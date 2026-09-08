use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::{Body, to_bytes};
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::Response;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Notify;
use tracing::warn;
use uuid::Uuid;

const CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);
const MAX_CACHE_ENTRIES: usize = 10_000;
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

pub(crate) const IDEMPOTENCY_KEY_HEADER: &str = "idempotency-key";
pub(crate) const IDEMPOTENCY_STATUS_HEADER: &str = "x-idempotency-status";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct StoredResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl StoredResponse {
    pub(crate) async fn from_response(response: Response) -> Result<Self, String> {
        let (parts, body) = response.into_parts();
        let body = to_bytes(body, MAX_RESPONSE_BYTES)
            .await
            .map_err(|err| format!("failed buffering idempotent response: {err}"))?;
        let headers = parts
            .headers
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.as_str().to_string(), value.to_string()))
            })
            .collect();
        Ok(Self {
            status: parts.status.as_u16(),
            headers,
            body: body.to_vec(),
        })
    }

    pub(crate) fn into_response(self) -> Response {
        let mut response = Response::new(Body::from(self.body));
        *response.status_mut() =
            StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        for (name, value) in self.headers {
            if let (Ok(name), Ok(value)) = (
                HeaderName::try_from(name.as_str()),
                HeaderValue::try_from(value),
            ) {
                response.headers_mut().append(name, value);
            }
        }
        response
    }

    fn is_success(&self) -> bool {
        StatusCode::from_u16(self.status).is_ok_and(|status| status.is_success())
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct CacheRecord {
    key_hash: String,
    fingerprint: String,
    expires_at_ms: u64,
    response: StoredResponse,
}

#[derive(Debug)]
enum EntryState {
    InFlight,
    Completed {
        response: Arc<StoredResponse>,
        expires_at_ms: u64,
    },
    Aborted,
}

#[derive(Debug)]
struct Entry {
    fingerprint: String,
    state: Mutex<EntryState>,
    notify: Notify,
}

#[derive(Debug)]
struct StoreInner {
    cache_dir: PathBuf,
    entries: Mutex<HashMap<String, Arc<Entry>>>,
    completed_writes: AtomicU64,
}

#[derive(Clone, Debug)]
pub(crate) struct IdempotencyStore {
    inner: Arc<StoreInner>,
}

pub(crate) enum Begin {
    Leader(Leader),
    Follower(Follower),
    Cached(Arc<StoredResponse>),
    Conflict,
}

pub(crate) struct Leader {
    store: Weak<StoreInner>,
    key_hash: String,
    entry: Arc<Entry>,
    finished: bool,
}

pub(crate) struct Follower {
    entry: Arc<Entry>,
}

pub(crate) enum FollowResult {
    Completed(Arc<StoredResponse>),
    Aborted,
}

impl IdempotencyStore {
    pub(crate) fn new(cache_dir: PathBuf) -> Self {
        if let Err(err) = fs::create_dir_all(&cache_dir) {
            warn!(
                path = %cache_dir.display(),
                error = %err,
                "failed creating idempotency cache directory; using memory cache only"
            );
        }

        let now_ms = unix_time_ms();
        let mut records = load_records(&cache_dir, now_ms);
        records.sort_unstable_by_key(|record| std::cmp::Reverse(record.expires_at_ms));

        let mut entries = HashMap::new();
        for record in records.into_iter().take(MAX_CACHE_ENTRIES) {
            entries.insert(
                record.key_hash,
                Arc::new(Entry {
                    fingerprint: record.fingerprint,
                    state: Mutex::new(EntryState::Completed {
                        response: Arc::new(record.response),
                        expires_at_ms: record.expires_at_ms,
                    }),
                    notify: Notify::new(),
                }),
            );
        }

        cleanup_files(&cache_dir, now_ms, MAX_CACHE_ENTRIES);
        Self {
            inner: Arc::new(StoreInner {
                cache_dir,
                entries: Mutex::new(entries),
                completed_writes: AtomicU64::new(0),
            }),
        }
    }

    pub(crate) fn begin(&self, key: &str, fingerprint: &str) -> Begin {
        let key_hash = hash_bytes(key.as_bytes());
        let now_ms = unix_time_ms();
        let mut entries = self.inner.entries.lock().expect("idempotency map poisoned");

        if let Some(entry) = entries.get(&key_hash).cloned() {
            if entry.fingerprint != fingerprint {
                return Begin::Conflict;
            }
            let state = entry.state.lock().expect("idempotency entry poisoned");
            match &*state {
                EntryState::InFlight => Begin::Follower(Follower {
                    entry: entry.clone(),
                }),
                EntryState::Completed {
                    response,
                    expires_at_ms,
                } if *expires_at_ms > now_ms => Begin::Cached(Arc::clone(response)),
                EntryState::Completed { .. } | EntryState::Aborted => {
                    drop(state);
                    entries.remove(&key_hash);
                    let entry = Arc::new(Entry {
                        fingerprint: fingerprint.to_string(),
                        state: Mutex::new(EntryState::InFlight),
                        notify: Notify::new(),
                    });
                    entries.insert(key_hash.clone(), Arc::clone(&entry));
                    Begin::Leader(Leader {
                        store: Arc::downgrade(&self.inner),
                        key_hash,
                        entry,
                        finished: false,
                    })
                }
            }
        } else {
            let entry = Arc::new(Entry {
                fingerprint: fingerprint.to_string(),
                state: Mutex::new(EntryState::InFlight),
                notify: Notify::new(),
            });
            entries.insert(key_hash.clone(), Arc::clone(&entry));
            Begin::Leader(Leader {
                store: Arc::downgrade(&self.inner),
                key_hash,
                entry,
                finished: false,
            })
        }
    }
}

impl Leader {
    pub(crate) async fn finish(mut self, response: StoredResponse) -> Arc<StoredResponse> {
        let response = Arc::new(response);
        let expires_at_ms = unix_time_ms().saturating_add(CACHE_TTL.as_millis() as u64);

        {
            let mut state = self.entry.state.lock().expect("idempotency entry poisoned");
            *state = EntryState::Completed {
                response: Arc::clone(&response),
                expires_at_ms,
            };
        }
        self.entry.notify.notify_waiters();
        self.finished = true;

        if let Some(store) = self.store.upgrade() {
            if response.is_success() {
                let record = CacheRecord {
                    key_hash: self.key_hash.clone(),
                    fingerprint: self.entry.fingerprint.clone(),
                    expires_at_ms,
                    response: (*response).clone(),
                };
                if let Err(err) = persist_record(&store.cache_dir, &record).await {
                    warn!(
                        key_hash = self.key_hash,
                        error = %err,
                        "failed persisting idempotent response"
                    );
                }
                trim_memory_entries(&store, MAX_CACHE_ENTRIES);
                if store.completed_writes.fetch_add(1, Ordering::Relaxed) % 128 == 0 {
                    let cache_dir = store.cache_dir.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        cleanup_files(&cache_dir, unix_time_ms(), MAX_CACHE_ENTRIES);
                    })
                    .await;
                }
            } else {
                remove_entry_if_same(&store, &self.key_hash, &self.entry);
            }
        }
        response
    }
}

impl Drop for Leader {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        {
            let mut state = self.entry.state.lock().expect("idempotency entry poisoned");
            *state = EntryState::Aborted;
        }
        self.entry.notify.notify_waiters();
        if let Some(store) = self.store.upgrade() {
            remove_entry_if_same(&store, &self.key_hash, &self.entry);
        }
    }
}

impl Follower {
    pub(crate) async fn wait(self) -> FollowResult {
        loop {
            let notified = self.entry.notify.notified();
            {
                let state = self.entry.state.lock().expect("idempotency entry poisoned");
                match &*state {
                    EntryState::Completed { response, .. } => {
                        return FollowResult::Completed(Arc::clone(response));
                    }
                    EntryState::Aborted => return FollowResult::Aborted,
                    EntryState::InFlight => {}
                }
            }
            notified.await;
        }
    }
}

pub(crate) fn validate_key(key: &str) -> Result<(), &'static str> {
    if key.is_empty() || key.len() > 255 || !key.bytes().all(|byte| (0x21..=0x7e).contains(&byte)) {
        Err("Idempotency-Key must contain 1 to 255 visible ASCII characters")
    } else {
        Ok(())
    }
}

pub(crate) fn request_fingerprint(
    model: &str,
    language: Option<&str>,
    response_format: &str,
    file_name: Option<&str>,
    file_bytes: &[u8],
) -> String {
    let mut hasher = Sha256::new();
    update_field(&mut hasher, model.as_bytes());
    update_field(&mut hasher, language.unwrap_or_default().as_bytes());
    update_field(&mut hasher, response_format.as_bytes());
    update_field(&mut hasher, file_name.unwrap_or_default().as_bytes());
    update_field(&mut hasher, file_bytes);
    hex::encode(hasher.finalize())
}

pub(crate) fn attach_headers(response: &mut Response, key: &str, status: &'static str) {
    if let Ok(value) = HeaderValue::try_from(key) {
        response.headers_mut().insert(IDEMPOTENCY_KEY_HEADER, value);
    }
    response
        .headers_mut()
        .insert(IDEMPOTENCY_STATUS_HEADER, HeaderValue::from_static(status));
}

fn update_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

fn hash_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn cache_path(cache_dir: &Path, key_hash: &str) -> PathBuf {
    cache_dir.join(format!("{key_hash}.json"))
}

async fn persist_record(cache_dir: &Path, record: &CacheRecord) -> Result<(), String> {
    tokio::fs::create_dir_all(cache_dir)
        .await
        .map_err(|err| err.to_string())?;
    let bytes = serde_json::to_vec(record).map_err(|err| err.to_string())?;
    let temporary = cache_dir.join(format!(".{}.tmp", Uuid::new_v4()));
    tokio::fs::write(&temporary, bytes)
        .await
        .map_err(|err| err.to_string())?;
    let destination = cache_path(cache_dir, &record.key_hash);
    match tokio::fs::rename(&temporary, &destination).await {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
            tokio::fs::remove_file(&destination)
                .await
                .map_err(|remove_err| remove_err.to_string())?;
            tokio::fs::rename(&temporary, destination)
                .await
                .map_err(|rename_err| rename_err.to_string())
        }
        Err(err) => Err(err.to_string()),
    }
}

fn load_records(cache_dir: &Path, now_ms: u64) -> Vec<CacheRecord> {
    let Ok(files) = fs::read_dir(cache_dir) else {
        return Vec::new();
    };
    files
        .filter_map(Result::ok)
        .filter_map(|file| fs::read(file.path()).ok())
        .filter_map(|bytes| serde_json::from_slice::<CacheRecord>(&bytes).ok())
        .filter(|record| record.expires_at_ms > now_ms && record.response.is_success())
        .collect()
}

fn cleanup_files(cache_dir: &Path, now_ms: u64, max_entries: usize) {
    let Ok(files) = fs::read_dir(cache_dir) else {
        return;
    };
    let mut records = files
        .filter_map(Result::ok)
        .filter_map(|file| {
            let path = file.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                let _ = fs::remove_file(path);
                return None;
            }
            let record = fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<CacheRecord>(&bytes).ok());
            match record {
                Some(record) if record.expires_at_ms > now_ms => Some((path, record.expires_at_ms)),
                _ => {
                    let _ = fs::remove_file(path);
                    None
                }
            }
        })
        .collect::<Vec<_>>();
    records.sort_unstable_by_key(|(_, expires_at_ms)| std::cmp::Reverse(*expires_at_ms));
    for (path, _) in records.into_iter().skip(max_entries) {
        let _ = fs::remove_file(path);
    }
}

fn trim_memory_entries(store: &StoreInner, max_entries: usize) {
    let mut entries = store.entries.lock().expect("idempotency map poisoned");
    if entries.len() <= max_entries {
        return;
    }
    let mut completed = entries
        .iter()
        .filter_map(|(key, entry)| {
            let state = entry.state.lock().ok()?;
            match &*state {
                EntryState::Completed { expires_at_ms, .. } => Some((key.clone(), *expires_at_ms)),
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    completed.sort_unstable_by_key(|(_, expires_at_ms)| *expires_at_ms);
    let remove_count = entries.len().saturating_sub(max_entries);
    for (key, _) in completed.into_iter().take(remove_count) {
        entries.remove(&key);
    }
}

fn remove_entry_if_same(store: &StoreInner, key_hash: &str, entry: &Arc<Entry>) {
    let mut entries = store.entries.lock().expect("idempotency map poisoned");
    if entries
        .get(key_hash)
        .is_some_and(|current| Arc::ptr_eq(current, entry))
    {
        entries.remove(key_hash);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(status: StatusCode, body: &'static str) -> StoredResponse {
        StoredResponse {
            status: status.as_u16(),
            headers: vec![("content-type".to_string(), "text/plain".to_string())],
            body: body.as_bytes().to_vec(),
        }
    }

    #[test]
    fn fingerprint_includes_effective_request_fields() {
        let base = request_fingerprint("model", Some("vi"), "json", Some("a.wav"), b"audio");
        assert_eq!(
            base,
            request_fingerprint("model", Some("vi"), "json", Some("a.wav"), b"audio")
        );
        assert_ne!(
            base,
            request_fingerprint("model", Some("vi"), "text", Some("a.wav"), b"audio")
        );
        assert_ne!(
            base,
            request_fingerprint("model", Some("vi"), "json", Some("a.wav"), b"different")
        );
    }

    #[test]
    fn validates_keys() {
        assert!(validate_key("meeting-123:segment-4").is_ok());
        assert!(validate_key("").is_err());
        assert!(validate_key("contains space").is_err());
        assert!(validate_key(&"x".repeat(256)).is_err());
    }

    #[tokio::test]
    async fn coalesces_and_detects_conflicts() {
        let temp = tempfile::tempdir().unwrap();
        let store = IdempotencyStore::new(temp.path().to_path_buf());
        let leader = match store.begin("key", "fingerprint") {
            Begin::Leader(leader) => leader,
            _ => panic!("expected leader"),
        };
        let follower = match store.begin("key", "fingerprint") {
            Begin::Follower(follower) => follower,
            _ => panic!("expected follower"),
        };
        assert!(matches!(store.begin("key", "different"), Begin::Conflict));

        leader.finish(response(StatusCode::OK, "done")).await;
        match follower.wait().await {
            FollowResult::Completed(result) => assert_eq!(result.body, b"done"),
            FollowResult::Aborted => panic!("leader unexpectedly aborted"),
        }
        assert!(matches!(
            store.begin("key", "fingerprint"),
            Begin::Cached(_)
        ));
    }

    #[tokio::test]
    async fn completed_success_survives_restart() {
        let temp = tempfile::tempdir().unwrap();
        {
            let store = IdempotencyStore::new(temp.path().to_path_buf());
            let leader = match store.begin("key", "fingerprint") {
                Begin::Leader(leader) => leader,
                _ => panic!("expected leader"),
            };
            leader.finish(response(StatusCode::OK, "durable")).await;
        }

        let reloaded = IdempotencyStore::new(temp.path().to_path_buf());
        match reloaded.begin("key", "fingerprint") {
            Begin::Cached(result) => assert_eq!(result.body, b"durable"),
            _ => panic!("expected durable cache hit"),
        }
    }

    #[test]
    fn expired_records_are_removed_on_startup() {
        let temp = tempfile::tempdir().unwrap();
        let key_hash = hash_bytes(b"expired");
        let record = CacheRecord {
            key_hash: key_hash.clone(),
            fingerprint: "fingerprint".to_string(),
            expires_at_ms: unix_time_ms().saturating_sub(1),
            response: response(StatusCode::OK, "stale"),
        };
        fs::write(
            cache_path(temp.path(), &key_hash),
            serde_json::to_vec(&record).unwrap(),
        )
        .unwrap();

        let store = IdempotencyStore::new(temp.path().to_path_buf());
        assert!(matches!(
            store.begin("expired", "fingerprint"),
            Begin::Leader(_)
        ));
        assert!(!cache_path(temp.path(), &key_hash).exists());
    }

    #[test]
    fn cleanup_bounds_durable_records() {
        let temp = tempfile::tempdir().unwrap();
        let now = unix_time_ms();
        for index in 0..3 {
            let key_hash = hash_bytes(format!("key-{index}").as_bytes());
            let record = CacheRecord {
                key_hash: key_hash.clone(),
                fingerprint: format!("fingerprint-{index}"),
                expires_at_ms: now + 1_000 + index,
                response: response(StatusCode::OK, "cached"),
            };
            fs::write(
                cache_path(temp.path(), &key_hash),
                serde_json::to_vec(&record).unwrap(),
            )
            .unwrap();
        }

        cleanup_files(temp.path(), now, 2);
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 2);
    }

    #[tokio::test]
    async fn errors_are_coalesced_but_not_persisted() {
        let temp = tempfile::tempdir().unwrap();
        let store = IdempotencyStore::new(temp.path().to_path_buf());
        let leader = match store.begin("key", "fingerprint") {
            Begin::Leader(leader) => leader,
            _ => panic!("expected leader"),
        };
        let follower = match store.begin("key", "fingerprint") {
            Begin::Follower(follower) => follower,
            _ => panic!("expected follower"),
        };
        leader
            .finish(response(StatusCode::INTERNAL_SERVER_ERROR, "failed"))
            .await;
        assert!(matches!(follower.wait().await, FollowResult::Completed(_)));
        assert!(matches!(
            store.begin("key", "fingerprint"),
            Begin::Leader(_)
        ));
        assert!(fs::read_dir(temp.path()).unwrap().next().is_none());
    }
}
