use std::fs;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{ResolvedModelConfig, WorkspacePaths};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetLockEntry {
    pub id: String,
    pub path: PathBuf,
    pub sha256: String,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub archive_url: Option<String>,
    #[serde(default)]
    pub archive_member: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssetStatus {
    Valid,
    Corrupted { actual_sha256: String },
    Missing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetReportEntry {
    pub id: String,
    pub path: PathBuf,
    pub status: AssetStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    pub entries: Vec<AssetReportEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetLock {
    pub generated_at: Option<String>,
    pub note: Option<String>,
    pub entries: Vec<AssetLockEntry>,
}

impl AssetLock {
    fn resolve_entry_path(workspace: &WorkspacePaths, entry: &AssetLockEntry) -> PathBuf {
        if entry.path.is_absolute() {
            entry.path.clone()
        } else {
            workspace.resolve(&entry.path)
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed reading asset lock file: {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("invalid asset lock JSON: {}", path.display()))
    }

    pub fn validate_all(&self, workspace: &WorkspacePaths) -> Result<()> {
        for entry in &self.entries {
            self.validate_entry(workspace, entry)?;
        }
        Ok(())
    }

    pub fn validate_entry(&self, workspace: &WorkspacePaths, entry: &AssetLockEntry) -> Result<()> {
        let path = Self::resolve_entry_path(workspace, entry);
        let resolved = path
            .canonicalize()
            .with_context(|| format!("locked asset missing: {}", path.display()))?;
        let actual = sha256_file(&resolved)?;
        if actual != entry.sha256 {
            return Err(anyhow!(
                "checksum mismatch for {}: expected {}, got {}",
                entry.id,
                entry.sha256,
                actual
            ));
        }
        Ok(())
    }

    pub fn validate_model_assets(
        &self,
        workspace: &WorkspacePaths,
        model: &ResolvedModelConfig,
    ) -> Result<()> {
        let mut paths_to_validate = Vec::new();
        let mut push_required = |label: &str, path: Option<&PathBuf>| -> Result<()> {
            let path = path
                .with_context(|| format!("missing resolved {label} path for model {}", model.id))?;
            let resolved = path
                .canonicalize()
                .with_context(|| format!("resolved {label} asset missing: {}", path.display()))?;
            paths_to_validate.push(resolved);
            Ok(())
        };

        push_required("encoder", model.encoder.as_ref())?;
        push_required("decoder", model.decoder.as_ref())?;
        push_required("joiner", model.joiner.as_ref())?;
        push_required("tokens", model.tokens.as_ref())?;

        if let Some(path) = &model.bpe_vocab {
            push_required("bpe vocab", Some(path))?;
        }

        if model.vad.enabled {
            let vad = model.vad_model_path.canonicalize().with_context(|| {
                format!(
                    "resolved VAD asset missing: {}",
                    model.vad_model_path.display()
                )
            })?;
            paths_to_validate.push(vad);
        }

        let mut locked_paths = std::collections::HashMap::new();
        for entry in &self.entries {
            let entry_path = Self::resolve_entry_path(workspace, entry);
            let canonical = entry_path.canonicalize().unwrap_or(entry_path);
            locked_paths.insert(canonical, entry);
        }

        for path in &paths_to_validate {
            let entry = locked_paths.get(path).copied().ok_or_else(|| {
                anyhow!(
                    "resolved model asset missing from asset lock: {}",
                    path.display()
                )
            })?;
            self.validate_entry(workspace, entry)?;
        }

        Ok(())
    }

    pub fn verify_all(&self, workspace: &WorkspacePaths) -> VerificationReport {
        let mut report_entries = Vec::new();
        for entry in &self.entries {
            let path = Self::resolve_entry_path(workspace, entry);
            let status = if !path.exists() {
                AssetStatus::Missing
            } else {
                match sha256_file(&path) {
                    Ok(actual) => {
                        if actual == entry.sha256 {
                            AssetStatus::Valid
                        } else {
                            AssetStatus::Corrupted {
                                actual_sha256: actual,
                            }
                        }
                    }
                    Err(_) => AssetStatus::Corrupted {
                        actual_sha256: String::new(),
                    },
                }
            };
            report_entries.push(AssetReportEntry {
                id: entry.id.clone(),
                path: entry.path.clone(),
                status,
            });
        }
        VerificationReport {
            entries: report_entries,
        }
    }

    pub async fn download_missing_assets(
        &self,
        workspace: &WorkspacePaths,
        force: bool,
    ) -> Result<()> {
        let report = self.verify_all(workspace);
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()?;
        let mut archive_downloads: std::collections::BTreeMap<String, Vec<&AssetLockEntry>> =
            std::collections::BTreeMap::new();

        for entry in &self.entries {
            let report_entry = report.entries.iter().find(|e| e.id == entry.id);
            let needs_download = match report_entry {
                Some(r) => match r.status {
                    AssetStatus::Valid => force,
                    _ => true,
                },
                None => true,
            };

            if needs_download {
                if let Some(url_str) = &entry.url {
                    let path = Self::resolve_entry_path(workspace, entry);

                    println!("Downloading asset {} from {}", entry.id, url_str);
                    if let Some(parent) = path.parent() {
                        std::fs::create_dir_all(parent)?;
                    }

                    let response = client.get(url_str).send().await.with_context(|| {
                        format!("Failed to download {}: network error", entry.id)
                    })?;

                    if !response.status().is_success() {
                        return Err(anyhow!(
                            "Failed to download {}: HTTP status {}",
                            entry.id,
                            response.status()
                        ));
                    }

                    // Download to a temporary file first
                    let temp_dir = tempfile::tempdir_in(path.parent().unwrap_or(&workspace.root))?;
                    let temp_file_path = temp_dir.path().join("download.tmp");

                    let mut file = std::fs::File::create(&temp_file_path)?;
                    let mut stream = response.bytes_stream();
                    use futures::StreamExt;

                    while let Some(chunk_result) = stream.next().await {
                        let chunk = chunk_result.with_context(|| {
                            format!("Failed to read stream chunk for {}", entry.id)
                        })?;
                        file.write_all(&chunk)?;
                    }
                    file.sync_all()?;
                    // Verify downloaded file hash
                    let actual = sha256_file(&temp_file_path)?;
                    if actual != entry.sha256 {
                        return Err(anyhow!(
                            "Downloaded file hash mismatch for {}: expected {}, got {}",
                            entry.id,
                            entry.sha256,
                            actual
                        ));
                    }

                    // Move to final destination
                    std::fs::rename(temp_file_path, &path)?;
                    println!("Successfully downloaded {} to {}", entry.id, path.display());
                } else if let (Some(archive_url), Some(_archive_member)) =
                    (&entry.archive_url, &entry.archive_member)
                {
                    archive_downloads
                        .entry(archive_url.clone())
                        .or_default()
                        .push(entry);
                } else {
                    println!(
                        "Warning: Asset {} is missing/corrupted but has no source URL configured",
                        entry.id
                    );
                }
            }
        }
        for (archive_url, entries) in archive_downloads {
            download_archive_entries(&client, workspace, &archive_url, &entries).await?;
        }
        Ok(())
    }
}

async fn download_archive_entries(
    client: &reqwest::Client,
    workspace: &WorkspacePaths,
    archive_url: &str,
    entries: &[&AssetLockEntry],
) -> Result<()> {
    println!(
        "Downloading archive {} for {} assets",
        archive_url,
        entries.len()
    );
    let temp_dir = tempfile::tempdir_in(&workspace.root)?;
    let archive_path = temp_dir.path().join("archive.tar.bz2");
    download_to_file(client, archive_url, &archive_path, "archive").await?;

    let mut wanted = entries
        .iter()
        .map(|entry| {
            (
                entry
                    .archive_member
                    .as_ref()
                    .expect("archive_member checked by caller")
                    .clone(),
                *entry,
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();

    let archive_file = fs::File::open(&archive_path)
        .with_context(|| format!("failed opening archive {}", archive_path.display()))?;
    let decoder = bzip2::read::BzDecoder::new(archive_file);
    let mut archive = tar::Archive::new(decoder);

    for file in archive
        .entries()
        .context("failed reading archive entries")?
    {
        let mut file = file.context("failed reading archive entry")?;
        let member = file
            .path()
            .context("failed reading archive member path")?
            .to_string_lossy()
            .replace('\\', "/");
        let Some(entry) = wanted.remove(&member) else {
            continue;
        };

        let dest = AssetLock::resolve_entry_path(workspace, entry);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        let parent = dest.parent().unwrap_or(&workspace.root);
        let extract_dir = tempfile::tempdir_in(parent)?;
        let temp_file_path = extract_dir.path().join("download.tmp");
        {
            let mut out = fs::File::create(&temp_file_path)?;
            std::io::copy(&mut file, &mut out)
                .with_context(|| format!("failed extracting {} from archive", entry.id))?;
            out.sync_all()?;
        }

        let actual = sha256_file(&temp_file_path)?;
        if actual != entry.sha256 {
            return Err(anyhow!(
                "Extracted file hash mismatch for {}: expected {}, got {}",
                entry.id,
                entry.sha256,
                actual
            ));
        }

        fs::rename(&temp_file_path, &dest)?;
        println!("Successfully extracted {} to {}", entry.id, dest.display());
    }

    if !wanted.is_empty() {
        let missing = wanted.keys().cloned().collect::<Vec<_>>().join(", ");
        return Err(anyhow!(
            "archive {} did not contain required members: {}",
            archive_url,
            missing
        ));
    }

    Ok(())
}

async fn download_to_file(
    client: &reqwest::Client,
    url: &str,
    path: &Path,
    label: &str,
) -> Result<()> {
    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("Failed to download {label}: network error"))?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Failed to download {label}: HTTP status {}",
            response.status()
        ));
    }

    let mut file = fs::File::create(path)?;
    let mut stream = response.bytes_stream();
    use futures::StreamExt;
    while let Some(chunk_result) = stream.next().await {
        let chunk =
            chunk_result.with_context(|| format!("Failed to read stream chunk for {label}"))?;
        file.write_all(&chunk)?;
    }
    file.sync_all()?;
    Ok(())
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let file =
        fs::File::open(path).with_context(|| format!("failed reading {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];

    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("failed reading {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::tempdir;

    use super::*;
    use crate::config::VadConfig;
    use crate::postprocess::PostprocessMode;

    #[test]
    fn verify_all_resolves_models_entries_against_models_dir() {
        let root = tempdir().expect("root tempdir");
        let models_dir = tempdir().expect("models tempdir");
        let workspace = WorkspacePaths {
            root: root.path().to_path_buf(),
            models_dir: models_dir.path().to_path_buf(),
        };

        let asset_path = models_dir.path().join("vad/ten-vad.int8.onnx");
        std::fs::create_dir_all(asset_path.parent().expect("vad parent")).expect("create vad dir");
        std::fs::write(&asset_path, b"vad-bytes").expect("write vad asset");

        let lock = AssetLock {
            generated_at: None,
            note: None,
            entries: vec![AssetLockEntry {
                id: "vad.ten".to_string(),
                path: PathBuf::from("models/vad/ten-vad.int8.onnx"),
                sha256: sha256_file(&asset_path).expect("hash vad asset"),
                url: None,
                archive_url: None,
                archive_member: None,
            }],
        };

        let report = lock.verify_all(&workspace);

        assert!(matches!(report.entries[0].status, AssetStatus::Valid));
    }

    #[test]
    fn validate_model_assets_requires_lock_entries_for_resolved_assets() {
        let root = tempdir().expect("root tempdir");
        let models_dir = tempdir().expect("models tempdir");
        let workspace = WorkspacePaths {
            root: root.path().to_path_buf(),
            models_dir: models_dir.path().to_path_buf(),
        };

        let model_dir = models_dir.path().join("stt/test-model");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        let encoder = model_dir.join("encoder.onnx");
        let decoder = model_dir.join("decoder.onnx");
        let joiner = model_dir.join("joiner.onnx");
        let tokens = model_dir.join("tokens.txt");
        let vad = models_dir.path().join("vad/ten-vad.int8.onnx");
        std::fs::create_dir_all(vad.parent().expect("vad parent")).expect("create vad dir");
        std::fs::write(&encoder, b"encoder").expect("write encoder");
        std::fs::write(&decoder, b"decoder").expect("write decoder");
        std::fs::write(&joiner, b"joiner").expect("write joiner");
        std::fs::write(&tokens, b"tokens").expect("write tokens");
        std::fs::write(&vad, b"vad").expect("write vad");

        let lock = AssetLock {
            generated_at: None,
            note: None,
            entries: vec![
                AssetLockEntry {
                    id: "encoder".to_string(),
                    path: PathBuf::from("models/stt/test-model/encoder.onnx"),
                    sha256: sha256_file(&encoder).expect("hash encoder"),
                    url: None,
                    archive_url: None,
                    archive_member: None,
                },
                AssetLockEntry {
                    id: "joiner".to_string(),
                    path: PathBuf::from("models/stt/test-model/joiner.onnx"),
                    sha256: sha256_file(&joiner).expect("hash joiner"),
                    url: None,
                    archive_url: None,
                    archive_member: None,
                },
                AssetLockEntry {
                    id: "tokens".to_string(),
                    path: PathBuf::from("models/stt/test-model/tokens.txt"),
                    sha256: sha256_file(&tokens).expect("hash tokens"),
                    url: None,
                    archive_url: None,
                    archive_member: None,
                },
                AssetLockEntry {
                    id: "vad.ten".to_string(),
                    path: PathBuf::from("models/vad/ten-vad.int8.onnx"),
                    sha256: sha256_file(&vad).expect("hash vad"),
                    url: None,
                    archive_url: None,
                    archive_member: None,
                },
            ],
        };
        let model = ResolvedModelConfig {
            id: "test-model".to_string(),
            language: "vi".to_string(),
            model_dir: model_dir.clone(),
            model_type: "transducer".to_string(),
            provider: "cpu".to_string(),
            num_threads: 2,
            postprocess_mode: PostprocessMode::CleanLower,
            capu_model_id: None,
            startup_probe_wav_path: None,
            startup_probe_expected_text: None,
            encoder: Some(encoder),
            decoder: Some(decoder),
            joiner: Some(joiner),
            tokens: Some(tokens),
            bpe_vocab: None,
            vad: VadConfig {
                enabled: true,
                profile: "ten".to_string(),
                threshold: 0.5,
                min_silence: 0.5,
                min_speech: 0.05,
                max_speech: 14.0,
            },
            vad_model_path: vad,
        };

        let err = lock
            .validate_model_assets(&workspace, &model)
            .expect_err("decoder should require a lock entry");

        assert!(
            err.to_string()
                .contains("resolved model asset missing from asset lock")
        );
    }

    #[tokio::test]
    async fn download_missing_assets_extracts_archive_members() -> Result<()> {
        let root = tempdir()?;
        let workspace = WorkspacePaths {
            root: root.path().to_path_buf(),
            models_dir: root.path().join("models"),
        };
        let member_bytes = b"archive member bytes";
        let archive_bytes = build_test_tar_bz2("bundle/member.txt", member_bytes)?;
        let server = std::net::TcpListener::bind("127.0.0.1:0")?;
        let url = format!("http://{}/archive.tar.bz2", server.local_addr()?);

        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = server.accept() {
                let mut buffer = [0_u8; 1024];
                let _ = stream.read(&mut buffer);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/octet-stream\r\nConnection: close\r\n\r\n",
                    archive_bytes.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.write_all(&archive_bytes);
            }
        });

        let target = workspace.models_dir.join("stt/test/member.txt");
        let lock = AssetLock {
            generated_at: None,
            note: None,
            entries: vec![AssetLockEntry {
                id: "archive.member".to_string(),
                path: PathBuf::from("models/stt/test/member.txt"),
                sha256: hex::encode(sha2::Sha256::digest(member_bytes)),
                url: None,
                archive_url: Some(url),
                archive_member: Some("bundle/member.txt".to_string()),
            }],
        };

        lock.download_missing_assets(&workspace, false).await?;

        assert_eq!(std::fs::read(target)?, member_bytes);
        Ok(())
    }

    fn build_test_tar_bz2(path: &str, bytes: &[u8]) -> Result<Vec<u8>> {
        let mut compressed = Vec::new();
        {
            let encoder = bzip2::write::BzEncoder::new(&mut compressed, bzip2::Compression::best());
            let mut builder = tar::Builder::new(encoder);
            let mut header = tar::Header::new_gnu();
            header.set_path(path)?;
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append(&header, bytes)?;
            let encoder = builder.into_inner()?;
            encoder.finish()?;
        }
        Ok(compressed)
    }
}
