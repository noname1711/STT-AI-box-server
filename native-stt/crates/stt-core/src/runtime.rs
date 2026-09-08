use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use tracing::{info, warn};

use crate::assets::resolve_offline_model_assets;
use crate::audio::{DecodedAudio, decode_audio_bytes, decode_audio_path};
use crate::checksum::AssetLock;
use crate::config::{
    AppRuntimeConfig, CapuModelConfig, ModelRegistry, ResolvedModelConfig, WorkspacePaths,
};
use crate::recognizer::{RecognizerRuntime, WarmedRecognizerRuntime};
use crate::types::{ModelSummary, TranscriptionResult};

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProbeResult {
    pub model_id: String,
    pub expected: Option<String>,
    pub actual: String,
    pub matches_expected: bool,
    #[serde(serialize_with = "serialize_pathbuf")]
    pub probe_path: PathBuf,
}

fn serialize_pathbuf<S: serde::Serializer>(
    path: &std::path::Path,
    s: S,
) -> Result<S::Ok, S::Error> {
    s.serialize_str(&path.display().to_string())
}

/// Status information for a single model, exposed via the admin status endpoint.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelStatus {
    pub id: String,
    /// Whether the model assets have been resolved and validated.
    pub resolved: bool,
    /// Whether the recognizer is loaded and cached.
    pub recognizer_loaded: bool,
    /// Whether the model is configured for warm-start at server startup.
    pub warm_start: bool,
    /// ISO 8601 timestamp of the last transcription request for this model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_used_at: Option<String>,
}

/// Result of a single model warmup operation, returned by the admin warmup endpoint.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WarmupResult {
    pub model: String,
    pub already_loaded: bool,
    pub elapsed_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probe: Option<ProbeResult>,
}

#[derive(Debug, Clone)]
pub struct SttRuntime {
    pub workspace: WorkspacePaths,
    pub runtime_config: AppRuntimeConfig,
    pub registry: ModelRegistry,
    pub asset_lock: AssetLock,
    resolved_models: Arc<Mutex<HashMap<String, ResolvedModelConfig>>>,
    recognizers: Arc<Mutex<HashMap<String, Arc<Mutex<WarmedRecognizerRuntime>>>>>,
    /// Tracks whether warm-start has completed. Used by `/health` readiness gating.
    pub is_ready: Arc<AtomicBool>,
}

impl SttRuntime {
    pub fn load(explicit_root: Option<PathBuf>, config_path: Option<&Path>) -> Result<Self> {
        let workspace = WorkspacePaths::discover(explicit_root, None)?;
        let runtime_config = workspace.load_runtime_config(config_path)?;
        let registry = workspace.load_registry(&runtime_config)?;
        let asset_lock = AssetLock::load(&workspace.resolve(&runtime_config.asset_lock_path))?;
        let asset_validation_started = Instant::now();

        info!(
            entries = asset_lock.entries.len(),
            "validating locked assets"
        );

        // Validate assets that are present
        for entry in &asset_lock.entries {
            let path = WorkspacePaths::resolve(&workspace, &entry.path);

            if path.exists() {
                asset_lock
                    .validate_entry(&workspace, entry)
                    .with_context(|| format!("Asset validation failed for ID: {}", entry.id))?;
            } else {
                info!(
                    asset_id = %entry.id,
                    path = %entry.path.display(),
                    "skipping validation for uninstalled asset"
                );
            }
        }

        info!(
            elapsed_ms = asset_validation_started.elapsed().as_millis(),
            "locked assets validated"
        );

        Ok(Self {
            workspace,
            runtime_config,
            registry,
            asset_lock,
            resolved_models: Arc::new(Mutex::new(HashMap::new())),
            recognizers: Arc::new(Mutex::new(HashMap::new())),
            is_ready: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn list_models(&self) -> Vec<ModelSummary> {
        self.registry
            .models
            .iter()
            .map(|model| ModelSummary {
                id: model.id.clone(),
                language: model.language.clone(),
                postprocess_mode: model.postprocess_mode,
            })
            .collect()
    }

    pub fn resolve_model(&self, model_id: &str) -> Result<ResolvedModelConfig> {
        if let Some(model) = self
            .resolved_models
            .lock()
            .map_err(|_| anyhow!("resolved model cache mutex poisoned"))?
            .get(model_id)
            .cloned()
        {
            return Ok(model);
        }

        let model = self
            .registry
            .models
            .iter()
            .find(|model| model.id == model_id)
            .ok_or_else(|| anyhow!("unknown model id: {model_id}"))?;
        let resolved = self.workspace.resolve_model(&self.runtime_config, model);
        let assets = resolve_offline_model_assets(
            &resolved.model_dir,
            &resolved.model_type,
            resolved.encoder.as_deref(),
            resolved.decoder.as_deref(),
            resolved.joiner.as_deref(),
            resolved.tokens.as_deref(),
            resolved.bpe_vocab.as_deref(),
        )?;
        let resolved = ResolvedModelConfig {
            model_dir: assets.model_dir.clone(),
            encoder: Some(assets.encoder.clone()),
            decoder: Some(assets.decoder.clone()),
            joiner: Some(assets.joiner.clone()),
            tokens: Some(assets.tokens.clone()),
            bpe_vocab: assets.bpe_vocab.clone(),
            ..resolved
        };
        self.asset_lock
            .validate_model_assets(&self.workspace, &resolved)?;
        self.resolved_models
            .lock()
            .map_err(|_| anyhow!("resolved model cache mutex poisoned"))?
            .insert(model_id.to_string(), resolved.clone());
        Ok(resolved)
    }

    pub fn default_model_id(&self) -> Result<&str> {
        self.registry
            .models
            .first()
            .map(|model| model.id.as_str())
            .context("model registry is empty")
    }

    pub fn resolve_capu_model(&self, capu_model_id: Option<&str>) -> Result<&CapuModelConfig> {
        let id = match capu_model_id {
            Some(id) => id,
            None => self
                .registry
                .capu_models
                .first()
                .map(|model| model.id.as_str())
                .context("CAPU model registry is empty")?,
        };
        self.registry
            .capu_models
            .iter()
            .find(|model| model.id == id)
            .ok_or_else(|| anyhow!("unknown CAPU model id: {id}"))
    }

    pub fn transcribe_path(&self, model_id: &str, path: &Path) -> Result<TranscriptionResult> {
        let decoded = decode_audio_path(path)?;
        self.transcribe_decoded(model_id, &decoded)
    }

    pub fn transcribe_bytes(
        &self,
        model_id: &str,
        filename: Option<&str>,
        bytes: &[u8],
    ) -> Result<TranscriptionResult> {
        let decoded = decode_audio_bytes(bytes, filename)?;
        self.transcribe_decoded(model_id, &decoded)
    }

    pub fn transcribe_decoded(
        &self,
        model_id: &str,
        decoded: &DecodedAudio,
    ) -> Result<TranscriptionResult> {
        let recognizer = self.warmed_recognizer(model_id)?;
        recognizer
            .lock()
            .map_err(|_| anyhow!("warmed recognizer mutex poisoned"))?
            .transcribe_samples(&decoded.samples, decoded.sample_rate)
    }

    pub fn warmed_recognizer(&self, model_id: &str) -> Result<Arc<Mutex<WarmedRecognizerRuntime>>> {
        let mut recognizers = self
            .recognizers
            .lock()
            .map_err(|_| anyhow!("warmed recognizer cache mutex poisoned"))?;

        if let Some(recognizer) = recognizers.get(model_id).cloned() {
            return Ok(recognizer);
        }

        // Hold the lock during construction to prevent duplicate loads under
        // concurrent requests for the same uncached model (Option A from the
        // review). This blocks other model lookups while a model loads, which
        // is acceptable for a small 2–3 model deployment.
        let model = self.resolve_model(model_id)?;
        let recognizer = Arc::new(Mutex::new(WarmedRecognizerRuntime::new(model)?));
        recognizers.insert(model_id.to_string(), Arc::clone(&recognizer));
        Ok(recognizer)
    }

    /// Check if a recognizer is already cached without loading it. Used for
    /// logging cache hit/miss in the HTTP layer.
    pub fn is_recognizer_cached(&self, model_id: &str) -> bool {
        self.recognizers
            .lock()
            .map(|r| r.contains_key(model_id))
            .unwrap_or(false)
    }

    pub fn run_probe(&self, model_id: Option<&str>) -> Result<ProbeResult> {
        let model_id = model_id.unwrap_or(self.default_model_id()?);
        let model = self.resolve_model(model_id)?;
        let runtime = RecognizerRuntime::new(model.clone());
        let result = runtime.transcribe_probe()?;
        let expected = model.startup_probe_expected_text.clone();
        let actual = result.text.trim().to_string();
        let matches_expected = expected
            .as_deref()
            .map(|text| text == actual)
            .unwrap_or(true);
        Ok(ProbeResult {
            model_id: model.id,
            expected,
            actual,
            matches_expected,
            probe_path: model
                .startup_probe_wav_path
                .context("missing startup probe wav")?,
        })
    }

    /// Prewarm all models listed in `warm_start_models` config. Called once at
    /// server startup, before marking the server as ready. Fails fast if any
    /// configured warm-start model cannot be loaded.
    pub fn warm_start(&self) -> Result<()> {
        let model_ids = &self.runtime_config.server.warm_start_models;
        if model_ids.is_empty() {
            info!("no warm_start_models configured; skipping warmup");
            self.is_ready.store(true, Ordering::Release);
            return Ok(());
        }

        let timeout_secs = self.runtime_config.server.warm_start_timeout_seconds;
        info!(
            models = ?model_ids,
            timeout_seconds = timeout_secs,
            "starting warm-start"
        );

        for model_id in model_ids {
            self.warm_one_model(model_id, timeout_secs, true)?;
        }

        self.is_ready.store(true, Ordering::Release);
        info!("warm-start complete; server is ready");
        Ok(())
    }

    /// Warm a single model: resolve, load recognizer, optionally run probe.
    /// Returns the elapsed time in milliseconds.
    fn warm_one_model(&self, model_id: &str, timeout_secs: u64, run_probe: bool) -> Result<u128> {
        let started = Instant::now();
        let already_loaded = self
            .recognizers
            .lock()
            .map_err(|_| anyhow!("warmed recognizer cache mutex poisoned"))?
            .contains_key(model_id);

        if already_loaded {
            info!(model_id, "model already loaded; skipping warmup");
            return Ok(0);
        }

        let model = self.resolve_model(model_id)?;
        self.warmed_recognizer(model_id)?;

        if run_probe && let Some(path) = model.startup_probe_wav_path.as_deref() {
            match self.transcribe_path(model_id, path) {
                Ok(_) => info!(model_id, "warm-start probe succeeded"),
                Err(err) => warn!(
                    model_id,
                    error = %err,
                    "warm-start probe failed (model is still loaded)"
                ),
            }
        }

        let elapsed_ms = started.elapsed().as_millis();
        if elapsed_ms > (timeout_secs as u128 * 1000) {
            anyhow::bail!(
                "warmup for {} exceeded timeout of {} seconds (took {}ms)",
                model_id,
                timeout_secs,
                elapsed_ms
            );
        }

        info!(model_id, elapsed_ms, "warm-start completed for model");
        Ok(elapsed_ms)
    }

    /// Warm specific models on demand (admin warmup endpoint). Returns results
    /// for each model. Does not fail fast — individual model failures are
    /// reported in the result.
    pub fn warmup_models(&self, model_ids: &[String], run_probe: bool) -> Vec<WarmupResult> {
        let timeout_secs = self.runtime_config.server.warm_start_timeout_seconds;
        let mut results = Vec::with_capacity(model_ids.len());

        for model_id in model_ids {
            let already_loaded = self
                .recognizers
                .lock()
                .map_err(|_| anyhow!("warmed recognizer cache mutex poisoned"))
                .map(|r| r.contains_key(model_id.as_str()))
                .unwrap_or(false);

            let started = Instant::now();
            let probe = if run_probe {
                match self.run_probe(Some(model_id)) {
                    Ok(result) => Some(result),
                    Err(err) => {
                        warn!(model_id, error = %err, "probe failed during admin warmup");
                        None
                    }
                }
            } else {
                // Just load the recognizer without probe
                match self.warm_one_model(model_id, timeout_secs, false) {
                    Ok(_) => None,
                    Err(err) => {
                        warn!(model_id, error = %err, "warmup failed during admin warmup");
                        results.push(WarmupResult {
                            model: model_id.clone(),
                            already_loaded,
                            elapsed_ms: started.elapsed().as_millis(),
                            probe: None,
                        });
                        continue;
                    }
                }
            };

            results.push(WarmupResult {
                model: model_id.clone(),
                already_loaded,
                elapsed_ms: started.elapsed().as_millis(),
                probe,
            });
        }

        results
    }

    /// Return status information for all registered models.
    pub fn model_status(&self) -> Vec<ModelStatus> {
        let warm_start_set: std::collections::HashSet<&str> = self
            .runtime_config
            .server
            .warm_start_models
            .iter()
            .map(|s| s.as_str())
            .collect();

        let resolved_ids: std::collections::HashSet<String> = self
            .resolved_models
            .lock()
            .map(|r| r.keys().cloned().collect())
            .unwrap_or_default();

        let loaded_ids: std::collections::HashSet<String> = self
            .recognizers
            .lock()
            .map(|r| r.keys().cloned().collect())
            .unwrap_or_default();

        self.registry
            .models
            .iter()
            .map(|model| ModelStatus {
                id: model.id.clone(),
                resolved: resolved_ids.contains(&model.id),
                recognizer_loaded: loaded_ids.contains(&model.id),
                warm_start: warm_start_set.contains(model.id.as_str()),
                last_used_at: None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, OnceLock};

    use tempfile::tempdir;

    use super::*;

    fn env_lock() -> &'static Mutex<()> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn load_honors_split_runtime_root_and_models_dir() -> Result<()> {
        let _guard = env_lock().lock().expect("env lock");
        let runtime_root = tempdir()?;
        let models_root = tempdir()?;

        std::fs::create_dir_all(runtime_root.path().join("config"))?;
        std::fs::create_dir_all(runtime_root.path().join("baselines/phase0"))?;
        let model_dir = models_root.path().join("stt/test-model");
        std::fs::create_dir_all(&model_dir)?;
        let vad_dir = models_root.path().join("vad");
        std::fs::create_dir_all(&vad_dir)?;

        let encoder = model_dir.join("encoder.onnx");
        let decoder = model_dir.join("decoder.onnx");
        let joiner = model_dir.join("joiner.onnx");
        let tokens = model_dir.join("tokens.txt");
        let vad = vad_dir.join("ten-vad.int8.onnx");

        std::fs::write(&encoder, b"encoder")?;
        std::fs::write(&decoder, b"decoder")?;
        std::fs::write(&joiner, b"joiner")?;
        std::fs::write(&tokens, b"tokens")?;
        std::fs::write(&vad, b"vad")?;

        std::fs::write(
            runtime_root.path().join("config/runtime.toml"),
            r#"models_config_path = "config/models.local.json"
asset_lock_path = "baselines/phase0/assets.lock.json"
provider = "cpu"
num_threads = 2

[vad]
enabled = true
profile = "ten"
threshold = 0.5
min_silence = 0.5
min_speech = 0.05
max_speech = 14.0

[server]
host = "0.0.0.0"
port = 8080
max_upload_mb = 128
max_audio_seconds = 14400
owned_by = "vit-stt"

[capu]
engine = "auto"
python_bin = "capu-worker/.venv/bin/python"
worker_dir = "capu-worker"
device = "cpu"
request_timeout_seconds = 120
"#,
        )?;
        std::fs::write(
            runtime_root.path().join("config/models.local.json"),
            r#"{
  "models": [
    {
      "id": "test-model",
      "language": "vi",
      "model_dir": "models/stt/test-model",
      "postprocess_mode": "clean_lower"
    }
  ]
}"#,
        )?;

        let lock = serde_json::json!({
            "entries": [
                {
                    "id": "encoder",
                    "path": "models/stt/test-model/encoder.onnx",
                    "sha256": crate::checksum::sha256_file(&encoder)?,
                    "url": serde_json::Value::Null
                },
                {
                    "id": "decoder",
                    "path": "models/stt/test-model/decoder.onnx",
                    "sha256": crate::checksum::sha256_file(&decoder)?,
                    "url": serde_json::Value::Null
                },
                {
                    "id": "joiner",
                    "path": "models/stt/test-model/joiner.onnx",
                    "sha256": crate::checksum::sha256_file(&joiner)?,
                    "url": serde_json::Value::Null
                },
                {
                    "id": "tokens",
                    "path": "models/stt/test-model/tokens.txt",
                    "sha256": crate::checksum::sha256_file(&tokens)?,
                    "url": serde_json::Value::Null
                },
                {
                    "id": "vad.ten",
                    "path": "models/vad/ten-vad.int8.onnx",
                    "sha256": crate::checksum::sha256_file(&vad)?,
                    "url": serde_json::Value::Null
                }
            ]
        });
        std::fs::write(
            runtime_root
                .path()
                .join("baselines/phase0/assets.lock.json"),
            serde_json::to_vec_pretty(&lock)?,
        )?;

        unsafe {
            std::env::set_var("VIT_STT_RUNTIME_ROOT", runtime_root.path());
            std::env::set_var("VIT_STT_MODELS_DIR", models_root.path());
        }

        let runtime = SttRuntime::load(None, None);

        unsafe {
            std::env::remove_var("VIT_STT_RUNTIME_ROOT");
            std::env::remove_var("VIT_STT_MODELS_DIR");
        }

        let runtime = runtime?;
        assert_eq!(runtime.workspace.root, runtime_root.path().canonicalize()?);
        assert_eq!(
            runtime.workspace.models_dir,
            models_root.path().canonicalize()?
        );

        let resolved = runtime.resolve_model("test-model")?;
        assert_eq!(resolved.model_dir, model_dir.canonicalize()?);
        assert_eq!(
            resolved.encoder.as_deref(),
            Some(encoder.canonicalize()?.as_path())
        );
        assert_eq!(resolved.vad_model_path, vad.canonicalize()?);

        Ok(())
    }
}
