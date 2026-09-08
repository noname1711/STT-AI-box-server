use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::postprocess::PostprocessMode;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_vad_profile")]
    pub profile: String,
    #[serde(default = "default_vad_threshold")]
    pub threshold: f32,
    #[serde(default = "default_vad_min_silence")]
    pub min_silence: f32,
    #[serde(default = "default_vad_min_speech")]
    pub min_speech: f32,
    #[serde(default = "default_vad_max_speech")]
    pub max_speech: f32,
}

fn default_true() -> bool {
    true
}
fn default_vad_profile() -> String {
    "ten".to_string()
}
fn default_vad_threshold() -> f32 {
    0.5
}
fn default_vad_min_silence() -> f32 {
    0.5
}
fn default_vad_min_speech() -> f32 {
    0.05
}
fn default_vad_max_speech() -> f32 {
    14.0
}
fn default_num_threads() -> i32 {
    2
}
fn default_provider() -> String {
    "cpu".to_string()
}
fn default_owned_by() -> String {
    "vit-stt".to_string()
}
fn default_max_upload_mb() -> u64 {
    256
}
fn default_max_audio_seconds() -> u64 {
    14400
}
fn default_warm_start_timeout_seconds() -> u64 {
    60
}
fn default_capu_engine() -> String {
    "auto".to_string()
}
fn default_capu_device() -> String {
    "cpu".to_string()
}
fn default_capu_timeout() -> u64 {
    240
}
fn default_capu_model_name() -> String {
    "vibert-capu".to_string()
}
fn default_capu_base_model_name() -> String {
    "base_model".to_string()
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    #[serde(default = "default_max_upload_mb")]
    pub max_upload_mb: u64,
    #[serde(default = "default_max_audio_seconds")]
    pub max_audio_seconds: u64,
    #[serde(default = "default_owned_by")]
    pub owned_by: String,
    /// Model IDs to prewarm at server startup. Vietnamese is prewarmed by
    /// default. English is lazy-loaded unless added here.
    #[serde(default)]
    pub warm_start_models: Vec<String>,
    /// Per-model warmup timeout in seconds. Default: 60.
    #[serde(default = "default_warm_start_timeout_seconds")]
    pub warm_start_timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapuModelConfig {
    pub id: String,
    pub model_dir: PathBuf,
    #[serde(default)]
    pub base_model_dir: Option<PathBuf>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub request_timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapuRuntimeConfig {
    #[serde(default = "default_capu_engine")]
    pub engine: String,
    #[serde(default)]
    pub worker_exe: Option<PathBuf>,
    pub python_bin: PathBuf,
    pub worker_dir: PathBuf,
    #[serde(default = "default_capu_model_name")]
    pub model_name: String,
    #[serde(default = "default_capu_base_model_name")]
    pub base_model_name: String,
    #[serde(default = "default_capu_device")]
    pub device: String,
    #[serde(default = "default_capu_timeout")]
    pub request_timeout_seconds: u64,
}

impl CapuRuntimeConfig {
    pub fn resolved_python_bin(&self, workspace: &WorkspacePaths) -> PathBuf {
        let python_bin = workspace.resolve(&self.python_bin);
        Self::resolve_python_bin_for_os(python_bin, cfg!(target_os = "windows"))
    }

    fn resolve_python_bin_for_os(python_bin: PathBuf, is_windows: bool) -> PathBuf {
        if is_windows {
            let mut python_bin = python_bin;
            let path_str = python_bin.to_string_lossy();
            if path_str.contains("bin/python.exe") {
                python_bin =
                    PathBuf::from(path_str.replace("bin/python.exe", "Scripts/python.exe"));
            } else if path_str.contains("bin\\python.exe") {
                python_bin =
                    PathBuf::from(path_str.replace("bin\\python.exe", "Scripts\\python.exe"));
            } else if path_str.contains("bin/python") {
                python_bin = PathBuf::from(path_str.replace("bin/python", "Scripts/python.exe"));
            } else if path_str.contains("bin\\python") {
                python_bin = PathBuf::from(path_str.replace("bin\\python", "Scripts\\python.exe"));
            } else if python_bin.file_name() == Some(std::ffi::OsStr::new("python")) {
                python_bin.set_file_name("python.exe");
            }
            python_bin
        } else {
            python_bin
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppRuntimeConfig {
    pub models_config_path: PathBuf,
    pub asset_lock_path: PathBuf,
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_num_threads")]
    pub num_threads: i32,
    pub vad: VadConfig,
    pub server: ServerConfig,
    pub capu: CapuRuntimeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub id: String,
    pub language: String,
    pub model_dir: PathBuf,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub num_threads: Option<i32>,
    pub postprocess_mode: PostprocessMode,
    #[serde(default)]
    pub capu_model_id: Option<String>,
    #[serde(default)]
    pub startup_probe_wav_path: Option<PathBuf>,
    #[serde(default)]
    pub startup_probe_expected_text: Option<String>,
    #[serde(default)]
    pub encoder: Option<PathBuf>,
    #[serde(default)]
    pub decoder: Option<PathBuf>,
    #[serde(default)]
    pub joiner: Option<PathBuf>,
    #[serde(default)]
    pub tokens: Option<PathBuf>,
    #[serde(default)]
    pub bpe_vocab: Option<PathBuf>,
    #[serde(default)]
    pub vad_threshold: Option<f32>,
    #[serde(default)]
    pub vad_min_silence: Option<f32>,
    #[serde(default)]
    pub vad_min_speech: Option<f32>,
    #[serde(default)]
    pub vad_max_speech: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub models: Vec<ModelConfig>,
    #[serde(default)]
    pub capu_models: Vec<CapuModelConfig>,
}

#[derive(Debug, Clone)]
pub struct ResolvedModelConfig {
    pub id: String,
    pub language: String,
    pub model_dir: PathBuf,
    pub model_type: String,
    pub provider: String,
    pub num_threads: i32,
    pub postprocess_mode: PostprocessMode,
    pub capu_model_id: Option<String>,
    pub startup_probe_wav_path: Option<PathBuf>,
    pub startup_probe_expected_text: Option<String>,
    pub encoder: Option<PathBuf>,
    pub decoder: Option<PathBuf>,
    pub joiner: Option<PathBuf>,
    pub tokens: Option<PathBuf>,
    pub bpe_vocab: Option<PathBuf>,
    pub vad: VadConfig,
    pub vad_model_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct WorkspacePaths {
    pub root: PathBuf,
    pub models_dir: PathBuf,
}

impl WorkspacePaths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let root = root.canonicalize().unwrap_or(root);
        let models_dir = std::env::var("VIT_STT_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root.join("models"));
        let models_dir = models_dir.canonicalize().unwrap_or(models_dir);
        Self { root, models_dir }
    }

    pub fn discover(
        explicit_root: Option<PathBuf>,
        explicit_models: Option<PathBuf>,
    ) -> Result<Self> {
        let mut searched_paths = Vec::new();

        // 1. Explicit root override
        if let Some(root) = explicit_root {
            searched_paths.push(format!("explicit path: {}", root.display()));
            if root.exists() {
                let root = root.canonicalize().unwrap_or(root);
                let models_dir = explicit_models
                    .clone()
                    .or_else(|| std::env::var("VIT_STT_MODELS_DIR").map(PathBuf::from).ok())
                    .unwrap_or_else(|| root.join("models"));
                let models_dir = models_dir.canonicalize().unwrap_or(models_dir);
                return Ok(Self { root, models_dir });
            }
        }

        // 2. Env variable VIT_STT_RUNTIME_ROOT
        if let Ok(env_root) = std::env::var("VIT_STT_RUNTIME_ROOT") {
            let root = PathBuf::from(env_root);
            searched_paths.push(format!("env VIT_STT_RUNTIME_ROOT: {}", root.display()));
            if root.exists() {
                let root = root.canonicalize().unwrap_or(root);
                let models_dir = explicit_models
                    .clone()
                    .or_else(|| std::env::var("VIT_STT_MODELS_DIR").map(PathBuf::from).ok())
                    .unwrap_or_else(|| root.join("models"));
                let models_dir = models_dir.canonicalize().unwrap_or(models_dir);
                return Ok(Self { root, models_dir });
            }
        }

        // 3. macOS bundle resources relative to the executable
        if let Ok(exe_path) = std::env::current_exe() {
            searched_paths.push(format!(
                "checking macOS bundle relative to exe: {}",
                exe_path.display()
            ));
            if let Some(contents_dir) = exe_path.parent().and_then(|p| p.parent())
                && contents_dir.file_name().and_then(|n| n.to_str()) == Some("Contents")
            {
                let resources_vit = contents_dir.join("Resources/vit-stt");
                searched_paths.push(format!(
                    "macOS bundle Resources/vit-stt: {}",
                    resources_vit.display()
                ));
                if resources_vit.join("config/runtime.toml").exists() {
                    let resources_vit = resources_vit.canonicalize().unwrap_or(resources_vit);
                    let models_dir = explicit_models
                        .clone()
                        .or_else(|| std::env::var("VIT_STT_MODELS_DIR").map(PathBuf::from).ok())
                        .unwrap_or_else(|| resources_vit.join("models"));
                    let models_dir = models_dir.canonicalize().unwrap_or(models_dir);
                    return Ok(Self {
                        root: resources_vit,
                        models_dir,
                    });
                }
                let resources_root = contents_dir.join("Resources");
                searched_paths.push(format!(
                    "macOS bundle Resources: {}",
                    resources_root.display()
                ));
                if resources_root.join("config/runtime.toml").exists() {
                    let resources_root = resources_root.canonicalize().unwrap_or(resources_root);
                    let models_dir = explicit_models
                        .clone()
                        .or_else(|| std::env::var("VIT_STT_MODELS_DIR").map(PathBuf::from).ok())
                        .unwrap_or_else(|| resources_root.join("models"));
                    let models_dir = models_dir.canonicalize().unwrap_or(models_dir);
                    return Ok(Self {
                        root: resources_root,
                        models_dir,
                    });
                }
            }

            // 4. Portable release directory relative to the executable (e.g. bin/../config/runtime.toml)
            if let Some(bin_dir) = exe_path.parent() {
                // Sibling of bin: bin_dir/../config/runtime.toml
                if let Some(portable_root) = bin_dir.parent() {
                    searched_paths.push(format!(
                        "portable release grandparent: {}",
                        portable_root.display()
                    ));
                    if portable_root.join("config/runtime.toml").exists() {
                        let portable_root = portable_root
                            .canonicalize()
                            .unwrap_or_else(|_| portable_root.to_path_buf());
                        let models_dir = explicit_models
                            .clone()
                            .or_else(|| std::env::var("VIT_STT_MODELS_DIR").map(PathBuf::from).ok())
                            .unwrap_or_else(|| portable_root.join("models"));
                        let models_dir = models_dir.canonicalize().unwrap_or(models_dir);
                        return Ok(Self {
                            root: portable_root,
                            models_dir,
                        });
                    }
                }
                // Right next to the executable: bin_dir/config/runtime.toml
                searched_paths.push(format!("portable release parent: {}", bin_dir.display()));
                if bin_dir.join("config/runtime.toml").exists() {
                    let bin_dir = bin_dir
                        .canonicalize()
                        .unwrap_or_else(|_| bin_dir.to_path_buf());
                    let models_dir = explicit_models
                        .clone()
                        .or_else(|| std::env::var("VIT_STT_MODELS_DIR").map(PathBuf::from).ok())
                        .unwrap_or_else(|| bin_dir.join("models"));
                    let models_dir = models_dir.canonicalize().unwrap_or(models_dir);
                    return Ok(Self {
                        root: bin_dir,
                        models_dir,
                    });
                }
            }
        }

        // 5. Development workspace fallback (checking current working directory)
        if let Ok(cwd) = std::env::current_dir() {
            searched_paths.push(format!("current working dir: {}", cwd.display()));
            if cwd.join("config/runtime.toml").exists() {
                let cwd = cwd.canonicalize().unwrap_or(cwd);
                let models_dir = explicit_models
                    .clone()
                    .or_else(|| std::env::var("VIT_STT_MODELS_DIR").map(PathBuf::from).ok())
                    .unwrap_or_else(|| cwd.join("models"));
                let models_dir = models_dir.canonicalize().unwrap_or(models_dir);
                return Ok(Self {
                    root: cwd,
                    models_dir,
                });
            }
        }

        // 6. Development workspace fallback via compile-time CARGO_MANIFEST_DIR
        if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
            let manifest_path = PathBuf::from(manifest_dir);
            if let Some(workspace_dir) = manifest_path.parent().and_then(|p| p.parent()) {
                searched_paths.push(format!(
                    "CARGO_MANIFEST_DIR workspace root: {}",
                    workspace_dir.display()
                ));
                if workspace_dir.join("config/runtime.toml").exists() {
                    let workspace_dir = workspace_dir
                        .canonicalize()
                        .unwrap_or_else(|_| workspace_dir.to_path_buf());
                    let models_dir = explicit_models
                        .clone()
                        .or_else(|| std::env::var("VIT_STT_MODELS_DIR").map(PathBuf::from).ok())
                        .unwrap_or_else(|| workspace_dir.join("models"));
                    let models_dir = models_dir.canonicalize().unwrap_or(models_dir);
                    return Ok(Self {
                        root: workspace_dir,
                        models_dir,
                    });
                }
            }
        }

        // 7. Return diagnostic error
        let error_msg = format!(
            "Could not discover vit-stt runtime root directory containing config/runtime.toml.\n\
             We searched in the following locations:\n  - {}\n\
             Please set the VIT_STT_RUNTIME_ROOT environment variable, or pass a valid directory.",
            searched_paths.join("\n  - ")
        );
        Err(anyhow!(error_msg))
    }

    pub fn resolve(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_path_buf()
        } else if let Ok(stripped) = path.strip_prefix("models") {
            self.models_dir.join(stripped)
        } else {
            self.root.join(path)
        }
    }

    pub fn runtime_config_path(&self) -> PathBuf {
        self.root.join("config/runtime.toml")
    }

    pub fn load_runtime_config(&self, path: Option<&Path>) -> Result<AppRuntimeConfig> {
        let path = path
            .map(|p| self.resolve(p))
            .unwrap_or_else(|| self.runtime_config_path());
        let text = fs::read_to_string(&path)
            .with_context(|| format!("failed reading runtime config: {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("invalid runtime config: {}", path.display()))
    }

    pub fn load_registry(&self, runtime: &AppRuntimeConfig) -> Result<ModelRegistry> {
        let path = self.resolve(&runtime.models_config_path);
        let text = fs::read_to_string(&path)
            .with_context(|| format!("failed reading models config: {}", path.display()))?;
        let registry: ModelRegistry = serde_json::from_str(&text)
            .with_context(|| format!("invalid models config: {}", path.display()))?;
        if registry.models.is_empty() {
            return Err(anyhow!("models config must contain at least one model"));
        }
        Ok(registry)
    }

    pub fn resolve_model(
        &self,
        runtime: &AppRuntimeConfig,
        model: &ModelConfig,
    ) -> ResolvedModelConfig {
        let model_dir = self.resolve(&model.model_dir);
        let resolve_opt = |path: &Option<PathBuf>| path.as_ref().map(|value| self.resolve(value));
        ResolvedModelConfig {
            id: model.id.clone(),
            language: model.language.clone(),
            model_dir: model_dir.clone(),
            model_type: model
                .model_type
                .clone()
                .unwrap_or_else(|| "transducer".to_string()),
            provider: model
                .provider
                .clone()
                .unwrap_or_else(|| runtime.provider.clone()),
            num_threads: model.num_threads.unwrap_or(runtime.num_threads),
            postprocess_mode: model.postprocess_mode,
            capu_model_id: model.capu_model_id.clone(),
            startup_probe_wav_path: resolve_opt(&model.startup_probe_wav_path),
            startup_probe_expected_text: model.startup_probe_expected_text.clone(),
            encoder: resolve_opt(&model.encoder),
            decoder: resolve_opt(&model.decoder),
            joiner: resolve_opt(&model.joiner),
            tokens: resolve_opt(&model.tokens),
            bpe_vocab: resolve_opt(&model.bpe_vocab),
            vad: VadConfig {
                enabled: runtime.vad.enabled,
                profile: runtime.vad.profile.clone(),
                threshold: model.vad_threshold.unwrap_or(runtime.vad.threshold),
                min_silence: model.vad_min_silence.unwrap_or(runtime.vad.min_silence),
                min_speech: model.vad_min_speech.unwrap_or(runtime.vad.min_speech),
                max_speech: model.vad_max_speech.unwrap_or(runtime.vad.max_speech),
            },
            vad_model_path: self.resolve(Path::new("models/vad/ten-vad.int8.onnx")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_resolve_python_bin_for_os() {
        // Test Unix path logic (should be unchanged when not Windows)
        let unix_path = PathBuf::from("capu-worker/.venv/bin/python");
        let res_unix = CapuRuntimeConfig::resolve_python_bin_for_os(unix_path.clone(), false);
        assert_eq!(res_unix, unix_path);

        // Test Windows path adaptation logic
        let win_path_1 = PathBuf::from("capu-worker/.venv/bin/python");
        let res_win_1 = CapuRuntimeConfig::resolve_python_bin_for_os(win_path_1, true);
        assert_eq!(
            res_win_1,
            PathBuf::from("capu-worker/.venv/Scripts/python.exe")
        );

        let win_path_2 = PathBuf::from("capu-worker\\.venv\\bin\\python");
        let res_win_2 = CapuRuntimeConfig::resolve_python_bin_for_os(win_path_2, true);
        assert_eq!(
            res_win_2,
            PathBuf::from("capu-worker\\.venv\\Scripts\\python.exe")
        );

        let win_path_3 = PathBuf::from("python");
        let res_win_3 = CapuRuntimeConfig::resolve_python_bin_for_os(win_path_3, true);
        assert_eq!(res_win_3, PathBuf::from("python.exe"));
    }

    #[test]
    fn resolve_model_uses_models_dir_for_vad_path() {
        let root = tempdir().expect("root tempdir");
        let models_dir = tempdir().expect("models tempdir");
        let workspace = WorkspacePaths {
            root: root.path().to_path_buf(),
            models_dir: models_dir.path().to_path_buf(),
        };
        let runtime = AppRuntimeConfig {
            models_config_path: PathBuf::from("config/models.local.json"),
            asset_lock_path: PathBuf::from("baselines/phase0/assets.lock.json"),
            provider: "cpu".to_string(),
            num_threads: 2,
            vad: VadConfig {
                enabled: true,
                profile: "ten".to_string(),
                threshold: 0.5,
                min_silence: 0.5,
                min_speech: 0.05,
                max_speech: 14.0,
            },
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
                max_upload_mb: 128,
                max_audio_seconds: 14400,
                owned_by: "vit-stt".to_string(),
                warm_start_models: Vec::new(),
                warm_start_timeout_seconds: 60,
            },
            capu: CapuRuntimeConfig {
                engine: "auto".to_string(),
                worker_exe: None,
                python_bin: PathBuf::from("capu-worker/.venv/bin/python"),
                worker_dir: PathBuf::from("capu-worker"),
                model_name: "vibert-capu".to_string(),
                base_model_name: "base_model".to_string(),
                device: "cpu".to_string(),
                request_timeout_seconds: 120,
            },
        };
        let model = ModelConfig {
            id: "test-model".to_string(),
            language: "vi".to_string(),
            model_dir: PathBuf::from("models/stt/test-model"),
            model_type: None,
            provider: None,
            num_threads: None,
            postprocess_mode: PostprocessMode::CleanLower,
            capu_model_id: None,
            startup_probe_wav_path: None,
            startup_probe_expected_text: None,
            encoder: None,
            decoder: None,
            joiner: None,
            tokens: None,
            bpe_vocab: None,
            vad_threshold: None,
            vad_min_silence: None,
            vad_min_speech: None,
            vad_max_speech: None,
        };

        let resolved = workspace.resolve_model(&runtime, &model);

        assert_eq!(
            resolved.vad_model_path,
            models_dir.path().join("vad/ten-vad.int8.onnx")
        );
    }
}
