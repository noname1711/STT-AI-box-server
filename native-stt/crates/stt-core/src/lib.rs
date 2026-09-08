pub mod assets;
pub mod audio;
pub mod checksum;
pub mod config;
pub mod postprocess;
pub mod recognizer;
pub mod runtime;
pub mod types;

pub use assets::{OfflineModelAssets, resolve_offline_model_assets};
pub use audio::{
    AudioNormalizationPolicy, AudioProbe, DecodedAudio, NormalizedAudioInfo, decode_audio_bytes,
    decode_audio_path, decode_to_f32_mono_16k, normalize_to_pcm16_wav, probe_audio, resolve_ffmpeg,
};
pub use checksum::{AssetLock, AssetLockEntry};
pub use config::{
    AppRuntimeConfig, CapuModelConfig, CapuRuntimeConfig, ModelConfig, ModelRegistry,
    ResolvedModelConfig, ServerConfig, VadConfig, WorkspacePaths,
};
pub use postprocess::{
    BuiltinPostprocessor, NoopPostprocessor, NormalizeTextPostprocessor, PostprocessMode,
    Postprocessor, clean_text, finalize_text, normalize_text,
};
pub use recognizer::{RecognizerRuntime, WarmedRecognizerRuntime};
pub use runtime::{ModelStatus, ProbeResult, SttRuntime, WarmupResult};
pub use types::{ModelSummary, RecognizedSegment, TranscriptionResult};
