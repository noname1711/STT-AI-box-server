use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow};
use tempfile::Builder;

#[derive(Debug, Clone)]
pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: i32,
    pub duration_seconds: f32,
}

#[derive(Debug, Clone, Default)]
pub struct AudioProbe {
    pub duration_seconds: Option<f64>,
    pub audio_stream_index: Option<usize>,
    pub codec_name: Option<String>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
}

#[derive(Debug, Clone)]
pub struct AudioNormalizationPolicy {
    pub sample_rate: u32,
    pub channels: u16,
    pub max_duration_seconds: Option<f64>,
    pub max_output_bytes: u64,
}

impl Default for AudioNormalizationPolicy {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            channels: 1,
            max_duration_seconds: None,
            max_output_bytes: 2 * 1024 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NormalizedAudioInfo {
    pub duration_seconds: f64,
    pub sample_rate: u32,
    pub channels: u16,
    pub bytes: u64,
}

#[derive(Debug, serde::Deserialize)]
struct FfprobeOutput {
    #[serde(default)]
    streams: Vec<FfprobeStream>,
    format: Option<FfprobeFormat>,
}

#[derive(Debug, serde::Deserialize)]
struct FfprobeStream {
    index: usize,
    codec_type: Option<String>,
    codec_name: Option<String>,
    sample_rate: Option<String>,
    channels: Option<u16>,
    duration: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct FfprobeFormat {
    duration: Option<String>,
}

/// Resolve the full path to the ffmpeg binary, or return an error with
/// platform-specific installation instructions when ffmpeg cannot be
/// found anywhere.
pub fn resolve_ffmpeg() -> Result<String> {
    // 1. Explicit FFMPEG env var overrides everything
    if let Ok(val) = std::env::var("FFMPEG") {
        let p = Path::new(&val);
        let exists = if cfg!(target_os = "windows") {
            p.is_file() || p.with_extension("exe").is_file()
        } else {
            p.is_file()
        };
        if exists {
            return Ok(val);
        }
    }

    // 2. Check common system installation paths first.
    //    These cover scenarios where PATH is minimal (launchd, systemd, etc.)
    //    while the binary exists at a standard location.
    #[cfg(target_os = "macos")]
    {
        let common_paths = [
            "/opt/homebrew/bin/ffmpeg", // Apple Silicon Homebrew
            "/usr/local/bin/ffmpeg",    // Intel Homebrew
            "/usr/bin/ffmpeg",          // Generic fallback
        ];
        for path in &common_paths {
            if Path::new(path).is_file() {
                return Ok(path.to_string());
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        let common_paths = ["/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"];
        for path in &common_paths {
            if Path::new(path).is_file() {
                return Ok(path.to_string());
            }
        }
    }
    #[cfg(target_os = "windows")]
    {
        // Use environment variables for robust path resolution on Windows
        let prog_data = std::env::var("ProgramData").unwrap_or_else(|_| "C:\\ProgramData".into());
        let prog_files =
            std::env::var("ProgramFiles").unwrap_or_else(|_| "C:\\Program Files".into());
        let common_paths = [
            Path::new(&prog_data)
                .join("chocolatey")
                .join("bin")
                .join("ffmpeg.exe"),
            Path::new(&prog_files)
                .join("ffmpeg")
                .join("bin")
                .join("ffmpeg.exe"),
        ];
        for path in &common_paths {
            if path.is_file() {
                return Ok(path.to_string_lossy().into_owned());
            }
        }
    }

    // 3. Check next to the running binary (bundled deployment)
    if let Ok(exe_path) = std::env::current_exe()
        && let Some(bin_dir) = exe_path.parent()
    {
        #[cfg(target_os = "windows")]
        let local_ffmpeg = bin_dir.join("ffmpeg.exe");
        #[cfg(not(target_os = "windows"))]
        let local_ffmpeg = bin_dir.join("ffmpeg");

        if local_ffmpeg.is_file() {
            return Ok(local_ffmpeg.to_string_lossy().into_owned());
        }
    }

    // 4. Fallback — search PATH directories manually.
    //    This avoids the silent launch failure when PATH doesn't contain ffmpeg.
    let bin_name = if cfg!(target_os = "windows") {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };

    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let candidate = dir.join(bin_name);
            if candidate.is_file() {
                return Ok(candidate.to_string_lossy().into_owned());
            }
        }
    }

    // 5. Nothing found — provide a clear, actionable error
    let install_hint = if cfg!(target_os = "macos") {
        "Install it via Homebrew: `brew install ffmpeg` or download a static build from https://evermeet.cx/ffmpeg/"
    } else if cfg!(target_os = "linux") {
        "Install it via your package manager (e.g. `apt install ffmpeg`, `dnf install ffmpeg`, or `pacman -S ffmpeg`)"
    } else if cfg!(target_os = "windows") {
        "Install it via Chocolatey (`choco install ffmpeg`), Scoop (`scoop install ffmpeg`), or download from https://ffmpeg.org/download.html"
    } else {
        "Install ffmpeg from https://ffmpeg.org/download.html"
    };

    Err(anyhow!(
        "ffmpeg not found. {install_hint}.\n\
         Alternatively, set the FFMPEG environment variable to the full path of the ffmpeg binary."
    ))
}

fn safe_suffix(filename: Option<&str>) -> &'static str {
    match filename
        .and_then(|value| Path::new(value).extension())
        .and_then(OsStr::to_str)
    {
        Some("wav") => ".wav",
        Some("mp3") => ".mp3",
        Some("m4a") => ".m4a",
        Some("flac") => ".flac",
        _ => ".wav",
    }
}

fn parse_seconds(value: Option<&str>) -> Option<f64> {
    value
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
}

pub fn probe_audio(path: &Path) -> Result<AudioProbe> {
    let resolved = path
        .canonicalize()
        .with_context(|| format!("audio path does not exist: {}", path.display()))?;
    let ffmpeg_path = resolve_ffmpeg()?;
    let ffprobe_path = Path::new(&ffmpeg_path)
        .with_file_name(if cfg!(target_os = "windows") {
            "ffprobe.exe"
        } else {
            "ffprobe"
        })
        .to_string_lossy()
        .into_owned();
    let probe_bin = if Path::new(&ffprobe_path).is_file() {
        ffprobe_path
    } else if cfg!(target_os = "windows") {
        "ffprobe.exe".to_string()
    } else {
        "ffprobe".to_string()
    };

    let output = Command::new(&probe_bin)
        .args([
            "-hide_banner",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,sample_rate,channels,duration",
            "-of",
            "json",
            resolved.to_string_lossy().as_ref(),
        ])
        .output()
        .with_context(|| format!("failed to launch ffprobe for {}", resolved.display()))?;

    if !output.status.success() {
        return Err(anyhow!(
            "ffprobe failed for {}: {}",
            resolved.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let parsed: FfprobeOutput =
        serde_json::from_slice(&output.stdout).context("failed to parse ffprobe JSON")?;
    let audio = parsed
        .streams
        .iter()
        .find(|stream| stream.codec_type.as_deref() == Some("audio"));
    let duration_seconds =
        parse_seconds(parsed.format.as_ref().and_then(|f| f.duration.as_deref()))
            .or_else(|| audio.and_then(|stream| parse_seconds(stream.duration.as_deref())));

    Ok(AudioProbe {
        duration_seconds,
        audio_stream_index: audio.map(|stream| stream.index),
        codec_name: audio.and_then(|stream| stream.codec_name.clone()),
        sample_rate: audio
            .and_then(|stream| stream.sample_rate.as_deref())
            .and_then(|value| value.parse::<u32>().ok()),
        channels: audio.and_then(|stream| stream.channels),
    })
}

fn estimated_pcm16_wav_bytes(
    duration_seconds: f64,
    sample_rate: u32,
    channels: u16,
) -> Option<u64> {
    let data_bytes =
        duration_seconds * sample_rate as f64 * channels as f64 * std::mem::size_of::<i16>() as f64;
    if !data_bytes.is_finite() || data_bytes < 0.0 {
        return None;
    }
    Some(data_bytes.ceil() as u64 + 44)
}

pub fn normalize_to_pcm16_wav(
    input: &Path,
    output: &Path,
    policy: &AudioNormalizationPolicy,
) -> Result<NormalizedAudioInfo> {
    let resolved_input = input
        .canonicalize()
        .with_context(|| format!("audio path does not exist: {}", input.display()))?;
    let probe = probe_audio(&resolved_input)?;
    if probe.audio_stream_index.is_none() {
        return Err(anyhow!("audio file has no audio stream"));
    }
    let duration = probe
        .duration_seconds
        .ok_or_else(|| anyhow!("audio duration is unavailable"))?;
    if let Some(max_duration) = policy.max_duration_seconds
        && duration > max_duration
    {
        return Err(anyhow!(
            "audio duration exceeds {max_duration} seconds ({duration:.3} seconds)"
        ));
    }
    let estimated_bytes = estimated_pcm16_wav_bytes(duration, policy.sample_rate, policy.channels)
        .ok_or_else(|| anyhow!("normalized WAV size cannot be estimated"))?;
    if estimated_bytes > policy.max_output_bytes {
        return Err(anyhow!(
            "normalized WAV would be approximately {estimated_bytes} bytes, exceeding the {} byte safety limit",
            policy.max_output_bytes
        ));
    }

    let ffmpeg_path = resolve_ffmpeg()?;
    let status = Command::new(&ffmpeg_path)
        .args([
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            resolved_input.to_string_lossy().as_ref(),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            &policy.channels.to_string(),
            "-ar",
            &policy.sample_rate.to_string(),
            "-c:a",
            "pcm_s16le",
            "-rf64",
            "never",
            output.to_string_lossy().as_ref(),
        ])
        .status()
        .with_context(|| format!("failed to launch ffmpeg for {}", resolved_input.display()))?;

    if !status.success() {
        return Err(anyhow!(
            "ffmpeg failed normalizing {}",
            resolved_input.display()
        ));
    }

    let output_probe = probe_audio(output)?;
    if output_probe.audio_stream_index.is_none() {
        return Err(anyhow!("normalized WAV has no audio stream"));
    }
    if output_probe.codec_name.as_deref() != Some("pcm_s16le") {
        return Err(anyhow!(
            "normalized WAV codec mismatch: expected pcm_s16le, got {}",
            output_probe.codec_name.as_deref().unwrap_or("unknown")
        ));
    }
    if output_probe.sample_rate != Some(policy.sample_rate) {
        return Err(anyhow!(
            "normalized WAV sample rate mismatch: expected {}, got {:?}",
            policy.sample_rate,
            output_probe.sample_rate
        ));
    }
    if output_probe.channels != Some(policy.channels) {
        return Err(anyhow!(
            "normalized WAV channel mismatch: expected {}, got {:?}",
            policy.channels,
            output_probe.channels
        ));
    }

    let bytes = fs::metadata(output)
        .with_context(|| format!("failed to stat normalized WAV {}", output.display()))?
        .len();
    if bytes > policy.max_output_bytes {
        return Err(anyhow!(
            "normalized WAV is {bytes} bytes, exceeding the {} byte safety limit",
            policy.max_output_bytes
        ));
    }

    Ok(NormalizedAudioInfo {
        duration_seconds: output_probe.duration_seconds.unwrap_or(duration),
        sample_rate: policy.sample_rate,
        channels: policy.channels,
        bytes,
    })
}

fn decode_via_ffmpeg(input: &Path) -> Result<DecodedAudio> {
    let ffmpeg_path = resolve_ffmpeg()?;
    let output = Command::new(&ffmpeg_path)
        .args([
            "-nostdin",
            "-v",
            "error",
            "-i",
            input.to_string_lossy().as_ref(),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-f",
            "f32le",
            "-c:a",
            "pcm_f32le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "pipe:1",
        ])
        .output()
        .with_context(|| format!("failed to launch ffmpeg for {}", input.display()))?;

    if !output.status.success() {
        return Err(anyhow!(
            "ffmpeg failed for {}: {}",
            input.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let samples = output
        .stdout
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4-byte chunk")))
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return Err(anyhow!("decoded audio is empty for {}", input.display()));
    }

    Ok(DecodedAudio {
        duration_seconds: samples.len() as f32 / 16_000.0,
        sample_rate: 16_000,
        samples,
    })
}

pub fn decode_to_f32_mono_16k(path: &Path) -> Result<DecodedAudio> {
    let resolved = path
        .canonicalize()
        .with_context(|| format!("audio path does not exist: {}", path.display()))?;
    decode_via_ffmpeg(&resolved)
}

pub fn decode_audio_path(path: &Path) -> Result<DecodedAudio> {
    decode_to_f32_mono_16k(path)
}

pub fn decode_audio_bytes(data: &[u8], filename: Option<&str>) -> Result<DecodedAudio> {
    let mut temp_file = Builder::new().suffix(safe_suffix(filename)).tempfile()?;
    temp_file.write_all(data)?;
    let path: PathBuf = temp_file.path().to_path_buf();
    temp_file.flush()?;
    let decoded = decode_via_ffmpeg(&path)?;
    fs::remove_file(path).ok();
    Ok(decoded)
}

pub fn copy_wav_segment(
    input: &Path,
    output: &Path,
    start_seconds: f32,
    duration_seconds: f32,
) -> Result<()> {
    let ffmpeg_path = resolve_ffmpeg()?;
    let status = Command::new(&ffmpeg_path)
        .args([
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            input.to_string_lossy().as_ref(),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ss",
            &format!("{start_seconds:.3}"),
            "-t",
            &format!("{duration_seconds:.3}"),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            output.to_string_lossy().as_ref(),
        ])
        .status()?;

    if !status.success() {
        return Err(anyhow!(
            "ffmpeg failed extracting segment from {}",
            input.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_test_wav(path: &Path, sample_rate: u32, channels: u16, frames: u32) {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).expect("create wav");
        for frame in 0..frames {
            for channel in 0..channels {
                let sample = if channel == 0 {
                    (frame % 512) as i16
                } else {
                    -((frame % 512) as i16)
                };
                writer.write_sample(sample).expect("write sample");
            }
        }
        writer.finalize().expect("finalize wav");
    }

    #[test]
    fn probe_audio_reports_first_audio_stream() {
        if resolve_ffmpeg().is_err() {
            return;
        }
        let temp = tempfile::tempdir().expect("tempdir");
        let input = temp.path().join("input.wav");
        write_test_wav(&input, 48_000, 2, 48_000);

        let probe = probe_audio(&input).expect("probe audio");

        assert_eq!(probe.audio_stream_index, Some(0));
        assert_eq!(probe.codec_name.as_deref(), Some("pcm_s16le"));
        assert_eq!(probe.sample_rate, Some(48_000));
        assert_eq!(probe.channels, Some(2));
        assert!(probe.duration_seconds.unwrap_or_default() > 0.9);
    }

    #[test]
    fn normalize_to_pcm16_wav_writes_seekable_16k_mono_wav() {
        if resolve_ffmpeg().is_err() {
            return;
        }
        let temp = tempfile::tempdir().expect("tempdir");
        let input = temp.path().join("input.wav");
        let output = temp.path().join("output.wav");
        write_test_wav(&input, 48_000, 2, 48_000);

        let info = normalize_to_pcm16_wav(
            &input,
            &output,
            &AudioNormalizationPolicy {
                max_output_bytes: 10 * 1024 * 1024,
                ..AudioNormalizationPolicy::default()
            },
        )
        .expect("normalize");

        assert_eq!(info.sample_rate, 16_000);
        assert_eq!(info.channels, 1);
        assert!(info.bytes > 44);

        let bytes = fs::read(&output).expect("read output");
        assert!(bytes.starts_with(b"RIFF"));
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_ne!(&bytes[4..8], &[0xff, 0xff, 0xff, 0xff]);

        let probe = probe_audio(&output).expect("probe normalized");
        assert_eq!(probe.codec_name.as_deref(), Some("pcm_s16le"));
        assert_eq!(probe.sample_rate, Some(16_000));
        assert_eq!(probe.channels, Some(1));
    }

    #[test]
    fn normalize_to_pcm16_wav_rejects_missing_audio_stream() {
        if resolve_ffmpeg().is_err() {
            return;
        }
        let temp = tempfile::tempdir().expect("tempdir");
        let input = temp.path().join("not-audio.txt");
        let output = temp.path().join("output.wav");
        fs::write(&input, b"not audio").expect("write text");

        let err = normalize_to_pcm16_wav(&input, &output, &AudioNormalizationPolicy::default())
            .expect_err("text file should fail");
        assert!(
            err.to_string().contains("ffprobe failed")
                || err.to_string().contains("no audio stream")
        );
    }
}
