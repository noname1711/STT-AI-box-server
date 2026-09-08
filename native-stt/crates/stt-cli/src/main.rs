use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command as StdCommand, Stdio};
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing_subscriber::EnvFilter;

use stt_capu::build_capu_postprocessor;
use stt_core::SttRuntime;
use stt_core::postprocess::{BuiltinPostprocessor, Postprocessor};

#[derive(Parser)]
#[command(version, about = "vit-stt CLI")]
struct Cli {
    #[arg(long)]
    workspace_root: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    #[command(about = "List the configured production model IDs")]
    ListModels,
    #[command(about = "Run a locked startup probe for a configured model")]
    Probe {
        #[arg(
            long,
            help = "Model ID to probe, for example vit_stt_vi_v2 or vit_stt_en_v2"
        )]
        model: Option<String>,
    },
    #[command(about = "Transcribe one audio file with a configured model")]
    TranscribeWav {
        #[arg(
            long,
            help = "Model ID to use, for example vit_stt_vi_v2 or vit_stt_en_v2"
        )]
        model: Option<String>,
        path: PathBuf,
        #[arg(long, default_value = "json", help = "Output format: json or text")]
        response_format: String,
    },
    #[command(about = "Run Vietnamese CAPU postprocessing on text")]
    PostprocessCapu { text: String },
    #[command(about = "Benchmark one audio file with a configured model")]
    Benchmark {
        #[arg(long, help = "Model ID to benchmark")]
        model: Option<String>,
        path: PathBuf,
    },
    #[command(about = "Download locked model assets from baselines/phase0/assets.lock.json")]
    DownloadModels {
        #[arg(
            long,
            help = "Re-download even assets whose checksums are already valid"
        )]
        force: bool,
    },
    #[command(about = "Verify checksums for every locked asset")]
    VerifyAssets,
    #[command(about = "Print runtime diagnostics without downloading models")]
    Doctor,
    #[command(about = "Verify the CAPU runtime and fixture output")]
    CheckCapu,
    #[command(about = "Prepare prerequisites and CAPU runtime; does not download model weights")]
    Setup,
    #[command(about = "Manage the optional stt-http system service")]
    Service {
        #[command(subcommand)]
        action: ServiceAction,
    },
    #[command(about = "Manage the optional llama-server LaunchDaemon")]
    LlamaService {
        #[command(subcommand)]
        action: LlamaServiceAction,
    },
}

/// Minimum required Python major version
const PYTHON_MIN_MAJOR: u32 = 3;
/// Minimum required Python minor version
const PYTHON_MIN_MINOR: u32 = 11;

/// Check whether a given Python binary meets the minimum required version.
fn python_version_ok(python: &PathBuf) -> bool {
    let output = match StdCommand::new(python)
        .arg("--version")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
    {
        Ok(o) => o,
        Err(_) => return false,
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // `python --version` prints to stdout or stderr depending on the version
    let combined = format!("{}{}", stdout, stderr);
    // Expected: "Python 3.11.x", "Python 3.12.x", etc.
    if let Some(ver_str) = combined.strip_prefix("Python ") {
        let parts: Vec<&str> = ver_str.trim().split('.').collect();
        if parts.len() >= 2
            && let (Ok(major), Ok(minor)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>())
        {
            return major > PYTHON_MIN_MAJOR
                || (major == PYTHON_MIN_MAJOR && minor >= PYTHON_MIN_MINOR);
        }
    }
    false
}

fn find_system_python() -> Option<PathBuf> {
    // 1. Try specific versioned executables (most precise)
    for ver in ["3.11", "3.12", "3.13", "3.14"] {
        let name = format!("python{}", ver);
        if StdCommand::new(&name)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok()
        {
            return Some(PathBuf::from(name));
        }
    }

    // 2. Try generic "python3" – but only if its version meets the requirement
    let python3 = PathBuf::from("python3");
    if python_version_ok(&python3) {
        return Some(python3);
    }

    // 3. Try generic "python" – only if its version meets the requirement
    let python = PathBuf::from("python");
    if python_version_ok(&python) {
        return Some(python);
    }

    // 4. Windows specific defaults (no version check needed – paths encode version)
    #[cfg(target_os = "windows")]
    {
        let common_paths = [
            "C:\\Program Files\\Python311\\python.exe",
            "C:\\Program Files\\Python312\\python.exe",
            "C:\\Program Files\\Python313\\python.exe",
            "C:\\Program Files\\Python310\\python.exe",
        ];
        for path in &common_paths {
            let p = PathBuf::from(path);
            if p.exists() {
                return Some(p);
            }
        }
    }

    // 5. macOS Homebrew locations (for manually installed Python >=3.11)
    #[cfg(target_os = "macos")]
    {
        let brew_paths = [
            PathBuf::from("/opt/homebrew/bin/python3.11"),
            PathBuf::from("/opt/homebrew/bin/python3.12"),
            PathBuf::from("/opt/homebrew/bin/python3.13"),
            PathBuf::from("/opt/homebrew/bin/python3.14"),
            PathBuf::from("/usr/local/bin/python3.11"),
            PathBuf::from("/usr/local/bin/python3.12"),
            PathBuf::from("/usr/local/bin/python3.13"),
            PathBuf::from("/usr/local/bin/python3.14"),
        ];
        for p in &brew_paths {
            if p.exists() {
                return Some(p.clone());
            }
        }
    }

    None
}

fn sys_prefers_vietnamese() -> bool {
    true
}

#[allow(dead_code)]
fn install_python_via_winget(vi: bool) -> Result<()> {
    if vi {
        println!("Đang cài đặt Python 3.11 qua winget...");
    } else {
        println!("Installing Python 3.11 via winget...");
    }
    let status = StdCommand::new("winget")
        .args([
            "install",
            "-e",
            "--id",
            "Python.Python.3.11",
            "--silent",
            "--accept-source-agreements",
            "--accept-package-agreements",
        ])
        .status()?;
    if !status.success() {
        if vi {
            return Err(anyhow::anyhow!("Cài đặt Python 3.11 qua winget thất bại"));
        } else {
            return Err(anyhow::anyhow!("winget installation of Python 3.11 failed"));
        }
    }
    if vi {
        println!("Python 3.11 đã được cài đặt thành công qua winget!");
    } else {
        println!("Python 3.11 successfully installed via winget!");
    }
    Ok(())
}

fn check_ffmpeg_installed() -> bool {
    check_ffmpeg_normalization().is_ok()
}

fn check_ffmpeg_normalization() -> Result<String> {
    let ffmpeg = stt_core::resolve_ffmpeg()?;
    let output = StdCommand::new(&ffmpeg)
        .args([
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-t",
            "0.01",
            "-f",
            "null",
            "-",
        ])
        .stdin(Stdio::null())
        .output()?;
    if !output.status.success() {
        anyhow::bail!(
            "FFmpeg synthetic conversion failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(ffmpeg)
}

async fn download_ffmpeg(bin_dir: &std::path::Path, vi: bool) -> Result<()> {
    let client = reqwest::Client::new();
    let url = if cfg!(target_os = "windows") {
        "https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v6.1/ffmpeg-6.1-win-64.zip"
    } else if cfg!(target_os = "macos") {
        "https://evermeet.cx/ffmpeg/get/zip"
    } else {
        "https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v6.1/ffmpeg-6.1-linux-64.zip"
    };

    if vi {
        println!("Đang tải xuống FFmpeg static build từ {}...", url);
    } else {
        println!("Downloading FFmpeg static build from {}...", url);
    }
    let response = client.get(url).send().await?;
    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Failed to download FFmpeg: HTTP status {}",
            response.status()
        ));
    }

    let temp_dir = tempfile::tempdir()?;
    let temp_zip_path = temp_dir.path().join("ffmpeg.zip");

    let mut file = fs::File::create(&temp_zip_path)?;
    let mut stream = response.bytes_stream();
    use futures::StreamExt;
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        file.write_all(&chunk)?;
    }
    file.sync_all()?;

    if vi {
        println!("Đang giải nén gói FFmpeg...");
    } else {
        println!("Extracting FFmpeg package...");
    }
    #[cfg(target_os = "windows")]
    {
        let status = StdCommand::new("powershell")
            .args([
                "-Command",
                &format!(
                    "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                    temp_zip_path.display(),
                    temp_dir.path().display()
                ),
            ])
            .status()?;
        if !status.success() {
            return Err(anyhow::anyhow!(
                "Failed to extract FFmpeg zip via PowerShell"
            ));
        }

        let ffmpeg_exe = temp_dir.path().join("ffmpeg.exe");
        if !ffmpeg_exe.exists() {
            return Err(anyhow::anyhow!("ffmpeg.exe not found in extracted files"));
        }
        fs::copy(&ffmpeg_exe, bin_dir.join("ffmpeg.exe"))?;
    }

    #[cfg(not(target_os = "windows"))]
    {
        let status = StdCommand::new("unzip")
            .args([
                "-o",
                &temp_zip_path.to_string_lossy(),
                "-d",
                &temp_dir.path().to_string_lossy(),
            ])
            .status()?;
        if !status.success() {
            return Err(anyhow::anyhow!("Failed to extract FFmpeg zip via unzip"));
        }

        let ffmpeg_bin = temp_dir.path().join("ffmpeg");
        if !ffmpeg_bin.exists() {
            return Err(anyhow::anyhow!(
                "ffmpeg binary not found in extracted files"
            ));
        }
        let target_ffmpeg = bin_dir.join("ffmpeg");
        fs::copy(&ffmpeg_bin, &target_ffmpeg)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&target_ffmpeg, fs::Permissions::from_mode(0o755))?;
        }
    }

    if vi {
        println!(
            "FFmpeg đã được cài đặt thành công tại {}!",
            bin_dir.display()
        );
    } else {
        println!("FFmpeg successfully installed in {}!", bin_dir.display());
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let vi = sys_prefers_vietnamese();

    macro_rules! lp {
        ($en:expr, $vi_text:expr) => {
            if vi {
                println!("{}", $vi_text);
            } else {
                println!("{}", $en);
            }
        };
        ($en:expr, $vi_text:expr, $($arg:tt)*) => {
            if vi {
                println!($vi_text, $($arg)*);
            } else {
                println!($en, $($arg)*);
            }
        };
    }

    // Handle service subcommands that don't need the runtime first.
    // SttRuntime::load() validates all asset checksums (~40s), so we skip
    // it for commands that only query/manage the service without models.
    if let Command::Service { action } = &cli.command {
        match action {
            ServiceAction::Status => return handle_service_status().await,
            ServiceAction::Stop => return handle_service_stop().await,
            ServiceAction::Start => return handle_service_start().await,
            ServiceAction::Uninstall => return handle_service_uninstall().await,
            ServiceAction::Logs { lines, follow } => {
                return handle_service_logs(cli.workspace_root.as_deref(), *lines, *follow).await;
            }
            _ => {}
        }
    }
    if let Command::LlamaService { action } = cli.command.clone() {
        return handle_llama_service(cli.workspace_root.as_deref(), action).await;
    }

    let runtime = SttRuntime::load(cli.workspace_root, None)?;

    match cli.command {
        Command::ListModels => {
            println!("{}", serde_json::to_string_pretty(&runtime.list_models())?);
        }
        Command::Probe { model } => {
            let probe = runtime.run_probe(model.as_deref())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "model_id": probe.model_id,
                    "probe_path": probe.probe_path,
                    "expected": probe.expected,
                    "actual": probe.actual,
                    "matches_expected": probe.matches_expected,
                }))?
            );
        }
        Command::TranscribeWav {
            model,
            path,
            response_format,
        } => {
            let model_id = model.unwrap_or_else(|| runtime.default_model_id().unwrap().to_string());
            let model_cfg = runtime.resolve_model(&model_id)?;
            let result = runtime.transcribe_path(&model_id, &path)?;
            let result = if model_cfg.postprocess_mode
                == stt_core::postprocess::PostprocessMode::Capu
            {
                let capu_model = runtime.resolve_capu_model(model_cfg.capu_model_id.as_deref())?;
                let (_, capu) = build_capu_postprocessor(
                    &runtime.workspace,
                    &runtime.runtime_config.capu,
                    capu_model,
                )?;
                capu.process_result(&result)?
            } else {
                BuiltinPostprocessor::new(model_cfg.postprocess_mode).process_result(&result)?
            };

            match response_format.as_str() {
                "text" => println!("{}", result.text),
                _ => println!("{}", serde_json::to_string_pretty(&result)?),
            }
        }
        Command::PostprocessCapu { text } => {
            let capu_model = runtime.resolve_capu_model(None)?;
            let (_, capu) = build_capu_postprocessor(
                &runtime.workspace,
                &runtime.runtime_config.capu,
                capu_model,
            )?;
            println!("{}", capu.process_text(&text)?);
        }
        Command::Benchmark { model, path } => {
            let model_id = model.unwrap_or_else(|| runtime.default_model_id().unwrap().to_string());
            let started = Instant::now();
            let result = runtime.transcribe_path(&model_id, &path)?;
            let elapsed = started.elapsed().as_secs_f32();
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "model": model_id,
                    "audio_duration_seconds": result.duration,
                    "decode_seconds": result.processing_time,
                    "wall_seconds": elapsed,
                    "rtf": if result.duration > 0.0 { elapsed / result.duration } else { 0.0 },
                    "text": result.text,
                }))?
            );
        }
        Command::DownloadModels { force } => {
            let lock_path = runtime
                .workspace
                .resolve(&runtime.runtime_config.asset_lock_path);
            let asset_lock = stt_core::checksum::AssetLock::load(&lock_path)?;
            lp!(
                "Checking for missing assets to download...",
                "Đang kiểm tra các tài nguyên còn thiếu để tải xuống..."
            );
            asset_lock
                .download_missing_assets(&runtime.workspace, force)
                .await?;
            lp!(
                "All missing assets downloaded successfully!",
                "Đã tải xuống tất cả tài nguyên còn thiếu thành công!"
            );
        }
        Command::VerifyAssets => {
            let lock_path = runtime
                .workspace
                .resolve(&runtime.runtime_config.asset_lock_path);
            let asset_lock = stt_core::checksum::AssetLock::load(&lock_path)?;
            let report = asset_lock.verify_all(&runtime.workspace);
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::CheckCapu => {
            lp!(
                "Checking CAPU postprocessing setup...",
                "Đang kiểm tra thiết lập hậu kỳ CAPU..."
            );
            let started = Instant::now();
            let capu_model = runtime.resolve_capu_model(None)?;
            let (_, capu) = build_capu_postprocessor(
                &runtime.workspace,
                &runtime.runtime_config.capu,
                capu_model,
            )?;
            let text = "rồi cũng hỗ trợ cho lâu lâu cũng cho gạo cho này kia";
            lp!(
                "Sending text for postprocessing: '{}'",
                "Gửi văn bản xử lý hậu kỳ: '{}'",
                text
            );
            let output = capu.process_text(text)?;
            lp!("CAPU Output: '{}'", "Kết quả CAPU: '{}'", output);
            lp!(
                "Time taken: {:.2}s",
                "Thời gian xử lý: {:.2}s",
                started.elapsed().as_secs_f32()
            );
            if output == "Rồi cũng hỗ trợ cho, lâu lâu cũng cho gạo cho này kia." {
                lp!(
                    "CAPU Check: SUCCESS (output matches expectation)",
                    "Kiểm tra CAPU: THÀNH CÔNG (kết quả khớp với mong đợi)"
                );
            } else {
                lp!(
                    "CAPU Check: FAILED (output mismatch)",
                    "Kiểm tra CAPU: THẤT BẠI (kết quả không khớp mong đợi)"
                );
            }
        }
        Command::Doctor => {
            lp!("--- System Diagnostics ---", "--- Chẩn đoán hệ thống ---");
            lp!(
                "Runtime root: {}",
                "Thư mục gốc runtime: {}",
                runtime.workspace.root.display()
            );
            lp!(
                "Models directory: {}",
                "Thư mục chứa model: {}",
                runtime.workspace.models_dir.display()
            );

            if let Ok(env_root) = std::env::var("VIT_STT_RUNTIME_ROOT") {
                lp!(
                    "VIT_STT_RUNTIME_ROOT (env): {}",
                    "VIT_STT_RUNTIME_ROOT (môi trường): {}",
                    env_root
                );
            } else {
                lp!(
                    "VIT_STT_RUNTIME_ROOT (env): <not set>",
                    "VIT_STT_RUNTIME_ROOT (môi trường): <chưa thiết lập>"
                );
            }

            if let Ok(env_models) = std::env::var("VIT_STT_MODELS_DIR") {
                lp!(
                    "VIT_STT_MODELS_DIR (env): {}",
                    "VIT_STT_MODELS_DIR (môi trường): {}",
                    env_models
                );
            } else {
                lp!(
                    "VIT_STT_MODELS_DIR (env): <not set>",
                    "VIT_STT_MODELS_DIR (môi trường): <chưa thiết lập>"
                );
            }

            match check_ffmpeg_normalization() {
                Ok(path) => lp!(
                    "FFmpeg normalization probe: SUCCESS ({})",
                    "Kiểm tra chuẩn hóa FFmpeg: THÀNH CÔNG ({})",
                    path
                ),
                Err(err) => lp!(
                    "FFmpeg normalization probe: FAILED ({})",
                    "Kiểm tra chuẩn hóa FFmpeg: THẤT BẠI ({})",
                    err
                ),
            }

            lp!(
                "\n--- Registered Models ---",
                "\n--- Các Model đã đăng ký ---"
            );
            for m in &runtime.registry.models {
                let resolved = runtime.workspace.resolve(&m.model_dir);
                lp!(
                    "- Model: {} (Language: {})",
                    "- Model: {} (Ngôn ngữ: {})",
                    m.id,
                    m.language
                );
                lp!("  Directory: {}", "  Thư mục: {}", resolved.display());
                lp!("  Exists: {}", "  Tồn tại: {}", resolved.exists());
            }

            lp!(
                "\n--- Asset Checksum Verification ---",
                "\n--- Xác minh mã kiểm tra Checksum của tài nguyên ---"
            );
            let lock_path = runtime
                .workspace
                .resolve(&runtime.runtime_config.asset_lock_path);
            let asset_lock = stt_core::checksum::AssetLock::load(&lock_path)?;
            let report = asset_lock.verify_all(&runtime.workspace);

            let mut missing_count = 0;
            let mut corrupted_count = 0;
            let mut valid_count = 0;

            for e in &report.entries {
                match e.status {
                    stt_core::checksum::AssetStatus::Valid => valid_count += 1,
                    stt_core::checksum::AssetStatus::Missing => {
                        missing_count += 1;
                        lp!(
                            "  [MISSING] {} at {}",
                            "  [THIẾU] {} tại {}",
                            e.id,
                            e.path.display()
                        );
                    }
                    stt_core::checksum::AssetStatus::Corrupted { .. } => {
                        corrupted_count += 1;
                        lp!(
                            "  [CORRUPTED] {} at {}",
                            "  [BỊ LỖI] {} tại {}",
                            e.id,
                            e.path.display()
                        );
                    }
                }
            }

            lp!(
                "Verification results: {} valid, {} missing, {} corrupted",
                "Kết quả xác minh: {} hợp lệ, {} thiếu, {} bị lỗi",
                valid_count,
                missing_count,
                corrupted_count
            );

            lp!(
                "\n--- CAPU Worker Probe ---",
                "\n--- Kiểm tra CAPU Worker ---"
            );
            let capu_model = runtime.resolve_capu_model(None)?;
            let spike = stt_capu::probe_onnx_or_pure_rust_support(&runtime.workspace, capu_model);
            lp!("CAPU model: {}", "Model CAPU: {}", capu_model.id);
            lp!(
                "CAPU ONNX support: {}",
                "Hỗ trợ ONNX CAPU: {}",
                spike.supported
            );
            lp!("Reason: {}", "Lý do: {}", spike.reason);

            if runtime.runtime_config.capu.worker_exe.is_some()
                || runtime
                    .runtime_config
                    .capu
                    .resolved_python_bin(&runtime.workspace)
                    .exists()
            {
                match build_capu_postprocessor(
                    &runtime.workspace,
                    &runtime.runtime_config.capu,
                    capu_model,
                ) {
                    Ok((_, capu)) => match capu.process_text("test") {
                        Ok(_) => lp!(
                            "CAPU Handshake/Probe: SUCCESS",
                            "Kiểm tra liên kết CAPU: THÀNH CÔNG"
                        ),
                        Err(err) => lp!(
                            "CAPU Handshake/Probe: FAILED ({})",
                            "Kiểm tra liên kết CAPU: THẤT BẠI ({})",
                            err
                        ),
                    },
                    Err(err) => {
                        lp!(
                            "CAPU Worker Spawn: FAILED ({})",
                            "Khởi chạy CAPU Worker: THẤT BẠI ({})",
                            err
                        );
                    }
                }
            } else {
                lp!(
                    "CAPU worker check skipped: no python or worker exe found",
                    "Bỏ qua kiểm tra CAPU worker: không tìm thấy tệp python hoặc worker exe"
                );
            }
        }
        Command::Setup => {
            lp!(
                "=== Starting Automated Setup Wizard ===",
                "=== Đang chạy Trình hướng dẫn thiết lập tự động ==="
            );
            lp!(
                "Setup prepares prerequisites only. It does not download model weights; run `stt-cli download-models` explicitly after setup.",
                "Setup chỉ chuẩn bị điều kiện chạy. Lệnh này không tải trọng số model; hãy chạy rõ ràng `stt-cli download-models` sau setup."
            );
            let root = &runtime.workspace.root;
            let bin_dir = std::env::current_exe()?
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| root.join("bin"));
            fs::create_dir_all(&bin_dir)?;

            // 1. Check/Install Python
            let python_bin = {
                #[allow(unused_mut)]
                let mut p = find_system_python();
                #[cfg(target_os = "windows")]
                if p.is_none() {
                    lp!(
                        "Python was not found in PATH or standard directories.",
                        "Không tìm thấy Python trong PATH hoặc các thư mục chuẩn."
                    );
                    if install_python_via_winget(vi).is_ok() {
                        p = find_system_python();
                    }
                }
                p
            };

            let python_bin = match python_bin {
                Some(p) => p,
                None => {
                    #[cfg(target_os = "windows")]
                    {
                        if vi {
                            return Err(anyhow::anyhow!(
                                "Không tìm thấy hoặc không thể cài đặt Python. Vui lòng cài đặt thủ công Python 3.11-3.13."
                            ));
                        } else {
                            return Err(anyhow::anyhow!(
                                "Failed to locate or install Python. Please install Python 3.11-3.13 manually."
                            ));
                        }
                    }
                    #[cfg(target_os = "macos")]
                    {
                        if vi {
                            return Err(anyhow::anyhow!(
                                "Không tìm thấy Python. Vui lòng cài đặt thủ công Python 3.11-3.13 (ví dụ sử dụng `brew install python@3.11`)."
                            ));
                        } else {
                            return Err(anyhow::anyhow!(
                                "Python not found. Please install Python 3.11-3.13 manually (e.g. using `brew install python@3.11`)."
                            ));
                        }
                    }
                    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
                    {
                        if vi {
                            return Err(anyhow::anyhow!(
                                "Không tìm thấy Python. Vui lòng cài đặt thủ công Python 3.11-3.13 (ví dụ sử dụng `sudo apt-get install python3-venv python3-pip`)."
                            ));
                        } else {
                            return Err(anyhow::anyhow!(
                                "Python not found. Please install Python 3.11-3.13 manually (e.g. using `sudo apt-get install python3-venv python3-pip`)."
                            ));
                        }
                    }
                }
            };
            lp!(
                "Using Python binary: {}",
                "Sử dụng đường dẫn Python: {}",
                python_bin.display()
            );

            // 2. Check/Install FFmpeg
            if !check_ffmpeg_installed() {
                lp!(
                    "FFmpeg was not found on your system.",
                    "Không tìm thấy FFmpeg trên hệ thống của bạn."
                );
                download_ffmpeg(&bin_dir, vi).await?;
            } else {
                lp!(
                    "FFmpeg is already installed and available.",
                    "FFmpeg đã được cài đặt và sẵn sàng sử dụng."
                );
            }

            // 3. CAPU virtual environment setup
            let worker_dir = runtime
                .workspace
                .resolve(&runtime.runtime_config.capu.worker_dir);
            let venv_dir = worker_dir.join(".venv");
            if !venv_dir.exists() {
                lp!(
                    "Creating local virtual environment at {}...",
                    "Đang tạo môi trường ảo Python cục bộ tại {}...",
                    venv_dir.display()
                );

                let use_uv = StdCommand::new("uv")
                    .arg("--version")
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
                    .is_ok();

                let status = if use_uv {
                    StdCommand::new("uv")
                        .args(["venv", ".venv"])
                        .current_dir(&worker_dir)
                        .status()?
                } else {
                    StdCommand::new(&python_bin)
                        .args(["-m", "venv", ".venv"])
                        .current_dir(&worker_dir)
                        .status()?
                };

                if !status.success() {
                    if vi {
                        return Err(anyhow::anyhow!("Tạo môi trường ảo Python thất bại"));
                    } else {
                        return Err(anyhow::anyhow!(
                            "Failed to create Python virtual environment"
                        ));
                    }
                }
            } else {
                lp!(
                    "Python virtual environment already exists at {}.",
                    "Môi trường ảo Python đã tồn tại tại {}.",
                    venv_dir.display()
                );
            }

            // Install python worker package
            lp!(
                "Installing/upgrading CAPU python package dependencies...",
                "Đang cài đặt/nâng cấp các gói phụ thuộc Python cho CAPU..."
            );
            let pip_exe = if cfg!(target_os = "windows") {
                venv_dir.join("Scripts/pip.exe")
            } else {
                venv_dir.join("bin/pip")
            };

            let use_uv = StdCommand::new("uv")
                .arg("--version")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok();

            let status = if use_uv {
                StdCommand::new("uv")
                    .args(["pip", "install", "-e", ".", "huggingface_hub"])
                    .current_dir(&worker_dir)
                    .status()?
            } else {
                let _ = StdCommand::new(&pip_exe)
                    .args(["install", "--upgrade", "pip"])
                    .current_dir(&worker_dir)
                    .status();

                StdCommand::new(&pip_exe)
                    .args(["install", "-e", ".", "huggingface_hub"])
                    .current_dir(&worker_dir)
                    .status()?
            };

            if !status.success() {
                if vi {
                    return Err(anyhow::anyhow!("Cài đặt các phụ thuộc Python thất bại"));
                } else {
                    return Err(anyhow::anyhow!("Failed to install Python dependencies"));
                }
            }
            lp!(
                "Python dependencies successfully installed!",
                "Các gói phụ thuộc Python đã được cài đặt thành công!"
            );

            // 4. Check whether model assets are ready. Setup intentionally does
            // not download model weights; use `download-models` for that.
            let lock_path = runtime
                .workspace
                .resolve(&runtime.runtime_config.asset_lock_path);
            let asset_lock = stt_core::checksum::AssetLock::load(&lock_path)?;
            let report = asset_lock.verify_all(&runtime.workspace);
            let missing_or_corrupt = report
                .entries
                .iter()
                .filter(|entry| !matches!(entry.status, stt_core::checksum::AssetStatus::Valid))
                .count();
            if missing_or_corrupt > 0 {
                lp!(
                    "Model asset check: {} missing or corrupted assets. Run `stt-cli download-models`, then `stt-cli verify-assets`.",
                    "Kiểm tra tài nguyên model: {} tài nguyên thiếu hoặc bị lỗi. Hãy chạy `stt-cli download-models`, sau đó `stt-cli verify-assets`.",
                    missing_or_corrupt
                );
                lp!(
                    "\n=== Setup Completed. Model download is a separate explicit step. ===",
                    "\n=== Thiết lập đã hoàn tất. Việc tải model là bước riêng cần chạy rõ ràng. ==="
                );
                return Ok(());
            }

            // 5. Run check-capu verification when assets are already available.
            lp!(
                "\n=== Running Verification checks ===",
                "\n=== Đang chạy các kiểm tra xác minh ==="
            );
            let capu_model = runtime.resolve_capu_model(None)?;
            let (_, capu) = build_capu_postprocessor(
                &runtime.workspace,
                &runtime.runtime_config.capu,
                capu_model,
            )?;
            let test_text = "rồi cũng hỗ trợ cho lâu lâu cũng cho gạo cho này kia";
            let output = capu.process_text(test_text)?;
            lp!("CAPU Output: '{}'", "Kết quả CAPU: '{}'", output);
            if output == "Rồi cũng hỗ trợ cho, lâu lâu cũng cho gạo cho này kia." {
                lp!(
                    "CAPU Check: SUCCESS (output matches expectation)",
                    "Kiểm tra CAPU: THÀNH CÔNG (kết quả khớp với mong đợi)"
                );
            } else {
                lp!(
                    "CAPU Check: WARNING (output mismatch)",
                    "Kiểm tra CAPU: CẢNH BÁO (kết quả không khớp mong đợi)"
                );
            }

            lp!(
                "\n=== Setup Completed Successfully! ===",
                "\n=== Thiết lập đã hoàn tất thành công! ==="
            );
        }
        Command::Service { action } => {
            handle_service(&runtime, action).await?;
        }
        Command::LlamaService { action } => {
            handle_llama_service(Some(runtime.workspace.root.as_path()), action).await?;
        }
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// Daemon Service Control Subcommands & Helpers
// -----------------------------------------------------------------------------

#[derive(Subcommand, Debug, Clone)]
enum ServiceAction {
    Config {
        #[arg(long)]
        bin_path: Option<PathBuf>,
        #[arg(long)]
        runtime_root: Option<PathBuf>,
        #[arg(long)]
        models_dir: Option<PathBuf>,
        #[arg(long)]
        user: Option<String>,
        #[arg(long)]
        port: Option<u16>,
        #[arg(long)]
        host: Option<String>,
        #[arg(long)]
        max_audio_seconds: Option<u64>,
    },
    Install,
    Uninstall,
    Start,
    Stop,
    Status,
    Health,
    #[command(about = "View recent stt-http service logs")]
    Logs {
        #[arg(long, default_value_t = 100)]
        lines: usize,
        #[arg(short, long)]
        follow: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ServiceConfig {
    bin_path: PathBuf,
    runtime_root: PathBuf,
    models_dir: PathBuf,
    user: String,
}

fn check_root_permission(path: &std::path::Path) -> bool {
    if let Some(parent) = path.parent() {
        let test_file = parent.join(".vit_stt_write_test");
        match fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&test_file)
        {
            Ok(_) => {
                let _ = fs::remove_file(test_file);
                true
            }
            Err(_) => false,
        }
    } else {
        false
    }
}

fn get_service_config(runtime: &SttRuntime, action_user: Option<String>) -> Result<ServiceConfig> {
    let service_config_path = runtime.workspace.root.join("config/service.json");
    if service_config_path.exists()
        && let Ok(text) = fs::read_to_string(&service_config_path)
        && let Ok(cfg) = serde_json::from_str::<ServiceConfig>(&text)
    {
        return Ok(cfg);
    }

    let workspace_root = runtime.workspace.root.clone();
    let models_dir = runtime.workspace.models_dir.clone();

    let mut bin_path = workspace_root.join("bin/stt-http");
    if !bin_path.exists()
        && let Ok(current_exe) = std::env::current_exe()
        && let Some(current_dir) = current_exe.parent()
    {
        let sibling_bin = current_dir.join("stt-http");
        if sibling_bin.exists() {
            bin_path = sibling_bin;
        } else {
            let target_release = workspace_root.join("target/release/stt-http");
            let target_debug = workspace_root.join("target/debug/stt-http");
            if target_release.exists() {
                bin_path = target_release;
            } else if target_debug.exists() {
                bin_path = target_debug;
            }
        }
    }

    let user = action_user
        .or_else(|| std::env::var("SUDO_USER").ok())
        .or_else(|| std::env::var("USER").ok())
        .unwrap_or_else(|| "root".to_string());

    let bin_path = bin_path.canonicalize().unwrap_or(bin_path);
    let runtime_root = workspace_root.canonicalize().unwrap_or(workspace_root);
    let models_dir = models_dir.canonicalize().unwrap_or(models_dir);

    Ok(ServiceConfig {
        bin_path,
        runtime_root,
        models_dir,
        user,
    })
}

fn save_service_config(workspace_root: &std::path::Path, cfg: &ServiceConfig) -> Result<()> {
    let service_config_path = workspace_root.join("config/service.json");
    if let Some(parent) = service_config_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json_str = serde_json::to_string_pretty(cfg)?;
    fs::write(&service_config_path, json_str)?;
    Ok(())
}

fn update_runtime_toml(
    workspace_root: &std::path::Path,
    host: &str,
    port: u16,
    max_audio_seconds: Option<u64>,
) -> Result<()> {
    let toml_path = workspace_root.join("config/runtime.toml");
    if !toml_path.exists() {
        return Ok(());
    }
    let toml_content = fs::read_to_string(&toml_path)?;
    let mut toml_val: toml::Value = toml::from_str(&toml_content)?;

    if let Some(server) = toml_val.get_mut("server").and_then(|s| s.as_table_mut()) {
        server.insert("host".to_string(), toml::Value::String(host.to_string()));
        server.insert("port".to_string(), toml::Value::Integer(port as i64));
        if let Some(secs) = max_audio_seconds {
            server.insert(
                "max_audio_seconds".to_string(),
                toml::Value::Integer(secs as i64),
            );
        }
    }

    let updated_content = toml::to_string(&toml_val)?;
    fs::write(&toml_path, updated_content)?;
    Ok(())
}

fn get_launchd_status() -> Result<Option<(Option<i32>, i32)>> {
    #[cfg(target_os = "macos")]
    {
        let output = StdCommand::new("launchctl")
            .args(["print", "system/com.vit-stt.http"])
            .output()?;
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut pid = None;
            let mut exit_code = 0;
            let mut has_service = false;
            for line in stdout.lines() {
                let line = line.trim();
                if line == "state = running" {
                    has_service = true;
                } else if let Some(value) = line.strip_prefix("pid = ") {
                    pid = value.parse::<i32>().ok();
                    has_service = true;
                } else if let Some(value) = line.strip_prefix("last exit code = ") {
                    exit_code = value.parse::<i32>().unwrap_or(0);
                    has_service = true;
                }
            }
            if has_service {
                return Ok(Some((pid, exit_code)));
            }
        }
    }

    let output = StdCommand::new("launchctl").arg("list").output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 && parts[2] == "com.vit-stt.http" {
            let pid = parts[0].parse::<i32>().ok();
            let exit_code = parts[1].parse::<i32>().unwrap_or(0);
            return Ok(Some((pid, exit_code)));
        }
    }
    Ok(None)
}

#[cfg(target_os = "macos")]
fn launchctl_bootout_system(label: &str) -> Result<std::process::ExitStatus> {
    Ok(StdCommand::new("launchctl")
        .args(["bootout", &format!("system/{label}")])
        .status()?)
}

#[cfg(target_os = "macos")]
fn launchctl_bootstrap_system(plist_path: &std::path::Path) -> Result<std::process::ExitStatus> {
    Ok(StdCommand::new("launchctl")
        .args(["bootstrap", "system", plist_path.to_string_lossy().as_ref()])
        .status()?)
}

#[cfg(target_os = "macos")]
fn launchctl_kickstart_system(label: &str) -> Result<std::process::ExitStatus> {
    Ok(StdCommand::new("launchctl")
        .args(["kickstart", "-k", &format!("system/{label}")])
        .status()?)
}

fn get_running_pids() -> Vec<i32> {
    let mut pids = Vec::new();
    if let Ok(output) = StdCommand::new("pgrep").args(["-x", "stt-http"]).output() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if let Ok(pid) = line.trim().parse::<i32>() {
                pids.push(pid);
            }
        }
    }
    pids
}

fn tail_log_files(paths: &[PathBuf], lines: usize, follow: bool) -> Result<()> {
    let existing_paths: Vec<&PathBuf> = paths.iter().filter(|path| path.exists()).collect();
    if existing_paths.is_empty() {
        println!("No log files found.");
        for path in paths {
            println!("Missing: {}", path.display());
        }
        return Ok(());
    }

    println!(
        "Viewing {} log file{}:",
        existing_paths.len(),
        if existing_paths.len() == 1 { "" } else { "s" }
    );
    for path in &existing_paths {
        println!("  {}", path.display());
    }

    let mut command = StdCommand::new("tail");
    command.arg("-n").arg(lines.to_string());
    if follow {
        command.arg("-f");
    }
    for path in existing_paths {
        command.arg(path);
    }
    let status = command.status()?;
    if !status.success() {
        return Err(anyhow::anyhow!("tail failed with status: {}", status));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn journalctl_logs(unit: &str, lines: usize, follow: bool) -> Result<()> {
    let mut command = StdCommand::new("journalctl");
    command.arg("-u").arg(unit).arg("-n").arg(lines.to_string());
    if follow {
        command.arg("-f");
    }
    let status = command.status()?;
    if !status.success() {
        return Err(anyhow::anyhow!(
            "journalctl failed for unit {} with status: {}",
            unit,
            status
        ));
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// llama-server LaunchDaemon Control
// -----------------------------------------------------------------------------

const LLAMA_SERVICE_LABEL: &str = "com.vit-stt.llama";
const LLAMA_SERVICE_PLIST: &str = "/Library/LaunchDaemons/com.vit-stt.llama.plist";
const LLAMA_DEFAULT_HF_REPO: &str = "unsloth/gemma-4-E4B-it-GGUF";
const LLAMA_DEFAULT_ALIAS: &str = "vit_small_4b";

#[derive(Subcommand, Debug, Clone)]
enum LlamaServiceAction {
    #[command(about = "Write llama-server LaunchDaemon configuration")]
    Config {
        #[arg(long)]
        bin_path: Option<PathBuf>,
        #[arg(long)]
        working_dir: Option<PathBuf>,
        #[arg(long)]
        user: Option<String>,
        #[arg(long, default_value = LLAMA_DEFAULT_HF_REPO)]
        hf: String,
        #[arg(long, default_value_t = 0.3)]
        temp: f32,
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,
        #[arg(long, default_value_t = 64)]
        top_k: i32,
        #[arg(long, default_value = "off")]
        reasoning: String,
        #[arg(short = 'c', long = "context", default_value_t = 32768)]
        context: u32,
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        #[arg(long, default_value_t = 8001)]
        port: u16,
        #[arg(long, default_value = LLAMA_DEFAULT_ALIAS)]
        alias: String,
    },
    #[command(about = "Install and start the llama-server LaunchDaemon")]
    Install {
        #[arg(long)]
        bin_path: Option<PathBuf>,
    },
    Uninstall,
    Start,
    Stop,
    Status,
    Health,
    #[command(about = "View recent llama-server LaunchDaemon logs")]
    Logs {
        #[arg(long, default_value_t = 100)]
        lines: usize,
        #[arg(short, long)]
        follow: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LlamaServiceConfig {
    bin_path: PathBuf,
    working_dir: PathBuf,
    user: String,
    hf: String,
    temp: f32,
    top_p: f32,
    top_k: i32,
    reasoning: String,
    context: u32,
    host: String,
    port: u16,
    alias: String,
}

fn workspace_root_or_cwd(workspace_root: Option<&std::path::Path>) -> Result<PathBuf> {
    Ok(workspace_root
        .map(|p| p.to_path_buf())
        .unwrap_or(std::env::current_dir()?))
}

fn llama_config_path(workspace_root: &std::path::Path) -> PathBuf {
    workspace_root.join("config/llama-service.json")
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn resolve_executable(name: &str) -> Option<PathBuf> {
    let output = StdCommand::new("which").arg(name).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if path.is_empty() {
        None
    } else {
        Some(PathBuf::from(path))
    }
}

fn get_llama_service_config(
    workspace_root: &std::path::Path,
    bin_path_override: Option<PathBuf>,
) -> Result<LlamaServiceConfig> {
    let config_path = llama_config_path(workspace_root);
    if config_path.exists() {
        let text = fs::read_to_string(&config_path)?;
        let mut cfg = serde_json::from_str::<LlamaServiceConfig>(&text)?;
        if let Some(path) = bin_path_override {
            cfg.bin_path = path.canonicalize().unwrap_or(path);
        }
        return Ok(cfg);
    }

    let bin_path = bin_path_override
        .or_else(|| resolve_executable("llama-server"))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Could not find llama-server. Install llama.cpp or run `stt-cli llama-service config --bin-path <path>`."
            )
        })?;
    let working_dir = workspace_root
        .canonicalize()
        .unwrap_or_else(|_| workspace_root.to_path_buf());
    let user = std::env::var("SUDO_USER")
        .ok()
        .or_else(|| std::env::var("USER").ok())
        .unwrap_or_else(|| "root".to_string());

    Ok(LlamaServiceConfig {
        bin_path: bin_path.canonicalize().unwrap_or(bin_path),
        working_dir,
        user,
        hf: LLAMA_DEFAULT_HF_REPO.to_string(),
        temp: 0.3,
        top_p: 0.95,
        top_k: 64,
        reasoning: "off".to_string(),
        context: 32768,
        host: "0.0.0.0".to_string(),
        port: 8001,
        alias: LLAMA_DEFAULT_ALIAS.to_string(),
    })
}

fn save_llama_service_config(
    workspace_root: &std::path::Path,
    cfg: &LlamaServiceConfig,
) -> Result<()> {
    let config_path = llama_config_path(workspace_root);
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(config_path, serde_json::to_string_pretty(cfg)?)?;
    Ok(())
}

fn llama_logs_dir(workspace_root: &std::path::Path) -> PathBuf {
    let config_path = llama_config_path(workspace_root);
    if config_path.exists()
        && let Ok(text) = fs::read_to_string(&config_path)
        && let Ok(cfg) = serde_json::from_str::<LlamaServiceConfig>(&text)
    {
        return cfg.working_dir.join("logs");
    }
    workspace_root.join("logs")
}

fn get_launchd_status_for(label: &str) -> Result<Option<(Option<i32>, i32)>> {
    #[cfg(target_os = "macos")]
    {
        let output = StdCommand::new("launchctl")
            .args(["print", &format!("system/{label}")])
            .output()?;
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut pid = None;
            let mut exit_code = 0;
            let mut has_service = false;
            for line in stdout.lines() {
                let line = line.trim();
                if line == "state = running" {
                    has_service = true;
                } else if let Some(value) = line.strip_prefix("pid = ") {
                    pid = value.parse::<i32>().ok();
                    has_service = true;
                } else if let Some(value) = line.strip_prefix("last exit code = ") {
                    exit_code = value.parse::<i32>().unwrap_or(0);
                    has_service = true;
                }
            }
            if has_service {
                return Ok(Some((pid, exit_code)));
            }
        }
    }

    let output = StdCommand::new("launchctl").arg("list").output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 && parts[2] == label {
            let pid = parts[0].parse::<i32>().ok();
            let exit_code = parts[1].parse::<i32>().unwrap_or(0);
            return Ok(Some((pid, exit_code)));
        }
    }
    Ok(None)
}

fn get_llama_running_pids() -> Vec<i32> {
    let mut pids = Vec::new();
    if let Ok(output) = StdCommand::new("pgrep")
        .args(["-f", "llama-server.*vit_small_4b"])
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if let Ok(pid) = line.trim().parse::<i32>() {
                pids.push(pid);
            }
        }
    }
    pids
}

fn build_llama_plist(cfg: &LlamaServiceConfig, logs_dir: &std::path::Path) -> String {
    let args = [
        cfg.bin_path.display().to_string(),
        "-hf".to_string(),
        cfg.hf.clone(),
        "--temp".to_string(),
        cfg.temp.to_string(),
        "--top-p".to_string(),
        cfg.top_p.to_string(),
        "--top-k".to_string(),
        cfg.top_k.to_string(),
        "--reasoning".to_string(),
        cfg.reasoning.clone(),
        "-c".to_string(),
        cfg.context.to_string(),
        "--host".to_string(),
        cfg.host.clone(),
        "--port".to_string(),
        cfg.port.to_string(),
        "--alias".to_string(),
        cfg.alias.clone(),
    ];

    let mut plist = String::from(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
"#,
    );
    plist.push_str(&format!(
        "    <string>{}</string>\n    <key>ProgramArguments</key>\n    <array>\n",
        xml_escape(LLAMA_SERVICE_LABEL)
    ));
    for arg in args {
        plist.push_str(&format!("        <string>{}</string>\n", xml_escape(&arg)));
    }
    plist.push_str(&format!(
        r#"    </array>
    <key>WorkingDirectory</key>
    <string>{}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>SessionCreate</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{}</string>
    <key>StandardErrorPath</key>
    <string>{}</string>
"#,
        xml_escape(&cfg.working_dir.display().to_string()),
        xml_escape(&logs_dir.join("llama-server.log").display().to_string()),
        xml_escape(&logs_dir.join("llama-server.err.log").display().to_string())
    ));
    if cfg.user != "root" {
        plist.push_str(&format!(
            "    <key>UserName</key>\n    <string>{}</string>\n",
            xml_escape(&cfg.user)
        ));
    }
    plist.push_str("</dict>\n</plist>\n");
    plist
}

async fn handle_llama_service(
    workspace_root: Option<&std::path::Path>,
    action: LlamaServiceAction,
) -> Result<()> {
    let workspace_root = workspace_root_or_cwd(workspace_root)?;
    match action {
        LlamaServiceAction::Config {
            bin_path,
            working_dir,
            user,
            hf,
            temp,
            top_p,
            top_k,
            reasoning,
            context,
            host,
            port,
            alias,
        } => {
            let mut cfg = get_llama_service_config(&workspace_root, bin_path)?;
            if let Some(dir) = working_dir {
                cfg.working_dir = dir.canonicalize().unwrap_or(dir);
            }
            if let Some(user) = user {
                cfg.user = user;
            }
            cfg.hf = hf;
            cfg.temp = temp;
            cfg.top_p = top_p;
            cfg.top_k = top_k;
            cfg.reasoning = reasoning;
            cfg.context = context;
            cfg.host = host;
            cfg.port = port;
            cfg.alias = alias;
            save_llama_service_config(&workspace_root, &cfg)?;
            println!("--- llama-server LaunchDaemon Configuration ---");
            println!("Binary Path:       {}", cfg.bin_path.display());
            println!("Working Directory: {}", cfg.working_dir.display());
            println!("User:              {}", cfg.user);
            println!("HF Repo:           {}", cfg.hf);
            println!("Temp:              {}", cfg.temp);
            println!("Top P:             {}", cfg.top_p);
            println!("Top K:             {}", cfg.top_k);
            println!("Reasoning:         {}", cfg.reasoning);
            println!("Context:           {}", cfg.context);
            println!("Host:              {}", cfg.host);
            println!("Port:              {}", cfg.port);
            println!("Alias:             {}", cfg.alias);
            println!(
                "Saved to:          {}",
                llama_config_path(&workspace_root).display()
            );
        }
        LlamaServiceAction::Install { bin_path } => {
            #[cfg(not(target_os = "macos"))]
            {
                let _ = bin_path;
                return Err(anyhow::anyhow!(
                    "llama-service install currently supports macOS LaunchDaemons only."
                ));
            }
            #[cfg(target_os = "macos")]
            {
                let cfg = get_llama_service_config(&workspace_root, bin_path)?;
                if !cfg.bin_path.exists() {
                    return Err(anyhow::anyhow!(
                        "llama-server binary does not exist: {}",
                        cfg.bin_path.display()
                    ));
                }

                let plist_path = std::path::Path::new(LLAMA_SERVICE_PLIST);
                if !check_root_permission(plist_path) {
                    return Err(anyhow::anyhow!(
                        "Permission denied. Please run this command with sudo: 'sudo stt-cli llama-service install'"
                    ));
                }

                let logs_dir = cfg.working_dir.join("logs");
                fs::create_dir_all(&logs_dir)?;
                let plist = build_llama_plist(&cfg, &logs_dir);
                fs::write(plist_path, plist)?;
                save_llama_service_config(&workspace_root, &cfg)?;
                println!(
                    "Written macOS LaunchDaemon plist to {}",
                    plist_path.display()
                );

                let _ = launchctl_bootout_system(LLAMA_SERVICE_LABEL);
                let status = launchctl_bootstrap_system(plist_path)?;
                if !status.success() {
                    return Err(anyhow::anyhow!(
                        "Failed to bootstrap LaunchDaemon. Try: sudo launchctl print system/{}",
                        LLAMA_SERVICE_LABEL
                    ));
                }
                let status = launchctl_kickstart_system(LLAMA_SERVICE_LABEL)?;
                if !status.success() {
                    return Err(anyhow::anyhow!(
                        "Failed to kickstart LaunchDaemon after bootstrap."
                    ));
                }
                println!("llama-server LaunchDaemon bootstrapped and started successfully.");
            }
        }
        LlamaServiceAction::Uninstall => {
            #[cfg(not(target_os = "macos"))]
            {
                return Err(anyhow::anyhow!(
                    "llama-service uninstall currently supports macOS LaunchDaemons only."
                ));
            }
            #[cfg(target_os = "macos")]
            {
                let plist_path = std::path::Path::new(LLAMA_SERVICE_PLIST);
                if !check_root_permission(plist_path) {
                    return Err(anyhow::anyhow!(
                        "Permission denied. Please run this command with sudo: 'sudo stt-cli llama-service uninstall'"
                    ));
                }
                if plist_path.exists() {
                    let _ = launchctl_bootout_system(LLAMA_SERVICE_LABEL);
                    fs::remove_file(plist_path)?;
                    println!("Removed llama-server LaunchDaemon plist and booted out service.");
                } else {
                    println!("llama-server LaunchDaemon plist not found, nothing to do.");
                }
            }
        }
        LlamaServiceAction::Start => {
            #[cfg(not(target_os = "macos"))]
            {
                return Err(anyhow::anyhow!(
                    "llama-service start currently supports macOS LaunchDaemons only."
                ));
            }
            #[cfg(target_os = "macos")]
            {
                let plist_path = std::path::Path::new(LLAMA_SERVICE_PLIST);
                if !check_root_permission(plist_path) {
                    return Err(anyhow::anyhow!(
                        "Permission denied. Please run this command with sudo: 'sudo stt-cli llama-service start'"
                    ));
                }
                if !plist_path.exists() {
                    return Err(anyhow::anyhow!(
                        "Service is not installed. Please run 'sudo stt-cli llama-service install' first."
                    ));
                }
                let status = if get_launchd_status_for(LLAMA_SERVICE_LABEL)?.is_some() {
                    launchctl_kickstart_system(LLAMA_SERVICE_LABEL)?
                } else {
                    let bootstrap_status = launchctl_bootstrap_system(plist_path)?;
                    if !bootstrap_status.success() {
                        return Err(anyhow::anyhow!("Failed to bootstrap llama-server service."));
                    }
                    launchctl_kickstart_system(LLAMA_SERVICE_LABEL)?
                };
                if !status.success() {
                    return Err(anyhow::anyhow!("Failed to start llama-server service."));
                }
                println!("llama-server service started successfully.");
            }
        }
        LlamaServiceAction::Stop => {
            #[cfg(not(target_os = "macos"))]
            {
                return Err(anyhow::anyhow!(
                    "llama-service stop currently supports macOS LaunchDaemons only."
                ));
            }
            #[cfg(target_os = "macos")]
            {
                let plist_path = std::path::Path::new(LLAMA_SERVICE_PLIST);
                if !check_root_permission(plist_path) {
                    return Err(anyhow::anyhow!(
                        "Permission denied. Please run this command with sudo: 'sudo stt-cli llama-service stop'"
                    ));
                }
                if !plist_path.exists() {
                    return Err(anyhow::anyhow!("Service is not installed."));
                }
                let status = launchctl_bootout_system(LLAMA_SERVICE_LABEL)?;
                if !status.success() {
                    return Err(anyhow::anyhow!("Failed to stop llama-server service."));
                }
                println!("llama-server service stopped successfully.");
            }
        }
        LlamaServiceAction::Status => {
            #[cfg(not(target_os = "macos"))]
            {
                return Err(anyhow::anyhow!(
                    "llama-service status currently supports macOS LaunchDaemons only."
                ));
            }
            #[cfg(target_os = "macos")]
            {
                let plist_path = std::path::Path::new(LLAMA_SERVICE_PLIST);
                println!("Service Name:      {}", LLAMA_SERVICE_LABEL);
                println!(
                    "Installed:         {}",
                    if plist_path.exists() {
                        format!("Yes ({})", plist_path.display())
                    } else {
                        "No".to_string()
                    }
                );
                match get_launchd_status_for(LLAMA_SERVICE_LABEL)? {
                    Some((pid, exit_code)) => {
                        println!("Loaded in launchd: Yes");
                        if let Some(pid) = pid {
                            println!("Status:            Running (PID: {})", pid);
                        } else {
                            println!(
                                "Status:            Inactive (Last Exit Code: {})",
                                exit_code
                            );
                        }
                    }
                    None => {
                        println!("Loaded in launchd: No");
                        println!("Status:            Inactive");
                    }
                }
                let pids = get_llama_running_pids();
                if pids.is_empty() {
                    println!("Active Processes:  None");
                } else {
                    println!("Active Processes:  Running (PIDs: {:?})", pids);
                }
                if let Ok(cfg) = get_llama_service_config(&workspace_root, None) {
                    println!("Health URL:        http://{}:{}/health", cfg.host, cfg.port);
                    println!("Model Alias:       {}", cfg.alias);
                }
                if !plist_path.exists() {
                    println!(
                        "\nTo install the service, run:\n  sudo stt-cli llama-service install"
                    );
                }
            }
        }
        LlamaServiceAction::Health => {
            let cfg = get_llama_service_config(&workspace_root, None)?;
            let base_url = format!("http://{}:{}", cfg.host, cfg.port);
            println!("Testing llama-server health at: {}/health", base_url);
            let client = reqwest::Client::new();
            match client.get(format!("{}/health", base_url)).send().await {
                Ok(response) => {
                    let status = response.status();
                    let text = response.text().await.unwrap_or_default();
                    println!("Health Status:     {}", status);
                    println!("Health Body:       {}", text);
                    if !status.is_success() {
                        println!("Service Health:    UNHEALTHY");
                        return Ok(());
                    }
                }
                Err(err) => {
                    println!("Error connecting to llama-server: {}", err);
                    println!("Service Health:    UNREACHABLE");
                    return Ok(());
                }
            }

            println!(
                "Testing OpenAI-compatible models at: {}/v1/models",
                base_url
            );
            match client.get(format!("{}/v1/models", base_url)).send().await {
                Ok(response) => {
                    let status = response.status();
                    let text = response.text().await.unwrap_or_default();
                    println!("Models Status:     {}", status);
                    println!("Models Body:       {}", text);
                    if status.is_success() && text.contains(&cfg.alias) {
                        println!("Service Health:    OK");
                    } else if status.is_success() {
                        println!("Service Health:    WARNING (alias not found in /v1/models)");
                    } else {
                        println!("Service Health:    WARNING (/health OK, /v1/models failed)");
                    }
                }
                Err(err) => {
                    println!("Models Check:      FAILED ({})", err);
                    println!("Service Health:    WARNING (/health OK, /v1/models unreachable)");
                }
            }
        }
        LlamaServiceAction::Logs { lines, follow } => {
            let logs_dir = llama_logs_dir(&workspace_root);
            tail_log_files(
                &[
                    logs_dir.join("llama-server.log"),
                    logs_dir.join("llama-server.err.log"),
                ],
                lines,
                follow,
            )?;
        }
    }
    Ok(())
}

async fn handle_service_uninstall() -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let plist_path = std::path::Path::new("/Library/LaunchDaemons/com.vit-stt.http.plist");
        if !check_root_permission(plist_path) {
            return Err(anyhow::anyhow!(
                "Permission denied. Please run this command with sudo: 'sudo stt-cli service uninstall'"
            ));
        }

        if plist_path.exists() {
            let _ = launchctl_bootout_system("com.vit-stt.http");
            fs::remove_file(plist_path)?;
            println!("Removed LaunchDaemon plist and booted out service.");
        } else {
            println!("LaunchDaemon plist not found, nothing to do.");
        }
    }

    #[cfg(target_os = "linux")]
    {
        let service_path = std::path::Path::new("/etc/systemd/system/stt-http.service");
        if !check_root_permission(service_path) {
            return Err(anyhow::anyhow!(
                "Permission denied. Please run this command with sudo: 'sudo stt-cli service uninstall'"
            ));
        }

        if service_path.exists() {
            let _ = StdCommand::new("systemctl")
                .args(["stop", "stt-http"])
                .status();
            let _ = StdCommand::new("systemctl")
                .args(["disable", "stt-http"])
                .status();
            fs::remove_file(service_path)?;
            let _ = StdCommand::new("systemctl").arg("daemon-reload").status()?;
            println!("Removed systemd service file and stopped service.");
        } else {
            println!("systemd service file not found, nothing to do.");
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        return Err(anyhow::anyhow!(
            "Unsupported operating system for service uninstallation."
        ));
    }
    Ok(())
}

async fn handle_service_start() -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let plist_path = std::path::Path::new("/Library/LaunchDaemons/com.vit-stt.http.plist");
        if !check_root_permission(plist_path) {
            return Err(anyhow::anyhow!(
                "Permission denied. Please run this command with sudo: 'sudo stt-cli service start'"
            ));
        }
        if !plist_path.exists() {
            return Err(anyhow::anyhow!(
                "Service is not installed. Please run 'sudo stt-cli service install' first."
            ));
        }

        let status = if get_launchd_status()?.is_some() {
            launchctl_kickstart_system("com.vit-stt.http")?
        } else {
            let bootstrap_status = launchctl_bootstrap_system(plist_path)?;
            if !bootstrap_status.success() {
                return Err(anyhow::anyhow!("Failed to bootstrap service."));
            }
            launchctl_kickstart_system("com.vit-stt.http")?
        };
        if !status.success() {
            return Err(anyhow::anyhow!("Failed to start service."));
        }
        println!("Service started successfully.");
    }

    #[cfg(target_os = "linux")]
    {
        let service_path = std::path::Path::new("/etc/systemd/system/stt-http.service");
        if !check_root_permission(service_path) {
            return Err(anyhow::anyhow!(
                "Permission denied. Please run this command with sudo: 'sudo stt-cli service start'"
            ));
        }
        if !service_path.exists() {
            return Err(anyhow::anyhow!(
                "Service is not installed. Please run 'sudo stt-cli service install' first."
            ));
        }

        let status = StdCommand::new("systemctl")
            .args(["start", "stt-http"])
            .status()?;
        if status.success() {
            println!("Service started successfully.");
        } else {
            return Err(anyhow::anyhow!("Failed to start service."));
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        return Err(anyhow::anyhow!("Unsupported operating system."));
    }
    Ok(())
}

async fn handle_service_stop() -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let plist_path = std::path::Path::new("/Library/LaunchDaemons/com.vit-stt.http.plist");
        if !check_root_permission(plist_path) {
            return Err(anyhow::anyhow!(
                "Permission denied. Please run this command with sudo: 'sudo stt-cli service stop'"
            ));
        }
        if !plist_path.exists() {
            return Err(anyhow::anyhow!("Service is not installed."));
        }

        let status = launchctl_bootout_system("com.vit-stt.http")?;
        if status.success() {
            println!("Service stopped successfully.");
        } else {
            return Err(anyhow::anyhow!("Failed to stop service."));
        }
    }

    #[cfg(target_os = "linux")]
    {
        let service_path = std::path::Path::new("/etc/systemd/system/stt-http.service");
        if !check_root_permission(service_path) {
            return Err(anyhow::anyhow!(
                "Permission denied. Please run this command with sudo: 'sudo stt-cli service stop'"
            ));
        }
        if !service_path.exists() {
            return Err(anyhow::anyhow!("Service is not installed."));
        }

        let status = StdCommand::new("systemctl")
            .args(["stop", "stt-http"])
            .status()?;
        if status.success() {
            println!("Service stopped successfully.");
        } else {
            return Err(anyhow::anyhow!("Failed to stop service."));
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        return Err(anyhow::anyhow!("Unsupported operating system."));
    }
    Ok(())
}

async fn handle_service_status() -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let plist_path = std::path::Path::new("/Library/LaunchDaemons/com.vit-stt.http.plist");
        println!("Service Name:     com.vit-stt.http");
        println!(
            "Installed:        {}",
            if plist_path.exists() {
                format!("Yes ({})", plist_path.display())
            } else {
                "No".to_string()
            }
        );

        let launchd_status = get_launchd_status()?;
        match launchd_status {
            Some((pid, exit_code)) => {
                println!("Loaded in launchd: Yes");
                if let Some(p) = pid {
                    println!("Status:           Running (PID: {})", p);
                } else {
                    println!("Status:           Inactive (Last Exit Code: {})", exit_code);
                }
            }
            None => {
                println!("Loaded in launchd: No");
                println!("Status:           Inactive");
            }
        }

        let pids = get_running_pids();
        if !pids.is_empty() {
            println!("Active Processes: Running (PIDs: {:?})", pids);
        } else {
            println!("Active Processes: None");
        }

        if !plist_path.exists() {
            println!("\nTo install the service, run:\n  sudo stt-cli service install");
        }
    }

    #[cfg(target_os = "linux")]
    {
        let service_path = std::path::Path::new("/etc/systemd/system/stt-http.service");
        println!("Service Name:     stt-http");
        println!(
            "Installed:        {}",
            if service_path.exists() {
                format!("Yes ({})", service_path.display())
            } else {
                "No".to_string()
            }
        );

        if service_path.exists() {
            println!("\n--- systemctl status stt-http ---");
            let _ = StdCommand::new("systemctl")
                .args(["status", "stt-http"])
                .status();
        } else {
            println!("\nTo install the service, run:\n  sudo stt-cli service install");
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        return Err(anyhow::anyhow!("Unsupported operating system."));
    }
    Ok(())
}

async fn handle_service_logs(
    workspace_root: Option<&std::path::Path>,
    lines: usize,
    follow: bool,
) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        let _ = workspace_root;
        journalctl_logs("stt-http", lines, follow)?;
    }
    #[cfg(not(target_os = "linux"))]
    {
        let workspace_root = workspace_root_or_cwd(workspace_root)?;
        let service_config_path = workspace_root.join("config/service.json");
        let runtime_root = if service_config_path.exists() {
            fs::read_to_string(&service_config_path)
                .ok()
                .and_then(|text| serde_json::from_str::<ServiceConfig>(&text).ok())
                .map(|cfg| cfg.runtime_root)
                .unwrap_or_else(|| workspace_root.clone())
        } else {
            workspace_root
        };
        let logs_dir = runtime_root.join("logs");
        tail_log_files(
            &[
                logs_dir.join("stt-http.log"),
                logs_dir.join("stt-http.err.log"),
            ],
            lines,
            follow,
        )?;
    }
    Ok(())
}

async fn handle_service(runtime: &SttRuntime, action: ServiceAction) -> Result<()> {
    match action {
        ServiceAction::Config {
            bin_path,
            runtime_root,
            models_dir,
            user,
            port,
            host,
            max_audio_seconds,
        } => {
            let mut cfg = match get_service_config(runtime, user.clone()) {
                Ok(c) => c,
                Err(_) => {
                    let current_exe = std::env::current_exe()?;
                    let current_dir = current_exe.parent().unwrap().to_path_buf();
                    ServiceConfig {
                        bin_path: current_dir.join("stt-http"),
                        runtime_root: runtime.workspace.root.clone(),
                        models_dir: runtime.workspace.models_dir.clone(),
                        user: std::env::var("USER").unwrap_or_else(|_| "root".to_string()),
                    }
                }
            };

            if let Some(b) = bin_path {
                cfg.bin_path = b.canonicalize().unwrap_or(b);
            }
            if let Some(r) = runtime_root {
                cfg.runtime_root = r.canonicalize().unwrap_or(r);
            }
            if let Some(m) = models_dir {
                cfg.models_dir = m.canonicalize().unwrap_or(m);
            }
            if let Some(u) = user {
                cfg.user = u;
            }

            save_service_config(&runtime.workspace.root, &cfg)?;

            let current_host = host
                .clone()
                .unwrap_or_else(|| runtime.runtime_config.server.host.clone());
            let current_port = port.unwrap_or(runtime.runtime_config.server.port);

            if port.is_some() || host.is_some() {
                update_runtime_toml(&runtime.workspace.root, &current_host, current_port, None)?;
                println!("Updated port/host in runtime.toml.");
            }

            if let Some(secs) = max_audio_seconds {
                update_runtime_toml(
                    &runtime.workspace.root,
                    &current_host,
                    current_port,
                    Some(secs),
                )?;
                println!("Updated max_audio_seconds in runtime.toml.");
            }

            let srv = &runtime.runtime_config.server;
            println!("--- Service Configuration ---");
            println!("Binary Path:      {}", cfg.bin_path.display());
            println!("Runtime Root:     {}", cfg.runtime_root.display());
            println!("Models Directory: {}", cfg.models_dir.display());
            println!("User:             {}", cfg.user);
            println!("Port:             {}", srv.port);
            println!("Host:             {}", srv.host);
            println!(
                "Saved to:         {}",
                runtime.workspace.root.join("config/service.json").display()
            );
        }
        ServiceAction::Install => {
            let cfg = get_service_config(runtime, None)?;

            if !cfg.bin_path.exists() {
                return Err(anyhow::anyhow!(
                    "Executable binary does not exist: {}. Please compile stt-http first or configure the correct path using 'stt-cli service config --bin-path <path>'.",
                    cfg.bin_path.display()
                ));
            }

            let logs_dir = cfg.runtime_root.join("logs");
            if !logs_dir.exists() {
                fs::create_dir_all(&logs_dir)?;
            }

            #[cfg(unix)]
            {
                if let Ok(uid_str) = std::process::Command::new("id")
                    .args(["-u", &cfg.user])
                    .output()
                    && uid_str.status.success()
                {
                    let uid_str = String::from_utf8_lossy(&uid_str.stdout).trim().to_string();
                    if let Ok(gid_str) = std::process::Command::new("id")
                        .args(["-g", &cfg.user])
                        .output()
                    {
                        let gid_str = String::from_utf8_lossy(&gid_str.stdout).trim().to_string();
                        let _ = std::process::Command::new("chown")
                            .args([
                                format!("{}:{}", uid_str, gid_str),
                                logs_dir.to_string_lossy().into_owned(),
                            ])
                            .status();
                    }
                }
            }

            #[cfg(target_os = "macos")]
            {
                let plist_path =
                    std::path::Path::new("/Library/LaunchDaemons/com.vit-stt.http.plist");
                if !check_root_permission(plist_path) {
                    return Err(anyhow::anyhow!(
                        "Permission denied. Please run this command with sudo: 'sudo stt-cli service install'"
                    ));
                }

                let mut plist = format!(
                    r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.vit-stt.http</string>
    <key>ProgramArguments</key>
    <array>
        <string>{}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>VIT_STT_RUNTIME_ROOT</key>
        <string>{}</string>
        <key>VIT_STT_MODELS_DIR</key>
        <string>{}</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>SessionCreate</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{}</string>
    <key>StandardErrorPath</key>
    <string>{}</string>
"#,
                    cfg.bin_path.display(),
                    cfg.runtime_root.display(),
                    cfg.runtime_root.display(),
                    cfg.models_dir.display(),
                    logs_dir.join("stt-http.log").display(),
                    logs_dir.join("stt-http.err.log").display()
                );

                if cfg.user != "root" {
                    plist.push_str(&format!(
                        "    <key>UserName</key>\n    <string>{}</string>\n",
                        cfg.user
                    ));
                }
                plist.push_str("</dict>\n</plist>\n");

                fs::write(plist_path, plist)?;
                println!("Written macOS launchctl plist to {}", plist_path.display());

                let _ = launchctl_bootout_system("com.vit-stt.http");
                let status = launchctl_bootstrap_system(plist_path)?;
                if !status.success() {
                    return Err(anyhow::anyhow!(
                        "Failed to bootstrap macOS LaunchDaemon. Try: sudo launchctl print system/com.vit-stt.http"
                    ));
                }
                let status = launchctl_kickstart_system("com.vit-stt.http")?;
                if !status.success() {
                    return Err(anyhow::anyhow!(
                        "Failed to kickstart macOS LaunchDaemon after bootstrap."
                    ));
                }
                println!("macOS LaunchDaemon bootstrapped and started successfully.");
            }

            #[cfg(target_os = "linux")]
            {
                let service_path = std::path::Path::new("/etc/systemd/system/stt-http.service");
                if !check_root_permission(service_path) {
                    return Err(anyhow::anyhow!(
                        "Permission denied. Please run this command with sudo: 'sudo stt-cli service install'"
                    ));
                }

                let service_content = format!(
                    r#"[Unit]
Description=vit-stt HTTP Speech-to-Text Service
After=network.target

[Service]
Type=simple
User={}
WorkingDirectory={}
Environment="VIT_STT_RUNTIME_ROOT={}"
Environment="VIT_STT_MODELS_DIR={}"
ExecStart={}
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"#,
                    cfg.user,
                    cfg.runtime_root.display(),
                    cfg.runtime_root.display(),
                    cfg.models_dir.display(),
                    cfg.bin_path.display()
                );

                fs::write(service_path, service_content)?;
                println!("Written systemd service file to {}", service_path.display());

                let _ = StdCommand::new("systemctl").arg("daemon-reload").status()?;
                let _ = StdCommand::new("systemctl")
                    .args(["enable", "stt-http"])
                    .status()?;
                let status = StdCommand::new("systemctl")
                    .args(["start", "stt-http"])
                    .status()?;
                if status.success() {
                    println!("systemd service enabled and started successfully.");
                } else {
                    return Err(anyhow::anyhow!("Failed to start stt-http service."));
                }
            }

            #[cfg(not(any(target_os = "macos", target_os = "linux")))]
            {
                return Err(anyhow::anyhow!(
                    "Unsupported operating system for service installation."
                ));
            }
        }
        // These four subcommands delegate to standalone functions so they
        // don't require loading the runtime (which takes ~40s validating assets).
        ServiceAction::Uninstall => handle_service_uninstall().await?,
        ServiceAction::Start => handle_service_start().await?,
        ServiceAction::Stop => handle_service_stop().await?,
        ServiceAction::Status => handle_service_status().await?,
        ServiceAction::Health => {
            let srv = &runtime.runtime_config.server;
            let url = format!("http://{}:{}/health", srv.host, srv.port);
            println!("Testing health at: {}", url);
            let client = reqwest::Client::new();
            let res = client.get(&url).send().await;
            match res {
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() {
                        let text = response.text().await.unwrap_or_default();
                        println!("Response Status:  {}", status);
                        println!("Response Body:    {}", text);
                        if text.contains("\"status\":\"ok\"") || text.contains("ok") {
                            println!("Service Health:   OK");
                        } else {
                            println!("Service Health:   WARNING (Unexpected response body)");
                        }
                    } else {
                        println!("Response Status:  {}", status);
                        println!("Service Health:   UNHEALTHY");
                    }
                }
                Err(err) => {
                    println!("Error connecting to service: {}", err);
                    println!("Service Health:   UNREACHABLE (Is the server running?)");
                }
            }
        }
        ServiceAction::Logs { lines, follow } => {
            handle_service_logs(Some(runtime.workspace.root.as_path()), lines, follow).await?;
        }
    }
    Ok(())
}
