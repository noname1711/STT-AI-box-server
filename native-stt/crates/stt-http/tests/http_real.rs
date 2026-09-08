use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};

fn real_workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn free_port() -> Result<u16> {
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

/// Create a temporary workspace directory with config files and symlinks to the
/// real baselines/ and models/ directories. Returns (temp_dir, workspace_root)
/// where workspace_root is the canonicalized temp dir path.
fn setup_test_workspace() -> Result<(tempfile::TempDir, PathBuf)> {
    let real_root = real_workspace_root();
    let temp_dir = tempfile::tempdir()?;
    let root = temp_dir.path().canonicalize()?;

    // Symlink baselines/ and models/ so relative paths in config resolve
    // through to the real asset files.
    std::os::unix::fs::symlink(real_root.join("baselines"), root.join("baselines"))?;
    std::os::unix::fs::symlink(real_root.join("models"), root.join("models"))?;

    // Create config/runtime.toml
    std::fs::create_dir_all(root.join("config"))?;
    std::fs::write(
        root.join("config/runtime.toml"),
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
warm_start_models = ["vit_stt_vi_v2"]

[capu]
engine = "auto"
python_bin = "capu-worker/.venv/bin/python"
worker_dir = "capu-worker"
device = "cpu"
request_timeout_seconds = 120
"#,
    )?;

    // Create config/models.local.json — use clean_lower to avoid CAPU
    // Python worker dependency in CI.
    std::fs::write(
        root.join("config/models.local.json"),
        r#"{
  "models": [
    {
      "id": "vit_stt_vi_v2",
      "language": "vi",
      "model_dir": "models/stt/gipformer-65M-rnnt",
      "postprocess_mode": "clean_lower",
      "vad_min_silence": 0.5,
      "vad_min_speech": 0.05,
      "vad_max_speech": 14.0,
      "startup_probe_wav_path": "baselines/phase0/vi_probe.wav",
      "startup_probe_expected_text": "ĐỊNH NGHĨA THẾ NÀO LÀ ĂN MẶC ĐẸP"
    }
  ]
}"#,
    )?;

    Ok((temp_dir, root))
}

fn wait_for_server(base_url: &str) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()?;
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        match client.get(format!("{base_url}/health")).send() {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            _ => thread::sleep(Duration::from_millis(250)),
        }
    }
    Err(anyhow!("stt-http failed to start within timeout"))
}

fn spawn_server(port: u16, workspace_root: &Path) -> Result<Child> {
    let child = Command::new(env!("CARGO_BIN_EXE_stt-http"))
        .current_dir(workspace_root)
        .args([
            "--workspace-root",
            &workspace_root.to_string_lossy(),
            "--host",
            "127.0.0.1",
            "-p",
            &port.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to spawn stt-http")?;
    wait_for_server(&format!("http://127.0.0.1:{port}"))?;
    Ok(child)
}

fn stop_server(mut child: Child) {
    let _ = child.kill();
    let _ = child.wait();
}

const EXPECTED_VI: &str = "định nghĩa thế nào là ăn mặc đẹp";

#[test]
fn health_and_models_and_transcribe_work() -> Result<()> {
    let (_temp, root) = setup_test_workspace()?;
    let port = free_port()?;
    let base_url = format!("http://127.0.0.1:{port}");
    let child = spawn_server(port, &root)?;
    let result = (|| -> Result<()> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()?;

        let health: serde_json::Value = client.get(format!("{base_url}/health")).send()?.json()?;
        assert_eq!(health["status"], "ok");
        assert_eq!(health["ready"], true);

        let models: serde_json::Value =
            client.get(format!("{base_url}/v1/models")).send()?.json()?;
        assert!(
            models["data"]
                .as_array()
                .unwrap()
                .iter()
                .any(|model| model["id"] == "vit_stt_vi_v2")
        );

        let wav = std::fs::read(root.join("baselines/phase0/vi_probe.wav"))?;
        let form = reqwest::blocking::multipart::Form::new()
            .part(
                "file",
                reqwest::blocking::multipart::Part::bytes(wav)
                    .file_name("vi_probe.wav")
                    .mime_str("audio/wav")?,
            )
            .text("model", "vit_stt_vi_v2")
            .text("response_format", "json");
        let response = client
            .post(format!("{base_url}/v1/audio/transcriptions"))
            .multipart(form)
            .send()?;
        assert!(response.headers().get("x-idempotency-status").is_none());
        let payload: serde_json::Value = response.json()?;
        assert_eq!(payload["text"], EXPECTED_VI);

        // An idempotent request executes once and returns the response contract.
        let wav = std::fs::read(root.join("baselines/phase0/vi_probe.wav"))?;
        let form = reqwest::blocking::multipart::Form::new()
            .part(
                "file",
                reqwest::blocking::multipart::Part::bytes(wav)
                    .file_name("vi_probe.wav")
                    .mime_str("audio/wav")?,
            )
            .text("model", "vit_stt_vi_v2")
            .text("response_format", "json");
        let response = client
            .post(format!("{base_url}/v1/audio/transcriptions"))
            .header("Idempotency-Key", "meeting-1-segment-1")
            .multipart(form)
            .send()?;
        assert_eq!(response.headers()["idempotency-key"], "meeting-1-segment-1");
        assert_eq!(response.headers()["x-idempotency-status"], "created");
        let payload: serde_json::Value = response.json()?;
        assert_eq!(payload["text"], EXPECTED_VI);

        // A fresh multipart boundary with the same effective request is cached.
        let wav = std::fs::read(root.join("baselines/phase0/vi_probe.wav"))?;
        let form = reqwest::blocking::multipart::Form::new()
            .part(
                "file",
                reqwest::blocking::multipart::Part::bytes(wav)
                    .file_name("vi_probe.wav")
                    .mime_str("audio/wav")?,
            )
            .text("model", "vit_stt_vi_v2")
            .text("response_format", "json");
        let response = client
            .post(format!("{base_url}/v1/audio/transcriptions"))
            .header("Idempotency-Key", "meeting-1-segment-1")
            .multipart(form)
            .send()?;
        assert_eq!(response.headers()["x-idempotency-status"], "cached");
        let payload: serde_json::Value = response.json()?;
        assert_eq!(payload["text"], EXPECTED_VI);

        // Reusing the key for different response semantics is a conflict.
        let wav = std::fs::read(root.join("baselines/phase0/vi_probe.wav"))?;
        let form = reqwest::blocking::multipart::Form::new()
            .part(
                "file",
                reqwest::blocking::multipart::Part::bytes(wav)
                    .file_name("vi_probe.wav")
                    .mime_str("audio/wav")?,
            )
            .text("model", "vit_stt_vi_v2")
            .text("response_format", "text");
        let response = client
            .post(format!("{base_url}/v1/audio/transcriptions"))
            .header("Idempotency-Key", "meeting-1-segment-1")
            .multipart(form)
            .send()?;
        assert_eq!(response.status(), reqwest::StatusCode::CONFLICT);
        assert_eq!(response.headers()["x-idempotency-status"], "conflict");
        let payload: serde_json::Value = response.json()?;
        assert_eq!(payload["error"]["code"], "idempotency_conflict");
        Ok(())
    })();
    stop_server(child);
    result
}

#[test]
fn test_error_localization() -> Result<()> {
    let (_temp, root) = setup_test_workspace()?;
    let port = free_port()?;
    let base_url = format!("http://127.0.0.1:{port}");
    let child = spawn_server(port, &root)?;
    let result = (|| -> Result<()> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;

        let wav = std::fs::read(root.join("baselines/phase0/vi_probe.wav"))?;

        // Test case 1: Non-existent model with Vietnamese preference
        let form = reqwest::blocking::multipart::Form::new()
            .part(
                "file",
                reqwest::blocking::multipart::Part::bytes(wav.clone())
                    .file_name("vi_probe.wav")
                    .mime_str("audio/wav")?,
            )
            .text("model", "non-existent-model");

        let resp = client
            .post(format!("{base_url}/v1/audio/transcriptions"))
            .header(reqwest::header::ACCEPT_LANGUAGE, "vi-VN,vi;q=0.9")
            .multipart(form)
            .send()?;

        assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);
        assert_eq!(
            resp.headers()
                .get(reqwest::header::CONTENT_LANGUAGE)
                .unwrap(),
            "vi"
        );

        let err_body: serde_json::Value = resp.json()?;
        assert_eq!(err_body["error"]["code"], "model_not_found");
        assert!(
            err_body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("Không tìm thấy model với mã ID: non-existent-model")
        );

        // Test case 2: Non-existent model with English preference
        let wav = std::fs::read(root.join("baselines/phase0/vi_probe.wav"))?;
        let form = reqwest::blocking::multipart::Form::new()
            .part(
                "file",
                reqwest::blocking::multipart::Part::bytes(wav)
                    .file_name("vi_probe.wav")
                    .mime_str("audio/wav")?,
            )
            .text("model", "non-existent-model");

        let resp = client
            .post(format!("{base_url}/v1/audio/transcriptions"))
            .header(reqwest::header::ACCEPT_LANGUAGE, "en-US,en;q=0.9")
            .multipart(form)
            .send()?;

        assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);
        assert_eq!(
            resp.headers()
                .get(reqwest::header::CONTENT_LANGUAGE)
                .unwrap(),
            "en"
        );

        let err_body: serde_json::Value = resp.json()?;
        assert_eq!(err_body["error"]["code"], "model_not_found");
        assert!(
            err_body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("unknown model id: non-existent-model")
        );

        // Test case 3: Audio decode error for a model with default language 'vi' (no Accept-Language)
        let form = reqwest::blocking::multipart::Form::new()
            .part(
                "file",
                reqwest::blocking::multipart::Part::bytes(b"invalid-audio-bytes".to_vec())
                    .file_name("bad.wav")
                    .mime_str("audio/wav")?,
            )
            .text("model", "vit_stt_vi_v2");

        let resp = client
            .post(format!("{base_url}/v1/audio/transcriptions"))
            .multipart(form)
            .send()?;

        assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
        assert_eq!(
            resp.headers()
                .get(reqwest::header::CONTENT_LANGUAGE)
                .unwrap(),
            "vi"
        );

        let err_body: serde_json::Value = resp.json()?;
        assert_eq!(err_body["error"]["code"], "audio_decode_failed");
        assert!(
            err_body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("Giải mã âm thanh thất bại")
        );

        // Test case 4: Audio decode error with explicit language field 'vi' (no Accept-Language)
        let form = reqwest::blocking::multipart::Form::new()
            .part(
                "file",
                reqwest::blocking::multipart::Part::bytes(b"invalid-audio-bytes".to_vec())
                    .file_name("bad.wav")
                    .mime_str("audio/wav")?,
            )
            .text("model", "vit_stt_vi_v2")
            .text("language", "vi");

        let resp = client
            .post(format!("{base_url}/v1/audio/transcriptions"))
            .multipart(form)
            .send()?;

        assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
        assert_eq!(
            resp.headers()
                .get(reqwest::header::CONTENT_LANGUAGE)
                .unwrap(),
            "vi"
        );

        let err_body: serde_json::Value = resp.json()?;
        assert_eq!(err_body["error"]["code"], "audio_decode_failed");
        assert!(
            err_body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("Giải mã âm thanh thất bại")
        );

        Ok(())
    })();
    stop_server(child);
    result
}

#[test]
fn transcription_requires_model_field() -> Result<()> {
    let (_temp, root) = setup_test_workspace()?;
    let port = free_port()?;
    let base_url = format!("http://127.0.0.1:{port}");
    let child = spawn_server(port, &root)?;
    let result = (|| -> Result<()> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;
        let wav = std::fs::read(root.join("baselines/phase0/vi_probe.wav"))?;
        let form = reqwest::blocking::multipart::Form::new().part(
            "file",
            reqwest::blocking::multipart::Part::bytes(wav)
                .file_name("vi_probe.wav")
                .mime_str("audio/wav")?,
        );

        let resp = client
            .post(format!("{base_url}/v1/audio/transcriptions"))
            .multipart(form)
            .send()?;

        assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
        let err_body: serde_json::Value = resp.json()?;
        assert_eq!(err_body["error"]["code"], "model_required");
        Ok(())
    })();
    stop_server(child);
    result
}

#[test]
fn health_reports_ready_after_warmup() -> Result<()> {
    let (_temp, root) = setup_test_workspace()?;
    let port = free_port()?;
    let base_url = format!("http://127.0.0.1:{port}");
    let child = spawn_server(port, &root)?;
    let result = (|| -> Result<()> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;

        let health: serde_json::Value = client.get(format!("{base_url}/health")).send()?.json()?;
        assert_eq!(health["status"], "ok");
        assert_eq!(health["ready"], true);
        Ok(())
    })();
    stop_server(child);
    result
}

#[test]
fn admin_model_status_works() -> Result<()> {
    let (_temp, root) = setup_test_workspace()?;
    let port = free_port()?;
    let base_url = format!("http://127.0.0.1:{port}");
    let child = spawn_server(port, &root)?;
    let result = (|| -> Result<()> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;

        let status: serde_json::Value = client
            .get(format!("{base_url}/admin/models/status"))
            .send()?
            .json()?;
        let models = status["models"].as_array().unwrap();
        assert!(!models.is_empty());
        let vi_model = models.iter().find(|m| m["id"] == "vit_stt_vi_v2").unwrap();
        assert_eq!(vi_model["recognizer_loaded"], true);
        assert_eq!(vi_model["warm_start"], true);
        Ok(())
    })();
    stop_server(child);
    result
}

#[test]
fn admin_warmup_works() -> Result<()> {
    let (_temp, root) = setup_test_workspace()?;
    let port = free_port()?;
    let base_url = format!("http://127.0.0.1:{port}");
    let child = spawn_server(port, &root)?;
    let result = (|| -> Result<()> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;

        // Vietnamese model should already be loaded (warm_start)
        let resp: serde_json::Value = client
            .post(format!("{base_url}/admin/warmup"))
            .header("Content-Type", "application/json")
            .body(r#"{"models": ["vit_stt_vi_v2"], "run_probe": false}"#)
            .send()?
            .json()?;
        let warmed = resp["warmed"].as_array().unwrap();
        assert_eq!(warmed.len(), 1);
        assert_eq!(warmed[0]["model"], "vit_stt_vi_v2");
        assert_eq!(warmed[0]["already_loaded"], true);
        Ok(())
    })();
    stop_server(child);
    result
}
