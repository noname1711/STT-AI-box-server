use std::ffi::OsStr;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::process::{Child, ChildStdin, ChildStdout, Command};
use std::sync::Mutex;
use tempfile::{TempDir, tempdir};
use tracing::info;

use stt_core::config::{CapuModelConfig, CapuRuntimeConfig, WorkspacePaths};
use stt_core::postprocess::{PostprocessMode, Postprocessor};
use stt_core::types::TranscriptionResult;

#[derive(Debug, thiserror::Error)]
pub enum CapuError {
    #[error("CAPU runtime unavailable: {0}")]
    Unavailable(String),
    #[error("CAPU request failed: {0}")]
    Request(String),
    #[error("CAPU request timed out: {0}")]
    Timeout(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapuResponse {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerRequest {
    command: String,
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerResponse {
    ok: bool,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug)]
pub struct OnnxCapuSpikeResult {
    pub supported: bool,
    pub reason: String,
}

pub fn probe_onnx_or_pure_rust_support(
    workspace: &WorkspacePaths,
    model: &CapuModelConfig,
) -> OnnxCapuSpikeResult {
    let model_dir = workspace.resolve(&model.model_dir);
    let model_has_onnx = std::fs::read_dir(&model_dir)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(|entry| entry.ok()))
        .map(|entry| entry.path())
        .any(|path| path.extension() == Some(OsStr::new("onnx")));

    if model_has_onnx {
        OnnxCapuSpikeResult {
            supported: true,
            reason: "CAPU directory already contains ONNX assets".to_string(),
        }
    } else {
        OnnxCapuSpikeResult {
            supported: false,
            reason: "CAPU assets are PyTorch-only (pytorch_model.bin + custom Python modules); no ONNX export or pure-Rust runtime is available in the checked-in assets".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
struct ResolvedCapuModel {
    id: String,
    model_dir: PathBuf,
    base_model_dir: PathBuf,
    device: String,
}

fn resolve_capu_model(
    workspace: &WorkspacePaths,
    config: &CapuRuntimeConfig,
    model: &CapuModelConfig,
) -> Result<ResolvedCapuModel> {
    let model_dir = workspace.resolve(&model.model_dir);
    ensure_capu_model_dir(&model_dir, &model.id)?;
    let base_model_dir = resolve_base_model_dir(
        workspace,
        &model_dir,
        model.base_model_dir.as_ref(),
        &config.base_model_name,
    )?;
    Ok(ResolvedCapuModel {
        id: model.id.clone(),
        model_dir,
        base_model_dir,
        device: model
            .device
            .clone()
            .unwrap_or_else(|| config.device.clone()),
    })
}

fn resolve_base_model_dir(
    workspace: &WorkspacePaths,
    model_dir: &Path,
    explicit_base_model_dir: Option<&PathBuf>,
    base_model_name: &str,
) -> Result<PathBuf> {
    let resolved = if let Some(base_model_dir) = explicit_base_model_dir {
        workspace.resolve(base_model_dir)
    } else if model_dir.join("base_model").is_dir() {
        model_dir.join("base_model")
    } else if let Some(parent) = model_dir.parent() {
        parent.join(base_model_name)
    } else {
        model_dir.join(base_model_name)
    };

    if resolved.join("vocab.txt").is_file() {
        Ok(resolved)
    } else {
        Err(anyhow!(
            "CAPU base model not found. Expected a directory containing vocab.txt at {}, or set base_model_dir explicitly",
            resolved.display()
        ))
    }
}

fn ensure_capu_model_dir(model_dir: &Path, model_id: &str) -> Result<()> {
    if model_dir.join("config.json").is_file() {
        return Ok(());
    }
    Err(anyhow!(
        "CAPU model '{}' not found at {}: missing config.json",
        model_id,
        model_dir.display()
    ))
}

struct WorkerCommand {
    request: WorkerRequest,
    response_tx: mpsc::Sender<Result<WorkerResponse>>,
}

struct WorkerProcess {
    child: Arc<Mutex<Child>>,
    request_tx: Option<mpsc::Sender<WorkerCommand>>,
    join_handle: Option<JoinHandle<()>>,
    timeout: Duration,
    poisoned: AtomicBool,
    #[allow(dead_code)]
    overlay_dir: TempDir,
}

impl WorkerProcess {
    fn send(&self, request: &WorkerRequest) -> Result<WorkerResponse> {
        if self.poisoned.load(Ordering::SeqCst) {
            return Err(CapuError::Unavailable("worker is not running".to_string()).into());
        }

        let request_tx = self
            .request_tx
            .as_ref()
            .context("CAPU worker channel unavailable")?;
        let (response_tx, response_rx) = mpsc::channel();
        request_tx
            .send(WorkerCommand {
                request: request.clone(),
                response_tx,
            })
            .map_err(|_| {
                self.poisoned.store(true, Ordering::SeqCst);
                CapuError::Unavailable("worker command channel closed".to_string())
            })?;

        match response_rx.recv_timeout(self.timeout) {
            Ok(response) => {
                if response.is_err() {
                    self.poisoned.store(true, Ordering::SeqCst);
                }
                response
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.poisoned.store(true, Ordering::SeqCst);
                self.terminate();
                Err(CapuError::Timeout(format!(
                    "worker did not respond within {} seconds",
                    self.timeout.as_secs()
                ))
                .into())
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.poisoned.store(true, Ordering::SeqCst);
                Err(CapuError::Unavailable("worker response channel closed".to_string()).into())
            }
        }
    }

    fn terminate(&self) {
        if let Ok(mut child) = self.child.lock() {
            if child.try_wait().ok().flatten().is_none() {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        self.request_tx.take();
        self.terminate();
        if let Some(join_handle) = self.join_handle.take() {
            let _ = join_handle.join();
        }
    }
}

#[derive(Clone)]
pub struct CapuClient {
    inner: Arc<WorkerProcess>,
}

impl std::fmt::Debug for CapuClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapuClient").finish_non_exhaustive()
    }
}

impl CapuClient {
    pub fn spawn(
        workspace: &WorkspacePaths,
        config: &CapuRuntimeConfig,
        capu_model: &CapuModelConfig,
    ) -> Result<Self> {
        let model = resolve_capu_model(workspace, config, capu_model)?;
        let timeout = Duration::from_secs(
            capu_model
                .request_timeout_seconds
                .unwrap_or(config.request_timeout_seconds)
                .max(1),
        );
        let worker_dir = workspace.resolve(&config.worker_dir);
        info!(
            python_bin = %config.python_bin.display(),
            worker_exe = ?config.worker_exe,
            worker_dir = %worker_dir.display(),
            capu_model_id = model.id.as_str(),
            device = model.device.as_str(),
            timeout_seconds = timeout.as_secs(),
            "spawning CAPU worker"
        );
        let overlay_dir = materialize_capu_overlay(&model)?;

        let python_bin = config.resolved_python_bin(workspace);

        let mut command = if let Some(worker_exe) = &config.worker_exe {
            let worker_exe = workspace.resolve(worker_exe);
            let mut command = Command::new(&worker_exe);
            command.current_dir(&workspace.root);
            command
        } else {
            let mut command = Command::new(&python_bin);
            command.arg("-m").arg("capu_worker").current_dir(worker_dir);
            command
        };
        command
            .arg("--model-dir")
            .arg(overlay_dir.path().join(&model.id))
            .arg("--device")
            .arg(&model.device)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = command.spawn().with_context(|| {
            if let Some(worker_exe) = &config.worker_exe {
                let worker_exe = workspace.resolve(worker_exe);
                format!("failed to spawn CAPU worker {}", worker_exe.display())
            } else {
                format!(
                    "failed to spawn CAPU worker with python {}",
                    python_bin.display()
                )
            }
        })?;

        let stdin = child
            .stdin
            .take()
            .context("CAPU worker stdin unavailable")?;
        let stdout = child
            .stdout
            .take()
            .context("CAPU worker stdout unavailable")?;
        let child = Arc::new(Mutex::new(child));
        let (request_tx, request_rx) = mpsc::channel();
        let child_for_thread = Arc::clone(&child);
        let join_handle = thread::spawn(move || {
            worker_loop(child_for_thread, stdin, stdout, request_rx);
        });
        let worker = WorkerProcess {
            child,
            request_tx: Some(request_tx),
            join_handle: Some(join_handle),
            timeout,
            poisoned: AtomicBool::new(false),
            overlay_dir,
        };

        let response = worker
            .send(&WorkerRequest {
                command: "ping".to_string(),
                text: String::new(),
            })
            .context("CAPU worker handshake failed")?;
        if !response.ok {
            return Err(anyhow!(
                "CAPU worker startup failed: {}",
                response
                    .error
                    .unwrap_or_else(|| "unknown error".to_string())
            ));
        }

        info!(
            capu_model_id = model.id.as_str(),
            device = model.device.as_str(),
            "CAPU worker ready"
        );

        Ok(Self {
            inner: Arc::new(worker),
        })
    }

    pub fn process_text(&self, text: &str) -> Result<String> {
        let response = self.inner.send(&WorkerRequest {
            command: "process_text".to_string(),
            text: text.to_string(),
        })?;

        if !response.ok {
            return Err(anyhow!(
                response
                    .error
                    .unwrap_or_else(|| "CAPU worker request failed".to_string())
            ));
        }

        response.text.context("CAPU worker returned no text")
    }
}

fn worker_loop(
    child: Arc<Mutex<Child>>,
    mut stdin: ChildStdin,
    stdout: ChildStdout,
    request_rx: mpsc::Receiver<WorkerCommand>,
) {
    let mut stdout = BufReader::new(stdout);

    for command in request_rx {
        let result = send_worker_request(&mut stdin, &mut stdout, &command.request);
        let should_stop = result.is_err();
        let _ = command.response_tx.send(result);
        if should_stop {
            if let Ok(mut child) = child.lock() {
                if child.try_wait().ok().flatten().is_none() {
                    let _ = child.kill();
                }
                let _ = child.wait();
            }
            break;
        }
    }
}

fn send_worker_request(
    stdin: &mut ChildStdin,
    stdout: &mut BufReader<ChildStdout>,
    request: &WorkerRequest,
) -> Result<WorkerResponse> {
    let line = serde_json::to_string(request)?;
    stdin.write_all(line.as_bytes())?;
    stdin.write_all(b"\n")?;
    stdin.flush()?;

    let mut response = String::new();
    let bytes = stdout.read_line(&mut response)?;
    if bytes == 0 {
        return Err(CapuError::Unavailable("worker closed stdout".to_string()).into());
    }

    let payload: WorkerResponse = serde_json::from_str(response.trim())?;
    Ok(payload)
}

fn materialize_capu_overlay(model: &ResolvedCapuModel) -> Result<TempDir> {
    let temp = tempdir()?;
    let overlay_model_dir = temp.path().join(&model.id);
    copy_dir_all(&model.model_dir, &overlay_model_dir)?;

    let patched = overlay_model_dir.join("config.json");
    let mut value: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&patched)?)?;
    value["pretrained_name_or_path"] =
        serde_json::Value::String(model.base_model_dir.display().to_string());
    std::fs::write(&patched, serde_json::to_vec_pretty(&value)?)?;

    Ok(temp)
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let target = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_all(&entry.path(), &target)?;
        } else if file_type.is_file() {
            std::fs::copy(entry.path(), target)?;
        } else if file_type.is_symlink() {
            #[cfg(unix)]
            {
                let target_path = std::fs::read_link(entry.path())?;
                std::os::unix::fs::symlink(target_path, target)?;
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct CapuPostprocessor {
    client: CapuClient,
}

impl CapuPostprocessor {
    pub fn new(client: CapuClient) -> Self {
        Self { client }
    }
}

impl Postprocessor for CapuPostprocessor {
    fn mode(&self) -> PostprocessMode {
        PostprocessMode::Capu
    }

    fn process_text(&self, text: &str) -> Result<String> {
        let cleaned = stt_core::clean_text(text);
        self.client.process_text(&cleaned)
    }

    fn process_result(&self, result: &TranscriptionResult) -> Result<TranscriptionResult> {
        if let Some(segments) = &result.segments {
            let mut processed_segments = Vec::with_capacity(segments.len());
            let mut full_text = Vec::with_capacity(segments.len());
            for segment in segments {
                let text = self.process_text(&segment.text)?;
                full_text.push(text.clone());
                processed_segments.push(stt_core::types::RecognizedSegment {
                    start: segment.start,
                    end: segment.end,
                    text,
                });
            }

            return Ok(TranscriptionResult {
                text: full_text.join(" "),
                model: result.model.clone(),
                language: result.language.clone(),
                duration: result.duration,
                processing_time: result.processing_time,
                segments: Some(processed_segments),
            });
        }

        Ok(TranscriptionResult {
            text: self.process_text(&result.text)?,
            model: result.model.clone(),
            language: result.language.clone(),
            duration: result.duration,
            processing_time: result.processing_time,
            segments: None,
        })
    }
}

pub fn build_capu_postprocessor(
    workspace: &WorkspacePaths,
    config: &CapuRuntimeConfig,
    capu_model: &CapuModelConfig,
) -> Result<(Option<OnnxCapuSpikeResult>, CapuPostprocessor)> {
    let spike = probe_onnx_or_pure_rust_support(workspace, capu_model);
    let client = CapuClient::spawn(workspace, config, capu_model)?;
    Ok((Some(spike), CapuPostprocessor::new(client)))
}
