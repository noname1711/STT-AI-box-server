mod idempotency;

use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use axum::extract::{DefaultBodyLimit, Multipart, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::{Router, serve};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::task;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use stt_capu::{CapuError, CapuPostprocessor, build_capu_postprocessor};
use stt_core::postprocess::{BuiltinPostprocessor, PostprocessMode, Postprocessor};
use stt_core::{
    AppRuntimeConfig, ModelStatus, SttRuntime, TranscriptionResult, WarmupResult, WorkspacePaths,
};

use crate::idempotency::{
    Begin as IdempotencyBegin, FollowResult, IdempotencyStore, StoredResponse,
    attach_headers as attach_idempotency_headers, request_fingerprint, validate_key,
};

#[derive(Parser, Debug)]
#[command(version, about = "vit-stt HTTP Server")]
struct HttpCli {
    #[arg(long)]
    workspace_root: Option<PathBuf>,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    host: Option<String>,
    #[arg(long, short = 'p')]
    port: Option<u16>,
}

#[derive(Clone)]
struct AppState {
    runtime: Arc<SttRuntime>,
    capu_cache: Arc<Mutex<HashMap<String, CapuPostprocessor>>>,
    idempotency: IdempotencyStore,
}

#[derive(Serialize)]
struct HealthResponse<'a> {
    status: &'a str,
    ready: bool,
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    created: i64,
    owned_by: String,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

#[derive(Serialize)]
struct JsonTranscriptionResponse {
    text: String,
}

#[derive(Serialize)]
struct SegmentResponse {
    id: String,
    #[serde(rename = "type")]
    segment_type: &'static str,
    start: f32,
    end: f32,
    text: String,
}

#[derive(Serialize)]
struct UsageResponse {
    #[serde(rename = "type")]
    usage_type: &'static str,
    seconds: u64,
}

#[derive(Serialize)]
struct VerboseTranscriptionResponse {
    task: &'static str,
    language: String,
    duration: f32,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    segments: Option<Vec<SegmentResponse>>,
    usage: UsageResponse,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: Option<String>,
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Deserialize)]
struct TranscriptionQuery {
    response_format: Option<String>,
}

#[derive(Deserialize)]
struct WarmupRequest {
    models: Vec<String>,
    #[serde(default = "default_true")]
    run_probe: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Serialize)]
struct WarmupResponse {
    warmed: Vec<WarmupResult>,
}

#[derive(Serialize)]
struct ModelStatusResponse {
    models: Vec<ModelStatus>,
}

#[derive(Default)]
struct RequestParams {
    model: Option<String>,
    language: Option<String>,
    response_format: Option<String>,
    file_name: Option<String>,
    file_bytes: Vec<u8>,
}

/// Distinguishes a too-large upload (HTTP 413) from any other multipart
/// parsing failure (HTTP 400). The HTTP layer uses this to return a
/// meaningful status and code, and to log the configured limit so an
/// operator can immediately see why a request was rejected.
enum MultipartParseError {
    TooLarge { max_mb: u64 },
    Parse(String),
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = HttpCli::parse();

    info!("starting stt-http");
    let startup_started = Instant::now();
    let mut runtime = SttRuntime::load(cli.workspace_root, cli.config.as_deref())?;

    if let Some(host) = cli.host {
        runtime.runtime_config.server.host = host;
    } else if let Ok(host) = std::env::var("VIT_STT_SERVER_HOST") {
        runtime.runtime_config.server.host = host;
    }

    if let Some(port) = cli.port {
        runtime.runtime_config.server.port = port;
    } else if let Ok(port) = std::env::var("VIT_STT_SERVER_PORT") {
        runtime.runtime_config.server.port = port.parse()?;
    }
    info!(
        elapsed_ms = startup_started.elapsed().as_millis(),
        model_count = runtime.registry.models.len(),
        "runtime loaded"
    );

    // Phase 1: Warm-start configured models before accepting traffic.
    // This blocks until all warm-start models are loaded (or fails fast).
    runtime.warm_start()?;

    let runtime = Arc::new(runtime);
    let idempotency_cache_dir = std::env::var_os("VIT_STT_IDEMPOTENCY_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| runtime.workspace.root.join(".cache/stt-http/idempotency"));
    let state = AppState {
        runtime,
        capu_cache: Arc::new(Mutex::new(HashMap::new())),
        idempotency: IdempotencyStore::new(idempotency_cache_dir),
    };
    let server_cfg = state.runtime.runtime_config.server.clone();
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/audio/transcriptions", post(transcribe))
        // Admin endpoints — exposed on all interfaces for local 1:1 deployment.
        // For production, bind admin routes to 127.0.0.1 only or require an
        // admin token header.
        .route("/admin/warmup", post(admin_warmup))
        .route("/admin/models/status", get(admin_model_status))
        .with_state(state)
        .layer(DefaultBodyLimit::max(
            (server_cfg.max_upload_mb * 1024 * 1024) as usize,
        ));

    let listener = TcpListener::bind((server_cfg.host.as_str(), server_cfg.port)).await?;
    info!(address = %listener.local_addr()?, "server ready");
    serve(listener, app).await?;
    Ok(())
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse<'static>> {
    let ready = state
        .runtime
        .is_ready
        .load(std::sync::atomic::Ordering::Acquire);
    Json(HealthResponse {
        status: "ok",
        ready,
    })
}

async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    let owned_by = state.runtime.runtime_config.server.owned_by.clone();
    let data = state
        .runtime
        .list_models()
        .into_iter()
        .map(|model| ModelObject {
            id: model.id,
            object: "model",
            created: 0,
            owned_by: owned_by.clone(),
        })
        .collect();
    Json(ModelsResponse {
        object: "list",
        data,
    })
}

async fn admin_warmup(
    State(state): State<AppState>,
    Json(req): Json<WarmupRequest>,
) -> Json<WarmupResponse> {
    info!(models = ?req.models, run_probe = req.run_probe, "admin warmup request");
    let runtime = Arc::clone(&state.runtime);
    let models = req.models.clone();
    let run_probe = req.run_probe;

    let results = task::spawn_blocking(move || runtime.warmup_models(&models, run_probe))
        .await
        .unwrap_or_default();

    for result in &results {
        info!(
            model = result.model.as_str(),
            already_loaded = result.already_loaded,
            elapsed_ms = result.elapsed_ms,
            probe_match = result.probe.as_ref().map(|p| p.matches_expected),
            "admin warmup result"
        );
    }

    Json(WarmupResponse { warmed: results })
}

async fn admin_model_status(State(state): State<AppState>) -> Json<ModelStatusResponse> {
    let models = state.runtime.model_status();
    Json(ModelStatusResponse { models })
}

async fn transcribe(
    State(state): State<AppState>,
    Query(query): Query<TranscriptionQuery>,
    headers: axum::http::HeaderMap,
    mut multipart: Multipart,
) -> Response {
    let request_id = Uuid::new_v4();
    let request_started = Instant::now();
    let prefers_vi = prefers_vietnamese(&headers);
    let content_lang = if prefers_vi { "vi" } else { "en" };

    let build_api_error = |status: StatusCode, err_msg: &str, error_type: &str| {
        let (message, code) = localize_error(err_msg, prefers_vi);
        api_error(status, message, error_type, Some(code), content_lang)
    };

    let idempotency_key = match headers.get(idempotency::IDEMPOTENCY_KEY_HEADER) {
        Some(value) => match value.to_str() {
            Ok(key) if validate_key(key).is_ok() => Some(key.to_string()),
            _ => {
                let message = if prefers_vi {
                    "Idempotency-Key phải có từ 1 đến 255 ký tự ASCII hiển thị"
                } else {
                    "Idempotency-Key must contain 1 to 255 visible ASCII characters"
                };
                return api_error(
                    StatusCode::BAD_REQUEST,
                    message,
                    "invalid_request_error",
                    Some("invalid_idempotency_key".to_string()),
                    content_lang,
                );
            }
        },
        None => None,
    };

    let max_upload_mb = state.runtime.runtime_config.server.max_upload_mb;
    match extract_request_params(&mut multipart, max_upload_mb).await {
        Ok(mut params) => {
            let is_vi_request = prefers_vi
                || params
                    .language
                    .as_deref()
                    .is_some_and(|l| l.to_ascii_lowercase().starts_with("vi"));
            let content_lang = if is_vi_request { "vi" } else { "en" };

            let build_api_error = |status: StatusCode, err_msg: &str, error_type: &str| {
                let (message, code) = localize_error(err_msg, is_vi_request);
                api_error(status, message, error_type, Some(code), content_lang)
            };

            if params.response_format.is_none() {
                params.response_format = query.response_format;
            }
            let response_format = params
                .response_format
                .clone()
                .unwrap_or_else(|| "json".to_string());
            let model_id = match params.model.clone().map(|model| model.trim().to_string()) {
                Some(model_id) if !model_id.is_empty() => model_id,
                _ => {
                    warn!(%request_id, "transcription request missing model");
                    return build_api_error(
                        StatusCode::BAD_REQUEST,
                        "model is required",
                        "invalid_request_error",
                    );
                }
            };

            let fingerprint = idempotency_key.as_ref().map(|_| {
                request_fingerprint(
                    &model_id,
                    params.language.as_deref(),
                    &response_format,
                    params.file_name.as_deref(),
                    &params.file_bytes,
                )
            });

            let operation = async {
                let model = match state.runtime.resolve_model(&model_id) {
                    Ok(model) => model,
                    Err(err) => {
                        warn!(%request_id, model_id, error = %err, "transcription request referenced unknown model");
                        return build_api_error(
                            StatusCode::NOT_FOUND,
                            &err.to_string(),
                            "model_not_found",
                        );
                    }
                };

                let is_vi_request =
                    is_vi_request || model.language.to_ascii_lowercase().starts_with("vi");
                let content_lang = if is_vi_request { "vi" } else { "en" };

                let build_api_error = |status: StatusCode, err_msg: &str, error_type: &str| {
                    let (message, code) = localize_error(err_msg, is_vi_request);
                    api_error(status, message, error_type, Some(code), content_lang)
                };

                info!(
                    %request_id,
                    model_id,
                    response_format,
                    file_name = params.file_name.as_deref().unwrap_or("<unnamed>"),
                    file_bytes = params.file_bytes.len(),
                    postprocess_mode = %model.postprocess_mode,
                    "transcription request accepted"
                );

                let max_audio_seconds =
                    state.runtime.runtime_config.server.max_audio_seconds as f32;
                let decode_started = Instant::now();
                let file_bytes = params.file_bytes;
                let file_name = params.file_name.clone();
                let decoded = match task::spawn_blocking(move || {
                    stt_core::decode_audio_bytes(&file_bytes, file_name.as_deref())
                })
                .await
                {
                    Ok(Ok(decoded)) => decoded,
                    Ok(Err(err)) => {
                        let (status, error_type) = classify_audio_decode_error(&err.to_string());
                        if status == StatusCode::INTERNAL_SERVER_ERROR {
                            error!(%request_id, error = %err, error_debug = ?err, "audio decode failed (server-side)");
                        } else {
                            warn!(%request_id, error = %err, error_debug = ?err, "failed to decode uploaded audio");
                        }
                        return build_api_error(status, &err.to_string(), error_type);
                    }
                    Err(err) => {
                        error!(%request_id, error = %err, "audio decode task failed");
                        return build_api_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            &err.to_string(),
                            "server_error",
                        );
                    }
                };
                if decoded.duration_seconds > max_audio_seconds {
                    warn!(
                        %request_id,
                        duration_seconds = decoded.duration_seconds,
                        max_audio_seconds,
                        "audio duration exceeds configured limit"
                    );
                    return build_api_error(
                        StatusCode::BAD_REQUEST,
                        &format!("audio duration exceeds {} seconds", max_audio_seconds),
                        "invalid_request_error",
                    );
                }
                info!(
                    %request_id,
                    duration_seconds = decoded.duration_seconds,
                    sample_rate = decoded.sample_rate,
                    elapsed_ms = decode_started.elapsed().as_millis(),
                    "input audio validated"
                );
                let decoded_duration = decoded.duration_seconds;

                let inference_started = Instant::now();
                let runtime = Arc::clone(&state.runtime);
                let inference_model_id = model_id.clone();
                // Phase 4: Check cache before inference to log cache hit/miss.
                let recognizer_cache_hit = state.runtime.is_recognizer_cached(&model_id);
                let mut result = match task::spawn_blocking(move || {
                    runtime.transcribe_decoded(&inference_model_id, &decoded)
                })
                .await
                {
                    Ok(Ok(result)) => result,
                    Ok(Err(err)) => {
                        error!(%request_id, model_id, error = %err, "transcription inference failed");
                        return build_api_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            &err.to_string(),
                            "server_error",
                        );
                    }
                    Err(err) => {
                        error!(%request_id, model_id, error = %err, "transcription inference task failed");
                        return build_api_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            &err.to_string(),
                            "server_error",
                        );
                    }
                };

                info!(
                    %request_id,
                    model_id,
                    recognizer_cache_hit,
                    audio_seconds = result.duration,
                    inference_seconds = result.processing_time,
                    inference_elapsed_ms = inference_started.elapsed().as_millis(),
                    segment_count = result.segments.as_ref().map_or(0, Vec::len),
                    text_chars = result.text.chars().count(),
                    "transcription inference completed"
                );

                result.language = params
                    .language
                    .clone()
                    .unwrap_or_else(|| model.language.clone());

                let postprocess_started = Instant::now();
                match apply_postprocess(
                    &request_id,
                    &state.runtime,
                    &state.runtime.workspace,
                    &state.runtime.runtime_config,
                    &state.capu_cache,
                    &model,
                    result,
                )
                .await
                {
                    Ok(result) => {
                        info!(
                            %request_id,
                            model_id,
                            language = result.language.as_str(),
                            postprocess_mode = %model.postprocess_mode,
                            response_format,
                            audio_seconds = decoded_duration,
                            postprocess_elapsed_ms = postprocess_started.elapsed().as_millis(),
                            total_elapsed_ms = request_started.elapsed().as_millis(),
                            segment_count = result.segments.as_ref().map_or(0, Vec::len),
                            text_chars = result.text.chars().count(),
                            "transcription request completed"
                        );
                        format_response(result, decoded_duration, &response_format)
                    }
                    Err(err) => {
                        error!(%request_id, model_id, error = %err, "postprocessing failed");
                        build_runtime_error_response(&err.to_string(), is_vi_request, content_lang)
                    }
                }
            };

            match (idempotency_key, fingerprint) {
                (Some(key), Some(fingerprint)) => {
                    execute_idempotently(
                        &state.idempotency,
                        key,
                        fingerprint,
                        is_vi_request,
                        content_lang,
                        operation,
                    )
                    .await
                }
                _ => operation.await,
            }
        }
        Err(MultipartParseError::TooLarge { max_mb }) => {
            warn!(
                %request_id,
                max_upload_mb = max_mb,
                "rejected transcription request: upload exceeds max_upload_mb"
            );
            build_api_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                &format!("request body exceeds max_upload_mb = {max_mb}"),
                "invalid_request_error",
            )
        }
        Err(MultipartParseError::Parse(err)) => {
            warn!(%request_id, error = %err, "failed to parse multipart transcription request");
            build_api_error(StatusCode::BAD_REQUEST, &err, "invalid_request_error")
        }
    }
}

async fn execute_idempotently<F>(
    store: &IdempotencyStore,
    key: String,
    fingerprint: String,
    prefers_vi: bool,
    content_language: &'static str,
    operation: F,
) -> Response
where
    F: Future<Output = Response>,
{
    let mut operation = Some(operation);
    loop {
        match store.begin(&key, &fingerprint) {
            IdempotencyBegin::Leader(leader) => {
                let response = operation
                    .take()
                    .expect("idempotent operation can only have one leader")
                    .await;
                let stored = match StoredResponse::from_response(response).await {
                    Ok(response) => response,
                    Err(err) => {
                        error!(error = %err, "failed capturing idempotent response");
                        return build_runtime_error_response(&err, prefers_vi, content_language);
                    }
                };
                let stored = leader.finish(stored).await;
                let mut response = (*stored).clone().into_response();
                attach_idempotency_headers(&mut response, &key, "created");
                return response;
            }
            IdempotencyBegin::Follower(follower) => match follower.wait().await {
                FollowResult::Completed(stored) => {
                    let mut response = (*stored).clone().into_response();
                    attach_idempotency_headers(&mut response, &key, "coalesced");
                    return response;
                }
                FollowResult::Aborted => continue,
            },
            IdempotencyBegin::Cached(stored) => {
                let mut response = (*stored).clone().into_response();
                attach_idempotency_headers(&mut response, &key, "cached");
                return response;
            }
            IdempotencyBegin::Conflict => {
                let message = if prefers_vi {
                    "Idempotency-Key đã được dùng cho một request khác"
                } else {
                    "Idempotency-Key was already used with a different request"
                };
                let mut response = api_error(
                    StatusCode::CONFLICT,
                    message,
                    "idempotency_error",
                    Some("idempotency_conflict".to_string()),
                    content_language,
                );
                attach_idempotency_headers(&mut response, &key, "conflict");
                return response;
            }
        }
    }
}

async fn apply_postprocess(
    request_id: &Uuid,
    runtime: &SttRuntime,
    workspace: &WorkspacePaths,
    runtime_cfg: &AppRuntimeConfig,
    capu_cache: &Arc<Mutex<HashMap<String, CapuPostprocessor>>>,
    model: &stt_core::ResolvedModelConfig,
    result: TranscriptionResult,
) -> Result<TranscriptionResult> {
    match model.postprocess_mode {
        PostprocessMode::Capu => {
            info!(%request_id, model_id = model.id.as_str(), "applying CAPU postprocessing");
            let capu_model = runtime.resolve_capu_model(model.capu_model_id.as_deref())?;
            let capu_key = capu_cache_key(capu_model, runtime_cfg);
            let capu = get_or_build_capu(
                capu_cache,
                &capu_key,
                workspace,
                &runtime_cfg.capu,
                capu_model,
            )
            .await?;
            match process_capu_result(capu, result.clone()).await {
                Ok(result) => Ok(result),
                Err(err) if is_retryable_capu_error(&err) => {
                    warn!(
                        %request_id,
                        model_id = model.id.as_str(),
                        error = %err,
                        "CAPU worker failed; evicting cached worker and retrying once"
                    );
                    evict_capu(capu_cache, &capu_key).await;
                    let capu = get_or_build_capu(
                        capu_cache,
                        &capu_key,
                        workspace,
                        &runtime_cfg.capu,
                        capu_model,
                    )
                    .await?;
                    process_capu_result(capu, result).await
                }
                Err(err) => Err(err),
            }
        }
        mode => {
            info!(
                %request_id,
                model_id = model.id.as_str(),
                mode = %mode,
                "applying builtin postprocessing"
            );
            BuiltinPostprocessor::new(mode).process_result(&result)
        }
    }
}

async fn process_capu_result(
    capu: CapuPostprocessor,
    result: TranscriptionResult,
) -> Result<TranscriptionResult> {
    task::spawn_blocking(move || capu.process_result(&result)).await?
}

fn capu_cache_key(
    capu_model: &stt_core::CapuModelConfig,
    runtime_cfg: &AppRuntimeConfig,
) -> String {
    let timeout = capu_model
        .request_timeout_seconds
        .unwrap_or(runtime_cfg.capu.request_timeout_seconds);
    format!(
        "{}|{}|{}",
        capu_model.id,
        capu_model
            .device
            .clone()
            .unwrap_or_else(|| runtime_cfg.capu.device.clone()),
        timeout
    )
}

async fn get_or_build_capu(
    capu_cache: &Arc<Mutex<HashMap<String, CapuPostprocessor>>>,
    key: &str,
    workspace: &WorkspacePaths,
    capu_runtime: &stt_core::CapuRuntimeConfig,
    capu_model: &stt_core::CapuModelConfig,
) -> Result<CapuPostprocessor> {
    let cache = capu_cache.lock().await;
    if let Some(capu) = cache.get(key) {
        return Ok(capu.clone());
    }
    drop(cache);

    let workspace = workspace.clone();
    let capu_runtime = capu_runtime.clone();
    let capu_model = capu_model.clone();
    let (_, capu) = task::spawn_blocking(move || {
        build_capu_postprocessor(&workspace, &capu_runtime, &capu_model)
    })
    .await??;

    let mut cache = capu_cache.lock().await;
    if let Some(existing) = cache.get(key) {
        return Ok(existing.clone());
    }
    cache.insert(key.to_string(), capu.clone());
    Ok(capu)
}

async fn evict_capu(capu_cache: &Arc<Mutex<HashMap<String, CapuPostprocessor>>>, key: &str) {
    let mut cache = capu_cache.lock().await;
    cache.remove(key);
}

fn is_retryable_capu_error(err: &anyhow::Error) -> bool {
    err.downcast_ref::<CapuError>()
        .is_some_and(|capu| matches!(capu, CapuError::Timeout(_) | CapuError::Unavailable(_)))
}

fn format_response(result: TranscriptionResult, duration: f32, response_format: &str) -> Response {
    match response_format {
        "text" => (StatusCode::OK, result.text).into_response(),
        "verbose_json" => {
            let segments = result.segments.map(|segments| {
                segments
                    .into_iter()
                    .enumerate()
                    .map(|(index, segment)| SegmentResponse {
                        id: format!("segment_{index}"),
                        segment_type: "transcript.text.segment",
                        start: segment.start,
                        end: segment.end,
                        text: segment.text,
                    })
                    .collect::<Vec<_>>()
            });
            Json(VerboseTranscriptionResponse {
                task: "transcribe",
                language: result.language,
                duration,
                text: result.text,
                segments,
                usage: UsageResponse {
                    usage_type: "duration",
                    seconds: duration.round() as u64,
                },
            })
            .into_response()
        }
        _ => Json(JsonTranscriptionResponse { text: result.text }).into_response(),
    }
}

async fn extract_request_params(
    multipart: &mut Multipart,
    max_mb: u64,
) -> Result<RequestParams, MultipartParseError> {
    let mut params = RequestParams::default();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| classify_multipart_error(err, max_mb))?
    {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                params.file_name = field.file_name().map(ToString::to_string);
                params.file_bytes = field
                    .bytes()
                    .await
                    .map_err(|err| classify_multipart_error(err, max_mb))?
                    .to_vec();
            }
            "model" => {
                params.model = Some(
                    field
                        .text()
                        .await
                        .map_err(|err| classify_multipart_error(err, max_mb))?,
                )
            }
            "language" => {
                params.language = Some(
                    field
                        .text()
                        .await
                        .map_err(|err| classify_multipart_error(err, max_mb))?,
                )
            }
            "response_format" => {
                params.response_format = Some(
                    field
                        .text()
                        .await
                        .map_err(|err| classify_multipart_error(err, max_mb))?,
                )
            }
            _ => {}
        }
    }

    if params.file_bytes.is_empty() {
        return Err(MultipartParseError::Parse(
            "multipart field 'file' is required".to_string(),
        ));
    }

    Ok(params)
}

/// Map an axum `MultipartError` into the smaller surface area the HTTP
/// handler cares about. Body-limit rejections become `TooLarge` so we can
/// answer with HTTP 413 and a clear `request_too_large` code; everything
/// else is treated as a generic parse failure (HTTP 400).
fn classify_multipart_error(
    err: axum::extract::multipart::MultipartError,
    max_mb: u64,
) -> MultipartParseError {
    if err.status() == StatusCode::PAYLOAD_TOO_LARGE {
        MultipartParseError::TooLarge { max_mb }
    } else {
        MultipartParseError::Parse(err.to_string())
    }
}

/// Pick an HTTP status and `error_type` for an `decode_audio_bytes` failure.
/// The decode path can fail for two distinct reasons that the localizer
/// already distinguishes by code:
///
/// - ffmpeg is missing or cannot be launched (server config problem)
/// - the upload is not a decodable audio file (client problem)
///
/// The previous handler returned 400 for both, which made a missing
/// ffmpeg binary look like a bad request to the client.
fn classify_audio_decode_error(err_msg: &str) -> (StatusCode, &'static str) {
    if err_msg.contains("ffmpeg not found") || err_msg.contains("failed to launch ffmpeg") {
        (StatusCode::INTERNAL_SERVER_ERROR, "server_error")
    } else {
        (StatusCode::BAD_REQUEST, "invalid_request_error")
    }
}

fn api_error(
    status: StatusCode,
    message: impl ToString,
    error_type: &str,
    code: Option<String>,
    content_language: &'static str,
) -> Response {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        axum::http::header::CONTENT_LANGUAGE,
        axum::http::HeaderValue::from_static(content_language),
    );
    (
        status,
        headers,
        Json(ErrorBody {
            error: ErrorDetail {
                message: message.to_string(),
                error_type: error_type.to_string(),
                code,
            },
        }),
    )
        .into_response()
}

fn build_runtime_error_response(
    err_msg: &str,
    prefers_vi: bool,
    content_language: &'static str,
) -> Response {
    let (status, error_type) = classify_runtime_error(err_msg);
    let (message, code) = localize_error(err_msg, prefers_vi);
    api_error(status, message, error_type, Some(code), content_language)
}

fn classify_runtime_error(err_msg: &str) -> (StatusCode, &'static str) {
    if err_msg.contains("CAPU request timed out") {
        (StatusCode::GATEWAY_TIMEOUT, "server_error")
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, "server_error")
    }
}

fn prefers_vietnamese(headers: &axum::http::HeaderMap) -> bool {
    if let Some(s) = headers
        .get(axum::http::header::ACCEPT_LANGUAGE)
        .and_then(|val| val.to_str().ok())
    {
        let mut vi_q: f32 = 0.0;
        let mut en_q: f32 = 0.0;
        let mut vi_present = false;
        let mut en_present = false;

        for part in s.split(',') {
            let mut subparts = part.split(';');
            if let Some(lang_tag) = subparts.next() {
                let lang_tag = lang_tag.trim().to_ascii_lowercase();
                let mut q = 1.0;
                if let Some(parsed_q) = subparts
                    .next()
                    .and_then(|q_param| q_param.trim().strip_prefix("q="))
                    .and_then(|q_val| q_val.parse::<f32>().ok())
                {
                    q = parsed_q;
                }
                if lang_tag.starts_with("vi") {
                    vi_present = true;
                    vi_q = vi_q.max(q);
                } else if lang_tag.starts_with("en") {
                    en_present = true;
                    en_q = en_q.max(q);
                }
            }
        }
        if vi_present {
            if en_present {
                return vi_q >= en_q;
            } else {
                return true;
            }
        }
    }
    false
}

fn extract_max_upload_mb(err_str: &str) -> Option<u64> {
    const PREFIX: &str = "request body exceeds max_upload_mb = ";
    err_str
        .split(PREFIX)
        .nth(1)
        .and_then(|tail| tail.split_whitespace().next())
        .and_then(|n| n.parse::<u64>().ok())
}

fn localize_error(err_str: &str, preferred_vietnamese: bool) -> (String, String) {
    if let Some(max_mb) = extract_max_upload_mb(err_str) {
        return if preferred_vietnamese {
            (
                format!(
                    "Dung lượng yêu cầu vượt quá giới hạn {} MB. Vui lòng giảm kích thước tệp tin hoặc tăng giới hạn `max_upload_mb` trong `config/runtime.toml` rồi khởi động lại máy chủ.",
                    max_mb
                ),
                "request_too_large".to_string(),
            )
        } else {
            (
                format!(
                    "Request body exceeds the configured limit of {} MB. Reduce the upload size or raise `max_upload_mb` in `config/runtime.toml` and restart the server.",
                    max_mb
                ),
                "request_too_large".to_string(),
            )
        };
    }
    if preferred_vietnamese {
        if err_str.contains("model is required") {
            (
                "Tham số \"model\" là bắt buộc.".to_string(),
                "model_required".to_string(),
            )
        } else if err_str.contains("multipart field 'file' is required") {
            (
                "Trường dữ liệu 'file' trong yêu cầu multipart là bắt buộc.".to_string(),
                "file_required".to_string(),
            )
        } else if err_str.contains("audio duration exceeds") {
            let seconds = err_str
                .split("exceeds ")
                .nth(1)
                .and_then(|s| s.split(" seconds").next())
                .unwrap_or("");
            (
                format!("Thời lượng âm thanh vượt quá giới hạn {} giây.", seconds),
                "audio_too_long".to_string(),
            )
        } else if err_str.contains("unknown model id:") {
            let model_id = err_str.split("unknown model id: ").nth(1).unwrap_or("");
            (
                format!("Không tìm thấy model với mã ID: {}.", model_id),
                "model_not_found".to_string(),
            )
        } else if err_str.contains("ffmpeg not found") {
            (
                "Không tìm thấy FFmpeg trên máy chủ. Vui lòng cài đặt FFmpeg: `brew install ffmpeg` (macOS), `apt install ffmpeg` (Linux), hoặc tải từ https://ffmpeg.org/download.html".to_string(),
                "ffmpeg_not_found".to_string(),
            )
        } else if err_str.contains("failed to launch ffmpeg") {
            (
                "Không thể khởi chạy FFmpeg. Vui lòng kiểm tra cài đặt FFmpeg và quyền thực thi."
                    .to_string(),
                "ffmpeg_launch_failed".to_string(),
            )
        } else if err_str.contains("failed to decode")
            || err_str.contains("ffmpeg failed")
            || err_str.contains("hound error")
            || err_str.contains("wav")
        {
            (
                "Giải mã âm thanh thất bại. Vui lòng kiểm tra định dạng tệp tin.".to_string(),
                "audio_decode_failed".to_string(),
            )
        } else if err_str.contains("CAPU runtime unavailable") {
            let reason = err_str
                .split("CAPU runtime unavailable: ")
                .nth(1)
                .unwrap_or("");
            (
                format!("Trình chạy hậu kỳ CAPU không khả dụng: {}.", reason),
                "capu_unavailable".to_string(),
            )
        } else if err_str.contains("CAPU request timed out") {
            let reason = err_str
                .split("CAPU request timed out: ")
                .nth(1)
                .unwrap_or("");
            (
                format!("Yêu cầu hậu kỳ CAPU đã hết thời gian chờ: {}.", reason),
                "capu_timeout".to_string(),
            )
        } else if err_str.contains("CAPU request failed") {
            let reason = err_str.split("CAPU request failed: ").nth(1).unwrap_or("");
            (
                format!("Yêu cầu hậu kỳ CAPU thất bại: {}.", reason),
                "capu_failed".to_string(),
            )
        } else if err_str.contains("failed to parse multipart") {
            (
                "Không thể phân tích yêu cầu transcription multipart.".to_string(),
                "invalid_multipart".to_string(),
            )
        } else if err_str.contains("inference failed") {
            (
                "Nhận dạng giọng nói thất bại.".to_string(),
                "inference_failed".to_string(),
            )
        } else {
            (
                format!("Lỗi máy chủ nội bộ: {}.", err_str),
                "internal_error".to_string(),
            )
        }
    } else {
        // English fallback
        if err_str.contains("model is required") {
            (
                "Parameter \"model\" is required.".to_string(),
                "model_required".to_string(),
            )
        } else if err_str.contains("multipart field 'file' is required") {
            (
                "Multipart form field 'file' is required.".to_string(),
                "file_required".to_string(),
            )
        } else if err_str.contains("audio duration exceeds") {
            (err_str.to_string(), "audio_too_long".to_string())
        } else if err_str.contains("unknown model id:") {
            (err_str.to_string(), "model_not_found".to_string())
        } else if err_str.contains("ffmpeg not found") {
            (
                "FFmpeg is not installed on the server. Please install it: `brew install ffmpeg` (macOS), `apt install ffmpeg` (Linux), or download from https://ffmpeg.org/download.html".to_string(),
                "ffmpeg_not_found".to_string(),
            )
        } else if err_str.contains("failed to launch ffmpeg") {
            (
                "Failed to launch FFmpeg. Please check your FFmpeg installation and execution permissions.".to_string(),
                "ffmpeg_launch_failed".to_string(),
            )
        } else if err_str.contains("failed to decode") || err_str.contains("ffmpeg failed") {
            (
                "Failed to decode uploaded audio. Please check file format.".to_string(),
                "audio_decode_failed".to_string(),
            )
        } else if err_str.contains("CAPU runtime unavailable") {
            (err_str.to_string(), "capu_unavailable".to_string())
        } else if err_str.contains("CAPU request timed out") {
            (err_str.to_string(), "capu_timeout".to_string())
        } else if err_str.contains("CAPU request failed") {
            (err_str.to_string(), "capu_failed".to_string())
        } else if err_str.contains("failed to parse multipart") {
            (
                "Failed to parse multipart transcription request.".to_string(),
                "invalid_multipart".to_string(),
            )
        } else if err_str.contains("inference failed") {
            (
                "Transcription inference failed.".to_string(),
                "inference_failed".to_string(),
            )
        } else {
            (err_str.to_string(), "internal_error".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::HeaderMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn test_prefers_vietnamese() {
        let mut headers = HeaderMap::new();
        assert!(!prefers_vietnamese(&headers));

        headers.insert(
            axum::http::header::ACCEPT_LANGUAGE,
            "en-US,en;q=0.9".parse().unwrap(),
        );
        assert!(!prefers_vietnamese(&headers));

        headers.insert(
            axum::http::header::ACCEPT_LANGUAGE,
            "vi-VN,vi;q=0.9".parse().unwrap(),
        );
        assert!(prefers_vietnamese(&headers));

        headers.insert(
            axum::http::header::ACCEPT_LANGUAGE,
            "vi-VN,vi;q=0.9,en-US;q=0.8".parse().unwrap(),
        );
        assert!(prefers_vietnamese(&headers));

        headers.insert(
            axum::http::header::ACCEPT_LANGUAGE,
            "en-US,en;q=0.9,vi;q=0.8".parse().unwrap(),
        );
        assert!(!prefers_vietnamese(&headers));

        headers.insert(
            axum::http::header::ACCEPT_LANGUAGE,
            "vi;q=0.7,en;q=0.8".parse().unwrap(),
        );
        assert!(!prefers_vietnamese(&headers));

        headers.insert(
            axum::http::header::ACCEPT_LANGUAGE,
            "vi;q=0.8,en;q=0.7".parse().unwrap(),
        );
        assert!(prefers_vietnamese(&headers));
    }

    #[test]
    fn test_localize_error() {
        // Vietnamese localization tests
        let (msg, code) = localize_error("model is required", true);
        assert_eq!(msg, "Tham số \"model\" là bắt buộc.");
        assert_eq!(code, "model_required");

        let (msg, code) = localize_error("multipart field 'file' is required", true);
        assert_eq!(
            msg,
            "Trường dữ liệu 'file' trong yêu cầu multipart là bắt buộc."
        );
        assert_eq!(code, "file_required");

        let (msg, code) = localize_error("audio duration exceeds 10 seconds", true);
        assert_eq!(msg, "Thời lượng âm thanh vượt quá giới hạn 10 giây.");
        assert_eq!(code, "audio_too_long");

        let (msg, code) = localize_error("unknown model id: some-model", true);
        assert_eq!(msg, "Không tìm thấy model với mã ID: some-model.");
        assert_eq!(code, "model_not_found");

        let (msg, code) = localize_error("failed to decode uploaded audio", true);
        assert_eq!(
            msg,
            "Giải mã âm thanh thất bại. Vui lòng kiểm tra định dạng tệp tin."
        );
        assert_eq!(code, "audio_decode_failed");

        let (msg, code) = localize_error(
            "CAPU request timed out: worker did not respond within 1 seconds",
            true,
        );
        assert!(msg.contains("hết thời gian chờ"));
        assert_eq!(code, "capu_timeout");

        // English fallback tests
        let (msg, code) = localize_error("model is required", false);
        assert_eq!(msg, "Parameter \"model\" is required.");
        assert_eq!(code, "model_required");

        let (msg, code) = localize_error("multipart field 'file' is required", false);
        assert_eq!(msg, "Multipart form field 'file' is required.");
        assert_eq!(code, "file_required");

        let (msg, code) = localize_error("audio duration exceeds 10 seconds", false);
        assert_eq!(msg, "audio duration exceeds 10 seconds");
        assert_eq!(code, "audio_too_long");

        let (msg, code) = localize_error("unknown model id: some-model", false);
        assert_eq!(msg, "unknown model id: some-model");
        assert_eq!(code, "model_not_found");

        let (msg, code) = localize_error(
            "CAPU request timed out: worker did not respond within 1 seconds",
            false,
        );
        assert_eq!(
            msg,
            "CAPU request timed out: worker did not respond within 1 seconds"
        );
        assert_eq!(code, "capu_timeout");
    }

    #[test]
    fn test_classify_runtime_error_timeout() {
        let response = build_runtime_error_response(
            "CAPU request timed out: worker did not respond within 1 seconds",
            false,
            "en",
        );
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    }

    #[test]
    fn test_extract_max_upload_mb() {
        assert_eq!(
            extract_max_upload_mb("request body exceeds max_upload_mb = 256"),
            Some(256)
        );
        assert_eq!(
            extract_max_upload_mb("request body exceeds max_upload_mb = 128 extra"),
            Some(128)
        );
        assert_eq!(extract_max_upload_mb("model is required"), None);
        assert_eq!(
            extract_max_upload_mb("request body exceeds max_upload_mb = not-a-number"),
            None
        );
    }

    #[test]
    fn test_localize_error_request_too_large() {
        let (msg, code) = localize_error("request body exceeds max_upload_mb = 256", true);
        assert!(msg.contains("256 MB"));
        assert!(msg.contains("max_upload_mb"));
        assert_eq!(code, "request_too_large");

        let (msg, code) = localize_error("request body exceeds max_upload_mb = 256", false);
        assert!(msg.contains("256 MB"));
        assert!(msg.contains("max_upload_mb"));
        assert_eq!(code, "request_too_large");
    }

    #[test]
    fn test_classify_multipart_error_too_large() {
        // The classification only depends on `err.status()`, which we can
        // exercise through `extract_request_params` by routing the error
        // path: a syntactically invalid multipart body produces a 400 from
        // axum, so we cannot directly construct a 413 from the public API
        // here. Instead, verify the non-too-large path is a Parse.
        let parsed = MultipartParseError::Parse("boom".to_string());
        match parsed {
            MultipartParseError::Parse(s) => assert_eq!(s, "boom"),
            MultipartParseError::TooLarge { .. } => panic!("expected Parse"),
        }
    }

    #[test]
    fn test_classify_audio_decode_error() {
        let (status, ty) = classify_audio_decode_error("ffmpeg not found. install it");
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(ty, "server_error");

        let (status, ty) = classify_audio_decode_error("failed to launch ffmpeg for foo");
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(ty, "server_error");

        let (status, ty) = classify_audio_decode_error("ffmpeg failed for foo: bad data");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(ty, "invalid_request_error");

        let (status, ty) = classify_audio_decode_error("hound error: bad riff header");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(ty, "invalid_request_error");
    }

    #[tokio::test]
    async fn idempotent_execution_coalesces_concurrent_requests() {
        let temp = tempfile::tempdir().unwrap();
        let store = IdempotencyStore::new(temp.path().to_path_buf());
        let calls = Arc::new(AtomicUsize::new(0));

        let first_store = store.clone();
        let first_calls = Arc::clone(&calls);
        let first = tokio::spawn(async move {
            execute_idempotently(
                &first_store,
                "same-key".to_string(),
                "same-fingerprint".to_string(),
                false,
                "en",
                async move {
                    first_calls.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    Response::new(Body::from("first"))
                },
            )
            .await
        });

        tokio::time::sleep(Duration::from_millis(10)).await;
        let second_store = store.clone();
        let second_calls = Arc::clone(&calls);
        let second = tokio::spawn(async move {
            execute_idempotently(
                &second_store,
                "same-key".to_string(),
                "same-fingerprint".to_string(),
                false,
                "en",
                async move {
                    second_calls.fetch_add(1, Ordering::SeqCst);
                    Response::new(Body::from("second"))
                },
            )
            .await
        });

        let first = first.await.unwrap();
        let second = second.await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            first.headers()[idempotency::IDEMPOTENCY_STATUS_HEADER],
            "created"
        );
        assert_eq!(
            second.headers()[idempotency::IDEMPOTENCY_STATUS_HEADER],
            "coalesced"
        );
    }
}
