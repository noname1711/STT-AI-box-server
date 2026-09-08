
# Report: Parakeet First-Request Latency, Prewarming Tradeoff, and Deployment Recommendation for `vit-stt`

## 1. Executive Summary

The slower first request for the English Parakeet model in `vit-stt` is expected behavior caused by **lazy model initialization**. The HTTP server starts without loading every recognizer. The first request for `vit_stt_en_v2` creates and caches the Parakeet recognizer, initializes ONNX/sherpa runtime sessions, validates model assets, reads model files, and runs the first actual decode path. Later requests reuse the warmed recognizer, so they are faster.

For the intended deployment, a **24/7 local one-server-to-one-client EchoVox setup that may not be used constantly**, the recommended policy is:

```toml
[server]
warm_start_models = ["vit_stt_vi_v2"]
```

Do **not** prewarm English Parakeet by default. Keep English lazy-loaded unless the deployment needs English frequently, has enough RAM headroom, or must avoid any first-English-request delay during demos or acceptance testing.

The best long-term solution is not “always prewarm everything.” The best solution is **configurable model warmup**, plus an **admin/manual warmup endpoint**, improved timing logs, and optional idle unload for rarely used models.

---

## 2. Current Repo Behavior

### 2.1 `stt-http` starts without preloading recognizers

The HTTP service starts by loading the runtime and config, then binds the server. It logs runtime loading and server readiness, but it does not build the Parakeet or Gipformer recognizers before accepting requests.

The startup path currently does roughly this:

```rust
let mut runtime = SttRuntime::load(cli.workspace_root, cli.config.as_deref())?;

// host/port overrides...

let runtime = Arc::new(runtime);

let app = Router::new()
    .route("/health", get(health))
    .route("/v1/models", get(list_models))
    .route("/v1/audio/transcriptions", post(transcribe))
    .with_state(state);

serve(listener, app).await?;
```

This means the service can become “ready” while the actual STT recognizers are still unloaded.

---

### 2.2 Recognizers are lazy-loaded on first use

The request path calls:

```rust
let recognizer = self.warmed_recognizer(model_id)?;
```

inside `transcribe_decoded()`.

`warmed_recognizer()` first checks the recognizer cache. If the model is not already loaded, it resolves the model, creates `WarmedRecognizerRuntime`, inserts it into the cache, then returns it.

So for each model:

* first request: build recognizer, validate assets, initialize runtime sessions, then infer;
* later requests: reuse cached recognizer, then infer.

This explains why the first Parakeet request is noticeably slower.

---

### 2.3 The English model is Parakeet 0.6B int8

The English production model is configured as:

```json
{
  "id": "vit_stt_en_v2",
  "language": "en",
  "model_dir": "models/stt/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming",
  "model_type": "nemo_transducer",
  "postprocess_mode": "none"
}
```

The asset lock confirms the English Parakeet model uses separate ONNX files:

* `encoder.int8.onnx`
* `decoder.int8.onnx`
* `joiner.int8.onnx`
* `tokens.txt`
* probe WAV

Because this is a 0.6B-class int8 model, recognizer creation is not trivial. The first request has to load and initialize a relatively large model stack.

---

### 2.4 `WarmedRecognizerRuntime` keeps the recognizer alive

`WarmedRecognizerRuntime` stores a `sherpa_onnx::OfflineRecognizer`:

```rust
pub struct WarmedRecognizerRuntime {
    model: ResolvedModelConfig,
    recognizer: OfflineRecognizer,
}
```

The runtime stores warmed recognizers in a cache:

```rust
recognizers: Arc<Mutex<HashMap<String, Arc<Mutex<WarmedRecognizerRuntime>>>>>
```

Therefore, if English Parakeet is prewarmed, it remains resident in memory until the process restarts or explicit unload support is added.

---

## 3. Why the First Request Is Slower

The first request is slower because it includes more work than later requests.

### 3.1 Model resolution and checksum validation

When the model is first resolved, `resolve_model()` validates model assets before caching the resolved model.

Asset validation reads files and computes SHA-256 hashes.

For Parakeet, that means hashing multiple ONNX files. Later requests skip this because the resolved model config is cached.

---

### 3.2 Offline recognizer creation

`WarmedRecognizerRuntime::new()` creates the recognizer and logs:

```rust
"initializing offline recognizer"
"offline recognizer ready"
```

Recognizer construction uses sherpa-onnx config with the encoder, decoder, joiner, tokens, provider, thread count, model type, and decoding settings.

This is the main cold-start cost.

---

### 3.3 ONNX Runtime session initialization and graph optimization

Parakeet is served through sherpa-onnx, which uses ONNX Runtime. ONNX Runtime performs graph optimizations to improve performance. These optimizations can be performed online before inference or offline by saving an optimized graph to disk.

ONNX Runtime enables optimizations by default, including graph simplification, redundant node elimination, fusions, and extended optimizations for CPU/CUDA/ROCm execution providers.

This means first session creation can cost extra time. Later requests benefit from already-created sessions.

---

### 3.4 First inference path warmup

TensorFlow Serving documents a similar production issue: runtime components are often lazily initialized, and the first request after loading a model can have much higher latency. TensorFlow Serving handles this by running representative warmup requests at model load time.

This maps closely to `vit-stt`: the first Parakeet request is effectively acting as the warmup request.

---

### 3.5 VAD is rebuilt per request (persistent overhead)

`WarmedRecognizerRuntime::transcribe_samples()` creates a VAD detector for each request if VAD is enabled:

```rust
segment_with_vad(build_vad_for_model(&self.model)?, samples, sample_rate)?
```

The current VAD builder uses TEN-VAD on CPU.

This is **not just a first-request issue** — it is a persistent per-request overhead. Every transcription request, whether the recognizer is cached or not, pays the cost of constructing a new `VoiceActivityDetector`, loading the TEN-VAD ONNX model, and initializing the VAD session. For short audio clips this cost may be small relative to STT inference, but for a high-throughput deployment it adds up.

**Future improvement**: Cache the VAD detector alongside the recognizer in `WarmedRecognizerRuntime`, or create a separate VAD cache. This would eliminate redundant VAD initialization on every request.

---

## 4. Prewarming: What It Improves and What It Costs

### 4.1 What prewarming improves

Prewarming moves cold-start work from the first user request to server startup or an explicit warmup action.

If English Parakeet is prewarmed:

* the recognizer is created before user traffic;
* ONNX/sherpa sessions are initialized earlier;
* model files are loaded/read earlier;
* the first actual user English transcription becomes faster and more predictable.

This is the same general principle as TensorFlow Serving warmup, where representative requests are used during model loading to reduce lazy-initialization latency.

---

### 4.2 What prewarming costs

Prewarming English Parakeet means the English recognizer remains in memory.

Expected resource impact:

| Resource                      | Impact if English is prewarmed                                           |
| ----------------------------- | ------------------------------------------------------------------------ |
| RAM                           | Persistent increase while `stt-http` is running                          |
| CPU                           | Startup spike; low idle CPU afterward                                    |
| Disk I/O                      | Higher startup read/load activity                                        |
| GPU/VRAM                      | No impact with current CPU provider; possible impact if provider changes |
| Startup time                  | Longer startup                                                           |
| First English request latency | Lower and more predictable                                               |

The repo default provider is CPU.

The configured recognizer thread count is passed into sherpa config.

ONNX Runtime documents that thread spinning can provide faster inference but consume more CPU cycles, resources, and power. This does not prove that idle Parakeet will burn significant CPU in this specific build, but it is a reason to monitor idle CPU after prewarming.

---

### 4.3 Memory caveat

The exact RAM increase must be measured on the target deployment machine. A safe expectation for Parakeet 0.6B int8 is that keeping the recognizer resident may cost **hundreds of MB to over 1 GB** depending on ONNX Runtime session overhead, allocator behavior, OS page cache, and platform.

This should be treated as an estimate until measured. Guessing memory from model name alone is useful for planning, but not enough for final sizing.

---

## 5. How Production Inference Servers Usually Handle This

Production inference systems do not use one universal policy. They usually support a few model lifecycle modes.

### 5.1 NVIDIA Triton

NVIDIA Triton has three model control modes: `NONE`, `EXPLICIT`, and `POLL`. In the default `NONE` mode, Triton attempts to load all models at startup. In `EXPLICIT` mode, Triton loads only the models specified at startup and then requires explicit load/unload actions through model-control APIs.

**Crucially, Triton does not mark a model as READY until warmup completes.** Warmup is declared in `config.pbtxt` via `model_warmup` and runs as part of model initialization, not as a separate API call. This ties warmup into the health/readiness system.

Triton also notes that load/unload memory behavior can be affected by allocator behavior, and memory may not be released back to the OS right away.

### 5.2 TensorFlow Serving

TensorFlow Serving bundles warmup data as a TFRecord file (`assets.extra/tf_serving_warmup_requests`) inside the SavedModel directory. Warmup runs at model load time, before the server marks the model as ready. Maximum 1000 warmup records per model.

### 5.3 Open Inference Protocol (KServe V2)

The industry standard for health/readiness probes is the **Open Inference Protocol** (formerly KServe V2). It defines three REST endpoints:

| Endpoint | Method | Purpose | K8s Mapping |
|---|---|---|---|
| `/v2/health/live` | Server Live | Is the server process alive? | `livenessProbe` |
| `/v2/health/ready` | Server Ready | Are all models ready? | `readinessProbe` |
| `/v2/models/{name}/ready` | Model Ready | Is a specific model ready? | Informational |

This protocol is implemented by NVIDIA Triton, KServe, Seldon Core 2 / MLServer, AMD Inference Server, and partially by vLLM.

**Key design distinction**: Server-level readiness (`/v2/health/ready`) should be used for container probes, not model-specific endpoints. Using model-specific ready endpoints as K8s readiness probes can cause deadlock — the probe waits for model load, but model download won't start until the pod is Ready.

**Recommendation for `vit-stt`**: Do **not** implement the full Open Inference Protocol. It is designed for multi-model serving platforms (Triton, KServe, Seldon) in container/K8s environments. For a local 1:1 STT server, the protocol adds unnecessary complexity (3 new versioned endpoints, versioning layer, K8s-specific semantics).

Instead, **cherry-pick the most valuable OIP patterns**:

1. **Readiness-gating** — Don't report `/health` as ready until warmup completes. Currently `/health` returns `{"status":"ok"}` immediately, even before models are warmed. Add a `ready` field: `{"status":"ok","ready":false}` during warmup, `{"status":"ok","ready":true}` after.
2. **Per-model ready concept** — Expose through the existing `GET /admin/models/status` endpoint (already in the plan), not through a new `/v2/models/{name}/ready` route.
3. **Liveness vs readiness distinction** — Keep `/health` as liveness (process alive). The `ready` field signals readiness.

### 5.4 ONNX Runtime patterns

ONNX Runtime itself does not have a server-level warmup endpoint, but the community has well-established patterns:

* **InferenceSession as singleton**: Never create one per request. Session creation deserializes the model, resolves kernels, applies graph optimizations, allocates memory, and compiles execution plans.
* **Warmup via dummy inference**: After creating the session, run 1–2 dummy inferences to trigger final JIT compilation and cache population.
* **Offline optimization**: Serialize the optimized model to disk via `optimized_model_filepath` in `SessionOptions` to reduce subsequent session creation time.
* **Documented latency pattern**: Users report 968ms → 30ms → 11ms across the first 3 requests, confirming warmup necessity.

### 5.5 Mapping to `vit-stt`

| Production pattern         | `vit-stt` equivalent              |
| -------------------------- | --------------------------------- |
| Load all models at startup | Prewarm Vietnamese and English    |
| Explicit selected preload  | Prewarm only configured models    |
| Lazy load on first request | Current behavior                  |
| Manual model load          | Admin warmup endpoint             |
| Model unload               | Future idle eviction/admin unload |
| Readiness gating           | Don't report ready until warmup completes |

That last point matters if `vit-stt` later adds idle unload: unloading a recognizer may reduce internal memory use, but the operating system may not immediately show all memory returned.

---

## 6. Deployment Context

The target deployment is:

* local server,
* one server to one EchoVox client,
* 24/7 uptime,
* not necessarily used often,
* likely Vietnamese-first,
* English available but not necessarily common.

This changes the tradeoff.

For a busy public API, prewarming every supported model may make sense because any user can hit any model at any time. For a local 1:1 server, the first English request after restart affects only that local client and likely happens rarely.

Because the server is 24/7, the English cold start happens only after:

* service restart,
* machine reboot,
* update,
* crash,
* first-ever English use after deployment,
* explicit model eviction if added later.

So the key question is:

> Is avoiding one occasional slow English request worth keeping English Parakeet resident in RAM all day?

For this deployment, the answer is usually **no**.

---

## 7. Recommendation

### 7.1 Recommended default

Use this default:

```toml
[server]
warm_start_models = ["vit_stt_vi_v2"]
```

Keep English lazy:

```toml
[server]
lazy_models = ["vit_stt_en_v2"]
```

This gives the best balance:

* Vietnamese, the likely main path, is fast from the first user request.
* English remains available.
* English does not consume persistent memory unless actually used.
* The server stays lighter during long idle periods.
* The deployment avoids paying permanent RAM cost for rare English usage.

---

### 7.2 When to prewarm English too

Use this only when English is expected to be used often or must be demo-ready:

```toml
[server]
warm_start_models = ["vit_stt_vi_v2", "vit_stt_en_v2"]
```

Recommended for:

* demos,
* customer acceptance tests,
* deployments where English meetings are common,
* machines with plenty of RAM headroom,
* users who complain about the first English request delay.

Not recommended as the default for low-resource 24/7 local servers.

---

### 7.3 Manual warmup option

Add an admin warmup endpoint:

```text
POST /admin/warmup
```

Example request:

```json
{
  "models": ["vit_stt_en_v2"],
  "run_probe": true
}
```

Example response:

```json
{
  "warmed": [
    {
      "model": "vit_stt_en_v2",
      "already_loaded": false,
      "elapsed_ms": 1840
    }
  ]
}
```

This lets EchoVox or an operator warm English only when needed, for example:

* before a demo,
* before a known English meeting,
* when the user switches the STT language to English,
* during deployment checks.

This is better than forcing English Parakeet to sit in RAM 24/7.

---

### 7.4 Add model status endpoint

Add:

```text
GET /admin/models/status
```

Example response:

```json
{
  "models": [
    {
      "id": "vit_stt_vi_v2",
      "resolved": true,
      "recognizer_loaded": true,
      "warm_start": true,
      "last_used_at": "2026-07-06T14:32:10+07:00"
    },
    {
      "id": "vit_stt_en_v2",
      "resolved": false,
      "recognizer_loaded": false,
      "warm_start": false,
      "last_used_at": null
    }
  ]
}
```

This makes model lifecycle visible instead of guessing from latency.

---

## 8. Proposed Implementation

### 8.1 Add config fields

Extend `ServerConfig`:

```rust
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

    #[serde(default)]
    pub warm_start_models: Vec<String>,
}
```

Then in `runtime.toml`:

```toml
[server]
host = "0.0.0.0"
port = 8080
max_upload_mb = 256
max_audio_seconds = 14400
owned_by = "vit-stt"
warm_start_models = ["vit_stt_vi_v2"]
```

---

### 8.2 Add warm-start function

Add a function like:

```rust
fn warm_start_models(runtime: &SttRuntime) -> anyhow::Result<()> {
    for model_id in &runtime.runtime_config.server.warm_start_models {
        let started = std::time::Instant::now();

        let model = runtime.resolve_model(model_id)?;

        runtime.warmed_recognizer(model_id)?;

        if let Some(path) = model.startup_probe_wav_path.as_deref() {
            let _ = runtime.transcribe_path(model_id, path)?;
        }

        tracing::info!(
            model_id = model_id.as_str(),
            elapsed_ms = started.elapsed().as_millis(),
            "warm-start completed"
        );
    }

    Ok(())
}
```

Call this after config overrides but before `server ready`:

```rust
let mut runtime = SttRuntime::load(cli.workspace_root, cli.config.as_deref())?;

// apply host/port overrides...

warm_start_models(&runtime)?;

let runtime = Arc::new(runtime);
```

The probe decode should be optional but enabled by default for warm-started models. TensorFlow Serving recommends representative warmup requests, not merely loading the model object.

---

### 8.3 Avoid duplicate first-load race

Current `warmed_recognizer()` in `runtime.rs:191-209` has a subtle race window:

```rust
// Step 1: Acquire lock, check cache, release lock
if let Some(recognizer) = self.recognizers.lock()...get(model_id).cloned() {
    return Ok(recognizer);  // cache hit — fast path
}
// Lock is dropped here

// Step 2: Build recognizer (no lock held — other threads can enter)
let model = self.resolve_model(model_id)?;
let recognizer = Arc::new(Mutex::new(WarmedRecognizerRuntime::new(model)?));

// Step 3: Reacquire lock, insert
self.recognizers.lock()...insert(model_id.to_string(), Arc::clone(&recognizer));
```

Between steps 1 and 3, if two requests for the same uncached model arrive concurrently:

1. Request A checks cache → miss → drops lock → starts building recognizer
2. Request B checks cache → miss → drops lock → starts building recognizer
3. Both build the same recognizer (wasted CPU/RAM)
4. Both insert into cache (last writer wins, first writer's recognizer is orphaned)

For a one-client deployment this is unlikely, but it is still worth fixing.

**Option A — Hold lock during construction (simple, blocks all lookups)**:

```rust
pub fn warmed_recognizer(&self, model_id: &str) -> Result<Arc<Mutex<WarmedRecognizerRuntime>>> {
    let mut recognizers = self
        .recognizers
        .lock()
        .map_err(|_| anyhow!("warmed recognizer cache mutex poisoned"))?;

    if let Some(recognizer) = recognizers.get(model_id).cloned() {
        return Ok(recognizer);
    }

    let model = self.resolve_model(model_id)?;
    let recognizer = Arc::new(Mutex::new(WarmedRecognizerRuntime::new(model)?));

    recognizers.insert(model_id.to_string(), Arc::clone(&recognizer));

    Ok(recognizer)
}
```

**Tradeoff**: This blocks the entire recognizer cache while a model loads (potentially seconds for Parakeet). Other models cannot be looked up during construction. Acceptable for a small deployment with 2–3 models.

**Option B — Per-model initialization lock (cleaner, more complex)**:

Use a separate `HashMap<String, Arc<Mutex<()>>>` for per-model load locks. Only hold the global cache lock for the lookup; hold the per-model lock during construction. This allows concurrent loads of different models while preventing duplicate loads of the same model.

Recommendation: **Option A for initial implementation**, upgrade to Option B if concurrent model loading becomes a real need.

---

### 8.4 Add manual warmup endpoint

Pseudo-flow:

```rust
async fn warmup(
    State(state): State<AppState>,
    Json(req): Json<WarmupRequest>,
) -> Response {
    for model_id in req.models {
        let runtime = Arc::clone(&state.runtime);

        let result = task::spawn_blocking(move || {
            let started = Instant::now();

            let model = runtime.resolve_model(&model_id)?;
            runtime.warmed_recognizer(&model_id)?;

            if req.run_probe {
                if let Some(path) = model.startup_probe_wav_path.as_deref() {
                    let _ = runtime.transcribe_path(&model_id, path)?;
                }
            }

            Ok(WarmupResult {
                model: model_id,
                elapsed_ms: started.elapsed().as_millis(),
            })
        })
        .await;
    }
}
```

Protect it as an admin endpoint or only bind it to localhost. Since this is a local server, simplest option:

* only expose admin endpoints on `127.0.0.1`, or
* require an admin token.

---

### 8.5 Add optional idle unload later

Future config:

```toml
[server]
never_unload_models = ["vit_stt_vi_v2"]
idle_unload_after_seconds = 7200
```

Behavior:

| Model      | Policy                                            |
| ---------- | ------------------------------------------------- |
| Vietnamese | Warm at startup, never unload                     |
| English    | Lazy-load on first use, unload after 2 hours idle |
| CAPU       | Separate policy, probably tied to Vietnamese      |

However, idle unload should be a second-phase feature. It is more complex, and memory may not immediately return to the OS due to allocator behavior. NVIDIA Triton documents similar memory behavior when loading and unloading models.

---

### 8.6 Error handling during warmup

The warm-start function must handle model load failures gracefully. Two strategies:

**Strategy A — Fail fast (crash on startup)**:
If a configured warm-start model fails to load, the server refuses to start. This prevents a broken deployment from silently running without the expected model.

```rust
fn warm_start_models(runtime: &SttRuntime) -> anyhow::Result<()> {
    for model_id in &runtime.runtime_config.server.warm_start_models {
        let started = std::time::Instant::now();
        runtime.resolve_model(model_id)?;
        runtime.warmed_recognizer(model_id)?;
        // ... probe, log ...
    }
    Ok(())
}
```

**Strategy B — Log and continue**:
If a warm-start model fails to load, log a warning but let the server start. The model will be retried on first request.

```rust
fn warm_start_models(runtime: &SttRuntime) {
    for model_id in &runtime.runtime_config.server.warm_start_models {
        match warm_one(runtime, model_id) {
            Ok(elapsed) => tracing::info!(model_id, elapsed_ms = elapsed, "warm-start completed"),
            Err(err) => tracing::warn!(model_id, error = %err, "warm-start failed; will retry on first request"),
        }
    }
}
```

**Recommendation**: Strategy A for initial deployment. A server that starts without its expected model is a broken deployment. If the team later needs partial startup, add a `warm_start_optional` config field.

---

### 8.7 Warmup timeout

A stuck model load could block server startup indefinitely. Add a per-model timeout:

```rust
fn warm_one(runtime: &SttRuntime, model_id: &str, timeout: Duration) -> anyhow::Result<()> {
    let started = Instant::now();
    // ... resolve, load, probe ...
    if started.elapsed() > timeout {
        anyhow::bail!("warmup for {} exceeded timeout {:?}", model_id, timeout);
    }
    Ok(())
}
```

Config:

```toml
[server]
warm_start_models = ["vit_stt_vi_v2"]
warm_start_timeout_seconds = 60
```

Default: 60 seconds per model. Parakeet 0.6B int8 should load in under 10 seconds on modern hardware; 60s provides headroom.

---

### 8.8 Concurrent warmup

If `warm_start_models = ["vit_stt_vi_v2", "vit_stt_en_v2"]`, should they load sequentially or in parallel?

**Sequential** (simpler, lower peak memory):
```rust
for model_id in &config.warm_start_models {
    warm_one(runtime, model_id)?;
}
```

**Parallel** (faster startup, higher peak memory):
```rust
let handles: Vec<_> = config.warm_start_models.iter().map(|id| {
    let runtime = runtime.clone();
    let id = id.clone();
    std::thread::spawn(move || warm_one(&runtime, &id))
}).collect();
for h in handles { h.join()??; }
```

**Recommendation**: Sequential for initial implementation. Two models loading simultaneously could spike memory to 2x peak. With only 2 models, sequential startup adds ~5–10s which is acceptable.

---

### 8.9 Existing probe infrastructure

`stt-core` already has a `run_probe()` method in `runtime.rs:211-231`:

```rust
pub fn run_probe(&self, model_id: Option<&str>) -> Result<ProbeResult> {
    let model_id = model_id.unwrap_or(self.default_model_id()?);
    let model = self.resolve_model(model_id)?;
    let runtime = RecognizerRuntime::new(model.clone());
    let result = runtime.transcribe_probe()?;
    let expected = model.startup_probe_expected_text.clone();
    let actual = result.text.trim().to_string();
    let matches_expected = expected.as_deref().map(|text| text == actual).unwrap_or(true);
    Ok(ProbeResult { model_id, expected, actual, matches_expected, probe_path })
}
```

Note: `run_probe()` creates a **new** `RecognizerRuntime` each time (not warmed). This is correct for probe testing (isolates the probe from cached state), but the warmup path should use `warmed_recognizer()` to also populate the cache.

The admin warmup endpoint should reuse `run_probe()` for probe validation, or implement a warmed variant that uses the cached recognizer.

---

## 9. Recommended Deployment Modes

### Mode A: Lowest resource mode

```toml
[server]
warm_start_models = []
```

Use when:

* RAM is very limited,
* startup must be very fast,
* first request latency is acceptable.

Downside:

* first Vietnamese request is slower;
* first English request is slower.

Not recommended for polished customer deployment.

---

### Mode B: Recommended EchoVox 1:1 local deployment

```toml
[server]
warm_start_models = ["vit_stt_vi_v2"]
```

Use when:

* Vietnamese is primary,
* English is occasional,
* server runs 24/7,
* low idle resource use matters,
* user experience should be good for the common path.

This is the recommended default.

---

### Mode C: Demo / acceptance mode

```toml
[server]
warm_start_models = ["vit_stt_vi_v2", "vit_stt_en_v2"]
```

Use when:

* both Vietnamese and English will be tested,
* there is enough RAM,
* first-English latency would look bad,
* deployment is for demo or acceptance testing.

Downside:

* higher persistent memory use;
* slower service startup.

---

### Mode D: Adaptive mode

```toml
[server]
warm_start_models = ["vit_stt_vi_v2"]
never_unload_models = ["vit_stt_vi_v2"]
idle_unload_after_seconds = 7200
```

Use later if:

* English is used sometimes but not daily,
* memory is constrained,
* the team wants automatic cleanup.

This is a good future improvement, not the first thing to implement.

---

## 10. Measurement Plan

The team should measure on the real target server before finalizing default settings.

### 10.1 macOS measurement

Before warmup:

```bash
ps -o pid,rss,vsz,pcpu,comm -p $(pgrep -f stt-http)
```

Warm English:

```bash
curl -s \
  -F model=vit_stt_en_v2 \
  -F response_format=json \
  -F file=@models/stt/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/test_wavs/0.wav \
  http://127.0.0.1:8080/v1/audio/transcriptions >/dev/null
```

After warmup:

```bash
ps -o pid,rss,vsz,pcpu,comm -p $(pgrep -f stt-http)
```

Live monitor:

```bash
top -pid $(pgrep -f stt-http)
```

---

### 10.2 Windows PowerShell measurement

```powershell
Get-Process stt-http | Select-Object Id,CPU,WorkingSet64,PrivateMemorySize64
```

Run before and after English warmup.

---

### 10.3 Request latency test

Run the same request three times:

```bash
for i in 1 2 3; do
  echo "Run $i"
  time curl -s \
    -F model=vit_stt_en_v2 \
    -F response_format=json \
    -F file=@models/stt/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/test_wavs/0.wav \
    http://127.0.0.1:8080/v1/audio/transcriptions >/dev/null
done
```

Expected result:

| Run   | Expected behavior          |
| ----- | -------------------------- |
| Run 1 | Slower, includes cold load |
| Run 2 | Faster                     |
| Run 3 | Similar to Run 2           |

If prewarm works, Run 1 after server startup should already behave like Run 2/3.

---

### 10.4 Metrics to record

| Metric                  | Before warmup | After VI warmup | After EN warmup |
| ----------------------- | ------------: | --------------: | --------------: |
| RSS / Working Set       |               |                 |                 |
| Private memory          |               |                 |                 |
| Idle CPU                |               |                 |                 |
| First request total ms  |               |                 |                 |
| Second request total ms |               |                 |                 |
| Recognizer init ms      |               |                 |                 |
| Audio decode ms         |               |                 |                 |
| VAD ms                  |               |                 |                 |
| Inference ms            |               |                 |                 |
| Postprocess ms          |               |                 |                 |

---

## 11. Logging Improvements

The current HTTP logs already include request accepted, audio validation, inference completed, postprocess, and total completion timing.

However, the logs should be expanded to separate cold-start costs from actual inference.

Recommended fields:

```text
recognizer_cache_hit=true/false
model_resolve_ms=...
asset_validation_ms=...
recognizer_init_ms=...
probe_decode_ms=...
audio_decode_ms=...
vad_ms=...
stt_decode_ms=...
postprocess_ms=...
total_ms=...
```

Important detail: HTTP’s `inference_elapsed_ms` starts before `runtime.transcribe_decoded()`, so on the first request it includes lazy recognizer creation.

But the result’s `processing_time` starts inside `transcribe_samples()`, after the recognizer has been acquired.

Without explicit `recognizer_cache_hit` and `recognizer_init_ms`, benchmark results can be confusing.

---

## 12. Risk Assessment

### Risk 1: Prewarming English increases idle memory

If English is prewarmed, its recognizer remains cached. This increases persistent memory usage while the server runs.

Mitigation:

* do not prewarm English by default;
* add manual warmup;
* measure actual RSS/working set;
* add idle unload later if necessary.

---

### Risk 2: First English request remains slower

If English is lazy-loaded, the first English request after restart will still be slower.

Mitigation:

* document this behavior;
* warm English before demos;
* add admin warmup endpoint;
* prewarm English only for deployments that need it.

---

### Risk 3: Duplicate recognizer initialization under concurrent first requests

Two first-time requests for the same model could trigger duplicate loading.

Mitigation:

* add per-model load lock;
* or hold recognizer cache lock during initial construction for this small deployment.

---

### Risk 4: Idle unload may not fully return memory to OS

Even if unload is implemented, allocator behavior may prevent all memory from being visibly returned to the OS immediately. NVIDIA Triton documents similar behavior for load/unload workflows.

Mitigation:

* treat idle unload as best-effort;
* measure on target OS;
* avoid relying on unload as the only memory-control mechanism.

---

## 13. Final Recommendation

For the current EchoVox local 1:1 deployment:

```toml
[server]
warm_start_models = ["vit_stt_vi_v2"]
```

Do not prewarm English by default.

Add these features in order:

1. **Configurable warm-start list**

   * Default: `["vit_stt_vi_v2"]`
   * Optional demo config: `["vit_stt_vi_v2", "vit_stt_en_v2"]`

2. **Manual/admin warmup endpoint**

   * Allows English to be warmed before demos or known English meetings.

3. **Model status endpoint**

   * Shows whether each model is loaded, warm-started, and recently used.

4. **Better timing logs**

   * Separate model load, recognizer init, audio decode, VAD, STT decode, and postprocess.

5. **Duplicate load guard**

   * Prevent two first requests from loading the same model simultaneously.

6. **Readiness-gating on `/health`**

   * Add `ready` field to `/health` response: `false` during warmup, `true` after.
   * Cherry-picked from Open Inference Protocol. Do not implement the full `/v2/` protocol — it is designed for K8s multi-model serving, not local 1:1 deployments.

7. **Optional idle unload**

   * Future improvement for memory-constrained servers.

This policy gives the best balance for a 24/7 local server: the common Vietnamese path is fast, English remains available, and the server does not waste memory on an idle English recognizer all day.

---

## 14. EchoVox UI Integration Plan

### 14.1 Should prewarm/probe be in the EchoVox UI?

**Yes — as a diagnostic/testing tool in Advanced Settings, not as a primary workflow.**

Rationale:

* Desktop apps like Ollama, LM Studio, and ModxAI Studio all expose model status indicators and load/unload controls.
* The user occasionally needs to verify model health, especially before demos or after updates.
* The existing `run_probe()` in `stt-core` already supports this — it just needs an API endpoint and UI.
* Prewarming should remain **manual/on-demand** in the UI. The report's recommendation to prewarm only Vietnamese by default remains correct. The UI gives the operator a way to warm English before demos without changing the default config.

### 14.2 Where in EchoVox

**Not in CommonSettingsPanel** — that panel handles primary STT model selection and runtime device management.

**In AdvancedSettingsPanel** — add a "STT Model Status" section with:

* Model list with loaded/unloaded status indicators (green dot = loaded, gray = unloaded)
* "Warm" button per model (calls `POST /admin/warmup`)
* "Test" button per model (runs probe WAV, shows expected vs actual text with match/mismatch)
* Elapsed time display after warmup/test

### 14.3 UI sketch

```
┌─────────────────────────────────────────────────────────────┐
│ STT Model Status                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  vit_stt_vi_v2   ● Đã tải     [Làm nóng]  [Kiểm tra]      │
│  vit_stt_en_v2   ○ Chưa tải   [Làm nóng]  [Kiểm tra]      │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│  Lần kiểm tra gần nhất: vit_stt_vi_v2                       │
│  Kỳ vọng: "Định nghĩa thế nào là ăn mặc đẹp?"             │
│  Thực tế:  "Định nghĩa thế nào là ăn mặc đẹp?"             │
│  Kết quả:  ✓ Khớp (42ms)                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 14.4 Implementation requirements

| Layer | What | Where |
|---|---|---|
| vit-stt backend | `POST /admin/warmup` endpoint | `stt-http` crate |
| vit-stt backend | `GET /admin/models/status` endpoint | `stt-http` crate |
| EchoVox backend | Proxy to vit-stt admin endpoints (localhost-only) | `src/web/handlers/` |
| EchoVox frontend | STT Model Status panel | `frontend/src/app/features/settings/AdvancedSettingsPanel.tsx` |

### 14.5 API contract for EchoVox

**Warm a model**:
```
POST /admin/warmup
Content-Type: application/json

{
  "models": ["vit_stt_en_v2"],
  "run_probe": true
}

→ 200
{
  "warmed": [
    {
      "model": "vit_stt_en_v2",
      "already_loaded": false,
      "elapsed_ms": 1840,
      "probe": {
        "expected": "...",
        "actual": "...",
        "matches": true
      }
    }
  ]
}
```

**Get model status**:
```
GET /admin/models/status

→ 200
{
  "models": [
    {
      "id": "vit_stt_vi_v2",
      "resolved": true,
      "recognizer_loaded": true,
      "warm_start": true,
      "last_used_at": "2026-07-06T14:32:10+07:00"
    }
  ]
}
```

### 14.6 Security

Since this is a local 1:1 deployment:

* Admin endpoints should only be exposed on `127.0.0.1` (not `0.0.0.0`).
* Alternatively, require an admin token header for admin routes.
* EchoVox backend proxies admin requests to vit-stt; the frontend never calls vit-stt directly.

---

## 15. Detailed Phased Implementation Plan

### Phase 1: Configurable warm-start (foundation)

**Goal**: Server prewarms configured models at startup. `/health` reports readiness state.

**Changes**:

| File | Change |
|---|---|
| `stt-core/src/config.rs` | Add `warm_start_models: Vec<String>` and `warm_start_timeout_seconds: u64` to `ServerConfig` |
| `stt-core/src/runtime.rs` | Add `warm_start()` method that iterates `warm_start_models`, resolves, loads recognizer, runs probe, logs timing; add `is_ready: Arc<AtomicBool>` field to `SttRuntime` |
| `stt-http/src/main.rs` | Call `runtime.warm_start()?` after config overrides, before `server ready` log; set `is_ready` to `true` after warmup completes |
| `stt-http/src/main.rs` | Update `/health` handler to include `ready` field: `{"status":"ok","ready":false}` during warmup, `{"status":"ok","ready":true}` after |
| `config/runtime.toml` | Add `warm_start_models = ["vit_stt_vi_v2"]` |

**Validation**:
* Server starts and logs warm-start timing for Vietnamese.
* `/health` returns `{"status":"ok","ready":true}` after warmup completes.
* First Vietnamese transcription request is fast (no cold-start delay).
* English first request is still slow (lazy-loaded).
* `cargo test` passes.

**Exit criteria**: Vietnamese model is warmed at startup. English remains lazy. `/health` reports readiness. Timing logged.

---

### Phase 2: Race condition fix

**Goal**: Prevent duplicate recognizer construction under concurrent requests.

**Changes**:

| File | Change |
|---|---|
| `stt-core/src/runtime.rs` | Rewrite `warmed_recognizer()` to hold the cache lock during construction (Option A) |

**Validation**:
* Single-threaded behavior unchanged.
* Code review confirms no deadlock risk (single lock, no nested locks).

**Exit criteria**: `warmed_recognizer()` is race-free.

---

### Phase 3: Admin endpoints

**Goal**: Expose warmup and model status via HTTP.

**Changes**:

| File | Change |
|---|---|
| `stt-http/src/main.rs` | Add `POST /admin/warmup` and `GET /admin/models/status` routes |
| `stt-http/src/main.rs` | Bind admin routes only to `127.0.0.1` (separate listener or route guard) |
| `stt-core/src/runtime.rs` | Add `model_status()` method returning loaded/resolved/last_used info |

**Validation**:
* `curl -X POST http://127.0.0.1:8080/admin/warmup -d '{"models":["vit_stt_en_v2"],"run_probe":true}'` returns timing and probe result.
* `curl http://127.0.0.1:8080/admin/models/status` returns model list with loaded state.
* Admin endpoints are not reachable from external IPs.

**Exit criteria**: Admin warmup and status endpoints work on localhost.

---

### Phase 4: Logging improvements

**Goal**: Separate cold-start costs from inference in logs.

**Changes**:

| File | Change |
|---|---|
| `stt-core/src/runtime.rs` | Add `recognizer_cache_hit` field to transcription log; log `recognizer_init_ms` on cold load |
| `stt-http/src/main.rs` | Add timing fields to request logs: `model_resolve_ms`, `asset_validation_ms`, `recognizer_init_ms` |

**Validation**:
* First request logs show `recognizer_cache_hit=false` and `recognizer_init_ms`.
* Second request logs show `recognizer_cache_hit=true` and no `recognizer_init_ms`.

**Exit criteria**: Cold-start vs warm inference is distinguishable in logs.

---

### Phase 5: EchoVox UI integration

**Goal**: Model status and warmup controls in EchoVox Advanced Settings.

**Changes**:

| File | Change |
|---|---|
| EchoVox backend (`src/web/handlers/`) | Add proxy routes for `/api/app/stt/admin/warmup` and `/api/app/stt/admin/models/status` |
| EchoVox frontend (`AdvancedSettingsPanel.tsx`) | Add STT Model Status panel with warm/test buttons |
| EchoVox frontend (`api.ts`) | Add `warmSttModel()` and `getSttModelStatus()` API functions |

**Validation**:
* Advanced Settings shows model list with loaded/unloaded indicators.
* "Warm" button triggers warmup and shows elapsed time.
* "Test" button runs probe and shows expected vs actual text.

**Exit criteria**: Operator can warm and test STT models from EchoVox UI.

---

### Phase 6: VAD caching (future optimization)

**Goal**: Eliminate per-request VAD reconstruction overhead.

**Changes**:

| File | Change |
|---|---|
| `stt-core/src/recognizer.rs` | Cache `VoiceActivityDetector` in `WarmedRecognizerRuntime` or add a separate VAD cache |

**Validation**:
* Per-request VAD init time is eliminated for cached models.
* VAD behavior unchanged (same segments, same output).

**Exit criteria**: VAD detector is reused across requests for the same model.

---

### Phase 7: Idle unload (future, deferred)

**Goal**: Automatically unload models after idle period.

**Changes**:

| File | Change |
|---|---|
| `stt-core/src/config.rs` | Add `never_unload_models` and `idle_unload_after_seconds` to `ServerConfig` |
| `stt-core/src/runtime.rs` | Add last-used tracking, idle check timer, unload method |
| `stt-http/src/main.rs` | Run idle check on a timer (e.g., every 5 minutes) |

**Validation**:
* English model unloads after 2 hours idle.
* Vietnamese model never unloads.
* Memory decreases after unload (measure RSS).

**Exit criteria**: Idle models are automatically unloaded. Memory returns to OS (best-effort).

---

### Phase 8: Documentation cleanup, tests, and validation

**Goal**: Ensure all documentation is accurate, tests cover new functionality, and validation scripts are in place.

**Changes**:

| File | Change |
|---|---|
| `vit-stt/AGENTS.md` | Update "Phase Order" to include warmup feature; add warmup-related validation commands |
| `vit-stt/README.md` (if exists) | Document `warm_start_models` config, admin endpoints, deployment modes |
| `config/runtime.toml` | Add commented examples for `warm_start_models`, `warm_start_timeout_seconds` |
| `config/models.local.json` | Ensure `startup_probe_wav_path` and `startup_probe_expected_text` are set for all models |
| `stt-core/src/config.rs` | Add `#[serde(default)]` annotations for new fields; add doc comments |
| `stt-core/src/runtime.rs` | Add doc comments to `warm_start()`, `model_status()`, `ProbeResult` |
| `stt-http/src/main.rs` | Add doc comments to admin endpoint handlers |
| `stt-core/tests/warm_start.rs` (new) | Unit tests for warm-start logic: success, failure, timeout, empty list |
| `stt-core/tests/probe.rs` (new) | Unit tests for probe validation: match, mismatch, missing probe WAV |
| `stt-http/tests/admin_endpoints.rs` (new) | Integration tests for `/admin/warmup` and `/admin/models/status` |
| `scripts/validate_warmup.sh` (new) | Shell script to test warmup workflow end-to-end |

**Documentation updates**:

1. **AGENTS.md** — Add to "Validation Expectations":
   ```markdown
   For warmup work:
   - verify `warm_start_models` config is parsed correctly
   - verify warm-start logs appear on server startup
   - verify `POST /admin/warmup` returns timing and probe result
   - verify `GET /admin/models/status` returns loaded state
   - verify admin endpoints are localhost-only
   ```

2. **Config examples** — Add to `runtime.toml`:
   ```toml
   [server]
   # Models to prewarm at startup. Vietnamese is prewarmed by default.
   # English is lazy-loaded unless added here.
   warm_start_models = ["vit_stt_vi_v2"]
   # Timeout per model warmup (seconds). Default: 60.
   warm_start_timeout_seconds = 60
   ```

3. **Model registry** — Ensure all models have probe WAV paths:
   ```json
   {
     "id": "vit_stt_en_v2",
     "startup_probe_wav_path": "models/stt/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/test_wavs/0.wav",
     "startup_probe_expected_text": "..."
   }
   ```

**Test coverage**:

| Test | What it validates |
|---|---|
| `warm_start_empty_list` | Server starts normally with no warm-start models |
| `warm_start_single_model` | Vietnamese model is warmed and cached |
| `warm_start_invalid_model` | Server fails fast on unknown model ID |
| `warm_start_timeout` | Warmup fails if model load exceeds timeout |
| `probe_match` | Probe returns `matches_expected=true` for correct output |
| `probe_mismatch` | Probe returns `matches_expected=false` for wrong output |
| `probe_missing_wav` | Probe fails gracefully if probe WAV is missing |
| `admin_warmup_endpoint` | `POST /admin/warmup` returns 200 with timing |
| `admin_status_endpoint` | `GET /admin/models/status` returns model list |
| `admin_localhost_only` | Admin endpoints reject requests from non-localhost |
| `race_condition_prevented` | Two concurrent requests for same model don't duplicate load |
| `health_ready_after_warmup` | `/health` returns `ready:true` after warmup completes |
| `health_not_ready_during_warmup` | `/health` returns `ready:false` before warmup completes |

**Validation script** (`scripts/validate_warmup.sh`):

```bash
#!/bin/bash
set -euo pipefail

echo "=== Warmup Validation ==="

# 1. Start server with warm_start_models
echo "Starting stt-http..."
cargo run --bin stt-http -- --workspace-root . &
SERVER_PID=$!
sleep 5

# 2. Check server is ready
echo "Checking /health (should be ready after warmup)..."
HEALTH=$(curl -sf http://127.0.0.1:8080/health | jq -r '.ready')
if [ "$HEALTH" != "true" ]; then
  echo "FAIL: /health ready should be true after warmup"
  exit 1
fi
echo "OK: /health ready=true"

# 3. Check model status (should show Vietnamese loaded)
echo "Checking /admin/models/status..."
curl -sf http://127.0.0.1:8080/admin/models/status | jq .

# 4. Warm English
echo "Warming English model..."
curl -sf -X POST http://127.0.0.1:8080/admin/warmup \
  -H 'Content-Type: application/json' \
  -d '{"models":["vit_stt_en_v2"],"run_probe":true}' | jq .

# 5. Check model status again (should show both loaded)
echo "Checking /admin/models/status after warmup..."
curl -sf http://127.0.0.1:8080/admin/models/status | jq .

# 6. Test transcription (should be fast, no cold start)
echo "Testing Vietnamese transcription..."
time curl -sf \
  -F model=vit_stt_vi_v2 \
  -F response_format=json \
  -F file=@baselines/phase0/vi_probe.wav \
  http://127.0.0.1:8080/v1/audio/transcriptions | jq .

# 7. Test English transcription (should be fast, already warmed)
echo "Testing English transcription..."
time curl -sf \
  -F model=vit_stt_en_v2 \
  -F response_format=json \
  -F file=@models/stt/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/test_wavs/0.wav \
  http://127.0.0.1:8080/v1/audio/transcriptions | jq .

# 8. Cleanup
kill $SERVER_PID
echo "=== Validation complete ==="
```

**Exit criteria**:
* All new unit and integration tests pass.
* `cargo clippy` and `cargo fmt` pass on all changed crates.
* Documentation is accurate and examples work.
* Validation script runs end-to-end successfully.
* No outdated or conflicting documentation remains.

---

## 16. Industry References

| Topic | Source | URL |
|---|---|---|
| TF Serving warmup | Official docs | https://www.tensorflow.org/tfx/serving/saved_model_warmup |
| TF Serving performance | Official guide | https://www.tensorflow.org/tfx/serving/performance |
| Triton model config | NVIDIA docs | https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html |
| Triton restricted APIs | Customization guide | https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/inference_protocols.md |
| Open Inference Protocol | KServe V2 spec | https://github.com/open-inference/open-inference-protocol/blob/main/specification/protocol/inference_rest.md |
| KServe V2 health endpoints | KServe docs | https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol |
| ONNX Runtime patterns | Community blog | https://raymondclanan.com/blog/onnx-runtime-patterns/ |
| ONNX Runtime session docs | Official | https://microsoft-onnxruntime-40.mintlify.app/concepts/sessions |
| ONNX Runtime graph optimizations | Official | https://tomwildenhain-microsoft.github.io/onnxruntime/docs/performance/graph-optimizations.html |
| ONNX Runtime first-request issue | GitHub issue | https://github.com/microsoft/onnxruntime/issues/10131 |
| vLLM readiness probes | Docs | https://github.com/llm-d/llm-d/blob/0ffa2847/docs/readiness-probes.md |
| Vertex AI warmup pattern | Blog | https://oneuptime.com/blog/post/2026-02-17-how-to-implement-model-warm-up-and-traffic-splitting-on-vertex-ai-endpoints/view |
| Seldon readiness deadlock warning | Docs | https://docs.seldon.ai/seldon-core-2/user-guide/v2/rest/health |
| Ollama model manager UI | PR | https://github.com/ollama/ollama/pull/14531 |
| LM Studio model management | DeepWiki | https://deepwiki.com/lmstudio-ai/docs/5.4-model-management-and-configuration |