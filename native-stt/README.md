# vit-stt

`vit-stt` là workspace Rust cho nhận dạng giọng nói offline, tập trung vào
tiếng Việt và tiếng Anh. Repo cung cấp runtime STT tái sử dụng được, xử lý hậu
kỳ CAPU cho tiếng Việt, CLI để kiểm tra/probe/benchmark, và HTTP API tương thích
kiểu OpenAI cho endpoint transcription.

## Trạng thái hiện tại

- Phase 0 baseline lock: hoàn tất
- Phase 1 Rust core spike: hoàn tất
- Phase 2 CAPU parity: hoàn tất qua Python worker riêng
- Phase 3 CLI: hoàn tất
- Phase 4 HTTP service: hoàn tất
- Phase 5 standalone runtime hardening: đang là trọng tâm
- Phase 6 packaging: đã dựng/stage release cho macOS và Windows
- Phase 7 accelerators và advanced features: kế tiếp

## Repo này có gì

- `stt-core`: registry model, asset lock, decode audio, runtime sherpa-onnx,
  postprocessing, cấu hình và lỗi dùng chung
- `stt-capu`: giao diện CAPU và bridge tới CAPU worker
- `stt-http`: HTTP service với `/health`, `/v1/models`,
  `/v1/audio/transcriptions`
- `stt-cli`: lệnh list/probe/transcribe/benchmark/download/doctor/service
- `capu-worker`: Python worker riêng để giữ parity CAPU tiếng Việt

Model CAPU hiện dùng alias `vibert-capu`, trỏ tới
`models/capu/vibert-capu`, lấy từ gói hợp nhất `leakless/vibert-capu`.

## Yêu cầu hệ thống

- Rust toolchain có `cargo` nếu chạy từ source
- `ffmpeg` có trong `PATH`
- `llama-server` từ `llama.cpp`
- Python 3.11-3.13 cho CAPU worker khi chạy từ source hoặc khi release không có
  CAPU worker đóng gói sẵn
- Kết nối mạng để tải model ở lần chuẩn bị đầu tiên, trừ khi dùng offline model pack
- Dung lượng đĩa đủ cho model STT, VAD, CAPU và baseline assets


Trên macOS có thể cài nhanh:

- Cài homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

- Từ homebrew cài các phần mềm
```bash
brew install rustup ffmpeg llama.cpp python@3.12
```

- Setup rust
```bash
rustup-init
```

## Cài đặt và khởi động `llama.cpp` `llama-server`

`llama-server` là dependency bắt buộc cho phần CAPU/LLM experiments. Cách
nhanh nhất trên macOS là cài từ Homebrew:

```bash
brew install llama.cpp
which llama-server
llama-server --help
```

Nếu muốn tự build từ source:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON
cmake --build build -j
sudo cmake --install build
```

Khởi động `llama-server` với cấu hình mặc định của repo:

```bash
llama-server -hf unsloth/gemma-4-E4B-it-GGUF --temp 0.3 --top-p 0.95 --top-k 64 \
  --reasoning off -c 32768 --host 0.0.0.0 --port 8001 --alias vit_small_4b
```

Kiểm tra server đang chạy:

```bash
curl http://127.0.0.1:8001/health
```

Nếu muốn chạy nền trên macOS bằng LaunchDaemon, dùng helper của repo:

```bash
./bin/stt-cli llama-service config --bin-path "$(command -v llama-server)"
sudo ./bin/stt-cli llama-service install
./bin/stt-cli llama-service status
./bin/stt-cli llama-service health
./bin/stt-cli llama-service logs --lines 200
```

## Model mặc định

- `vit_stt_vi_v2`: tiếng Việt, dùng Gipformer và CAPU để viết hoa/thêm dấu câu
- `vit_stt_en_v2`: tiếng Anh, dùng NeMo Parakeet, không dùng CAPU

Registry mặc định chỉ chủ động expose hai model production này. Các file runtime
cần thiết được khóa checksum trong `baselines/phase0/assets.lock.json`.

Model CAPU mặc định là `vibert-capu`, nằm ở `models/capu/vibert-capu`. Không
cần tải thủ công nếu chạy:

```bash
./bin/stt-cli download-models
./bin/stt-cli verify-assets
./bin/stt-cli check-capu
```

## Phân phối model và pre-seed server

Gói release mặc định không chứa model weights.

Khi bàn giao server offline/on-premise, VietInnotech có thể tải model trực tiếp
lên server trước khi bàn giao bằng:

```bash
./scripts/check-server-deps.sh
./scripts/prepare-server-models.sh
```

Hoặc chạy thủ công:

```bash
./bin/stt-cli download-models
./bin/stt-cli verify-assets
./bin/stt-cli check-capu
./bin/stt-cli probe --model vit_stt_vi_v2
./bin/stt-cli probe --model vit_stt_en_v2
```

Vì khách hàng nhận server đã có model files, thư mục triển khai phải có:

```text
LICENSE
THIRD_PARTY_NOTICES.md
MODEL_BOM.md
DEPENDENCY_BOM.md
licenses/
```

Model weights không bị VietInnotech sửa đổi.

Chỉ dùng `VIT_STT_INCLUDE_MODELS=1` khi cố ý tạo release archive có model files:

```bash
VIT_STT_INCLUDE_MODELS=1 ./scripts/package-release /tmp/vit-stt-with-models
```

Chế độ này cần rà soát pháp lý rõ ràng về quyền phân phối lại model.


## Cách 1: Triển khai bằng release đóng gói

Release portable hiện hỗ trợ:

- macOS ARM64: `vit-stt-macos-arm64.tar.gz`
- Windows x86_64: `vit-stt-windows-x86_64.zip`

Linux không phải deployment target chính thức cho khách hàng.

Môi trường phát triển chính: macOS.

Gói release có sẵn binary `stt-cli`, `stt-http`, cấu hình mặc định, baseline đã
khóa và CAPU worker runtime. Model weights không nằm trong archive public để giữ
kích thước gói hợp lý và tránh vấn đề giấy phép phân phối lại.

### macOS hoặc Windows

```bash
tar -xzf vit-stt-<platform>.tar.gz -C /opt/vit-stt
cd /opt/vit-stt

./bin/stt-cli setup
./bin/stt-cli download-models
./bin/stt-cli verify-assets
./bin/stt-cli doctor
./bin/stt-cli check-capu
./bin/stt-cli probe --model vit_stt_vi_v2
./bin/stt-cli probe --model vit_stt_en_v2
```

### Windows PowerShell

```powershell
Expand-Archive .\vit-stt-windows-x86_64.zip -DestinationPath C:\vit-stt
cd C:\vit-stt

.\bin\stt-cli.exe setup
.\bin\stt-cli.exe download-models
.\bin\stt-cli.exe verify-assets
.\bin\stt-cli.exe doctor
.\bin\stt-cli.exe check-capu
.\bin\stt-cli.exe probe --model vit_stt_vi_v2
.\bin\stt-cli.exe probe --model vit_stt_en_v2
```

## Cách 2: Triển khai bằng clone repo

Cách này phù hợp cho máy dev, máy staging, hoặc khi cần tự build binary từ
source.

```bash
git clone https://github.com/VietInnotech/vit-stt.git
cd vit-stt

./scripts/bootstrap-dev
```

`./scripts/bootstrap-dev` sẽ chạy:

```bash
cargo run -p stt-cli -- --workspace-root "$PWD" setup
cargo run -p stt-cli -- --workspace-root "$PWD" download-models
cargo run -p stt-cli -- --workspace-root "$PWD" verify-assets
cargo run -p stt-cli -- --workspace-root "$PWD" check-capu
```

Sau khi bootstrap xong, kiểm tra model:

```bash
cargo run -p stt-cli -- list-models
cargo run -p stt-cli -- probe --model vit_stt_vi_v2
cargo run -p stt-cli -- probe --model vit_stt_en_v2
```

Chạy thử transcription bằng CLI:

```bash
cargo run -p stt-cli -- transcribe-wav baselines/phase0/vi_probe.wav \
  --model vit_stt_vi_v2 \
  --response-format text
```

Build binary release local:

```bash
cargo build --release -p stt-cli -p stt-http
```

Chạy binary vừa build:

```bash
./target/release/stt-cli doctor
./target/release/stt-http --host 0.0.0.0 -p 8080
```

## Chạy HTTP service

Mặc định service bind theo `config/runtime.toml`: host `0.0.0.0`, port `8080`.
Có thể override bằng CLI flag:

```bash
./bin/stt-http --host 0.0.0.0 -p 8080
```

Nếu chạy từ source:

```bash
cargo run -p stt-http -- --host 0.0.0.0 -p 8080
```

Nếu chỉ muốn cho máy local truy cập:

```bash
./bin/stt-http --host 127.0.0.1 -p 8080
```

Cũng có thể dùng biến môi trường:

```bash
VIT_STT_SERVER_HOST=127.0.0.1 VIT_STT_SERVER_PORT=8080 ./bin/stt-http
```

## Kiểm tra HTTP API

Từ terminal khác:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

Transcribe trả JSON:

```bash
curl -F model=vit_stt_vi_v2 \
  -F response_format=json \
  -F file=@baselines/phase0/vi_probe.wav \
  http://127.0.0.1:8080/v1/audio/transcriptions
```

Transcribe trả text:

```bash
curl -F model=vit_stt_vi_v2 \
  -F response_format=text \
  -F file=@baselines/phase0/vi_probe.wav \
  http://127.0.0.1:8080/v1/audio/transcriptions
```

Endpoint transcription nhận multipart form:

- `file`: file audio
- `model`: model id, ví dụ `vit_stt_vi_v2`
- `language`: tùy chọn
- `response_format`: `json`, `text`, hoặc `verbose_json`

Runtime hiện decode audio qua `ffmpeg` và chuẩn hóa về PCM mono 16 kHz.

Client có thể gửi header `Idempotency-Key` (1–255 ký tự ASCII hiển thị) để
retry an toàn. Cùng key và cùng nội dung request sẽ dùng chung request đang
chạy hoặc trả lại kết quả thành công đã lưu; cùng key với nội dung khác trả
HTTP 409 (`code: idempotency_conflict`). Response echo `Idempotency-Key` và
trả `X-Idempotency-Status: created|coalesced|cached|conflict`. Cache thành
công tồn tại 24 giờ trong `.cache/stt-http/idempotency`; có thể đổi thư mục
bằng `VIT_STT_IDEMPOTENCY_CACHE_DIR`.

### Warm-start và readiness

Mặc định, server prewarm model tiếng Việt khi khởi động. Model tiếng Anh
lazy-load khi có request đầu tiên. Cấu hình trong `[server]` của
`config/runtime.toml`:

```toml
[server]
# Prewarm tiếng Việt khi khởi động (mặc định: rỗng)
warm_start_models = ["vit_stt_vi_v2"]
# Timeout mỗi model (giây). Mặc định: 60.
warm_start_timeout_seconds = 60
```

Nếu cần cả tiếng Anh sẵn sàng (demo, acceptance test):

```toml
[server]
warm_start_models = ["vit_stt_vi_v2", "vit_stt_en_v2"]
```

`/health` trả `ready: true` sau khi warm-start hoàn tất:

```json
{"status": "ok", "ready": true}
```

### Admin endpoints

#### POST /admin/warmup

Warm model trên demand. Hữu ích khi cần chuẩn bị tiếng Anh trước demo.

```bash
curl -X POST http://127.0.0.1:8080/admin/warmup \
  -H 'Content-Type: application/json' \
  -d '{"models": ["vit_stt_en_v2"], "run_probe": true}'
```

Response:

```json
{
  "warmed": [
    {
      "model": "vit_stt_en_v2",
      "already_loaded": false,
      "elapsed_ms": 1840,
      "probe": {
        "model_id": "vit_stt_en_v2",
        "expected": "...",
        "actual": "...",
        "matches_expected": true,
        "probe_path": "..."
      }
    }
  ]
}
```

#### GET /admin/models/status

Xem trạng thái loaded/unloaded của từng model.

```bash
curl http://127.0.0.1:8080/admin/models/status
```

Response:

```json
{
  "models": [
    {
      "id": "vit_stt_vi_v2",
      "resolved": true,
      "recognizer_loaded": true,
      "warm_start": true,
      "last_used_at": null
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

### Chế độ triển khai

| Chế độ | `warm_start_models` | Khi nào dùng |
| --- | --- | --- |
| Tiết kiệm RAM | `[]` | RAM hạn chế, startup nhanh |
| **Mặc định EchoVox 1:1** | `["vit_stt_vi_v2"]` | Tiếng Việt chính, tiếng Anh偶尔 |
| Demo/acceptance | `["vit_stt_vi_v2", "vit_stt_en_v2"]` | Cả hai ngôn ngữ cần sẵn sàng |

### Giới hạn upload và thời lượng audio

HTTP layer áp dụng hai giới hạn phía server, độc lập với nhau. Cả hai đều cấu
hình được ở `[server]` trong `config/runtime.toml` và áp dụng cho mọi request
tới `POST /v1/audio/transcriptions`, bất kể client gửi gì.

| Giới hạn | Mặc định | Nơi áp dụng | Khi vượt |
| --- | --- | --- | --- |
| `max_upload_mb` (kích thước multipart body) | 256 MB | axum `DefaultBodyLimit`, trước khi parse | HTTP 413, `code: request_too_large` |
| `max_audio_seconds` (độ dài audio sau khi decode) | 14400 s (4 h) | Sau khi ffmpeg decode, trước inference | HTTP 400, `code: audio_too_long` |

File MP3 cuộc họp 3.5 giờ ở 128 kbps khoảng 200 MB, nên cap 256 MB hiện bao
được trường hợp đó với dư dả cho bitrate cao hơn. File lớn hơn cần hoặc nén
lại / re-encode, hoặc nâng `max_upload_mb` rồi khởi động lại server.

Khi vượt `max_upload_mb`, response trả về JSON error body với giá trị cap hiện
tại trong thông báo, header `content-language: en|vi` dựa trên `Accept-Language`
của client, thay vì trả 400 chung chung kiểu "Error parsing multipart/form-data
request". Log phía server cũng ghi lại `request_id` và `max_upload_mb` đang áp
dụng để khi operator đọc log biết ngay là cap nào vừa bắn và request nào bị
ảnh hưởng.

Ví dụ request vượt cap:

```bash
$ curl -i -F model=vit_stt_vi_v2 -F file=@rat-mp3-300mb.mp3 \
    http://127.0.0.1:8080/v1/audio/transcriptions
HTTP/1.1 413 Payload Too Large
content-language: en
content-type: application/json

{
  "error": {
    "message": "Request body exceeds the configured limit of 256 MB. Reduce the upload size or raise `max_upload_mb` in `config/runtime.toml` and restart the server.",
    "type": "invalid_request_error",
    "code": "request_too_large"
  }
}
```

Các mã lỗi phổ biến khác của endpoint:

- `model_required` (HTTP 400) — thiếu field `model`
- `model_not_found` (HTTP 404) — alias model không tồn tại trong registry
- `file_required` (HTTP 400) — multipart không có field `file`
- `audio_too_long` (HTTP 400) — audio sau khi decode dài hơn `max_audio_seconds`
- `audio_decode_failed` (HTTP 400) — file audio không decode được bằng ffmpeg
- `ffmpeg_not_found` / `ffmpeg_launch_failed` (HTTP 500) — server chưa cài hoặc
  không chạy được ffmpeg; lỗi phía server, không phải lỗi client
- `capu_unavailable` / `capu_timeout` / `capu_failed` (HTTP 500/504) — lỗi khi
  gọi CAPU worker cho model tiếng Việt

## Lệnh CLI thường dùng

```bash
# Liệt kê model production
cargo run -p stt-cli -- list-models

# Tải toàn bộ locked assets còn thiếu
cargo run -p stt-cli -- download-models

# Kiểm tra checksum assets
cargo run -p stt-cli -- verify-assets

# Chẩn đoán runtime
cargo run -p stt-cli -- doctor

# Kiểm tra CAPU worker và fixture output
cargo run -p stt-cli -- check-capu

# Probe model tiếng Việt
cargo run -p stt-cli -- probe --model vit_stt_vi_v2

# Probe model tiếng Anh
cargo run -p stt-cli -- probe --model vit_stt_en_v2

# Benchmark một file audio
cargo run -p stt-cli -- benchmark \
  --model vit_stt_vi_v2 \
  baselines/phase0/vi_probe.wav
```

Với release đã giải nén, thay `cargo run -p stt-cli --` bằng `./bin/stt-cli`.
Trên Windows, dùng `.\bin\stt-cli.exe`.

## Cấu hình

- Runtime config: `config/runtime.toml` (đã ignore git — copy từ `runtime.example.toml`)
- Model registry local: `config/models.local.json` (đã ignore git — copy từ `models.example.json`)
- Registry mẫu: `config/models.example.json`
- Service plist mẫu cho LaunchDaemon trên macOS: `deploy/com.vit-stt.http.plist` và `deploy/com.vit-stt.llama.plist`
- Asset lock: `baselines/phase0/assets.lock.json`

Runtime tự tìm workspace/runtime root theo thứ tự:

1. `--workspace-root`
2. `VIT_STT_RUNTIME_ROOT`
3. thư mục cha/gần binary trong layout portable
4. macOS bundle resources
5. current working directory

Nếu muốn để model ở ngoài thư mục release, đặt:

```bash
VIT_STT_MODELS_DIR=/duong/dan/models ./bin/stt-http
```

## Setup config, đổi alias, nâng cấp model

Sửa model ở hai nơi:

- `config/models.local.json`: alias runtime, đường dẫn model, language,
  postprocess, CAPU mapping, startup probe
- `baselines/phase0/assets.lock.json`: danh sách file cần tải, URL, đường dẫn,
  checksum SHA-256

Đổi alias STT:

```json
{
  "id": "vit_stt_vi_v3",
  "language": "vi",
  "model_dir": "models/stt/gipformer-65M-rnnt",
  "postprocess_mode": "capu",
  "capu_model_id": "vibert-capu"
}
```

Sau khi đổi `id`, client HTTP phải gửi alias mới:

```bash
curl -F model=vit_stt_vi_v3 \
  -F response_format=text \
  -F file=@baselines/phase0/vi_probe.wav \
  http://127.0.0.1:8080/v1/audio/transcriptions
```

Đổi alias CAPU:

```json
{
  "capu_models": [
    {
      "id": "vibert-capu-v2",
      "model_dir": "models/capu/vibert-capu-v2"
    }
  ],
  "models": [
    {
      "id": "vit_stt_vi_v2",
      "capu_model_id": "vibert-capu-v2"
    }
  ]
}
```

Khi nâng cấp model:

1. Đặt file model mới dưới `models/stt/...` hoặc `models/capu/...`.
2. Cập nhật `config/models.local.json`.
3. Cập nhật `baselines/phase0/assets.lock.json` với `path`, `url` hoặc
   `archive_url`, và `sha256`.
4. Nếu output probe thay đổi, cập nhật baseline có chủ ý; không sửa expectation
   im lặng.
5. Chạy validation:

```bash
./bin/stt-cli download-models
./bin/stt-cli verify-assets
./bin/stt-cli doctor
./bin/stt-cli check-capu
./bin/stt-cli probe --model vit_stt_vi_v2
./bin/stt-cli probe --model vit_stt_en_v2
```

Tính checksum trên macOS:

```bash
shasum -a 256 path/to/model-file
```

Tính checksum trên Windows PowerShell:

```powershell
Get-FileHash path\to\model-file -Algorithm SHA256
```

## macOS service helper

Dịch vụ chạy trên macOS sử dụng LaunchDaemon làm trình quản lý tiến trình nền (Option A). `stt-cli` hỗ trợ cài đặt và quản lý các daemon này.

### 1. Cài đặt HTTP Service

```bash
sudo ./bin/stt-cli service install
./bin/stt-cli service status
./bin/stt-cli service health
./bin/stt-cli service logs --lines 200
./bin/stt-cli service logs --follow
```

Các file cấu hình service local như `config/service.json`,
`config/llama-service.json` được ignore khỏi git. Ngoài ra, các file
`config/runtime.toml` và `config/models.local.json` cũng được ignore —
mỗi máy cần copy từ file `.example.*` tương ứng và điều chỉnh riêng.

Mẫu plist cho LaunchDaemon có sẵn tại `deploy/com.vit-stt.http.plist` và
`deploy/com.vit-stt.llama.plist`:

```bash
# Cài đặt hoặc cập nhật HTTP service
sudo cp deploy/com.vit-stt.http.plist /Library/LaunchDaemons/
sudo launchctl bootstrap system /Library/LaunchDaemons/com.vit-stt.http.plist

# Cài đặt hoặc cập nhật llama-server service
sudo cp deploy/com.vit-stt.llama.plist /Library/LaunchDaemons/
sudo launchctl bootstrap system /Library/LaunchDaemons/com.vit-stt.llama.plist
```

### 2. Cấu hình chống Sleep khi tắt màn hình

Mặc định máy Mac sẽ tự động ngủ (sleep) khi màn hình tắt, làm ngắt quãng các tiến trình chạy nền. Để giữ dịch vụ hoạt động liên tục khi cắm sạc:

- **Qua Terminal (Khuyên dùng):**
  ```bash
  sudo pmset -c sleep 0
  ```
- **Qua GUI:** Vào `System Settings` -> `Lock Screen` (hoặc `Battery` / `Energy Saver`) và bật tùy chọn **"Prevent automatic sleeping on power adapter when the display is off"** (Ngăn tự động ngủ khi cắm sạc lúc màn hình tắt).

## llama.cpp service helper

Repo có helper để tạo LaunchDaemon cho `llama-server` dùng trong thí nghiệm
LLM/CAPU. Process mặc định:

```bash
llama-server -hf unsloth/gemma-4-E4B-it-GGUF --temp 0.3 --top-p 0.95 --top-k 64 --reasoning off -c 32768 --host 0.0.0.0 --port 8001 --alias Tom_tat
```

Cấu hình binary path rồi cài service:

```bash
./bin/stt-cli llama-service config --bin-path "$(command -v llama-server)"
sudo ./bin/stt-cli llama-service install
./bin/stt-cli llama-service status
./bin/stt-cli llama-service health
./bin/stt-cli llama-service logs --lines 200
```

Mẫu plist cho LaunchDaemon cũng có sẵn tại `deploy/com.vit-stt.llama.plist`, sẵn
sàng để copy và cài đặt thủ công nếu không dùng CLI.

## Đóng gói release

Build release binaries:

```bash
cargo build --release -p stt-cli -p stt-http
```

Stage layout release:

```bash
scripts/package-release release/vit-stt
```

Nếu build theo target cụ thể:

```bash
cargo build --release --target aarch64-apple-darwin
VIT_STT_RELEASE_BIN_DIR=target/aarch64-apple-darwin/release \
  scripts/package-release release/vit-stt-macos-arm64
```

Layout release mặc định:

```text
vit-stt/
  bin/
    stt-cli
    stt-http
  config/
    runtime.toml
    models.local.json
  baselines/
    phase0/
  models/
    stt/
    vad/
    capu/
  runtime/
    capu/
```

Để tạo gói code/runtime không kèm model:

```bash
VIT_STT_INCLUDE_MODELS=0 scripts/package-release release/vit-stt-code-runtime
```

Nếu có CAPU worker đã đóng gói:

```bash
VIT_STT_CAPU_WORKER_SRC=/path/to/vit-stt-capu-worker-full-numpy \
  scripts/package-release release/vit-stt
```

GitHub Actions build release trên runner native cho từng platform, stage layout
portable, và smoke-test `stt-cli --help` cùng `list-models`. Decode validation
đầy đủ vẫn cần model assets và nên chạy trên máy local hoặc VM native trước khi
phát hành.

## Kiểm thử và validation

Trước khi báo hoàn tất thay đổi code, chạy tối thiểu:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

Với thay đổi liên quan HTTP hoặc runtime, kiểm tra thêm:

```bash
cargo run -p stt-cli -- verify-assets
cargo run -p stt-cli -- doctor
cargo run -p stt-cli -- check-capu
cargo run -p stt-cli -- probe --model vit_stt_vi_v2
cargo run -p stt-cli -- probe --model vit_stt_en_v2
```

Với HTTP API, cần xác nhận:

- `GET /health` trả `{"status":"ok","ready":true}` sau warm-start
- `GET /v1/models`
- `POST /v1/audio/transcriptions`
- response dạng JSON và text
- output tiếng Việt baseline khớp `Định nghĩa thế nào là ăn mặc đẹp?`

Với warm-start và admin endpoints, kiểm tra thêm:

- `GET /admin/models/status` trả model list với loaded state
- `POST /admin/warmup` trả timing và probe result
- Model tiếng Việt prewarm khi có `warm_start_models = ["vit_stt_vi_v2"]`
- Log server hiển thị `warm-start completed` và `recognizer_cache_hit`

## Lưu ý CAPU

CAPU là hành vi bắt buộc cho tiếng Việt, không phải tính năng phụ. Nếu
`check-capu` lỗi, không fallback thủ công sang postprocessing yếu hơn; cần sửa
CAPU runtime hoặc model CAPU trước khi phục vụ production.

CAPU hiện được bọc sau interface Rust ổn định nhưng implementation production
vẫn dùng Python worker riêng vì asset đang là PyTorch/custom Python. Không đổi
model CAPU mặc định sang candidate khác nếu chưa cập nhật baseline và benchmark.

## Giấy phép và model assets

- Không commit model artifact lớn nếu chưa có yêu cầu rõ ràng.
- Không giả định được phép phân phối thương mại model upstream.
- Ưu tiên manifest, checksum, download script và tài liệu thay vì copy binary.
- Model files nên nằm ở `models/` hoặc thư mục external được trỏ bằng
  `VIT_STT_MODELS_DIR`.

## Cấu trúc repo

```text
vit-stt/
  Cargo.toml
  baselines/phase0/
  capu-worker/
  config/
  crates/
    stt-core/
    stt-capu/
    stt-http/
    stt-cli/
  deploy/
    com.vit-stt.http.plist
    com.vit-stt.llama.plist
    Dockerfile
    docker-compose.yml
    *.service (systemd)
  models/
  scripts/
```

## Tích hợp với ứng dụng khác

Ứng dụng bên ngoài nên dùng HTTP API tương thích OpenAI để gọi STT. Nếu cần tích
hợp sâu hơn, có thể phụ thuộc trực tiếp vào `stt-core`/`stt-capu` và tự viết
adapter ở tầng ứng dụng.
