# Bàn giao server vit-stt

## Tương thích với EchoVox

EchoVox gửi clip giọng nói Ogg Opus trực tiếp tới
`POST /v1/audio/transcriptions` và kèm `Idempotency-Key` ổn định. Vì vậy FFmpeg
phải tiếp tục có sẵn để decode input.

Server gộp các request trùng đang chạy, trả lại kết quả thành công đã cache
trong 24 giờ, và trả HTTP 409 nếu cùng key được dùng với fingerprint khác.
Cache idempotency lưu status và response, không lưu audio cuộc họp thô. Cần giữ
thư mục `VIT_STT_IDEMPOTENCY_CACHE_DIR` qua các lần restart nếu deployment yêu
cầu retry liên tục.

## Nền tảng hỗ trợ

Deployment target chính thức:

- macOS Apple Silicon;
- Windows x86_64.

Linux không phải nền tảng bàn giao chính thức cho khách hàng.

Môi trường phát triển chính là macOS.

## Mô hình triển khai

Gói release mặc định của `vit-stt` không bundle model weights.

Khi bàn giao offline/on-premise cho khách hàng, VietInnotech chuẩn bị server bằng cách:

1. cài FFmpeg/FFprobe riêng trên server;
2. giải nén hoặc cài đặt `vit-stt`;
3. chạy tải model và kiểm tra checksum;
4. chạy kiểm tra CAPU và probe model;
5. để lại đầy đủ license notice, model BOM, dependency BOM và tài liệu bàn giao trong thư mục triển khai.

## FFmpeg / FFprobe

FFmpeg/FFprobe không nằm trong repo và không nằm trong gói proprietary release.

Cài riêng trên server trước khi bàn giao.

Ghi lại thông tin phiên bản:

```bash
ffmpeg -version
ffprobe -version
```

## Chuẩn bị model trên server

Chạy:

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

## File pháp lý bắt buộc trong thư mục triển khai

Thư mục triển khai phải có:

```text
LICENSE
THIRD_PARTY_NOTICES.md
MODEL_BOM.md
DEPENDENCY_BOM.md
licenses/
```

## Ghi chú về model license

Model weights không bị VietInnotech sửa đổi.

Server có thể chứa model files từ:

* Gipformer;
* NVIDIA Parakeet / sherpa-onnx packaged assets;
* leakless/vibert-capu / dragonSwing/vibert-capu;
* VAD assets.

Các model này vẫn là tài sản hoặc thành phần bên thứ ba và tuân theo license gốc của từng nguồn.

Không được tuyên bố VietInnotech sở hữu model weights bên thứ ba.

Không áp dụng hạn chế pháp lý hoặc kỹ thuật bổ sung lên model CC-BY-SA nếu hạn chế đó cản trở quyền được license cho phép.

## Tư thế pháp lý trước khi bàn giao

* `ten-vad.int8.onnx` lấy từ release assets của `k2-fsa/sherpa-onnx`; repo nguồn dùng Apache-2.0. Khi bàn giao có asset này, giữ kèm license Apache-2.0 và notices liên quan.
* `leakless/vibert-capu` và upstream `dragonSwing/vibert-capu` khai báo CC-BY-SA-4.0 cho model CAPU. Khi bàn giao có các file này, giữ kèm license CC-BY-SA-4.0, attribution, source URL, checksum, và trạng thái không sửa model weights.
* `models/capu/vibert-capu/base_model/` chứa tokenizer/config lấy từ `FPTAI/vibert-base-cased`. Tại lần kiểm tra nguồn ngày 2026-06-30, Hugging Face API và model card không có metadata license rõ ràng cho nguồn này. Không đóng gói các file `base_model/` vào offline model pack có thể phân phối lại cho tới khi VietInnotech ghi nhận license grant, permission, hoặc nguồn thay thế.
* Không thấy yêu cầu attribution riêng ngoài việc giữ source names, source URLs, license texts, checksums, và ghi rõ VietInnotech không sửa model weights trong `MODEL_BOM.md` và `THIRD_PARTY_NOTICES.md`.
