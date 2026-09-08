# stt-http

HTTP server crate for `vit-stt`.

Endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/transcriptions`

The service is intended to preserve compatibility with the current OpenAI-style transcription contract used by consuming apps.

## Audio formats

The multipart `file` may be WAV, Ogg Opus, or another format supported by the
configured FFmpeg runtime. The service decodes input to 16 kHz mono PCM before
inference. EchoVox persists and uploads Ogg Opus directly; WAV remains fully
backward compatible.

## Idempotent transcription

`POST /v1/audio/transcriptions` accepts an optional `Idempotency-Key` header
containing 1–255 visible ASCII characters.

- The first request registers the key and performs inference.
- A concurrent duplicate with the same semantic fingerprint waits for the
  original computation.
- A completed duplicate returns the cached status/body without decoding or
  inference.
- Reusing the key with different audio, model, language, or response format
  returns HTTP 409 with `code: idempotency_conflict`.
- Requests without the header keep the original behavior.

Responses echo `Idempotency-Key`, include the server request ID, and return
`X-Idempotency-Status: created|coalesced|cached|conflict`.

Successful records are stored without raw audio for 24 hours in the bounded
durable cache. The default directory is
`.cache/stt-http/idempotency`; override it with
`VIT_STT_IDEMPOTENCY_CACHE_DIR`.
