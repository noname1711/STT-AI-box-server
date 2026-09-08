#!/usr/bin/env python3
"""HL Meet v19 line-protocol bridge to the native VIT-STT HTTP backend.

Input modes:
  PCM\t<raw_pcm_path>\t<vi|en|auto>\t<client_id>
  WAV\t<wav_path>\t<vi|en|auto>\t<client_id>

Legacy input remains accepted:
  <raw_pcm_path>\t<vi|en|auto>\t<client_id>

Output:
  READY
  OK\t<vi|en>\t<text>\t<pcm|wav>
  SKIP\t<reason>
  ERR\t<reason>

PCM is the realtime transport/capture format. FINAL correction windows are
materialized as WAV by meeting-server and sent unchanged to native stt-http.
Both lanes use the same company VIT model stack.
"""

from __future__ import annotations

import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

SAMPLE_RATE = 16_000
CHANNELS = 1
BITS_PER_SAMPLE = 16
BASE_URL = os.getenv("MEETING_VIT_STT_HTTP_URL", "http://127.0.0.1:8091").rstrip("/")
CONNECT_WAIT_SECONDS = max(1, int(os.getenv("MEETING_VIT_STT_CONNECT_TIMEOUT", "120")))
REQUEST_TIMEOUT = max(1, int(os.getenv("MEETING_VIT_STT_REQUEST_TIMEOUT", "180")))

MODELS = {
    "vi": "vit_stt_vi_v2",
    "en": "vit_stt_en_v2",
}

VI_WORDS = {
    "tôi", "bạn", "anh", "chị", "em", "là", "có", "không", "và", "của",
    "cho", "một", "những", "này", "đó", "được", "xin", "chào", "mọi", "người",
    "hôm", "nay", "chúng", "ta", "sẽ", "đang", "với", "trong", "thì", "rất",
}
EN_WORDS = {
    "i", "you", "he", "she", "we", "they", "the", "a", "an", "is", "are",
    "am", "was", "were", "and", "or", "to", "of", "for", "in", "on", "with",
    "this", "that", "hello", "today", "will", "can", "not", "have", "has", "be",
}


def eprint(*args) -> None:
    print(*args, file=sys.stderr, flush=True)


def clean_text(value: str) -> str:
    return " ".join((value or "").replace("\t", " ").replace("\r", " ").replace("\n", " ").split())


def pcm16_to_wav(raw: bytes) -> bytes:
    if len(raw) & 1:
        raw = raw[:-1]
    data_size = len(raw)
    byte_rate = SAMPLE_RATE * CHANNELS * (BITS_PER_SAMPLE // 8)
    block_align = CHANNELS * (BITS_PER_SAMPLE // 8)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE", b"fmt ", 16, 1, CHANNELS,
        SAMPLE_RATE, byte_rate, block_align, BITS_PER_SAMPLE, b"data", data_size,
    )
    return header + raw


def looks_like_pcm16_wav(data: bytes) -> bool:
    return len(data) >= 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def multipart_body(model: str, language: str, wav_bytes: bytes):
    boundary = "----HLMeetVITSTT" + uuid.uuid4().hex
    b = boundary.encode("ascii")
    chunks = []

    def field(name: str, value: str) -> None:
        chunks.extend([
            b"--" + b + b"\r\n",
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode("utf-8"),
            b"\r\n",
        ])

    field("model", model)
    field("language", language)
    field("response_format", "json")
    chunks.extend([
        b"--" + b + b"\r\n",
        b'Content-Disposition: form-data; name="file"; filename="utterance.wav"\r\n',
        b"Content-Type: audio/wav\r\n\r\n",
        wav_bytes,
        b"\r\n",
        b"--" + b + b"--\r\n",
    ])
    return boundary, b"".join(chunks)


def health_ready() -> bool:
    req = urllib.request.Request(BASE_URL + "/health", headers={"Connection": "close"})
    try:
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status != 200:
                return False
            data = json.loads(response.read().decode("utf-8", errors="replace"))
            return data.get("status") == "ok" and data.get("ready") is True
    except Exception:
        return False


def wait_for_backend() -> bool:
    deadline = time.monotonic() + CONNECT_WAIT_SECONDS
    while time.monotonic() < deadline:
        if health_ready():
            return True
        time.sleep(0.5)
    return False


def transcribe_once_wav(wav_bytes: bytes, language: str) -> str:
    model = MODELS[language]
    boundary, body = multipart_body(model, language, wav_bytes)
    req = urllib.request.Request(
        BASE_URL + "/v1/audio/transcriptions",
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "Accept": "application/json",
            "Accept-Language": language,
            "Connection": "close",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"native HTTP {exc.code}: {detail[:600]}") from exc
    return clean_text(str(payload.get("text", "")))


def repetition_penalty(words):
    penalty = 0.0
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            penalty += 1.2
    if len(words) >= 6:
        uniq = len(set(words)) / len(words)
        if uniq < 0.45:
            penalty += (0.45 - uniq) * 8.0
    return penalty


def language_score(text: str, language: str) -> float:
    words = re.findall(r"[^\W\d_]+", text.casefold(), flags=re.UNICODE)
    if not words:
        return -100.0
    score = min(len(words), 12) * 0.08 - repetition_penalty(words)
    if language == "vi":
        score += sum(word in VI_WORDS for word in words) * 0.55
        score += sum(
            any(ch in "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ" for ch in word)
            for word in words
        ) * 0.35
    else:
        score += sum(word in EN_WORDS for word in words) * 0.55
        score -= sum(any(ch in "ăâđêôơư" for ch in word) for word in words) * 1.0
    return score


def transcribe_wav_bytes(wav_bytes: bytes, pref: str):
    pref = pref if pref in {"vi", "en", "auto"} else "vi"
    if pref in {"vi", "en"}:
        return pref, transcribe_once_wav(wav_bytes, pref), "single"

    # Explicit VI/EN remains the intended accuracy path. Auto uses both company
    # models and a lightweight language score; no external ASR model is involved.
    vi_text = transcribe_once_wav(wav_bytes, "vi")
    en_text = transcribe_once_wav(wav_bytes, "en")
    if language_score(en_text, "en") > language_score(vi_text, "vi"):
        return "en", en_text, "dual-vit"
    return "vi", vi_text, "dual-vit"


def parse_request(line: str):
    parts = line.split("\t")
    if parts and parts[0] in {"PCM", "WAV"}:
        if len(parts) < 3:
            raise ValueError("bad-request")
        mode = parts[0].lower()
        path = Path(parts[1])
        pref = parts[2].strip().lower()
        client_id = parts[3].strip() if len(parts) >= 4 else ""
        return mode, path, pref, client_id

    # v18/v17 compatibility: path<TAB>language<TAB>client_id means raw PCM.
    path = Path(parts[0])
    pref = parts[1].strip().lower() if len(parts) >= 2 else "vi"
    client_id = parts[2].strip() if len(parts) >= 3 else ""
    return "pcm", path, pref, client_id


def main() -> int:
    if not wait_for_backend():
        eprint(f"native VIT-STT backend not ready at {BASE_URL}")
        return 2

    print("READY", flush=True)

    for raw_line in sys.stdin:
        line = raw_line.rstrip("\r\n")
        if not line:
            print("SKIP\tempty-request", flush=True)
            continue
        if line == "QUIT":
            return 0

        try:
            mode, path, pref, client_id = parse_request(line)
            payload = path.read_bytes()
            if not payload:
                print("SKIP\tempty-audio", flush=True)
                continue

            if mode == "pcm":
                wav_bytes = pcm16_to_wav(payload)
            else:
                if not looks_like_pcm16_wav(payload):
                    raise RuntimeError("invalid-wav")
                wav_bytes = payload

            lang, text, decode_mode = transcribe_wav_bytes(wav_bytes, pref)
            text = clean_text(text)
            if not text:
                print("SKIP\tempty-transcript", flush=True)
                continue

            eprint(
                f"native decode lane={mode} lang={lang} mode={decode_mode} "
                f"chars={len(text)} client={client_id or '-'} file={path.name}"
            )
            print(f"OK\t{lang}\t{text}\t{mode}-{decode_mode}", flush=True)
        except Exception as exc:
            message = clean_text(str(exc))[:600]
            eprint(f"native decode error: {message}")
            print(f"ERR\t{message}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
