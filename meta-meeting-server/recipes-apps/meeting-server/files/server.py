#!/usr/bin/env python3

import asyncio
import json
import logging
import math
import os
import re
import shutil
import signal
import struct
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from aiohttp import WSMsgType, web


APP_VERSION = "20.0.0-rc3"
RUNTIME_DIR = Path("/run/meeting-server")
PREVIEW_DIR = RUNTIME_DIR / "preview"
STATE_DIR = Path(os.getenv("MEETING_STATE_DIR", "/var/lib/meeting-server"))
FINAL_SPOOL_DIR = STATE_DIR / "spool"

# V20 post-ASR anonymous speaker tracking.
# This lane never participates in VIT recognition/arbitration.
SPEAKER_TMP_DIR = STATE_DIR / "speaker-tmp"
SPEAKER_ENABLED = os.getenv("MEETING_SPEAKER_ENABLED", "1") == "1"
SPEAKER_THRESHOLD = float(
    os.getenv("MEETING_SPEAKER_THRESHOLD", "0.60")
)
SPEAKER_TIMEOUT_SECONDS = int(
    os.getenv("MEETING_SPEAKER_TIMEOUT_SECONDS", "60")
)
# Reject only speaker embeddings from very short clips. STT still consumes
# every captured sample exactly as before; this guard is post-ASR metadata only.
SPEAKER_MIN_AUDIO_MS = max(
    0,
    int(os.getenv("MEETING_SPEAKER_MIN_AUDIO_MS", "1000")),
)
SPEAKER_NICE = max(
    0,
    min(19, int(os.getenv("MEETING_SPEAKER_NICE", "15"))),
)
UNRESOLVED_DIR = STATE_DIR / "unresolved"
STATE_MIN_FREE_MB = max(128, int(os.getenv("MEETING_STATE_MIN_FREE_MB", "1024")))
STATE_MIN_FREE_BYTES = STATE_MIN_FREE_MB * 1024 * 1024

HTTP_PORT = int(os.getenv("MEETING_HTTP_PORT", "8080"))
SAMPLE_RATE = 16000
SAMPLE_BYTES = 2
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_BYTES

# Capture / segmentation. These limits constrain inference windows only; a person
# can continue speaking for arbitrarily long periods.
SEGMENT_SOFT_SECONDS = float(os.getenv("MEETING_SEGMENT_SOFT_SECONDS", "8.0"))
SEGMENT_HARD_SECONDS = float(os.getenv("MEETING_SEGMENT_HARD_SECONDS", "12.0"))
SEGMENT_SOFT_BYTES = int(BYTES_PER_SECOND * SEGMENT_SOFT_SECONDS)
SEGMENT_HARD_BYTES = int(BYTES_PER_SECOND * SEGMENT_HARD_SECONDS)
SEGMENT_OVERLAP_MS = int(os.getenv("MEETING_SEGMENT_OVERLAP_MS", "800"))
SEGMENT_OVERLAP_BYTES = BYTES_PER_SECOND * SEGMENT_OVERLAP_MS // 1000
MIN_UTTERANCE_MS = int(os.getenv("MEETING_MIN_UTTERANCE_MS", "250"))
MIN_UTTERANCE_BYTES = BYTES_PER_SECOND * MIN_UTTERANCE_MS // 1000

# Server-owned VAD. Browser Automatic mode always sends PCM; the browser never
# decides which speech is allowed to reach the AI Box.
STREAM_FRAME_MS = 20
STREAM_FRAME_BYTES = BYTES_PER_SECOND * STREAM_FRAME_MS // 1000
WS_MAX_PCM_MESSAGE_BYTES = max(
    STREAM_FRAME_BYTES,
    int(os.getenv("MEETING_WS_MAX_PCM_MESSAGE_BYTES", "262144")),
)
WS_SEND_TIMEOUT_SECONDS = max(
    0.25,
    float(os.getenv("MEETING_WS_SEND_TIMEOUT_SECONDS", "1.5")),
)
PCM_REFRAME_YIELD_FRAMES = max(
    8, int(os.getenv("MEETING_PCM_REFRAME_YIELD_FRAMES", "32"))
)
# Bound network fan-in independently of AI throughput. A normal browser streams
# at 1x realtime; generous burst/rate headroom tolerates scheduler/network jitter
# without allowing one broken LAN client to feed minutes of PCM per second.
MAX_CLIENTS = max(1, min(128, int(os.getenv("MEETING_MAX_CLIENTS", "16"))))
PCM_MAX_REALTIME_FACTOR = max(1.25, min(8.0, float(
    os.getenv("MEETING_PCM_MAX_REALTIME_FACTOR", "4.0")
)))
PCM_RATE_BURST_SECONDS = max(2.0, min(30.0, float(
    os.getenv("MEETING_PCM_RATE_BURST_SECONDS", "10.0")
)))
PCM_RATE_BURST_BYTES = int(BYTES_PER_SECOND * PCM_RATE_BURST_SECONDS)
STREAM_PRE_ROLL_MS = int(os.getenv("MEETING_STREAM_PRE_ROLL_MS", "600"))
STREAM_PRE_ROLL_FRAMES = max(1, STREAM_PRE_ROLL_MS // STREAM_FRAME_MS)
STREAM_ONSET_MS = int(os.getenv("MEETING_STREAM_ONSET_MS", "80"))
STREAM_ONSET_FRAMES = max(1, STREAM_ONSET_MS // STREAM_FRAME_MS)
STREAM_HANGOVER_MS = int(os.getenv("MEETING_STREAM_HANGOVER_MS", "700"))
STREAM_HANGOVER_FRAMES = max(1, STREAM_HANGOVER_MS // STREAM_FRAME_MS)
STREAM_SOFT_SILENCE_MS = int(os.getenv("MEETING_STREAM_SOFT_SILENCE_MS", "320"))
STREAM_SOFT_SILENCE_FRAMES = max(1, STREAM_SOFT_SILENCE_MS // STREAM_FRAME_MS)
STREAM_MIN_THRESHOLD = float(os.getenv("MEETING_STREAM_MIN_THRESHOLD", "0.006"))
STREAM_NOISE_MULTIPLIER = float(os.getenv("MEETING_STREAM_NOISE_MULTIPLIER", "2.4"))
STREAM_MIN_PEAK = float(os.getenv("MEETING_STREAM_MIN_PEAK", "0.012"))
STREAM_NOISE_LEARN_ENABLED = os.getenv("MEETING_STREAM_NOISE_LEARN_ENABLED", "1") == "1"
STREAM_NOISE_LEARN_MIN_MS = int(os.getenv("MEETING_STREAM_NOISE_LEARN_MIN_MS", "1800"))
STREAM_NOISE_REJECT_BLEND = float(os.getenv("MEETING_STREAM_NOISE_REJECT_BLEND", "0.35"))
STREAM_NOISE_MAX_FLOOR = float(os.getenv("MEETING_STREAM_NOISE_MAX_FLOOR", "0.0065"))

# v19 logical-turn continuity + VIT-only rolling FINAL WAV correction.
LOGICAL_TURN_GAP_MS = int(os.getenv("MEETING_LOGICAL_TURN_GAP_MS", "1500"))
FINAL_LEFT_CONTEXT_SECONDS = float(os.getenv("MEETING_FINAL_LEFT_CONTEXT_SECONDS", "10"))
FINAL_MAX_WINDOW_SECONDS = float(os.getenv("MEETING_FINAL_MAX_WINDOW_SECONDS", "24"))
FINAL_LEFT_CONTEXT_BYTES = int(BYTES_PER_SECOND * FINAL_LEFT_CONTEXT_SECONDS)
FINAL_MAX_WINDOW_BYTES = int(BYTES_PER_SECOND * FINAL_MAX_WINDOW_SECONDS)
FINAL_MERGE_TAIL_WORDS = int(os.getenv("MEETING_FINAL_MERGE_TAIL_WORDS", "100"))

# LIVE lane. This is disposable and lower priority than captured FINAL speech.
PARTIAL_ENABLED = os.getenv("MEETING_PARTIAL_ENABLED", "1") == "1"
PARTIAL_MIN_AUDIO_MS = int(os.getenv("MEETING_PARTIAL_MIN_AUDIO_MS", "1200"))
PARTIAL_INTERVAL_MS = int(os.getenv("MEETING_PARTIAL_INTERVAL_MS", "900"))
PARTIAL_WINDOW_MS = int(os.getenv("MEETING_PARTIAL_WINDOW_MS", "5000"))
PARTIAL_TIMEOUT_SECONDS = int(os.getenv("MEETING_PARTIAL_TIMEOUT_SECONDS", "60"))
PARTIAL_MIN_BYTES = BYTES_PER_SECOND * PARTIAL_MIN_AUDIO_MS // 1000
PARTIAL_INTERVAL_BYTES = BYTES_PER_SECOND * PARTIAL_INTERVAL_MS // 1000
PARTIAL_WINDOW_BYTES = BYTES_PER_SECOND * PARTIAL_WINDOW_MS // 1000

# Final queue is intentionally unbounded. v19 does not delete captured speech
# merely because inference is temporarily slower than capture.
QUEUE_CAPACITY = int(os.getenv("MEETING_STT_QUEUE_CAPACITY", "0"))

# Non-destructive cross-talk hold retained for multi-client rooms. It may annotate
# echo suspicion but production v19 never drops the weaker capture here.
PRE_STT_GATE_HOLD_MS = int(os.getenv("MEETING_PRE_STT_GATE_HOLD_MS", "500"))
PRE_STT_GATE_PENDING_CAPACITY = int(os.getenv("MEETING_PRE_STT_GATE_PENDING_CAPACITY", "128"))
PRE_STT_AVG_RMS_RATIO = float(os.getenv("MEETING_PRE_STT_AVG_RMS_RATIO", "1.60"))
PRE_STT_PEAK_RMS_RATIO = float(os.getenv("MEETING_PRE_STT_PEAK_RMS_RATIO", "1.25"))
PRE_STT_START_DELTA_MS = 650
PRE_STT_END_DELTA_MS = 900
PRE_STT_MIN_OVERLAP_RATIO = 0.70
PRE_STT_DESTRUCTIVE = os.getenv("MEETING_PRE_STT_DESTRUCTIVE", "0") == "1"

SUPPORTED_LANGUAGES = {"vi", "en"}
MAX_HISTORY_LIMIT = 1000
MAX_SESSION_TURNS = 2000
MAX_TRANSCRIPT_CHAR_RATE = 55
MIN_TRANSCRIPT_CHAR_BUDGET = 220

VIT_TIMEOUT_SECONDS = int(os.getenv("MEETING_VIT_STT_REQUEST_TIMEOUT", "180"))

# Translation is post-ASR only and never participates in transcript arbitration.
# v19.5 has two source lanes using the same accuracy-first MADLAD model:
#   LIVE: disposable translation of already-confirmed VIT LIVE text for UX.
#   FINAL: authoritative translation of the finalized logical-turn source.
TRANSLATION_ENABLED = os.getenv("MEETING_TRANSLATION_ENABLED", "1") == "1"
TRANSLATION_TIMEOUT_SECONDS = int(os.getenv("MEETING_TRANSLATION_TIMEOUT_SECONDS", "300"))
TRANSLATION_IDLE_POLL_MS = int(os.getenv("MEETING_TRANSLATION_IDLE_POLL_MS", "120"))
TRANSLATION_PROGRESSIVE_ENABLED = (
    os.getenv("MEETING_TRANSLATION_PROGRESSIVE_ENABLED", "1") == "1"
)
TRANSLATION_LIVE_PREVIEW_ENABLED = (
    os.getenv("MEETING_TRANSLATION_LIVE_PREVIEW_ENABLED", "0") == "1"
)
TRANSLATION_LIVE_MIN_WORDS = max(3, int(
    os.getenv("MEETING_TRANSLATION_LIVE_MIN_WORDS", "6")
))
TRANSLATION_LIVE_MIN_INTERVAL_MS = max(300, int(
    os.getenv("MEETING_TRANSLATION_LIVE_MIN_INTERVAL_MS", "1200")
))
TRANSLATION_LIVE_TIMEOUT_SECONDS = max(5, min(60, int(
    os.getenv("MEETING_TRANSLATION_LIVE_TIMEOUT_SECONDS", "20")
)))
# Keep each disposable MADLAD preview bounded. The VIT LIVE window is already
# short; this cap is a safety rail against pathological tokenization / unusually
# fast speech holding the single native translator lane in front of FINAL work.
TRANSLATION_LIVE_MAX_WORDS = max(6, min(64, int(
    os.getenv("MEETING_TRANSLATION_LIVE_MAX_WORDS", "24")
)))
TRANSLATOR_REQUESTED_DEVICE = str(
    os.getenv("MEETING_TRANSLATOR_DEVICE", "cpu")
).strip().lower() or "cpu"
TRANSLATOR_SOCKET_PATH = str(
    os.getenv(
        "MEETING_TRANSLATOR_SOCKET",
        "/run/meeting-translator/translator.sock",
    )
).strip()
TRANSLATOR_CONNECT_TIMEOUT_SECONDS = max(5, min(300, int(
    os.getenv("MEETING_TRANSLATOR_CONNECT_TIMEOUT_SECONDS", "180")
)))
# v19.4 uses a document-oriented MT model. Wider sentence-aware chunks preserve
# enough semantic context for Vietnamese pronouns, terminology and long clauses
# while remaining comfortably below the translator's 512-token source ceiling.
TRANSLATION_CHUNK_TRIGGER_WORDS = max(16, int(os.getenv("MEETING_TRANSLATION_CHUNK_TRIGGER_WORDS", "72")))
TRANSLATION_CHUNK_WORDS = max(24, int(os.getenv("MEETING_TRANSLATION_CHUNK_WORDS", "96")))
TRANSLATION_CHUNK_MIN_TAIL_WORDS = max(6, int(os.getenv("MEETING_TRANSLATION_CHUNK_MIN_TAIL_WORDS", "18")))
TRANSLATION_PROGRESSIVE_FIRST_WORDS = max(
    16, int(os.getenv("MEETING_TRANSLATION_PROGRESSIVE_FIRST_WORDS", "72"))
)
TRANSLATION_PROGRESSIVE_MIN_NEW_WORDS = max(
    16, int(os.getenv("MEETING_TRANSLATION_PROGRESSIVE_MIN_NEW_WORDS", "96"))
)
TRANSLATION_RECOVERY_MAX_DEPTH = max(
    1, min(4, int(os.getenv("MEETING_TRANSLATION_RECOVERY_MAX_DEPTH", "3")))
)
TRANSLATION_ROUNDTRIP_QA = os.getenv("MEETING_TRANSLATION_ROUNDTRIP_QA", "1") == "1"
TRANSLATION_ROUNDTRIP_MIN_SIMILARITY = min(
    0.90,
    max(0.10, float(os.getenv("MEETING_TRANSLATION_ROUNDTRIP_MIN_SIMILARITY", "0.34"))),
)
TRANSLATION_ROUNDTRIP_MIN_WORDS = max(
    3, int(os.getenv("MEETING_TRANSLATION_ROUNDTRIP_MIN_WORDS", "5"))
)
TRANSLATION_ROUNDTRIP_MAX_WORDS = max(
    16, int(os.getenv("MEETING_TRANSLATION_ROUNDTRIP_MAX_WORDS", "90"))
)
TRANSLATION_PRIMARY_GREEDY = os.getenv("MEETING_TRANSLATION_PRIMARY_GREEDY", "1") == "1"
TRANSLATOR_NICE = max(0, min(19, int(os.getenv("MEETING_TRANSLATOR_NICE", "10"))))
TRANSLATOR_ALLOW_OPUS_FALLBACK = os.getenv("MEETING_TRANSLATOR_ALLOW_OPUS_FALLBACK", "0") == "1"
TRANSLATION_PROTECTED_TERMS = tuple(
    term.strip()
    for term in os.getenv(
        "MEETING_TRANSLATION_PROTECTED_TERMS",
        "HL Meet,NVIDIA,Jetson,Orin Nano,CUDA,CTranslate2,NVMe,MADLAD,Gipformer,Parakeet",
    ).split(",")
    if term.strip()
)
TRANSLATOR_RESTART_BACKOFF_SECONDS = max(5, int(os.getenv("MEETING_TRANSLATOR_RESTART_BACKOFF_SECONDS", "30")))
TRANSLATOR_RESTART_BACKOFF_MAX_SECONDS = max(
    TRANSLATOR_RESTART_BACKOFF_SECONDS,
    int(os.getenv("MEETING_TRANSLATOR_RESTART_BACKOFF_MAX_SECONDS", "120")),
)
VI_WRITTEN_FORM_ENABLED = os.getenv("MEETING_VI_WRITTEN_FORM_ENABLED", "1") == "1"
VI_ITN_PERCENT_ENABLED = os.getenv("MEETING_VI_ITN_PERCENT_ENABLED", "1") == "1"
VI_ITN_COMMON_ENABLED = os.getenv("MEETING_VI_ITN_COMMON_ENABLED", "1") == "1"

ALLOWED_WEB_ORIGINS = {
    item.strip().rstrip("/")
    for item in os.getenv("MEETING_ALLOWED_WEB_ORIGINS", "").split(",")
    if item.strip()
}
ALLOW_PAGES_DEV = os.getenv("MEETING_ALLOW_PAGES_DEV", "0") == "1"
ALLOW_LOCALHOST_DEV = os.getenv("MEETING_ALLOW_LOCALHOST_DEV", "1") == "1"
ALLOW_AUTO_LANGUAGE = os.getenv("MEETING_ALLOW_AUTO_LANGUAGE", "0") == "1"

logging.basicConfig(
    level=os.getenv("MEETING_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("meeting-server")


@dataclass
class Client:
    ws: web.WebSocketResponse
    client_id: str
    name: str = "Guest"
    room: str = "default"
    joined_at: float = field(default_factory=time.time)
    mic_state: str = "off"
    language_pref: str = "vi"
    # Snapshot at the start of an acoustic segment. UI language changes while a
    # person is speaking apply only to the next natural segment/turn.
    segment_language_pref: str = "vi"

    capture_active: bool = False
    capture_mode: str = "legacy"
    current_id: Optional[str] = None
    current_audio: bytearray = field(default_factory=bytearray)
    speech_started_ms: int = 0
    speech_started_mono: float = 0.0
    turn_id: str = ""
    segment_index: int = 0
    segment_leading_overlap_bytes: int = 0

    avg_rms: float = 0.0
    peak_rms: float = 0.0
    audio_frames: int = 0
    audio_rms_sum: float = 0.0
    speech_like_frames: int = 0

    pre_roll: deque = field(
        default_factory=lambda: deque(maxlen=STREAM_PRE_ROLL_FRAMES)
    )
    vad_onset_frames: int = 0
    vad_silent_frames: int = 0
    noise_floor: float = 0.003
    pending_noise_floor: float = 0.0

    # Bounded acoustic context from already-captured speech in the same logical
    # turn. It is raw PCM only; no previous transcript is fed back to the model.
    logical_context_audio: bytearray = field(default_factory=bytearray)
    logical_last_end_mono: float = 0.0

    partial_task: Optional[asyncio.Task] = None
    partial_last_bytes: int = 0
    partial_revision: int = 0
    partial_last_text: str = ""
    partial_candidate_text: str = ""
    partial_candidate_hits: int = 0
    partial_confirmed_text: str = ""

    # WebSocket messages are transport packets, not VAD frames. Buffer and
    # reframe PCM into exact 20 ms units before VAD/statistics consume it.
    pcm_remainder: bytearray = field(default_factory=bytearray)
    pcm_budget_bytes: float = field(default_factory=lambda: float(PCM_RATE_BURST_BYTES))
    pcm_budget_updated_mono: float = field(default_factory=time.monotonic)
    # A transport-evicted client remains counted until its handler finally exits.
    # This prevents a slow/half-closed socket from becoming an untracked PCM sender.
    closing: bool = False


@dataclass
class Utterance:
    utterance_id: str
    client_id: str
    speaker: str
    room: str
    language_pref: str
    turn_id: str
    segment_index: int

    raw_path: Path
    wav_path: Path
    started_ms: int
    ended_ms: int
    server_started_ms: int
    server_ended_ms: int
    avg_rms: float
    peak_rms: float
    speech_ratio: float
    audio_ms: int
    context_prefix_ms: int
    end_reason: str
    preview_text: str = ""
    gate_received_at: float = field(default_factory=time.monotonic)
    retain_audio: bool = False


class LineDaemon:
    """Persistent line protocol with strict request/response synchronization.

    After a timeout, EOF, or pipe failure the child stream is no longer trusted:
    a late response could otherwise be consumed by the next request. Reset the
    child immediately so every future request starts on a fresh protocol stream.
    """

    def __init__(self, argv, name, *, nice_value=0):
        self.argv = argv
        self.name = name
        self.nice_value = max(0, min(19, int(nice_value or 0)))
        self.proc = None
        self.lock = asyncio.Lock()
        self.reset_count = 0
        # Optional fields after READY are daemon-specific runtime metadata.
        # meeting-vit-stt still emits plain READY; meeting-translator emits
        # READY\t<actual-device>\t<actual-compute-type>.
        self.ready_metadata = []

    def is_ready(self):
        return bool(
            self.proc is not None
            and self.proc.returncode is None
        )

    async def _kill_untrusted(self, proc, reason):
        if proc is None:
            return
        if self.proc is proc:
            self.proc = None
            self.ready_metadata = []
        self.reset_count += 1
        log.warning("resetting %s after %s", self.name, reason)
        try:
            if proc.returncode is None:
                proc.kill()
        except ProcessLookupError:
            pass
        except Exception:
            log.exception("failed killing untrusted %s", self.name)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except Exception:
            pass

    async def start(self):
        if self.proc is not None and self.proc.returncode is None:
            return
        log.info("starting %s: %s", self.name, self.argv)
        proc = await asyncio.create_subprocess_exec(
            *self.argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.proc = proc
        if self.nice_value > 0:
            try:
                os.setpriority(os.PRIO_PROCESS, proc.pid, self.nice_value)
                log.info(
                    "set %s pid=%d nice=%d",
                    self.name, proc.pid, self.nice_value,
                )
            except Exception:
                log.exception(
                    "failed setting %s nice=%d",
                    self.name, self.nice_value,
                )
        try:
            ready = await asyncio.wait_for(proc.stdout.readline(), timeout=180)
        except asyncio.CancelledError:
            await self._kill_untrusted(proc, "startup-cancelled")
            raise
        except Exception:
            await self._kill_untrusted(proc, "startup-timeout-or-io-error")
            raise
        ready_line = ready.decode(errors="replace").rstrip("\r\n")
        ready_fields = ready_line.split("\t") if ready_line else []
        if not ready_fields or ready_fields[0] != "READY":
            try:
                err = await asyncio.wait_for(proc.stderr.read(), timeout=2)
            except asyncio.TimeoutError:
                err = b""
            await self._kill_untrusted(proc, "bad-ready-handshake")
            raise RuntimeError(
                f"{self.name} failed startup: {ready!r} "
                f"{err.decode(errors='replace')[-4000:]}"
            )
        self.ready_metadata = ready_fields[1:]
        asyncio.create_task(self._drain_stderr(proc))

    async def _drain_stderr(self, proc):
        while proc.stderr is not None:
            line = await proc.stderr.readline()
            if not line:
                return
            log.info("%s: %s", self.name, line.decode(errors="replace").rstrip())

    async def ask(self, line: str, timeout: int) -> str:
        async with self.lock:
            await self.start()
            proc = self.proc
            if proc is None or proc.stdin is None or proc.stdout is None:
                raise RuntimeError(f"{self.name} pipe unavailable")
            line = line.replace("\n", " ").replace("\r", " ")
            try:
                proc.stdin.write((line + "\n").encode())
                await proc.stdin.drain()
                response = await asyncio.wait_for(
                    proc.stdout.readline(), timeout=timeout
                )
            except asyncio.CancelledError:
                await self._kill_untrusted(proc, "request-cancelled")
                raise
            except asyncio.TimeoutError:
                await self._kill_untrusted(proc, "request-timeout")
                raise
            except Exception:
                await self._kill_untrusted(proc, "pipe-io-error")
                raise
            if not response:
                await self._kill_untrusted(proc, "unexpected-eof")
                raise RuntimeError(f"{self.name} exited")
            return response.decode(errors="replace").rstrip("\r\n")

    async def stop(self):
        proc = self.proc
        if proc is None:
            return
        self.proc = None
        self.ready_metadata = []
        try:
            if proc.stdin is not None:
                proc.stdin.write(b"QUIT\n")
                await proc.stdin.drain()
            await asyncio.wait_for(proc.wait(), timeout=5)
        except Exception:
            try:
                if proc.returncode is None:
                    proc.kill()
            except Exception:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except Exception:
                pass


class UnixLineDaemon:
    """Persistent line protocol over a systemd-owned Unix socket.

    The heavy translator process lives in meeting-translator.service, outside
    the meeting-server cgroup. A request timeout or socket failure therefore
    resets only this client connection; it can never kill or OOM the HTTP/mic
    server. The translator-side proxy consumes every native response even after
    a client disconnect, preserving strict request/response synchronization.
    """

    def __init__(self, path, name, *, connect_timeout=180):
        self.path = str(path)
        self.name = name
        self.connect_timeout = max(1, int(connect_timeout))
        self.reader = None
        self.writer = None
        self.lock = asyncio.Lock()
        self.reset_count = 0
        self.ready_metadata = []

    def is_ready(self):
        return bool(
            self.reader is not None
            and self.writer is not None
            and not self.writer.is_closing()
        )

    async def _reset_connection(self, reason, *, count=True):
        writer = self.writer
        self.reader = None
        self.writer = None
        self.ready_metadata = []
        if count:
            self.reset_count += 1
            log.warning("resetting %s connection after %s", self.name, reason)
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def start(self):
        if self.is_ready():
            return
        deadline = time.monotonic() + self.connect_timeout
        last_error = None
        while True:
            try:
                reader, writer = await asyncio.open_unix_connection(self.path)
                ready = await asyncio.wait_for(reader.readline(), timeout=5)
                ready_line = ready.decode(errors="replace").rstrip("\r\n")
                ready_fields = ready_line.split("\t") if ready_line else []
                if not ready_fields or ready_fields[0] != "READY":
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"{self.name} bad READY handshake: {ready_line!r}"
                    )
                self.reader = reader
                self.writer = writer
                self.ready_metadata = ready_fields[1:]
                log.info(
                    "connected %s socket=%s metadata=%s",
                    self.name,
                    self.path,
                    self.ready_metadata,
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"{self.name} socket unavailable: {last_error}"
                    ) from exc
                await asyncio.sleep(0.25)

    async def ask(self, line: str, timeout: int) -> str:
        async with self.lock:
            await self.start()
            reader = self.reader
            writer = self.writer
            if reader is None or writer is None:
                raise RuntimeError(f"{self.name} socket unavailable")
            line = line.replace("\n", " ").replace("\r", " ")
            try:
                writer.write((line + "\n").encode())
                await writer.drain()
                response = await asyncio.wait_for(
                    reader.readline(), timeout=timeout
                )
            except asyncio.CancelledError:
                await self._reset_connection("request-cancelled")
                raise
            except asyncio.TimeoutError:
                await self._reset_connection("request-timeout")
                raise
            except Exception:
                await self._reset_connection("socket-io-error")
                raise
            if not response:
                await self._reset_connection("unexpected-eof")
                raise RuntimeError(f"{self.name} socket closed")
            return response.decode(errors="replace").rstrip("\r\n")

    async def stop(self):
        await self._reset_connection("service-stop", count=False)


class Meeting:
    def __init__(self):
        self.clients = {}
        self.queue = asyncio.Queue(maxsize=max(0, QUEUE_CAPACITY))
        self.pending_gate = []
        self.pending_gate_lock = asyncio.Lock()
        self.history_cache = []
        self.turn_rows = {}
        self.turn_pending = defaultdict(int)
        self.turn_finalize_tasks = {}
        self.background_tasks = set()
        self.stopping = False
        self.critical_worker_failures = 0
        self.translation_worker_failures = 0
        # Reserve WebSocket handshakes that passed admission but have not yet
        # been installed in self.clients. ws.prepare() yields to the event loop,
        # so a burst of simultaneous handshakes could otherwise oversubscribe
        # MAX_CLIENTS before any handler records its client.
        self.client_admissions_inflight = 0

        self.started_monotonic = time.monotonic()
        self.worker_task = None
        self.gate_task = None

        # Only one ASR stack remains in v19.
        self.vit_stt = LineDaemon(["/usr/bin/meeting-vit-stt"], "meeting-vit-stt")
        self.vit_available = False
        self.stt_engine = "vit-only-dual-lane-v19.2.4.1"
        # Native sherpa inference is serialized and cannot be hard-preempted. Keep
        # at most one disposable LIVE request globally so multiple clients can
        # never queue a train of previews in front of authoritative FINAL work.
        self.vit_preview_global_task = None
        self.vit_preview_global_client_id = ""

        # V20 translation remains strictly post-ASR.
        # Canonical VIT text is never modified by translation.
        self.translation_model_root = Path(
            "/opt/meeting/models/translation/envit5"
        )

        translation_files = (
            "model.bin",
            "config.json",
            "spiece.model",
            "PROVENANCE.txt",
        )

        translation_models_present = all(
            (self.translation_model_root / name).is_file()
            for name in translation_files
        )

        self.translation_model_family = (
            "envit5-int8"
            if translation_models_present
            else "missing"
        )

        self.translator = UnixLineDaemon(
            TRANSLATOR_SOCKET_PATH,
            "meeting-translator",
            connect_timeout=TRANSLATOR_CONNECT_TIMEOUT_SECONDS,
        )
        self.translator_available = False
        self.translator_restart_failures = 0
        self.translator_retry_not_before = 0.0
        self.translation_restart_deferred = 0
        self.speaker_id = UnixLineDaemon(
            "/run/meeting-speaker-v20/speaker.sock",
            "meeting-speaker-id",
            connect_timeout=max(15, SPEAKER_TIMEOUT_SECONDS),
        )
        self.speaker_available = False
        self.speaker_requests = 0
        self.speaker_results = 0
        self.speaker_errors = 0
        self.speaker_skipped = 0

        self.translation_queue = asyncio.Queue(maxsize=16)
        self.translation_pending_keys = set()
        self.translation_active_keys = set()
        self.translation_task = None

        # Disposable LIVE translation has its own latest-wins queue. There is
        # only one native translator process, so LineDaemon.lock still serializes
        # native inference. FINAL work always suppresses starting new LIVE work.
        self.translation_live_queue = asyncio.Queue()
        self.translation_live_pending_keys = set()
        self.translation_live_active_keys = set()
        self.translation_live_latest = {}
        self.translation_live_sequence = 0
        self.translation_live_last_started_mono = 0.0
        self.translation_live_task = None
        self.translation_warm_task = None

        # v19.2.3: cache already translated stable chunks for a logical turn.
        # This prevents re-translating the complete cumulative transcript
        # after every accepted FINAL STT segment.
        self.translation_chunk_cache = {}

        # Telemetry.
        self.queue_high_watermark = 0
        self.gate_forwarded = 0
        self.gate_dropped = 0
        self.gate_groups = 0
        self.stream_frames = 0
        self.stream_speech_starts = 0
        self.stream_segments = 0
        self.stream_autocuts = 0
        self.stream_softcuts = 0
        self.stream_silence_ends = 0
        self.stream_noise_updates = 0
        self.pcm_rate_rejects = 0
        self.client_limit_rejects = 0
        self.protocol_state_rejects = 0
        self.logical_turns_started = 0
        self.logical_turns_continued = 0
        self.logical_turns_finalized = 0
        self.dropped_audio_seconds = 0.0

        self.vit_preview_requests = 0
        self.vit_preview_results = 0
        self.vit_preview_errors = 0
        self.vit_preview_skipped_busy = 0
        self.vit_preview_skipped_final_guard = 0
        self.vit_preview_skipped_global_busy = 0
        self.vit_preview_skipped_vit_locked = 0
        self.vit_preview_cancelled_for_final = 0
        self.capture_stopped_stt_unavailable = 0
        self.slow_client_evictions = 0
        self.vit_final_requests = 0
        self.vit_final_results = 0
        self.vit_final_errors = 0
        self.vit_final_corrections = 0
        self.vit_final_appends = 0
        self.vit_final_unresolved = 0
        self.vit_final_recovery_requests = 0
        self.final_wav_windows = 0
        self.retained_audio_segments = 0
        self.spool_orphans_recovered = 0
        self.unresolved_audio_segments_current = 0
        self.storage_pressure_events = 0

        self.translation_queued = 0
        self.translation_completed = 0
        self.translation_errors = 0
        self.translation_stale_skipped = 0
        self.translation_chunk_requests = 0
        self.translation_chunk_failures = 0
        self.translation_chunk_recoveries = 0
        self.translation_chunk_recovery_failures = 0
        self.translation_chunk_retry_suppressed = 0
        self.translation_identical_retry_skipped = 0
        self.translation_queue_coalesced = 0
        self.translation_token_limit_recoveries = 0
        self.translation_progressive_deferred = 0
        self.translation_stale_visible_updates = 0
        # HLMEET_PROGRESSIVE_FOLLOW_V19_7_4: useful lagging cumulative output remains visible while
        # the worker immediately follows the newest authoritative STT row.
        self.translation_progressive_debounce_reloads = 0
        self.translation_progressive_lagging_visible = 0
        self.translation_progressive_requeued = 0
        self.translation_progressive_final_preempted = 0
        # HLMEET_TRANSLATE_EFFICIENT_V19_7_5
        self.translation_progressive_chunk_preempted = 0
        self.translation_boundary_dedupes = 0
        self.translation_anchor_failures = 0
        self.translation_roundtrip_checks = 0
        self.translation_roundtrip_failures = 0
        self.translation_roundtrip_recoveries = 0
        self.translation_request_last_ms = 0
        self.translation_request_max_ms = 0
        self.translation_request_total_ms = 0
        self.translation_live_queued = 0
        self.translation_live_completed = 0
        self.translation_live_errors = 0
        self.translation_live_skipped_short = 0
        self.translation_live_skipped_busy = 0
        self.translation_live_skipped_unavailable = 0
        self.translation_live_coalesced = 0
        # v19.5: stale LIVE output is never visible. Keep the legacy counter in
        # health at zero for dashboard compatibility and count actual drops.
        self.translation_live_superseded_visible = 0
        self.translation_live_superseded_dropped = 0
        self.translation_live_invalidated_final = 0
        self.translation_live_source_trimmed = 0
        # HLMEET_TRANSLATE_EFFICIENT_V19_7_5: once cumulative translation exists, LIVE has no display
        # value on web v4.2 and must not spend the shared EnViT5 lane.
        self.translation_live_skipped_cumulative = 0
        self.translation_live_final_guard_dropped = 0
        # A disposable LIVE timeout resets only the Unix client connection.
        # It must never arm the expensive FINAL translator restart backoff.
        self.translation_live_request_timeouts = 0
        # HLMEET_LIVE_REVISION_SELFHEAL_V1
        # A disposable LIVE timeout invalidates only that preview/socket.
        # Later confirmed STT revisions may reconnect immediately; FINAL
        # remains authoritative and latest-wins prevents LIVE backlog.
        self.translation_live_reconnect_pending = False
        self.translation_live_reconnect_attempts = 0
        self.translation_live_reconnect_successes = 0
        self.translation_live_proxy_busy = 0

    def init_storage(self):
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        FINAL_SPOOL_DIR.mkdir(parents=True, exist_ok=True)
        UNRESOLVED_DIR.mkdir(parents=True, exist_ok=True)
        SPEAKER_TMP_DIR.mkdir(parents=True, exist_ok=True)

        for stale_speaker_wav in SPEAKER_TMP_DIR.glob("*.wav"):
            try:
                stale_speaker_wav.unlink()
            except Exception:
                log.exception(
                    "failed removing stale speaker WAV %s",
                    stale_speaker_wav,
                )

        # Runtime previews are disposable. FINAL audio is on persistent NVMe.
        # If power was lost while FINAL files were waiting in the in-memory
        # queue, quarantine the orphaned files instead of leaving them in the
        # active spool forever or deleting captured speech.
        for preview in PREVIEW_DIR.iterdir():
            if preview.is_file():
                try:
                    preview.unlink()
                except Exception:
                    log.exception("failed removing stale preview %s", preview)

        for orphan in FINAL_SPOOL_DIR.iterdir():
            if not orphan.is_file():
                continue
            try:
                # Atomic spool writes use .tmp until fsync completes. Preserve the
                # original audio extension when quarantining an interrupted write
                # so operators can still inspect/recover the captured PCM/WAV.
                logical_name = orphan.name[:-4] if orphan.name.endswith(".tmp") else orphan.name
                logical = Path(logical_name)
                reason = "startup-partial" if orphan.name.endswith(".tmp") else "startup-orphan"
                target = UNRESOLVED_DIR / (
                    logical.stem + f"-{reason}" + logical.suffix
                )
                if target.exists():
                    target = UNRESOLVED_DIR / (
                        logical.stem + f"-{reason}-" + uuid.uuid4().hex[:8]
                        + logical.suffix
                    )
                shutil.move(str(orphan), str(target))
                self.spool_orphans_recovered += 1
            except Exception:
                log.exception("failed quarantining spool orphan %s", orphan)

        self.unresolved_audio_segments_current = sum(
            1 for path in UNRESOLVED_DIR.glob("*.wav") if path.is_file()
        )

    async def start(self):
        self.init_storage()
        try:
            await self.vit_stt.start()
            self.vit_available = True
        except Exception:
            self.vit_available = False
            log.exception("VIT-STT worker failed startup")
            await self.vit_stt.stop()

        self.worker_task = asyncio.create_task(self._utterance_worker())
        self.gate_task = asyncio.create_task(self._pre_stt_gate_worker())
        self.translation_task = asyncio.create_task(
            self._translation_worker()
        )
        # V20 MVP is FINAL-only. Do not create the legacy LIVE worker when
        # live translation is disabled.
        if TRANSLATION_LIVE_PREVIEW_ENABLED:
            self.translation_live_task = asyncio.create_task(
                self._translation_live_worker()
            )
        else:
            self.translation_live_task = None
        # Do not make the HTTP/WebSocket server wait for a 3B CUDA model load.
        # STT/capture becomes usable immediately after authoritative VIT is ready;
        # translation warms independently and health exposes that state.
        if TRANSLATION_ENABLED:
            self.translation_warm_task = self._spawn(
                self._warm_translator()
            )
        # Only capture/STT-control workers are product-critical. Translation is
        # optional and must never restart HTTP/WebSocket/mic if its coordinator
        # exits unexpectedly; health exposes translation worker state separately.
        for name, task in (
            ("utterance-worker", self.worker_task),
            ("pre-stt-gate-worker", self.gate_task),
        ):
            task.add_done_callback(
                lambda completed, worker_name=name: self._critical_task_done(
                    worker_name, completed
                )
            )
        self.translation_task.add_done_callback(self._translation_task_done)
        log.info(
            "meeting-server %s ready; engine=%s vit=%s translator=%s",
            APP_VERSION,
            self.stt_engine,
            "ready" if self.vit_available else "unavailable",
            "warming" if TRANSLATION_ENABLED else "disabled",
        )

    async def stop(self):
        self.stopping = True
        if self.gate_task is not None:
            self.gate_task.cancel()
            try:
                await self.gate_task
            except asyncio.CancelledError:
                pass

        if self.worker_task is not None:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

        # LIVE previews are disposable and must not outlive shutdown or hold the
        # VIT line daemon while captured FINAL audio is being preserved.
        partials = []
        for client in list(self.clients.values()):
            task = client.partial_task
            client.partial_task = None
            if task is not None and not task.done():
                task.cancel()
                partials.append(task)
        if partials:
            await asyncio.gather(*partials, return_exceptions=True)

        # The queue itself is memory-only. Preserve every waiting FINAL file on
        # persistent storage before process exit so shutdown cannot orphan it.
        while True:
            try:
                queued = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            key = (queued.client_id, queued.turn_id)
            self._retain_audio(queued, "service-stop-queued")
            remaining = max(0, self.turn_pending.get(key, 0) - 1)
            if remaining:
                self.turn_pending[key] = remaining
            else:
                self.turn_pending.pop(key, None)
            self.queue.task_done()

        for task in list(self.turn_finalize_tasks.values()):
            task.cancel()
        if self.turn_finalize_tasks:
            await asyncio.gather(*self.turn_finalize_tasks.values(), return_exceptions=True)
        self.turn_finalize_tasks.clear()

        for task in list(self.background_tasks):
            task.cancel()
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        async with self.pending_gate_lock:
            pending = list(self.pending_gate)
            self.pending_gate.clear()
        for utterance in pending:
            self._retain_audio(utterance, "service-stop-pending")

        if self.translation_live_task is not None:
            self.translation_live_task.cancel()
            try:
                await self.translation_live_task
            except asyncio.CancelledError:
                pass

        if self.translation_task is not None:
            self.translation_task.cancel()
            try:
                await self.translation_task
            except asyncio.CancelledError:
                pass

        await self.translator.stop()
        await self.speaker_id.stop()
        await self.vit_stt.stop()

    def _spawn(self, coro):
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    def _critical_task_done(self, name, task):
        """Fail the service if a core infinite worker exits unexpectedly.

        A silently dead utterance/gate worker is worse than a bounded systemd
        restart: health could otherwise remain green while FINAL audio accumulates
        forever. SIGTERM lets aiohttp run normal cleanup before systemd restarts.
        """
        if self.stopping or task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        self.critical_worker_failures += 1
        if exc is None:
            log.critical("critical worker exited unexpectedly: %s", name)
        else:
            log.critical(
                "critical worker failed: %s: %s",
                name, exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception:
            log.exception("failed terminating after critical worker failure")

    def _translation_task_done(self, task):
        """Translation failure degrades translation only; never kill mic/HTTP/STT."""
        if self.stopping or task.cancelled():
            return
        self.translation_worker_failures += 1
        self.translator_available = False
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is None:
            log.error("translation worker exited unexpectedly")
        else:
            log.error(
                "translation worker failed without affecting capture: %s",
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    def _critical_workers_ready(self):
        tasks = (self.worker_task, self.gate_task)
        return all(task is not None and not task.done() for task in tasks)

    def _translation_worker_ready(self):
        return bool(
            self.translation_task is not None
            and not self.translation_task.done()
        )

    def _authoritative_stt_busy(self):
        return bool(
            self.queue.qsize() > 0
            or len(self.pending_gate) > 0
            or any(int(value or 0) > 0 for value in self.turn_pending.values())
        )

    @staticmethod
    def _reset_pcm_budget(client):
        client.pcm_budget_bytes = float(PCM_RATE_BURST_BYTES)
        client.pcm_budget_updated_mono = time.monotonic()

    def _consume_pcm_budget(self, client, byte_count):
        now = time.monotonic()
        elapsed = max(0.0, now - client.pcm_budget_updated_mono)
        client.pcm_budget_updated_mono = now
        refill = elapsed * BYTES_PER_SECOND * PCM_MAX_REALTIME_FACTOR
        client.pcm_budget_bytes = min(
            float(PCM_RATE_BURST_BYTES),
            max(0.0, client.pcm_budget_bytes) + refill,
        )
        if byte_count > client.pcm_budget_bytes:
            self.pcm_rate_rejects += 1
            return False
        client.pcm_budget_bytes -= byte_count
        return True

    def _preview_task_done(self, task, client_id):
        if self.vit_preview_global_task is task:
            self.vit_preview_global_task = None
            self.vit_preview_global_client_id = ""
        owner = self.clients.get(client_id)
        if owner is not None and owner.partial_task is task:
            owner.partial_task = None

    async def _cancel_partial_for_final(self, client):
        """Cancel the single global LIVE request before a FINAL is queued.

        Native sherpa inference itself is blocking and may finish after its HTTP
        client disconnects, so cancellation cannot promise hard preemption. The
        global admission invariant is what matters: no second/third disposable
        preview may already be waiting behind it.
        """
        tasks = []
        global_task = self.vit_preview_global_task
        if global_task is not None and not global_task.done():
            tasks.append(global_task)

        local_task = client.partial_task
        if (
            local_task is not None
            and not local_task.done()
            and local_task not in tasks
        ):
            tasks.append(local_task)

        if not tasks:
            return

        owner_id = self.vit_preview_global_client_id
        owner = self.clients.get(owner_id) if owner_id else None
        if owner is not None and owner.partial_task in tasks:
            owner.partial_task = None
        if client.partial_task in tasks:
            client.partial_task = None
        self.vit_preview_global_task = None
        self.vit_preview_global_client_id = ""

        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException) and not isinstance(
                result, asyncio.CancelledError
            ):
                log.warning(
                    "partial cancellation cleanup failed client=%s error=%s",
                    client.client_id,
                    result,
                )
        self.vit_preview_cancelled_for_final += len(tasks)

    def _translator_mark_ready(self):
        self.translator_available = True
        self.translator_restart_failures = 0
        self.translator_retry_not_before = 0.0

    def _translator_runtime_device(self):
        metadata = list(getattr(self.translator, "ready_metadata", []) or [])
        return str(metadata[0]).strip().lower() if metadata else ""

    def _translator_runtime_compute_type(self):
        metadata = list(getattr(self.translator, "ready_metadata", []) or [])
        return str(metadata[1]).strip().lower() if len(metadata) >= 2 else ""

    def _translator_is_cuda(self):
        return self._translator_runtime_device() == "cuda"

    def _translator_live_capable(self):
        """Allow disposable LIVE translation only on validated EnViT5 CPU."""
        return bool(
            self.translator_available
            and self.translation_model_family == "envit5-int8"
            and self._translator_runtime_device() == "cpu"
        )

    def _translator_warming(self):
        return bool(
            TRANSLATION_ENABLED
            and not self.translator_available
            and self.translation_warm_task is not None
            and not self.translation_warm_task.done()
        )

    async def _ensure_translator_ready(self, failure_reason):
        if self.translator_available and self.translator.is_ready():
            return True
        await self._wait_translator_backoff()
        try:
            # Serialize socket connection establishment with requests. The heavy
            # 3B process itself is owned/restarted by meeting-translator.service.
            async with self.translator.lock:
                await self.translator.start()
            self._translator_mark_ready()
            return True
        except asyncio.CancelledError:
            raise
        except Exception:
            self._translator_mark_failed(failure_reason)
            log.exception("translation worker failed to become ready")
            return False

    async def _warm_translator(self):
        if not TRANSLATION_ENABLED:
            return
        # Socket availability follows the isolated model lifecycle. Fail a short
        # connect attempt quickly, then keep retrying in this background task so
        # /api/health and mic capture never wait for MADLAD warm/restart.
        while not self.stopping and not self.translator_available:
            ready = await self._ensure_translator_ready("startup")
            if ready:
                log.info(
                    "translation warm ready device=%s compute=%s",
                    self._translator_runtime_device() or "unknown",
                    self._translator_runtime_compute_type() or "unknown",
                )
                return

    def _translator_mark_failed(self, reason):
        self.translator_available = False
        self.translator_restart_failures = min(16, self.translator_restart_failures + 1)
        exponent = min(6, self.translator_restart_failures - 1)
        delay = min(
            TRANSLATOR_RESTART_BACKOFF_MAX_SECONDS,
            TRANSLATOR_RESTART_BACKOFF_SECONDS * (2 ** exponent),
        )
        self.translator_retry_not_before = max(
            self.translator_retry_not_before, time.monotonic() + delay
        )
        log.warning(
            "translator unavailable reason=%s failures=%d retry_in_s=%d",
            reason, self.translator_restart_failures, delay,
        )

    async def _wait_translator_backoff(self):
        delay = self.translator_retry_not_before - time.monotonic()
        if delay > 0:
            self.translation_restart_deferred += 1
            await asyncio.sleep(delay)

    def _state_disk_free_bytes(self):
        try:
            return int(shutil.disk_usage(STATE_DIR).free)
        except Exception:
            return -1

    def _storage_ready(self, required_bytes=0):
        free_bytes = self._state_disk_free_bytes()
        if free_bytes < 0:
            return False
        return free_bytes >= STATE_MIN_FREE_BYTES + max(0, int(required_bytes))

    @staticmethod
    def _clean_text(text):
        return " ".join(str(text or "").replace("\x00", " ").split())

    @staticmethod
    def _words(text):
        return re.findall(r"[^\W_]+", str(text or "").casefold(), flags=re.UNICODE)

    @staticmethod
    def _word_matches(text):
        return list(re.finditer(r"[^\W_]+", str(text or ""), flags=re.UNICODE))

    @classmethod
    def _lexical_similarity(cls, left, right):
        """Word-level similarity used only for preview stability / diagnostics.

        v19.2 called this helper from the LIVE lane but never defined it, which
        turned every second overlapping preview hypothesis into an
        AttributeError. Keep it independent from FINAL arbitration.
        """
        left_words = cls._words(left)
        right_words = cls._words(right)
        if not left_words or not right_words:
            return 0.0
        return SequenceMatcher(
            None,
            left_words,
            right_words,
            autojunk=False,
        ).ratio()

    _VI_DIGITS = {
        "không": 0, "linh": None, "lẻ": None,
        "một": 1, "mốt": 1,
        "hai": 2,
        "ba": 3,
        "bốn": 4, "tư": 4,
        "năm": 5, "lăm": 5,
        "sáu": 6,
        "bảy": 7,
        "tám": 8,
        "chín": 9,
    }
    _VI_NUMBER_WORDS = set(_VI_DIGITS) | {
        "mười", "mươi", "trăm", "nghìn", "ngàn", "triệu", "tỷ", "phẩy",
    }

    @classmethod
    def _vi_parse_under_1000(cls, words):
        words = [w for w in words if w not in {"linh", "lẻ"}]
        if not words:
            return 0

        value = 0
        pos = 0

        # Hundreds.
        if "trăm" in words:
            idx = words.index("trăm")
            if idx == 0:
                hundreds = 1
            else:
                hundreds = cls._VI_DIGITS.get(words[idx - 1])
                if hundreds is None:
                    return None
            value += int(hundreds) * 100
            words = words[idx + 1:]
            pos = 0

        if not words:
            return value

        # Tens.
        if words[0] == "mười":
            value += 10
            pos = 1
        elif len(words) >= 2 and words[1] == "mươi":
            tens = cls._VI_DIGITS.get(words[0])
            if tens is None:
                return None
            value += int(tens) * 10
            pos = 2

        # Remaining unit. Be conservative: reject unknown structures instead of
        # silently changing transcript content.
        remaining = [
            w for w in words[pos:]
            if w not in {"linh", "lẻ"}
        ]
        if remaining:
            if len(remaining) != 1:
                return None
            unit = cls._VI_DIGITS.get(remaining[0])
            if unit is None:
                return None
            value += int(unit)
        return value

    @classmethod
    def _vi_parse_integer_words(cls, words):
        if not words:
            return None
        scales = {
            "tỷ": 1_000_000_000,
            "triệu": 1_000_000,
            "nghìn": 1_000,
            "ngàn": 1_000,
        }
        total = 0
        group = []
        last_scale = float("inf")
        saw_number = False

        for word in words:
            if word in scales:
                scale = scales[word]
                if scale >= last_scale:
                    return None
                group_value = cls._vi_parse_under_1000(group)
                if group_value is None:
                    return None
                if not group:
                    group_value = 1
                total += group_value * scale
                group = []
                last_scale = scale
                saw_number = True
            else:
                if word not in cls._VI_NUMBER_WORDS or word == "phẩy":
                    return None
                group.append(word)
                saw_number = True

        group_value = cls._vi_parse_under_1000(group)
        if group_value is None:
            return None
        total += group_value
        return total if saw_number else None

    @classmethod
    def _vi_parse_number_words(cls, words):
        if not words:
            return None
        if words.count("phẩy") > 1:
            return None
        if "phẩy" not in words:
            value = cls._vi_parse_integer_words(words)
            return str(value) if value is not None else None

        idx = words.index("phẩy")
        integer = cls._vi_parse_integer_words(words[:idx])
        decimals = words[idx + 1:]
        if integer is None or not decimals:
            return None

        digits = []
        for word in decimals:
            digit = cls._VI_DIGITS.get(word)
            if digit is None:
                return None
            digits.append(str(digit))
        return f"{integer}." + "".join(digits)

    @classmethod
    def _vi_itn_percent(cls, text):
        """Conservative ITN: only number words directly before 'phần trăm'.

        This intentionally does not normalize arbitrary Vietnamese number words;
        phrases such as "một trong những" must remain lexical text.
        """
        clean = cls._clean_text(text)
        if not clean or not VI_ITN_PERCENT_ENABLED:
            return clean

        tokens = clean.split()
        out = []
        i = 0
        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and tokens[i].casefold() == "phần"
                and tokens[i + 1].casefold() == "trăm"
            ):
                # Search only the contiguous numeric phrase immediately before
                # "phần trăm", bounded to avoid consuming normal prose.
                start = len(out)
                while (
                    start > 0
                    and len(out) - start < 12
                    and out[start - 1].casefold() in cls._VI_NUMBER_WORDS
                ):
                    start -= 1

                numeric_words = [w.casefold() for w in out[start:]]
                rendered = cls._vi_parse_number_words(numeric_words)
                if rendered is not None and numeric_words:
                    del out[start:]
                    out.append(rendered + "%")
                    i += 2
                    continue

                # ASR may already emit Arabic numerals even when it spells out
                # the percent unit.
                if out and re.fullmatch(r"\d+(?:[.,]\d+)?", out[-1]):
                    out[-1] = out[-1].replace(",", ".") + "%"
                    i += 2
                    continue

            out.append(tokens[i])
            i += 1

        return cls._clean_text(" ".join(out))

    @classmethod
    def _vi_itn_common(cls, text):
        """Conservative Vietnamese ITN for high-confidence numeric contexts.

        Canonical ASR text is never changed. Only presentation text is
        normalized, and only where surrounding unit/label words make a numeric
        interpretation substantially less ambiguous.
        """
        clean = cls._clean_text(text)
        if not clean or not VI_ITN_COMMON_ENABLED:
            return clean

        tokens = clean.split()
        if not tokens:
            return clean

        def numeric_span(start, max_words=10):
            end = start
            while (
                end < len(tokens)
                and end - start < max_words
                and tokens[end].casefold() in cls._VI_NUMBER_WORDS
                and tokens[end].casefold() != "phẩy"
            ):
                end += 1
            best = None
            for candidate_end in range(end, start, -1):
                value = cls._vi_parse_integer_words(
                    [w.casefold() for w in tokens[start:candidate_end]]
                )
                if value is not None:
                    best = (candidate_end, value)
                    break
            return best

        out = []
        i = 0
        count_units = {
            "tuổi", "năm", "tháng", "ngày", "giờ", "phút",
            "giây", "tỉnh", "chữ", "lần", "người", "điều",
        }
        labels_before_number = {
            "khu", "điều", "chương", "mục", "phần", "số",
        }

        while i < len(tokens):
            folded = tokens[i].casefold()

            # Spoken year: "năm hai nghìn mười tám" -> "năm 2018".
            # Restrict to a normal modern/historical year range.
            if folded == "năm" and i + 1 < len(tokens):
                parsed = numeric_span(i + 1, 8)
                if parsed is not None:
                    end, value = parsed
                    if 1000 <= value <= 2099:
                        out.append(tokens[i])
                        out.append(str(value))
                        i = end
                        continue

            # Numbered labels: "khu mười hai" -> "khu 12".
            if folded in labels_before_number and i + 1 < len(tokens):
                parsed = numeric_span(i + 1, 8)
                if parsed is not None:
                    end, value = parsed
                    out.append(tokens[i])
                    out.append(str(value))
                    i = end
                    continue

            # Numeric phrase immediately before an unambiguous counting unit.
            # Search the unit boundary explicitly because Vietnamese "năm" can
            # mean both digit five and the unit year.
            if folded in cls._VI_NUMBER_WORDS and folded != "phẩy":
                unit_match = None
                upper = min(len(tokens), i + 11)
                for unit_idx in range(i + 1, upper):
                    if tokens[unit_idx].casefold() not in count_units:
                        continue
                    value = cls._vi_parse_integer_words(
                        [w.casefold() for w in tokens[i:unit_idx]]
                    )
                    if value is not None:
                        unit_match = (unit_idx, value)
                        break
                if unit_match is not None:
                    unit_idx, value = unit_match
                    out.append(str(value))
                    out.append(tokens[unit_idx])
                    i = unit_idx + 1
                    continue

            out.append(tokens[i])
            i += 1

        return cls._clean_text(" ".join(out))

    @classmethod
    def _vi_written_form(cls, text, *, is_final=False):
        """Presentation-only VI written form.

        Canonical ASR text is never modified by this function. This is a
        deliberately conservative fallback for the Yocto appliance, where the
        locked ViBERT-CaPU model/worker is not currently packaged. It restores
        high-confidence punctuation/case and percentage ITN after rolling merge.
        """
        clean = cls._clean_text(text)
        if not clean or not VI_WRITTEN_FORM_ENABLED:
            return clean

        clean = cls._vi_itn_percent(clean)
        clean = cls._vi_itn_common(clean)

        # High-confidence discourse boundaries. These are presentation hints,
        # not ASR evidence and never feed rolling merge.
        sentence_markers = (
            "hơn thế nữa",
            "chưa kể đến",
            "tuy nhiên",
            "mặt khác",
            "do đó",
            "vì vậy",
            "đấy là",
        )
        comma_markers = (
            "nhưng",
            "cho nên",
            "đồng thời",
            "nhất là",
            "đặc biệt là",
            "trong khi",
        )

        for marker in sentence_markers:
            clean = re.sub(
                rf"\s+({re.escape(marker)})\s+",
                r". \1 ",
                clean,
                flags=re.IGNORECASE,
            )
        for marker in comma_markers:
            clean = re.sub(
                rf"\s+({re.escape(marker)})\s+",
                r", \1 ",
                clean,
                flags=re.IGNORECASE,
            )

        clean = re.sub(r"\s+([,.;:!?%])", r"\1", clean)
        clean = re.sub(r"([,;:!?])(?=\S)", r"\1 ", clean)
        clean = cls._clean_text(clean)

        # A small proper-name pass is safe for common product-domain names and
        # materially improves Vietnamese readability without touching canonical.
        proper_names = {
            "việt nam": "Việt Nam",
            "hồ chí minh": "Hồ Chí Minh",
        }
        for source, target in proper_names.items():
            clean = re.sub(
                rf"\b{re.escape(source)}\b",
                target,
                clean,
                flags=re.IGNORECASE,
            )

        # Capitalize sentence starts without lowercasing any ASR token.
        chars = list(clean)
        capitalize_next = True
        for idx, ch in enumerate(chars):
            if capitalize_next and ch.isalpha():
                chars[idx] = ch.upper()
                capitalize_next = False
            if ch in ".!?":
                capitalize_next = True
        clean = "".join(chars)

        if is_final:
            clean = clean.rstrip(" ,;:")
            question_patterns = (
                r"\bcó\b.{0,80}\bkhông$",
                r"\bphải không$",
                r"\bđúng không$",
                r"\bđược không$",
                r"\bthế nào$",
                r"\btại sao$",
                r"\bvì sao$",
                r"\bbao nhiêu$",
                r"\bở đâu$",
                r"\bkhi nào$",
                r"\bai$",
            )
            folded = clean.casefold()
            terminal = "?" if any(
                re.search(pattern, folded)
                for pattern in question_patterns
            ) else "."
            if not clean.endswith((".", "?", "!")):
                clean += terminal

        return cls._clean_text(clean)

    @classmethod
    def _vi_translation_form(cls, text, *, is_final=False):
        """Vietnamese source form used only by machine translation.

        This deliberately does NOT run numeric ITN.  A display-only heuristic
        must never be allowed to change the semantic source seen by NMT.  We do
        restore conservative punctuation/case because sentence boundaries are
        useful context for document-level translation.
        """
        clean = cls._clean_text(text)
        if not clean:
            return clean

        sentence_markers = (
            "hơn thế nữa", "chưa kể đến", "tuy nhiên", "mặt khác",
            "do đó", "vì vậy", "đấy là",
        )
        comma_markers = (
            "nhưng", "cho nên", "đồng thời", "nhất là", "đặc biệt là",
            "trong khi",
        )
        for marker in sentence_markers:
            clean = re.sub(
                rf"\s+({re.escape(marker)})\s+",
                r". \1 ",
                clean,
                flags=re.IGNORECASE,
            )
        for marker in comma_markers:
            clean = re.sub(
                rf"\s+({re.escape(marker)})\s+",
                r", \1 ",
                clean,
                flags=re.IGNORECASE,
            )

        clean = re.sub(r"\s+([,.;:!?])", r"\1", clean)
        clean = re.sub(r"([,;:!?])(?=\S)", r"\1 ", clean)
        clean = cls._clean_text(clean)

        # Casing only: no lexical replacement and no number rewriting.
        for source, target in {
            "việt nam": "Việt Nam",
            "hồ chí minh": "Hồ Chí Minh",
        }.items():
            clean = re.sub(
                rf"\b{re.escape(source)}\b",
                target,
                clean,
                flags=re.IGNORECASE,
            )

        chars = list(clean)
        capitalize_next = True
        for idx, ch in enumerate(chars):
            if capitalize_next and ch.isalpha():
                chars[idx] = ch.upper()
                capitalize_next = False
            if ch in ".!?":
                capitalize_next = True
        clean = "".join(chars)

        if is_final:
            clean = clean.rstrip(" ,;:")
            folded = clean.casefold()
            question_patterns = (
                r"\bcó\b.{0,80}\bkhông$", r"\bphải không$",
                r"\bđúng không$", r"\bđược không$", r"\bthế nào$",
                r"\btại sao$", r"\bvì sao$", r"\bbao nhiêu$",
                r"\bở đâu$", r"\bkhi nào$", r"\bai$",
            )
            terminal = "?" if any(
                re.search(pattern, folded) for pattern in question_patterns
            ) else "."
            if not clean.endswith((".", "?", "!")):
                clean += terminal

        return cls._clean_text(clean)

    @classmethod
    def _translation_input_text(cls, canonical, language, *, is_final=False):
        canonical = cls._clean_text(canonical)
        if language == "vi":
            return cls._vi_translation_form(canonical, is_final=is_final)
        return canonical

    @classmethod
    def _display_text(cls, canonical, language, *, is_final=False):
        canonical = cls._clean_text(canonical)
        if language == "vi":
            return cls._vi_written_form(canonical, is_final=is_final)
        return canonical

    @staticmethod
    def _has_disallowed_script(text):
        for ch in str(text or ""):
            cp = ord(ch)
            if (
                0x3040 <= cp <= 0x30FF
                or 0x3400 <= cp <= 0x4DBF
                or 0x4E00 <= cp <= 0x9FFF
                or 0xF900 <= cp <= 0xFAFF
                or 0x1100 <= cp <= 0x11FF
                or 0x3130 <= cp <= 0x318F
                or 0xAC00 <= cp <= 0xD7AF
                or 0x0400 <= cp <= 0x052F
                or 0x0600 <= cp <= 0x06FF
            ):
                return True
        return False

    @classmethod
    def _quality_reason(cls, text, duration_ms=None, *, reject_repetition=True):
        text = cls._clean_text(text)
        if not text:
            return "empty"
        if cls._has_disallowed_script(text):
            return "unsupported-script"
        if sum(1 for ch in text if ch.isalpha()) < 2:
            return "no-lexical-content"

        words = cls._words(text)
        if len(words) >= 3:
            run = 1
            for i in range(1, len(words)):
                if words[i] == words[i - 1]:
                    run += 1
                    if run >= 3 and reject_repetition:
                        return "consecutive-word-loop"
                else:
                    run = 1

        if len(words) >= 6:
            for i in range(len(words) - 5):
                if (
                    reject_repetition
                    and words[i:i + 2]
                    == words[i + 2:i + 4]
                    == words[i + 4:i + 6]
                ):
                    return "repeated-bigram"

        # Detect only contiguous repetition loops.
        #
        # Do NOT reject merely because an n-gram occurs many times
        # at different positions in legitimate speech.
        #
        # 1-word loops are handled above by consecutive-word-loop.
        # 2-word loops are handled above by repeated-bigram.
        # Here we detect contiguous blocks of 3..8 words repeated
        # three times consecutively.
        if reject_repetition and len(words) >= 9:
            max_span = min(8, len(words) // 3)
            for span in range(3, max_span + 1):
                for i in range(len(words) - (3 * span) + 1):
                    block = words[i:i + span]
                    if (
                        block == words[i + span:i + (2 * span)]
                        and
                        block == words[i + (2 * span):i + (3 * span)]
                    ):
                        return "repeated-trigram"

        if duration_ms is not None:
            duration_s = max(0.45, float(duration_ms) / 1000.0)
            budget = max(MIN_TRANSCRIPT_CHAR_BUDGET, int(duration_s * MAX_TRANSCRIPT_CHAR_RATE))
            if len(text) > budget:
                return "implausible-length"
            if len(words) >= 8 and len(words) / duration_s > 7.0:
                return "implausible-word-rate"
            if duration_s >= 8.0 and len(text) <= 18 and len(words) <= 4:
                return "sparse-long-audio"
            if duration_s >= 12.0 and len(text) <= 26 and len(words) <= 5:
                return "sparse-very-long-audio"
        return None

    @classmethod
    def _parse_vit_reply(cls, reply):
        out = {"ok": False, "language": "", "text": "", "reason": "invalid-reply"}
        if not isinstance(reply, str):
            return out
        if reply.startswith("ERR\t"):
            out["reason"] = reply.split("\t", 1)[1] or "worker-error"
            return out
        if reply.startswith("SKIP\t"):
            out["reason"] = reply.split("\t", 1)[1] or "worker-skip"
            return out
        if reply.startswith("OK\t"):
            parts = reply.split("\t", 3)
            if len(parts) >= 3:
                out["language"] = parts[1].strip().lower()
                out["text"] = cls._clean_text(parts[2])
                out["ok"] = bool(out["text"])
                out["reason"] = None if out["ok"] else "empty"
                if len(parts) >= 4:
                    out["decode_mode"] = parts[3].strip().lower()
            return out
        # Legacy bridge protocol.
        if "\t" in reply:
            language, text = reply.split("\t", 1)
            out["language"] = language.strip().lower()
            out["text"] = cls._clean_text(text)
            out["ok"] = bool(out["text"])
            out["reason"] = None if out["ok"] else "empty"
        return out

    @staticmethod
    def _pcm_stats(data):
        n = len(data) // 2
        if n <= 0:
            return 0.0, 0.0
        total = 0.0
        peak = 0.0
        view = memoryview(data)[: n * 2].cast("h")
        for sample in view:
            value = float(sample) / 32768.0
            total += value * value
            peak = max(peak, abs(value))
        return math.sqrt(total / n), peak

    @staticmethod
    def _atomic_write_bytes(path: Path, payload: bytes):
        """Durably publish one spool file without exposing partial contents."""
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            try:
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                directory_fd = os.open(str(path.parent), flags)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        except Exception:
            # Keep a successfully fsynced .tmp on persistent storage for startup
            # quarantine instead of pretending the captured audio was committed.
            raise

    @classmethod
    def _write_wav(cls, path: Path, pcm: bytes):
        if len(pcm) & 1:
            pcm = pcm[:-1]
        data_size = len(pcm)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            1,
            SAMPLE_RATE,
            BYTES_PER_SECOND,
            SAMPLE_BYTES,
            16,
            b"data",
            data_size,
        )
        cls._atomic_write_bytes(path, header + pcm)

    @classmethod
    def _materialize_final_audio(cls, raw_path, wav_path, novel_pcm, final_pcm):
        """Durably materialize a paired FINAL PCM/WAV set on NVMe."""
        raw_tmp = raw_path.with_name(raw_path.name + ".tmp")
        wav_tmp = wav_path.with_name(wav_path.name + ".tmp")
        try:
            wav_payload_pcm = final_pcm[:-1] if len(final_pcm) & 1 else final_pcm
            data_size = len(wav_payload_pcm)
            wav_header = struct.pack(
                "<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + data_size, b"WAVE",
                b"fmt ", 16, 1, 1, SAMPLE_RATE, BYTES_PER_SECOND, SAMPLE_BYTES,
                16, b"data", data_size,
            )
            for tmp, payload in (
                (raw_tmp, novel_pcm),
                (wav_tmp, wav_header + wav_payload_pcm),
            ):
                with open(tmp, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            # Both payloads are complete before either final filename is visible.
            os.replace(raw_tmp, raw_path)
            os.replace(wav_tmp, wav_path)
            try:
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                directory_fd = os.open(str(raw_path.parent), flags)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                # Some filesystems do not support directory fsync. File fsync +
                # atomic rename still prevents readers from seeing partial bytes.
                pass
        except Exception:
            # Leave any fsynced temporary/committed member in the persistent spool.
            # init_storage() quarantines it on the next start; captured speech is
            # never silently deleted after an interrupted commit.
            raise

    def _sparse_preview_reason(self, utterance, text):
        # Phase6B verbatim-first: reject only an actually empty hypothesis.
        # Valid one-word/two-word FINAL text must not depend on LIVE corroboration.
        if not self._words(text):
            return "empty-transcript"
        return None

    def _learn_noise_from_rejected(self, utterance, reason):
        if not STREAM_NOISE_LEARN_ENABLED:
            return
        if int(getattr(utterance, "audio_ms", 0) or 0) < STREAM_NOISE_LEARN_MIN_MS:
            return
        if str(reason or "") not in {
            "empty", "empty-transcript", "vit-final-empty",
            "sparse-long-audio", "sparse-very-long-audio",
            "sparse-unconfirmed-preview",
            "sparse-preview-mismatch",
        }:
            return
        client = self.clients.get(utterance.client_id)
        if client is None:
            return
        target = min(
            STREAM_NOISE_MAX_FLOOR,
            max(0.0010, float(utterance.avg_rms) * STREAM_NOISE_REJECT_BLEND),
        )
        if target > client.noise_floor and target > client.pending_noise_floor:
            client.pending_noise_floor = target
            log.info(
                "noise-learn scheduled client=%s reason=%s avg=%.6f current=%.6f target=%.6f",
                utterance.client_id, reason, float(utterance.avg_rms),
                float(client.noise_floor), float(target),
            )

    def _retain_audio(self, utterance, reason):
        UNRESOLVED_DIR.mkdir(parents=True, exist_ok=True)
        safe_reason = re.sub(r"[^A-Za-z0-9_-]", "-", str(reason or "unresolved"))[:32]
        retained = False
        retained_wav = False
        for source in (utterance.raw_path, utterance.wav_path):
            try:
                if source.exists():
                    target = UNRESOLVED_DIR / f"{utterance.utterance_id}-{safe_reason}{source.suffix}"
                    if target.exists():
                        target = UNRESOLVED_DIR / (
                            f"{utterance.utterance_id}-{safe_reason}-"
                            f"{uuid.uuid4().hex[:8]}{source.suffix}"
                        )
                    shutil.move(str(source), str(target))
                    retained = True
                    retained_wav = retained_wav or source.suffix == ".wav"
            except Exception:
                log.exception("failed retaining %s", source)
        if retained:
            self.retained_audio_segments += 1
        if retained_wav:
            self.unresolved_audio_segments_current += 1
        utterance.retain_audio = retained
        return retained

    @staticmethod
    def _segment_id(turn_id, index):
        base = safe_utterance_id(turn_id)
        suffix = f"-s{index + 1}"
        return (base[: max(1, 64 - len(suffix))] + suffix)[:64]

    def _reset_segment_stats(self, client, pcm=b""):
        client.audio_frames = 0
        client.audio_rms_sum = 0.0
        client.peak_rms = 0.0
        client.speech_like_frames = 0
        if pcm:
            rms, peak = self._pcm_stats(pcm)
            frame_bytes = max(2, BYTES_PER_SECOND * STREAM_FRAME_MS // 1000)
            frames = max(1, len(pcm) // frame_bytes)
            client.audio_frames = frames
            client.audio_rms_sum = rms * frames
            client.peak_rms = peak
            client.speech_like_frames = frames

    def _cancel_turn_finalize(self, client_id, turn_id):
        key = (client_id, turn_id)
        task = self.turn_finalize_tasks.pop(key, None)
        if task is not None:
            task.cancel()

    def _reset_logical_turn(self, client):
        if client.turn_id:
            self.translation_live_latest.pop(
                (client.client_id, client.turn_id), None
            )
        client.turn_id = ""
        client.segment_index = 0
        client.logical_context_audio.clear()
        client.logical_last_end_mono = 0.0
        client.segment_leading_overlap_bytes = 0

    def _begin_segment(self, client, *, turn_id=None, pre_roll=None):
        now_mono = time.monotonic()
        now_ms = int(time.time() * 1000)

        if turn_id:
            explicit = safe_utterance_id(turn_id)
            if client.turn_id != explicit:
                self._reset_logical_turn(client)
                client.turn_id = explicit
                self.logical_turns_started += 1
        else:
            continue_turn = (
                bool(client.turn_id)
                and client.logical_last_end_mono > 0.0
                and (now_mono - client.logical_last_end_mono) * 1000.0 <= LOGICAL_TURN_GAP_MS
            )
            if continue_turn:
                self._cancel_turn_finalize(client.client_id, client.turn_id)
                self.logical_turns_continued += 1
            else:
                self._reset_logical_turn(client)
                client.turn_id = safe_utterance_id(uuid.uuid4().hex)
                self.logical_turns_started += 1

        client.current_id = self._segment_id(client.turn_id, client.segment_index)
        client.segment_language_pref = (
            client.language_pref
            if client.language_pref in SUPPORTED_LANGUAGES
            else "vi"
        )
        client.current_audio.clear()
        client.segment_leading_overlap_bytes = 0
        client.partial_last_bytes = 0
        client.partial_revision = 0
        client.partial_last_text = ""
        client.partial_candidate_text = ""
        client.partial_candidate_hits = 0
        client.partial_confirmed_text = ""
        self._reset_segment_stats(client)

        buffered = list(pre_roll or [])
        if buffered:
            duration_ms = len(buffered) * STREAM_FRAME_MS
            client.speech_started_ms = now_ms - duration_ms
            client.speech_started_mono = now_mono - duration_ms / 1000.0
            for pcm, rms, peak, _ in buffered:
                client.current_audio.extend(pcm)
                client.audio_frames += 1
                client.audio_rms_sum += rms
                client.peak_rms = max(client.peak_rms, peak)
                threshold = max(STREAM_MIN_THRESHOLD, client.noise_floor * STREAM_NOISE_MULTIPLIER)
                if rms >= threshold and peak >= max(STREAM_MIN_PEAK, threshold * 1.45):
                    client.speech_like_frames += 1
        else:
            client.speech_started_ms = now_ms
            client.speech_started_mono = now_mono

        client.mic_state = "speaking"
        self.stream_speech_starts += 1

    async def _append_active_frame(self, client, data, *, rms=None, peak=None, speech_like=None):
        if not client.current_id:
            return
        if rms is None or peak is None:
            rms, peak = self._pcm_stats(data)
        client.current_audio.extend(data)
        client.audio_frames += 1
        client.audio_rms_sum += rms
        client.peak_rms = max(client.peak_rms, peak)
        if speech_like is None:
            threshold = max(STREAM_MIN_THRESHOLD, client.noise_floor * STREAM_NOISE_MULTIPLIER)
            speech_like = rms >= threshold and peak >= max(STREAM_MIN_PEAK, threshold * 1.45)
        if speech_like:
            client.speech_like_frames += 1
            # Never start fresh disposable inference on a silence frame. Native
            # blocking inference cannot be force-preempted by closing HTTP.
            self.maybe_schedule_partial(client)

        if len(client.current_audio) >= SEGMENT_HARD_BYTES:
            self.stream_autocuts += 1
            await self._flush_client_segment(client, keep_overlap=True, end_reason="hard-cut")

    def maybe_schedule_partial(self, client):
        if not PARTIAL_ENABLED or not self.vit_available:
            return
        if client.segment_language_pref not in SUPPORTED_LANGUAGES:
            return
        if not client.current_id:
            return
        audio_bytes = len(client.current_audio)
        if audio_bytes < PARTIAL_MIN_BYTES:
            return
        if client.partial_last_bytes and audio_bytes - client.partial_last_bytes < PARTIAL_INTERVAL_BYTES:
            return
        if client.partial_task is not None and not client.partial_task.done():
            return
        # There is one non-preemptible native recognizer lane. Never let multiple
        # clients build a FIFO of disposable LIVE requests ahead of FINAL work.
        global_task = self.vit_preview_global_task
        if global_task is not None and not global_task.done():
            self.vit_preview_skipped_global_busy += 1
            return
        # FINAL speech always wins. Do not begin a disposable preview while final
        # audio is waiting or the cross-talk gate is holding completed segments.
        if self.queue.qsize() > 0 or len(self.pending_gate) > 0:
            self.vit_preview_skipped_busy += 1
            return
        # The queue may already be empty because the FINAL worker has taken its
        # item. The line lock is therefore the authoritative "VIT busy" signal.
        if self.vit_stt.lock.locked():
            self.vit_preview_skipped_vit_locked += 1
            return
        # Avoid launching a non-preemptible native LIVE decode immediately before
        # a deterministic hard-cut FINAL, and once VAD has started a silence tail.
        if (
            client.vad_silent_frames > 0
            or SEGMENT_HARD_BYTES - audio_bytes <= PARTIAL_INTERVAL_BYTES
        ):
            self.vit_preview_skipped_final_guard += 1
            return

        pcm = bytes(client.current_audio[-PARTIAL_WINDOW_BYTES:])
        client.partial_last_bytes = audio_bytes
        client.partial_revision += 1
        revision = client.partial_revision
        task = self._spawn(
            self._run_partial(
                client_id=client.client_id,
                room=client.room,
                speaker=client.name,
                turn_id=client.turn_id,
                utterance_id=client.current_id,
                language=client.segment_language_pref,
                revision=revision,
                pcm=pcm,
            )
        )
        client.partial_task = task
        self.vit_preview_global_task = task
        self.vit_preview_global_client_id = client.client_id
        task.add_done_callback(
            lambda completed, owner_id=client.client_id: self._preview_task_done(
                completed, owner_id
            )
        )

    async def _run_partial(self, *, client_id, room, speaker, turn_id, utterance_id, language, revision, pcm):
        path = PREVIEW_DIR / f"preview-{client_id}-{uuid.uuid4().hex}.pcm"
        try:
            if self.queue.qsize() > 0 or len(self.pending_gate) > 0:
                self.vit_preview_skipped_busy += 1
                return
            path.write_bytes(pcm)
            self.vit_preview_requests += 1
            reply = await self.vit_stt.ask(
                f"PCM\t{path}\t{language}\t{client_id}",
                timeout=PARTIAL_TIMEOUT_SECONDS,
            )
            candidate = self._parse_vit_reply(reply)
            if not candidate.get("ok"):
                return
            text = self._clean_text(candidate.get("text"))
            duration_ms = len(pcm) * 1000 // BYTES_PER_SECOND
            if candidate.get("language") not in SUPPORTED_LANGUAGES:
                return
            if self._quality_reason(text, duration_ms=duration_ms) is not None:
                return

            client = self.clients.get(client_id)
            if client is None or client.current_id != utterance_id or client.turn_id != turn_id:
                return
            # v19.2.1: one LIVE decode is not enough evidence.
            # Require two similar overlapping PCM hypotheses.
            if client.partial_candidate_text:
                similarity = self._lexical_similarity(
                    client.partial_candidate_text,
                    text,
                )

                if similarity >= 0.72:
                    client.partial_candidate_hits += 1
                else:
                    client.partial_candidate_text = text
                    client.partial_candidate_hits = 1
                    return
            else:
                client.partial_candidate_text = text
                client.partial_candidate_hits = 1
                return

            if client.partial_candidate_hits < 2:
                return

            client.partial_confirmed_text = text

            if text == client.partial_last_text:
                return

            client.partial_last_text = text
            self.vit_preview_results += 1
            await self._queue_live_translation(
                client_id=client_id,
                room=room,
                turn_id=turn_id,
                utterance_id=utterance_id,
                source_language=candidate.get("language") or language,
                source_text=text,
                revision=revision,
            )
            await self.broadcast(
                room,
                {
                    "type": "transcript_partial",
                    "id": turn_id,
                    "utterance_id": utterance_id,
                    "turn_id": turn_id,
                    "room": room,
                    "client_id": client_id,
                    "speaker": speaker,
                    "source_language": candidate.get("language") or language,
                    "text": text,
                    "translation": "",
                    "revision": revision,
                    "is_final": False,
                    "asr_source": "vit-stt-live-pcm",
                    "audio_ms": duration_ms,
                    "created_at": time.time(),
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self.vit_preview_errors += 1
            # LIVE is disposable. A bridge timeout/cancel resets that child, but
            # it is not proof that the authoritative native VIT backend is down.
            # Only FINAL failure is allowed to change global VIT availability.
            log.exception("VIT live preview failed turn=%s utterance=%s", turn_id, utterance_id)
        finally:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    async def _flush_client_segment(self, client, *, keep_overlap=False, end_reason="segment"):
        if not client.current_id:
            return False

        # A LIVE preview is disposable. Do not let it occupy the serialized VIT
        # bridge while authoritative FINAL audio is being queued. Cancellation
        # resets only the tiny line bridge; the native stt-http service remains.
        await self._cancel_partial_for_final(client)

        pcm = bytes(client.current_audio)
        if len(pcm) < MIN_UTTERANCE_BYTES:
            client.current_audio.clear()
            client.current_id = None
            self._reset_segment_stats(client)
            return False

        ended_mono = time.monotonic()
        ended_ms = int(time.time() * 1000)
        frames = max(1, client.audio_frames)
        avg_rms = client.audio_rms_sum / frames
        peak_rms = client.peak_rms
        speech_ratio = min(1.0, max(0.0, client.speech_like_frames / frames))

        leading = min(client.segment_leading_overlap_bytes, len(pcm))
        novel_pcm = pcm[leading:]
        if not novel_pcm:
            novel_pcm = pcm
            leading = 0

        # Build the FINAL WAV from bounded left acoustic context +
        # current novel speech. If the total exceeds the configured window, only
        # old context is trimmed; current captured speech is preserved.
        left_context = bytes(client.logical_context_audio[-FINAL_LEFT_CONTEXT_BYTES:])
        max_context = max(0, FINAL_MAX_WINDOW_BYTES - len(novel_pcm))
        if len(left_context) > max_context:
            left_context = left_context[-max_context:] if max_context > 0 else b""
        final_pcm = left_context + novel_pcm
        context_prefix_ms = len(left_context) * 1000 // BYTES_PER_SECOND

        spool_id = f"final-{client.client_id}-{uuid.uuid4().hex}"
        raw_path = FINAL_SPOOL_DIR / f"{spool_id}.pcm"
        wav_path = FINAL_SPOOL_DIR / f"{spool_id}.wav"
        required_bytes = len(novel_pcm) + len(final_pcm) + 4096
        storage_pressure = not self._storage_ready(required_bytes)
        free_bytes = self._state_disk_free_bytes()
        if free_bytes >= 0 and free_bytes < required_bytes + 16 * 1024 * 1024:
            raise RuntimeError("final-audio-storage-exhausted")
        if storage_pressure:
            # Preserve the current captured segment while there is still ample
            # reserve, then stop accepting additional continuous capture so a
            # runaway backlog cannot fill the root filesystem.
            keep_overlap = False
            self.storage_pressure_events += 1
            log.error(
                "storage pressure: stopping capture client=%s free_mb=%.1f reserve_mb=%d",
                client.client_id,
                free_bytes / (1024 * 1024) if free_bytes >= 0 else -1.0,
                STATE_MIN_FREE_MB,
            )

        await asyncio.to_thread(
            self._materialize_final_audio,
            raw_path,
            wav_path,
            novel_pcm,
            final_pcm,
        )
        self.final_wav_windows += 1

        utterance = Utterance(
            utterance_id=client.current_id,
            client_id=client.client_id,
            speaker=client.name,
            room=client.room,
            language_pref=client.segment_language_pref,
            turn_id=client.turn_id,
            segment_index=client.segment_index,
            raw_path=raw_path,
            wav_path=wav_path,
            started_ms=client.speech_started_ms,
            ended_ms=ended_ms,
            server_started_ms=int(client.speech_started_mono * 1000),
            server_ended_ms=int(ended_mono * 1000),
            avg_rms=avg_rms,
            peak_rms=peak_rms,
            speech_ratio=speech_ratio,
            audio_ms=len(novel_pcm) * 1000 // BYTES_PER_SECOND,
            context_prefix_ms=context_prefix_ms,
            end_reason=end_reason,
            preview_text=client.partial_confirmed_text,
        )

        # Update bounded context only with novel speech to avoid duplicate hard-cut
        # overlap. This buffer is RAM-only and never grows with speaking duration.
        client.logical_context_audio.extend(novel_pcm)
        if len(client.logical_context_audio) > FINAL_LEFT_CONTEXT_BYTES:
            del client.logical_context_audio[:-FINAL_LEFT_CONTEXT_BYTES]

        await self.submit_utterance(utterance)
        self.stream_segments += 1
        client.logical_last_end_mono = ended_mono
        if storage_pressure and client.capture_active:
            client.capture_active = False
            client.mic_state = "off"
            await self.broadcast(
                client.room,
                {
                    "type": "error",
                    "code": "storage-pressure",
                    "message": "Capture stopped to preserve system disk reserve",
                    "free_mb": round(
                        max(0, self._state_disk_free_bytes()) / (1024 * 1024), 1
                    ),
                },
            )

        overlap = b""
        if keep_overlap and SEGMENT_OVERLAP_BYTES > 0:
            overlap = pcm[-min(len(pcm), SEGMENT_OVERLAP_BYTES):]

        client.segment_index += 1
        client.partial_last_bytes = 0
        client.partial_revision = 0
        client.partial_last_text = ""

        if keep_overlap:
            client.current_id = self._segment_id(client.turn_id, client.segment_index)
            client.current_audio = bytearray(overlap)
            client.segment_leading_overlap_bytes = len(overlap)
            overlap_ms = len(overlap) * 1000 // BYTES_PER_SECOND
            client.speech_started_ms = ended_ms - overlap_ms
            client.speech_started_mono = ended_mono - overlap_ms / 1000.0
            self._reset_segment_stats(client, overlap)
        else:
            client.current_id = None
            client.current_audio.clear()
            client.segment_leading_overlap_bytes = 0
            self._reset_segment_stats(client)
            self._schedule_turn_finalize(client.client_id, client.turn_id, LOGICAL_TURN_GAP_MS)

        log.info(
            "segment flush id=%s turn=%s seg=%d reason=%s novel_ms=%d context_ms=%d",
            utterance.utterance_id,
            utterance.turn_id,
            utterance.segment_index,
            end_reason,
            utterance.audio_ms,
            context_prefix_ms,
        )
        return True

    async def ingest_stream_frame(self, client, data):
        if not client.capture_active:
            return
        self.stream_frames += 1
        rms, peak = self._pcm_stats(data)
        if not client.current_id and client.pending_noise_floor > client.noise_floor:
            old_floor = client.noise_floor
            client.noise_floor = min(STREAM_NOISE_MAX_FLOOR, client.pending_noise_floor)
            client.pending_noise_floor = 0.0
            self.stream_noise_updates += 1
            log.info("noise-learn applied client=%s old=%.6f new=%.6f",
                     client.client_id, old_floor, client.noise_floor)
        threshold = max(STREAM_MIN_THRESHOLD, client.noise_floor * STREAM_NOISE_MULTIPLIER)
        speech_like = rms >= threshold and peak >= max(STREAM_MIN_PEAK, threshold * 1.45)

        client.pre_roll.append((bytes(data), rms, peak, time.monotonic()))

        if not client.current_id:
            if rms < threshold:
                client.noise_floor = max(0.0010, client.noise_floor * 0.985 + rms * 0.015)
            client.vad_onset_frames = client.vad_onset_frames + 1 if speech_like else 0
            if client.vad_onset_frames < STREAM_ONSET_FRAMES:
                return

            buffered = list(client.pre_roll)
            client.pre_roll.clear()
            client.vad_onset_frames = 0
            client.vad_silent_frames = 0
            self._begin_segment(client, pre_roll=buffered)
            await self.broadcast_room_status(client.room)
            return

        await self._append_active_frame(client, data, rms=rms, peak=peak, speech_like=speech_like)
        client.vad_silent_frames = 0 if speech_like else client.vad_silent_frames + 1

        if (
            len(client.current_audio) >= SEGMENT_SOFT_BYTES
            and client.vad_silent_frames >= STREAM_SOFT_SILENCE_FRAMES
        ):
            self.stream_softcuts += 1
            await self._flush_client_segment(client, keep_overlap=False, end_reason="soft-silence")
            client.vad_silent_frames = 0
            client.pre_roll.clear()
            client.mic_state = "listening"
            await self.broadcast_room_status(client.room)
            return

        if client.vad_silent_frames >= STREAM_HANGOVER_FRAMES:
            self.stream_silence_ends += 1
            await self._flush_client_segment(client, keep_overlap=False, end_reason="silence")
            client.vad_silent_frames = 0
            client.pre_roll.clear()
            client.mic_state = "listening"
            await self.broadcast_room_status(client.room)

    async def ingest_pcm_payload(self, client, data, *, continuous):
        """Reframe arbitrary WebSocket PCM packets into exact 20 ms frames.

        Remove the consumed prefix only once per transport message. Repeated
        bytearray front-deletes are O(n) memmoves and become quadratic for a
        large-but-valid WebSocket packet. Yield periodically so one client cannot
        monopolize the event loop with hundreds of frames in one message.
        """
        if not data:
            return
        if len(data) > WS_MAX_PCM_MESSAGE_BYTES:
            raise ValueError("pcm-message-too-large")
        if len(data) % SAMPLE_BYTES:
            raise ValueError("pcm16-odd-byte-count")
        if not self._consume_pcm_budget(client, len(data)):
            raise ValueError("pcm-rate-limit")

        client.pcm_remainder.extend(data)
        complete_bytes = (
            len(client.pcm_remainder) // STREAM_FRAME_BYTES
        ) * STREAM_FRAME_BYTES
        if complete_bytes <= 0:
            return

        framed = bytes(client.pcm_remainder[:complete_bytes])
        del client.pcm_remainder[:complete_bytes]

        frame_count = 0
        for offset in range(0, len(framed), STREAM_FRAME_BYTES):
            frame = framed[offset:offset + STREAM_FRAME_BYTES]
            if continuous:
                await self.ingest_stream_frame(client, frame)
            elif client.current_id:
                rms, peak = self._pcm_stats(frame)
                threshold = max(
                    STREAM_MIN_THRESHOLD,
                    client.noise_floor * STREAM_NOISE_MULTIPLIER,
                )
                speech_like = (
                    rms >= threshold
                    and peak >= max(STREAM_MIN_PEAK, threshold * 1.45)
                )
                await self._append_active_frame(
                    client,
                    frame,
                    rms=rms,
                    peak=peak,
                    speech_like=speech_like,
                )
            frame_count += 1
            if frame_count % PCM_REFRAME_YIELD_FRAMES == 0:
                await asyncio.sleep(0)

    async def _stop_capture_for_stt_unavailable(self, client):
        """Bound backlog after an authoritative FINAL proves VIT unavailable.

        Preserve at most the segment already in flight, then stop continuous
        capture explicitly. This avoids an outage turning into an unbounded disk
        and descriptor backlog while keeping already-captured speech on the NVMe
        durability path.
        """
        if not client.capture_active or client.capture_mode != "continuous-v19":
            return False
        client.capture_active = False
        client.mic_state = "off"
        self.capture_stopped_stt_unavailable += 1
        try:
            await self._flush_pcm_remainder(client)
            if client.current_id:
                await self._flush_client_segment(
                    client,
                    keep_overlap=False,
                    end_reason="stt-unavailable",
                )
        except Exception:
            log.exception(
                "failed preserving active segment during VIT outage client=%s",
                client.client_id,
            )
        await self.broadcast_room_status(client.room)
        return True

    async def _flush_pcm_remainder(self, client):
        """Preserve a final sub-20 ms PCM tail when a segment is already active."""
        if not client.pcm_remainder:
            return
        tail = bytes(client.pcm_remainder)
        client.pcm_remainder.clear()
        if client.current_id and len(tail) >= SAMPLE_BYTES:
            rms, peak = self._pcm_stats(tail)
            threshold = max(
                STREAM_MIN_THRESHOLD,
                client.noise_floor * STREAM_NOISE_MULTIPLIER,
            )
            speech_like = (
                rms >= threshold
                and peak >= max(STREAM_MIN_PEAK, threshold * 1.45)
            )
            await self._append_active_frame(
                client,
                tail,
                rms=rms,
                peak=peak,
                speech_like=speech_like,
            )

    @staticmethod
    def _duration_ms(utterance):
        return max(1, int(utterance.audio_ms or (utterance.server_ended_ms - utterance.server_started_ms)))

    @classmethod
    def _overlap_ratio(cls, left, right):
        overlap = max(
            0,
            min(left.server_ended_ms, right.server_ended_ms)
            - max(left.server_started_ms, right.server_started_ms),
        )
        shorter = min(cls._duration_ms(left), cls._duration_ms(right))
        return overlap / shorter if shorter > 0 else 0.0

    @classmethod
    def _same_capture_window(cls, left, right):
        if left.room != right.room or left.client_id == right.client_id:
            return False
        return (
            abs(left.server_started_ms - right.server_started_ms) <= PRE_STT_START_DELTA_MS
            and abs(left.server_ended_ms - right.server_ended_ms) <= PRE_STT_END_DELTA_MS
            and cls._overlap_ratio(left, right) >= PRE_STT_MIN_OVERLAP_RATIO
        )

    @classmethod
    def _likely_room_echo(cls, dominant, weaker):
        if not cls._same_capture_window(dominant, weaker):
            return False
        if dominant.avg_rms <= 0.0 or weaker.avg_rms <= 0.0:
            return False
        avg_ratio = dominant.avg_rms / max(weaker.avg_rms, 1e-9)
        peak_ratio = dominant.peak_rms / max(weaker.peak_rms, 1e-9)
        return avg_ratio >= PRE_STT_AVG_RMS_RATIO and peak_ratio >= PRE_STT_PEAK_RMS_RATIO

    async def submit_utterance(self, utterance):
        self.turn_pending[(utterance.client_id, utterance.turn_id)] += 1
        async with self.pending_gate_lock:
            if len(self.pending_gate) >= PRE_STT_GATE_PENDING_CAPACITY:
                await self.queue.put(utterance)
                self.gate_forwarded += 1
                self.queue_high_watermark = max(self.queue_high_watermark, self.queue.qsize())
                return True
            self.pending_gate.append(utterance)

        await self.broadcast(
            utterance.room,
            {
                "type": "processing",
                "stage": "vit-final-wav",
                "utterance_id": utterance.utterance_id,
                "turn_id": utterance.turn_id,
                "segment_index": utterance.segment_index,
                "queue": self.queue.qsize(),
            },
        )
        return True

    async def _pre_stt_gate_worker(self):
        interval = max(0.05, min(PRE_STT_GATE_HOLD_MS / 4000.0, 0.20))
        while True:
            await asyncio.sleep(interval)
            await self._flush_pre_stt_gate()

    async def _flush_pre_stt_gate(self):
        now = time.monotonic()
        hold_s = max(0.0, PRE_STT_GATE_HOLD_MS / 1000.0)
        while True:
            async with self.pending_gate_lock:
                ready = [x for x in self.pending_gate if now - x.gate_received_at >= hold_s]
                if not ready:
                    return
                seed = min(ready, key=lambda x: x.gate_received_at)
                group = [x for x in self.pending_gate if x is seed or self._same_capture_window(seed, x)]
                for item in group:
                    self.pending_gate.remove(item)
            await self._forward_pre_stt_group(group)
            now = time.monotonic()

    async def _forward_pre_stt_group(self, group):
        if not group:
            return
        self.gate_groups += 1
        dominant = max(group, key=lambda x: (x.avg_rms, x.peak_rms))
        for item in group:
            echo_suspected = item is not dominant and self._likely_room_echo(dominant, item)
            if echo_suspected and PRE_STT_DESTRUCTIVE:
                # Compatibility switch only. Production v19 ships this OFF.
                self.gate_dropped += 1
                self.turn_pending[(item.client_id, item.turn_id)] = max(
                    0, self.turn_pending[(item.client_id, item.turn_id)] - 1
                )
                self._retain_audio(item, "legacy-destructive-gate")
                continue
            await self.queue.put(item)
            self.gate_forwarded += 1
            self.queue_high_watermark = max(self.queue_high_watermark, self.queue.qsize())
            await self.broadcast(
                item.room,
                {
                    "type": "processing",
                    "stage": "vit-final-wav",
                    "utterance_id": item.utterance_id,
                    "turn_id": item.turn_id,
                    "segment_index": item.segment_index,
                    "queue": self.queue.qsize(),
                    "echo_suspected": echo_suspected,
                },
            )
        await self.broadcast_room_status(group[0].room)

    @classmethod
    def _merge_rolling_text(cls, previous, window_text):
        """Replace the previous transcript tail with a long-window VIT decode.

        The WAV starts with bounded acoustic left context, so its first words
        should align somewhere near the tail of the already-published turn. A
        strong alignment lets the new long-window hypothesis rewrite that tail,
        which is the v19 self-correction mechanism. Weak alignment never deletes
        old text; it falls back to conservative append/dedup.
        """
        previous = cls._clean_text(previous)
        window_text = cls._clean_text(window_text)
        if not previous:
            return window_text, "initial", 1.0
        if not window_text:
            return previous, "empty-window", 0.0
        if previous == window_text:
            return previous, "identical", 1.0

        prev_matches = cls._word_matches(previous)
        win_matches = cls._word_matches(window_text)
        prev_words = [m.group(0).casefold() for m in prev_matches]
        win_words = [m.group(0).casefold() for m in win_matches]
        if not prev_words or not win_words:
            return previous + " " + window_text, "append-no-words", 0.0

        tail_start = max(0, len(prev_words) - FINAL_MERGE_TAIL_WORDS)
        prev_tail = prev_words[tail_start:]
        sm = SequenceMatcher(None, prev_tail, win_words, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size > 0 and b.b <= 18]
        block = max(blocks, key=lambda b: (b.size, -b.b), default=None)

        if block is not None and block.size >= 4:
            offset = tail_start + block.a - block.b
            if max(0, len(prev_words) - FINAL_MERGE_TAIL_WORDS) <= offset < len(prev_words):
                existing_slice = prev_words[offset:]
                compare_len = min(len(existing_slice), len(win_words))
                ratio = SequenceMatcher(
                    None,
                    existing_slice[:compare_len],
                    win_words[:compare_len],
                    autojunk=False,
                ).ratio() if compare_len else 0.0
                if ratio >= 0.42:
                    char_start = prev_matches[offset].start()
                    prefix = previous[:char_start].rstrip(" ,.;:!?-–—")
                    merged = (prefix + " " + window_text).strip() if prefix else window_text
                    return cls._clean_text(merged), "tail-rewrite", ratio

        # Conservative suffix-prefix exact overlap when long-window alignment is
        # too weak to justify rewriting existing words.
        best = 0
        for count in range(1, min(24, len(prev_words), len(win_words)) + 1):
            if prev_words[-count:] == win_words[:count]:
                best = count
        if best > 0:
            if best >= len(win_matches):
                return previous, "overlap-only", 1.0
            suffix = window_text[win_matches[best].start():].lstrip(" ,.;:!?-–—")
            merged = previous.rstrip() + (" " + suffix if suffix else "")
            return cls._clean_text(merged), "append-overlap", best / max(1, min(len(prev_words), len(win_words)))

        return cls._clean_text(previous.rstrip() + " " + window_text), "append-unaligned", 0.0

    async def _utterance_worker(self):
        while True:
            utterance = await self.queue.get()
            key = (utterance.client_id, utterance.turn_id)
            try:
                await self._process_utterance(utterance)
            except asyncio.CancelledError:
                self._retain_audio(utterance, "final-cancelled")
                raise
            except Exception:
                self.vit_final_errors += 1
                self._retain_audio(utterance, "final-error")
                log.exception("VIT FINAL failed id=%s", utterance.utterance_id)
                await self.broadcast(
                    utterance.room,
                    {
                        "type": "error",
                        "utterance_id": utterance.utterance_id,
                        "turn_id": utterance.turn_id,
                        "message": "VIT FINAL failed; audio retained",
                    },
                )
            finally:
                remaining = max(0, self.turn_pending.get(key, 0) - 1)
                if remaining:
                    self.turn_pending[key] = remaining
                else:
                    self.turn_pending.pop(key, None)
                if not utterance.retain_audio:
                    for path in (utterance.raw_path, utterance.wav_path):
                        try:
                            path.unlink(missing_ok=True)
                        except Exception:
                            pass
                self.queue.task_done()
                await self.broadcast_room_status(utterance.room)

    async def _process_utterance(self, utterance):
        started = time.time()
        self.vit_final_requests += 1
        try:
            reply = await self.vit_stt.ask(
                f"WAV\t{utterance.wav_path}\t{utterance.language_pref}\t{utterance.client_id}",
                timeout=VIT_TIMEOUT_SECONDS,
            )
            self.vit_available = True
        except Exception:
            self.vit_available = False
            # The worker owns the single error accounting point; avoid counting
            # one failed FINAL twice as it propagates through this helper.
            raise

        candidate = self._parse_vit_reply(reply)
        full_window_ms = utterance.context_prefix_ms + utterance.audio_ms
        if not candidate.get("ok"):
            self.vit_final_unresolved += 1
            reject_reason = candidate.get("reason") or "vit-final-empty"
            self._learn_noise_from_rejected(utterance, reject_reason)
            self._retain_audio(utterance, reject_reason)
            return
        if candidate.get("language") not in SUPPORTED_LANGUAGES:
            self.vit_final_unresolved += 1
            self._retain_audio(utterance, "unsupported-language")
            return

        text = self._clean_text(candidate.get("text"))
        reason = self._quality_reason(
            text,
            duration_ms=full_window_ms,
        )

        # v19.2.4: English rolling windows can occasionally repeat words/ngrams
        # when a long left acoustic context is present even though the novel
        # current segment decodes cleanly. Before rejecting that failure class,
        # re-decode only the novel segment with the SAME English VIT model.
        # This is a same-model recovery lane; it is not ensemble voting.
        segment_only_recovered = False
        if (
            utterance.context_prefix_ms > 0
            and candidate.get("language") in {"en", "vi"}
            and reason in {
                "consecutive-word-loop",
                "repeated-bigram",
                "repeated-trigram",
            }
        ):
            solo_wav = FINAL_SPOOL_DIR / (
                f"solo-{utterance.client_id}-{uuid.uuid4().hex}.wav"
            )
            try:
                raw_pcm = await asyncio.to_thread(utterance.raw_path.read_bytes)
                await asyncio.to_thread(
                    self._write_wav,
                    solo_wav,
                    raw_pcm,
                )
                self.vit_final_recovery_requests += 1
                solo_reply = await self.vit_stt.ask(
                    f"WAV\t{solo_wav}\t{utterance.language_pref}\t"
                    f"{utterance.client_id}",
                    timeout=VIT_TIMEOUT_SECONDS,
                )
                solo = self._parse_vit_reply(solo_reply)
                solo_text = (
                    self._clean_text(solo.get("text"))
                    if solo.get("ok")
                    else ""
                )
                solo_reason = (
                    self._quality_reason(
                        solo_text,
                        duration_ms=utterance.audio_ms,
                        reject_repetition=False,
                    )
                    if (
                        solo_text
                        and solo.get("language") in SUPPORTED_LANGUAGES
                    )
                    else "segment-only-invalid"
                )
                if solo_reason is None:
                    text = solo_text
                    reason = None
                    segment_only_recovered = True
            finally:
                try:
                    solo_wav.unlink(missing_ok=True)
                except Exception:
                    pass

        # Accuracy-first policy:
        # repetition alone is never sufficient to delete valid speech.
        #
        # With no acoustic left context there is no useful segment-only
        # re-decode to perform, because the current candidate already
        # represents only novel audio. Re-evaluate while ignoring repetition
        # heuristics, but retain every other quality guard.
        if (
            utterance.context_prefix_ms <= 0
            and reason in {
                "consecutive-word-loop",
                "repeated-bigram",
                "repeated-trigram",
            }
        ):
            reason = self._quality_reason(
                text,
                duration_ms=full_window_ms,
                reject_repetition=False,
            )

        if reason is None:
            reason = self._sparse_preview_reason(
                utterance,
                text,
            )

        if reason is not None:
            self.vit_final_unresolved += 1
            self._learn_noise_from_rejected(utterance, reason)
            self._retain_audio(utterance, reason)
            await self.broadcast(
                utterance.room,
                {
                    "type": "processing_done",
                    "utterance_id": utterance.utterance_id,
                    "turn_id": utterance.turn_id,
                    "unresolved": True,
                    "filter_reason": reason,
                    "audio_retained": True,
                },
            )
            return

        row = self.turn_rows.get((utterance.client_id, utterance.turn_id))
        previous_text = (
            row.get("canonical_text", row.get("text", ""))
            if row else ""
        )
        merged_text, merge_mode, merge_similarity = self._merge_rolling_text(previous_text, text)

        # A v19.2.4 segment-only recovery contains only novel audio, not the
        # rolling left context. If lexical overlap cannot be aligned to already
        # published text, conservative plain append is therefore correct.
        if segment_only_recovered:
            if (
                previous_text
                and merge_mode in {"append-unaligned", "append-no-words"}
            ):
                merged_text = self._clean_text(
                    previous_text + " " + text
                )
                merge_mode = "recovery-append-segment-only"
            else:
                merge_mode = "recovery-" + merge_mode

        # If the long context window cannot be aligned safely to the already
        # published tail, never append the whole context window and duplicate old
        # speech. Re-decode only the current segment with the same VIT model, then
        # append that segment conservatively. This is a same-model recovery lane,
        # not ensemble voting.
        if (
            not segment_only_recovered
            and previous_text
            and merge_mode in {"append-unaligned", "append-no-words"}
        ):
            solo_wav = FINAL_SPOOL_DIR / f"solo-{utterance.client_id}-{uuid.uuid4().hex}.wav"
            try:
                raw_pcm = await asyncio.to_thread(utterance.raw_path.read_bytes)
                await asyncio.to_thread(self._write_wav, solo_wav, raw_pcm)
                self.vit_final_recovery_requests += 1
                solo_reply = await self.vit_stt.ask(
                    f"WAV\t{solo_wav}\t{utterance.language_pref}\t{utterance.client_id}",
                    timeout=VIT_TIMEOUT_SECONDS,
                )
                solo = self._parse_vit_reply(solo_reply)
                solo_text = self._clean_text(solo.get("text")) if solo.get("ok") else ""
                if (
                    solo_text
                    and solo.get("language") in SUPPORTED_LANGUAGES
                    and self._quality_reason(solo_text, duration_ms=utterance.audio_ms, reject_repetition=False) is None
                ):
                    merged_text, solo_mode, solo_similarity = self._merge_rolling_text(
                        previous_text, solo_text
                    )
                    # A segment-only decode has no old acoustic context. If exact
                    # overlap still cannot be found, plain append is correct here.
                    if solo_mode == "append-unaligned":
                        merged_text = self._clean_text(previous_text + " " + solo_text)
                        solo_mode = "append-segment-only"
                    merge_mode = "recovery-" + solo_mode
                    merge_similarity = solo_similarity
                    text = solo_text
            finally:
                try:
                    solo_wav.unlink(missing_ok=True)
                except Exception:
                    pass

        if not merged_text:
            return

        if "tail-rewrite" in merge_mode and previous_text and merged_text != previous_text:
            self.vit_final_corrections += 1
        elif "append" in merge_mode or merge_mode == "initial":
            self.vit_final_appends += 1

        processing_ms = int((time.time() - started) * 1000)
        await self._upsert_turn_row(
            utterance,
            merged_text,
            result_language=candidate.get("language") or utterance.language_pref,
            window_text=text,
            merge_mode=merge_mode,
            merge_similarity=merge_similarity,
            processing_ms=processing_ms,
        )
        # V20 speaker tracking is post-ASR. The transcript row
        # has already been committed/broadcast above, so speaker inference
        # cannot delay or alter canonical recognition.
        if SPEAKER_ENABLED:
            try:
                speaker_pcm = await asyncio.to_thread(
                    utterance.raw_path.read_bytes
                )
                if speaker_pcm:
                    self._spawn(
                        self._assign_speaker_pcm(
                            key=(
                                utterance.client_id,
                                utterance.turn_id,
                            ),
                            room=utterance.room,
                            segment_index=utterance.segment_index,
                            audio_ms=utterance.audio_ms,
                            started_ms=utterance.started_ms,
                            ended_ms=utterance.ended_ms,
                            pcm=speaker_pcm,
                        )
                    )
            except Exception:
                self.speaker_errors += 1
                log.exception(
                    "failed snapshotting speaker PCM turn=%s segment=%s",
                    utterance.turn_id,
                    utterance.segment_index,
                )

        self.vit_final_results += 1

    async def _upsert_turn_row(self, utterance, merged_text, *, result_language, window_text, merge_mode, merge_similarity, processing_ms):
        key = (utterance.client_id, utterance.turn_id)
        existing = self.turn_rows.get(key)
        now = time.time()
        display_text = self._display_text(
            merged_text,
            result_language,
            is_final=False,
        )
        translation_input_text = self._translation_input_text(
            merged_text,
            result_language,
            is_final=False,
        )
        if existing is None:
            row = {
                "type": "transcript",
                "id": utterance.turn_id,
                "room": utterance.room,
                "client_id": utterance.client_id,
                "speaker": utterance.speaker,
                "speaker_id": "",
                "speaker_segments": [],
                "speaker_status": "pending" if SPEAKER_ENABLED else "disabled",
                "source_language": result_language,
                "canonical_text": merged_text,
                "display_text": display_text,
                "translation_input_text": translation_input_text,
                "text": display_text,
                "translation": "",
                "started_ms": utterance.started_ms,
                "ended_ms": utterance.ended_ms,
                "turn_id": utterance.turn_id,
                "segment_index": utterance.segment_index,
                "segment_count": utterance.segment_index + 1,
                "audio_ms": utterance.audio_ms,
                "context_prefix_ms": utterance.context_prefix_ms,
                "created_at": now,
                "updated_at": now,
                "processing_ms": processing_ms,
                "asr_source": "vit-stt-final-wav",
                "verification_status": "vit-long-window-final",
                "stability": "rolling-final",
                "revision": 1,
                "source_revision": 1,
                "is_final": False,
                "merge_mode": merge_mode,
                "merge_similarity": round(float(merge_similarity), 4),
                "last_window_text": window_text,
            }
            self.turn_rows[key] = row
            self.history_cache.append(row)
            self.history_cache = self.history_cache[-MAX_SESSION_TURNS:]
            self._prune_turn_state()
            await self.broadcast(utterance.room, dict(row))
        else:
            previous_text = existing.get(
                "canonical_text",
                existing.get("text", ""),
            )
            previous_translation = self._clean_text(
                existing.get("translation", "")
            )
            existing.update(
                canonical_text=merged_text,
                display_text=display_text,
                translation_input_text=translation_input_text,
                text=display_text,
                source_language=result_language,
                ended_ms=utterance.ended_ms,
                segment_index=utterance.segment_index,
                segment_count=max(int(existing.get("segment_count") or 0), utterance.segment_index + 1),
                audio_ms=int(existing.get("audio_ms") or 0) + utterance.audio_ms,
                context_prefix_ms=utterance.context_prefix_ms,
                updated_at=now,
                processing_ms=processing_ms,
                asr_source="vit-stt-final-wav",
                verification_status=(
                    "vit-long-window-corrected"
                    if merge_mode == "tail-rewrite" and merged_text != previous_text
                    else "vit-long-window-final"
                ),
                stability="corrected" if merge_mode == "tail-rewrite" and merged_text != previous_text else "rolling-final",
                revision=int(existing.get("revision") or 1) + 1,
                source_revision=int(existing.get("source_revision") or 1) + 1,
                is_final=False,
                merge_mode=merge_mode,
                merge_similarity=round(float(merge_similarity), 4),
                last_window_text=window_text,
            )

            # Keep the last-known-good translation visible while a newer ASR
            # revision is being translated. Clearing it here caused the UI to
            # oscillate between useful text and "no stable translation" during
            # continuous speech. Metadata makes the stale relationship explicit
            # without pretending the old translation matches the new source.
            if (
                previous_translation
                and int(existing.get("translation_source_revision") or 0)
                != int(existing.get("source_revision") or 0)
            ):
                existing["translation_pending"] = True
                existing["translation_pending_source_revision"] = int(
                    existing.get("source_revision") or 0
                )
                existing["translation_status"] = "updating-source"
                self.translation_stale_visible_updates += 1

            outgoing = dict(existing)
            outgoing["type"] = "transcript_replace"
            outgoing["replace_id"] = existing["id"]
            await self.broadcast(utterance.room, outgoing)

        # v19.3 progressive translation:
        # coalesce small rolling corrections instead of translating every
        # accepted STT window. This reduces stale NMT work during continuous
        # speech while still providing useful progressive updates.
        if TRANSLATION_ENABLED and TRANSLATION_PROGRESSIVE_ENABLED:
            current = self.turn_rows.get(key)
            if current is not None:
                source_words = len(
                    self._words(
                        current.get(
                            "translation_input_text",
                            current.get("canonical_text", ""),
                        )
                    )
                )
                last_queued_words = int(
                    current.get("translation_last_queued_words") or 0
                )
                should_queue = (
                    (
                        last_queued_words == 0
                        and source_words >= TRANSLATION_PROGRESSIVE_FIRST_WORDS
                    )
                    or (
                        last_queued_words > 0
                        and source_words - last_queued_words
                        >= TRANSLATION_PROGRESSIVE_MIN_NEW_WORDS
                    )
                )
                if should_queue:
                    queued = await self._queue_translation_job(
                        utterance.client_id,
                        utterance.turn_id,
                    )
                    if queued:
                        current["translation_last_queued_words"] = source_words
                else:
                    self.translation_progressive_deferred += 1

        await self.broadcast_room_status(utterance.room)

    async def _queue_translation_job(self, client_id, turn_id):
        """Queue exactly one logical-FINAL translation job per turn.

        V20 translation is post-ASR only. Non-final revisions, LIVE text and
        progressive state are never admitted to the native translator queue.
        """
        if not TRANSLATION_ENABLED:
            return False

        key = (client_id, turn_id)
        row = self.turn_rows.get(key)

        if row is None or not bool(row.get("is_final")):
            return False

        if (
            key in self.translation_pending_keys
            or key in self.translation_active_keys
        ):
            self.translation_queue_coalesced += 1
            return False

        self.translation_pending_keys.add(key)

        try:
            self.translation_queue.put_nowait(key)
        except asyncio.QueueFull:
            self.translation_pending_keys.discard(key)
            self.translation_errors += 1
            row["translation_pending"] = False
            row["translation_status"] = "translation-queue-full"
            return False

        self.translation_queued += 1
        return True

    def _prune_turn_state(self):
        """Bound metadata to rows still retained in the in-memory history."""
        if len(self.turn_rows) <= MAX_SESSION_TURNS:
            return
        live_keys = {
            (str(row.get("client_id", "")), str(row.get("turn_id", "")))
            for row in self.history_cache
        }
        for key in list(self.turn_rows):
            if (
                key in live_keys
                or self.turn_pending.get(key, 0) > 0
                or key in self.translation_pending_keys
                or key in self.translation_active_keys
            ):
                continue
            self.turn_rows.pop(key, None)
            self.turn_pending.pop(key, None)
            self.translation_pending_keys.discard(key)
            for cache_key in list(self.translation_chunk_cache):
                if cache_key[:2] == key:
                    self.translation_chunk_cache.pop(cache_key, None)

    @classmethod
    def _translation_live_source_span(cls, text):
        """Return a bounded suffix of a confirmed VIT LIVE hypothesis.

        LIVE is disposable and uses a rolling acoustic window. Bounding only the
        preview source limits worst-case GPU occupancy without changing canonical
        STT or authoritative FINAL translation input.
        """
        clean = cls._clean_text(text)
        matches = cls._word_matches(clean)
        if len(matches) <= TRANSLATION_LIVE_MAX_WORDS:
            return clean, False
        start = matches[-TRANSLATION_LIVE_MAX_WORDS].start()
        bounded = clean[start:].lstrip(" ,.;:!?-–—")
        return cls._clean_text(bounded), True

    @classmethod
    def _translation_quality_reason(cls, source, translated):
        source = cls._clean_text(source)
        translated = cls._clean_text(translated)
        if not translated:
            return "empty"
        if cls._has_disallowed_script(translated):
            return "disallowed-script"
        src_words = cls._words(source)
        out_words = cls._words(translated)

        # Defensive guard for the runaway failure class observed before the
        # Marian source-EOS fix. Never publish pathological decoder output.
        if re.search(r"([<>•])(?:\s*\1){3,}", translated):
            return "runaway-symbols"

        if len(out_words) >= 3:
            run = 1
            for i in range(1, len(out_words)):
                if out_words[i] == out_words[i - 1]:
                    run += 1
                    if run >= 3:
                        return "runaway-consecutive-word"
                else:
                    run = 1

        if len(out_words) >= 6:
            for i in range(len(out_words) - 5):
                if (
                    out_words[i:i + 2]
                    == out_words[i + 2:i + 4]
                    == out_words[i + 4:i + 6]
                ):
                    return "runaway-bigram"

        if len(out_words) >= 9:
            trigrams = {}
            for i in range(len(out_words) - 2):
                gram = tuple(out_words[i:i + 3])
                trigrams[gram] = trigrams.get(gram, 0) + 1
            if max(trigrams.values(), default=0) >= 3:
                return "runaway-trigram"

        if src_words and len(out_words) > max(80, len(src_words) * 5):
            return "runaway-length"
        if len(out_words) >= 10 and len(set(out_words)) / max(1, len(out_words)) < 0.20:
            return "runaway-repetition"

        # Fail closed only on severe, deterministic information failures. The
        # thresholds are intentionally conservative so technical terms/names do
        # not make a legitimate bilingual sentence look like source copying.
        if len(src_words) >= 8 and len(out_words) >= 8:
            copy_ratio = SequenceMatcher(
                None, src_words, out_words, autojunk=False
            ).ratio()
            if copy_ratio >= 0.92:
                return "source-copy"
        if len(src_words) >= 12 and len(out_words) < max(3, len(src_words) // 4):
            return "severe-undertranslation"
        return None

    @staticmethod
    def _translation_numeric_anchors(text):
        """Return typed numeric anchors while treating written percent forms alike."""
        anchors = []
        pattern = (
            r"(?<!\w)\d+(?:[.,:/-]\d+)*"
            r"(?:\s*(?:%|‰|percent|per\s+cent|phần\s+trăm))?"
        )
        for match in re.finditer(pattern, str(text or ""), flags=re.IGNORECASE):
            raw = match.group(0).strip()
            percent = bool(re.search(
                r"(?:%|percent|per\s+cent|phần\s+trăm)\s*$",
                raw, flags=re.IGNORECASE,
            ))
            permille = raw.endswith("‰")
            core = re.sub(
                r"\s*(?:%|‰|percent|per\s+cent|phần\s+trăm)\s*$",
                "", raw, flags=re.IGNORECASE,
            )
            groups = tuple(re.findall(r"\d+", core))
            separators = re.findall(r"[.,:/-]", core)
            if not groups:
                continue
            if percent or permille:
                anchors.append(("percent" if percent else "permille", groups))
            elif any(sep in {":", "/", "-"} for sep in separators) or len(separators) >= 2:
                anchors.append(("structured", groups))
            elif len(separators) == 1 and len(groups) == 2:
                if len(groups[1]) == 3 and 1 <= len(groups[0]) <= 3:
                    anchors.append(("integer", (groups[0] + groups[1],)))
                else:
                    anchors.append(("decimal", groups))
            else:
                anchors.append(("integer", ("".join(groups),)))
        return anchors

    @staticmethod
    def _en_parse_number_words(words):
        small = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
            "fourteen": 14, "fifteen": 15, "sixteen": 16,
            "seventeen": 17, "eighteen": 18, "nineteen": 19,
        }
        tens = {
            "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        }
        if not words:
            return None
        total = 0
        current = 0
        seen = False
        for word in words:
            word = word.casefold()
            if word == "and":
                continue
            if word in small:
                current += small[word]
                seen = True
            elif word in tens:
                current += tens[word]
                seen = True
            elif word == "hundred" and current > 0:
                current *= 100
            elif word in {"thousand", "million", "billion"} and current > 0:
                scale = {"thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}[word]
                total += current * scale
                current = 0
            else:
                return None
        return total + current if seen else None

    @classmethod
    def _translation_spoken_percent_anchors(cls, text):
        """Normalize spoken VI/EN percentages for validation only.

        This never rewrites NMT input/output. It lets QA recognize equivalent
        forms such as "mười lăm phần trăm", "fifteen percent", and "15%".
        """
        raw = cls._clean_text(text)
        anchors = []
        for kind, groups in cls._translation_numeric_anchors(raw):
            if kind == "percent":
                anchors.append(("percent", groups))

        tokens = cls._words(raw)
        for idx in range(len(tokens)):
            # Vietnamese: <number words> phần trăm
            if idx + 1 < len(tokens) and tokens[idx] == "phần" and tokens[idx + 1] == "trăm":
                low = max(0, idx - 12)
                for start in range(low, idx):
                    phrase = tokens[start:idx]
                    value = cls._vi_parse_number_words(phrase)
                    if value is not None:
                        groups = tuple(re.findall(r"\d+", value))
                        if groups:
                            anchors.append(("percent", groups))
                        break
            # English: <number words> percent / per cent
            is_percent = tokens[idx] == "percent" or (
                idx + 1 < len(tokens) and tokens[idx] == "per" and tokens[idx + 1] == "cent"
            )
            if is_percent:
                low = max(0, idx - 10)
                for start in range(low, idx):
                    value = cls._en_parse_number_words(tokens[start:idx])
                    if value is not None:
                        anchors.append(("percent", (str(value),)))
                        break
        # Every detector above is anchored to a distinct textual percent marker
        # ("%", "phần trăm", "percent", or "per cent"). Keep duplicates by
        # value: "15% ... 15%" carries two facts and the target must preserve
        # both occurrences rather than satisfying QA with only one.
        return anchors

    @staticmethod
    def _translation_id_anchors(text):
        raw = str(text or "")
        anchors = set()
        # Mixed technical IDs/versions such as GPT-5, ISO-9001, Q4, H264.
        for token in re.findall(
            r"(?<![A-Za-z0-9])(?:[A-Za-z]{1,12}[-_.]?[0-9]{1,12}|[0-9]{1,12}[-_.]?[A-Za-z]{1,12})(?![A-Za-z0-9])",
            raw,
        ):
            anchors.add(token.casefold())
        return sorted(anchors)

    @classmethod
    def _translation_anchor_reason(cls, source, translated):
        """Reject deterministic information loss that should never be translated.

        Arabic numbers are language-independent anchors.  Keeping this guard on
        the semantic NMT source (not display ITN) avoids false confidence from a
        presentation rewrite while catching dropped years, percentages and IDs.
        """
        source_percentages = cls._translation_spoken_percent_anchors(source)
        if source_percentages:
            remaining_percentages = list(
                cls._translation_spoken_percent_anchors(translated)
            )
            for anchor in source_percentages:
                if anchor in remaining_percentages:
                    remaining_percentages.remove(anchor)
                else:
                    return "missing-percent-anchor"

        # Percent anchors are compared semantically above so written and spoken
        # equivalents do not get double-counted as ordinary integers.
        source_numbers = [
            anchor for anchor in cls._translation_numeric_anchors(source)
            if anchor[0] not in {"percent", "permille"}
        ]
        if source_numbers:
            target_numbers = [
                anchor for anchor in cls._translation_numeric_anchors(translated)
                if anchor[0] not in {"percent", "permille"}
            ]
            remaining = list(target_numbers)
            for anchor in source_numbers:
                if anchor in remaining:
                    remaining.remove(anchor)
                else:
                    return "missing-numeric-anchor"

        # Preserve explicit all-caps Latin acronyms from English source.  Do not
        # apply this to ordinary title-cased names because Vietnamese may render
        # those names with harmless punctuation/casing differences.
        acronyms = re.findall(r"(?<![A-Za-z])(?:[A-Z]{2,8})(?![A-Za-z])", str(source or ""))
        folded_target = str(translated or "").casefold()
        for acronym in acronyms:
            token = re.escape(acronym.casefold())
            if re.search(rf"(?<![a-z0-9]){token}(?![a-z0-9])", folded_target) is None:
                return "missing-acronym-anchor"

        for anchor in cls._translation_id_anchors(source):
            token = re.escape(anchor)
            if re.search(rf"(?<![a-z0-9]){token}(?![a-z0-9])", folded_target) is None:
                return "missing-id-anchor"

        # Product/domain terms are validation-only anchors. They are never
        # replaced with placeholders in NMT input, avoiding tokenizer/context
        # damage while preventing silent corruption of known technical names.
        folded_source = str(source or "").casefold()
        for term in TRANSLATION_PROTECTED_TERMS:
            folded_term = term.casefold()
            if folded_term in folded_source and folded_term not in folded_target:
                return "missing-protected-term"
        return None

    @classmethod
    def _roundtrip_similarity(cls, source, backtranslated):
        source_words = cls._words(source)
        back_words = cls._words(backtranslated)
        if not source_words or not back_words:
            return 0.0

        sequence = SequenceMatcher(
            None, source_words, back_words, autojunk=False
        ).ratio()
        source_set = set(source_words)
        back_set = set(back_words)
        recall = len(source_set & back_set) / max(1, len(source_set))

        source_chars = "".join(ch for ch in cls._clean_text(source).casefold() if ch.isalnum())
        back_chars = "".join(ch for ch in cls._clean_text(backtranslated).casefold() if ch.isalnum())
        char_ratio = SequenceMatcher(
            None, source_chars, back_chars, autojunk=False
        ).ratio() if source_chars and back_chars else 0.0

        return 0.55 * sequence + 0.30 * recall + 0.15 * char_ratio

    async def _roundtrip_verify_chunk(self, direction, source, candidate):
        """Final-only semantic sanity check using the same bilingual model.

        Round-trip agreement is not a proof of a correct translation, so the
        threshold is intentionally low.  It is a fail-closed detector for large
        meaning drift such as a fluent but unrelated translation.  Progressive
        updates skip this expensive check; logical FINAL is authoritative.
        """
        word_count = len(self._words(source))
        if (
            not TRANSLATION_ROUNDTRIP_QA
            or self.translation_model_family != "madlad400-3b-mt-int8"
            or word_count < TRANSLATION_ROUNDTRIP_MIN_WORDS
            or word_count > TRANSLATION_ROUNDTRIP_MAX_WORDS
        ):
            return candidate, None, None, False

        reverse_direction = "en-vi" if direction == "vi-en" else "vi-en"
        self.translation_roundtrip_checks += 1
        backtranslated, back_reason = await self._translate_chunk_once(
            reverse_direction,
            candidate,
            greedy=True,
        )
        if back_reason is None and backtranslated:
            score = self._roundtrip_similarity(source, backtranslated)
            if score >= TRANSLATION_ROUNDTRIP_MIN_SIMILARITY:
                return candidate, None, score, False
        else:
            score = 0.0

        # A forward greedy retry is useful only when the primary decode used a
        # wider beam. MADLAD v19.4.x production already starts with greedy, so an
        # identical second request adds latency without creating a new hypothesis.
        if not TRANSLATION_PRIMARY_GREEDY:
            greedy_candidate, greedy_reason = await self._translate_chunk_once(
                direction,
                source,
                greedy=True,
            )
            if greedy_reason is None and greedy_candidate:
                anchor_reason = self._translation_anchor_reason(source, greedy_candidate)
                if anchor_reason is None:
                    greedy_back, greedy_back_reason = await self._translate_chunk_once(
                        reverse_direction,
                        greedy_candidate,
                        greedy=True,
                    )
                    if greedy_back_reason is None and greedy_back:
                        greedy_score = self._roundtrip_similarity(source, greedy_back)
                        if greedy_score >= TRANSLATION_ROUNDTRIP_MIN_SIMILARITY:
                            self.translation_roundtrip_recoveries += 1
                            return greedy_candidate, None, greedy_score, True
                        score = max(score, greedy_score)
        else:
            self.translation_identical_retry_skipped += 1

        self.translation_roundtrip_failures += 1
        return "", "roundtrip-low-similarity", score, False

    @classmethod
    def _sanitize_translation_candidate(cls, direction, text):
        candidate = cls._clean_text(text)
        if direction != "en-vi":
            return candidate

        # v19.2.4.1 EN->VI sanitizer, kept post-translation and isolated
        # from canonical STT text.
        candidate = re.sub(
            r"</?\s*[A-Za-z][^>]{0,80}>",
            " ",
            candidate,
        )
        candidate = re.sub(
            r"[¶♩♪♫♬]+",
            " ",
            candidate,
        )
        candidate = cls._clean_text(candidate)
        return re.sub(r"\s+([,.;:!?])", r"\1", candidate)

    @classmethod
    def _translation_split_index(cls, words):
        """Choose a clause-aware split near the configured chunk limit."""
        if len(words) <= TRANSLATION_CHUNK_WORDS:
            return len(words)

        high = min(len(words) - 1, TRANSLATION_CHUNK_WORDS)
        low = max(10, high - 14)
        boundary_words = {
            "and", "but", "because", "so", "while", "when", "if", "that",
            "which", "however", "therefore", "meanwhile",
            "và", "nhưng", "vì", "nên", "mà", "khi", "nếu", "để",
            "cho", "còn", "tuy", "do", "trong", "với",
        }

        # Prefer explicit punctuation, then a conjunction/discourse boundary.
        for idx in range(high, low - 1, -1):
            previous = words[idx - 1] if idx > 0 else ""
            current = words[idx] if idx < len(words) else ""
            if previous.rstrip().endswith((",", ";", ":")):
                return idx
            if re.sub(r"^[^\wÀ-ỹ]+|[^\wÀ-ỹ]+$", "", current.casefold()) in boundary_words:
                return idx

        return high

    @classmethod
    def _translation_chunks(cls, text):
        clean = cls._clean_text(text)
        if not clean:
            return []
        if len(clean.split()) <= TRANSLATION_CHUNK_TRIGGER_WORDS:
            return [clean]

        # Preserve sentence boundaries, but do not make every short sentence an
        # isolated NMT request. Pack adjacent sentences up to the semantic word
        # budget so the model keeps discourse context and avoids tiny fragments.
        sentences = [
            p.strip()
            for p in re.findall(
                r"[^.!?…]+(?:[.!?…]+|$)",
                clean,
            )
            if p.strip()
        ] or [clean]

        units = []
        for sentence in sentences:
            words = sentence.split()
            while len(words) > TRANSLATION_CHUNK_WORDS:
                split_at = cls._translation_split_index(words)
                split_at = max(8, min(split_at, len(words) - 1))
                units.append(cls._clean_text(" ".join(words[:split_at])))
                words = words[split_at:]
            if words:
                units.append(cls._clean_text(" ".join(words)))

        chunks = []
        current = []
        current_words = 0
        for unit in units:
            unit_words = len(unit.split())
            if current and current_words + unit_words > TRANSLATION_CHUNK_WORDS:
                chunks.append(cls._clean_text(" ".join(current)))
                current = []
                current_words = 0
            current.append(unit)
            current_words += unit_words
        if current:
            chunks.append(cls._clean_text(" ".join(current)))

        # Avoid a tiny context-starved tail when it can safely be merged.
        if len(chunks) >= 2:
            tail = chunks[-1].split()
            prev = chunks[-2].split()
            if (
                len(tail) < TRANSLATION_CHUNK_MIN_TAIL_WORDS
                and len(prev) + len(tail)
                <= TRANSLATION_CHUNK_WORDS + TRANSLATION_CHUNK_MIN_TAIL_WORDS
            ):
                chunks[-2] = cls._clean_text(
                    chunks[-2] + " " + chunks[-1]
                )
                chunks.pop()

        return [c for c in chunks if c]

    @classmethod
    def _join_translation_parts(cls, parts):
        """Join adjacent translated chunks without deleting target text.

        Translation source chunks are non-overlapping. Target-only boundary
        deduplication can therefore erase a legitimate repeated phrase. Keep
        joining lossless; source-aware overlap handling belongs in ASR merge,
        not in NMT output assembly.
        """
        clean_parts = [cls._clean_text(p) for p in parts if cls._clean_text(p)]
        return cls._clean_text(" ".join(clean_parts)), 0

    @classmethod
    def _translation_recovery_split(cls, text):
        words = cls._clean_text(text).split()
        if len(words) < 8:
            return []
        midpoint = len(words) // 2
        low = max(4, midpoint - 5)
        high = min(len(words) - 4, midpoint + 5)
        boundary_words = {
            "and", "but", "because", "so", "while", "when", "if",
            "và", "nhưng", "vì", "nên", "mà", "khi", "nếu", "để",
        }
        split_at = midpoint
        best_distance = len(words)
        for idx in range(low, high + 1):
            word = re.sub(
                r"^[^\wÀ-ỹ]+|[^\wÀ-ỹ]+$",
                "",
                words[idx].casefold(),
            )
            punctuation_boundary = words[idx - 1].endswith((",", ";", ":"))
            if word in boundary_words or punctuation_boundary:
                distance = abs(idx - midpoint)
                if distance < best_distance:
                    split_at = idx
                    best_distance = distance
        return [
            cls._clean_text(" ".join(words[:split_at])),
            cls._clean_text(" ".join(words[split_at:])),
        ]

    async def _translate_chunk_once(
        self, direction, source, *, greedy=False, timeout_seconds=None,
        live_preview=False,
    ):
        self.translation_chunk_requests += 1
        # EnViT5 policy: LIVE=beam1, FINAL=beam1, QA recovery=beam2 once.
        # Keep the historical `greedy` keyword internal for call-site stability;
        # on EnViT5 it means the explicit beam2 recovery protocol.
        if live_preview:
            request_direction = f"{direction}-live"
        elif greedy and self.translation_model_family == "envit5-int8":
            request_direction = f"{direction}-beam2"
        else:
            request_direction = direction
        request_started = time.monotonic()
        try:
            reply = await self.translator.ask(
                f"{request_direction}\t{source}",
                timeout=(
                    TRANSLATION_TIMEOUT_SECONDS
                    if timeout_seconds is None
                    else int(timeout_seconds)
                ),
            )
            self._translator_mark_ready()
        except asyncio.TimeoutError:
            if live_preview:
                # LIVE is disposable. Do not arm authoritative FINAL backoff.
                self.translator_available = False
                self.translator_retry_not_before = 0.0
                self.translation_live_reconnect_pending = True
                self.translation_live_request_timeouts += 1
                return "", "request-timeout"
            self._translator_mark_failed("request-timeout")
            return "", "request-timeout"
        except Exception:
            self._translator_mark_failed("request-failed")
            return "", "request-failed"
        finally:
            elapsed_ms = int((time.monotonic() - request_started) * 1000)
            self.translation_request_last_ms = elapsed_ms
            self.translation_request_max_ms = max(
                self.translation_request_max_ms, elapsed_ms
            )
            self.translation_request_total_ms += elapsed_ms

        if not reply.startswith("OK\t"):
            if reply.startswith("ERR\tproxy-busy"):
                return "", "proxy-busy"
            if reply.startswith("ERR\tinput-too-long"):
                return "", "input-too-long"
            return "", "worker-error"

        candidate = self._sanitize_translation_candidate(
            direction,
            reply.split("\t", 1)[1],
        )
        reason = self._translation_quality_reason(source, candidate)
        if reason is None:
            reason = self._translation_anchor_reason(source, candidate)
            if reason is not None:
                self.translation_anchor_failures += 1
        return candidate, reason

    async def _translate_chunk_with_recovery(
        self,
        direction,
        source,
        *,
        depth=0,
    ):
        """Beam1 primary; beam2 once after deterministic QA reject; then split."""
        candidate, reason = await self._translate_chunk_once(direction, source)
        if reason is None:
            return candidate, None, depth > 0

        if reason in {"request-failed", "request-timeout", "worker-error"}:
            return "", reason, depth > 0

        word_count = len(self._words(source))
        if reason == "input-too-long":
            self.translation_token_limit_recoveries += 1

        beam2_attempted = False
        if (
            depth == 0
            and reason != "input-too-long"
            and self.translation_model_family == "envit5-int8"
        ):
            beam2_attempted = True
            self.translation_chunk_recoveries += 1
            beam2_candidate, beam2_reason = await self._translate_chunk_once(
                direction,
                source,
                greedy=True,
            )
            if beam2_reason is None and beam2_candidate:
                return beam2_candidate, None, True
            reason = beam2_reason or reason

        # Do not create context-starved subfragments. EnViT5 already consumed the
        # only permitted beam2 recovery at depth zero.
        if word_count < 8:
            if depth == 0 and beam2_attempted:
                self.translation_chunk_recovery_failures += 1
            return "", reason, depth > 0 or beam2_attempted

        if depth >= TRANSLATION_RECOVERY_MAX_DEPTH:
            if depth == 0 and beam2_attempted:
                self.translation_chunk_recovery_failures += 1
            return "", reason, depth > 0 or beam2_attempted

        recovery_sources = self._translation_recovery_split(source)
        if len(recovery_sources) != 2:
            if depth == 0 and beam2_attempted:
                self.translation_chunk_recovery_failures += 1
            return "", reason, depth > 0 or beam2_attempted

        if depth == 0 and not beam2_attempted:
            self.translation_chunk_recoveries += 1

        recovered_parts = []
        last_reason = reason
        for recovery_source in recovery_sources:
            recovered, recovery_reason, _ = await self._translate_chunk_with_recovery(
                direction,
                recovery_source,
                depth=depth + 1,
            )
            if recovery_reason is not None or not recovered:
                if depth == 0:
                    self.translation_chunk_recovery_failures += 1
                return "", recovery_reason or last_reason, True
            recovered_parts.append(recovered)

        combined, _ = self._join_translation_parts(recovered_parts)
        combined_reason = self._translation_quality_reason(source, combined)
        if combined_reason is None:
            combined_reason = self._translation_anchor_reason(source, combined)
            if combined_reason is not None:
                self.translation_anchor_failures += 1
        if combined_reason is not None:
            if depth == 0:
                self.translation_chunk_recovery_failures += 1
            return "", combined_reason, True

        return combined, None, True

    async def _queue_live_translation(
        self, *, client_id, room, turn_id, utterance_id,
        source_language, source_text, revision,
    ):
        """Latest-wins disposable translation of confirmed VIT LIVE text.

        This lane never writes turn_rows/history and therefore cannot become an
        authoritative transcript/translation. It exists only to reduce perceived
        latency while the FINAL STT/translation lanes preserve correctness.
        """
        if not (TRANSLATION_ENABLED and TRANSLATION_LIVE_PREVIEW_ENABLED):
            return False
        source_text = self._translation_input_text(
            source_text, source_language, is_final=False
        )
        source_text, source_trimmed = self._translation_live_source_span(source_text)
        if source_trimmed:
            self.translation_live_source_trimmed += 1
        if len(self._words(source_text)) < TRANSLATION_LIVE_MIN_WORDS:
            self.translation_live_skipped_short += 1
            return False

        key = (client_id, turn_id)
        current_row = self.turn_rows.get(key)
        if (
            current_row is not None
            and self._clean_text(current_row.get("translation", ""))
        ):
            # HLMEET_TRANSLATE_EFFICIENT_V19_7_5: cumulative progressive/FINAL translation already owns
            # this turn in web v4.2. Drop any saved LIVE snapshot and do not
            # enqueue native work whose result the browser will ignore.
            self.translation_live_latest.pop(key, None)
            self.translation_live_skipped_cumulative += 1
            return False

        # LIVE remains a fast bootstrap only before cumulative translation.
        # A LIVE request-timeout resets only this server's Unix client.
        # Treat that as transient: accept the newest confirmed STT revision so
        # the LIVE worker can reconnect.
        if (
            not self._translator_live_capable()
            and not self.translation_live_reconnect_pending
        ):
            self.translation_live_skipped_unavailable += 1
            return False

        self.translation_live_sequence += 1
        payload = {
            "sequence": self.translation_live_sequence,
            "client_id": client_id,
            "room": room,
            "turn_id": turn_id,
            "utterance_id": utterance_id,
            "source_language": str(source_language or "").lower(),
            "source_text": source_text,
            "revision": int(revision or 0),
        }
        previous = self.translation_live_latest.get(key)
        self.translation_live_latest[key] = payload
        if previous is not None:
            self.translation_live_coalesced += 1

        if key in self.translation_live_pending_keys or key in self.translation_live_active_keys:
            return True
        self.translation_live_pending_keys.add(key)
        await self.translation_live_queue.put(key)
        self.translation_live_queued += 1
        return True

    async def _translation_live_worker(self):
        while True:
            key = await self.translation_live_queue.get()
            self.translation_live_pending_keys.discard(key)
            self.translation_live_active_keys.add(key)
            processed_sequence = 0
            try:
                payload = self.translation_live_latest.get(key)
                if payload is None:
                    continue

                current_row = self.turn_rows.get(key)
                if (
                    current_row is not None
                    and self._clean_text(current_row.get("translation", ""))
                ):
                    # HLMEET_TRANSLATE_EFFICIENT_V19_7_5: queued before cumulative handoff; cancel before
                    # touching the native translator.
                    self.translation_live_latest.pop(key, None)
                    self.translation_live_skipped_cumulative += 1
                    continue

                # FINAL translation always wins access to the one native daemon.
                # Dropping a disposable preview is preferable to delaying an
                # authoritative logical-turn result.
                if (
                    self.translation_queue.qsize() > 0
                    or self.translation_active_keys
                    or self._authoritative_stt_busy()
                ):
                    self.translation_live_skipped_busy += 1
                    continue
                if self.translation_live_reconnect_pending:
                    self.translation_live_reconnect_attempts += 1
                    ready = await self._ensure_translator_ready(
                        "live-reconnect"
                    )
                    if ready:
                        self.translation_live_reconnect_pending = False
                        self.translation_live_reconnect_successes += 1
                    else:
                        self.translation_live_skipped_unavailable += 1
                        continue

                if not self._translator_live_capable():
                    self.translation_live_skipped_unavailable += 1
                    continue

                now = time.monotonic()
                wait_s = (
                    TRANSLATION_LIVE_MIN_INTERVAL_MS / 1000.0
                    - (now - self.translation_live_last_started_mono)
                )
                if wait_s > 0:
                    await asyncio.sleep(wait_s)
                    # Use the newest source for this turn after the debounce.
                    payload = self.translation_live_latest.get(key)
                    if payload is None:
                        continue

                    current_row = self.turn_rows.get(key)
                    if (
                        current_row is not None
                        and self._clean_text(current_row.get("translation", ""))
                    ):
                        # HLMEET_TRANSLATE_EFFICIENT_V19_7_5: progressive cumulative became visible during
                        # debounce; do not start obsolete LIVE inference.
                        self.translation_live_latest.pop(key, None)
                        self.translation_live_skipped_cumulative += 1
                        continue

                    if (
                        self.translation_queue.qsize() > 0
                        or self.translation_active_keys
                        or self._authoritative_stt_busy()
                    ):
                        self.translation_live_skipped_busy += 1
                        continue

                client = self.clients.get(payload["client_id"])
                if client is None or client.turn_id != payload["turn_id"]:
                    continue

                source_language = payload["source_language"]
                if source_language not in SUPPORTED_LANGUAGES:
                    continue
                direction = "vi-en" if source_language == "vi" else "en-vi"
                target_language = "en" if source_language == "vi" else "vi"
                processed_sequence = int(payload["sequence"])
                self.translation_live_last_started_mono = time.monotonic()

                translated, reason = await self._translate_chunk_once(
                    direction,
                    payload["source_text"],
                    timeout_seconds=TRANSLATION_LIVE_TIMEOUT_SECONDS,
                    live_preview=True,
                )
                if reason == "proxy-busy":
                    # The old timed-out native LIVE inference is still
                    # being drained by the isolated proxy. Do not queue
                    # this disposable revision behind it; the next STT
                    # revision will retry latest-wins.
                    self.translation_live_proxy_busy += 1
                    self.translation_live_skipped_busy += 1
                    continue
                if reason is not None or not translated:
                    self.translation_live_errors += 1
                    continue

                client = self.clients.get(payload["client_id"])
                if client is None or client.turn_id != payload["turn_id"]:
                    continue

                latest = self.translation_live_latest.get(key)
                if latest is None:
                    # Preserve FINAL invalidation semantics. A missing latest
                    # snapshot means the disposable LIVE lane was explicitly
                    # invalidated/removed; do not resurrect it.
                    self.translation_live_superseded_dropped += 1
                    continue

                superseded = bool(
                    int(latest.get("sequence") or 0) != processed_sequence
                )
                if superseded:
                    # HLMEET_LIVE_LATENCY_V19_7_2
                    #
                    # This inference already completed successfully. VIT LIVE
                    # may have advanced while EnViT5 was running, but discarding
                    # the completed translation creates starvation during
                    # continuous speech. Expose it as last-good/lagging LIVE;
                    # `finally` still reschedules the newest source snapshot.
                    #
                    # FINAL translation remains authoritative and the FINAL
                    # guard below can still suppress this disposable preview.
                    self.translation_live_superseded_visible += 1

                current_row = self.turn_rows.get(key)
                if (
                    key in self.translation_pending_keys
                    or key in self.translation_active_keys
                    or (current_row is not None and bool(current_row.get("is_final")))
                ):
                    self.translation_live_final_guard_dropped += 1
                    continue

                await self.broadcast(
                    payload["room"],
                    {
                        "type": "translation_partial",
                        "turn_id": payload["turn_id"],
                        "utterance_id": payload["utterance_id"],
                        "client_id": payload["client_id"],
                        "source_language": source_language,
                        "translation_language": target_language,
                        "translation": translated,
                        "translation_status": (
                            "translated-live-preview-lagging"
                            if superseded
                            else "translated-live-preview"
                        ),
                        "source_text": payload["source_text"],
                        "source_revision": payload["revision"],
                        "created_at": time.time(),
                    },
                )
                self.translation_live_completed += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                self.translation_live_errors += 1
                log.exception("LIVE translation worker error")
            finally:
                self.translation_live_active_keys.discard(key)
                self.translation_live_queue.task_done()

                # If newer LIVE text arrived while native inference was running,
                # retain only that newest snapshot and schedule one more token.
                latest = self.translation_live_latest.get(key)
                latest_sequence = int(latest.get("sequence") or 0) if latest else 0
                client = self.clients.get(key[0])
                turn_still_active = bool(
                    client is not None and client.turn_id == key[1]
                )
                if (
                    processed_sequence > 0
                    and latest_sequence > processed_sequence
                    and turn_still_active
                    and key not in self.translation_live_pending_keys
                ):
                    self.translation_live_pending_keys.add(key)
                    self.translation_live_queue.put_nowait(key)
                    self.translation_live_queued += 1
                elif not turn_still_active:
                    self.translation_live_latest.pop(key, None)

    async def _translation_worker(self):
        """Authoritative V20 FINAL-only OPUS worker.

        No progressive retries and no self-requeue loop exist here. A failed
        auxiliary service may recover for a later turn, while STT/control-plane
        work continues independently.
        """
        while True:
            key = await self.translation_queue.get()
            self.translation_pending_keys.discard(key)
            self.translation_active_keys.add(key)

            try:
                client_id, turn_id = key
                current = self.turn_rows.get(key)

                if current is None or not bool(current.get("is_final")):
                    self.translation_stale_skipped += 1
                    continue

                source_language = str(
                    current.get("source_language") or ""
                ).lower()

                if source_language not in {"vi", "en"}:
                    self.translation_stale_skipped += 1
                    continue

                source_text = self._clean_text(
                    current.get(
                        "translation_input_text",
                        current.get(
                            "canonical_text",
                            current.get("text", ""),
                        ),
                    )
                )

                if not source_text:
                    self.translation_stale_skipped += 1
                    continue

                source_revision = int(
                    current.get("source_revision") or 0
                )
                direction = (
                    "vi-en"
                    if source_language == "vi"
                    else "en-vi"
                )
                target_language = (
                    "en"
                    if source_language == "vi"
                    else "vi"
                )

                current["translation_pending"] = True
                current["translation_pending_source_revision"] = (
                    source_revision
                )
                current["translation_status"] = "translating-final-source"

                ready = await self._ensure_translator_ready(
                    "final-request"
                )

                if not ready:
                    current = self.turn_rows.get(key)
                    if current is not None:
                        current["translation_pending"] = False
                        current.pop(
                            "translation_pending_source_revision",
                            None,
                        )
                        current["translation_status"] = (
                            "translator-unavailable"
                        )
                        current["revision"] = int(
                            current.get("revision") or 1
                        ) + 1
                        current["updated_at"] = time.time()

                        outgoing = dict(current)
                        outgoing["type"] = "transcript_replace"
                        outgoing["replace_id"] = current["id"]
                        await self.broadcast(
                            current["room"],
                            outgoing,
                        )

                    self.translation_errors += 1
                    continue

                chunks = (
                    self._translation_chunks(source_text)
                    or [source_text]
                )
                translated_parts = []

                for chunk in chunks:
                    started = time.monotonic()
                    self.translation_chunk_requests += 1

                    try:
                        reply = await self.translator.ask(
                            f"{direction}\t{chunk}",
                            timeout=TRANSLATION_TIMEOUT_SECONDS,
                        )
                    except Exception:
                        self._translator_mark_failed(
                            "final-request-failed"
                        )
                        raise

                    elapsed_ms = int(
                        (time.monotonic() - started) * 1000
                    )
                    self.translation_request_last_ms = elapsed_ms
                    self.translation_request_max_ms = max(
                        self.translation_request_max_ms,
                        elapsed_ms,
                    )
                    self.translation_request_total_ms += elapsed_ms

                    if not reply.startswith("OK\t"):
                        self.translation_chunk_failures += 1
                        raise RuntimeError(
                            f"translator reply: {reply[:240]}"
                        )

                    candidate = self._clean_text(
                        reply.split("\t", 1)[1]
                    )

                    if not candidate:
                        self.translation_chunk_failures += 1
                        raise RuntimeError(
                            "translator returned empty text"
                        )

                    translated_parts.append(candidate)

                translated, boundary_dedupes = (
                    self._join_translation_parts(
                        translated_parts
                    )
                )
                self.translation_boundary_dedupes += (
                    boundary_dedupes
                )

                reason = self._translation_quality_reason(
                    source_text,
                    translated,
                )
                if reason is None:
                    reason = self._translation_anchor_reason(
                        source_text,
                        translated,
                    )
                    if reason is not None:
                        self.translation_anchor_failures += 1

                if reason is not None:
                    raise RuntimeError(
                        f"translation rejected: {reason}"
                    )

                latest = self.turn_rows.get(key)

                if latest is None:
                    self.translation_stale_skipped += 1
                    continue

                latest_source = self._clean_text(
                    latest.get(
                        "translation_input_text",
                        latest.get(
                            "canonical_text",
                            latest.get("text", ""),
                        ),
                    )
                )

                if (
                    not bool(latest.get("is_final"))
                    or int(
                        latest.get("source_revision") or 0
                    ) != source_revision
                    or latest_source != source_text
                ):
                    self.translation_stale_skipped += 1
                    continue

                latest["translation"] = translated
                latest["translation_pending"] = False
                latest.pop(
                    "translation_pending_source_revision",
                    None,
                )
                latest["translation_language"] = (
                    target_language
                )
                latest["translation_source_revision"] = (
                    source_revision
                )
                latest["translation_source_text"] = source_text
                latest["translation_status"] = (
                    "translated-final-source-chunked"
                    if len(chunks) > 1
                    else "translated-final-source"
                )
                latest["translation_updated_at"] = time.time()
                latest["revision"] = int(
                    latest.get("revision") or 1
                ) + 1
                latest["updated_at"] = time.time()

                outgoing = dict(latest)
                outgoing["type"] = "transcript_replace"
                outgoing["replace_id"] = latest["id"]

                await self.broadcast(
                    latest["room"],
                    outgoing,
                )

                self.translation_completed += 1
                self._translator_mark_ready()

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.translation_errors += 1
                log.warning(
                    "FINAL translation failed turn=%s error=%s",
                    key[1],
                    exc,
                )

                row = self.turn_rows.get(key)
                if row is not None and bool(row.get("is_final")):
                    row["translation_pending"] = False
                    row.pop(
                        "translation_pending_source_revision",
                        None,
                    )
                    row["translation_status"] = (
                        "translation-error"
                    )
                    row["revision"] = int(
                        row.get("revision") or 1
                    ) + 1
                    row["updated_at"] = time.time()

                    outgoing = dict(row)
                    outgoing["type"] = "transcript_replace"
                    outgoing["replace_id"] = row["id"]

                    try:
                        await self.broadcast(
                            row["room"],
                            outgoing,
                        )
                    except Exception:
                        log.exception(
                            "failed broadcasting translation error"
                        )

                # Deliberately no requeue here. A later FINAL turn gets one
                # fresh attempt after the bounded restart backoff.

            finally:
                self.translation_active_keys.discard(key)
                self.translation_queue.task_done()

    async def _assign_speaker_pcm(
        self,
        *,
        key,
        room,
        segment_index,
        audio_ms,
        started_ms,
        ended_ms,
        pcm,
    ):
        if not SPEAKER_ENABLED or not pcm:
            self.speaker_skipped += 1
            return

        # Speaker re-ID quality guard only. Do not alter, trim, or reject STT.
        if int(audio_ms) < SPEAKER_MIN_AUDIO_MS:
            self.speaker_skipped += 1
            return

        wav_path = SPEAKER_TMP_DIR / (
            f"speaker-{safe_utterance_id(key[1])}-"
            f"{segment_index}-{uuid.uuid4().hex}.wav"
        )

        try:
            await asyncio.to_thread(
                self._write_wav,
                wav_path,
                pcm,
            )

            self.speaker_requests += 1

            reply = await self.speaker_id.ask(
                f"WAV\t{wav_path}\t{room}",
                timeout=SPEAKER_TIMEOUT_SECONDS,
            )

            # A valid protocol response proves that the speaker
            # process itself is alive, even if the clip is too short.
            self.speaker_available = True

            if not reply.startswith("OK\t"):
                self.speaker_skipped += 1
                return

            fields = reply.split("\t")

            if len(fields) < 4:
                self.speaker_errors += 1
                return

            speaker_id = fields[1].strip()

            try:
                score = float(fields[2])
            except ValueError:
                score = 0.0

            match_mode = fields[3].strip()

            def speaker_float(index, default):
                if len(fields) <= index:
                    return default
                try:
                    return float(fields[index])
                except ValueError:
                    return default

            second_score = speaker_float(4, -1.0)
            margin = speaker_float(
                5,
                1.0 if second_score < -0.5 else score - second_score,
            )
            # Phase3 native appends the actual pre-decision similarities.
            # These are especially important when match_mode == "new", where
            # the assigned-speaker score remains 1.0 for protocol compatibility.
            raw_best_score = speaker_float(6, score)
            raw_second_score = speaker_float(7, second_score)
            raw_margin = speaker_float(8, margin)

            if not speaker_id:
                self.speaker_errors += 1
                return

            row = self.turn_rows.get(key)

            if row is None:
                self.speaker_skipped += 1
                return

            segments = list(
                row.get("speaker_segments") or []
            )

            # Replace an existing result for this segment index
            # instead of ever duplicating it.
            segments = [
                item
                for item in segments
                if int(item.get("segment_index", -1))
                != int(segment_index)
            ]

            segments.append(
                {
                    "segment_index": int(segment_index),
                    "speaker_id": speaker_id,
                    "score": round(score, 4),
                    "second_score": round(second_score, 4),
                    "margin": round(margin, 4),
                    "raw_best_score": round(raw_best_score, 4),
                    "raw_second_score": round(raw_second_score, 4),
                    "raw_margin": round(raw_margin, 4),
                    "match": match_mode,
                    "audio_ms": int(audio_ms),
                    "started_ms": int(started_ms),
                    "ended_ms": int(ended_ms),
                }
            )

            segments.sort(
                key=lambda item: int(
                    item.get("segment_index", 0)
                )
            )

            totals = {}

            for item in segments:
                sid = str(
                    item.get("speaker_id") or ""
                )
                if not sid:
                    continue

                totals[sid] = (
                    totals.get(sid, 0)
                    + max(
                        1,
                        int(item.get("audio_ms") or 0),
                    )
                )

            dominant = (
                max(
                    totals,
                    key=totals.get,
                )
                if totals
                else speaker_id
            )

            # Speaker metadata only. Never touch canonical text.
            row["speaker_id"] = dominant
            row["speaker_segments"] = segments
            row["speaker_status"] = "identified"
            row["speaker_last_score"] = round(score, 4)
            row["speaker_last_second_score"] = round(second_score, 4)
            row["speaker_last_margin"] = round(margin, 4)
            row["speaker_last_raw_best_score"] = round(raw_best_score, 4)
            row["speaker_last_raw_second_score"] = round(raw_second_score, 4)
            row["speaker_last_raw_margin"] = round(raw_margin, 4)
            row["revision"] = int(
                row.get("revision") or 1
            ) + 1
            row["updated_at"] = time.time()

            outgoing = dict(row)
            outgoing["type"] = "transcript_replace"
            outgoing["replace_id"] = row["id"]

            await self.broadcast(
                row["room"],
                outgoing,
            )

            self.speaker_results += 1

        except asyncio.CancelledError:
            raise
        except Exception:
            self.speaker_available = False
            self.speaker_errors += 1
            log.exception(
                "speaker assignment failed turn=%s segment=%s",
                key[1],
                segment_index,
            )
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _schedule_turn_finalize(self, client_id, turn_id, delay_ms):
        if not turn_id:
            return
        key = (client_id, turn_id)
        old = self.turn_finalize_tasks.pop(key, None)
        if old is not None:
            old.cancel()
        task = asyncio.create_task(self._finalize_turn_after_gap(client_id, turn_id, delay_ms))
        self.turn_finalize_tasks[key] = task

        def done(_):
            current = self.turn_finalize_tasks.get(key)
            if current is task:
                self.turn_finalize_tasks.pop(key, None)
        task.add_done_callback(done)

    async def _finalize_turn_after_gap(self, client_id, turn_id, delay_ms):
        try:
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000.0)

            client = self.clients.get(client_id)
            if client is not None and client.turn_id == turn_id and client.current_id:
                return

            key = (client_id, turn_id)
            # Never finalize before all captured FINAL windows for this turn have
            # completed. Captured audio has priority over UI timing.
            while self.turn_pending.get(key, 0) > 0:
                await asyncio.sleep(0.08)

            row = self.turn_rows.get(key)
            if row is not None and not row.get("is_final"):
                row["is_final"] = True
                row["stability"] = "finalized"
                row["verification_status"] = "vit-long-window-finalized"

                canonical = self._clean_text(
                    row.get("canonical_text", row.get("text", ""))
                )
                source_language = str(row.get("source_language", "")).lower()
                final_display = self._display_text(
                    canonical,
                    source_language,
                    is_final=True,
                )
                final_translation_input = self._translation_input_text(
                    canonical,
                    source_language,
                    is_final=True,
                )
                row["canonical_text"] = canonical
                row["display_text"] = final_display
                row["translation_input_text"] = final_translation_input
                row["text"] = final_display

                # FINAL written-form may differ from the progressive source by
                # punctuation/ITN only. Keep last-known-good translation visible
                # until the authoritative FINAL replacement is ready.
                if (
                    self._clean_text(row.get("translation", ""))
                    and self._clean_text(
                        row.get("translation_source_text", "")
                    ) != final_translation_input
                ):
                    row["translation_pending"] = True
                    row["translation_pending_source_revision"] = int(
                        row.get("source_revision") or 0
                    )
                    row["translation_status"] = "updating-final-source"
                    self.translation_stale_visible_updates += 1

                row["revision"] = int(row.get("revision") or 1) + 1
                row["updated_at"] = time.time()
                outgoing = dict(row)
                outgoing["type"] = "transcript_replace"
                outgoing["replace_id"] = row["id"]
                await self.broadcast(row["room"], outgoing)
                self.logical_turns_finalized += 1
                if TRANSLATION_ENABLED:
                    row["translation_last_queued_words"] = len(
                        self._words(final_translation_input)
                    )
                    await self._queue_translation_job(
                        client_id,
                        turn_id,
                    )

            client = self.clients.get(client_id)
            if client is not None and client.turn_id == turn_id and not client.current_id:
                self._reset_logical_turn(client)
                client.mic_state = "listening" if client.capture_active else "off"
                await self.broadcast_room_status(client.room)
        except asyncio.CancelledError:
            raise

    def room_clients(self, room):
        return [
            {
                "client_id": c.client_id,
                "name": c.name,
                "mic_state": c.mic_state,
                "language": c.language_pref,
                "joined_at": c.joined_at,
            }
            for c in self.clients.values()
            if c.room == room
        ]

    def room_stats(self, room):
        rows = [x for x in self.history_cache if x.get("room") == room]
        return {
            "turns": len(rows),
            "participants": len(self.room_clients(room)),
            "queue": self.queue.qsize(),
            "queue_capacity": QUEUE_CAPACITY,
            "queue_high_watermark": self.queue_high_watermark,
            "pre_stt_pending": sum(1 for x in self.pending_gate if x.room == room),
        }

    async def broadcast(self, room, payload):
        encoded = json.dumps(payload, ensure_ascii=False)
        targets = [
            (client_id, client)
            for client_id, client in list(self.clients.items())
            if client.room == room and not client.closing
        ]

        async def send_one(client_id, client):
            try:
                await asyncio.wait_for(
                    client.ws.send_str(encoded),
                    timeout=WS_SEND_TIMEOUT_SECONDS,
                )
                return None
            except Exception:
                return client_id

        if not targets:
            return
        dead = await asyncio.gather(
            *(send_one(client_id, client) for client_id, client in targets)
        )
        for client_id in dead:
            if client_id is None:
                continue
            # Keep the session in self.clients until its WebSocket handler exits.
            # Otherwise a half-closed slow peer is no longer counted by
            # MAX_CLIENTS yet can still deliver buffered PCM to its local handler.
            client = self.clients.get(client_id)
            if client is None or client.closing:
                continue
            client.closing = True
            self.slow_client_evictions += 1
            if not client.ws.closed:
                try:
                    await asyncio.wait_for(
                        client.ws.close(code=1011, message=b"slow-client"),
                        timeout=0.5,
                    )
                except Exception:
                    pass

    async def broadcast_room_status(self, room):
        await self.broadcast(
            room,
            {
                "type": "room_status",
                "participants": self.room_clients(room),
                "stats": self.room_stats(room),
            },
        )


meeting = Meeting()


def origin_allowed(origin: str, request: Optional[web.Request] = None) -> bool:
    if not origin:
        return True
    try:
        parsed = urlsplit(origin)
        host = (parsed.hostname or "").lower()
        normalized = origin.rstrip("/")
        if normalized in ALLOWED_WEB_ORIGINS:
            return True
        if ALLOW_PAGES_DEV and parsed.scheme == "https" and host.endswith(".pages.dev"):
            return True
        if ALLOW_LOCALHOST_DEV and host in {"localhost", "127.0.0.1", "::1"}:
            return True
        if request is not None and normalized == f"{request.scheme}://{request.host}".rstrip("/"):
            return True
    except Exception:
        return False
    return False


@web.middleware
async def cors_middleware(request, handler):
    origin = request.headers.get("Origin", "")
    allowed = origin_allowed(origin, request)
    if request.method == "OPTIONS":
        if origin and not allowed:
            return web.Response(status=403, text="Origin not allowed")
        response = web.Response(status=204)
    else:
        response = await handler(request)

    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    if origin and allowed:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        requested = request.headers.get("Access-Control-Request-Headers", "")
        response.headers["Access-Control-Allow-Headers"] = requested or "Content-Type"
        response.headers["Access-Control-Max-Age"] = "86400"
        if request.headers.get("Access-Control-Request-Private-Network") == "true":
            response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


def normalized_room(value):
    value = str(value or "default").strip()[:64] or "default"
    return re.sub(r"[^A-Za-z0-9._-]", "-", value)


def safe_name(value):
    value = str(value or "Guest").strip()[:64]
    return value or "Guest"


def safe_language_pref(value):
    value = str(value or "vi").strip().lower()
    if value in {"vi", "en"}:
        return value
    if value == "auto" and ALLOW_AUTO_LANGUAGE:
        return "auto"
    return "vi"


def safe_utterance_id(value):
    value = str(value or uuid.uuid4().hex).strip()[:80]
    value = re.sub(r"[^A-Za-z0-9_-]", "-", value)
    value = value.strip("-")[:64]
    return value or uuid.uuid4().hex


async def root_info(request):
    return web.json_response(
        {
            "service": "meeting-server",
            "version": APP_VERSION,
            "mode": "api-only",
            "asr": "VIT-only",
            "health": "/api/health",
            "websocket": "/ws",
        }
    )


async def health(request):
    rooms = {c.room for c in meeting.clients.values()}
    return web.json_response(
        {
            # ok is liveness for backward compatibility; ready is product
            # readiness and requires authoritative VIT plus healthy storage.
            "ok": True,
            "ready": bool(
                meeting.vit_available
                and meeting._storage_ready()
                and meeting._critical_workers_ready()
            ),
            "version": APP_VERSION,
            "uptime_s": int(time.monotonic() - meeting.started_monotonic),
            "clients": len(meeting.clients),
            "client_admissions_inflight": meeting.client_admissions_inflight,
            "rooms": len(rooms),
            "queue": meeting.queue.qsize(),
            "queue_capacity": QUEUE_CAPACITY,
            "queue_high_watermark": meeting.queue_high_watermark,
            "critical_workers_ready": meeting._critical_workers_ready(),
            "critical_worker_failures": meeting.critical_worker_failures,
            "translation_worker_failures": meeting.translation_worker_failures,
            "pre_stt_pending": len(meeting.pending_gate),
            "pre_stt_gate_forwarded": meeting.gate_forwarded,
            "pre_stt_gate_dropped": meeting.gate_dropped,
            "pre_stt_destructive": PRE_STT_DESTRUCTIVE,
            "stt_engine": meeting.stt_engine,
            "asr_stack": "vit-stt-only",
            "vit_available": meeting.vit_available,
            "vit_daemon_resets": meeting.vit_stt.reset_count,
            "whisper_small_available": False,
            "whisper_turbo_available": False,
            "translation_enabled": TRANSLATION_ENABLED,
            "translation_available": meeting.translator_available,
            "speaker_enabled": SPEAKER_ENABLED,
            "speaker_available": meeting.speaker_available,
            "speaker_engine": "sherpa-onnx-embedding-v20",
            "speaker_threshold": SPEAKER_THRESHOLD,
            "speaker_min_audio_ms": SPEAKER_MIN_AUDIO_MS,
            "speaker_requests": meeting.speaker_requests,
            "speaker_results": meeting.speaker_results,
            "speaker_errors": meeting.speaker_errors,
            "speaker_skipped": meeting.speaker_skipped,
            "translation_warming": meeting._translator_warming(),
            "translation_requested_device": TRANSLATOR_REQUESTED_DEVICE,
            "translation_transport": "isolated-unix-socket-v20",
            "translation_socket": TRANSLATOR_SOCKET_PATH,
            "translation_device": meeting._translator_runtime_device() or "unknown",
            "translation_compute_type": (
                meeting._translator_runtime_compute_type() or "unknown"
            ),
            "translation_daemon_resets": meeting.translator.reset_count,
            "translation_restart_failures": meeting.translator_restart_failures,
            "translation_restart_deferred": meeting.translation_restart_deferred,
            "translation_retry_in_s": round(
                max(0.0, meeting.translator_retry_not_before - time.monotonic()), 1
            ),
            "translation_opus_fallback_allowed": TRANSLATOR_ALLOW_OPUS_FALLBACK,
            "partial_engine": "vit-stt-short-pcm",
            "final_engine": "vit-stt-long-wav",
            "final_decoder": "modified_beam_search",
            "final_max_active_paths": 4,
            "stream_frames": meeting.stream_frames,
            "stream_speech_starts": meeting.stream_speech_starts,
            "stream_segments": meeting.stream_segments,
            "stream_autocuts": meeting.stream_autocuts,
            "stream_softcuts": meeting.stream_softcuts,
            "stream_silence_ends": meeting.stream_silence_ends,
            "stream_noise_updates": meeting.stream_noise_updates,
            "pcm_rate_rejects": meeting.pcm_rate_rejects,
            "client_limit_rejects": meeting.client_limit_rejects,
            "protocol_state_rejects": meeting.protocol_state_rejects,
            "stream_noise_max_floor": STREAM_NOISE_MAX_FLOOR,
            "logical_turns_started": meeting.logical_turns_started,
            "logical_turns_continued": meeting.logical_turns_continued,
            "logical_turns_finalized": meeting.logical_turns_finalized,
            "logical_turn_gap_ms": LOGICAL_TURN_GAP_MS,
            "segment_soft_seconds": SEGMENT_SOFT_SECONDS,
            "segment_hard_seconds": SEGMENT_HARD_SECONDS,
            "segment_overlap_ms": SEGMENT_OVERLAP_MS,
            "final_left_context_seconds": FINAL_LEFT_CONTEXT_SECONDS,
            "final_max_window_seconds": FINAL_MAX_WINDOW_SECONDS,
            "vit_preview_requests": meeting.vit_preview_requests,
            "vit_preview_results": meeting.vit_preview_results,
            "vit_preview_errors": meeting.vit_preview_errors,
            "vit_preview_skipped_busy": meeting.vit_preview_skipped_busy,
            "vit_preview_skipped_final_guard": meeting.vit_preview_skipped_final_guard,
            "vit_preview_skipped_global_busy": meeting.vit_preview_skipped_global_busy,
            "vit_preview_skipped_vit_locked": meeting.vit_preview_skipped_vit_locked,
            "vit_preview_cancelled_for_final": meeting.vit_preview_cancelled_for_final,
            "capture_stopped_stt_unavailable": meeting.capture_stopped_stt_unavailable,
            "slow_client_evictions": meeting.slow_client_evictions,
            "vit_native_decode_preemptible": False,
            "vit_final_requests": meeting.vit_final_requests,
            "vit_final_results": meeting.vit_final_results,
            "vit_final_errors": meeting.vit_final_errors,
            "vit_final_corrections": meeting.vit_final_corrections,
            "vit_final_appends": meeting.vit_final_appends,
            "vit_final_unresolved": meeting.vit_final_unresolved,
            "vit_final_recovery_requests": meeting.vit_final_recovery_requests,
            "final_wav_windows": meeting.final_wav_windows,
            "retained_audio_segments": meeting.retained_audio_segments,
            "translation_queue": meeting.translation_queue.qsize(),
            "translation_queued": meeting.translation_queued,
            "translation_completed": meeting.translation_completed,
            "translation_errors": meeting.translation_errors,
            "translation_progressive_enabled": TRANSLATION_PROGRESSIVE_ENABLED,
            "translation_live_preview_enabled": TRANSLATION_LIVE_PREVIEW_ENABLED,
            "translation_live_queue": meeting.translation_live_queue.qsize(),
            "translation_live_active": len(meeting.translation_live_active_keys),
            "translation_live_queued": meeting.translation_live_queued,
            "translation_live_completed": meeting.translation_live_completed,
            "translation_live_errors": meeting.translation_live_errors,
            "translation_live_skipped_short": meeting.translation_live_skipped_short,
            "translation_live_skipped_busy": meeting.translation_live_skipped_busy,
            "translation_live_skipped_unavailable": meeting.translation_live_skipped_unavailable,
            "translation_live_coalesced": meeting.translation_live_coalesced,
            "translation_live_superseded_visible": meeting.translation_live_superseded_visible,
            "translation_live_superseded_dropped": meeting.translation_live_superseded_dropped,
            "translation_live_invalidated_final": meeting.translation_live_invalidated_final,
            "translation_live_source_trimmed": meeting.translation_live_source_trimmed,
            "translation_live_skipped_cumulative": meeting.translation_live_skipped_cumulative,
            "translation_live_final_guard_dropped": meeting.translation_live_final_guard_dropped,
            "translation_live_request_timeouts": meeting.translation_live_request_timeouts,
            "translation_live_reconnect_pending": meeting.translation_live_reconnect_pending,
            "translation_live_reconnect_attempts": meeting.translation_live_reconnect_attempts,
            "translation_live_reconnect_successes": meeting.translation_live_reconnect_successes,
            "translation_live_proxy_busy": meeting.translation_live_proxy_busy,
            "translation_worker_ready": meeting._translation_worker_ready(),
            "translation_live_worker_ready": bool(
                meeting.translation_live_task is not None
                and not meeting.translation_live_task.done()
            ),
            "translation_live_min_words": TRANSLATION_LIVE_MIN_WORDS,
            "translation_live_max_words": TRANSLATION_LIVE_MAX_WORDS,
            "translation_live_min_interval_ms": TRANSLATION_LIVE_MIN_INTERVAL_MS,
            "translation_stale_skipped": meeting.translation_stale_skipped,
            "translation_chunk_requests": meeting.translation_chunk_requests,
            "translation_chunk_failures": meeting.translation_chunk_failures,
            "translation_chunk_recoveries": meeting.translation_chunk_recoveries,
            "translation_chunk_recovery_failures": meeting.translation_chunk_recovery_failures,
            "translation_chunk_retry_suppressed": meeting.translation_chunk_retry_suppressed,
            "translation_identical_retry_skipped": meeting.translation_identical_retry_skipped,
            "translation_queue_coalesced": meeting.translation_queue_coalesced,
            "translation_token_limit_recoveries": meeting.translation_token_limit_recoveries,
            "translation_progressive_deferred": meeting.translation_progressive_deferred,
            "translation_stale_visible_updates": meeting.translation_stale_visible_updates,
            "translation_progressive_debounce_reloads": meeting.translation_progressive_debounce_reloads,
            "translation_progressive_lagging_visible": meeting.translation_progressive_lagging_visible,
            "translation_progressive_requeued": meeting.translation_progressive_requeued,
            "translation_progressive_final_preempted": meeting.translation_progressive_final_preempted,
            "translation_progressive_chunk_preempted": meeting.translation_progressive_chunk_preempted,
            "translation_boundary_dedupes": meeting.translation_boundary_dedupes,
            "translation_anchor_failures": meeting.translation_anchor_failures,
            "translation_roundtrip_checks": meeting.translation_roundtrip_checks,
            "translation_roundtrip_failures": meeting.translation_roundtrip_failures,
            "translation_roundtrip_recoveries": meeting.translation_roundtrip_recoveries,
            "translation_chunk_trigger_words": TRANSLATION_CHUNK_TRIGGER_WORDS,
            "translation_chunk_words": TRANSLATION_CHUNK_WORDS,
            "translation_progressive_first_words": TRANSLATION_PROGRESSIVE_FIRST_WORDS,
            "translation_progressive_min_new_words": TRANSLATION_PROGRESSIVE_MIN_NEW_WORDS,
            "translation_recovery_max_depth": TRANSLATION_RECOVERY_MAX_DEPTH,
            "translation_roundtrip_qa": TRANSLATION_ROUNDTRIP_QA,
            "translation_primary_greedy": TRANSLATION_PRIMARY_GREEDY,
            "translation_nice": TRANSLATOR_NICE,
            "translation_request_last_ms": meeting.translation_request_last_ms,
            "translation_request_max_ms": meeting.translation_request_max_ms,
            "translation_request_avg_ms": round(
                meeting.translation_request_total_ms / max(1, meeting.translation_chunk_requests), 1
            ),
            "translation_roundtrip_min_similarity": TRANSLATION_ROUNDTRIP_MIN_SIMILARITY,
            "translation_model_family": meeting.translation_model_family,
            "translation_engine": (
                "ctranslate2-opus-v20-cpu-float32"
                if meeting.translation_model_family == "opus-v20"
                else "unavailable"
            ),
            "translation_policy": "v20-final-only-opus-isolated-no-self-requeue",
            "translation_recovery_beam": 2,
            "translation_protected_terms": list(TRANSLATION_PROTECTED_TERMS),
            "vi_written_form_enabled": VI_WRITTEN_FORM_ENABLED,
            "vi_itn_percent_enabled": VI_ITN_PERCENT_ENABLED,
            "vi_itn_common_enabled": VI_ITN_COMMON_ENABLED,
            "canonical_display_separated": True,
            "canonical_display_translation_separated": True,
            "unresolved_audio_segments": meeting.unresolved_audio_segments_current,
            "spool_orphans_recovered": meeting.spool_orphans_recovered,
            "storage_pressure_events": meeting.storage_pressure_events,
            "storage_ready": meeting._storage_ready(),
            "state_min_free_mb": STATE_MIN_FREE_MB,
            "state_disk_free_mb": round(meeting._state_disk_free_bytes() / (1024 * 1024), 1) if meeting._state_disk_free_bytes() >= 0 else -1,
            "dropped_audio_seconds": round(meeting.dropped_audio_seconds, 3),
            "supported_languages": sorted(SUPPORTED_LANGUAGES),
            "auto_language_enabled": ALLOW_AUTO_LANGUAGE,
            "sample_rate": SAMPLE_RATE,
            "stream_frame_ms": STREAM_FRAME_MS,
            "stream_frame_bytes": STREAM_FRAME_BYTES,
            "ws_max_pcm_message_bytes": WS_MAX_PCM_MESSAGE_BYTES,
            "max_clients": MAX_CLIENTS,
            "pcm_max_realtime_factor": PCM_MAX_REALTIME_FACTOR,
            "pcm_rate_burst_seconds": PCM_RATE_BURST_SECONDS,
            "session_storage": "transcript-memory-final-audio-nvme-spool",
            "transcript_persistence": False,
            "final_audio_spool": str(FINAL_SPOOL_DIR),
            "allowed_web_origins": sorted(ALLOWED_WEB_ORIGINS),
            "frontend": "static-longform-v19",
        }
    )


async def room_info(request):
    room = normalized_room(request.query.get("room", "default"))
    return web.json_response({"room": room, "participants": meeting.room_clients(room), "stats": meeting.room_stats(room)})


async def history(request):
    room = normalized_room(request.query.get("room", "default"))
    try:
        limit = int(request.query.get("limit", "500"))
    except ValueError:
        limit = 500
    limit = min(max(limit, 1), MAX_HISTORY_LIMIT)
    rows = [dict(x) for x in meeting.history_cache if x.get("room") == room]
    return web.json_response(rows[-limit:])


async def export_transcript(request):
    room = normalized_room(request.query.get("room", "default"))
    fmt = str(request.query.get("format", "txt")).lower()
    rows = [dict(x) for x in meeting.history_cache if x.get("room") == room]
    if fmt == "json":
        body = json.dumps({"room": room, "exported_at": time.time(), "transcripts": rows}, ensure_ascii=False, indent=2).encode()
        content_type = "application/json"
        suffix = "json"
    else:
        lines = [f"Meeting room: {room}", ""]
        for row in rows:
            stamp = datetime.fromtimestamp(row["created_at"], tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"[{stamp}] {row['speaker']} [{row['source_language'].upper()}]")
            lines.append(row["text"])
            lines.append("")
        body = "\n".join(lines).encode("utf-8")
        content_type = "text/plain"
        suffix = "txt"
    response = web.Response(body=body, content_type=content_type, charset="utf-8")
    response.headers["Content-Disposition"] = f'attachment; filename="meeting-{room}.{suffix}"'
    return response


async def websocket(request):
    origin = request.headers.get("Origin", "")
    if origin and not origin_allowed(origin, request):
        raise web.HTTPForbidden(text="WebSocket origin not allowed")

    if len(meeting.clients) + meeting.client_admissions_inflight >= MAX_CLIENTS:
        meeting.client_limit_rejects += 1
        raise web.HTTPServiceUnavailable(text="AI Box client limit reached")

    # Reserve the slot before the first await. aiohttp runs handlers on one
    # event loop, but ws.prepare() yields; without this reservation a burst of
    # simultaneous handshakes can all pass the limit check.
    meeting.client_admissions_inflight += 1
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=1024 * 1024)
    try:
        await ws.prepare(request)
        client_id = uuid.uuid4().hex[:12]
        client = Client(ws=ws, client_id=client_id)
        meeting.clients[client_id] = client
    finally:
        meeting.client_admissions_inflight = max(
            0, meeting.client_admissions_inflight - 1
        )

    await ws.send_json(
        {
            "type": "hello",
            "client_id": client_id,
            "version": APP_VERSION,
            "server_time": time.time(),
            "capture_protocol": "continuous-v19-vit-only",
            "asr_stack": "vit-stt-only",
        }
    )

    try:
        async for message in ws:
            if client.closing:
                break
            if message.type == WSMsgType.TEXT:
                try:
                    data = json.loads(message.data)
                except Exception:
                    continue
                kind = data.get("type")

                if kind == "join":
                    if client.capture_active or client.current_id:
                        meeting.protocol_state_rejects += 1
                        await ws.send_json({
                            "type": "error",
                            "code": "join-during-capture",
                            "message": "Stop the current capture before changing room/session",
                        })
                        continue
                    old_room = client.room
                    client.name = safe_name(data.get("name"))
                    client.room = normalized_room(data.get("room"))
                    client.language_pref = safe_language_pref(data.get("language"))
                    client.joined_at = time.time()
                    client.mic_state = "off"
                    await ws.send_json(
                        {
                            "type": "joined",
                            "client_id": client.client_id,
                            "name": client.name,
                            "room": client.room,
                            "language": client.language_pref,
                            "version": APP_VERSION,
                            "capture_protocol": "continuous-v19-vit-only",
                        }
                    )
                    await meeting.broadcast_room_status(client.room)
                    if old_room != client.room:
                        await meeting.broadcast_room_status(old_room)

                elif kind == "language_pref":
                    client.language_pref = safe_language_pref(data.get("language"))
                    await meeting.broadcast_room_status(client.room)

                elif kind == "capture_start":
                    if client.capture_active and client.capture_mode == "continuous-v19":
                        if "language" in data:
                            client.language_pref = safe_language_pref(data.get("language"))
                        await ws.send_json({
                            "type": "capture_ready",
                            "mode": "server-vad-vit-only",
                            "already_active": True,
                        })
                        continue
                    if client.current_id:
                        meeting.protocol_state_rejects += 1
                        await ws.send_json({
                            "type": "error",
                            "code": "capture-mode-conflict",
                            "message": "Finish the active speech segment before starting continuous capture",
                        })
                        continue
                    if not meeting.vit_available:
                        await ws.send_json({
                            "type": "error",
                            "code": "stt-unavailable",
                            "message": "Capture unavailable: authoritative STT is not ready",
                        })
                        continue
                    if not meeting._storage_ready():
                        await ws.send_json(
                            {
                                "type": "error",
                                "code": "storage-pressure",
                                "message": "Capture unavailable: storage reserve is low",
                                "free_mb": round(
                                    max(0, meeting._state_disk_free_bytes()) / (1024 * 1024), 1
                                ),
                            }
                        )
                        continue
                    if "language" in data:
                        client.language_pref = safe_language_pref(data.get("language"))
                    client.capture_active = True
                    client.capture_mode = "continuous-v19"
                    client.pre_roll.clear()
                    client.vad_onset_frames = 0
                    client.vad_silent_frames = 0
                    client.current_id = None
                    client.current_audio.clear()
                    client.pcm_remainder.clear()
                    meeting._reset_logical_turn(client)
                    client.partial_last_bytes = 0
                    client.partial_revision = 0
                    client.partial_last_text = ""
                    client.mic_state = "listening"
                    await ws.send_json({"type": "capture_ready", "mode": "server-vad-vit-only"})
                    await meeting.broadcast_room_status(client.room)

                elif kind == "capture_stop":
                    client.capture_active = False
                    turn_id = client.turn_id
                    await meeting._flush_pcm_remainder(client)
                    if client.current_id:
                        await meeting._flush_client_segment(client, keep_overlap=False, end_reason="capture-stop")
                    if turn_id:
                        meeting._schedule_turn_finalize(client.client_id, turn_id, 0)
                    client.pre_roll.clear()
                    client.vad_onset_frames = 0
                    client.vad_silent_frames = 0
                    client.mic_state = "off"
                    await meeting.broadcast_room_status(client.room)

                elif kind == "speech_start":
                    if client.capture_active or client.current_id:
                        meeting.protocol_state_rejects += 1
                        await ws.send_json({
                            "type": "error",
                            "code": "speech-already-active",
                            "message": "Finish the active capture before starting PTT speech",
                        })
                        continue
                    if not meeting.vit_available:
                        await ws.send_json({
                            "type": "error",
                            "code": "stt-unavailable",
                            "message": "PTT unavailable: authoritative STT is not ready",
                        })
                        continue
                    if not meeting._storage_ready():
                        await ws.send_json({
                            "type": "error",
                            "code": "storage-pressure",
                            "message": "PTT unavailable: storage reserve is low",
                            "free_mb": round(
                                max(0, meeting._state_disk_free_bytes()) / (1024 * 1024), 1
                            ),
                        })
                        continue
                    if "language" in data:
                        client.language_pref = safe_language_pref(data.get("language"))
                    client.capture_mode = "ptt"
                    client.capture_active = False
                    client.pcm_remainder.clear()
                    meeting._reset_logical_turn(client)
                    meeting._begin_segment(client, turn_id=safe_utterance_id(data.get("utterance_id")))
                    client.speech_started_ms = int(data.get("started_ms") or time.time() * 1000)
                    client.mic_state = "speaking"
                    await meeting.broadcast_room_status(client.room)

                elif kind == "speech_end" and client.current_id:
                    turn_id = client.turn_id
                    await meeting._flush_pcm_remainder(client)
                    await meeting._flush_client_segment(client, keep_overlap=False, end_reason="ptt-end")
                    meeting._schedule_turn_finalize(client.client_id, turn_id, 0)
                    client.mic_state = "listening"
                    await meeting.broadcast_room_status(client.room)

                elif kind == "ping":
                    await ws.send_json({"type": "pong", "client_ts": data.get("client_ts"), "server_ts": time.time() * 1000})

            elif message.type == WSMsgType.BINARY:
                payload = bytes(message.data)
                if not payload:
                    continue
                try:
                    if (
                        client.capture_active
                        and client.capture_mode == "continuous-v19"
                        and not meeting.vit_available
                    ):
                        stopped = await meeting._stop_capture_for_stt_unavailable(client)
                        if stopped and not ws.closed:
                            await ws.send_json({
                                "type": "error",
                                "code": "stt-unavailable",
                                "message": "Capture stopped: authoritative STT became unavailable",
                            })
                        continue
                    if client.capture_active and client.capture_mode == "continuous-v19":
                        await meeting.ingest_pcm_payload(
                            client, payload, continuous=True
                        )
                    elif client.current_id:
                        await meeting.ingest_pcm_payload(
                            client, payload, continuous=False
                        )
                except ValueError as exc:
                    log.warning(
                        "closing invalid PCM client=%s reason=%s bytes=%d",
                        client.client_id,
                        exc,
                        len(payload),
                    )
                    await ws.close(code=1009, message=str(exc).encode()[:120])
                    break

            elif message.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                break
    finally:
        client.closing = True
        old_room = client.room
        turn_id = client.turn_id
        try:
            await meeting._flush_pcm_remainder(client)
        except Exception:
            log.exception("disconnect PCM tail flush failed client=%s", client.client_id)
        if client.current_id:
            try:
                await meeting._flush_client_segment(client, keep_overlap=False, end_reason="disconnect")
            except Exception:
                log.exception("disconnect flush failed client=%s", client.client_id)
        if turn_id:
            meeting._schedule_turn_finalize(client.client_id, turn_id, 0)
        meeting.clients.pop(client_id, None)
        await meeting.broadcast_room_status(old_room)

    return ws


async def options_handler(request):
    return web.Response(status=204)


async def on_startup(app):
    await meeting.start()


async def on_cleanup(app):
    await meeting.stop()


def create_app():
    app = web.Application(client_max_size=1024 * 1024, middlewares=[cors_middleware])
    app.router.add_get("/", root_info)
    app.router.add_get("/ws", websocket)
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/room", room_info)
    app.router.add_get("/api/history", history)
    app.router.add_get("/api/export", export_transcript)
    app.router.add_route("OPTIONS", "/{tail:.*}", options_handler)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_SPOOL_DIR.mkdir(parents=True, exist_ok=True)
    UNRESOLVED_DIR.mkdir(parents=True, exist_ok=True)
    log.info("HL Meet %s VIT-only API on HTTP port %d", APP_VERSION, HTTP_PORT)
    web.run_app(create_app(), host="0.0.0.0", port=HTTP_PORT)
