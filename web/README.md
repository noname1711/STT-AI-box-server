# HL Meet v19.1 — STT + Translation Test Console

Standalone static frontend for the HL Meet v19.1 VIT-only backend.

## Runtime contract

The frontend keeps the latest authoritative transcript row by `id`. It handles:

- `transcript_partial` for VIT LIVE preview
- `transcript` for normal transcript events
- `transcript_replace` for FINAL corrections and translation updates

HL Meet v19.1 attaches accepted translation output to the authoritative FINAL row:

```text
translation
translation_language
translation_status = translated-final-source
```

Translation is rendered only from those server-provided fields. The browser does
not call a translation API and does not translate partial text itself.

## Local test

```bash
cd hl-meet-v19.1-web-translation
python3 -m http.server 4173
```
ssh-keygen -f ~/.ssh/known_hosts -R meeting-server.local
ssh hungle@meeting-server.local

cd ~/YOCTO/web

python3 -m http.server 8000 --bind 127.0.0.1

http://localhost:8000

Open:

```text
http://localhost:4173
```

Set **AI Box backend** to the current Jetson address, for example:

```text
http://192.168.0.107:8080
```

Then Connect → Start microphone.

## What the UI shows

Primary view:

- continuous source transcript
- VIT LIVE draft
- EN↔VI translation from FINAL source only

v19.1 metrics:

- ASR queue
- dropped audio
- VIT preview errors
- VIT FINAL errors
- translation queue
- accepted translations
- translation errors
- FINAL decoder / max active paths

Translation errors reported by `/api/health` are backend quality/runtime
counters. A rejected translation is intentionally not shown as accepted text.

## Export

`Export TXT` contains both the merged source transcript and accepted
translations, followed by per-segment diagnostics.

## Microphone-selection diagnostic build

This package adds an explicit browser microphone selector and a **Refresh** button.
Press **Refresh** once to grant microphone permission and reveal device labels, choose
the intended microphone, then start capture. The selected physical input and its
reported sample rate/channel count are written to the event log.

This diagnostic build intentionally leaves the PCM worklet, backend protocol, VAD,
ASR, and translation behavior unchanged so microphone selection can be tested as a
single isolated variable.
