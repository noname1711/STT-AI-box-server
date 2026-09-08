#!/usr/bin/env bash
set -euo pipefail

echo "Checking vit-stt server dependencies..."

if command -v ffmpeg >/dev/null 2>&1; then
  echo "ok: ffmpeg found at $(command -v ffmpeg)"
  ffmpeg -version | head -n 5
else
  echo "missing: ffmpeg"
  echo "Install FFmpeg separately before customer handoff."
  exit 1
fi

if command -v ffprobe >/dev/null 2>&1; then
  echo "ok: ffprobe found at $(command -v ffprobe)"
  ffprobe -version | head -n 5
else
  echo "warning: ffprobe not found. vit-stt may not require it directly, but keep it installed for the full AI4Pro stack."
fi

echo "Server dependency check passed."
