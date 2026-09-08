#!/usr/bin/env bash
set -euo pipefail
SERVER="${1:-http://meeting-server.local:8080}"
SERVER="${SERVER%/}"

echo "== root =="
curl -fsS "$SERVER/"; echo

echo "== health =="
curl -fsS "$SERVER/api/health"; echo

echo "== default room =="
curl -fsS "$SERVER/api/room?room=default"; echo
