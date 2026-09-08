#!/usr/bin/env python3
import json
import os
import time
import urllib.request

URL = os.getenv("VIT_STT_READINESS_URL", "http://127.0.0.1:8091/health")
TIMEOUT = max(5, int(os.getenv("VIT_STT_READINESS_TIMEOUT", "150")))

deadline = time.monotonic() + TIMEOUT
while time.monotonic() < deadline:
    try:
        request = urllib.request.Request(URL, headers={"Connection": "close"})
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        if response.status == 200 and payload.get("status") == "ok" and payload.get("ready") is True:
            raise SystemExit(0)
    except Exception:
        pass
    time.sleep(0.5)

raise SystemExit(1)
