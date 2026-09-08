#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAIL=0
pass(){ echo "PASS: $*"; }
fail(){ echo "FAIL: $*"; FAIL=1; }
sha(){ sha256sum "$1" | awk '{print $1}'; }

[[ "$(sha "$ROOT/meta-meeting-server/recipes-apps/meeting-server/files/server.py")" == "8c4829ef917984f173626fa5bbd33a9a2c2986523827198050c1ff4d8f974568" ]] && pass "meeting-server source" || fail "meeting-server drift"
[[ "$(sha "$ROOT/native-stt/Cargo.lock")" == "f7b145ffbc25311b3cbeb599e740a576c675bc915cff12345435dd375e08c8a8" ]] && pass "Cargo.lock" || fail "Cargo.lock drift"
[[ "$(sha "$ROOT/web/app.js")" == "7e2747c0f43619afe15cc94d9f149a23c1b150dc08e75c1a05a53a9850ff2a91" ]] && pass "web app.js" || fail "web app.js drift"
[[ "$(sha "$ROOT/web/pcm-worklet.js")" == "a12275b4468b9e100f0376473539b78a3b2fc20d73e9c39e6904976f795472ea" ]] && pass "web pcm-worklet.js" || fail "web worklet drift"

for f in \
  scripts/rebuild.sh \
  scripts/build-native-stt.sh \
  scripts/build-stt-models.sh \
  scripts/build-translation-model.sh \
  scripts/build-assets.sh \
  manifests/pins.env \
  manifests/build.env \
  meta-meeting-server/recipes-core/images/meeting-server-image.bb; do
  [[ -f "$ROOT/$f" ]] && pass "$f" || fail "missing $f"
done

if find "$ROOT" -type f -size +95M ! -path "$ROOT/.git/*" ! -path "$ROOT/.cache/*" -print -quit | grep -q .; then
  fail "source file >95 MiB found"
  find "$ROOT" -type f -size +95M ! -path "$ROOT/.git/*" ! -path "$ROOT/.cache/*" -printf '%s %p\n'
else
  pass "no source file >95 MiB"
fi

if find "$ROOT" -type f \( -name '*.before-*' -o -name '*.bak' -o -name '*.orig' \) -print -quit | grep -q .; then
  fail "historical/backup filename found"
  find "$ROOT" -type f \( -name '*.before-*' -o -name '*.bak' -o -name '*.orig' \) -print
else
  pass "no historical backup files"
fi

for p in \
  meta-meeting-server/recipes-ai/meeting-translation-models/files/envit5-ct2.tar \
  meta-meeting-server/recipes-ai/meeting-vit-stt/files/vit-stt-bin-orin-aarch64.tar.zst \
  meta-meeting-server/recipes-ai/meeting-vit-stt/files/vit-stt-meeting-models.tar.zst; do
  [[ ! -e "$ROOT/$p" ]] && pass "generated asset absent: $p" || fail "generated asset committed/present: $p"
done

MACHINE_PATH_PATTERN='(/home/[^/[:space:]]+/YOCTO|[$]HOME/YOCTO)'

if grep -RInE --exclude-dir=.git --exclude-dir=.cache --exclude='verify-source.sh' --include='*.sh' --include='*.py' --include='*.bb' --include='*.bbappend' --include='*.conf' --include='*.env' "$MACHINE_PATH_PATTERN" "$ROOT" >/tmp/hlmeet_clean_abs.$$ 2>/dev/null; then
  fail "machine-specific path found"
  cat /tmp/hlmeet_clean_abs.$$
else
  pass "no machine-specific source path"
fi
rm -f /tmp/hlmeet_clean_abs.$$

if find "$ROOT" -mindepth 2 -name .git -print -quit | grep -q .; then
  fail "nested .git found"
else
  pass "no nested Git repositories"
fi

grep -Fxq 'NO_FREEFORM_REWRITE_OF_CANONICAL=TRUE' "$ROOT/manifests/pins.env" && pass "canonical transcript invariant pinned" || fail "canonical transcript invariant missing"

if grep -RIn --exclude-dir=.git --exclude-dir=.cache -E 'vit-stt-bin-orin-aarch64\.tar\.zst\.sha256|vit-stt-meeting-models\.tar\.zst\.sha256' "$ROOT" >/tmp/hlmeet_old_outer.$$ 2>/dev/null; then
  fail "historical outer-tar checksum reference remains"
  cat /tmp/hlmeet_old_outer.$$
else
  pass "no historical outer-tar checksum dependency"
fi
rm -f /tmp/hlmeet_old_outer.$$

if [[ "$FAIL" -ne 0 ]]; then
  echo "SOURCE_REPO_GATE=FAIL"
  exit 2
fi

echo "SOURCE_REPO_GATE=PASS"
