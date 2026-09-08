#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/manifests/pins.env"

fail() { echo "FAIL: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"; }
for c in rustup file tar zstd sha256sum gcc g++ ar; do need "$c"; done

SDK_ENV="${1:-${SDK_ENV:-}}"
if [[ -z "$SDK_ENV" ]]; then
    SDK_ENV="$(find "$REPO_ROOT/.cache/sdk-orin" -maxdepth 1 -type f -name 'environment-setup-*-oe4t-linux' -print -quit 2>/dev/null || true)"
fi
[[ -n "$SDK_ENV" && -f "$SDK_ENV" ]] || fail "Yocto SDK env not found. Pass it as arg1 or set SDK_ENV."

WORK="$REPO_ROOT/.cache/native-stt"
CROSS="$WORK/cross-tools"
STABLE="$WORK/stable"
TARGET_DIR="$WORK/target"
SOURCE_REAL="$REPO_ROOT/native-stt"
SOURCE_DIR="$STABLE/YOCTO/native-stt"
OUT="$REPO_ROOT/meta-meeting-server/recipes-ai/meeting-vit-stt/files/vit-stt-bin-orin-aarch64.tar.zst"

[[ "$(sha256sum "$SOURCE_REAL/Cargo.lock" | awk '{print $1}')" == "$CARGO_LOCK_SHA256" ]] || fail "Cargo.lock drift"

rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal
rustup target add --toolchain "$RUST_TOOLCHAIN" aarch64-unknown-linux-gnu

set +u
# shellcheck disable=SC1090
source "$SDK_ENV"
set -u

TARGET_CC="$CC"
TARGET_CXX="$CXX"
TARGET_AR="$AR"
TARGET_SYSROOT="$SDKTARGETSYSROOT"
TARGET_CC_BIN="${TARGET_CC%% *}"
[[ "$("$TARGET_CC_BIN" -dumpmachine)" == "aarch64-oe4t-linux" ]] || fail "SDK compiler is not aarch64-oe4t-linux"

rm -rf "$CROSS" "$STABLE" "$TARGET_DIR"
mkdir -p "$CROSS" "$(dirname "$SOURCE_DIR")" "$TARGET_DIR" "$(dirname "$OUT")"
cp -a "$SOURCE_REAL" "$SOURCE_DIR"

cat > "$CROSS/cc" <<EOF
#!/bin/sh
exec $TARGET_CC "\$@"
EOF
cat > "$CROSS/cxx" <<EOF
#!/bin/sh
exec $TARGET_CXX "\$@"
EOF
cat > "$CROSS/ar" <<EOF
#!/bin/sh
exec $TARGET_AR "\$@"
EOF
chmod 0755 "$CROSS/cc" "$CROSS/cxx" "$CROSS/ar"

unset CC CXX AR LD CPP AS NM RANLIB STRIP OBJCOPY OBJDUMP || true
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS RUSTFLAGS CARGO_ENCODED_RUSTFLAGS || true

export HOST_CC="$(command -v gcc)"
export HOST_CXX="$(command -v g++)"
export HOST_AR="$(command -v ar)"
export CC_x86_64_unknown_linux_gnu="$HOST_CC"
export CXX_x86_64_unknown_linux_gnu="$HOST_CXX"
export AR_x86_64_unknown_linux_gnu="$HOST_AR"
export CFLAGS_x86_64_unknown_linux_gnu=""
export CXXFLAGS_x86_64_unknown_linux_gnu=""

export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="$CROSS/cc"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_AR="$CROSS/ar"
export CC_aarch64_unknown_linux_gnu="$CROSS/cc"
export CXX_aarch64_unknown_linux_gnu="$CROSS/cxx"
export AR_aarch64_unknown_linux_gnu="$CROSS/ar"
export CC_AARCH64_UNKNOWN_LINUX_GNU="$CROSS/cc"
export CXX_AARCH64_UNKNOWN_LINUX_GNU="$CROSS/cxx"
export AR_AARCH64_UNKNOWN_LINUX_GNU="$CROSS/ar"
export PKG_CONFIG_ALLOW_CROSS=1
export PKG_CONFIG_SYSROOT_DIR="$TARGET_SYSROOT"
export BINDGEN_EXTRA_CLANG_ARGS="--sysroot=$TARGET_SYSROOT"
export CARGO_PROFILE_RELEASE_DEBUG=0
unset CARGO_PROFILE_RELEASE_STRIP || true
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="--remap-path-prefix=$STABLE=/usr/src/debug/vit-stt --remap-path-prefix=$HOME=/usr/src/debug/vit-stt -C debuginfo=0"

cd "$SOURCE_DIR"
CARGO_TARGET_DIR="$TARGET_DIR" \
  cargo "+$RUST_TOOLCHAIN" \
    --config "target.aarch64-unknown-linux-gnu.linker=\"$CROSS/cc\"" \
    --config "target.aarch64-unknown-linux-gnu.ar=\"$CROSS/ar\"" \
    build --release --locked --target aarch64-unknown-linux-gnu \
    -p stt-cli -p stt-http

BIN="$TARGET_DIR/aarch64-unknown-linux-gnu/release"
for b in stt-cli stt-http; do
    [[ -x "$BIN/$b" ]] || fail "missing binary: $b"
    file "$BIN/$b" | grep -q 'ARM aarch64' || fail "$b is not ARM aarch64"
done

ACT_HTTP="$(sha256sum "$BIN/stt-http" | awk '{print $1}')"
ACT_CLI="$(sha256sum "$BIN/stt-cli" | awk '{print $1}')"
[[ "$ACT_HTTP" == "$STT_HTTP_SHA256" ]] || fail "stt-http SHA drift: $ACT_HTTP"
[[ "$ACT_CLI" == "$STT_CLI_SHA256" ]] || fail "stt-cli SHA drift: $ACT_CLI"

PACK="$WORK/packroot"
rm -rf "$PACK"
mkdir -p "$PACK/bin"
install -m 0755 "$BIN/stt-http" "$PACK/bin/stt-http"
install -m 0755 "$BIN/stt-cli" "$PACK/bin/stt-cli"

TMP="$OUT.tmp"
rm -f "$TMP" "$OUT"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner --format=gnu \
  -C "$PACK" -cf - bin | zstd -19 -T1 -q -o "$TMP"
mv -f "$TMP" "$OUT"

echo "PASS: native STT rebuilt and certified binary SHA values matched"
echo "OUTPUT=$OUT"
sha256sum "$OUT"
