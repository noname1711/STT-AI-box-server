#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$REPO_ROOT/manifests/pins.env"
# shellcheck disable=SC1091
source "$REPO_ROOT/manifests/build.env"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

need() {
    command -v "$1" >/dev/null 2>&1 ||
        fail "missing host command: $1"
}

for c in git curl tar zstd python3 sha256sum; do
    need "$c"
done

IMAGE_TARGET="${HLMEET_IMAGE_TARGET:-${IMAGE_TARGET:-}}"
[[ -n "$IMAGE_TARGET" ]] ||
    fail "image target is ambiguous. Set HLMEET_IMAGE_TARGET=<bitbake-image>"

#
# 1. Bootstrap exact pinned OE4T tree
#
TDD="$REPO_ROOT/tegra-demo-distro"

if [[ ! -x "$TDD/setup-env" ]]; then
    TDD="$REPO_ROOT/.cache/upstream/tegra-demo-distro"

    if [[ ! -d "$TDD/.git" ]]; then
        rm -rf "$TDD"
        git clone "$TEGRA_DEMO_DISTRO_UPSTREAM" "$TDD"
    fi

    git -C "$TDD" fetch origin "$TEGRA_DEMO_DISTRO_COMMIT"
    git -C "$TDD" checkout --detach "$TEGRA_DEMO_DISTRO_COMMIT"
    git -C "$TDD" submodule update --init --recursive
fi

BUILD_DIR="$REPO_ROOT/.cache/yocto-build"
mkdir -p "$BUILD_DIR"

set +u
# shellcheck disable=SC1090
source "$TDD/setup-env" \
    --machine "$MACHINE" \
    --distro "$DISTRO" \
    "$BUILD_DIR"
set -u

[[ -n "${BUILDDIR:-}" && -d "$BUILDDIR" ]] ||
    fail "setup-env did not create BUILDDIR"

#
# If this is a retry after an interrupted run, keep the HL Meet layer OUT
# while the bootstrap SDK is built. Its recipes intentionally reference
# generated local assets that do not exist yet.
#
if bitbake-layers show-layers 2>/dev/null |
       grep -Fq "$REPO_ROOT/meta-meeting-server"; then
    echo "INFO: temporarily removing HL Meet layer for bootstrap SDK"
    bitbake-layers remove-layer "$REPO_ROOT/meta-meeting-server"
fi

#
# 2. Build upstream target SDK first.
#
echo "===== BUILD YOCTO SDK ====="

bitbake "$SDK_IMAGE_TARGET" -c populate_sdk

SDK_INSTALLER="$(
    find "$BUILDDIR/tmp/deploy/sdk" \
        -maxdepth 1 \
        -type f \
        -name '*.sh' \
        -print 2>/dev/null |
    sort |
    tail -n1
)"

[[ -n "$SDK_INSTALLER" && -f "$SDK_INSTALLER" ]] ||
    fail "SDK installer not found"

SDK_DIR="$REPO_ROOT/.cache/sdk-orin"
rm -rf "$SDK_DIR"

bash "$SDK_INSTALLER" -y -d "$SDK_DIR"

SDK_ENV="$(
    find "$SDK_DIR" \
        -maxdepth 1 \
        -type f \
        -name 'environment-setup-*-oe4t-linux' \
        -print -quit
)"

[[ -n "$SDK_ENV" && -f "$SDK_ENV" ]] ||
    fail "installed SDK environment not found"

#
# 3. Reproduce every generated production asset.
#
echo "===== BUILD STT MODEL ASSETS ====="
"$REPO_ROOT/scripts/build-stt-models.sh"

echo "===== BUILD TRANSLATION MODEL ASSET ====="
"$REPO_ROOT/scripts/build-translation-model.sh"

echo "===== BUILD NATIVE STT ASSET ====="
"$REPO_ROOT/scripts/build-native-stt.sh" "$SDK_ENV"

#
# 4. Only now expose the HL Meet layer to BitBake.
#
echo "===== ADD HL MEET LAYER ====="

if ! bitbake-layers show-layers |
       grep -Fq "$REPO_ROOT/meta-meeting-server"; then
    bitbake-layers add-layer "$REPO_ROOT/meta-meeting-server"
fi

#
# Reproduce the currently certified lab/development image profile.
#
if ! grep -Fq 'HL_MEET_ACCESS_PROFILE = "lab"' \
        "$BUILDDIR/conf/local.conf"; then
    printf '\nHL_MEET_ACCESS_PROFILE = "lab"\n' \
        >> "$BUILDDIR/conf/local.conf"
fi

if ! grep -Fq 'LICENSE_FLAGS_ACCEPTED += "commercial"' \
        "$BUILDDIR/conf/local.conf"; then
    printf '\nLICENSE_FLAGS_ACCEPTED += "commercial"\n' \
        >> "$BUILDDIR/conf/local.conf"
fi

#
# 5. Build final HL Meet image.
#
echo "===== BUILD FINAL IMAGE ====="

bitbake "$IMAGE_TARGET"

echo
echo "PASS: fresh source rebuild completed"
echo "BUILDDIR=$BUILDDIR"
echo "IMAGE_TARGET=$IMAGE_TARGET"
