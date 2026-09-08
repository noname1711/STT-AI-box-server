# HL Meet STT Source

This repository is the source-only rebuild package for the HL Meet Jetson Orin Nano speech-to-text appliance.

It contains the product source, the native Rust STT runtime source, the Yocto layer, web client, model/build pins, and scripts needed to regenerate the large model/native bundles and rebuild the image.

Large generated model and binary archives are intentionally not committed. The rebuild scripts fetch pinned upstream sources/models and verify certified SHA-256 values before packaging them.

## Layout

- `meta-meeting-server/` — HL Meet Yocto layer and system services.
- `native-stt/` — native Rust STT runtime source.
- `web/` — browser microphone/test client.
- `manifests/pins.env` — exact toolchain, model, upstream revision, and certified hash pins.
- `manifests/build.env` — machine/distro/image build targets.
- `scripts/rebuild.sh` — clean rebuild entry point.
- `scripts/verify-source.sh` — static source-integrity gate.

## Verify the repository

```bash
./scripts/verify-source.sh
```

Expected result:

```text
SOURCE_REPO_GATE=PASS
```

## Rebuild

On a supported Linux/Yocto build host with Internet access:

```bash
./scripts/rebuild.sh
```

The rebuild flow downloads the exact pinned OE4T/Yocto source, builds the target SDK, rebuilds the native ARM64 STT runtime, downloads and verifies the VI/EN ASR models, regenerates the EnViT5 CTranslate2 model, and builds `meeting-server-image`.

The canonical STT transcript is never free-form rewritten. `NO_FREEFORM_REWRITE_OF_CANONICAL=TRUE` is pinned in `manifests/pins.env`.

## Reproducibility boundary

The repository is source-complete, but a practical clean rebuild also requires a compatible Linux build host, Internet access to the pinned upstream repositories/model files, and enough disk/RAM for Yocto. Final deletion of an older local Yocto tree should only happen after a fresh clone of this repository completes `./scripts/rebuild.sh` successfully.
