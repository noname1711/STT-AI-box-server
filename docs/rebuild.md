# Clean rebuild

1. Clone the repository onto a supported Linux build host.
2. Run `./scripts/verify-source.sh`.
3. Ensure the host has standard Yocto build prerequisites plus: `git`, `curl`, `tar`, `zstd`, `python3`, `python3-venv`, `gcc`, `g++`, `file`, `sha256sum`, and `rustup`.
4. Run `./scripts/rebuild.sh`.
5. Do not modify `manifests/pins.env` for a certified rebuild.

The build intentionally regenerates large artifacts instead of storing them in Git:

- native ARM64 `stt-http` / `stt-cli`
- Vietnamese Zipformer model bundle
- English Parakeet model bundle
- EnViT5 CTranslate2 translation bundle

The scripts verify certified inner binary/model hashes before BitBake packages the image.

Internal Yocto recipe filenames and upstream model identifiers may contain version numbers. Those names are part of BitBake/upstream identity and are intentionally preserved. Human-facing repository folders and scripts are versionless.
