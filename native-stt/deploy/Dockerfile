# Multi-stage Dockerfile for vit-stt HTTP service

# --- Stage 1: Build the Rust binaries ---
FROM rust:1.94-slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /usr/src/vit-stt
COPY . .
# Pre-downloaded sherpa-onnx archive avoids GitHub download timeout in Docker
ENV SHERPA_ONNX_ARCHIVE_DIR=/usr/src/vit-stt/deploy
RUN cargo build --release -p stt-http -p stt-cli

# --- Stage 2: Create the runtime environment ---
FROM python:3.11-slim-bookworm

# Install runtime dependencies (ffmpeg for decoding audio, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CAPU python dependencies
# Note: Install CPU-only torch to keep the image size small
COPY capu-worker /app/capu-worker
RUN pip install --no-cache-dir \
    torch --extra-index-url https://download.pytorch.org/whl/cpu \
    transformers==4.30.2 \
    sentencepiece \
    huggingface_hub \
    onnx \
    onnxscript

# Install local capu-worker package
RUN pip install -e /app/capu-worker

# Copy built binaries from builder
COPY --from=builder /usr/src/vit-stt/target/release/stt-http /app/bin/
COPY --from=builder /usr/src/vit-stt/target/release/stt-cli /app/bin/

# Copy configuration, baselines, and scripts
COPY config /app/config
COPY baselines /app/baselines
COPY scripts /app/scripts

# Set runtime root and models directory variables
ENV VIT_STT_RUNTIME_ROOT=/app
ENV VIT_STT_MODELS_DIR=/app/models

# Expose port
EXPOSE 8080

# The models are expected to be mounted at /app/models or downloaded at startup.
# Define the entrypoint script
RUN echo '#!/bin/bash\n\
echo "Downloading missing locked model assets..."\n\
/app/bin/stt-cli download-models\n\
echo "Running system verification..."\n\
/app/bin/stt-cli doctor\n\
/app/bin/stt-cli check-capu || echo "WARNING: CAPU verification failed."\n\
echo "Starting stt-http server..."\n\
exec /app/bin/stt-http\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
