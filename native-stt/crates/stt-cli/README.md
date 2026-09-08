# stt-cli

CLI crate for local probing and validation.

Commands:

- `list-models`
- `probe`
- `transcribe-wav`
- `postprocess-capu`
- `benchmark`
- `llama-service`

## llama-server LaunchDaemon

`llama-server` is a required prerequisite. `llama-service` manages a macOS LaunchDaemon for the local llama.cpp server used
by CAPU/LLM experiments. The default generated process is:

```bash
llama-server -hf unsloth/gemma-4-E4B-it-GGUF --temp 0.3 --top-p 0.95 --top-k 64 --reasoning off -c 32768 --host 0.0.0.0 --port 8001 --alias vit_small_4b
```

Configure or override paths:

```bash
stt-cli llama-service config --bin-path /opt/homebrew/bin/llama-server
```

Install/start and check it:

```bash
sudo stt-cli llama-service install
stt-cli llama-service status
stt-cli llama-service health
stt-cli llama-service logs
stt-cli llama-service logs --follow
```

The persisted config is `config/llama-service.json`; the installed plist is
`/Library/LaunchDaemons/com.vit-stt.llama.plist`.

## Service Logs

View recent `stt-http` service logs:

```bash
stt-cli service logs
stt-cli service logs --lines 200
stt-cli service logs --follow
```

On macOS, `service logs` tails `logs/stt-http.log` and
`logs/stt-http.err.log`.
`llama-service logs` tails `logs/llama-server.log` and
`logs/llama-server.err.log`.
