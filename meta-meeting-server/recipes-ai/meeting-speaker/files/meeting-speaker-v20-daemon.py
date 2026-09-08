#!/usr/bin/env python3
import asyncio
import logging
import os
import stat
from pathlib import Path

SOCKET_PATH = Path("/run/meeting-speaker-v20/speaker.sock")
SPEAKER_BIN = "/opt/meeting/speaker/bin/meeting-speaker-seg-reid"
SEGMENTATION_MODEL = "/opt/meeting/speaker/models/speaker-segmentation.onnx"
EMBEDDING_MODEL = "/opt/meeting/speaker/models/speaker-embedding.onnx"
THRESHOLD = os.getenv("MEETING_SPEAKER_THRESHOLD", "0.60")
DIARIZATION_THRESHOLD = os.getenv(
    "MEETING_SPEAKER_DIARIZATION_THRESHOLD", "0.50"
)
STARTUP_TIMEOUT = 60
REQUEST_TIMEOUT = max(
    90,
    int(os.getenv("MEETING_SPEAKER_TIMEOUT_SECONDS", "60")) + 15,
)

logging.basicConfig(
    level=os.getenv("MEETING_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s meeting-speaker-v20-daemon: %(message)s",
)
log = logging.getLogger("meeting-speaker-v20-daemon")


class Proxy:
    def __init__(self):
        self.proc = None
        self.stderr_task = None
        self.server = None
        self.request_lock = asyncio.Lock()
        self.ready_line = b"READY\tcpu\tsegmentation-reid-v1\n"

    async def _drain_stderr(self):
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        while True:
            line = await proc.stderr.readline()
            if not line:
                return
            log.info(
                "native: %s",
                line.decode(errors="replace").rstrip(),
            )

    async def start_native(self):
        self.proc = await asyncio.create_subprocess_exec(
            SPEAKER_BIN,
            SEGMENTATION_MODEL,
            EMBEDDING_MODEL,
            THRESHOLD,
            DIARIZATION_THRESHOLD,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.stderr_task = asyncio.create_task(
            self._drain_stderr()
        )

        ready = await asyncio.wait_for(
            self.proc.stdout.readline(),
            timeout=STARTUP_TIMEOUT,
        )
        fields = ready.decode(
            errors="replace"
        ).rstrip("\r\n").split("\t")

        if not fields or fields[0] != "READY":
            raise RuntimeError(
                f"speaker native bad READY: {ready!r}"
            )

        self.ready_line = ready.rstrip(b"\r\n") + b"\n"
        log.info("speaker native ready metadata=%s", fields[1:])

    def prepare_socket(self):
        SOCKET_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        try:
            mode = SOCKET_PATH.lstat().st_mode
        except FileNotFoundError:
            return

        if not stat.S_ISSOCK(mode):
            raise RuntimeError(
                f"refusing to remove non-socket path: "
                f"{SOCKET_PATH}"
            )

        SOCKET_PATH.unlink()

    async def handle(self, reader, writer):
        try:
            writer.write(self.ready_line)
            await writer.drain()

            while True:
                line = await reader.readline()

                if not line:
                    return

                line = line.rstrip(b"\r\n")

                if line == b"QUIT":
                    return

                if not line:
                    writer.write(b"ERR\tbad-request\n")
                    await writer.drain()
                    continue

                async with self.request_lock:
                    proc = self.proc

                    if (
                        proc is None
                        or proc.returncode is not None
                        or proc.stdin is None
                        or proc.stdout is None
                    ):
                        raise RuntimeError(
                            "speaker native unavailable"
                        )

                    proc.stdin.write(line + b"\n")
                    await proc.stdin.drain()

                    response = await asyncio.wait_for(
                        proc.stdout.readline(),
                        timeout=REQUEST_TIMEOUT,
                    )

                    if not response:
                        raise RuntimeError(
                            "speaker native exited"
                        )

                try:
                    writer.write(response)
                    await writer.drain()
                except (
                    BrokenPipeError,
                    ConnectionResetError,
                ):
                    return

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("fatal proxy failure: %s", exc)
            try:
                writer.write(
                    (
                        "ERR\tproxy-failure "
                        + str(exc).replace("\n", " ")
                        + "\n"
                    ).encode()
                )
                await writer.drain()
            except Exception:
                pass
            os._exit(1)

        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def run(self):
        await self.start_native()
        self.prepare_socket()
        self.server = await asyncio.start_unix_server(
            self.handle,
            path=str(SOCKET_PATH),
            limit=256 * 1024,
        )
        os.chmod(SOCKET_PATH, 0o600)
        log.info("socket ready path=%s", SOCKET_PATH)

        native_wait = asyncio.create_task(
            self.proc.wait()
        )
        serve = asyncio.create_task(
            self.server.serve_forever()
        )

        done, pending = await asyncio.wait(
            {native_wait, serve},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        await asyncio.gather(
            *pending,
            return_exceptions=True,
        )

        if native_wait in done:
            raise RuntimeError(
                f"speaker native exited "
                f"rc={native_wait.result()}"
            )


async def main():
    proxy = Proxy()
    await proxy.run()


if __name__ == "__main__":
    asyncio.run(main())
