from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class WorkerRequest:
    command: str
    text: str


@dataclass(slots=True)
class WorkerResponse:
    ok: bool
    text: str | None = None
    error: str | None = None
