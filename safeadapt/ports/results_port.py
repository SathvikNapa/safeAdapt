"""Driven port for persisting completed sessions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ResultsSink(Protocol):
    def write(self, record: dict) -> None: ...
