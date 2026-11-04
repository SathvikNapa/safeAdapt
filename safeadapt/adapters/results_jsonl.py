"""Append-only JSONL results sink — one line per completed session."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from safeadapt.domain.models import SessionResult


class JsonlResultsSink:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def session_to_record(result: SessionResult) -> dict:
    record = asdict(result)
    record["condition"] = result.condition.value
    record["persona"] = result.persona.value if result.persona else None
    record["ablation_arm"] = result.ablation_arm.value
    record["escalated"] = result.escalated
    record["safety_fired_count"] = result.safety_fired_count
    record["recovered"] = result.recovered
    record["mean_social_bias"] = result.mean_social_bias
    return record
