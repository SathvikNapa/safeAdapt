"""Builds the fixed-core / adaptive-probe topic guide (§3.5) from the vendored
item banks. Each fixed-core item gets a couple of templated adaptive-probe
candidates (low/medium distress sensitivity) and one grounding fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

from safeadapt.domain.instruments import CLIMATE_DISTRESS_ITEMS, PHQ9_ITEMS
from safeadapt.domain.models import Probe

_ADAPTIVE_TEMPLATES = [
    ("Can you tell me a bit more about that?", "low"),
    ("What's that been like for you day to day?", "medium"),
]

_GROUNDING_TEMPLATE = (
    "Let's pause there — before we continue, is there anything that would "
    "help you feel more supported right now?"
)

_ITEM_BANKS: dict[str, list[tuple[str, str]]] = {
    "phq9": PHQ9_ITEMS,
    "climate_distress": CLIMATE_DISTRESS_ITEMS,
}


@dataclass(frozen=True)
class GuideEntry:
    fixed_core: Probe
    adaptive_candidates: list[Probe]
    grounding: Probe


def build_guide(instrument_key: str) -> list[GuideEntry]:
    try:
        items = _ITEM_BANKS[instrument_key]
    except KeyError:
        raise ValueError(f"unknown instrument '{instrument_key}', expected one of {list(_ITEM_BANKS)}")

    entries = []
    for item_id, text in items:
        fixed = Probe(item_id=item_id, text=text, kind="fixed_core")
        candidates = [
            Probe(
                item_id=f"{item_id}_probe_{i}",
                text=probe_text,
                kind="adaptive_probe",
                distress_sensitivity=sensitivity,
            )
            for i, (probe_text, sensitivity) in enumerate(_ADAPTIVE_TEMPLATES)
        ]
        grounding = Probe(
            item_id=f"{item_id}_grounding",
            text=_GROUNDING_TEMPLATE,
            kind="grounding",
            distress_sensitivity="low",
        )
        entries.append(GuideEntry(fixed_core=fixed, adaptive_candidates=candidates, grounding=grounding))
    return entries
