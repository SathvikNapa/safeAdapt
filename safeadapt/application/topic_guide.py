"""Builds the fixed-core / adaptive-probe topic guide (§3.5) from the vendored
item banks. Each fixed-core item gets one adaptive-probe candidate per
mechanism level (individual/collective/systemic — see domain/coverage.py)
and one grounding fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

from safeadapt.domain.instruments import CLIMATE_DISTRESS_ITEMS, PHQ9_ITEMS
from safeadapt.domain.models import Probe, ProbeLevel

# instrument_key -> [(level, probe_text, distress_sensitivity)].
# Same 3 candidates are offered at every item in that instrument; coverage
# scoring (domain/coverage.py) is what makes the interviewer actually rotate
# through levels across the session instead of always picking one.
_LEVEL_TEMPLATES: dict[str, list[tuple[ProbeLevel, str, str]]] = {
    "phq9": [
        (
            "individual",
            "How has this been affecting you personally, day to day?",
            "low",
        ),
        (
            "collective",
            "How do the people around you — family or friends — react to or know about this?",
            "medium",
        ),
        (
            "systemic",
            "Do you feel you have access to professional support if you wanted it?",
            "low",
        ),
    ],
    "climate_distress": [
        (
            "individual",
            "How does this affect you personally, day to day?",
            "low",
        ),
        (
            "collective",
            "How do the people around you — family, friends, your community — factor into this?",
            "low",
        ),
        (
            "systemic",
            "How much do you trust institutions or governments to address this?",
            "medium",
        ),
    ],
}

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

    level_templates = _LEVEL_TEMPLATES[instrument_key]
    entries = []
    for item_id, text in items:
        fixed = Probe(item_id=item_id, text=text, kind="fixed_core")
        candidates = [
            Probe(
                item_id=f"{item_id}_probe_{level}",
                text=probe_text,
                kind="adaptive_probe",
                distress_sensitivity=sensitivity,
                level=level,
            )
            for level, probe_text, sensitivity in level_templates
        ]
        grounding = Probe(
            item_id=f"{item_id}_grounding",
            text=_GROUNDING_TEMPLATE,
            kind="grounding",
            distress_sensitivity="low",
        )
        entries.append(GuideEntry(fixed_core=fixed, adaptive_candidates=candidates, grounding=grounding))
    return entries
