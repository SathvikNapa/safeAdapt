"""Coverage(a_t) for the utility function (§3.2) — real, not the 1.0 stub.

Adapted from Sabherwal, Goldwert, Vlasceanu & Weber (2025): a small set of
probes spanning complementary levels (individual/collective/systemic) beats
both single-mechanism depth and "kitchen sink" breadth. Here that principle
drives *probe selection*: a candidate's coverage score is highest when its
level is the least-represented one so far in the session, so utility scoring
(ALPHA * coverage + BETA * emergence) naturally rotates the interviewer
through levels across the fixed-core items rather than repeating whichever
level happens to score highest on emergence alone.

Candidates with no level (e.g. a pure placeholder) get a neutral 0.5.
"""

from __future__ import annotations

from safeadapt.domain.models import Probe


def level_counts(history: list[tuple[Probe, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for probe, _ in history:
        if probe.kind == "adaptive_probe" and probe.level:
            counts[probe.level] = counts.get(probe.level, 0) + 1
    return counts


def coverage_scores(candidates: list[Probe], history: list[tuple[Probe, str]]) -> list[float]:
    counts = level_counts(history)
    return [
        1.0 / (1 + counts.get(c.level, 0)) if c.level is not None else 0.5
        for c in candidates
    ]
