"""U(a_t) = α·Coverage(a_t) + β·Emergence(a_t), subject to A'(a_t) ≤ τ_p (§3.2).

Pure scoring/selection. Coverage, emergence, and projected affect are computed
upstream (they need a ToM port call) and passed in already-scored per candidate.
"""

from __future__ import annotations

from safeadapt.domain.models import Probe, ScoredProbe

ALPHA = 0.6  # weight on fixed-core coverage progress
BETA = 0.4   # weight on emergent-theme novelty


def score_candidates(
    candidates: list[Probe],
    *,
    coverage: list[float],
    emergence: list[float],
    projected_affect: list[float],
    tau_p: float,
) -> list[ScoredProbe]:
    if not (len(candidates) == len(coverage) == len(emergence) == len(projected_affect)):
        raise ValueError("candidates/coverage/emergence/projected_affect must be the same length")
    scored = []
    for probe, cov, emg, proj in zip(candidates, coverage, emergence, projected_affect):
        scored.append(
            ScoredProbe(
                probe=probe,
                coverage=cov,
                emergence=emg,
                projected_affect=proj,
                utility=ALPHA * cov + BETA * emg,
                allowed=proj <= tau_p,
            )
        )
    return scored


def select_action(scored: list[ScoredProbe], *, grounding_fallback: Probe) -> ScoredProbe:
    """Highest-utility allowed candidate; if every candidate exceeds τ_p, fall
    back to a validated grounding question (§3.2)."""
    allowed = [s for s in scored if s.allowed]
    if not allowed:
        return ScoredProbe(
            probe=grounding_fallback,
            coverage=0.0,
            emergence=0.0,
            projected_affect=0.0,
            utility=0.0,
            allowed=True,
        )
    return max(allowed, key=lambda s: s.utility)
