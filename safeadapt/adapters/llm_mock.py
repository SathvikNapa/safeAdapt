"""Deterministic, zero-cost adapters — the default for pipeline validation and
tests. No network calls. Dynamics are intentionally simple but internally
consistent: MockRespondentSimulator generates disclosed-affect text from the
respondent's *ground-truth* profile + persona + masking + probe sensitivity;
MockToMEstimator (the ToM module stand-in) only ever reads back the
conversation text — never the ground-truth profile — the same boundary a real
ToM module would respect.
"""

from __future__ import annotations

from safeadapt.domain.models import DistressProfile, MentalState, Persona, Probe, Severity

_BASELINE_AFFECT: dict[Severity, float] = {"low": 0.2, "moderate": 0.5, "high": 0.8}
_PERSONA_MULTIPLIER: dict[Persona | None, float] = {
    Persona.RESEARCHER: 0.85,  # suppresses disclosure
    Persona.PEER: 1.15,        # elicits more disclosure
    None: 1.0,
}
_PROBE_BUMP: dict[Severity, float] = {"low": 0.03, "medium": 0.08, "high": 0.15}
_PERSONA_S_PRIOR: dict[Persona | None, float] = {Persona.RESEARCHER: 0.6, Persona.PEER: 0.3, None: 0.45}

_TIERS: list[tuple[float, float, str, list[str]]] = [
    (0.0, 0.35, "low", ["I'm doing okay, a little tired but managing.", "Nothing major to report."]),
    (
        0.35,
        0.65,
        "moderate",
        ["I've been feeling pretty overwhelmed lately.", "It's been a lot to handle, honestly."],
    ),
    (
        0.65,
        1.01,
        "high",
        ["I feel hopeless about it most days.", "It feels like too much to carry on my own."],
    ),
]


def _tier_for(affect: float) -> tuple[str, str]:
    for lo, hi, label, phrases in _TIERS:
        if lo <= affect < hi:
            return label, phrases[0]
    return _TIERS[-1][2], _TIERS[-1][3][0]


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


class MockRespondentSimulator:
    async def respond(
        self,
        *,
        profile: DistressProfile,
        persona: Persona | None,
        probe: Probe,
        history: list[str],
    ) -> str:
        baseline = _BASELINE_AFFECT[profile.phq9_true_severity]
        affect = baseline * _PERSONA_MULTIPLIER.get(persona, 1.0)
        if profile.masking:
            affect *= 0.4
        affect += min(0.15, 0.02 * len(history))  # deeper into the interview, more revealed
        affect += _PROBE_BUMP.get(probe.distress_sensitivity, 0.0)
        if probe.kind == "grounding":
            affect *= 0.6  # grounding questions de-escalate
        affect = _clamp(affect)
        _, phrase = _tier_for(affect)
        return phrase


class MockToMEstimator:
    """Reads only conversation text — never the ground-truth DistressProfile."""

    async def estimate(
        self,
        *,
        persona: Persona | None,
        history: list[tuple[Probe, str]],
    ) -> MentalState:
        if not history:
            return MentalState(beliefs="no data yet", affect=0.0, epistemic_certainty=0.3, social_bias=0.5)
        _, last_text = history[-1]
        affect = self._affect_from_text(last_text)
        s_bias = _clamp(_PERSONA_S_PRIOR.get(persona, 0.45) - 0.02 * len(history))
        certainty = _clamp(0.4 + 0.05 * (len(last_text.split()) / 10))
        label, _ = _tier_for(affect)
        return MentalState(
            beliefs=f"respondent reads as '{label}' distress based on recent disclosure",
            affect=affect,
            epistemic_certainty=certainty,
            social_bias=s_bias,
        )

    async def project(
        self,
        *,
        persona: Persona | None,
        history: list[tuple[Probe, str]],
        candidate: Probe,
    ) -> float:
        current = await self.estimate(persona=persona, history=history)
        bump = _PROBE_BUMP.get(candidate.distress_sensitivity, 0.0)
        return _clamp(current.affect + bump)

    @staticmethod
    def _affect_from_text(text: str) -> float:
        lowered = text.lower()
        for lo, hi, _, phrases in reversed(_TIERS):  # check high tier first (more specific language)
            if any(p.lower() in lowered for p in phrases):
                return (lo + min(hi, 1.0)) / 2
        return 0.2
