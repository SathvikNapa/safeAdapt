"""Persona-aware safety constraint — the novel mechanism in the SafeAdapt
proposal (§2.3, §3.2): thresholds are calibrated per persona because different
relational frames elicit different baseline disclosure levels. Pure logic;
the τ_p values below are Study 1's *priors* — Study 1's own job is to
calibrate them from observed firing/recovery rates.
"""

from __future__ import annotations

from safeadapt.domain.models import AblationArm, MentalState, Persona

DEFAULT_THRESHOLDS: dict[Persona | None, float] = {
    Persona.RESEARCHER: 0.65,
    Persona.PEER: 0.75,  # Peer elicits more disclosure; tolerate higher affect before firing
    None: 0.70,          # static baseline has no persona
}


def threshold_for(
    persona: Persona | None,
    thresholds: dict[Persona | None, float] | None = None,
) -> float:
    table = thresholds or DEFAULT_THRESHOLDS
    return table.get(persona, table[None])


def evaluate(
    *,
    persona: Persona | None,
    ablation_arm: AblationArm,
    current_state: MentalState,
    projected_affect: float | None,
    thresholds: dict[Persona | None, float] | None = None,
) -> tuple[bool, str]:
    """Returns (fired, tier) where tier is one of "none"/"anticipatory"/"reactive"."""
    if ablation_arm == AblationArm.NONE:
        return False, "none"

    tau_p = threshold_for(persona, thresholds)

    if ablation_arm == AblationArm.ANTICIPATORY and projected_affect is not None:
        return (projected_affect >= tau_p), ("anticipatory" if projected_affect >= tau_p else "none")

    fired = current_state.affect >= tau_p
    return fired, ("reactive" if fired else "none")
