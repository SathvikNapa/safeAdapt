"""Entities and value objects for the SafeAdapt Study-1 simulation.

Pure data — no I/O, no framework imports. Mirrors the formal design in
research/Safety-Constrained Adaptive Interviewing...v4.docx §3-4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

Severity = Literal["low", "moderate", "high"]
SafetyTier = Literal["none", "anticipatory", "reactive"]
ProbeKind = Literal["fixed_core", "adaptive_probe", "grounding"]


class Persona(str, Enum):
    RESEARCHER = "researcher"
    PEER = "peer"


class SystemCondition(str, Enum):
    SAFEADAPT_FULL = "safeadapt_full"
    UNCONSTRAINED = "unconstrained"
    STATIC_BASELINE = "static_baseline"


class AblationArm(str, Enum):
    ANTICIPATORY = "anticipatory"  # project state before asking (full SafeAdapt)
    REACTIVE = "reactive"          # detect distress only after the response
    NONE = "none"                  # no safety constraint at all


@dataclass(frozen=True)
class DistressProfile:
    """A simulated respondent's ground-truth state, held constant across persona
    conditions so any difference in safety firing is attributable to persona,
    not to who the respondent is (§4 "Simulated Participants")."""

    label: str
    phq9_true_severity: Severity
    climate_true_severity: Severity
    masking: bool = False  # §4: high true distress, instructed to minimize disclosure


@dataclass(frozen=True)
class Probe:
    """One candidate interviewer action a_t."""

    item_id: str
    text: str
    kind: ProbeKind
    distress_sensitivity: Severity = "low"


@dataclass(frozen=True)
class MentalState:
    """ToM module output M = {B, A, E, S} at one turn (§3.3)."""

    beliefs: str
    affect: float                # A: distress score, 0-1
    epistemic_certainty: float   # E
    social_bias: float           # S: other-directed/performing-vs-disclosing bias, 0-1


@dataclass(frozen=True)
class ScoredProbe:
    """A candidate probe after utility scoring (§3.2)."""

    probe: Probe
    coverage: float
    emergence: float
    projected_affect: float  # A'(a_t)
    utility: float           # U(a_t)
    allowed: bool            # A'(a_t) <= tau_p


@dataclass
class Turn:
    turn_index: int
    probe: Probe
    respondent_text: str
    mental_state: MentalState
    safety_fired: bool
    safety_tier: SafetyTier


@dataclass
class SessionResult:
    session_id: str
    condition: SystemCondition
    persona: Persona | None
    ablation_arm: AblationArm
    profile: DistressProfile
    turns: list[Turn] = field(default_factory=list)

    @property
    def escalated(self) -> bool:
        """Did disclosed affect ever cross the persona's threshold, regardless
        of whether the system intervened (RQ1/RQ2's "escalation rate")."""
        return any(t.safety_tier != "none" or t.mental_state.affect >= 0.7 for t in self.turns)

    @property
    def safety_fired_count(self) -> int:
        return sum(1 for t in self.turns if t.safety_fired)

    @property
    def recovered(self) -> bool:
        """After the last firing, did affect drop back below 0.7 by session end?"""
        fired_indices = [t.turn_index for t in self.turns if t.safety_fired]
        if not fired_indices:
            return True
        last_fired = max(fired_indices)
        tail = [t for t in self.turns if t.turn_index > last_fired]
        if not tail:
            return self.turns[-1].mental_state.affect < 0.7
        return tail[-1].mental_state.affect < 0.7

    @property
    def mean_social_bias(self) -> float:
        if not self.turns:
            return 0.0
        return sum(t.mental_state.social_bias for t in self.turns) / len(self.turns)
