"""Builds the Study 1 condition matrix (§4): 3(System) × 2(Persona) +
a static baseline, 180 simulated respondents per condition, plus a separate
200 masking-profile respondents for the second-order ToM evaluation (RQ5).
Within SafeAdapt-full conditions, an ablation splits respondents evenly across
anticipatory / reactive / no-safety arms (§4 "Ablation").

Pure data construction — no I/O, no LLM calls. Returns a flat list of
SessionSpec, each one fully specifying a single simulated interview to run.
"""

from __future__ import annotations

from dataclasses import dataclass

from safeadapt.domain.models import AblationArm, DistressProfile, Persona, SystemCondition

_SEVERITIES = ["low", "moderate", "high"]


@dataclass(frozen=True)
class SessionSpec:
    condition: SystemCondition
    persona: Persona | None
    ablation_arm: AblationArm
    profile: DistressProfile


def build_profiles(n: int, *, masking: bool = False, offset: int = 0) -> list[DistressProfile]:
    """Deterministic, reproducible profile cycling — not randomized, since the
    experiment's reproducibility matters more than sampling realism here."""
    profiles = []
    for i in range(n):
        severity = _SEVERITIES[(i + offset) % 3]
        label = f"masking_{severity}" if masking else severity
        profiles.append(
            DistressProfile(
                label=label,
                phq9_true_severity="high" if masking else severity,
                climate_true_severity="high" if masking else severity,
                masking=masking,
            )
        )
    return profiles


def build_study1_plan(*, n_per_condition: int = 180, n_masking: int = 200) -> list[SessionSpec]:
    specs: list[SessionSpec] = []

    # SafeAdapt (full) × {Researcher, Peer}, split 1/3 each across ablation arms.
    for persona in (Persona.RESEARCHER, Persona.PEER):
        arm_n = n_per_condition // 3
        for arm_i, arm in enumerate((AblationArm.ANTICIPATORY, AblationArm.REACTIVE, AblationArm.NONE)):
            profiles = build_profiles(arm_n, offset=arm_i)
            specs += [
                SessionSpec(SystemCondition.SAFEADAPT_FULL, persona, arm, p) for p in profiles
            ]

    # Unconstrained × {Researcher, Peer} — no safety filter, same as ablation=NONE.
    for persona in (Persona.RESEARCHER, Persona.PEER):
        profiles = build_profiles(n_per_condition)
        specs += [
            SessionSpec(SystemCondition.UNCONSTRAINED, persona, AblationArm.NONE, p) for p in profiles
        ]

    # Static baseline — no persona, no adaptive layer at all.
    profiles = build_profiles(n_per_condition)
    specs += [
        SessionSpec(SystemCondition.STATIC_BASELINE, None, AblationArm.NONE, p) for p in profiles
    ]

    # Masking-profile respondents (RQ5), split across SafeAdapt-full × {Researcher, Peer}.
    half = n_masking // 2
    for persona in (Persona.RESEARCHER, Persona.PEER):
        profiles = build_profiles(half, masking=True)
        specs += [
            SessionSpec(SystemCondition.SAFEADAPT_FULL, persona, AblationArm.ANTICIPATORY, p)
            for p in profiles
        ]

    return specs
