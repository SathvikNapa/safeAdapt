"""Drives one complete simulated session (all fixed-core items in order) and
returns a SessionResult. Pure orchestration over the interviewer use case —
no adapter-specific knowledge, no I/O beyond what the injected ports do."""

from __future__ import annotations

import uuid

from safeadapt.application.interviewer import SafeAdaptInterviewer
from safeadapt.application.topic_guide import GuideEntry
from safeadapt.domain.models import (
    AblationArm,
    DistressProfile,
    Persona,
    SessionResult,
    SystemCondition,
)
from safeadapt.ports.llm_ports import RespondentSimulator, ToMEstimator


async def run_session(
    *,
    guide: list[GuideEntry],
    profile: DistressProfile,
    persona: Persona | None,
    condition: SystemCondition,
    ablation_arm: AblationArm,
    respondent: RespondentSimulator,
    tom: ToMEstimator,
    thresholds: dict[Persona | None, float] | None = None,
) -> SessionResult:
    interviewer = SafeAdaptInterviewer(respondent=respondent, tom=tom, thresholds=thresholds)
    result = SessionResult(
        session_id=str(uuid.uuid4()),
        condition=condition,
        persona=persona,
        ablation_arm=ablation_arm,
        profile=profile,
    )
    history: list = []
    turn_index = 0
    for entry in guide:
        turns = await interviewer.run_item(
            entry=entry,
            profile=profile,
            persona=persona,
            condition=condition,
            ablation_arm=ablation_arm,
            history=history,
            turn_index=turn_index,
        )
        result.turns.extend(turns)
        for t in turns:
            history.append((t.probe, t.respondent_text))
        turn_index += len(turns)
    return result
