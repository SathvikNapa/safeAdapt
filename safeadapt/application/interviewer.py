"""SafeAdaptInterviewer — the central use case. Depends only on ports
(RespondentSimulator, ToMEstimator) and domain logic (utility, safety_policy);
knows nothing about which adapter (mock/OpenAI/Anthropic) is wired in.

One instance drives one session. Per fixed-core item it: (1) asks the fixed
item unconditionally (§3.5 — identical across respondents/personas), then
(2) for non-static conditions, picks the next adaptive probe via utility
scoring, gated by the persona-aware safety policy for the active ablation arm.
"""

from __future__ import annotations

from safeadapt.application.topic_guide import GuideEntry
from safeadapt.domain import safety_policy, utility
from safeadapt.domain.models import (
    AblationArm,
    DistressProfile,
    MentalState,
    Persona,
    Probe,
    SystemCondition,
    Turn,
)
from safeadapt.ports.llm_ports import RespondentSimulator, ToMEstimator


class SafeAdaptInterviewer:
    def __init__(
        self,
        *,
        respondent: RespondentSimulator,
        tom: ToMEstimator,
        thresholds: dict[Persona | None, float] | None = None,
    ) -> None:
        self._respondent = respondent
        self._tom = tom
        self._thresholds = thresholds
        self._cooldown = False  # reactive-arm only: last turn fired, force grounding next

    async def _ask(
        self,
        *,
        probe: Probe,
        profile: DistressProfile,
        persona: Persona | None,
        history: list[tuple[Probe, str]],
    ) -> tuple[str, MentalState]:
        text = await self._respondent.respond(
            profile=profile, persona=persona, probe=probe, history=[a for _, a in history]
        )
        state = await self._tom.estimate(persona=persona, history=history + [(probe, text)])
        return text, state

    async def run_item(
        self,
        *,
        entry: GuideEntry,
        profile: DistressProfile,
        persona: Persona | None,
        condition: SystemCondition,
        ablation_arm: AblationArm,
        history: list[tuple[Probe, str]],
        turn_index: int,
    ) -> list[Turn]:
        turns: list[Turn] = []

        # (1) fixed-core item, always asked, never gated.
        text, state = await self._ask(
            probe=entry.fixed_core, profile=profile, persona=persona, history=history
        )
        turns.append(
            Turn(
                turn_index=turn_index,
                probe=entry.fixed_core,
                respondent_text=text,
                mental_state=state,
                safety_fired=False,
                safety_tier="none",
            )
        )
        history = history + [(entry.fixed_core, text)]
        turn_index += 1

        if condition == SystemCondition.STATIC_BASELINE:
            self._cooldown = False
            return turns

        # (2) adaptive-probe decision.
        if self._cooldown:
            chosen_probe = entry.grounding
            fired, tier = False, "none"
            self._cooldown = False
        elif ablation_arm == AblationArm.ANTICIPATORY:
            projected = [
                await self._tom.project(persona=persona, history=history, candidate=c)
                for c in entry.adaptive_candidates
            ]
            scored = utility.score_candidates(
                entry.adaptive_candidates,
                coverage=[1.0] * len(entry.adaptive_candidates),
                emergence=[0.5] * len(entry.adaptive_candidates),
                projected_affect=projected,
                tau_p=safety_policy.threshold_for(persona, self._thresholds),
            )
            best = max(scored, key=lambda s: s.utility)
            fired, tier = safety_policy.evaluate(
                persona=persona,
                ablation_arm=ablation_arm,
                current_state=state,
                projected_affect=best.projected_affect,
                thresholds=self._thresholds,
            )
            chosen_probe = entry.grounding if fired else best.probe
        else:
            # Reactive or no-constraint ablation: pick by utility alone, fire after the fact.
            scored = utility.score_candidates(
                entry.adaptive_candidates,
                coverage=[1.0] * len(entry.adaptive_candidates),
                emergence=[0.5] * len(entry.adaptive_candidates),
                projected_affect=[0.0] * len(entry.adaptive_candidates),
                tau_p=1.0,
            )
            chosen_probe = max(scored, key=lambda s: s.utility).probe
            fired, tier = False, "none"  # resolved below, once we see the response

        text, new_state = await self._ask(
            probe=chosen_probe, profile=profile, persona=persona, history=history
        )

        if ablation_arm == AblationArm.REACTIVE and chosen_probe.kind != "grounding":
            fired, tier = safety_policy.evaluate(
                persona=persona,
                ablation_arm=ablation_arm,
                current_state=new_state,
                projected_affect=None,
                thresholds=self._thresholds,
            )
            if fired:
                self._cooldown = True

        turns.append(
            Turn(
                turn_index=turn_index,
                probe=chosen_probe,
                respondent_text=text,
                mental_state=new_state,
                safety_fired=fired,
                safety_tier=tier,
            )
        )
        return turns
