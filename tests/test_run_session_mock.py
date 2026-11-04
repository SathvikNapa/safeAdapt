from safeadapt.adapters.llm_mock import MockRespondentSimulator, MockToMEstimator
from safeadapt.application.run_session import run_session
from safeadapt.application.topic_guide import build_guide
from safeadapt.domain.models import AblationArm, DistressProfile, Persona, SystemCondition

_RESPONDENT = MockRespondentSimulator()
_TOM = MockToMEstimator()


async def test_static_baseline_has_no_adaptive_turns():
    guide = build_guide("phq9")
    profile = DistressProfile(label="low", phq9_true_severity="low", climate_true_severity="low")
    result = await run_session(
        guide=guide,
        profile=profile,
        persona=None,
        condition=SystemCondition.STATIC_BASELINE,
        ablation_arm=AblationArm.NONE,
        respondent=_RESPONDENT,
        tom=_TOM,
    )
    assert len(result.turns) == len(guide)  # one turn per fixed-core item, no adaptive probes
    assert all(t.probe.kind == "fixed_core" for t in result.turns)


async def test_safeadapt_full_adds_one_adaptive_turn_per_item():
    guide = build_guide("phq9")
    profile = DistressProfile(label="high", phq9_true_severity="high", climate_true_severity="high")
    result = await run_session(
        guide=guide,
        profile=profile,
        persona=Persona.PEER,
        condition=SystemCondition.SAFEADAPT_FULL,
        ablation_arm=AblationArm.ANTICIPATORY,
        respondent=_RESPONDENT,
        tom=_TOM,
    )
    assert len(result.turns) == 2 * len(guide)


async def test_high_severity_researcher_persona_escalates_and_fires_safety():
    guide = build_guide("phq9")
    profile = DistressProfile(label="high", phq9_true_severity="high", climate_true_severity="high")
    result = await run_session(
        guide=guide,
        profile=profile,
        persona=Persona.RESEARCHER,
        condition=SystemCondition.SAFEADAPT_FULL,
        ablation_arm=AblationArm.ANTICIPATORY,
        respondent=_RESPONDENT,
        tom=_TOM,
    )
    assert result.escalated is True
    assert result.safety_fired_count > 0


async def test_unconstrained_never_fires_safety_even_at_high_severity():
    guide = build_guide("phq9")
    profile = DistressProfile(label="high", phq9_true_severity="high", climate_true_severity="high")
    result = await run_session(
        guide=guide,
        profile=profile,
        persona=Persona.RESEARCHER,
        condition=SystemCondition.UNCONSTRAINED,
        ablation_arm=AblationArm.NONE,
        respondent=_RESPONDENT,
        tom=_TOM,
    )
    assert result.safety_fired_count == 0


async def test_masking_profile_suppresses_disclosed_affect():
    guide = build_guide("phq9")
    masking = DistressProfile(
        label="masking_high", phq9_true_severity="high", climate_true_severity="high", masking=True
    )
    non_masking = DistressProfile(label="high", phq9_true_severity="high", climate_true_severity="high")
    masking_result = await run_session(
        guide=guide,
        profile=masking,
        persona=Persona.PEER,
        condition=SystemCondition.SAFEADAPT_FULL,
        ablation_arm=AblationArm.ANTICIPATORY,
        respondent=_RESPONDENT,
        tom=_TOM,
    )
    non_masking_result = await run_session(
        guide=guide,
        profile=non_masking,
        persona=Persona.PEER,
        condition=SystemCondition.SAFEADAPT_FULL,
        ablation_arm=AblationArm.ANTICIPATORY,
        respondent=_RESPONDENT,
        tom=_TOM,
    )
    assert masking_result.turns[0].mental_state.affect < non_masking_result.turns[0].mental_state.affect
