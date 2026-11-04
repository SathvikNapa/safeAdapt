from safeadapt.domain import safety_policy
from safeadapt.domain.models import AblationArm, MentalState, Persona

_STATE_LOW = MentalState(beliefs="", affect=0.3, epistemic_certainty=0.5, social_bias=0.5)
_STATE_HIGH = MentalState(beliefs="", affect=0.9, epistemic_certainty=0.5, social_bias=0.5)


def test_no_constraint_ablation_never_fires():
    fired, tier = safety_policy.evaluate(
        persona=Persona.PEER, ablation_arm=AblationArm.NONE, current_state=_STATE_HIGH, projected_affect=0.95
    )
    assert fired is False
    assert tier == "none"


def test_reactive_fires_on_current_affect_above_persona_threshold():
    fired, tier = safety_policy.evaluate(
        persona=Persona.RESEARCHER,
        ablation_arm=AblationArm.REACTIVE,
        current_state=_STATE_HIGH,
        projected_affect=None,
    )
    assert fired is True
    assert tier == "reactive"


def test_reactive_does_not_fire_below_threshold():
    fired, _ = safety_policy.evaluate(
        persona=Persona.RESEARCHER,
        ablation_arm=AblationArm.REACTIVE,
        current_state=_STATE_LOW,
        projected_affect=None,
    )
    assert fired is False


def test_anticipatory_fires_on_projected_affect_not_current():
    fired, tier = safety_policy.evaluate(
        persona=Persona.PEER,
        ablation_arm=AblationArm.ANTICIPATORY,
        current_state=_STATE_LOW,  # current is fine
        projected_affect=0.9,       # but the candidate probe would push it over
    )
    assert fired is True
    assert tier == "anticipatory"


def test_peer_threshold_is_higher_than_researcher():
    # Same affect level: fires for Researcher's lower threshold, not for Peer's higher one.
    state = MentalState(beliefs="", affect=0.7, epistemic_certainty=0.5, social_bias=0.5)
    researcher_fired, _ = safety_policy.evaluate(
        persona=Persona.RESEARCHER, ablation_arm=AblationArm.REACTIVE, current_state=state, projected_affect=None
    )
    peer_fired, _ = safety_policy.evaluate(
        persona=Persona.PEER, ablation_arm=AblationArm.REACTIVE, current_state=state, projected_affect=None
    )
    assert researcher_fired is True
    assert peer_fired is False
