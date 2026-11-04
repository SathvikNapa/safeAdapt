from safeadapt.domain.models import Probe
from safeadapt.domain.utility import score_candidates, select_action

_P1 = Probe(item_id="p1", text="t1", kind="adaptive_probe", distress_sensitivity="low")
_P2 = Probe(item_id="p2", text="t2", kind="adaptive_probe", distress_sensitivity="medium")
_GROUNDING = Probe(item_id="g", text="grounding", kind="grounding")


def test_score_candidates_computes_utility_and_allowed():
    scored = score_candidates(
        [_P1, _P2], coverage=[1.0, 0.5], emergence=[0.0, 1.0], projected_affect=[0.3, 0.9], tau_p=0.7
    )
    assert scored[0].utility == 0.6 * 1.0 + 0.4 * 0.0
    assert scored[0].allowed is True
    assert scored[1].allowed is False  # projected 0.9 exceeds tau_p 0.7


def test_select_action_picks_highest_utility_allowed_candidate():
    scored = score_candidates(
        [_P1, _P2], coverage=[0.2, 0.9], emergence=[0.2, 0.9], projected_affect=[0.1, 0.1], tau_p=0.7
    )
    chosen = select_action(scored, grounding_fallback=_GROUNDING)
    assert chosen.probe is _P2  # higher coverage+emergence, both allowed


def test_select_action_falls_back_to_grounding_when_all_exceed_threshold():
    scored = score_candidates(
        [_P1, _P2], coverage=[1.0, 1.0], emergence=[1.0, 1.0], projected_affect=[0.9, 0.95], tau_p=0.7
    )
    chosen = select_action(scored, grounding_fallback=_GROUNDING)
    assert chosen.probe is _GROUNDING


def test_score_candidates_rejects_mismatched_lengths():
    try:
        score_candidates([_P1], coverage=[1.0, 2.0], emergence=[1.0], projected_affect=[1.0], tau_p=0.5)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched list lengths")
