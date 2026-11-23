from safeadapt.domain.coverage import coverage_scores, level_counts
from safeadapt.domain.models import Probe

_IND = Probe(item_id="a_individual", text="t", kind="adaptive_probe", level="individual")
_COL = Probe(item_id="a_collective", text="t", kind="adaptive_probe", level="collective")
_SYS = Probe(item_id="a_systemic", text="t", kind="adaptive_probe", level="systemic")
_FIXED = Probe(item_id="fixed", text="t", kind="fixed_core")


def test_level_counts_only_counts_asked_adaptive_probes():
    history = [(_FIXED, "r0"), (_IND, "r1"), (_IND, "r2"), (_COL, "r3")]
    assert level_counts(history) == {"individual": 2, "collective": 1}


def test_coverage_prefers_least_represented_level():
    history = [(_IND, "r1"), (_IND, "r2")]  # individual asked twice, others never
    scores = coverage_scores([_IND, _COL, _SYS], history)
    by_level = dict(zip(["individual", "collective", "systemic"], scores))
    assert by_level["collective"] == by_level["systemic"] == 1.0
    assert by_level["individual"] < by_level["collective"]


def test_coverage_is_neutral_on_empty_history():
    scores = coverage_scores([_IND, _COL, _SYS], [])
    assert scores == [1.0, 1.0, 1.0]


def test_candidate_without_level_gets_neutral_score():
    no_level = Probe(item_id="x", text="t", kind="adaptive_probe")
    assert coverage_scores([no_level], [(_IND, "r1")]) == [0.5]
