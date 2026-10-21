"""Logique de calibrage du garde-fou (scripts/evaluate.py), sur des scores réalistes de nomic-embed-text."""
from scripts.evaluate import gate_refusals, passes_gate, suggest_thresholds


def _s(sem, kw):
    return {"question": "q", "sem": sem, "kw": kw}


# Ordres de grandeur observés : les cosinus nomic sont élevés même hors sujet.
VALID = [_s(0.710, 0.171), _s(0.764, 0.307), _s(0.80, 0.55), _s(0.74, 0.30)]
OFF_TOPIC = [_s(0.735, 0.138), _s(0.637, 0.111), _s(0.60, 0.0), _s(0.58, 0.12)]


def test_default_threshold_is_a_noop_with_realistic_scores():
    """Régression : avec 0.30, aucune question sans réponse n'était refusée."""
    wrongly, caught = gate_refusals(VALID, OFF_TOPIC, min_sem=0.30, min_kw=0.15)
    assert (wrongly, caught) == (0, 0)


def test_suggested_thresholds_keep_a_safety_margin_below_weakest_valid_question():
    sem, kw = suggest_thresholds(VALID)
    assert sem == 0.66 and kw == 0.15


def test_suggested_thresholds_never_refuse_a_valid_question_and_catch_some_off_topic():
    sem, kw = suggest_thresholds(VALID)
    wrongly, caught = gate_refusals(VALID, OFF_TOPIC, sem, kw)
    assert wrongly == 0 and caught >= 2


def test_gate_passes_when_either_signal_is_strong():
    assert passes_gate(_s(0.9, 0.0), 0.66, 0.15)
    assert passes_gate(_s(0.1, 0.4), 0.66, 0.15)
    assert not passes_gate(_s(0.5, 0.05), 0.66, 0.15)


def test_thresholds_are_never_negative():
    assert suggest_thresholds([_s(0.02, 0.01)]) == (0.0, 0.0)
