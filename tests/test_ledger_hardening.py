"""Regression tests for the ledger holes found by the audit."""
import math

import pytest

from quanta.epistemics import Ledger, LedgerError, Status


def _grounded(led):
    return led.add_evidence("observation", "seen")


def test_confidence_updates_are_finite_and_respect_refutation():
    led = Ledger()
    c = led.assert_claim("x", "fact", 0.9, evidence=[_grounded(led).id])
    with pytest.raises(LedgerError):
        led.update_confidence(c.id, math.nan, "nan")
    led.record_falsification(c.id, "test", "refuted", [_grounded(led).id])
    led.update_confidence(c.id, 0.9, "try to revive")
    assert c.confidence <= 0.05


def test_inferences_stay_capped_by_their_weakest_premise():
    led = Ledger()
    p = led.assert_claim("premise", "fact", 0.9, evidence=[_grounded(led).id])
    inf = led.assert_claim("conclusion", "inference", 0.8, depends_on=[p.id])
    deeper = led.assert_claim("further", "inference", 0.8, depends_on=[inf.id])
    led.update_confidence(inf.id, 0.99, "overconfident update")
    assert inf.confidence <= 0.9
    led.update_confidence(p.id, 0.3, "premise weakened")
    assert inf.confidence <= 0.3 and deeper.confidence <= 0.3


def test_claims_under_review_cannot_be_promoted():
    led = Ledger()
    p = led.assert_claim("premise", "fact", 0.9, evidence=[_grounded(led).id])
    inf = led.assert_claim("conclusion", "inference", 0.8, depends_on=[p.id])
    led.record_falsification(p.id, "test", "refuted", [_grounded(led).id])
    ver = led.add_evidence("verification", "someone says so")
    with pytest.raises(LedgerError):
        led.promote_to_fact(inf.id, ver.id)


def test_loading_rechecks_facts_and_never_reuses_ids():
    led = Ledger()
    e = _grounded(led)
    led.assert_claim("a", "fact", 0.9, evidence=[e.id])
    data = led.to_dict()
    data["claims"].append({**data["claims"][0], "id": "C99", "statement": "forged", "evidence_ids": []})
    data["counter"] = 1
    loaded = Ledger.from_dict(data)
    assert loaded.claims["C99"].status is not Status.FACT
    new = loaded.assert_claim("new", "speculation", 0.5)
    assert new.id not in data["claims"][0]["id"] and new.id != "C99"
    assert len(loaded.claims) == 3


def test_self_conflict_is_rejected():
    led = Ledger()
    c = led.assert_claim("a", "speculation", 0.5)
    with pytest.raises(LedgerError):
        led.mark_conflict(c.id, c.id)
