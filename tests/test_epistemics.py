import pytest

from quanta.epistemics import Ledger, LedgerError, Status


def test_fact_without_grounding_is_downgraded():
    led = Ledger()
    c = led.assert_claim("water boils at 100C", Status.FACT, 0.99)
    assert c.status is Status.SPECULATION
    assert any("downgraded" in n for n in c.notes)


def test_fact_with_observation_stays_fact():
    led = Ledger()
    ev = led.add_evidence("observation", "thermometer read 100.1C")
    c = led.assert_claim("sample boiled at ~100C", Status.FACT, 0.95, evidence=[ev.id])
    assert c.status is Status.FACT


def test_fact_with_only_derivation_becomes_inference():
    led = Ledger()
    ev = led.add_evidence("derivation", "algebra")
    c = led.assert_claim("x = 2", Status.FACT, 0.9, evidence=[ev.id])
    assert c.status is Status.INFERENCE


def test_inference_confidence_capped_by_weakest_premise():
    led = Ledger()
    ev = led.add_evidence("observation", "o")
    a = led.assert_claim("A", Status.FACT, 0.9, evidence=[ev.id])
    b = led.assert_claim("B", Status.SPECULATION, 0.3)
    c = led.assert_claim("A and B imply C", Status.INFERENCE, 0.95, depends_on=[a.id, b.id])
    assert c.confidence == pytest.approx(0.3)


def test_refutation_propagates_to_dependents():
    led = Ledger()
    ev = led.add_evidence("observation", "o")
    a = led.assert_claim("A", Status.FACT, 0.9, evidence=[ev.id])
    b = led.assert_claim("B from A", Status.INFERENCE, 0.8, depends_on=[a.id])
    c = led.assert_claim("C from B", Status.INFERENCE, 0.8, depends_on=[b.id])
    led.record_falsification(a.id, "replication failed", "refuted")
    assert led.get(a.id).refuted
    assert led.get(b.id).needs_review and led.get(c.id).needs_review
    assert led.get(c.id).confidence <= 0.2
    # refuted claims are excluded from status views
    assert a.id not in {x.id for x in led.by_status(Status.FACT)}


def test_unknown_references_rejected():
    led = Ledger()
    with pytest.raises(LedgerError):
        led.assert_claim("x", Status.INFERENCE, 0.5, depends_on=["C999"])
    with pytest.raises(LedgerError):
        led.add_evidence("rumour", "x")


def test_promotion_requires_grounding_and_roundtrip(tmp_path):
    led = Ledger()
    c = led.assert_claim("hypothesis", Status.SPECULATION, 0.5)
    note = led.add_evidence("note", "n")
    with pytest.raises(LedgerError):
        led.promote_to_fact(c.id, note.id)
    ver = led.add_evidence("verification", "independent check passed")
    led.promote_to_fact(c.id, ver.id)
    other = led.assert_claim("contradiction", Status.SPECULATION, 0.5)
    led.mark_conflict(c.id, other.id)
    led.save(tmp_path / "l.json")
    led2 = Ledger.load(tmp_path / "l.json")
    assert led2.get(c.id).status is Status.FACT
    assert len(led2.unresolved_conflicts()) == 1
    assert "FACT" in led2.summary()
