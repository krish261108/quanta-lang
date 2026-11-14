import math

import pytest

from quanta import stats
from quanta.uncertainty import Calibration, DiminishingReturns, agreement_confidence, should_ask_human


def test_sign_test():
    assert stats.binom_two_sided_p(5, 10) == pytest.approx(1.0)
    assert stats.binom_two_sided_p(10, 10) == pytest.approx(2 / 1024)


def test_paired_bootstrap_detects_real_improvement_and_not_noise():
    import random
    rng = random.Random(1)
    base = [rng.random() for _ in range(80)]
    better = [b + 0.05 + rng.gauss(0, 0.02) for b in base]
    same = [b + rng.gauss(0, 0.02) for b in base]
    pc = stats.paired_bootstrap(base, better, seed=3)
    assert pc.ci_low > 0.03 and pc.wins > 70
    pc0 = stats.paired_bootstrap(base, same, seed=3)
    assert pc0.ci_low < 0 < pc0.ci_high
    # deterministic given the seed
    assert stats.paired_bootstrap(base, same, seed=3) == pc0


def test_calibration_metrics():
    cal = Calibration()
    for _ in range(80):
        cal.add(0.8, True)
    for _ in range(20):
        cal.add(0.8, False)
    assert cal.ece() == pytest.approx(0.0, abs=1e-9)
    assert cal.brier() == pytest.approx(0.16)
    over = Calibration()
    for _ in range(10):
        over.add(0.99, False)
    assert over.ece() > 0.9 and over.log_score() < math.log(0.05)


def test_should_ask_human():
    assert should_ask_human(0.95, 0.5).ask is False
    assert should_ask_human(0.5, 0.8).ask is True
    assert should_ask_human(0.5, 0.8, human_available=False).ask is False
    # irreversibility raises the stakes
    assert should_ask_human(0.85, 0.4, ask_cost=0.1).ask is False
    assert should_ask_human(0.85, 0.4, ask_cost=0.1, irreversible=True).ask is True


def test_diminishing_returns():
    dr = DiminishingReturns(patience=2, min_delta=0.01)
    assert not dr.update(0.1)
    assert not dr.update(0.3)
    assert not dr.update(0.5)
    assert not dr.update(0.505)
    assert dr.update(0.506)


def test_agreement_confidence():
    ans, conf = agreement_confidence(["a", "a", "b", "a"])
    assert ans == "a" and conf == pytest.approx(4 / 6)
    assert agreement_confidence([]) == (None, 0.0)
