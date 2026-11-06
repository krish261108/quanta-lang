"""The small-sample model evidence option (Genome.evidence = "gprior")."""
from __future__ import annotations

import math
import random

from quanta.genome import Genome
from quanta.science.discovery import DiscoveryLoop, gprior_log_evidence


def test_gprior_evidence_basic_properties():
    assert gprior_log_evidence(10.0, 10.0, 20, 0) == 0.0          # the null model itself
    better = gprior_log_evidence(1.0, 10.0, 20, 2)
    worse = gprior_log_evidence(5.0, 10.0, 20, 2)
    assert better > worse > -math.inf
    assert gprior_log_evidence(1.0, 10.0, 20, 2) > gprior_log_evidence(1.0, 10.0, 20, 6)   # same fit, more params
    assert gprior_log_evidence(0.0, 10.0, 8, 7) == -math.inf     # no residual degrees of freedom


def test_flexible_curve_does_not_win_by_chance_at_ten_points():
    """At n = 10 a high-capacity model whose fit is 10x tighter by chance should not beat
    a correct 3-parameter law (BIC with a plug-in scale gives it ~38% here)."""
    n, tss = 10, 100.0
    rss_law = 0.7                      # 1 - R^2 = 0.007
    rss_flex = 0.07                    # fits the remaining two degrees of freedom 10x better
    law = gprior_log_evidence(rss_law, tss, n, 2)
    flex = gprior_log_evidence(rss_flex, tss, n, 7)
    p_flex = 1.0 / (1.0 + math.exp(law - flex))
    assert p_flex < 0.05


class _Quadratic:
    domain = (0.1, 10.0)

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def measure(self, x):
        return 1.0 + 0.5 * x - 0.08 * x * x + self.rng.gauss(0, 0.05)


def test_gprior_loop_runs_and_default_is_unchanged():
    g = Genome(falsification_rounds=1, flexible_baseline=True, likelihood="student_t", evidence="gprior")
    res = DiscoveryLoop(g, seed=1).run(_Quadratic(1))
    assert res.answer in ("quadratic", "cubic", "none") and 0.0 <= res.credence <= 1.0
    assert Genome().fingerprint() == Genome(evidence="bic").fingerprint()


class _NarrowBump:
    domain = (0.1, 10.0)

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def measure(self, x):
        return 0.5 + 3.0 * math.exp(-((x - 5.0) ** 2) / (2 * 0.35 ** 2)) + self.rng.gauss(0, 0.03)


class _Line:
    domain = (0.1, 10.0)

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def measure(self, x):
        return 1.0 + 0.7 * x + self.rng.gauss(0, 0.05)


def test_gated_capabilities_join_only_when_core_laws_fail():
    from quanta.science.synth import X, Model, SynthHypothesis, unary

    def cap():
        return SynthHypothesis(Model((unary("E", unary("Q", X)),)), name="cap:bump", origin="capability",
                               init=[-5.0, -4.0])
    g = Genome(falsification_rounds=1, flexible_baseline=True, likelihood="student_t", gate_capabilities=True)
    line = DiscoveryLoop(g, seed=2, extra_hypotheses=[cap()]).run(_Line(2))
    assert "cap:bump" not in line.hypotheses_considered
    for seed in (1, 2, 3):
        bump = DiscoveryLoop(g, seed=seed, extra_hypotheses=[cap()]).run(_NarrowBump(seed))
        assert "cap:bump" in bump.hypotheses_considered and bump.answer == "cap:bump"
    ungated = DiscoveryLoop(Genome(falsification_rounds=1, flexible_baseline=True, likelihood="student_t"),
                            seed=2, extra_hypotheses=[cap()]).run(_Line(2))
    assert "cap:bump" in ungated.hypotheses_considered
