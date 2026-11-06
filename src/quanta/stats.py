"""Small, dependency-free statistics used for model comparison and for gating
self-improvements.

Everything here is deterministic given a seed so that every decision the
system makes about "is this better?" can be reproduced exactly.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


def mean(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("mean of empty sequence")
    return sum(xs) / len(xs)


def variance(xs: Sequence[float], ddof: int = 1) -> float:
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - ddof)


def stdev(xs: Sequence[float], ddof: int = 1) -> float:
    return math.sqrt(variance(xs, ddof))


def quantile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default 'linear' method)."""
    if not xs:
        raise ValueError("quantile of empty sequence")
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


# ---------------------------------------------------------------------------
# Special functions
# ---------------------------------------------------------------------------

def binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided sign-test p-value for k successes out of n at p=0.5."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def student_t_logpdf(r: float, scale: float, nu: float) -> float:
    z = r / scale
    return (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
            - 0.5 * math.log(nu * math.pi) - math.log(scale)
            - (nu + 1) / 2 * math.log1p(z * z / nu))


def gaussian_logpdf(r: float, scale: float) -> float:
    z = r / scale
    return -0.5 * math.log(2 * math.pi) - math.log(scale) - 0.5 * z * z


# ---------------------------------------------------------------------------
# Paired comparison used by the self-improvement gate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairedComparison:
    n: int
    mean_delta: float
    ci_low: float
    ci_high: float
    wins: int
    losses: int
    ties: int
    sign_p: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "n": self.n, "mean_delta": self.mean_delta, "ci_low": self.ci_low,
            "ci_high": self.ci_high, "wins": self.wins, "losses": self.losses,
            "ties": self.ties, "sign_p": self.sign_p, "confidence": self.confidence,
        }


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    n_boot: int = 4000,
    confidence: float = 0.95,
    seed: int = 0,
    tie_tol: float = 1e-9,
) -> PairedComparison:
    """Percentile bootstrap CI on the mean paired difference (candidate - baseline).

    Pairing matters: both systems are run on exactly the same tasks with the
    same seeds, so per-task difficulty cancels out of the comparison.
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired comparison needs equal-length samples")
    if not baseline:
        raise ValueError("paired comparison needs at least one pair")
    deltas = [c - b for b, c in zip(baseline, candidate)]
    n = len(deltas)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        boots.append(s / n)
    alpha = 1.0 - confidence
    wins = sum(1 for d in deltas if d > tie_tol)
    losses = sum(1 for d in deltas if d < -tie_tol)
    return PairedComparison(
        n=n,
        mean_delta=mean(deltas),
        ci_low=quantile(boots, alpha / 2),
        ci_high=quantile(boots, 1 - alpha / 2),
        wins=wins,
        losses=losses,
        ties=n - wins - losses,
        sign_p=binom_two_sided_p(wins, wins + losses),
        confidence=confidence,
    )
