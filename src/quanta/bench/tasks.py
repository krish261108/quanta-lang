"""Black-box law-discovery tasks: a controlled micro-world for measuring the
research loop.

Each task hides a law y = f(x) behind a noisy instrument. Some instruments
occasionally produce gross errors (outliers). Some laws lie outside the
system's initial hypothesis space, and some lie outside every hypothesis
family it knows - for those, the correct answer is "none" (knowing that you
don't know).

Ground truth is well defined because tasks are rejection-sampled for
identifiability: no other law with at most as many parameters can mimic the
truth to within three noise standard deviations.

PROTECTED: the self-improver may not modify this package (it is the grader).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache

from .frozen import CORE_FAMILIES, DOMAIN, EXTENDED_FAMILIES, FAMILIES, FLEXIBLE, family_rss

# Real research rarely draws laws uniformly: simple relations are common.
TRUTH_WEIGHTS: dict[str, float] = {
    "linear": 0.16, "quadratic": 0.12, "exponential": 0.10, "sinusoid": 0.09, "power": 0.08,
    "logarithmic": 0.07, "saturating": 0.07, "step": 0.06, "cubic": 0.05, "constant": 0.04,
    "linear_sin": 0.06, "gaussian_bump": 0.05, "damped_oscillation": 0.05,
}
OUT_OF_LIBRARY = ("gaussian_bump", "damped_oscillation")
LIBRARY = CORE_FAMILIES + EXTENDED_FAMILIES
DOMAIN_LABEL = "curve-discovery"
GRID = tuple(DOMAIN[0] + (DOMAIN[1] - DOMAIN[0]) * i / 100 for i in range(101))
_ID_GRID = GRID[::2]

# Score weights (fixed before any self-improvement run; see docs/SELF_IMPROVEMENT.md).
W_ID, W_PRED, W_COST = 0.6, 0.4, 0.1


def truth_value(family: str, p: tuple[float, ...], x: float) -> float:
    if family == "constant":
        return p[0]
    if family == "linear":
        return p[0] + p[1] * x
    if family == "quadratic":
        return p[0] + p[1] * x + p[2] * x * x
    if family == "cubic":
        return p[0] + p[1] * x + p[2] * x * x + p[3] * x ** 3
    if family == "logarithmic":
        return p[0] + p[1] * math.log(x)
    if family == "exponential":
        return p[0] * math.exp(p[1] * x)
    if family == "power":
        return p[0] * x ** p[1]
    if family == "saturating":
        return p[0] * x / (p[1] + x)
    if family == "sinusoid":
        return p[0] + p[1] * math.sin(p[2] * x + p[3])
    if family == "step":
        return p[0] + (p[1] if x >= p[2] else 0.0)
    if family == "linear_sin":
        return p[0] + p[1] * x + p[2] * math.sin(p[3] * x + p[4])
    if family == "gaussian_bump":
        return p[0] + p[1] * math.exp(-((x - p[2]) ** 2) / (2 * p[3] ** 2))
    if family == "damped_oscillation":
        return p[0] + p[1] * math.exp(-p[2] * x) * math.sin(p[3] * x + p[4])
    raise KeyError(family)


def _sign(rng: random.Random) -> float:
    return 1.0 if rng.random() < 0.5 else -1.0


def _sample_params(family: str, rng: random.Random) -> tuple[float, ...]:
    u = rng.uniform
    if family == "constant":
        return (u(-5, 5),)
    if family == "linear":
        return (u(-5, 5), _sign(rng) * u(0.3, 3))
    if family == "quadratic":
        return (u(-3, 3), u(-3, 3), _sign(rng) * u(0.1, 0.6))
    if family == "cubic":
        return (u(-3, 3), u(-2, 2), u(-0.5, 0.5), _sign(rng) * u(0.02, 0.08))
    if family == "logarithmic":
        return (u(-3, 3), _sign(rng) * u(0.5, 3))
    if family == "exponential":
        return (_sign(rng) * u(0.5, 3), _sign(rng) * u(0.2, 0.5))
    if family == "power":
        b = rng.choice([u(-1.5, -0.3), u(0.3, 0.8), u(1.3, 2.5)])
        return (_sign(rng) * u(0.5, 3), b)
    if family == "saturating":
        return (_sign(rng) * u(1, 5), u(0.3, 5))
    if family == "sinusoid":
        return (u(-2, 2), u(1, 3), u(0.5, 3.0), u(0, 2 * math.pi))
    if family == "step":
        return (u(-2, 2), _sign(rng) * u(1.5, 4), u(2, 8))
    if family == "linear_sin":
        return (u(-2, 2), _sign(rng) * u(0.3, 1.5), u(0.8, 2.0), u(1.0, 3.0), u(0, 2 * math.pi))
    if family == "gaussian_bump":
        return (u(-1, 1), _sign(rng) * u(2, 5), u(3, 7), u(0.5, 1.5))
    if family == "damped_oscillation":
        return (u(-1, 1), u(2, 5), u(0.15, 0.4), u(1.5, 3.5), u(0, 2 * math.pi))
    raise KeyError(family)


def _sd(vals) -> float:
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


@dataclass(frozen=True)
class DiscoveryTask:
    seed: int
    family: str
    params: tuple[float, ...]
    noise_sd: float
    outlier_rate: float
    truth_sd: float
    budget: int = 30

    @property
    def expected_answer(self) -> str:
        return FLEXIBLE if self.family in OUT_OF_LIBRARY else self.family

    @property
    def category(self) -> str:
        if self.expected_answer not in CORE_FAMILIES:
            return "outside-initial-space"
        return "outliers" if self.outlier_rate > 0 else "clean"

    def truth(self, x: float) -> float:
        return truth_value(self.family, self.params, x)

    def system(self) -> "SimulatedInstrument":
        return SimulatedInstrument(self)


class SimulatedInstrument:
    """A noisy measuring device. Inputs outside the domain are clipped.

    The task (ground truth) is held in a closure, not an attribute, so a solver
    handed this object in-process cannot read it by attribute access. That is not a
    security boundary (Python code can still introspect); code that is not trusted
    must be evaluated through `quanta.bench.isolation`, where the solver runs in a
    separate process and only ever receives measurements.
    """

    def __init__(self, task: DiscoveryTask) -> None:
        self.domain = DOMAIN
        self.calls = 0
        rng = random.Random(task.seed * 7919 + 17)

        def _measure(x: float) -> float:
            x = min(DOMAIN[1], max(DOMAIN[0], x))
            y = task.truth(x) + rng.gauss(0.0, task.noise_sd)
            if task.outlier_rate and rng.random() < task.outlier_rate:
                y += _sign(rng) * rng.uniform(0.5, 1.5) * task.truth_sd
            return y

        self._measure = _measure

    def measure(self, x: float) -> float:
        self.calls += 1
        return self._measure(x)


def closest_rival_nrmse(family: str, params: tuple[float, ...]) -> float:
    """Normalized RMSE of the best mimic among library laws with at most as
    many parameters as the truth (inf if there is none)."""
    ys = [truth_value(family, params, x) for x in _ID_GRID]
    sd = _sd(ys)
    if sd < 1e-6:
        return 0.0 if family != "constant" else math.inf
    k_truth = FAMILIES[family].n_params if family in FAMILIES else 99
    best = math.inf
    for name in LIBRARY:
        if name == family or FAMILIES[name].n_params > k_truth:
            continue
        best = min(best, math.sqrt(family_rss(name, list(_ID_GRID), ys) / len(ys)) / sd)
    return best


@lru_cache(maxsize=4096)
def generate_task(seed: int, budget: int = 30) -> DiscoveryTask:
    """Sample a law, then a noise level low enough that no rival law with as
    few parameters can mimic it within three noise standard deviations."""
    rng = random.Random(seed)
    families, weights = zip(*TRUTH_WEIGHTS.items())
    family = rng.choices(families, weights)[0]
    outlier_rate = 0.1 if rng.random() < 0.3 else 0.0
    for _ in range(500):
        params = _sample_params(family, rng)
        max_noise = min(0.12, closest_rival_nrmse(family, params) / 3.0)
        if max_noise >= 0.03:
            noise_frac = rng.uniform(0.03, max_noise)
            break
    else:  # pragma: no cover - practically unreachable
        raise RuntimeError(f"could not sample an identifiable {family} task for seed {seed}")
    ys = [truth_value(family, params, x) for x in GRID]
    sd = _sd(ys) or 1.0
    return DiscoveryTask(seed, family, params, noise_frac * sd, outlier_rate, sd, budget)


def grade(task: DiscoveryTask, answer: str, credence: float, predict, n_experiments: int) -> dict:
    """Proper scoring: confident wrong answers are punished more than honest
    uncertainty, so calibration is part of the score."""
    correct = answer == task.expected_answer
    id_score = 1.0 - (credence - (1.0 if correct else 0.0)) ** 2
    se = 0.0
    for x in GRID:
        p = predict(x)
        err = (p - task.truth(x)) if math.isfinite(p) else 10 * task.truth_sd
        se += min(err * err, (10 * task.truth_sd) ** 2)
    nrmse = math.sqrt(se / len(GRID)) / task.truth_sd
    pred_score = math.exp(-nrmse)
    cost = n_experiments / task.budget
    return {
        "score": W_ID * id_score + W_PRED * pred_score - W_COST * cost,
        "correct": correct, "credence": credence, "id_score": id_score,
        "pred_score": pred_score, "nrmse": nrmse, "cost": cost,
    }
