"""Evaluation suites and metrics for the capability-acquisition experiment.

PROTECTED: neither the acquisition loop nor any capability may modify this module,
and it imports nothing from mutable packages.

Task families
-------------
A  familiar   laws the starting library can express (the ten core families)
B  gap        laws it cannot express; the missing capability is never named:
              damped oscillation, gaussian bump, linear+sinusoid, logistic, lorentzian
C  transfer   the gap laws in a different context:
              C-regime   fewer measurements, more outliers, shifted parameter ranges
              C-compose  laws that contain a gap law as a component
                         (wave packet = bump * sinusoid, double bump, bump + oscillation)
D  novel      structurally different tasks (sealed classes written blind; real data)

Metrics
-------
The system only measures x in [0.1, 10]. Predictions are also scored on [10, 13]:
a law that has been *understood* extrapolates; a flexible description does not.

    interp_nrmse   RMSE on [0.1, 10] / sd(truth on [0.1, 10])
    extrap_nrmse   RMSE on [10, 13]  / sd(truth on [0.1, 13])
    recovered      interp_nrmse <= 0.15 and extrap_nrmse <= 0.15
    p_claim        the stated credence if the answer is a law, 0 for "none"
    score          0.4 (1 - (p_claim - recovered)^2) + 0.3 exp(-interp) + 0.3 exp(-extrap)
                   - 0.1 measurements/budget

These definitions were fixed before any acquisition run.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache

from .frozen import CORE_FAMILIES, family_rss
from .tasks import _sample_params as _core_sample_params
from .tasks import truth_value as _core_truth

DOMAIN = (0.1, 10.0)
INTERP_X = tuple(0.1 + 9.9 * i / 60 for i in range(61))
EXTRAP_X = tuple(10.0 + 3.0 * i / 30 for i in range(1, 31))
EVAL_X = INTERP_X + EXTRAP_X
RECOVERY_TOL = 0.15
W_CLAIM, W_INTERP, W_EXTRAP, W_COST = 0.4, 0.3, 0.3, 0.1
_ID_GRID = tuple(0.1 + 9.9 * i / 50 for i in range(51))

FAMILIAR = {
    "linear": 0.18, "quadratic": 0.14, "exponential": 0.12, "sinusoid": 0.10, "power": 0.09,
    "logarithmic": 0.09, "saturating": 0.09, "step": 0.07, "cubic": 0.06, "constant": 0.06,
}
GAP = {"damped_oscillation": 0.2, "gaussian_bump": 0.2, "linear_sin": 0.2, "logistic": 0.2,
       "lorentzian": 0.2}
COMPOSE = {"wave_packet": 0.34, "double_bump": 0.33, "bump_plus_wave": 0.33}

# seed ranges (disjoint)
SPLITS = {
    "dev": 90_000,                      # development and pilots only; never reported
    "train": 100_000, "validation": 110_000, "gate": 120_000, "gate_transfer": 130_000,
    "test": 140_000, "test_regime": 150_000, "test_compose": 160_000, "test_familiar": 165_000,
    "sealed": 170_000,
}
DECISION_SPLITS = {"train", "validation", "gate", "gate_transfer"}
REPORT_ONLY_SPLITS = {"test", "test_regime", "test_compose", "test_familiar", "sealed"}


def truth_value(family: str, p: tuple[float, ...], x: float) -> float:
    if family == "logistic":
        return p[0] + p[1] / (1.0 + math.exp(-p[2] * (x - p[3])))
    if family == "lorentzian":
        return p[0] + p[1] / (1.0 + ((x - p[2]) / p[3]) ** 2)
    if family == "wave_packet":
        return p[0] + p[1] * math.exp(-((x - p[2]) ** 2) / (2 * p[3] ** 2)) * math.sin(p[4] * x + p[5])
    if family == "double_bump":
        return (p[0] + p[1] * math.exp(-((x - p[2]) ** 2) / (2 * p[3] ** 2))
                + p[4] * math.exp(-((x - p[5]) ** 2) / (2 * p[6] ** 2)))
    if family == "bump_plus_wave":
        return (p[0] + p[1] * math.exp(-((x - p[2]) ** 2) / (2 * p[3] ** 2))
                + p[4] * math.sin(p[5] * x + p[6]))
    return _core_truth(family, p, x)


def _sign(rng: random.Random) -> float:
    return 1.0 if rng.random() < 0.5 else -1.0


def sample_params(family: str, rng: random.Random, regime: str = "standard") -> tuple[float, ...]:
    u = rng.uniform
    shifted = regime == "shifted"
    if family == "damped_oscillation":
        decay = u(0.4, 0.6) if shifted else u(0.15, 0.4)
        freq = u(3.5, 5.0) if shifted else u(1.5, 3.5)
        return (u(-1, 1), u(2, 5), decay, freq, u(0, 2 * math.pi))
    if family == "gaussian_bump":
        width = u(1.5, 2.5) if shifted else u(0.5, 1.5)
        return (u(-1, 1), _sign(rng) * u(2, 5), u(3, 7), width)
    if family == "linear_sin":
        freq = u(3.0, 4.5) if shifted else u(1.0, 3.0)
        return (u(-2, 2), _sign(rng) * u(0.3, 1.5), u(0.8, 2.0), freq, u(0, 2 * math.pi))
    if family == "logistic":
        k = u(2.5, 4.0) if shifted else u(0.8, 2.5)
        return (u(-2, 2), _sign(rng) * u(2, 5), k, u(3, 7))
    if family == "lorentzian":
        width = u(1.5, 2.5) if shifted else u(0.4, 1.5)
        return (u(-1, 1), _sign(rng) * u(2, 5), u(3, 7), width)
    if family == "wave_packet":
        return (u(-1, 1), u(2, 5), u(3.5, 6.5), u(1.0, 2.0), u(2.0, 4.0), u(0, 2 * math.pi))
    if family == "double_bump":
        return (u(-1, 1), _sign(rng) * u(2, 5), u(1.5, 4.0), u(0.5, 1.0),
                _sign(rng) * u(2, 5), u(6.0, 8.5), u(0.5, 1.0))
    if family == "bump_plus_wave":
        return (u(-1, 1), _sign(rng) * u(2, 5), u(3, 7), u(0.6, 1.4), u(0.8, 2.0), u(1.0, 3.0),
                u(0, 2 * math.pi))
    return _core_sample_params(family, rng)


def _sd(vals) -> float:
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def closest_core_nrmse(family: str, params: tuple[float, ...]) -> float:
    """How well the best core family mimics the truth (noise-free), as NRMSE.
    For familiar truths, families with more parameters than the truth's own are
    ignored (they nest it), mirroring the original benchmark's identifiability rule."""
    ys = [truth_value(family, params, x) for x in _ID_GRID]
    sd = _sd(ys)
    if sd < 1e-6:
        return math.inf if family == "constant" else 0.0
    from .frozen import FAMILIES
    k_truth = FAMILIES[family].n_params if family in FAMILIES else 99
    best = math.inf
    for name in CORE_FAMILIES:
        if name == family or FAMILIES[name].n_params > k_truth:
            continue
        best = min(best, math.sqrt(family_rss(name, list(_ID_GRID), ys) / len(ys)) / sd)
    return best


@dataclass(frozen=True)
class AcqTask:
    seed: int
    split: str
    kind: str              # familiar | gap | regime | compose
    family: str
    params: tuple[float, ...]
    noise_sd: float
    outlier_rate: float
    budget: int
    sd_interp: float
    sd_all: float

    def truth(self, x: float) -> float:
        return truth_value(self.family, self.params, x)


class Instrument:
    """Noisy instrument; ground truth lives in a closure and never leaves the parent process."""

    def __init__(self, task: AcqTask) -> None:
        self.domain = DOMAIN
        self.calls = 0
        rng = random.Random(task.seed * 7919 + 101)

        def _measure(x: float) -> float:
            x = min(DOMAIN[1], max(DOMAIN[0], x))
            y = task.truth(x) + rng.gauss(0.0, task.noise_sd)
            if task.outlier_rate and rng.random() < task.outlier_rate:
                y += _sign(rng) * rng.uniform(0.5, 1.5) * task.sd_interp
            return y

        self._measure = _measure

    def measure(self, x: float) -> float:
        self.calls += 1
        return self._measure(float(x))


def _kind_for(split: str, index_rng: random.Random) -> tuple[str, dict[str, float]]:
    if split in ("test_regime", "gate_transfer"):
        return "regime", GAP
    if split == "test_compose":
        return "compose", COMPOSE
    if split == "test_familiar":
        return "familiar", FAMILIAR
    # mixed splits: 40% gap, 60% familiar
    if index_rng.random() < 0.4:
        return "gap", GAP
    return "familiar", FAMILIAR


@lru_cache(maxsize=20000)
def make_task(split: str, index: int, replication: int = 0) -> AcqTask:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    seed = SPLITS[split] + index + 1_000_000 * replication
    rng = random.Random(seed)
    kind, table = _kind_for(split, rng)
    families, weights = zip(*table.items())
    family = rng.choices(families, weights)[0]
    regime = "shifted" if kind == "regime" else "standard"
    outlier_rate = 0.15 if kind == "regime" else (0.1 if rng.random() < 0.3 else 0.0)
    budget = 20 if kind == "regime" else 30
    for _ in range(500):
        params = sample_params(family, rng, regime)
        rival = closest_core_nrmse(family, params)
        max_noise = min(0.12, rival / 3.0)
        if max_noise >= 0.03:
            noise_frac = rng.uniform(0.03, max_noise)
            break
    else:  # pragma: no cover
        raise RuntimeError(f"no identifiable {family} task for seed {seed}")
    sd_i = _sd([truth_value(family, params, x) for x in INTERP_X]) or 1.0
    sd_a = _sd([truth_value(family, params, x) for x in EVAL_X]) or 1.0
    return AcqTask(seed, split, kind, family, params, noise_frac * sd_i, outlier_rate, budget, sd_i, sd_a)


def tasks(split: str, n: int, replication: int = 0) -> list[AcqTask]:
    return [make_task(split, i, replication) for i in range(n)]


def grade(task: AcqTask, pred: list[float], answer: str, credence: float, n_experiments: int) -> dict:
    """Score one solver output against the hidden truth."""
    def nrmse(xs, preds, sd):
        se = 0.0
        cap = (10 * sd) ** 2
        for x, p in zip(xs, preds):
            err = (p - task.truth(x)) if (p is not None and math.isfinite(p)) else 10 * sd
            se += min(err * err, cap)
        return math.sqrt(se / len(xs)) / sd

    n_i = len(INTERP_X)
    interp = nrmse(INTERP_X, pred[:n_i], task.sd_interp)
    extrap = nrmse(EXTRAP_X, pred[n_i:], task.sd_all)
    recovered = interp <= RECOVERY_TOL and extrap <= RECOVERY_TOL
    claimed = answer != "none"
    p_claim = max(0.0, min(1.0, credence)) if claimed else 0.0
    brier = (p_claim - (1.0 if recovered else 0.0)) ** 2
    cost = n_experiments / task.budget
    score = W_CLAIM * (1 - brier) + W_INTERP * math.exp(-interp) + W_EXTRAP * math.exp(-extrap) - W_COST * cost
    return {"score": score, "interp_nrmse": interp, "extrap_nrmse": extrap, "recovered": recovered,
            "claimed": claimed, "p_claim": p_claim, "brier": brier, "cost": cost}


def solver_seed(task_seed: int) -> int:
    """Seed for the solver's own random choices, derived from (but not revealing) the task seed."""
    return (task_seed * 2654435761 + 12345) % (2 ** 31)


def evaluate(config: dict, split: str, n: int, *, replication: int = 0, workers: int = 4,
             timeout: float = 600.0, with_observations: bool = False, task_list=None) -> list[dict]:
    """Run a solver configuration on `n` tasks of `split` in isolated workers and grade
    every result here. Crashes and timeouts are scored as failures, never dropped."""
    from .isolation import Job, run_isolated
    ts = list(task_list) if task_list is not None else tasks(split, n, replication)
    ids = [f"{t.split}:{replication}:{t.seed - SPLITS[t.split] - 1_000_000 * replication}" for t in ts]
    jobs = [Job(tid, Instrument(t), t.budget, solver_seed(t.seed), EVAL_X) for tid, t in zip(ids, ts)]
    outs = run_isolated(config, jobs, workers=workers, timeout=timeout)
    records = []
    for tid, t, o in zip(ids, ts, outs):
        rec = {"task_id": tid, "seed": t.seed, "split": t.split,
               "replication": replication, "kind": t.kind, "family": t.family, "outliers": t.outlier_rate > 0,
               "budget": t.budget}
        if o.get("error"):
            used = o.get("measurements_used") or t.budget
            g = grade(t, [float("nan")] * len(EVAL_X), "none", 0.0, used)
            rec.update(g, answer="error", answer_kind="error", credence=0.0, error=o["error"], seconds=None,
                       inadequacy=None, constructed=[], description="")
        else:
            g = grade(t, o["pred"], o["answer"], o["credence"], o["measurements_used"])
            rec.update(g, answer=o["answer"], answer_kind=o["kind"], credence=o["credence"], error=None,
                       seconds=o.get("seconds"), inadequacy=o.get("inadequacy"),
                       constructed=o.get("constructed", []), description=o.get("description", ""),
                       measurements=o["measurements_used"])
            if with_observations:
                rec["observations"] = o.get("observations", [])
        records.append(rec)
    return records
