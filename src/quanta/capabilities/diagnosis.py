"""Capability-gap detection and diagnosis.

Failure is *self-assessed* from the solver's own output and observations; the
evaluator's verdict is never consulted. A task counts as failed when

* no available law explained the data (the flexible description won),
* the best law is inconclusive (low credence), or
* the chosen law leaves statistically significant structure in its residuals
  (this is what catches confident wrong answers).

Each failure gets a structured diagnosis. A diagnosis is a HYPOTHESIS about why the
task failed: it is recorded with its evidence and confidence, and later checked
against what the research step actually found (see `acquire.py`). The failure-mode
labels are generic descriptions of residual structure; they do not name any law.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Sequence

from ..science.hypotheses import CORE_FAMILIES, FamilyHypothesis, FitError, solve_wls

FREQ_GRID = tuple(0.3 + (6.0 - 0.3) * i / 57 for i in range(58))
N_EFF_FREQ = (6.0 - 0.3) * 9.9 / math.pi
CREDENCE_FAIL = 0.6
ALPHA = 0.01

MODES = {
    "modulated_oscillation": ("Oscillation whose amplitude changes systematically with x.",
                              "A law combining an envelope with an oscillation is missing."),
    "added_oscillation": ("Oscillation left over on top of the best law.",
                          "A law with an oscillatory component added to a trend is missing."),
    "localized_feature": ("Residual structure concentrated in one part of the domain.",
                          "A law describing a localized feature (a peak or dip confined to part of the "
                          "domain) is missing."),
    "transition": ("Residuals change level part-way through the domain.",
                   "A law describing a smooth transition between two levels is missing."),
    "systematic_trend": ("Residuals follow a smooth trend.",
                         "The available laws miss a systematic trend; a different functional form is "
                         "needed."),
    "heavy_tails": ("A few measurements deviate far more than the rest.",
                    "The noise model, not the law, may be inadequate (heavy-tailed residuals)."),
    "unexplained": ("Misfit without a recognizable residual signature.",
                    "Some structure is unexplained; the form of the missing law is unknown."),
}


@dataclass
class GapDiagnosis:
    task: str
    current_approach: str
    observed_failure: str
    likely_failure_mode: str
    known_capabilities_used: list[str]
    missing_capability_hypothesis: str
    evidence: dict
    confidence: float
    proposed_next_experiment: str
    status: str = "hypothesis"
    mode: str = "unexplained"
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _r2(y: Sequence[float], fitted: Sequence[float]) -> float:
    m = sum(y) / len(y)
    ss = sum((v - m) ** 2 for v in y)
    if ss <= 0:
        return 0.0
    return max(0.0, 1.0 - sum((a - b) ** 2 for a, b in zip(y, fitted)) / ss)


def _lsq_fit(rows, ys):
    try:
        c = solve_wls(rows, ys)
    except FitError:
        return None
    return [sum(ci * ri for ci, ri in zip(c, row)) for row in rows], c


def periodicity(xs, r) -> dict:
    """Best sinusoid in the residuals; p-value with a look-elsewhere correction.
    Under white noise, R^2 of 2 regressors ~ Beta(1, (n-3)/2)."""
    best = (0.0, None, None)
    for w in FREQ_GRID:
        rows = [(1.0, math.sin(w * x), math.cos(w * x)) for x in xs]
        out = _lsq_fit(rows, r)
        if out is None:
            continue
        r2 = _r2(r, out[0])
        if r2 > best[0]:
            best = (r2, w, out[1])
    r2, w, coef = best
    n = len(xs)
    p_single = (1.0 - r2) ** ((n - 3) / 2.0) if n > 3 else 1.0
    p_adj = min(1.0, 1.0 - (1.0 - p_single) ** N_EFF_FREQ)
    return {"r2": r2, "freq": w, "p_adj": p_adj, "coef": coef}


def modulation(xs, r, w) -> float:
    """Ratio of oscillation amplitude in the upper vs lower half of the inputs."""
    mid = sorted(xs)[len(xs) // 2]

    def amp(sel):
        pts = [(x, v) for x, v in zip(xs, r) if sel(x)]
        if len(pts) < 5:
            return None
        rows = [(1.0, math.sin(w * x), math.cos(w * x)) for x, _ in pts]
        out = _lsq_fit(rows, [v for _, v in pts])
        if out is None:
            return None
        return math.hypot(out[1][1], out[1][2])

    lo, hi = amp(lambda x: x < mid), amp(lambda x: x >= mid)
    if not lo or not hi:
        return 1.0
    return hi / lo


def localization(xs, r) -> dict:
    """Largest share of residual energy inside a window of 25% of the domain,
    relative to the share of points in that window."""
    pts = sorted(zip(xs, r))
    lo, hi = pts[0][0], pts[-1][0]
    width = 0.25 * (hi - lo)
    total = sum(v * v for _, v in pts) or 1e-300
    best = (0.0, 0.0, 0.0)
    for x0, _ in pts:
        inside = [v for x, v in pts if x0 <= x <= x0 + width]
        if len(inside) < 3:
            continue
        share = sum(v * v for v in inside) / total
        frac = len(inside) / len(pts)
        ratio = share / frac
        if ratio > best[0]:
            best = (ratio, share, x0)
    return {"ratio": best[0], "energy_share": best[1], "window_start": best[2]}


def level_shift(xs, r) -> float:
    pts = sorted(zip(xs, r))
    best = 0.0
    for i in range(3, len(pts) - 3):
        t = pts[i][0]
        rows = [(1.0, 1.0 if x >= t else 0.0) for x, _ in pts]
        out = _lsq_fit(rows, [v for _, v in pts])
        if out is not None:
            best = max(best, _r2([v for _, v in pts], out[0]))
    return best


def tail_ratio(r) -> float:
    med = sorted(abs(v) for v in r)[len(r) // 2] or 1e-300
    return max(abs(v) for v in r) / (1.4826 * med)


def trend_r2(xs, r) -> float:
    rows = [(1.0, x, x * x) for x in xs]
    out = _lsq_fit(rows, r)
    return _r2(r, out[0]) if out else 0.0


def best_named_fit(xs, ys, robust: bool = True):
    best = None
    for name in CORE_FAMILIES:
        try:
            f = FamilyHypothesis(name).fit(xs, ys, robust=robust)
        except (FitError, OverflowError, ValueError, ZeroDivisionError):
            continue
        if math.isfinite(f.bic) and (best is None or f.bic < best.bic):
            best = f
    return best


def residual_signature(xs, ys, fit) -> dict:
    r = [y - fit.predict(x) for x, y in zip(xs, ys)]
    per = periodicity(xs, r)
    sig = {"named_law": fit.name, "n": len(xs), "periodic_r2": round(per["r2"], 4),
           "periodic_p_adj": per["p_adj"], "periodic_freq": per["freq"],
           "localization": localization(xs, r), "level_shift_r2": round(level_shift(xs, r), 4),
           "tail_ratio": round(tail_ratio(r), 3), "trend_r2": round(trend_r2(xs, r), 4)}
    sig["modulation"] = round(modulation(xs, r, per["freq"]), 3) if per["freq"] else 1.0
    return sig


def classify(sig: dict) -> tuple[str, float]:
    """Map a residual signature to a failure mode and a confidence in that reading."""
    if sig["periodic_p_adj"] < ALPHA:
        conf = 1.0 - sig["periodic_p_adj"]
        m = sig["modulation"]
        if m > 2.0 or m < 0.5:
            return "modulated_oscillation", conf * min(1.0, abs(math.log(m)) / math.log(4))
        return "added_oscillation", conf
    loc = sig["localization"]
    if loc["ratio"] > 2.2 and loc["energy_share"] > 0.5:
        return "localized_feature", min(1.0, (loc["ratio"] - 1.0) / 3.0)
    if sig["level_shift_r2"] > 0.5:
        return "transition", sig["level_shift_r2"]
    if sig["trend_r2"] > 0.4:
        return "systematic_trend", sig["trend_r2"]
    if sig["tail_ratio"] > 6.0:
        return "heavy_tails", min(1.0, (sig["tail_ratio"] - 6.0) / 6.0 + 0.5)
    return "unexplained", 0.3


def detect_failure(output: dict) -> str | None:
    """Self-assessed failure from the solver's output alone."""
    if output.get("error"):
        return f"solver error: {output['error'][:200]}"
    if output.get("answer") == "none":
        return "no available law explained the data (a flexible description beat every law)"
    if output.get("credence", 1.0) < CREDENCE_FAIL:
        return f"inconclusive: the best law has credence {output.get('credence', 0):.2f}"
    return None


def diagnose(task_id: str, output: dict, known_capabilities: Sequence[str]) -> GapDiagnosis | None:
    """Return a diagnosis if the task failed (including confident answers whose law
    leaves significant residual structure), else None."""
    obs = output.get("observations") or []
    if len(obs) < 8:
        return None
    xs = [o[0] for o in obs]
    ys = [o[1] for o in obs]
    failure = detect_failure(output)
    fit = best_named_fit(xs, ys)
    if fit is None:
        return None
    sig = residual_signature(xs, ys, fit)
    if failure is None:
        # the answer looked fine to the solver; is there significant leftover structure?
        if output.get("kind") == "named" and sig["periodic_p_adj"] < ALPHA / 10:
            failure = (f"the chosen law '{output.get('answer')}' leaves a significant periodic residual "
                       f"(p={sig['periodic_p_adj']:.1e})")
        else:
            return None
    mode, conf = classify(sig)
    desc, hyp = MODES[mode]
    return GapDiagnosis(
        task=task_id,
        current_approach=(f"named-law competition ({len(CORE_FAMILIES)} core laws + "
                          f"{len(known_capabilities)} acquired capabilities) with a flexible baseline"),
        observed_failure=failure,
        likely_failure_mode=desc,
        known_capabilities_used=list(CORE_FAMILIES) + list(known_capabilities),
        missing_capability_hypothesis=hyp,
        evidence={k: v for k, v in sig.items()},
        confidence=round(conf, 3),
        proposed_next_experiment=("Search the program space for laws that explain these observations; "
                                  "keep a law only if it predicts held-out observations better than the "
                                  "flexible description, then test it on other tasks with the same failure."),
        mode=mode,
    )
