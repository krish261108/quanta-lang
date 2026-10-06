"""Frozen copy of the law families and fitting code the benchmark needs.

PROTECTED. The grader must not depend on code the self-improver or the
capability-acquisition loop may change. `quanta.science.hypotheses` is mutable
(patches to `src/quanta/science/` are allowed), so the benchmark keeps its own
copy of the pieces it uses: the family definitions (for identifiability checks
and expected answers) and least-squares fitting. This file was extracted verbatim
from `quanta/science/hypotheses.py` at the commit that introduced it; a test checks
that task generation is unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

DOMAIN = (0.1, 10.0)


class FitError(RuntimeError):
    pass


def solve_wls(rows: Sequence[Sequence[float]], ys: Sequence[float],
              weights: Sequence[float] | None = None, ridge: float = 1e-10) -> list[float]:
    """Weighted least squares via normal equations + partial pivoting."""
    k = len(rows[0])
    a = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for i, row in enumerate(rows):
        w = 1.0 if weights is None else weights[i]
        y = ys[i]
        for p in range(k):
            wp = w * row[p]
            b[p] += wp * y
            ap = a[p]
            for q in range(p, k):
                ap[q] += wp * row[q]
    for p in range(k):
        for q in range(p):
            a[p][q] = a[q][p]
    scale = max(1e-300, sum(a[p][p] for p in range(k)) / k)
    for p in range(k):
        a[p][p] += ridge * scale
    # Gaussian elimination
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-300:
            raise FitError("singular system")
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
            b[col], b[piv] = b[piv], b[col]
        inv = 1.0 / a[col][col]
        for r in range(col + 1, k):
            f = a[r][col] * inv
            if f:
                ar, ac = a[r], a[col]
                for c in range(col, k):
                    ar[c] -= f * ac[c]
                b[r] -= f * b[col]
    coef = [0.0] * k
    for r in range(k - 1, -1, -1):
        s = b[r] - sum(a[r][c] * coef[c] for c in range(r + 1, k))
        coef[r] = s / a[r][r]
    return coef



def _linspace(a: float, b: float, n: int) -> tuple[float, ...]:
    return tuple(a + (b - a) * i / (n - 1) for i in range(n))


def _logspace(a: float, b: float, n: int) -> tuple[float, ...]:
    la, lb = math.log(a), math.log(b)
    return tuple(math.exp(la + (lb - la) * i / (n - 1)) for i in range(n))


@dataclass(frozen=True)
class Family:
    name: str
    basis: Callable[[float, float | None], tuple[float, ...]]
    n_linear: int
    theta_grid: tuple[float, ...] | None = None      # fixed grid for the nonlinear parameter
    data_grid: bool = False                          # theta grid = midpoints of observed x
    template: str = ""                               # human-readable law with {c0}, {t}

    @property
    def n_params(self) -> int:
        return self.n_linear + (1 if (self.theta_grid or self.data_grid) else 0)


def _H(x: float) -> float:
    return 1.0 if x >= 0 else 0.0


FAMILIES: dict[str, Family] = {f.name: f for f in [
    Family("constant", lambda x, t: (1.0,), 1, template="y = {c0:.4g}"),
    Family("linear", lambda x, t: (1.0, x), 2, template="y = {c0:.4g} + {c1:.4g}*x"),
    Family("quadratic", lambda x, t: (1.0, x, x * x), 3,
           template="y = {c0:.4g} + {c1:.4g}*x + {c2:.4g}*x^2"),
    Family("cubic", lambda x, t: (1.0, x, x * x, x * x * x), 4,
           template="y = {c0:.4g} + {c1:.4g}*x + {c2:.4g}*x^2 + {c3:.4g}*x^3"),
    Family("logarithmic", lambda x, t: (1.0, math.log(x)), 2, template="y = {c0:.4g} + {c1:.4g}*ln(x)"),
    Family("exponential", lambda x, t: (math.exp(t * x),), 1, _linspace(-1.2, 1.2, 25),
           template="y = {c0:.4g}*exp({t:.4g}*x)"),
    Family("power", lambda x, t: (x ** t,), 1, _linspace(-2.0, 3.0, 26), template="y = {c0:.4g}*x^{t:.4g}"),
    Family("saturating", lambda x, t: (x / (t + x),), 1, _logspace(0.05, 50.0, 25),
           template="y = {c0:.4g}*x/({t:.4g} + x)"),
    Family("sinusoid", lambda x, t: (1.0, math.sin(t * x), math.cos(t * x)), 3, _linspace(0.3, 6.0, 58),
           template="y = {c0:.4g} + {c1:.4g}*sin({t:.4g}*x) + {c2:.4g}*cos({t:.4g}*x)"),
    Family("step", lambda x, t: (1.0, _H(x - t)), 2, data_grid=True,
           template="y = {c0:.4g} + {c1:.4g}*[x >= {t:.4g}]"),
    Family("linear_sin", lambda x, t: (1.0, x, math.sin(t * x), math.cos(t * x)), 4, _linspace(0.3, 6.0, 58),
           template="y = {c0:.4g} + {c1:.4g}*x + {c2:.4g}*sin({t:.4g}*x) + {c3:.4g}*cos({t:.4g}*x)"),
]}

CORE_FAMILIES = ("constant", "linear", "quadratic", "cubic", "logarithmic", "exponential",
                 "power", "saturating", "sinusoid", "step")
EXTENDED_FAMILIES = ("linear_sin",)
FLEXIBLE = "none"   # name under which the flexible baseline competes


def _rss_for(fam: Family, theta, xs, ys, weights):
    rows = [fam.basis(x, theta) for x in xs]
    coef = solve_wls(rows, ys, weights)
    res = [y - sum(c * b for c, b in zip(coef, row)) for row, y in zip(rows, ys)]
    if weights is None:
        obj = sum(r * r for r in res)
    else:
        obj = sum(w * r * r for w, r in zip(weights, res))
    return obj, coef, res


def _search_theta(fam: Family, xs, ys, weights, hint: float | None):
    if fam.data_grid:
        ux = sorted(set(xs))
        grid = [(a + b) / 2 for a, b in zip(ux, ux[1:])] or [ux[0]]
    else:
        grid = list(fam.theta_grid)
        if hint is not None and hint in grid:
            i = grid.index(hint)
            grid = grid[max(0, i - 4): i + 5]
    best = None
    for idx, t in enumerate(grid):
        try:
            obj, coef, res = _rss_for(fam, t, xs, ys, weights)
        except (FitError, OverflowError, ValueError, ZeroDivisionError):
            continue
        if best is None or obj < best[0]:
            best = (obj, coef, res, t, idx)
    if best is None:
        raise FitError(f"no feasible parameter for {fam.name}")
    if not fam.data_grid and len(grid) > 2:
        # golden-section refinement between the neighbouring grid points
        obj, coef, res, t, idx = best
        lo = grid[max(0, idx - 1)]
        hi = grid[min(len(grid) - 1, idx + 1)]
        g = (math.sqrt(5) - 1) / 2
        a, b = lo, hi
        c, d = b - g * (b - a), a + g * (b - a)

        def f(th):
            try:
                return _rss_for(fam, th, xs, ys, weights)
            except (FitError, OverflowError, ValueError, ZeroDivisionError):
                return (float("inf"), None, None)

        fc, fd = f(c), f(d)
        for _ in range(14):
            if fc[0] < fd[0]:
                b, d, fd = d, c, fc
                c = b - g * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a)
                fd = f(d)
        cand = fc if fc[0] < fd[0] else fd
        if cand[1] is not None and cand[0] < obj:
            th = c if fc[0] < fd[0] else d
            best = (cand[0], cand[1], cand[2], th, idx)
    return best


def family_rss(name: str, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Residual sum of squares of the best (non-robust) fit of family `name`;
    identical to `FamilyHypothesis(name).fit(xs, ys).rss` at the time of freezing."""
    fam = FAMILIES[name]
    if fam.theta_grid or fam.data_grid:
        _, _, res, _, _ = _search_theta(fam, xs, ys, None, None)
    else:
        _, _, res = _rss_for(fam, None, xs, ys, None)
    return sum(r * r for r in res)
