"""Hypotheses are executable predictive models, not sentences.

A hypothesis can be fitted to data, makes quantitative predictions, and is
scored by how well it explains the data *after paying for its complexity*
(BIC). Two kinds exist:

* `FamilyHypothesis`  - a named law (linear, exponential, sinusoid, ...),
  fitted by variable projection: a 1-D search over the single nonlinear
  parameter with exact weighted least squares for the linear coefficients.
* `ExpressionHypothesis` - any formula over `x` with free parameters, parsed
  through a whitelist (no arbitrary code execution) and fitted with
  multi-start Nelder-Mead. This is how a language model proposes novel
  hypotheses that the same machinery can then test.

Robust fitting (Student-t likelihood, via iteratively reweighted least
squares) stops a few corrupted measurements from dictating the conclusion.
"""
from __future__ import annotations

import ast
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..stats import gaussian_logpdf, student_t_logpdf

DOMAIN = (0.1, 10.0)


class FitError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------

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


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------

@dataclass
class Fit:
    name: str
    n_params: int                    # free parameters of the mean function
    predict_fn: Callable[[float], float]
    params: dict[str, float]
    loglik: float
    rss: float
    n: int
    description: str = ""

    def predict(self, x: float) -> float:
        try:
            v = self.predict_fn(x)
        except (OverflowError, ValueError, ZeroDivisionError):
            return float("nan")
        return v if math.isfinite(v) else float("nan")

    @property
    def bic(self) -> float:
        """Bayesian information criterion (+1 parameter for the noise scale);
        inf when the hypothesis cannot be assessed on this little data."""
        if not math.isfinite(self.loglik):
            return math.inf
        return -2.0 * self.loglik + (self.n_params + 1) * math.log(max(self.n, 2))


MIN_RESIDUAL_DOF = 2


def _loglik(residuals: Sequence[float], *, robust: bool, nu: float,
            weights: Sequence[float] | None, floor: float, k: int) -> float:
    """Log-likelihood at the unbiased noise-scale estimate RSS/(n-k).

    Using the unbiased (rather than maximum-likelihood) scale stops
    high-capacity hypotheses from looking spuriously precise on small samples.
    Hypotheses without at least MIN_RESIDUAL_DOF residual degrees of freedom
    cannot be assessed at all and get -inf (they sit out the comparison).
    """
    n = len(residuals)
    dof = n - k
    if dof < MIN_RESIDUAL_DOF:
        return -math.inf
    if robust:
        w = weights or [1.0] * n
        s = max(math.sqrt(sum(wi * r * r for wi, r in zip(w, residuals)) / dof), floor)
        return sum(student_t_logpdf(r, s, nu) for r in residuals)
    s = max(math.sqrt(sum(r * r for r in residuals) / dof), floor)
    return sum(gaussian_logpdf(r, s) for r in residuals)


def _scale_floor(ys: Sequence[float]) -> float:
    spread = max(ys) - min(ys) if ys else 1.0
    return 1e-6 * (spread if spread > 0 else 1.0)


# ---------------------------------------------------------------------------
# Named families (variable projection)
# ---------------------------------------------------------------------------

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


class FamilyHypothesis:
    def __init__(self, family: Family | str) -> None:
        self.family = FAMILIES[family] if isinstance(family, str) else family
        self.name = self.family.name
        self.n_params = self.family.n_params
        self._hint: float | None = None

    def fit(self, xs: Sequence[float], ys: Sequence[float], *, robust: bool = False,
            nu: float = 4.0, warm: bool = False) -> Fit:
        fam = self.family
        has_theta = bool(fam.theta_grid or fam.data_grid)
        hint = self._hint if (warm and has_theta) else None
        weights = None
        floor = _scale_floor(ys)
        iters = 4 if robust else 1
        theta = None
        for it in range(iters):
            if has_theta:
                obj, coef, res, theta, _ = _search_theta(fam, xs, ys, weights, hint)
                if not fam.data_grid:
                    hint = min(fam.theta_grid, key=lambda g: abs(g - theta))
            else:
                obj, coef, res = _rss_for(fam, None, xs, ys, weights)
            if robust and it < iters - 1:
                s = max(1.4826 * _median([abs(r) for r in res]), floor)
                weights = [(nu + 1) / (nu + (r / s) ** 2) for r in res]
        if has_theta and not fam.data_grid:
            self._hint = min(fam.theta_grid, key=lambda g: abs(g - theta))
        rss = sum(r * r for r in res)
        ll = _loglik(res, robust=robust, nu=nu, weights=weights, floor=floor, k=fam.n_params)
        coef_t = tuple(coef)
        th = theta

        def predict(x, coef_t=coef_t, th=th):
            return sum(c * b for c, b in zip(coef_t, fam.basis(x, th)))

        params = {f"c{i}": c for i, c in enumerate(coef_t)}
        if th is not None:
            params["t"] = th
        desc = fam.template.format(**params) if fam.template else fam.name
        return Fit(fam.name, fam.n_params, predict, params, ll, rss, len(xs), desc)


class FlexibleHypothesis:
    """'None of the named laws': a linear spline with fixed equally spaced
    knots. It can follow almost any smooth curve but pays a large complexity
    penalty, so it wins only when every named law fits markedly worse."""

    def __init__(self, n_knots: int = 6, domain: tuple[float, float] = DOMAIN, name: str = FLEXIBLE) -> None:
        lo, hi = domain
        self.knots = tuple(lo + (hi - lo) * (i + 1) / (n_knots + 1) for i in range(n_knots))
        self.name = name
        self.n_params = n_knots + 2

    def _basis(self, x: float) -> tuple[float, ...]:
        return (1.0, x, *(max(0.0, x - k) for k in self.knots))

    def fit(self, xs, ys, *, robust: bool = False, nu: float = 4.0, warm: bool = False) -> Fit:
        floor = _scale_floor(ys)
        weights = None
        for it in range(4 if robust else 1):
            rows = [self._basis(x) for x in xs]
            coef = solve_wls(rows, ys, weights, ridge=1e-8)
            res = [y - sum(c * b for c, b in zip(coef, row)) for row, y in zip(rows, ys)]
            if robust and it < 3:
                s = max(1.4826 * _median([abs(r) for r in res]), floor)
                weights = [(nu + 1) / (nu + (r / s) ** 2) for r in res]
        coef_t = tuple(coef)
        basis = self._basis

        def predict(x):
            return sum(c * b for c, b in zip(coef_t, basis(x)))

        ll = _loglik(res, robust=robust, nu=nu, weights=weights, floor=floor, k=self.n_params)
        return Fit(self.name, self.n_params, predict, {f"c{i}": c for i, c in enumerate(coef_t)},
                   ll, sum(r * r for r in res), len(xs),
                   f"unnamed smooth curve (linear spline, {len(self.knots)} knots)")


# ---------------------------------------------------------------------------
# Safe formula hypotheses (proposed by a model or a person)
# ---------------------------------------------------------------------------

_ALLOWED_FUNCS: dict[str, Callable[..., float]] = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp, "log": math.log,
    "sqrt": math.sqrt, "abs": abs, "tanh": math.tanh, "atan": math.atan,
    "min": min, "max": max,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}
_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b}
_UNOPS = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}


class FormulaError(ValueError):
    pass


def compile_formula(formula: str, params: Sequence[str], variable: str = "x") -> Callable[[float, Sequence[float]], float]:
    """Compile `formula` into f(x, theta) using only arithmetic, whitelisted
    math functions, the variable and the declared parameters."""
    if len(formula) > 500:
        raise FormulaError("formula too long")
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"invalid formula: {e.msg}") from None
    index = {p: i for i, p in enumerate(params)}
    if variable in index:
        raise FormulaError("parameter name collides with the variable")

    def build(node):
        if isinstance(node, ast.Expression):
            return build(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            v = float(node.value)
            return lambda x, th: v
        if isinstance(node, ast.Name):
            if node.id == variable:
                return lambda x, th: x
            if node.id in index:
                i = index[node.id]
                return lambda x, th: th[i]
            if node.id in _ALLOWED_CONSTS:
                v = _ALLOWED_CONSTS[node.id]
                return lambda x, th: v
            raise FormulaError(f"unknown name {node.id!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            op = _BINOPS[type(node.op)]
            l, r = build(node.left), build(node.right)
            return lambda x, th: op(l(x, th), r(x, th))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            op = _UNOPS[type(node.op)]
            o = build(node.operand)
            return lambda x, th: op(o(x, th))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS \
                and not node.keywords:
            fn = _ALLOWED_FUNCS[node.func.id]
            args = [build(a) for a in node.args]
            return lambda x, th: fn(*(a(x, th) for a in args))
        raise FormulaError(f"disallowed syntax: {ast.dump(node)[:80]}")

    return build(tree)


def nelder_mead(f: Callable[[list[float]], float], x0: Sequence[float], *, step: float = 0.5,
                max_iter: int = 400, tol: float = 1e-10) -> tuple[list[float], float]:
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] = p[i] + (step if p[i] == 0 else step * abs(p[i]))
        simplex.append(p)
    vals = [f(p) for p in simplex]
    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * (abs(vals[0]) + tol):
            break
        centroid = [sum(p[j] for p in simplex[:-1]) / n for j in range(n)]
        worst = simplex[-1]
        refl = [c + (c - w) for c, w in zip(centroid, worst)]
        fr = f(refl)
        if fr < vals[0]:
            exp_ = [c + 2 * (c - w) for c, w in zip(centroid, worst)]
            fe = f(exp_)
            simplex[-1], vals[-1] = (exp_, fe) if fe < fr else (refl, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = refl, fr
        else:
            con = [c + 0.5 * (w - c) for c, w in zip(centroid, worst)]
            fc = f(con)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = con, fc
            else:
                best = simplex[0]
                simplex = [best] + [[b + 0.5 * (p - b) for b, p in zip(best, q)] for q in simplex[1:]]
                vals = [vals[0]] + [f(p) for p in simplex[1:]]
    i = min(range(n + 1), key=lambda i: vals[i])
    return simplex[i], vals[i]


@dataclass
class ExpressionHypothesis:
    name: str
    formula: str
    params: tuple[str, ...]
    restarts: int = 8
    seed: int = 0
    _fn: Callable = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.params = tuple(self.params)
        self._fn = compile_formula(self.formula, self.params)
        self.n_params = len(self.params)

    def fit(self, xs, ys, *, robust: bool = False, nu: float = 4.0, warm: bool = False) -> Fit:
        fn = self._fn
        floor = _scale_floor(ys)
        scale = max(1.0, max(abs(y) for y in ys))

        def residuals(th):
            out = []
            for x, y in zip(xs, ys):
                try:
                    v = fn(x, th)
                except (OverflowError, ValueError, ZeroDivisionError):
                    return None
                if isinstance(v, complex) or not math.isfinite(v):
                    return None
                out.append(y - v)
            return out

        def loss(th):
            r = residuals(th)
            if r is None:
                return float("inf")
            if robust:
                s = max(1e-3 * scale, 1e-12)
                return sum(math.log1p((ri / s) ** 2 / nu) for ri in r)
            return sum(ri * ri for ri in r)

        rng = random.Random(self.seed)
        best_th, best_v = None, float("inf")
        for k in range(self.restarts):
            x0 = [1.0] * self.n_params if k == 0 else [rng.uniform(-3, 3) for _ in range(self.n_params)]
            th, v = nelder_mead(loss, x0)
            if v < best_v:
                best_th, best_v = th, v
        if best_th is None or not math.isfinite(best_v):
            raise FitError(f"could not fit {self.name}")
        res = residuals(best_th)
        th_t = tuple(best_th)
        ll = _loglik(res, robust=robust, nu=nu, weights=None, floor=floor, k=self.n_params)
        params = dict(zip(self.params, th_t))
        return Fit(self.name, self.n_params, lambda x: fn(x, th_t), params, ll,
                   sum(r * r for r in res), len(xs),
                   f"y = {self.formula} with " + ", ".join(f"{k}={v:.4g}" for k, v in params.items()))


def make_hypotheses(names: Sequence[str]) -> list:
    out = []
    for n in names:
        out.append(FlexibleHypothesis() if n == FLEXIBLE else FamilyHypothesis(n))
    return out
