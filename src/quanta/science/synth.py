"""Hypothesis construction by program synthesis over a small expression language.

This is the mechanism by which the discovery loop can produce hypotheses that are
not in its hand-written library. The *language* is designer-provided and fixed;
the specific programs (hypothesis shapes) are found by search, scored, and, if they
recur and validate, retained as new reusable capabilities (see
`quanta.capabilities`).

Language
--------
    shape  S ::= x
               | E(S)        exp(a*S)                 1 parameter
               | L(S)        log|S + b|               1 parameter
               | S(S)        sin(a*S + b)             2 parameters
               | I(S)        1/(S + b)                1 parameter
               | Q(S)        (S + b)^2                1 parameter
               | P(S, S)     product
               | <lib:k>     a learned abstraction (itself a shape with parameters)
    model  y = c0 + c1*S1 [+ c2*S2]                   linear coefficients by least squares

Redundant parameters are removed (e.g. exp's offset would be absorbed by the
linear coefficient), so every parameter is identifiable in principle.

Search is a beam search from small shapes to larger ones. Every candidate is
scored by BIC plus a *structure cost*: minus the log prior probability of the
program under a probabilistic grammar, plus a look-elsewhere charge for each
frequency searched. The more programs the search can reach, the more evidence a
constructed hypothesis needs to win, which is what keeps construction from
explaining noise.

Every program is compiled from generated Python source that must pass the same
static policy (`check_source`) applied to capability artifacts: only arithmetic,
calls to five whitelisted helpers, and the names `x` and `p`.
"""
from __future__ import annotations

import ast
import itertools
import math
import random
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from .hypotheses import Fit, FitError, _loglik, _median, _scale_floor, nelder_mead, solve_wls

UNARY = ("E", "L", "S", "I", "Q")
N_PARAMS = {"E": 1, "L": 1, "S": 2, "I": 1, "Q": 1}

# Grid of initial values per unary parameter slot.
INIT_GRID = {
    ("E", 0): (-1.0, -0.4, -0.15, 0.15, 0.4, 1.0),
    ("L", 0): (0.0, 1.0, 5.0),
    ("S", 0): (0.4, 0.8, 1.3, 1.9, 2.6, 3.5, 4.8),
    ("S", 1): (0.0, 1.6),
    ("I", 0): (0.5, 3.0, -11.0),
    ("Q", 0): (-8.0, -5.0, -2.0, 0.0),
}

# Probabilistic grammar used as the structure prior (designer-provided, fixed).
P_LEAF, P_UNARY, P_PROD, P_LIB = 0.40, 0.35, 0.15, 0.10
P_TERMS = {1: 0.6, 2: 0.4}
# Effective number of independent frequencies searched for one sin node on the
# benchmark domain: (w_max - w_min) * range / pi  with w in [0.3, 6], range 9.9.
FREQ_TRIALS = (6.0 - 0.3) * 9.9 / math.pi

HELPERS = {"_E", "_L", "_S", "_I", "_Q"}


def _E(z: float) -> float:
    return math.exp(z) if z < 700.0 else 1e300


def _L(z: float) -> float:
    a = abs(z)
    return math.log(a) if a > 1e-300 else -690.0


def _I(z: float) -> float:
    if abs(z) > 1e-12:
        return 1.0 / z
    return 1e12 if z >= 0 else -1e12


def _Q(z: float) -> float:
    return z * z


NAMESPACE = {"_E": _E, "_L": _L, "_S": math.sin, "_I": _I, "_Q": _Q, "__builtins__": {}}


# ---------------------------------------------------------------------------
# Static policy for generated / proposed code
# ---------------------------------------------------------------------------

class PolicyError(ValueError):
    pass


_ALLOWED_EXPR_NODES = (ast.Expression, ast.Lambda, ast.arguments, ast.arg, ast.BinOp, ast.UnaryOp,
                       ast.Call, ast.Name, ast.Load, ast.Constant, ast.Subscript, ast.Add, ast.Sub,
                       ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)


def check_source(src: str) -> None:
    """Accept only `lambda x, p: <arithmetic over x, p[i], numbers and helper calls>`."""
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise PolicyError(f"syntax error: {e.msg}") from None
    if not isinstance(tree.body, ast.Lambda) or [a.arg for a in tree.body.args.args] != ["x", "p"]:
        raise PolicyError("program must be `lambda x, p: ...`")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_EXPR_NODES):
            raise PolicyError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in HELPERS | {"x", "p"}:
            raise PolicyError(f"disallowed name: {node.id}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in HELPERS):
            raise PolicyError("only helper calls are allowed")
        if isinstance(node, ast.Call) and node.keywords:
            raise PolicyError("keyword arguments are not allowed")
        if isinstance(node, ast.Subscript):
            if not (isinstance(node.value, ast.Name) and node.value.id == "p"
                    and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int)):
                raise PolicyError("only p[<int>] subscripts are allowed")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise PolicyError("only numeric constants are allowed")


def compile_source(src: str) -> Callable[[float, Sequence[float]], float]:
    check_source(src)
    return eval(compile(src, "<quanta-program>", "eval"), dict(NAMESPACE))  # policy-checked above


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    op: str                                   # "x", one of UNARY, "P", or "LIB"
    children: tuple["Shape", ...] = ()
    lib: "Abstraction | None" = None

    @property
    def size(self) -> int:
        if self.op == "x":
            return 0
        if self.op == "LIB":
            return 1
        return 1 + sum(c.size for c in self.children)

    @property
    def n_params(self) -> int:
        if self.op == "x":
            return 0
        if self.op == "LIB":
            return self.lib.n_params
        return N_PARAMS.get(self.op, 0) + sum(c.n_params for c in self.children)

    @property
    def n_freq(self) -> int:
        if self.op == "LIB":
            return self.lib.n_freq
        return (1 if self.op == "S" else 0) + sum(c.n_freq for c in self.children)

    @property
    def key(self) -> str:
        if self.op == "x":
            return "x"
        if self.op == "LIB":
            return f"<{self.lib.name}>"
        return f"{self.op}({','.join(c.key for c in self.children)})"

    def expand_key(self) -> str:
        """Key with abstractions inlined (used to compare structures)."""
        if self.op == "x":
            return "x"
        if self.op == "LIB":
            return self.lib.structure or f"<{self.lib.name}>"
        return f"{self.op}({','.join(c.expand_key() for c in self.children)})"

    def source(self, offset: int = 0) -> tuple[str, int]:
        """Python expression for this shape using p[offset...]; returns (src, next_offset)."""
        if self.op == "x":
            return "x", offset
        if self.op == "LIB":
            return f"({relocate(self.lib.source, offset)})", offset + self.lib.n_params
        if self.op == "P":
            a, o = self.children[0].source(offset)
            b, o = self.children[1].source(o)
            return f"({a})*({b})", o
        inner, o = self.children[0].source(offset)
        if self.op == "E":
            return f"_E(p[{o}]*({inner}))", o + 1
        if self.op == "L":
            return f"_L(({inner})+p[{o}])", o + 1
        if self.op == "S":
            return f"_S(p[{o}]*({inner})+p[{o + 1}])", o + 2
        if self.op == "I":
            return f"_I(({inner})+p[{o}])", o + 1
        if self.op == "Q":
            return f"_Q(({inner})+p[{o}])", o + 1
        raise ValueError(self.op)

    def param_slots(self) -> list[tuple[str, int]]:
        """(op, slot) for each parameter in source order (for initialization grids)."""
        if self.op == "x":
            return []
        if self.op == "LIB":
            return [("LIB", i) for i in range(self.lib.n_params)]
        if self.op == "P":
            return self.children[0].param_slots() + self.children[1].param_slots()
        return self.children[0].param_slots() + [(self.op, i) for i in range(N_PARAMS[self.op])]

    def log_prior(self, n_lib: int) -> float:
        """log P(shape) under the structure grammar."""
        p_lib = P_LIB if n_lib else 0.0
        z = P_LEAF + P_UNARY + P_PROD + p_lib
        if self.op == "x":
            return math.log(P_LEAF / z)
        if self.op == "LIB":
            return math.log(p_lib / z / n_lib)
        if self.op == "P":
            return math.log(P_PROD / z) + sum(c.log_prior(n_lib) for c in self.children)
        return math.log(P_UNARY / z / len(UNARY)) + self.children[0].log_prior(n_lib)

    def subshapes(self) -> list["Shape"]:
        out = [self]
        for c in self.children:
            out.extend(c.subshapes())
        return out


X = Shape("x")


def unary(op: str, child: Shape) -> Shape:
    return Shape(op, (child,))


def prod(a: Shape, b: Shape) -> Shape:
    a, b = sorted((a, b), key=lambda s: s.key)
    return Shape("P", (a, b))


_PARAM_RE = re.compile(r"p\[(\d+)\]")


def relocate(src: str, offset: int) -> str:
    """Shift every parameter index p[i] in `src` by `offset`."""
    if offset == 0:
        return src
    return _PARAM_RE.sub(lambda m: f"p[{int(m.group(1)) + offset}]", src)


@dataclass(frozen=True)
class Abstraction:
    """A learned, named shape: a reusable building block acquired from experience.

    It is stored as generated, policy-checked source code over `x` and its own
    parameters `p[0..n_params-1]` (relocated when embedded in a larger program),
    so the artifact that is tested, reviewed and hashed is exactly the code that
    runs."""
    name: str
    source: str
    n_params: int
    n_freq: int = 0
    structure: str = ""
    inits: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        check_source(f"lambda x, p: {self.source}")
        used = {int(m) for m in _PARAM_RE.findall(self.source)}
        if used and (min(used) < 0 or max(used) >= self.n_params):
            raise PolicyError("parameter indices out of range")

    @classmethod
    def from_shape(cls, name: str, shape: "Shape", inits=()) -> "Abstraction":
        src, n = shape.source(0)
        return cls(name, src, n, shape.n_freq, shape.expand_key(), tuple(tuple(i) for i in inits))


def lib_leaf(ab: Abstraction) -> Shape:
    return Shape("LIB", lib=ab)


# ---------------------------------------------------------------------------
# Models and fitting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Model:
    shapes: tuple[Shape, ...]

    @property
    def key(self) -> str:
        return "c+" + "+".join("c*" + s.key for s in sorted(self.shapes, key=lambda s: s.key))

    @property
    def size(self) -> int:
        return sum(s.size for s in self.shapes)

    @property
    def n_nonlinear(self) -> int:
        return sum(s.n_params for s in self.shapes)

    @property
    def n_params(self) -> int:
        return 1 + len(self.shapes) + self.n_nonlinear

    def sources(self) -> list[str]:
        out, o = [], 0
        for s in self.shapes:
            src, o2 = s.source(o)
            out.append(src)
            o = o2
        return out

    def log_prior(self, n_lib: int) -> float:
        """Structure log prior including a look-elsewhere charge per frequency searched."""
        lp = math.log(P_TERMS[len(self.shapes)]) + sum(s.log_prior(n_lib) for s in self.shapes)
        return lp - sum(s.n_freq for s in self.shapes) * math.log(FREQ_TRIALS)

    def param_slots(self) -> list[tuple[str, int]]:
        return [slot for s in self.shapes for slot in s.param_slots()]

    def describe(self, params: Sequence[float], coefs: Sequence[float]) -> str:
        srcs = self.sources()
        terms = [f"{coefs[0]:.4g}"] + [f"{c:.4g}*[{s}]" for c, s in zip(coefs[1:], srcs)]
        body = " + ".join(terms)
        for i, v in enumerate(params):
            body = body.replace(f"p[{i}]", f"{v:.4g}")
        return "y = " + body.replace("_E", "exp").replace("_L", "log|.|").replace("_S", "sin") \
            .replace("_I", "inv").replace("_Q", "sq")


_COMPILED: dict[str, Callable] = {}


def compiled(src: str) -> Callable:
    fn = _COMPILED.get(src)
    if fn is None:
        fn = compile_source(f"lambda x, p: {src}")
        _COMPILED[src] = fn
    return fn


@dataclass
class FitResult:
    model: Model
    params: list[float]
    coefs: list[float]
    rss: float
    loglik: float
    n: int
    weights: list[float] | None = None

    @property
    def bic(self) -> float:
        if not math.isfinite(self.loglik):
            return math.inf
        return -2.0 * self.loglik + (self.model.n_params + 1) * math.log(max(self.n, 2))

    def predictor(self) -> Callable[[float], float]:
        fns = [compiled(s) for s in self.model.sources()]
        params, coefs = list(self.params), list(self.coefs)

        def predict(x: float) -> float:
            try:
                return coefs[0] + sum(c * f(x, params) for c, f in zip(coefs[1:], fns))
            except (OverflowError, ValueError, ZeroDivisionError):
                return float("nan")

        return predict


def _columns(fns, xs, params):
    cols = []
    for f in fns:
        col = []
        for x in xs:
            v = f(x, params)
            if not math.isfinite(v) or abs(v) > 1e150:
                return None
            col.append(v)
        cols.append(col)
    return cols


def _solve(cols, ys, weights):
    rows = [(1.0, *[c[i] for c in cols]) for i in range(len(ys))]
    coefs = solve_wls(rows, ys, weights, ridge=1e-9)
    res = [y - sum(c * r for c, r in zip(coefs, row)) for row, y in zip(rows, ys)]
    return coefs, res


def fit_model(model: Model, xs: Sequence[float], ys: Sequence[float], *, init: Sequence[float] | None = None,
              robust: bool = False, nu: float = 4.0, rng: random.Random | None = None,
              max_starts: int = 3, max_iter: int = 250, grid_cap: int = 400) -> FitResult:
    """Variable projection: Nelder-Mead over the nonlinear parameters, exact weighted
    least squares for the linear coefficients."""
    rng = rng or random.Random(0)
    fns = [compiled(s) for s in model.sources()]
    k = model.n_nonlinear
    weights: list[float] | None = None
    floor = _scale_floor(ys)

    def objective(p):
        try:
            cols = _columns(fns, xs, p)
            if cols is None:
                return math.inf
            _, res = _solve(cols, ys, weights)
        except (FitError, OverflowError, ValueError, ZeroDivisionError):
            return math.inf
        if weights is None:
            return sum(r * r for r in res)
        return sum(w * r * r for w, r in zip(weights, res))

    best_p: list[float] = []
    for it in range(4 if robust else 1):
        if k == 0:
            best_p = []
        else:
            starts = _starts(model, init if it == 0 else best_p, objective, rng, max_starts, grid_cap)
            best_v = math.inf
            for s in starts:
                p, v = nelder_mead(objective, s, step=0.25, max_iter=max_iter, tol=1e-9)
                if v < best_v:
                    best_p, best_v = list(p), v
            if not math.isfinite(best_v):
                raise FitError(f"could not fit {model.key}")
        if robust and it < 3:
            cols = _columns(fns, xs, best_p)
            if cols is None:
                raise FitError(f"could not fit {model.key}")
            _, res = _solve(cols, ys, weights)
            s = max(1.4826 * _median([abs(r) for r in res]), floor)
            weights = [(nu + 1) / (nu + (r / s) ** 2) for r in res]
    cols = _columns(fns, xs, best_p)
    if cols is None:
        raise FitError(f"could not fit {model.key}")
    coefs, res = _solve(cols, ys, weights)
    ll = _loglik(res, robust=robust, nu=nu, weights=weights, floor=floor, k=model.n_params)
    return FitResult(model, best_p, list(coefs), sum(r * r for r in res), ll, len(xs), weights)


def _starts(model: Model, init, objective, rng, max_starts, grid_cap=400):
    """Starting points for Nelder-Mead: a supplied warm start (plus jittered copies),
    otherwise the best points of an initialization grid. Learned abstractions supply
    their own typical parameter values instead of a grid."""
    slots = model.param_slots()
    if init is not None and len(init) == len(slots):
        base = list(init)
        return [base] + [[v + rng.gauss(0, 0.15 * (abs(v) + 0.3)) for v in base] for _ in range(max_starts - 1)]
    grids = [(None,) if op == "LIB" else INIT_GRID[(op, slot)] for op, slot in slots]
    combos = list(itertools.product(*grids))
    if len(combos) > grid_cap:
        rng.shuffle(combos)
        combos = combos[:grid_cap]
    candidates = []
    for combo in combos:
        for lib_fill in _lib_option_lists(model):
            p = list(combo)
            for idx, val in lib_fill.items():
                p[idx] = val
            p = [0.5 if v is None else v for v in p]
            candidates.append((objective(p), p))
    candidates.sort(key=lambda t: t[0])
    return [p for _, p in candidates[:max_starts]] or [[0.5] * len(slots)]


def _lib_option_lists(model: Model) -> list[dict[int, float]]:
    """Assignments of learned initial values to the parameter slots of LIB leaves."""
    per_leaf: list[list[dict[int, float]]] = []
    offset = 0

    def walk(s: Shape):
        nonlocal offset
        if s.op == "x":
            return
        if s.op == "LIB":
            opts = []
            inits = s.lib.inits or (tuple(0.5 for _ in range(s.lib.n_params)),)
            for init in inits[:3]:
                opts.append({offset + i: v for i, v in enumerate(init)})
            per_leaf.append(opts)
            offset += s.lib.n_params
            return
        if s.op == "P":
            walk(s.children[0])
            walk(s.children[1])
            return
        walk(s.children[0])
        offset += N_PARAMS[s.op]

    for s in model.shapes:
        walk(s)
    if not per_leaf:
        return [{}]
    out = []
    for combo in itertools.product(*per_leaf):
        merged: dict[int, float] = {}
        for d in combo:
            merged.update(d)
        out.append(merged)
    return out[:9]


def score(fit: FitResult, n_lib: int) -> float:
    """MDL-style score (lower is better): BIC plus twice the structure cost."""
    return fit.bic - 2.0 * fit.model.log_prior(n_lib)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def base_shapes(library: Sequence[Abstraction] = ()) -> list[Shape]:
    return [X] + [unary(op, X) for op in UNARY] + [prod(X, X)] + [lib_leaf(a) for a in library]


def _children(model: Model, library: Sequence[Abstraction], max_size: int) -> list[Model]:
    out = []
    bases = base_shapes(library)
    for i, s in enumerate(model.shapes):
        rest = model.shapes[:i] + model.shapes[i + 1:]
        for op in UNARY:
            ns = unary(op, s)
            if model.size + 1 <= max_size:
                out.append(Model(rest + (ns,)))
        for b in bases:
            ns = prod(s, b)
            if model.size + 1 + b.size <= max_size:
                out.append(Model(rest + (ns,)))
    if len(model.shapes) == 1:
        for b in bases:
            # a repeated term is only identifiable if it has its own nonlinear parameters
            if model.size + b.size <= max_size and (b.key != model.shapes[0].key or b.n_params > 0):
                out.append(Model(model.shapes + (b,)))
    return out


@dataclass
class Candidate:
    model: Model
    fit: FitResult
    score: float


def synthesize(xs: Sequence[float], ys: Sequence[float], *, library: Sequence[Abstraction] = (),
               max_size: int = 3, beam: int = 6, robust: bool = False, nu: float = 4.0,
               seed: int = 0, top_k: int = 5) -> list[Candidate]:
    """Beam search over models, best (lowest score) first. Every model returned has
    at least two residual degrees of freedom."""
    rng = random.Random(seed)
    n_lib = len(library)
    seen: dict[str, Candidate] = {}

    def evaluate(model: Model, init=None) -> Candidate | None:
        # cheap fit during search; the finalists are refitted carefully below
        if model.key in seen:
            return seen[model.key]
        if len(xs) - model.n_params < 2:
            return None
        try:
            f = fit_model(model, xs, ys, init=init, robust=robust, nu=nu, rng=rng,
                          max_starts=1, max_iter=120, grid_cap=40)
        except FitError:
            return None
        if not math.isfinite(f.bic):
            return None
        c = Candidate(model, f, score(f, n_lib))
        seen[model.key] = c
        return c

    frontier = [c for c in (evaluate(Model((s,))) for s in base_shapes(library)) if c]
    frontier.sort(key=lambda c: c.score)
    frontier = frontier[:beam]
    for _ in range(max_size):
        nxt = []
        for parent in frontier:
            for child in _children(parent.model, library, max_size):
                if child.key in seen:
                    continue
                c = evaluate(child)
                if c:
                    nxt.append(c)
        if not nxt:
            break
        nxt.sort(key=lambda c: c.score)
        frontier = nxt[:beam]
    ranked = sorted(seen.values(), key=lambda c: c.score)[: 2 * top_k]
    final = []
    for c in ranked:
        try:
            f = fit_model(c.model, xs, ys, robust=robust, nu=nu, rng=rng, init=c.fit.params)
            f2 = fit_model(c.model, xs, ys, robust=robust, nu=nu, rng=rng)
            f = f if f.rss <= f2.rss else f2
        except FitError:
            continue
        final.append(Candidate(c.model, f, score(f, n_lib)))
    final.sort(key=lambda c: c.score)
    return final[:top_k]


# ---------------------------------------------------------------------------
# A hypothesis wrapper the discovery loop can use
# ---------------------------------------------------------------------------

class SynthHypothesis:
    """A constructed (or learned) model competing alongside the named laws.

    `log_prior_offset` carries the structure cost so that a program found by
    searching a large space must earn proportionally more evidence."""

    def __init__(self, model: Model, *, name: str | None = None, init: Sequence[float] | None = None,
                 log_prior_offset: float = 0.0, origin: str = "constructed") -> None:
        self.model = model
        self.name = name or f"synth:{model.key}"
        self.n_params = model.n_params
        self._init = list(init) if init is not None else None
        self.log_prior_offset = log_prior_offset
        self.origin = origin

    def fit(self, xs, ys, *, robust: bool = False, nu: float = 4.0, warm: bool = False) -> Fit:
        starts = 3 if self._init is None else (1 if warm else 2)
        f = fit_model(self.model, xs, ys, init=self._init, robust=robust, nu=nu, max_starts=starts)
        self._init = list(f.params)
        predict = f.predictor()
        params = {f"p{i}": v for i, v in enumerate(f.params)}
        params.update({f"c{i}": v for i, v in enumerate(f.coefs)})
        return Fit(self.name, self.model.n_params, predict, params, f.loglik, f.rss, f.n,
                   self.model.describe(f.params, f.coefs))
