"""Frozen copy of the static code policy for capability artifacts.

PROTECTED. The reviewer must not rely on code the acquisition loop may change, so
it re-checks artifacts with this copy (extracted verbatim from
`quanta/science/synth.py` when it was introduced).
"""
from __future__ import annotations

import ast
import math
import re

HELPERS = {"_E", "_L", "_S", "_I", "_Q"}


class PolicyError(ValueError):
    pass


_ALLOWED_EXPR_NODES = (ast.Expression, ast.Lambda, ast.arguments, ast.arg, ast.BinOp, ast.UnaryOp,
                       ast.Call, ast.Name, ast.Load, ast.Constant, ast.Subscript, ast.Add, ast.Sub,
                       ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)


MAX_SOURCE_CHARS = 4000
MAX_DEPTH = 60
MAX_PARAMS = 64
MAX_CONSTANT = 1e6
MAX_EXPONENT = 8
_CANONICAL_PARAM = re.compile(r"p\[\d+\]")


def _depth(node, limit: int) -> int:
    """Nesting depth of an AST, computed without recursion (stops at `limit`)."""
    deepest, stack = 0, [(node, 1)]
    while stack:
        n, d = stack.pop()
        deepest = max(deepest, d)
        if deepest > limit:
            return deepest
        stack.extend((c, d + 1) for c in ast.iter_child_nodes(n))
    return deepest


def check_source(src: str) -> None:
    """Accept only `lambda x, p: <arithmetic over x, p[i], numbers and helper calls>`."""
    if len(src) > MAX_SOURCE_CHARS:
        raise PolicyError("program too long")
    try:
        tree = ast.parse(src, mode="eval")
    except (SyntaxError, MemoryError, RecursionError) as e:
        raise PolicyError(f"cannot parse: {type(e).__name__}") from None
    if _depth(tree, MAX_DEPTH) > MAX_DEPTH:
        raise PolicyError("program nested too deeply")
    if not isinstance(tree.body, ast.Lambda) or [a.arg for a in tree.body.args.args] != ["x", "p"]:
        raise PolicyError("program must be `lambda x, p: ...`")
    subscripts = 0
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
            idx = node.slice
            if not (isinstance(node.value, ast.Name) and node.value.id == "p" and isinstance(idx, ast.Constant)
                    and type(idx.value) is int and 0 <= idx.value < MAX_PARAMS):
                raise PolicyError("only p[<non-negative int>] subscripts are allowed")
            subscripts += 1
        if isinstance(node, ast.Constant):
            v = node.value
            if type(v) not in (int, float) or not math.isfinite(v) or abs(v) > MAX_CONSTANT:
                raise PolicyError("only finite numeric constants of moderate size are allowed")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if not (isinstance(node.right, ast.Constant) and abs(node.right.value) <= MAX_EXPONENT):
                raise PolicyError("exponents must be small numeric constants")
    # Every parameter reference must use the canonical spelling p[N], so that the index
    # range check and relocation (which work on the text) see all of them.
    if len(_CANONICAL_PARAM.findall(src)) != subscripts:
        raise PolicyError("parameter indices must be written as p[N]")
