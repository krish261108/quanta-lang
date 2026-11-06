"""Frozen copy of the static code policy for capability artifacts.

PROTECTED. The reviewer must not rely on code the acquisition loop may change, so
it re-checks artifacts with this copy (extracted verbatim from
`quanta/science/synth.py` when it was introduced).
"""
from __future__ import annotations

import ast

HELPERS = {"_E", "_L", "_S", "_I", "_Q"}


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
