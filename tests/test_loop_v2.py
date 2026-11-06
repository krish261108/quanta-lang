"""Fixes to the acquisition loop found by the independent correctness review."""
from __future__ import annotations

import math
import random

from quanta.capabilities.acquire import align_params, artifact_expanded_key, expanded_key, random_model
from quanta.capabilities.artifacts import build_artifact
from quanta.capabilities.diagnosis import diagnose
from quanta.science.synth import X, Model, unary
from quanta.science.toolkit import load_artifact


def test_terms_in_a_different_order_get_their_own_starting_values():
    a = Model((unary("E", X), unary("S", X)))
    b = Model((unary("S", X), unary("E", X)))
    assert expanded_key(a) == expanded_key(b)
    # b's parameters: S(a, b) then E(a); reorder into a's order E(a), S(a, b)
    assert align_params(b, [2.0, 0.3, -0.25], a) == [-0.25, 2.0, 0.3]


def test_a_law_written_through_a_capability_is_recognized_as_that_capability():
    cap_model = Model((unary("S", X), X))
    art = build_artifact("cap-0001", 1, cap_model, [[2.0, 0.0]], name="t", description="d")
    _, blocks = load_artifact(art)
    from quanta.science.synth import lib_leaf
    through_block = Model((lib_leaf(blocks[0]), X))
    assert expanded_key(through_block) == artifact_expanded_key(art)


def test_random_control_respects_the_registered_size_range():
    from quanta.capabilities.acquire import AcquisitionSettings, RandomAcquisitionLoop
    rng = random.Random(0)
    assert max(random_model(rng.randint(1, 3), rng).size for _ in range(5000)) >= 4   # the generator can
    loop = RandomAcquisitionLoop.__new__(RandomAcquisitionLoop)
    loop.s = AcquisitionSettings(max_candidates_per_round=50, synth_max_size=3)
    loop.rng, loop.attempted, loop._failed, loop._robust = random.Random(1), set(), [], True
    cands = loop._design([], {"capabilities": []})
    assert cands and all(1 <= c["model"].size <= 3 for c in cands)


def _obs(f, n=24, sd=0.02):
    rng = random.Random(3)
    xs = [0.1 + 9.9 * i / (n - 1) for i in range(n)]
    return [[x, f(x) + rng.gauss(0, sd)] for x in xs]


def test_confident_wrong_answers_are_diagnosed_from_the_loops_view():
    out = {"answer": "linear", "credence": 0.97, "answer_kind": "named",
           "observations": _obs(lambda x: 1 + 0.5 * x + 1.2 * math.sin(2.1 * x))}
    d = diagnose("t", out, [])
    assert d is not None and d.mode in ("added_oscillation", "modulated_oscillation")


def test_solver_errors_count_as_failures():
    d = diagnose("t", {"error": "TimeoutError: worker killed", "answer": "error"}, [])
    assert d is not None and d.mode == "solver_error"
