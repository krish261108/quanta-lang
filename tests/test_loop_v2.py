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


def test_leakage_gate_sees_evaluations_the_proposer_did_not_report(tmp_path, monkeypatch):
    from quanta.bench import acquisition as A
    from quanta.bench import review as R
    from quanta.genome import Genome
    from quanta.science.toolkit import Toolkit
    monkeypatch.setitem(R.GATES, "n_gate", 3)
    monkeypatch.setitem(R.GATES, "n_transfer", 3)
    monkeypatch.setitem(R.GATES, "n_repro", 1)
    svc = R.EvaluationService(tmp_path / "e", workers=1)
    base = Toolkit(Genome(), []).to_config()
    svc.records(base, "gate", 2, block=0)                 # a direct, unreported held-out evaluation
    art = build_artifact("cap-0001", 1, Model((unary("E", X),)), [[-0.3]], name="t", description="d")
    cand = {"genome": base["genome"], "capabilities": [art]}
    d = R.Reviewer(svc, tmp_path / "r", workers=1).review(base, cand, art, "missing", set())
    assert not d.gates["no_leakage"]["passed"]
    assert A.make_task("gate", 0).seed in svc.unreviewed_seeds


def test_reviewer_rejects_validation_records_of_another_pair(tmp_path, monkeypatch):
    from quanta.bench import review as R
    from quanta.genome import Genome
    from quanta.science.toolkit import Toolkit
    svc = R.EvaluationService(tmp_path / "e", workers=1)
    base = Toolkit(Genome(), []).to_config()
    art = build_artifact("cap-0001", 1, Model((unary("E", X),)), [[-0.3]], name="t", description="d")
    other = build_artifact("cap-0002", 1, Model((unary("S", X),)), [[1.0, 0.0]], name="t", description="d")
    v = svc.validate(base, {"genome": base["genome"], "capabilities": [other]}, 3)
    cand = {"genome": base["genome"], "capabilities": [art]}
    d = R.Reviewer(svc, tmp_path / "r", workers=1).review(base, cand, art, v["validation_id"], set())
    assert not d.gates["validation"]["passed"] and "different pair" in d.gates["validation"]["value"]
