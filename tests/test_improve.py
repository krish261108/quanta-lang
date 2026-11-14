import json

import pytest

from quanta.bench.harness import SuiteResult, diagnose, run_suite
from quanta.genome import Genome
from quanta.governance import Governor
from quanta.improve import (REPO_ROOT, Candidate, DiagnosticProposer, EvaluationError, Evaluator, GateConfig,
                            LLMProposer, SelfImprover, gate, patch_path_allowed)
from quanta.llm import ScriptedModel


@pytest.mark.parametrize("path,ok", [
    ("src/quanta/science/discovery.py", True), ("src/quanta/science/new_strategy.py", True),
    ("src/quanta/governance.py", False), ("src/quanta/improve.py", False), ("src/quanta/bench/tasks.py", False),
    ("src/quanta/tools.py", False), ("src/quanta/epistemics.py", False), ("src/quanta/genome.py", False),
    ("tests/test_governance_invariants.py", False), ("src/quanta/science/../governance.py", False),
    ("../outside.py", False), ("/etc/passwd", False), ("src/quanta/solver.py", False),
])
def test_only_mutable_paths_can_be_patched(path, ok):
    assert patch_path_allowed(path) is ok


def _fake_suite(scores, cats=None):
    results = [{"seed": i, "score": s, "correct": True, "credence": 0.9, "category": (cats or ["clean"] * len(scores))[i],
                "stop_reason": "confident", "cost": 0.5, "nrmse": 0.1, "asked_human": False,
                "expected": "linear", "answer": "linear"} for i, s in enumerate(scores)]
    return SuiteResult(Genome(), "selection", results, diagnose(results))


def test_gate_rejects_null_difference_and_accepts_real_gain():
    base = run_suite(Genome(), "selection", 30)
    same = run_suite(Genome(), "selection", 30)
    assert not gate(base, same).accept                        # A/A test: no difference -> reject
    better = _fake_suite([0.8] * 40)
    worse = _fake_suite([0.7] * 39 + [0.71])
    assert gate(worse, better).accept
    assert not gate(better, worse).accept


def test_gate_blocks_category_regression():
    cats = ["clean"] * 30 + ["outliers"] * 10
    base = _fake_suite([0.5] * 40, cats)
    cand_results = [dict(r, score=r["score"] + 0.2) for r in base.results]
    for r in cand_results[30:]:
        r["correct"] = False
    cand = SuiteResult(Genome(), "selection", cand_results, diagnose(cand_results))
    dec = gate(base, cand, GateConfig(min_category_size=5))
    assert not dec.accept and any("outliers" in r for r in dec.reasons)


def test_diagnostic_proposer_targets_failure_modes():
    import random
    d = run_suite(Genome(), "selection", 60).diagnostics
    props = DiagnosticProposer().propose(Genome(), d, [], random.Random(0), 8)
    changed = [set(c) for c, _, _ in props]
    assert {"falsification_rounds"} in changed                 # baseline is confidently wrong sometimes
    assert any({"flexible_baseline", "expand_on_inadequacy"} <= c for c in changed)
    assert all(why for _, why, _ in props)
    # proposals already evaluated are not repeated
    hist = [{"genome": Genome().mutate(**c).fingerprint()} for c, _, _ in props]
    again = DiagnosticProposer().propose(Genome(), d, hist, random.Random(0), 8)
    assert not {json.dumps(c, sort_keys=True) for c, _, _ in again} & {json.dumps(c, sort_keys=True) for c, _, _ in props}


def test_self_improver_end_to_end(tmp_path):
    imp = SelfImprover(Evaluator(jobs=2), [DiagnosticProposer()], run_dir=tmp_path, generations=2,
                       candidates_per_generation=4, n_selection=40, n_confirmation=40, n_test=60, patience=2)
    res = imp.run(Genome())
    assert (tmp_path / "report.md").exists() and (tmp_path / "summary.json").exists()
    events = [json.loads(l) for l in (tmp_path / "lineage.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "baseline"
    for a in res.adopted:                                      # every adoption passed both gates
        cid = a["candidate"]
        sel = [e for e in events if e.get("candidate") == cid and e["event"] == "evaluated"][0]
        conf = [e for e in events if e.get("candidate") == cid and e["event"] == "confirmation"][-1]
        assert sel["selection_gate"]["accept"] and conf["confirmation_gate"]["accept"]
    report = (tmp_path / "report.md").read_text()
    assert "## Established facts" in report and "## Speculation" in report and "held-out" in report
    assert len(res.test_initial.results) == 60


def test_sandbox_refuses_protected_patch():
    ev = Evaluator(jobs=1)
    cand = Candidate("bad", Genome(), patch={"src/quanta/governance.py": "MAX_AUTO_APPROVE = 3\n"})
    with pytest.raises(EvaluationError, match="non-mutable"):
        ev.evaluate(cand, "selection", 4)


def test_sandbox_evaluates_code_patch_and_catches_breakage():
    ev = Evaluator(jobs=1)
    src = (REPO_ROOT / "src/quanta/science/discovery.py").read_text()
    benign = Candidate("ok", Genome(), patch={"src/quanta/science/discovery.py": src + "\n# harmless\n"})
    sandboxed = ev.evaluate(benign, "selection", 4)
    in_process = run_suite(Genome(), "selection", 4)
    assert sandboxed.scores == pytest.approx(in_process.scores)
    broken = Candidate("broken", Genome(), patch={"src/quanta/science/discovery.py": src + "\nraise ImportError\n"})
    with pytest.raises(EvaluationError, match="benchmark failed"):
        ev.evaluate(broken, "selection", 4)


def test_code_adoption_requires_human_authorization(tmp_path):
    imp = SelfImprover(Evaluator(), [], run_dir=tmp_path, governor=Governor())
    assert imp._authorize(Candidate("g", Genome()))           # bounded genome change: LOW risk
    assert not imp._authorize(Candidate("p", Genome(), patch={"src/quanta/science/x.py": ""}))  # HIGH risk


def test_llm_proposer_validates_proposals():
    import random
    model = ScriptedModel([
        {"tool_calls": [("submit_proposals", {"proposals": [{"changes": {"approval_ceiling": 3}, "rationale": "x"}]})]},
        {"tool_calls": [("submit_proposals", {"proposals": [
            {"changes": {"falsification_rounds": 3, "temperature": 99}, "rationale": "test leaders harder"}]})]},
    ])
    props = LLMProposer(model).propose(Genome(), {"accuracy": 0.8}, [], random.Random(0), 2)
    assert props == [({"falsification_rounds": 3, "temperature": 4.0}, "test leaders harder", None)]


def test_inert_parameters_are_not_mutated():
    import random
    from quanta.improve import _random_mutation, active_params
    g = Genome()
    assert not {"t_dof", "prior_pseudocount", "expand_on_inadequacy", "explore_prob", "n_candidates"} & set(active_params(g))
    assert "t_dof" in active_params(g.mutate(likelihood="student_t"))
    assert "n_candidates" in active_params(g.mutate(falsification_rounds=2))
    rng = random.Random(0)
    for _ in range(200):
        changes, _ = _random_mutation(g, rng)
        assert set(changes) <= set(active_params(g))


def test_near_misses_are_combined():
    import random
    d = run_suite(Genome(), "selection", 40).diagnostics
    hist = [{"changes": {"likelihood": "student_t"}, "genome": "a", "mean_delta": 0.01, "passed_gate": False},
            {"changes": {"design": "disagreement"}, "genome": "b", "mean_delta": 0.02, "passed_gate": False},
            {"changes": {"temperature": 1.3}, "genome": "c", "mean_delta": -0.01, "passed_gate": False}]
    props = DiagnosticProposer().propose(Genome(), d, hist, random.Random(0), 10)
    assert any(c == {"design": "disagreement", "likelihood": "student_t"} and "near-miss" in why
               for c, why, _ in props)
