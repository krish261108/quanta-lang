"""Tests for the capability-acquisition subsystem and its protected evaluator."""
from __future__ import annotations

import json
import math
import random

import pytest

from quanta.bench import acquisition as A
from quanta.bench import external as EXT
from quanta.bench import review as R
from quanta.bench.isolation import Job, run_isolated
from quanta.capabilities.artifacts import build_artifact, run_unit_tests, static_check
from quanta.capabilities.diagnosis import GapDiagnosis, diagnose
from quanta.capabilities.registry import CapabilityRecord, Registry, RegistryError
from quanta.genome import Genome
from quanta.science.discovery import DiscoveryLoop
from quanta.science.synth import X, Model, unary
from quanta.science.toolkit import ArtifactError, Toolkit, artifact_hash, load_artifact

S0 = Genome(falsification_rounds=2, likelihood="student_t", flexible_baseline=True, pool_equivalent=True)
GRADE_FIELDS = {"score", "recovered", "family", "kind", "interp_nrmse", "extrap_nrmse", "brier", "p_claim",
                "claimed", "cost", "outliers"}


def _artifact(cid="cap-0001", shape=None):
    model = Model((shape or unary("I", unary("E", X)),))
    return build_artifact(cid, 1, model, [[-1.5, 5.5e-4]], name="test", description="test artifact")


def _record(reg, art):
    rec = CapabilityRecord(capability_id=art["capability_id"], version=1, name="t", description="d",
                           origin="test", triggering_failure={}, hypothesis="h", artifact=art)
    return reg.propose(rec)


# -- registry -----------------------------------------------------------------

def test_registry_lifecycle_and_replay(tmp_path):
    reg = Registry(tmp_path / "reg.jsonl")
    art = _artifact(reg.new_id())
    _record(reg, art)
    assert reg.get(art["capability_id"]).status == "EXPERIMENTAL"
    with pytest.raises(RegistryError):
        _record(reg, art)                                    # no overwrite of (id, version)
    with pytest.raises(RegistryError):
        reg.set_status(art["capability_id"], 1, "ACTIVE", "no review")
    with pytest.raises(RegistryError):
        reg.set_status(art["capability_id"], 1, "ACTIVE", "rejected review",
                       review={"review_id": "r1", "approved": False})
    reg.add_evidence(art["capability_id"], 1, "tests", {"passed": True})
    reg.set_status(art["capability_id"], 1, "ACTIVE", "approved", review={"review_id": "r1", "approved": True},
                   confidence=0.8)
    assert reg.active_artifacts() == [art]
    replay = Registry(tmp_path / "reg.jsonl")                  # the log alone reproduces the view
    r = replay.get(art["capability_id"])
    assert r.status == "ACTIVE" and r.tests == {"passed": True} and r.confidence == 0.8
    assert replay.new_id() == "cap-0002"


def test_rejected_is_final(tmp_path):
    reg = Registry(tmp_path / "reg.jsonl")
    art = _artifact()
    _record(reg, art)
    reg.set_status("cap-0001", 1, "REJECTED", "failed gates")
    with pytest.raises(RegistryError):
        reg.set_status("cap-0001", 1, "ACTIVE", "x", review={"review_id": "r", "approved": True})
    assert reg.summary()["REJECTED"] == 1


# -- code policy and artifacts --------------------------------------------------

@pytest.mark.parametrize("src", [
    "__import__('os').system('true')",
    "x.__class__",
    "(lambda: 1)()",
    "open('/etc/passwd')",
    "p[0].real",
    "[c for c in ()]",
    "globals()",
])
def test_policy_rejects_malicious_sources(src):
    art = _artifact()
    art["terms"][0]["source"] = src
    art["sha256"] = artifact_hash(art)
    assert static_check(art), src
    with pytest.raises((ArtifactError, ValueError)):
        load_artifact(art)


def test_valid_artifact_loads_and_hash_mismatch_is_rejected():
    art = _artifact()
    assert static_check(art) == []
    family, blocks = load_artifact(art)
    assert family.name == "cap:cap-0001" and len(blocks) == 1
    bad = json.loads(json.dumps(art))
    bad["terms"][0]["source"] = bad["terms"][0]["source"].replace("p[1]", "p[0]")
    with pytest.raises(ArtifactError):
        load_artifact(bad)                                    # content no longer matches its hash


def test_artifact_unit_tests_in_subprocess():
    out = run_unit_tests(_artifact(), trials=6)
    assert out["passed"], out
    assert out["policy"] and out["finite"] and out["deterministic"]


# -- isolation ------------------------------------------------------------------

class _Line:
    def __init__(self):
        self.domain = (0.1, 10.0)
        self.calls = 0

    def measure(self, x):
        self.calls += 1
        return 2.0 + 0.5 * x


def test_isolation_enforces_budget_and_is_deterministic():
    config = Toolkit(S0, []).to_config()
    eval_x = (1.0, 5.0, 12.0)
    jobs = [Job(f"j{i}", _Line(), 9, 100 + i, eval_x) for i in range(4)]
    one = run_isolated(config, jobs, workers=1)
    jobs2 = [Job(f"j{i}", _Line(), 9, 100 + i, eval_x) for i in range(4)]
    two = run_isolated(config, jobs2, workers=2)
    assert [r["task_id"] for r in one] == [f"j{i}" for i in range(4)]
    for a, b, j in zip(one, two, jobs):
        assert a["error"] is None
        assert a["measurements_used"] <= 9 and j.instrument.calls == a["measurements_used"]
        assert a["pred"] == b["pred"] and a["answer"] == b["answer"]
    assert abs(one[0]["pred"][2] - 8.0) < 0.5               # a clean line extrapolates


def test_isolation_records_failures_instead_of_dropping():
    bad = {"genome": S0.to_dict(), "capabilities": [{"interface": "nope", "capability_id": "x", "terms": []}]}
    out = run_isolated(bad, [Job("j0", _Line(), 9, 1, (1.0,))], workers=1)
    assert out[0]["error"] and "configuration" in out[0]["error"]
    slow = run_isolated(Toolkit(S0, []).to_config(), [Job("j1", _Line(), 9, 1, (1.0,))], workers=1, timeout=0.01)
    assert slow[0]["error"]


# -- evaluator splits and the proposer's view ----------------------------------------

def test_split_seed_ranges_are_disjoint_and_tasks_deterministic():
    spans = []
    for rep in (0, 1):
        for s, base in A.SPLITS.items():
            lo = base + A.REPLICATION_STRIDE * rep
            spans.append((lo, lo + A.SPLIT_WIDTH, s, rep))
    spans.append((EXT.SEED_BASE, EXT.SEED_BASE + EXT.TASKS_PER_DATASET * len(EXT.DATASETS), "external", 0))
    spans.sort()
    for (lo1, hi1, *_), (lo2, _hi2, *_) in zip(spans, spans[1:]):
        assert hi1 <= lo2
    assert A.make_task("gate", 17) == A.make_task("gate", 17)
    assert A.make_task("gate", 17).seed != A.make_task("gate", 17, 1).seed
    with pytest.raises(ValueError):
        A.make_task("gate", A.SPLIT_WIDTH)


def test_solve_training_hides_grades(tmp_path):
    svc = R.EvaluationService(tmp_path / "evals", workers=2)
    outs = svc.solve_training(Toolkit(S0, []).to_config(), 4)
    assert len(outs) == 4
    for o in outs:
        assert not (GRADE_FIELDS & set(o)), set(o) & GRADE_FIELDS
        assert o["observations"]
    assert all(A.SPLITS["train"] <= s < A.SPLITS["train"] + A.SPLIT_WIDTH for s in svc.seen_seeds)


def test_reviewer_integrity_and_leakage_gates(tmp_path, monkeypatch):
    monkeypatch.setitem(R.GATES, "n_gate", 4)
    monkeypatch.setitem(R.GATES, "n_transfer", 4)
    monkeypatch.setitem(R.GATES, "n_repro", 2)
    svc = R.EvaluationService(tmp_path / "evals", workers=2)
    rev = R.Reviewer(svc, tmp_path / "rev", workers=2, first_block=3)
    base = Toolkit(S0, []).to_config()
    art = _artifact()
    good = {"genome": base["genome"], "capabilities": [art]}
    changed_genome = {"genome": {**base["genome"], "stop_posterior": 0.5}, "capabilities": [art]}
    d = rev.review(base, changed_genome, art, "missing", set())
    assert not d.approved and not d.gates["integrity"]["passed"] and d.block == 3
    two = {"genome": base["genome"], "capabilities": [art, _artifact("cap-0002")]}
    assert not rev.review(base, two, art, "missing", set()).gates["integrity"]["passed"]
    gate_seed = A.make_task("gate", 0).seed
    d = rev.review(base, good, art, "missing", {gate_seed})
    assert not d.approved and not d.gates["no_leakage"]["passed"]
    d = rev.review(base, good, art, "missing", set())
    assert not d.approved and d.gates["validation"]["value"] == "validation records not found"
    assert rev.state["next_block"] == 7                       # every review consumed a fresh block


# -- diagnosis ------------------------------------------------------------------

def _obs(f, n=24, sd=0.02, seed=0):
    rng = random.Random(seed)
    xs = [0.1 + 9.9 * i / (n - 1) for i in range(n)]
    return [[x, f(x) + rng.gauss(0, sd)] for x in xs]


def test_diagnosis_flags_failures_with_required_fields():
    clean = {"answer": "linear", "credence": 0.97, "kind": "named", "observations": _obs(lambda x: 1 + 2 * x)}
    assert diagnose("t0", clean, []) is None
    bump = {"answer": "none", "credence": 0.9, "kind": "none",
            "observations": _obs(lambda x: 3 * math.exp(-((x - 5) ** 2) / 0.8))}
    d = diagnose("t1", bump, ["cap-0001"])
    assert isinstance(d, GapDiagnosis) and d.status == "hypothesis"
    for f in ("task", "current_approach", "observed_failure", "likely_failure_mode", "known_capabilities_used",
              "missing_capability_hypothesis", "evidence", "confidence", "proposed_next_experiment"):
        assert getattr(d, f) not in (None, ""), f
    assert "cap-0001" in d.known_capabilities_used


# -- external data (offline) ------------------------------------------------------------

def test_external_tasks_pool_and_grading():
    csv_text = "rownames,time,value\n" + "\n".join(f"{i},{1900 + i},{(i % 7) * 1.5 + 0.1 * i}" for i in range(60))
    pts = EXT.parse(csv_text.encode())
    assert len(pts) == 60
    tasks = EXT.make_tasks("toy", pts)
    assert len(tasks) == EXT.TASKS_PER_DATASET
    t = tasks[0]
    assert all(x <= EXT.MEASURE_MAX for x, _ in t.pool + t.interp) and all(x > EXT.MEASURE_MAX for x, _ in t.extrap)
    assert not set(t.pool) & set(t.interp)
    inst = EXT.PoolInstrument(t)
    seen = [inst.measure_at(5.0)[0] for _ in range(3)]
    assert len(set(seen)) == 3 and all(x in {p[0] for p in t.pool} for x in seen)
    perfect = [y for _, y in t.interp] + [y for _, y in t.extrap]
    g = EXT.grade(t, perfect)
    assert g["interp_nrmse"] == 0 and g["extrap_nrmse"] == 0 and g["score_ext"] == 1
    base = [{"task_id": f"{d}:{k}", "dataset": d, "score_ext": 0.5} for d in "abc" for k in range(3)]
    cand = [{**r, "score_ext": 0.6} for r in base]
    c = EXT.cluster_bootstrap(base, cand, n_boot=200)
    assert c["n_datasets"] == 3 and abs(c["mean_delta"] - 0.1) < 1e-9 and c["ci_low"] > 0


# -- legacy behaviour --------------------------------------------------------------------

class _Sine:
    domain = (0.1, 10.0)

    def __init__(self):
        self.rng = random.Random(5)

    def measure(self, x):
        return 1.0 + 2.0 * math.sin(1.3 * x + 0.4) + self.rng.gauss(0, 0.05)


def test_new_options_default_off_reproduce_the_legacy_loop():
    g = Genome(falsification_rounds=1)
    a = DiscoveryLoop(g, seed=3).run(_Sine())
    b = DiscoveryLoop(g, seed=3, separate_streams=False, predict_x=None).run(_Sine())
    assert (a.answer, a.credence, a.n_experiments) == (b.answer, b.credence, b.n_experiments)
    assert [o.x for o in a.observations] == [o.x for o in b.observations]
    assert Genome().fingerprint() == Genome(pool_equivalent=False).fingerprint()
