import pytest

from quanta.memory import MemoryStore
from quanta.planning import PlanError, TaskGraph, TaskStatus


def test_memory_search_and_persistence(tmp_path):
    path = tmp_path / "mem.db"
    with MemoryStore(path) as mem:
        mem.add_knowledge("physics", "gravitational acceleration is 9.81 m/s^2",
                          status="fact", confidence=0.99, verified=True)
        mem.add_knowledge("cooking", "pasta water should be salted", confidence=0.6)
        pid = mem.add_procedure("debugging", "bisect the commit history to find a regression")
        mem.record_outcome(pid, True)
        mem.record_episode("run1", "step", "ran pytest; 3 failures")
    with MemoryStore(path) as mem:  # survives process restart
        hits = mem.search("gravitational acceleration")
        assert hits and "9.81" in hits[0].content
        proc = mem.search("regression bisect", kinds=("procedure",), verified_only=True)
        assert proc and proc[0].successes == 1
        assert mem.search("pasta", verified_only=True) == []
        assert mem.stats()["knowledge"] == 2
        assert len(mem.recent(run_id="run1")) == 1
        assert "verified" in mem.context_block("gravity acceleration")


def test_memory_count_by_meta_uses_only_verified():
    mem = MemoryStore()
    for fam in ["linear", "linear", "sinusoid"]:
        mem.add_knowledge("curve-discovery", f"truth was {fam}", verified=True, meta={"family": fam})
    mem.add_knowledge("curve-discovery", "guess", verified=False, meta={"family": "cubic"})
    counts = mem.count_by_meta("knowledge", "curve-discovery", "family")
    assert counts == {"linear": 2, "sinusoid": 1}


def test_memory_handles_punctuation_queries():
    mem = MemoryStore()
    mem.add_knowledge("q", "what's up? (nothing) AND/OR \"quotes\"")
    assert mem.search('what\'s "up" AND (OR)')
    assert mem.search("!!!") == []


def test_plan_from_spec_orders_and_validates():
    g = TaskGraph.from_spec([
        {"id": "c", "title": "C", "deps": ["a", "b"]},
        {"id": "a", "title": "A"},
        {"id": "b", "title": "B", "deps": ["a"]},
    ])
    assert g.topological_order() == ["a", "b", "c"]
    assert [t.id for t in g.ready()] == ["a"]
    with pytest.raises(PlanError):
        TaskGraph.from_spec([{"id": "x", "deps": ["y"]}, {"id": "y", "deps": ["x"]}])
    with pytest.raises(PlanError):
        TaskGraph.from_spec([{"id": "x", "deps": ["missing"]}])


def test_plan_retry_failure_and_blocking():
    g = TaskGraph.from_spec([{"id": "a", "max_attempts": 2}, {"id": "b", "deps": ["a"]}])
    g.start("a"); g.fail("a", "boom")
    assert g.get("a").status is TaskStatus.PENDING
    g.start("a"); g.fail("a", "boom again")
    assert g.get("a").status is TaskStatus.FAILED
    assert g.get("b").status is TaskStatus.BLOCKED
    assert g.is_complete() and not g.succeeded()
    g.unblock("a")
    assert g.get("b").status is TaskStatus.PENDING
    g.start("a"); g.complete("a", "ok")
    g.start("b"); g.complete("b", "ok")
    assert g.succeeded() and g.progress() == 1.0


def test_plan_refinement_rewires_dependents(tmp_path):
    g = TaskGraph.from_spec([{"id": "a"}, {"id": "b", "deps": ["a"]}, {"id": "c", "deps": ["b"]}])
    g.start("a"); g.complete("a")
    g.refine("b", [{"id": "b1"}, {"id": "b2", "deps": ["b1"]}])
    assert g.get("b").status is TaskStatus.REFINED
    assert set(g.get("c").deps) == {"b.b1", "b.b2"}
    assert [t.id for t in g.ready()] == ["b.b1"]
    g.save(tmp_path / "plan.json")
    g2 = TaskGraph.load(tmp_path / "plan.json")
    for tid in ["b.b1", "b.b2", "c"]:
        g2.start(tid); g2.complete(tid)
    assert g2.succeeded()


def test_plan_scales_to_thousands_of_actions():
    n = 3000
    spec = [{"id": f"t{i}", "deps": [f"t{i - 1}"] if i % 50 else []} for i in range(n)]
    g = TaskGraph.from_spec(reversed(spec))
    assert len(g.tasks) == n
    assert g.critical_path_length() == 50
    done = 0
    while not g.is_complete():
        for t in g.ready():
            g.start(t.id); g.complete(t.id); done += 1
    assert done == n and not g.is_stuck()
