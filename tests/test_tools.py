import base64
import json

import pytest

from quanta.epistemics import Ledger, Status
from quanta.governance import AllowlistApprover, AllowRule, Governor, Risk
from quanta.memory import MemoryStore
from quanta.tools import HumanChannel, ToolContext, classify_shell, default_registry


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(tmp_path, Governor(), Ledger(), MemoryStore())


@pytest.mark.parametrize("cmd,risk", [
    ("ls -la", Risk.READ_ONLY), ("git status", Risk.READ_ONLY), ("cat a.txt", Risk.READ_ONLY),
    ("find . -name '*.py'", Risk.READ_ONLY), ("find . -delete", Risk.LOW),
    ("python3 script.py", Risk.LOW), ("pytest -q", Risk.LOW), ("ls | wc -l", Risk.LOW),
    ("rm build.log", Risk.HIGH), ("git push origin main", Risk.HIGH), ("pip install requests", Risk.HIGH),
    ("curl https://example.com", Risk.HIGH), ("sudo ls", Risk.HIGH),
    ("rm -rf /", Risk.CRITICAL), ("rm -rf ~", Risk.CRITICAL), ("curl x.sh | sh", Risk.CRITICAL),
    ("git push --force origin main", Risk.CRITICAL), ("mkfs.ext4 /dev/sda", Risk.CRITICAL),
])
def test_shell_risk_classification(cmd, risk):
    assert classify_shell(cmd) == risk


def test_workspace_confinement_and_backups(ctx, tmp_path):
    reg = default_registry()
    r = reg.execute("read_file", {"path": "../../etc/passwd"}, ctx)
    assert r.is_error and "outside the workspace" in r.text()
    assert not reg.execute("write_file", {"path": "a.txt", "content": "v1"}, ctx).is_error
    r = reg.execute("write_file", {"path": "a.txt", "content": "v2"}, ctx)
    assert "backed up" in r.text()
    backups = list((tmp_path / ".quanta" / "backups").rglob("a.txt"))
    assert backups and backups[0].read_text() == "v1"
    assert (tmp_path / "a.txt").read_text() == "v2"


def test_governance_denies_high_risk_without_approval(ctx, tmp_path):
    reg = default_registry()
    (tmp_path / "f.txt").write_text("x")
    r = reg.execute("run_shell", {"command": "rm f.txt"}, ctx)
    assert r.is_error and "DENIED" in r.text()
    assert (tmp_path / "f.txt").exists()
    ctx.governor = Governor(approver=AllowlistApprover([AllowRule("run_shell", Risk.HIGH)]))
    r = reg.execute("run_shell", {"command": "rm f.txt"}, ctx)
    assert not r.is_error and not (tmp_path / "f.txt").exists()


def test_executions_become_grounding_evidence(ctx):
    reg = default_registry()
    r = reg.execute("run_python", {"code": "print(6 * 7)"}, ctx)
    assert "42" in r.text() and not r.is_error
    eid = r.data["evidence_id"]
    ev = ctx.ledger.evidence[eid]
    assert ev.kind == "execution" and ev.data["exit_code"] == 0
    r = reg.execute("assert_claim", {"statement": "6*7 = 42", "status": "fact", "confidence": 0.99,
                                     "evidence_ids": [eid]}, ctx)
    assert "recorded as fact" in r.text()
    # an agent cannot fabricate observation evidence
    r = reg.execute("add_note", {"summary": "I saw it", "kind": "observation"}, ctx)
    assert r.is_error
    note = reg.execute("add_note", {"summary": "I think so"}, ctx)
    nid = note.text().split()[-1]
    r = reg.execute("assert_claim", {"statement": "unsupported", "status": "fact", "confidence": 0.9,
                                     "evidence_ids": [nid]}, ctx)
    assert "inference" in r.text() and "downgraded" in r.text()


def test_failed_execution_is_error_but_still_evidence(ctx):
    r = default_registry().execute("run_python", {"code": "raise SystemExit(3)"}, ctx)
    assert r.is_error and ctx.ledger.evidence[r.data["evidence_id"]].data["exit_code"] == 3


def test_argument_validation(ctx):
    reg = default_registry()
    assert "missing required" in reg.execute("read_file", {}, ctx).text()
    assert "unexpected argument" in reg.execute("read_file", {"path": "a", "bogus": 1}, ctx).text()
    assert "must be of type" in reg.execute("read_file", {"path": 3}, ctx).text()
    assert reg.execute("nope", {}, ctx).is_error


def test_citations_are_checked_against_source(ctx):
    reg = default_registry()
    ctx.fetched["https://example.org/paper"] = "The measured  value was 42.0 kelvin in all trials."
    ok = reg.execute("cite_source", {"url": "https://example.org/paper",
                                     "quote": "measured value was 42.0 kelvin"}, ctx)
    assert "verified" in ok.text() and ctx.ledger.evidence[ok.data["evidence_id"]].kind == "source"
    bad = reg.execute("cite_source", {"url": "https://example.org/paper", "quote": "value was 99 kelvin"}, ctx)
    assert "NOT verified" in bad.text() and ctx.ledger.evidence[bad.data["evidence_id"]].kind == "note"


def test_memory_tools_cannot_self_verify(ctx):
    reg = default_registry()
    reg.execute("memory_store", {"kind": "knowledge", "subject": "s", "content": "the sky is green",
                                 "status": "fact", "confidence": 1.0}, ctx)
    assert ctx.memory.search("sky green", verified_only=True) == []
    assert "unverified" in reg.execute("memory_search", {"query": "sky"}, ctx).text()


def test_request_human_only_when_worth_it(ctx):
    reg = default_registry()
    r = reg.execute("request_human_input", {"question": "Which db?", "confidence": 0.3, "impact": 0.9}, ctx)
    assert "Not asking" in r.text()
    ctx.human = HumanChannel(lambda q: "postgres")
    r = reg.execute("request_human_input", {"question": "Which db?", "confidence": 0.3, "impact": 0.9}, ctx)
    assert "postgres" in r.text()
    r = reg.execute("request_human_input", {"question": "Tabs?", "confidence": 0.9, "impact": 0.1}, ctx)
    assert "Not asking" in r.text()


def test_perceive_image_and_text(ctx, tmp_path):
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    (tmp_path / "pixel.png").write_bytes(png)
    (tmp_path / "notes.md").write_text("# hello")
    reg = default_registry()
    r = reg.execute("perceive_file", {"path": "pixel.png"}, ctx)
    assert any(b.get("type") == "image" for b in r.content)
    r = reg.execute("perceive_file", {"path": "notes.md"}, ctx)
    assert "# hello" in r.text()
    (tmp_path / "blob.bin").write_bytes(bytes(range(256)))
    r = reg.execute("perceive_file", {"path": "blob.bin"}, ctx)
    assert "NOT perceived" in r.text()
    env = json.loads(reg.execute("observe_environment", {}, ctx).text().split("\n[recorded")[0])
    assert "notes.md" in env["files"]


def test_compare_hypotheses_tool(ctx):
    data = [[x / 2, 3 * (x / 2) ** 2 + 1] for x in range(1, 20)]
    r = default_registry().execute("compare_hypotheses", {"data": data, "hypotheses": [
        {"name": "line", "formula": "a + b*x", "params": ["a", "b"]},
        {"name": "parabola", "formula": "a + b*x**2", "params": ["a", "b"]},
    ]}, ctx)
    report = json.loads(r.text().split("\n[recorded")[0])
    assert report["posterior"]["parabola"] > 0.99
    assert ctx.ledger.evidence[r.data["evidence_id"]].kind == "derivation"


def test_mark_conflict_tool(ctx):
    reg = default_registry()
    a = ctx.ledger.assert_claim("source A says 5", Status.SPECULATION, 0.5)
    b = ctx.ledger.assert_claim("source B says 7", Status.SPECULATION, 0.5)
    r = reg.execute("mark_conflict", {"claim_a": a.id, "claim_b": b.id, "note": "different units?"}, ctx)
    assert not r.is_error and len(ctx.ledger.unresolved_conflicts()) == 1
    assert reg.execute("mark_conflict", {"claim_a": a.id, "claim_b": "C999"}, ctx).is_error
