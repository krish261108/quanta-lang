"""Human-control invariants.

PROTECTED: the self-improver refuses candidates that modify this file or
quanta/governance.py, and every code-patch candidate must pass this file in
its sandbox before it is even benchmarked.
"""
import json

import pytest

from quanta.governance import (
    Action, AllowlistApprover, AllowRule, AuditLog, Budget, BudgetExceeded,
    DenyAllApprover, Governor, KillSwitch, KillSwitchTripped, Risk,
)


class AlwaysYes:
    name = "always-yes"

    def __init__(self):
        self.calls = 0

    def approve(self, action):
        from quanta.governance import Decision
        self.calls += 1
        return Decision(True, "yes", self.name)


def test_high_and_critical_never_auto_approved_even_if_configured():
    approver = DenyAllApprover()
    gov = Governor(approver=approver, auto_approve_up_to=Risk.CRITICAL)
    assert gov.auto_approve_up_to == Risk.LOW
    assert gov.authorize(Action("t", {}, Risk.LOW)).allowed
    assert not gov.authorize(Action("t", {}, Risk.HIGH)).allowed
    assert not gov.authorize(Action("t", {}, Risk.CRITICAL)).allowed


def test_high_risk_goes_to_approver():
    yes = AlwaysYes()
    gov = Governor(approver=yes)
    assert gov.authorize(Action("deploy", {}, Risk.HIGH)).allowed
    assert yes.calls == 1
    gov.authorize(Action("read", {}, Risk.READ_ONLY))
    assert yes.calls == 1


def test_critical_cannot_be_preauthorized():
    rules = [AllowRule("rm", max_risk=Risk.CRITICAL)]
    gov = Governor(approver=AllowlistApprover(rules))
    assert not gov.authorize(Action("rm", {}, Risk.CRITICAL)).allowed
    assert gov.authorize(Action("rm", {}, Risk.HIGH)).allowed


def test_allowlist_match_function():
    rules = [AllowRule("shell", Risk.HIGH, match=lambda a: a.get("command", "").startswith("pytest"))]
    gov = Governor(approver=AllowlistApprover(rules))
    assert gov.authorize(Action("shell", {"command": "pytest -q"}, Risk.HIGH)).allowed
    assert not gov.authorize(Action("shell", {"command": "git push"}, Risk.HIGH)).allowed


def test_kill_switch_blocks_everything(tmp_path):
    stop = tmp_path / "STOP"
    gov = Governor(approver=AlwaysYes(), kill_switch=KillSwitch(stop))
    assert gov.authorize(Action("t", {}, Risk.READ_ONLY)).allowed
    stop.write_text("halt")
    assert not gov.authorize(Action("t", {}, Risk.READ_ONLY)).allowed
    with pytest.raises(KillSwitchTripped):
        gov.check_running()


def test_budget_exhaustion_blocks_and_children_charge_parent():
    gov = Governor(budget=Budget(max_steps=3))
    child = gov.child(max_steps=10)
    child.budget.charge(steps=3)
    assert gov.budget.steps == 3
    assert not gov.authorize(Action("t", {}, Risk.READ_ONLY)).allowed
    with pytest.raises(BudgetExceeded):
        child.check_running()


def test_audit_log_is_tamper_evident(tmp_path):
    path = tmp_path / "audit.jsonl"
    gov = Governor(audit=AuditLog(path))
    for i in range(5):
        gov.authorize(Action("t", {"i": i}, Risk.LOW))
    log = AuditLog(path)
    assert log.verify() == (True, None)
    lines = path.read_text().splitlines()
    entry = json.loads(lines[2])
    entry["allowed"] = not entry["allowed"]
    lines[2] = json.dumps(entry)
    path.write_text("\n".join(lines) + "\n")
    ok, bad = AuditLog(path).verify()
    assert not ok and bad == 2


def test_every_decision_is_audited():
    gov = Governor()
    gov.authorize(Action("a", {}, Risk.LOW))
    gov.authorize(Action("b", {}, Risk.HIGH))
    assert [e["tool"] for e in gov.audit.entries] == ["a", "b"]
    assert [e["allowed"] for e in gov.audit.entries] == [True, False]


def test_default_approver_denies():
    assert not Governor().authorize(Action("x", {}, Risk.HIGH)).allowed
