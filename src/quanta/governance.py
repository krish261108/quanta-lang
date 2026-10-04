"""Human control: every side-effecting action passes through one chokepoint.

Invariants (enforced here and covered by tests/test_governance_invariants.py,
which the self-improver is not allowed to modify):

1. Actions classified HIGH or CRITICAL are never auto-approved. The
   auto-approval ceiling is clamped to LOW in code; no configuration, genome
   or model output can raise it.
2. When the kill switch is tripped, no further action is authorized.
3. When the budget is exhausted, no further action is authorized.
4. Every authorization decision is written to a hash-chained, append-only
   audit log, so after-the-fact tampering is detectable.
5. With no human available, the default approver denies.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Protocol


class Risk(IntEnum):
    READ_ONLY = 0   # observes only
    LOW = 1         # local, reversible side effects inside the workspace
    HIGH = 2        # hard to reverse, external side effects, spends money
    CRITICAL = 3    # irreversible, or affects people/systems outside the sandbox


# Hard ceiling for automatic approval. Not configurable on purpose.
MAX_AUTO_APPROVE = Risk.LOW


@dataclass(frozen=True)
class Action:
    tool: str
    args: dict[str, Any]
    risk: Risk
    description: str = ""
    actor: str = "agent"


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    approver: str


class GovernanceError(RuntimeError):
    pass


class BudgetExceeded(GovernanceError):
    pass


class KillSwitchTripped(GovernanceError):
    pass


# ---------------------------------------------------------------------------
# Approvers: who may authorize HIGH / CRITICAL actions
# ---------------------------------------------------------------------------

class Approver(Protocol):
    name: str

    def approve(self, action: Action) -> Decision: ...


class DenyAllApprover:
    """Default when no human is reachable: refuse anything needing approval."""

    name = "deny-all"

    def approve(self, action: Action) -> Decision:
        return Decision(False, "no human approver available; action requires authorization", self.name)


class InteractiveApprover:
    """Ask a human on the terminal."""

    name = "interactive"

    def __init__(self, input_fn: Callable[[str], str] = input, out=sys.stderr) -> None:
        self.input_fn = input_fn
        self.out = out

    def approve(self, action: Action) -> Decision:
        print(f"\n[approval required] risk={action.risk.name} tool={action.tool}\n"
              f"  {action.description}\n  args={json.dumps(action.args, default=str)[:800]}",
              file=self.out)
        try:
            answer = self.input_fn("Approve? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        ok = answer in ("y", "yes")
        return Decision(ok, "approved by human" if ok else "denied by human", self.name)


@dataclass
class AllowRule:
    """Pre-authorization for a specific tool up to a risk level.

    `match` optionally restricts it further: it is called with the action args.
    CRITICAL actions can never be pre-authorized.
    """
    tool: str
    max_risk: Risk = Risk.HIGH
    match: Callable[[dict], bool] | None = None
    note: str = ""


class AllowlistApprover:
    name = "allowlist"

    def __init__(self, rules: list[AllowRule], fallback: Approver | None = None) -> None:
        self.rules = rules
        self.fallback = fallback or DenyAllApprover()

    def approve(self, action: Action) -> Decision:
        if action.risk < Risk.CRITICAL:
            for rule in self.rules:
                if rule.tool == action.tool and action.risk <= min(rule.max_risk, Risk.HIGH):
                    if rule.match is None or rule.match(action.args):
                        return Decision(True, f"pre-authorized: {rule.note or rule.tool}", self.name)
        return self.fallback.approve(action)


# ---------------------------------------------------------------------------
# Budget, kill switch, audit log
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    max_steps: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_seconds: float | None = None
    steps: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    started_at: float = field(default_factory=time.time)
    parent: "Budget | None" = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def charge(self, *, steps: int = 0, tokens: int = 0, cost_usd: float = 0.0) -> None:
        with self._lock:
            self.steps += steps
            self.tokens += tokens
            self.cost_usd += cost_usd
        if self.parent is not None:
            self.parent.charge(steps=steps, tokens=tokens, cost_usd=cost_usd)

    def exhausted_reason(self) -> str | None:
        if self.max_steps is not None and self.steps >= self.max_steps:
            return f"step budget exhausted ({self.steps}/{self.max_steps})"
        if self.max_tokens is not None and self.tokens >= self.max_tokens:
            return f"token budget exhausted ({self.tokens}/{self.max_tokens})"
        if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            return f"cost budget exhausted (${self.cost_usd:.2f}/${self.max_cost_usd:.2f})"
        if self.max_seconds is not None and time.time() - self.started_at >= self.max_seconds:
            return "wall-clock budget exhausted"
        if self.parent is not None:
            return self.parent.exhausted_reason()
        return None

    def child(self, fraction: float = 0.25, max_steps: int | None = None) -> "Budget":
        """A sub-budget for a sub-agent; spending is charged to the parent too."""
        def part(limit, used):
            return None if limit is None else max(0, (limit - used)) * fraction
        return Budget(
            max_steps=max_steps,
            max_tokens=None if self.max_tokens is None else int(part(self.max_tokens, self.tokens)),
            max_cost_usd=part(self.max_cost_usd, self.cost_usd),
            max_seconds=None if self.max_seconds is None
            else part(self.max_seconds, time.time() - self.started_at),
            parent=self,
        )

    def snapshot(self) -> dict:
        return {"steps": self.steps, "tokens": self.tokens, "cost_usd": round(self.cost_usd, 4),
                "elapsed_s": round(time.time() - self.started_at, 1),
                "limits": {"steps": self.max_steps, "tokens": self.max_tokens,
                           "cost_usd": self.max_cost_usd, "seconds": self.max_seconds}}


class KillSwitch:
    """Tripped programmatically or by creating a file (e.g. `touch STOP`)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._tripped = False
        self.reason = ""

    def trip(self, reason: str = "tripped") -> None:
        self._tripped = True
        self.reason = reason

    def is_tripped(self) -> bool:
        if self._tripped:
            return True
        if self.path is not None and self.path.exists():
            self.reason = f"stop file present: {self.path}"
            return True
        return False


GENESIS_HASH = "0" * 64


def _entry_hash(prev: str, body: dict) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev + canonical).encode("utf-8")).hexdigest()


class AuditLog:
    """Append-only JSONL log; each entry commits to the previous one's hash."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.entries: list[dict] = []
        self._lock = threading.Lock()
        self.head = GENESIS_HASH
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.entries.append(entry)
                    self.head = entry["hash"]

    def append(self, event: str, **fields: Any) -> dict:
        with self._lock:
            body = {"ts": time.time(), "event": event, **fields}
            h = _entry_hash(self.head, body)
            entry = {**body, "prev": self.head, "hash": h}
            self.entries.append(entry)
            self.head = h
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
            return entry

    @staticmethod
    def verify_entries(entries: list[dict]) -> tuple[bool, int | None]:
        """Return (ok, index_of_first_bad_entry)."""
        prev = GENESIS_HASH
        for i, entry in enumerate(entries):
            body = {k: v for k, v in entry.items() if k not in ("prev", "hash")}
            if entry.get("prev") != prev or _entry_hash(prev, body) != entry.get("hash"):
                return False, i
            prev = entry["hash"]
        return True, None

    def verify(self) -> tuple[bool, int | None]:
        return self.verify_entries(self.entries)


# ---------------------------------------------------------------------------
# The governor
# ---------------------------------------------------------------------------

class Governor:
    def __init__(
        self,
        *,
        approver: Approver | None = None,
        budget: Budget | None = None,
        kill_switch: KillSwitch | None = None,
        audit: AuditLog | None = None,
        auto_approve_up_to: Risk = Risk.LOW,
    ) -> None:
        self.approver = approver or DenyAllApprover()
        self.budget = budget or Budget()
        self.kill_switch = kill_switch or KillSwitch()
        self.audit = audit or AuditLog()
        # Invariant 1: the ceiling can be lowered but never raised above LOW.
        self.auto_approve_up_to = Risk(min(int(auto_approve_up_to), int(MAX_AUTO_APPROVE)))

    def check_running(self) -> None:
        """Raise if the run must stop (kill switch or budget)."""
        if self.kill_switch.is_tripped():
            raise KillSwitchTripped(self.kill_switch.reason or "kill switch tripped")
        reason = self.budget.exhausted_reason()
        if reason:
            raise BudgetExceeded(reason)

    def authorize(self, action: Action) -> Decision:
        if self.kill_switch.is_tripped():
            decision = Decision(False, f"kill switch: {self.kill_switch.reason}", "governor")
        elif (reason := self.budget.exhausted_reason()) is not None:
            decision = Decision(False, reason, "governor")
        elif action.risk <= self.auto_approve_up_to:
            decision = Decision(True, f"auto-approved ({action.risk.name})", "governor")
        else:
            decision = self.approver.approve(action)
        self.audit.append(
            "authorize", tool=action.tool, risk=action.risk.name, actor=action.actor,
            description=action.description[:500],
            args=json.dumps(action.args, default=str)[:2000],
            allowed=decision.allowed, reason=decision.reason, approver=decision.approver,
        )
        return decision

    def child(self, fraction: float = 0.25, max_steps: int | None = None) -> "Governor":
        """Governor for a sub-agent: same approver, kill switch and audit log;
        a sub-budget that also charges the parent."""
        return Governor(
            approver=self.approver,
            budget=self.budget.child(fraction, max_steps),
            kill_switch=self.kill_switch,
            audit=self.audit,
            auto_approve_up_to=self.auto_approve_up_to,
        )
