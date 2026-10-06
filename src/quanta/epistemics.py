"""The evidence ledger: an explicit, mechanically-enforced distinction between
established facts, inferences and speculation.

The rules are enforced by code, not by asking a model to be careful:

* A claim may only be recorded as FACT if it cites at least one piece of
  *grounding* evidence (a direct observation, an executed command, a cited
  source, or an independent verification). Otherwise it is downgraded.
* An INFERENCE must rest on other claims or on evidence; otherwise it is
  SPECULATION.
* An inference cannot be more confident than its weakest premise.
* When a claim is refuted, everything that (transitively) depends on it is
  flagged for review and its confidence is capped. This is how a single
  discovered error propagates instead of silently surviving in conclusions.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class Status(str, Enum):
    FACT = "fact"
    INFERENCE = "inference"
    SPECULATION = "speculation"


GROUNDING_KINDS = frozenset({"observation", "execution", "source", "verification"})
EVIDENCE_KINDS = GROUNDING_KINDS | {"derivation", "falsification", "note"}

# Confidence cap applied to claims that depend on a refuted claim.
REFUTED_DEPENDENT_CAP = 0.2


@dataclass
class Evidence:
    id: str
    kind: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class FalsificationAttempt:
    test: str
    outcome: str  # "survived" | "refuted" | "inconclusive"
    evidence_ids: list[str] = field(default_factory=list)
    at: float = field(default_factory=time.time)


@dataclass
class Claim:
    id: str
    statement: str
    status: Status
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    falsifications: list[FalsificationAttempt] = field(default_factory=list)
    author: str = "agent"
    refuted: bool = False
    needs_review: bool = False
    notes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def survived_falsification(self) -> int:
        return sum(1 for f in self.falsifications if f.outcome == "survived")


class LedgerError(ValueError):
    pass


class Ledger:
    """Thread-safe store of evidence and claims with epistemic discipline."""

    def __init__(self) -> None:
        self.evidence: dict[str, Evidence] = {}
        self.claims: dict[str, Claim] = {}
        self._counter = 0
        self._lock = threading.RLock()

    # -- ids -------------------------------------------------------------
    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    # -- evidence --------------------------------------------------------
    def add_evidence(self, kind: str, summary: str, data: dict | None = None) -> Evidence:
        if kind not in EVIDENCE_KINDS:
            raise LedgerError(f"unknown evidence kind {kind!r}; expected one of {sorted(EVIDENCE_KINDS)}")
        with self._lock:
            ev = Evidence(self._next_id("E"), kind, summary, dict(data or {}))
            self.evidence[ev.id] = ev
            return ev

    # -- claims ----------------------------------------------------------
    def assert_claim(
        self,
        statement: str,
        status: Status | str,
        confidence: float,
        *,
        evidence: Iterable[str] = (),
        depends_on: Iterable[str] = (),
        author: str = "agent",
    ) -> Claim:
        status = Status(status)
        evidence = list(evidence)
        depends_on = list(depends_on)
        if not 0.0 <= confidence <= 1.0:
            raise LedgerError("confidence must be in [0, 1]")
        with self._lock:
            for eid in evidence:
                if eid not in self.evidence:
                    raise LedgerError(f"unknown evidence id {eid!r}")
            for cid in depends_on:
                if cid not in self.claims:
                    raise LedgerError(f"unknown claim id {cid!r}")
            notes: list[str] = []
            grounded = any(self.evidence[e].kind in GROUNDING_KINDS for e in evidence)
            if status is Status.FACT and not grounded:
                status = Status.INFERENCE if (depends_on or evidence) else Status.SPECULATION
                notes.append("downgraded from fact: no grounding evidence (observation/execution/source/verification)")
            if status is Status.INFERENCE and not (depends_on or evidence):
                status = Status.SPECULATION
                notes.append("downgraded from inference: no premises or evidence")
            if status is Status.INFERENCE and depends_on:
                cap = min(self.claims[c].confidence for c in depends_on)
                if confidence > cap:
                    notes.append(f"confidence capped at weakest premise ({cap:.3f})")
                    confidence = cap
            needs_review = any(self.claims[c].refuted or self.claims[c].needs_review for c in depends_on)
            if needs_review:
                confidence = min(confidence, REFUTED_DEPENDENT_CAP)
                notes.append("depends on a refuted or under-review claim")
            claim = Claim(
                id=self._next_id("C"), statement=statement, status=status,
                confidence=confidence, evidence_ids=evidence, depends_on=depends_on,
                author=author, needs_review=needs_review, notes=notes,
            )
            self.claims[claim.id] = claim
            return claim

    def get(self, claim_id: str) -> Claim:
        try:
            return self.claims[claim_id]
        except KeyError:
            raise LedgerError(f"unknown claim id {claim_id!r}") from None

    def update_confidence(self, claim_id: str, confidence: float, note: str) -> Claim:
        with self._lock:
            c = self.get(claim_id)
            if c.needs_review or c.refuted:
                confidence = min(confidence, REFUTED_DEPENDENT_CAP)
            c.confidence = max(0.0, min(1.0, confidence))
            c.notes.append(note)
            return c

    def promote_to_fact(self, claim_id: str, verification_evidence: str) -> Claim:
        """Promote a claim after an independent verification grounded it."""
        with self._lock:
            c = self.get(claim_id)
            ev = self.evidence.get(verification_evidence)
            if ev is None or ev.kind not in GROUNDING_KINDS:
                raise LedgerError("promotion to fact requires grounding evidence")
            if c.refuted:
                raise LedgerError("cannot promote a refuted claim")
            c.status = Status.FACT
            c.evidence_ids.append(verification_evidence)
            c.notes.append(f"promoted to fact by {verification_evidence}")
            return c

    def record_falsification(self, claim_id: str, test: str, outcome: str,
                             evidence: Iterable[str] = ()) -> Claim:
        if outcome not in ("survived", "refuted", "inconclusive"):
            raise LedgerError("outcome must be survived, refuted or inconclusive")
        with self._lock:
            c = self.get(claim_id)
            c.falsifications.append(FalsificationAttempt(test, outcome, list(evidence)))
            if outcome == "refuted":
                c.refuted = True
                c.confidence = min(c.confidence, 0.05)
                if c.status is Status.FACT:
                    c.status = Status.SPECULATION
                c.notes.append(f"refuted by test: {test}")
                self._propagate_review(claim_id)
            return c

    def _propagate_review(self, root: str) -> None:
        frontier = [root]
        seen = {root}
        while frontier:
            current = frontier.pop()
            for c in self.claims.values():
                if current in c.depends_on and c.id not in seen:
                    seen.add(c.id)
                    c.needs_review = True
                    c.confidence = min(c.confidence, REFUTED_DEPENDENT_CAP)
                    c.notes.append(f"premise {current} refuted or under review")
                    frontier.append(c.id)

    def mark_conflict(self, a: str, b: str) -> None:
        with self._lock:
            ca, cb = self.get(a), self.get(b)
            if b not in ca.contradicts:
                ca.contradicts.append(b)
            if a not in cb.contradicts:
                cb.contradicts.append(a)

    # -- queries ---------------------------------------------------------
    def by_status(self, status: Status | str) -> list[Claim]:
        status = Status(status)
        return [c for c in self.claims.values() if c.status is status and not c.refuted]

    def refuted(self) -> list[Claim]:
        return [c for c in self.claims.values() if c.refuted]

    def unresolved_conflicts(self) -> list[tuple[Claim, Claim]]:
        out = []
        for c in self.claims.values():
            for other_id in c.contradicts:
                other = self.claims[other_id]
                if c.id < other_id and not c.refuted and not other.refuted:
                    out.append((c, other))
        return out

    def summary(self, max_items: int = 40) -> str:
        """Compact text view for injecting into a model's context."""
        lines = []
        for status in Status:
            items = sorted(self.by_status(status), key=lambda c: -c.confidence)[:max_items]
            if items:
                lines.append(f"## {status.value.upper()}")
                for c in items:
                    flag = " [NEEDS REVIEW]" if c.needs_review else ""
                    lines.append(f"- {c.id} (p={c.confidence:.2f}){flag}: {c.statement}")
        ref = self.refuted()
        if ref:
            lines.append("## REFUTED")
            lines.extend(f"- {c.id}: {c.statement}" for c in ref[:max_items])
        conflicts = self.unresolved_conflicts()
        if conflicts:
            lines.append("## UNRESOLVED CONFLICTS")
            lines.extend(f"- {a.id} vs {b.id}" for a, b in conflicts)
        return "\n".join(lines) if lines else "(ledger empty)"

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> dict:
        with self._lock:
            return {
                "counter": self._counter,
                "evidence": [asdict(e) for e in self.evidence.values()],
                "claims": [
                    {**asdict(c), "status": c.status.value}
                    for c in self.claims.values()
                ],
            }

    @classmethod
    def from_dict(cls, data: dict) -> "Ledger":
        led = cls()
        led._counter = data.get("counter", 0)
        for e in data.get("evidence", []):
            led.evidence[e["id"]] = Evidence(**e)
        for c in data.get("claims", []):
            c = dict(c)
            c["status"] = Status(c["status"])
            c["falsifications"] = [FalsificationAttempt(**f) for f in c.get("falsifications", [])]
            led.claims[c["id"]] = Claim(**c)
        return led

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
