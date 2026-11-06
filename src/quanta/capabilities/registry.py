"""Persistent, versioned capability registry.

An append-only JSONL event log; the current view is obtained by replaying it.
Nothing is ever overwritten: evidence is appended, status changes are events,
and a changed artifact is a new version with the previous one preserved.

A capability becomes ACTIVE only through a status event that carries an
approving review from the independent reviewer (`quanta.bench.review`); the
registry refuses ACTIVE without one. (In-process checks like this are an
integrity guard and an audit trail, not a security boundary.)
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATUSES = ("EXPERIMENTAL", "ACTIVE", "REJECTED", "DEPRECATED")
EVIDENCE_FIELDS = ("tests", "selection_evidence", "heldout_evidence", "transfer_evidence", "review",
                   "retry_evidence")


class RegistryError(ValueError):
    pass


@dataclass
class CapabilityRecord:
    capability_id: str
    version: int
    name: str
    description: str
    origin: str
    triggering_failure: dict
    hypothesis: str
    artifact: dict
    prerequisites: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    tests: dict = field(default_factory=dict)
    selection_evidence: dict = field(default_factory=dict)
    heldout_evidence: dict = field(default_factory=dict)
    transfer_evidence: dict = field(default_factory=dict)
    review: dict = field(default_factory=dict)
    retry_evidence: dict = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    status: str = "EXPERIMENTAL"
    status_history: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class Registry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.records: dict[tuple[str, int], CapabilityRecord] = {}
        self._counter = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._apply(json.loads(line))

    # -- event log ---------------------------------------------------------
    def _append(self, event: dict) -> None:
        event = {"ts": time.time(), **event}
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
            self._apply(event)

    def _apply(self, e: dict) -> None:
        kind = e["event"]
        key = (e["capability_id"], e["version"])
        if kind == "propose":
            rec = CapabilityRecord(**e["record"])
            self.records[key] = rec
            n = int(e["capability_id"].split("-")[-1]) if e["capability_id"].split("-")[-1].isdigit() else 0
            self._counter = max(self._counter, n)
        elif kind == "evidence":
            rec = self.records[key]
            getattr(rec, e["field"]).update(e["value"])
            rec.updated_at = e["ts"]
        elif kind == "status":
            rec = self.records[key]
            rec.status = e["status"]
            rec.status_history.append({"ts": e["ts"], "status": e["status"], "reason": e["reason"],
                                       "review_id": e.get("review_id")})
            if "confidence" in e:
                rec.confidence = e["confidence"]
            if e.get("limitations"):
                rec.limitations.extend(e["limitations"])
            rec.updated_at = e["ts"]
        else:
            raise RegistryError(f"unknown event {kind!r}")

    # -- operations ----------------------------------------------------------
    def new_id(self) -> str:
        return f"cap-{self._counter + 1:04d}"

    def propose(self, record: CapabilityRecord) -> CapabilityRecord:
        key = (record.capability_id, record.version)
        if key in self.records:
            raise RegistryError(f"{key} already exists; propose a new version instead of overwriting")
        if record.status != "EXPERIMENTAL":
            raise RegistryError("new capabilities start as EXPERIMENTAL")
        self._append({"event": "propose", "capability_id": record.capability_id, "version": record.version,
                      "record": asdict(record)})
        return self.records[key]

    def add_evidence(self, capability_id: str, version: int, field_name: str, value: dict) -> None:
        if field_name not in EVIDENCE_FIELDS:
            raise RegistryError(f"unknown evidence field {field_name!r}")
        self._get(capability_id, version)
        self._append({"event": "evidence", "capability_id": capability_id, "version": version,
                      "field": field_name, "value": value})

    def set_status(self, capability_id: str, version: int, status: str, reason: str, *,
                   review: dict | None = None, confidence: float | None = None,
                   limitations: list[str] | None = None) -> None:
        if status not in STATUSES:
            raise RegistryError(f"unknown status {status!r}")
        rec = self._get(capability_id, version)
        if status == "ACTIVE":
            if not review or not review.get("approved") or not review.get("review_id"):
                raise RegistryError("ACTIVE requires an approving independent review")
            if rec.status != "EXPERIMENTAL":
                raise RegistryError(f"only EXPERIMENTAL capabilities can become ACTIVE (is {rec.status})")
        if rec.status in ("REJECTED", "DEPRECATED") and status != rec.status:
            raise RegistryError(f"{rec.status} is final; propose a new version instead")
        event: dict[str, Any] = {"event": "status", "capability_id": capability_id, "version": version,
                                 "status": status, "reason": reason,
                                 "review_id": (review or {}).get("review_id")}
        if confidence is not None:
            event["confidence"] = confidence
        if limitations:
            event["limitations"] = limitations
        self._append(event)

    def _get(self, capability_id: str, version: int) -> CapabilityRecord:
        try:
            return self.records[(capability_id, version)]
        except KeyError:
            raise RegistryError(f"unknown capability {capability_id} v{version}") from None

    def get(self, capability_id: str, version: int | None = None) -> CapabilityRecord:
        if version is None:
            versions = [v for (c, v) in self.records if c == capability_id]
            if not versions:
                raise RegistryError(f"unknown capability {capability_id}")
            version = max(versions)
        return self._get(capability_id, version)

    def by_status(self, status: str) -> list[CapabilityRecord]:
        return [r for r in self.records.values() if r.status == status]

    def active_artifacts(self) -> list[dict]:
        """Artifacts of ACTIVE capabilities (latest active version per id), in creation order."""
        latest: dict[str, CapabilityRecord] = {}
        for r in sorted(self.records.values(), key=lambda r: (r.created_at, r.version)):
            if r.status == "ACTIVE":
                latest[r.capability_id] = r
        return [r.artifact for r in latest.values()]

    def summary(self) -> dict:
        out = {s: 0 for s in STATUSES}
        for r in self.records.values():
            out[r.status] += 1
        return out
