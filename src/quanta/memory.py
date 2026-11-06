"""Persistent long-term memory backed by SQLite (full-text search via FTS5).

Three stores:

* episodic   - what happened (actions, observations, outcomes) per run
* semantic   - knowledge: statements with epistemic status, confidence, source
* procedural - reusable strategies/lessons with a track record
  (successes/failures from *verified* outcomes)

Learning rule: only verified experience changes behaviour. Unverified items
are stored (so they can be audited and later verified) but retrieval for
decision-making defaults to `verified_only=True` for procedures and priors.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

KINDS = ("episode", "knowledge", "procedure")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items(
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,
  ts REAL NOT NULL,
  run_id TEXT,
  subject TEXT NOT NULL DEFAULT '',
  content TEXT NOT NULL,
  status TEXT,
  confidence REAL,
  verified INTEGER NOT NULL DEFAULT 0,
  successes INTEGER NOT NULL DEFAULT 0,
  failures INTEGER NOT NULL DEFAULT 0,
  meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS items_kind_subject ON items(kind, subject);
CREATE INDEX IF NOT EXISTS items_run ON items(run_id);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
  subject, content, content='items', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS items_ai AFTER INSERT ON items BEGIN
  INSERT INTO items_fts(rowid, subject, content) VALUES (new.id, new.subject, new.content);
END;
CREATE TRIGGER IF NOT EXISTS items_ad AFTER DELETE ON items BEGIN
  INSERT INTO items_fts(items_fts, rowid, subject, content) VALUES('delete', old.id, old.subject, old.content);
END;
CREATE TRIGGER IF NOT EXISTS items_au AFTER UPDATE OF subject, content ON items BEGIN
  INSERT INTO items_fts(items_fts, rowid, subject, content) VALUES('delete', old.id, old.subject, old.content);
  INSERT INTO items_fts(rowid, subject, content) VALUES (new.id, new.subject, new.content);
END;
"""

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class MemoryItem:
    id: int
    kind: str
    ts: float
    run_id: str | None
    subject: str
    content: str
    status: str | None
    confidence: float | None
    verified: bool
    successes: int
    failures: int
    meta: dict[str, Any]
    score: float = 0.0

    @property
    def success_rate(self) -> float:
        """Laplace-smoothed success rate."""
        return (self.successes + 1) / (self.successes + self.failures + 2)


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            try:
                self._conn.executescript(_FTS_SCHEMA)
                self.fts = True
            except sqlite3.OperationalError:
                self.fts = False
            self._conn.commit()

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writes ----------------------------------------------------------
    def _insert(self, kind: str, subject: str, content: str, *, run_id: str | None = None,
                status: str | None = None, confidence: float | None = None,
                verified: bool = False, meta: dict | None = None) -> int:
        if kind not in KINDS:
            raise ValueError(f"unknown memory kind {kind!r}")
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO items(kind, ts, run_id, subject, content, status, confidence, verified, meta)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (kind, time.time(), run_id, subject, content, status, confidence,
                 int(verified), json.dumps(meta or {}, default=str)),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def record_episode(self, run_id: str, subject: str, content: str, meta: dict | None = None) -> int:
        return self._insert("episode", subject, content, run_id=run_id, meta=meta)

    def add_knowledge(self, subject: str, content: str, *, status: str = "speculation",
                      confidence: float = 0.5, verified: bool = False,
                      run_id: str | None = None, meta: dict | None = None) -> int:
        return self._insert("knowledge", subject, content, run_id=run_id, status=status,
                            confidence=confidence, verified=verified, meta=meta)

    def add_procedure(self, name: str, content: str, *, meta: dict | None = None,
                      run_id: str | None = None) -> int:
        return self._insert("procedure", name, content, run_id=run_id, meta=meta)

    def record_outcome(self, item_id: int, success: bool) -> None:
        """Record a *verified* outcome of applying a procedure/knowledge item."""
        col = "successes" if success else "failures"
        with self._lock:
            self._conn.execute(f"UPDATE items SET {col} = {col} + 1, verified = 1 WHERE id = ?", (item_id,))
            self._conn.commit()

    def mark_verified(self, item_id: int, verified: bool = True) -> None:
        with self._lock:
            self._conn.execute("UPDATE items SET verified = ? WHERE id = ?", (int(verified), item_id))
            self._conn.commit()

    # -- reads -----------------------------------------------------------
    @staticmethod
    def _row(row: sqlite3.Row, score: float = 0.0) -> MemoryItem:
        return MemoryItem(
            id=row["id"], kind=row["kind"], ts=row["ts"], run_id=row["run_id"],
            subject=row["subject"], content=row["content"], status=row["status"],
            confidence=row["confidence"], verified=bool(row["verified"]),
            successes=row["successes"], failures=row["failures"],
            meta=json.loads(row["meta"] or "{}"), score=score,
        )

    def get(self, item_id: int) -> MemoryItem | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        return self._row(row) if row else None

    def search(self, query: str, *, kinds: Iterable[str] = KINDS, limit: int = 10,
               verified_only: bool = False, subject: str | None = None) -> list[MemoryItem]:
        kinds = tuple(kinds)
        tokens = _TOKEN.findall(query.lower())
        if not tokens:
            return []
        where = [f"i.kind IN ({','.join('?' * len(kinds))})"]
        params: list[Any] = list(kinds)
        if verified_only:
            where.append("i.verified = 1")
        if subject is not None:
            where.append("i.subject = ?")
            params.append(subject)
        with self._lock:
            if self.fts:
                match = " OR ".join(f'"{t}"' for t in tokens)
                sql = (
                    "SELECT i.*, bm25(items_fts) AS rank FROM items_fts JOIN items i ON i.id = items_fts.rowid"
                    f" WHERE items_fts MATCH ? AND {' AND '.join(where)} ORDER BY rank LIMIT ?"
                )
                rows = self._conn.execute(sql, [match, *params, limit * 3]).fetchall()
                items = [self._row(r, score=-float(r["rank"])) for r in rows]
            else:  # token-overlap fallback
                rows = self._conn.execute(
                    f"SELECT * FROM items i WHERE {' AND '.join(where)}", params).fetchall()
                qset = set(tokens)
                items = []
                for r in rows:
                    words = set(_TOKEN.findall((r["subject"] + " " + r["content"]).lower()))
                    overlap = len(qset & words)
                    if overlap:
                        items.append(self._row(r, score=overlap / len(qset)))
        # Verified experience with a good track record ranks higher.
        for it in items:
            if it.kind == "procedure":
                it.score *= 0.5 + it.success_rate
            if it.verified:
                it.score *= 1.25
        items.sort(key=lambda it: -it.score)
        return items[:limit]

    def recent(self, *, kind: str = "episode", run_id: str | None = None, limit: int = 20) -> list[MemoryItem]:
        sql = "SELECT * FROM items WHERE kind = ?"
        params: list[Any] = [kind]
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(run_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row(r) for r in rows]

    def count_by_meta(self, kind: str, subject: str, key: str, *, verified_only: bool = True) -> Counter:
        """Histogram of `meta[key]` over items, e.g. which hypothesis family
        turned out to be true in past verified tasks of a domain."""
        sql = "SELECT meta FROM items WHERE kind = ? AND subject = ?"
        if verified_only:
            sql += " AND verified = 1"
        with self._lock:
            rows = self._conn.execute(sql, (kind, subject)).fetchall()
        counts: Counter = Counter()
        for (meta_json,) in rows:
            value = json.loads(meta_json or "{}").get(key)
            if value is not None:
                counts[value] += 1
        return counts

    def stats(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, COUNT(*), SUM(verified) FROM items GROUP BY kind").fetchall()
        out = {f"{k}": int(n) for k, n, _ in rows}
        out.update({f"{k}_verified": int(v or 0) for k, _, v in rows})
        return out

    def context_block(self, query: str, *, limit: int = 8) -> str:
        """Relevant memories formatted for a model prompt."""
        items = self.search(query, kinds=("knowledge", "procedure"), limit=limit)
        if not items:
            return "(no relevant memories)"
        lines = []
        for it in items:
            tag = "verified" if it.verified else "unverified"
            if it.kind == "procedure":
                lines.append(f"- [procedure, {tag}, {it.successes}W/{it.failures}L] {it.subject}: {it.content}")
            else:
                lines.append(f"- [knowledge, {tag}, {it.status}, p={it.confidence}] {it.subject}: {it.content}")
        return "\n".join(lines)
