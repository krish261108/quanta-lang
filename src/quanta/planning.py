"""Long-horizon plans as dependency graphs.

A plan is a DAG of tasks with acceptance criteria. The graph supports
thousands of nodes, hierarchical refinement (replace a task by a sub-plan
without breaking the tasks that depended on it), bounded retries, failure
propagation, and JSON checkpointing so a run can be resumed after a crash.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"      # exhausted its attempts
    BLOCKED = "blocked"    # a dependency failed
    SKIPPED = "skipped"    # deliberately dropped during replanning
    REFINED = "refined"    # replaced by a sub-plan


TERMINAL = {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.SKIPPED, TaskStatus.REFINED}
SATISFIED = {TaskStatus.DONE, TaskStatus.SKIPPED, TaskStatus.REFINED}


class PlanError(ValueError):
    pass


@dataclass
class Task:
    id: str
    title: str
    description: str = ""
    deps: list[str] = field(default_factory=list)
    acceptance: str = ""
    role: str = "generalist"
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    result: str = ""
    errors: list[str] = field(default_factory=list)
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class TaskGraph:
    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}
        self._order: list[str] = []

    # -- construction ----------------------------------------------------
    def add(self, task: Task) -> Task:
        if task.id in self.tasks:
            raise PlanError(f"duplicate task id {task.id!r}")
        for d in task.deps:
            if d not in self.tasks:
                raise PlanError(f"task {task.id!r} depends on unknown task {d!r}")
        # deps all pre-exist, so adding a node can never create a cycle
        self.tasks[task.id] = task
        self._order.append(task.id)
        return task

    @staticmethod
    def _task_from_item(item: dict, **overrides: Any) -> Task:
        fields = {
            "id": str(item["id"]).strip(), "title": str(item.get("title", item["id"])),
            "description": str(item.get("description", "")),
            "deps": list(item.get("deps", []) or []),
            "acceptance": str(item.get("acceptance", "")),
            "role": str(item.get("role", "generalist")),
            "max_attempts": int(item.get("max_attempts", 3)),
        }
        fields.update(overrides)
        return Task(**fields)

    @staticmethod
    def _toposort(by_id: dict[str, dict]) -> list[str]:
        """Order item ids so dependencies come first; deps outside `by_id`
        are ignored here (the caller validates them)."""
        order: list[str] = []
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(tid: str) -> None:
            if tid in done:
                return
            if tid in visiting:
                raise PlanError(f"dependency cycle through {tid!r}")
            visiting.add(tid)
            for d in by_id[tid].get("deps", []) or []:
                if d in by_id:
                    visit(d)
            visiting.discard(tid)
            done.add(tid)
            order.append(tid)

        for tid in by_id:
            visit(tid)
        return order

    @classmethod
    def from_spec(cls, spec: Iterable[dict]) -> "TaskGraph":
        """Build from a (possibly model-written) list of task dicts in any order.
        Validates ids, dependencies and acyclicity."""
        by_id: dict[str, dict] = {}
        for item in spec:
            tid = str(item.get("id", "")).strip()
            if not tid:
                raise PlanError("every task needs a non-empty id")
            if tid in by_id:
                raise PlanError(f"duplicate task id {tid!r}")
            by_id[tid] = {**item, "id": tid}
        for tid, item in by_id.items():
            for d in item.get("deps", []) or []:
                if d not in by_id:
                    raise PlanError(f"task {tid!r} depends on unknown task {d!r}")
        g = cls()
        for tid in cls._toposort(by_id):
            g.add(cls._task_from_item(by_id[tid]))
        return g

    # -- state transitions ----------------------------------------------
    def get(self, tid: str) -> Task:
        try:
            return self.tasks[tid]
        except KeyError:
            raise PlanError(f"unknown task {tid!r}") from None

    def _is_ready(self, t: Task) -> bool:
        return t.status is TaskStatus.PENDING and all(self.tasks[d].status in SATISFIED for d in t.deps)

    def ready(self) -> list[Task]:
        return [self.tasks[tid] for tid in self._order if self._is_ready(self.tasks[tid])]

    def start(self, tid: str) -> Task:
        t = self.get(tid)
        if not self._is_ready(t):
            raise PlanError(f"task {tid!r} is not ready (status={t.status.value})")
        t.status = TaskStatus.RUNNING
        t.attempts += 1
        return t

    def complete(self, tid: str, result: str = "") -> Task:
        t = self.get(tid)
        if t.status is not TaskStatus.RUNNING:
            raise PlanError(f"task {tid!r} is not running")
        t.status = TaskStatus.DONE
        t.result = result
        return t

    def fail(self, tid: str, error: str) -> Task:
        """Record a failed attempt. Retries until max_attempts, then fails
        permanently and blocks everything downstream."""
        t = self.get(tid)
        if t.status is not TaskStatus.RUNNING:
            raise PlanError(f"task {tid!r} is not running")
        t.errors.append(error)
        if t.attempts >= t.max_attempts:
            t.status = TaskStatus.FAILED
            self._block_dependents(tid)
        else:
            t.status = TaskStatus.PENDING
        return t

    def skip(self, tid: str, reason: str) -> Task:
        t = self.get(tid)
        if t.status in TERMINAL:
            raise PlanError(f"task {tid!r} already terminal")
        t.status = TaskStatus.SKIPPED
        t.result = f"skipped: {reason}"
        return t

    def _block_dependents(self, tid: str) -> None:
        frontier = [tid]
        while frontier:
            cur = frontier.pop()
            for t in self.tasks.values():
                if cur in t.deps and t.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                    t.status = TaskStatus.BLOCKED
                    t.errors.append(f"blocked: dependency {cur} failed")
                    frontier.append(t.id)

    def unblock(self, tid: str) -> None:
        """Re-open a failed task (e.g. after replanning provided a new approach)."""
        t = self.get(tid)
        if t.status is not TaskStatus.FAILED:
            raise PlanError(f"task {tid!r} is not failed")
        t.status = TaskStatus.PENDING
        t.attempts = 0
        frontier = [tid]
        while frontier:
            cur = frontier.pop()
            for other in self.tasks.values():
                if cur in other.deps and other.status is TaskStatus.BLOCKED:
                    other.status = TaskStatus.PENDING
                    frontier.append(other.id)

    def refine(self, tid: str, subtasks: list[dict]) -> list[Task]:
        """Hierarchical decomposition: replace `tid` by a sub-plan.

        Sub-task ids are namespaced as `tid.<id>`. Sub-tasks inherit the
        parent's dependencies (sub-tasks with no internal deps depend on the
        parent's deps). Tasks that depended on the parent now wait for the
        parent, which counts as satisfied only when all children are done.
        """
        parent = self.get(tid)
        if parent.status in TERMINAL:
            raise PlanError(f"cannot refine terminal task {tid!r}")
        local_ids = {str(s["id"]) for s in subtasks}
        by_id: dict[str, dict] = {}
        for s in subtasks:
            deps = s.get("deps", []) or []
            internal = [f"{tid}.{d}" for d in deps if d in local_ids]
            external = [d for d in deps if d not in local_ids]
            for d in external:
                if d not in self.tasks:
                    raise PlanError(f"subtask depends on unknown task {d!r}")
            sid = f"{tid}.{s['id']}"
            if sid in by_id or sid in self.tasks:
                raise PlanError(f"duplicate task id {sid!r}")
            by_id[sid] = {**s, "id": sid, "deps": (internal + external) or list(parent.deps)}
        created = []
        for sid in self._toposort(by_id):
            created.append(self.add(self._task_from_item(by_id[sid], parent=tid)))
        parent.children = [t.id for t in created]
        parent.status = TaskStatus.REFINED
        # Dependents of the parent must now wait for the children's completion.
        for t in self.tasks.values():
            if tid in t.deps and t.parent != tid:
                t.deps = [d for d in t.deps if d != tid] + parent.children
        return created

    # -- queries ---------------------------------------------------------
    def is_complete(self) -> bool:
        return all(t.status in TERMINAL for t in self.tasks.values())

    def succeeded(self) -> bool:
        return all(t.status in SATISFIED for t in self.tasks.values())

    def is_stuck(self) -> bool:
        """Nothing running, nothing ready, but not complete."""
        if self.is_complete():
            return False
        running = any(t.status is TaskStatus.RUNNING for t in self.tasks.values())
        return not running and not self.ready()

    def progress(self) -> float:
        work = [t for t in self.tasks.values() if t.status is not TaskStatus.REFINED]
        if not work:
            return 1.0
        return sum(1 for t in work if t.status in SATISFIED) / len(work)

    def topological_order(self) -> list[str]:
        return list(self._order)

    def critical_path_length(self) -> int:
        depth: dict[str, int] = {}
        for tid in self._order:
            t = self.tasks[tid]
            depth[tid] = 1 + max((depth[d] for d in t.deps), default=0)
        return max(depth.values(), default=0)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.tasks.values():
            out[t.status.value] = out.get(t.status.value, 0) + 1
        return out

    def summary(self, max_items: int = 30) -> str:
        lines = [f"progress={self.progress():.0%} counts={self.counts()}"]
        for tid in self._order[:max_items]:
            t = self.tasks[tid]
            if t.status is TaskStatus.REFINED:
                continue
            deps = f" deps={t.deps}" if t.deps else ""
            lines.append(f"- [{t.status.value}] {t.id}: {t.title}{deps}")
        if len(self._order) > max_items:
            lines.append(f"... ({len(self._order) - max_items} more)")
        return "\n".join(lines)

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> dict:
        return {"order": self._order,
                "tasks": [{**asdict(t), "status": t.status.value} for t in self.tasks.values()]}

    @classmethod
    def from_dict(cls, data: dict) -> "TaskGraph":
        g = cls()
        for t in data["tasks"]:
            t = dict(t)
            t["status"] = TaskStatus(t["status"])
            g.tasks[t["id"]] = Task(**t)
        g._order = list(data["order"])
        return g

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TaskGraph":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
