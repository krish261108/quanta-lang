"""The open-ended research loop for problems with no prescribed solution.

    frame -> research -> hypothesize -> plan -> execute -> update -> verify -> decide
                 ^                                                            |
                 +----------------- revise (next iteration) <-----------------+
                                                                              v
                                                                           deliver

* frame       : objective, success criteria, unknowns, assumptions; clarifying
                questions go to a human only if the expected loss justifies it
* research    : literature/docs/web; citations are checked against sources
* hypothesize : at least `genome.min_hypotheses` competing, falsifiable hypotheses
* plan        : a dependency graph of tasks with acceptance criteria
* execute     : each task runs in a fresh specialist sub-agent; failures are
                retried with the error, then trigger replanning
* update      : hypothesis credences updated by Bayes from stated likelihood
                ratios, tied to the ledger (refuted hypotheses collapse)
* verify      : fresh-context verifier tries to refute the key claims
* decide      : stop on confidence, diminishing returns, or iteration cap
* deliver     : report generated from the ledger + lessons stored in memory

State is checkpointed after every phase and every task, so a run survives
crashes and can be resumed with `ResearchLoop.resume`.
"""
from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agent import ROLE_TOOLS, AgentFactory, READ_TOOLS, submit_tool
from .epistemics import Ledger, Status
from .genome import Genome
from .governance import AuditLog, Budget, GovernanceError, Governor, KillSwitch
from .llm import Model
from .memory import MemoryStore
from .planning import PlanError, TaskGraph, TaskStatus
from .report import render_report
from .tools import HumanChannel, ToolContext, ToolRegistry, default_registry
from .uncertainty import DiminishingReturns, should_ask_human
from . import verify

S = {"type": "string"}
N = {"type": "number"}
B = {"type": "boolean"}
STRS = {"type": "array", "items": S}
LR_CLIP = (1e-3, 1e3)


@dataclass
class RunState:
    run_id: str
    problem: str
    phase: str = "frame"
    iteration: int = 1
    frame: dict | None = None
    research: dict | None = None
    hypotheses: list[dict] = field(default_factory=list)
    plan: dict | None = None
    replans: int = 0
    failure_context: str = ""
    verification: list[dict] = field(default_factory=list)
    progress: list[float] = field(default_factory=list)
    result: dict | None = None
    human_log: list[dict] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    spent: dict = field(default_factory=lambda: {"steps": 0, "tokens": 0, "cost_usd": 0.0})
    stop_reason: str = ""
    done: bool = False

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RunState":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class ResearchLoop:
    def __init__(self, problem: str, *, model: Model, run_dir: str | Path, workspace: str | Path | None = None,
                 genome: Genome | None = None, governor: Governor | None = None,
                 memory: MemoryStore | None = None, human: HumanChannel | None = None,
                 registry: ToolRegistry | None = None, verifier_model: Model | None = None,
                 max_tasks_per_iteration: int = 40) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.run_dir / "state.json"
        if state_path.exists():
            self.state = RunState.load(state_path)
        else:
            self.state = RunState(run_id=f"run-{uuid.uuid4().hex[:8]}", problem=problem)
        ledger_path = self.run_dir / "ledger.json"
        self.ledger = Ledger.load(ledger_path) if ledger_path.exists() else Ledger()
        genome_path = self.run_dir / "genome.json"
        if genome is None and genome_path.exists():
            genome = Genome.load(genome_path)        # a resumed run keeps its strategy
        self.genome = genome or Genome()
        self.genome.save(genome_path)
        self.governor = governor or Governor(
            budget=Budget(max_steps=400, max_cost_usd=25.0),
            audit=AuditLog(self.run_dir / "audit.jsonl"), kill_switch=KillSwitch(self.run_dir / "STOP"))
        sp = self.state.spent
        self.governor.budget.charge(steps=int(sp["steps"]), tokens=int(sp["tokens"]), cost_usd=float(sp["cost_usd"]))
        self.memory = memory
        self.human = human or HumanChannel()
        self.workspace = Path(workspace or self.run_dir / "workspace")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.registry = registry or default_registry()
        self.model = model
        self.factory = AgentFactory(model, self.registry, subagent_effort=self.genome.subagent_effort,
                                    allow_delegation=self.genome.delegate)
        self.verifier_factory = AgentFactory(verifier_model or model, self.registry,
                                             subagent_effort=self.genome.main_effort, allow_delegation=False)
        self.ctx = ToolContext(self.workspace, self.governor, self.ledger, memory, self.state.run_id,
                               "orchestrator", self.human)
        self.max_tasks_per_iteration = max_tasks_per_iteration

    @classmethod
    def resume(cls, run_dir: str | Path, **kwargs) -> "ResearchLoop":
        state = RunState.load(Path(run_dir) / "state.json")
        return cls(state.problem, run_dir=run_dir, **kwargs)

    # -- infrastructure ------------------------------------------------------
    def _checkpoint(self) -> None:
        b = self.governor.budget
        self.state.spent = {"steps": b.steps, "tokens": b.tokens, "cost_usd": b.cost_usd}
        self.state.save(self.run_dir / "state.json")
        self.ledger.save(self.run_dir / "ledger.json")

    def _log(self, **event) -> None:
        self.state.log.append({"ts": time.time(), "iteration": self.state.iteration, **event})

    def _agent(self, role: str, submit, *, max_steps: int = 25, tool_names=None, system_extra: str = ""):
        note = self.genome.phase_note(self.state.phase)
        extra = "\n".join(x for x in (system_extra, note) if x)
        return self.factory.create(role, self.ctx, submit=submit, max_steps=max_steps,
                                   effort=self.genome.main_effort, system_extra=extra, tool_names=tool_names)

    def _brief(self, *, ledger_items: int = 25) -> str:
        st = self.state
        parts = [f"# Problem\n{st.problem}"]
        if st.frame:
            f = st.frame
            parts.append("# Frame\nObjective: " + f.get("objective", "") + "\nSuccess criteria:\n"
                         + "\n".join(f"- {c}" for c in f.get("success_criteria", []))
                         + ("\nAnswers from the human:\n" + "\n".join(
                             f"- Q: {a['question']} A: {a['answer']}" for a in f.get("answers", []))
                            if f.get("answers") else ""))
        if st.hypotheses:
            parts.append("# Competing hypotheses (posterior)\n" + "\n".join(
                f"- {h['id']} [{h['claim_id']}] p={h['posterior']:.2f}: {h['statement']}" for h in st.hypotheses))
        if st.plan:
            parts.append("# Plan\n" + TaskGraph.from_dict(st.plan).summary(20))
        parts.append("# Evidence ledger\n" + self.ledger.summary(ledger_items))
        if self.memory is not None:
            parts.append("# Relevant long-term memory\n" + self.memory.context_block(st.problem))
        parts.append(f"# Status\niteration {st.iteration}/{self.genome.max_iterations}; "
                     f"budget used {self.governor.budget.snapshot()}")
        return "\n\n".join(parts)

    # -- main loop ---------------------------------------------------------------
    def run(self) -> Path:
        while not self.state.done:
            phase = self.state.phase
            try:
                getattr(self, f"_phase_{phase}")()
            except GovernanceError as e:
                self.state.stop_reason = f"halted: {e}"
                self._log(phase=phase, event="halted", reason=str(e))
                if phase == "deliver":
                    self._finalize(None)
                else:
                    self.state.phase = "deliver"
            self._checkpoint()
        return self.run_dir / "report.md"

    # -- phases --------------------------------------------------------------------
    def _phase_frame(self) -> None:
        def validate(a):
            errs = []
            for k in ("objective",):
                if not str(a.get(k, "")).strip():
                    errs.append(f"{k} must be non-empty")
            for k in ("success_criteria", "unknowns"):
                if not a.get(k):
                    errs.append(f"{k} must list at least one item")
            return errs

        tool = submit_tool("submit_frame", "Submit the problem frame.", {
            "objective": S, "success_criteria": STRS, "unknowns": STRS, "assumptions": STRS,
            "constraints": STRS, "risks": STRS,
            "clarifying_questions": {"type": "array", "items": {"type": "object", "properties": {
                "question": S, "impact": N, "confidence_without_answer": N},
                "required": ["question", "impact", "confidence_without_answer"]}}},
            ["objective", "success_criteria", "unknowns"], validate)
        res = self._agent("planner", tool, max_steps=12).run(
            self._brief() + "\n\nFrame this problem: state the real objective, measurable success "
            "criteria, what we need to know but don't (unknowns), assumptions, constraints and risks. "
            "List clarifying questions only if the answer would change the approach; estimate the impact "
            "of being wrong (0-1) and your confidence without an answer (0-1).")
        self._log(phase="frame", stop=res.stop_reason, steps=res.steps)
        if res.submitted is None:
            self.state.stop_reason = f"framing produced no valid frame ({res.stop_reason})"
            self.state.phase = "deliver"
            return
        frame = dict(res.submitted)
        frame["answers"] = []
        for q in frame.get("clarifying_questions", []):
            d = should_ask_human(float(q["confidence_without_answer"]), float(q["impact"]),
                                 human_available=self.human.available)
            if d.ask:
                answer = self.human.ask(q["question"])
                ev = self.ledger.add_evidence("source", f"human answer: {q['question'][:150]}",
                                              {"question": q["question"], "answer": answer})
                frame["answers"].append({"question": q["question"], "answer": answer, "evidence": ev.id})
                self.state.human_log.append({"question": q["question"], "answer": answer, "why": d.reason})
            else:
                self.ledger.assert_claim(f"Assumed without asking: {q['question']} ({d.reason})",
                                         Status.SPECULATION, float(q["confidence_without_answer"]), author="frame")
        for a in frame.get("assumptions", []):
            self.ledger.assert_claim(f"Assumption: {a}", Status.SPECULATION, 0.5, author="frame")
        self.state.frame = frame
        if self.memory is not None:
            self.memory.record_episode(self.state.run_id, "frame", json.dumps(frame)[:4000])
        self.state.phase = "research" if self.genome.research_depth > 0 else "hypothesize"

    def _phase_research(self) -> None:
        tool = submit_tool("submit_research", "Submit what you found.", {
            "findings": {"type": "array", "items": {"type": "object", "properties": {
                "statement": S, "claim_ids": STRS}, "required": ["statement"]}},
            "open_questions": STRS, "conflicts": STRS}, ["findings"])
        res = self._agent("researcher", tool, max_steps=12 * self.genome.research_depth).run(
            self._brief() + "\n\nResearch what is already known that bears on the unknowns: documentation, "
            "literature, prior work, data. Record each important claim with assert_claim and honest status; "
            "cite sources with cite_source. Where sources disagree, record both claims and explain which is "
            "better supported. Identify what remains unknown.")
        self._log(phase="research", stop=res.stop_reason, steps=res.steps)
        self.state.research = res.submitted or {"findings": [], "note": f"no submission ({res.stop_reason})"}
        self.state.phase = "hypothesize"

    def _phase_hypothesize(self) -> None:
        k = self.genome.min_hypotheses

        def validate(a):
            hs = a.get("hypotheses", [])
            errs = []
            if len(hs) < k:
                errs.append(f"need at least {k} competing hypotheses, got {len(hs)}")
            ids = [h.get("id") for h in hs]
            if len(set(ids)) != len(ids):
                errs.append("hypothesis ids must be unique")
            for h in hs:
                if not h.get("statement") or not h.get("falsification_test"):
                    errs.append(f"hypothesis {h.get('id')} needs a statement and a falsification test")
                if not 0 < float(h.get("prior", 0)) <= 1:
                    errs.append(f"hypothesis {h.get('id')} prior must be in (0, 1]")
            return errs

        tool = submit_tool("submit_hypotheses", "Submit competing hypotheses.", {
            "hypotheses": {"type": "array", "items": {"type": "object", "properties": {
                "id": S, "statement": S, "predictions": STRS, "falsification_test": S, "prior": N},
                "required": ["id", "statement", "predictions", "falsification_test", "prior"]}}},
            ["hypotheses"], validate)
        revise = ("\n\nThis is iteration %d. Revise the hypotheses in light of the evidence: keep (same id), "
                  "drop, or add. Consider hypotheses nobody has proposed yet." % self.state.iteration
                  if self.state.iteration > 1 else "")
        res = self._agent("scientist", tool, max_steps=10).run(
            self._brief() + f"\n\nPropose at least {k} genuinely competing hypotheses (answers, explanations "
            "or solution approaches). For each: concrete predictions that differ from the others, a test that "
            "could falsify it, and a prior probability." + revise)
        self._log(phase="hypothesize", stop=res.stop_reason, steps=res.steps)
        if res.submitted is None:
            self.state.stop_reason = f"no valid hypotheses ({res.stop_reason})"
            self.state.phase = "deliver"
            return
        old = {h["id"]: h for h in self.state.hypotheses}
        new = []
        for h in res.submitted["hypotheses"]:
            prev = old.get(h["id"])
            prior = prev["posterior"] if prev else float(h["prior"])
            claim_id = prev["claim_id"] if prev else self.ledger.assert_claim(
                f"Hypothesis {h['id']}: {h['statement']}", Status.SPECULATION, prior, author="hypothesize").id
            new.append({**h, "prior": prior, "posterior": prior, "claim_id": claim_id})
        total = sum(h["prior"] for h in new)
        for h in new:
            h["prior"] = h["posterior"] = h["prior"] / total
        self.state.hypotheses = new
        self.state.phase = "plan"

    def _phase_plan(self) -> None:
        def validate(a):
            tasks = a.get("tasks", [])
            if not tasks:
                return ["the plan needs at least one task"]
            if len(tasks) > 200:
                return ["at most 200 tasks per iteration; group work into larger tasks"]
            bad = [t.get("role") for t in tasks if t.get("role") not in ROLE_TOOLS]
            if bad:
                return [f"unknown roles {bad}; choose from {sorted(ROLE_TOOLS)}"]
            try:
                TaskGraph.from_spec(tasks)
            except PlanError as e:
                return [str(e)]
            return []

        tool = submit_tool("submit_plan", "Submit the task graph.", {
            "tasks": {"type": "array", "items": {"type": "object", "properties": {
                "id": S, "title": S, "description": S, "deps": STRS, "acceptance": S,
                "role": {"type": "string", "enum": sorted(ROLE_TOOLS)}},
                "required": ["id", "title", "description", "deps", "acceptance", "role"]}}},
            ["tasks"], validate)
        failure = f"\n\nThe previous plan failed:\n{self.state.failure_context}\nPlan a different approach." \
            if self.state.failure_context else ""
        res = self._agent("planner", tool, max_steps=10).run(
            self._brief() + "\n\nWrite an executable plan for this iteration: tasks that build what is needed "
            "and run the experiments/checks that discriminate between the hypotheses. Each task needs a role, "
            "dependencies and an acceptance criterion that can be checked." + failure)
        self._log(phase="plan", stop=res.stop_reason, steps=res.steps)
        if res.submitted is None:
            self.state.stop_reason = f"no valid plan ({res.stop_reason})"
            self.state.phase = "deliver"
            return
        self.state.plan = TaskGraph.from_spec(res.submitted["tasks"]).to_dict()
        self.state.failure_context = ""
        self.state.phase = "execute"

    def _run_task(self, task, results: dict[str, str]) -> tuple[bool, str, list | None]:
        ledger = self.ledger

        def validate(a):
            errs = [f"unknown evidence id {e!r}" for e in a.get("evidence_ids", []) if e not in ledger.evidence]
            errs += [f"unknown claim id {c!r}" for c in a.get("claim_ids", []) if c not in ledger.claims]
            return errs

        subtask = {"type": "object", "properties": {"id": S, "title": S, "description": S, "deps": STRS,
                                                     "acceptance": S, "role": {"type": "string",
                                                                               "enum": sorted(ROLE_TOOLS)}},
                   "required": ["id", "title", "description", "acceptance", "role"]}
        tool = submit_tool("submit_task_result", "Submit the outcome of your task. If the task is too large "
                           "to finish, set success=false and propose subtasks that decompose it.", {
                               "success": B, "summary": S, "evidence_ids": STRS, "claim_ids": STRS,
                               "subtasks": {"type": "array", "items": subtask}},
                           ["success", "summary"], validate)
        ctx = self.ctx.child(f"task:{task.id}/{task.role}", self.governor.child(0.25))
        agent = self.factory.create(task.role, ctx, submit=tool, max_steps=30)
        deps = "\n".join(f"- {d}: {results.get(d, '')[:600]}" for d in task.deps)
        retry = ("\n\nPrevious attempts failed:\n" + "\n".join(f"- {e}" for e in task.errors)
                 + "\nUse a different approach.") if task.errors else ""
        res = agent.run(self._brief(ledger_items=15)
                        + f"\n\n# Your task: {task.id} - {task.title}\n{task.description}\n"
                          f"Acceptance criterion: {task.acceptance}\nResults of dependencies:\n{deps or '(none)'}"
                        + retry)
        if res.submitted is None:
            return False, f"no result submitted ({res.stop_reason})", None
        sub = res.submitted
        return bool(sub["success"]), sub["summary"], (sub.get("subtasks") or None) if not sub["success"] else None

    def _phase_execute(self) -> None:
        graph = TaskGraph.from_dict(self.state.plan)
        for t in graph.tasks.values():          # tasks interrupted by a crash restart cleanly
            if t.status is TaskStatus.RUNNING:
                t.status = TaskStatus.PENDING
        done = 0
        while done < self.max_tasks_per_iteration:
            self.governor.check_running()
            ready = graph.ready()
            if not ready:
                break
            batch = ready[:4] if self.genome.delegate else ready[:1]
            results = {t.id: t.result for t in graph.tasks.values() if t.result}
            for t in batch:
                graph.start(t.id)
            if len(batch) > 1:
                with ThreadPoolExecutor(max_workers=len(batch)) as ex:
                    outcomes = list(ex.map(lambda t: self._run_task(t, results), batch))
            else:
                outcomes = [self._run_task(batch[0], results)]
            for t, (ok, summary, subtasks) in zip(batch, outcomes):
                if ok:
                    graph.complete(t.id, summary)
                elif subtasks and t.parent is None:
                    # hierarchical refinement: replace the task by its sub-plan
                    graph.tasks[t.id].status = TaskStatus.PENDING
                    try:
                        graph.refine(t.id, subtasks)
                        summary = f"decomposed into {len(subtasks)} subtasks: {summary}"
                    except PlanError as e:
                        graph.tasks[t.id].status = TaskStatus.RUNNING
                        graph.fail(t.id, f"{summary} (decomposition rejected: {e})")
                else:
                    graph.fail(t.id, summary)
                self._log(phase="execute", task=t.id, ok=ok, summary=summary[:300])
            done += len(batch)
            self.state.plan = graph.to_dict()
            self._checkpoint()
        failed = [t for t in graph.tasks.values() if t.status is TaskStatus.FAILED]
        if failed and self.state.replans < 2:
            self.state.replans += 1
            self.state.failure_context = "\n".join(f"- {t.id} ({t.title}): {t.errors[-1]}" for t in failed)
            self.state.phase = "plan"
        else:
            self.state.phase = "update"

    def _phase_update(self) -> None:
        ids = {h["id"] for h in self.state.hypotheses}

        def validate(a):
            errs = [f"unknown hypothesis {u.get('hypothesis_id')!r}" for u in a.get("updates", [])
                    if u.get("hypothesis_id") not in ids]
            errs += [f"likelihood ratio must be > 0 for {u.get('hypothesis_id')}" for u in a.get("updates", [])
                     if not float(u.get("likelihood_ratio", 0)) > 0]
            return errs

        tool = submit_tool("submit_update", "Submit likelihood ratios for the new evidence.", {
            "updates": {"type": "array", "items": {"type": "object", "properties": {
                "hypothesis_id": S, "likelihood_ratio": N, "justification": S, "evidence_ids": STRS},
                "required": ["hypothesis_id", "likelihood_ratio", "justification"]}}},
            ["updates"], validate)
        res = self._agent("scientist", tool, max_steps=8, tool_names=READ_TOOLS).run(
            self._brief() + "\n\nFor each hypothesis, how much more (or less) likely is the evidence gathered "
            "this iteration if the hypothesis is true than if it is false? Give a likelihood ratio (1 = "
            "uninformative), a justification and the evidence ids. Be conservative: weak evidence deserves "
            "ratios near 1.")
        self._log(phase="update", stop=res.stop_reason, steps=res.steps)
        lrs = {u["hypothesis_id"]: min(LR_CLIP[1], max(LR_CLIP[0], float(u["likelihood_ratio"])))
               for u in (res.submitted or {}).get("updates", [])}
        self._apply_posterior(lrs)
        self.state.phase = "verify"

    def _apply_posterior(self, lrs: dict[str, float]) -> None:
        hs = self.state.hypotheses
        for h in hs:
            lr = lrs.get(h["id"], 1.0)
            if self.ledger.get(h["claim_id"]).refuted:
                lr = min(lr, 0.01)
            h["posterior"] = h["posterior"] * lr
        total = sum(h["posterior"] for h in hs) or 1.0
        for h in hs:
            h["posterior"] /= total
            self.ledger.update_confidence(h["claim_id"], h["posterior"], "posterior after update")
        hs.sort(key=lambda h: -h["posterior"])

    def _phase_verify(self) -> None:
        mode = self.genome.verification
        if mode == "none" or not self.state.hypotheses:
            self.state.phase = "decide"
            return
        leader = self.state.hypotheses[0]
        candidates = sorted((c for c in self.ledger.claims.values()
                             if c.status is Status.INFERENCE and not c.refuted and c.confidence >= 0.5
                             and not any(f.outcome == "survived" for f in c.falsifications)),
                            key=lambda c: -c.confidence)[:6]
        claims = [self.ledger.get(leader["claim_id"]), *candidates]
        ctx = self.ctx.child("verifier", self.governor.child(0.3))
        summary, _ = verify.review(self.verifier_factory, ctx, claims, mode=mode)
        self.state.verification.append({"iteration": self.state.iteration, "mode": mode, **summary})
        self._log(phase="verify", **{k: v for k, v in summary.items() if k != "error"})
        if leader["claim_id"] in summary.get("refuted", []):
            self._apply_posterior({})
        self.state.phase = "decide"

    def _phase_decide(self) -> None:
        st, g = self.state, self.genome
        leader = st.hypotheses[0] if st.hypotheses else None
        p = leader["posterior"] if leader else 0.0
        last = st.verification[-1] if st.verification and st.verification[-1]["iteration"] == st.iteration else None
        checked = len(last["supported"]) + len(last["refuted"]) + len(last["insufficient"]) if last else 0
        v = (len(last["supported"]) / checked) if checked else (1.0 if g.verification == "none" else 0.0)
        signal = p * (0.5 + 0.5 * v)
        st.progress.append(signal)
        leader_ok = leader is not None and (last is None or leader["claim_id"] in last["supported"])
        stall = DiminishingReturns(patience=2, min_delta=0.02, history=list(st.progress))
        if st.iteration >= g.max_iterations:
            reason = "iteration limit reached"
        elif p >= 0.9 and leader_ok and (g.verification == "none" or last is not None):
            reason = "leading hypothesis is confident and survived verification"
        elif stall.should_stop():
            reason = "diminishing returns"
        else:
            reason = ""
        self._log(phase="decide", signal=signal, leader_p=p, reason=reason or "continue")
        if reason:
            st.stop_reason = st.stop_reason or reason
            st.phase = "deliver"
        else:
            st.iteration += 1
            st.phase = "hypothesize"

    def _phase_deliver(self) -> None:
        ledger = self.ledger

        def validate(a):
            errs = [f"unknown claim id {c!r}" for c in a.get("key_claim_ids", []) if c not in ledger.claims]
            if not 0 <= float(a.get("confidence", -1)) <= 1:
                errs.append("confidence must be in [0, 1]")
            return errs

        tool = submit_tool("submit_result", "Submit the final result.", {
            "answer": S, "confidence": N, "key_claim_ids": STRS, "limitations": STRS,
            "reproduction_steps": STRS, "next_steps": STRS},
            ["answer", "confidence", "key_claim_ids", "limitations", "reproduction_steps"], validate)
        submitted = None
        if not self.state.stop_reason.startswith("halted"):
            res = self._agent("generalist", tool, max_steps=8, tool_names=READ_TOOLS).run(
                self._brief(ledger_items=60) + "\n\nDeliver the strongest result the evidence supports. Base the "
                "answer only on the ledger; cite the key claim ids it rests on; state limitations honestly; give "
                "steps another person could follow to reproduce it. Your stated confidence will be capped by "
                "the confidence of the claims you cite.")
            self._log(phase="deliver", stop=res.stop_reason, steps=res.steps)
            submitted = res.submitted
        self._finalize(submitted)

    # -- delivery ----------------------------------------------------------------------
    def _finalize(self, submitted: dict | None) -> None:
        st = self.state
        limitations = []
        if submitted:
            key = submitted.get("key_claim_ids", [])
            if key:
                final = self.ledger.assert_claim(f"Final answer: {submitted['answer']}", Status.INFERENCE,
                                                 float(submitted["confidence"]), depends_on=key, author="deliver")
            else:
                final = self.ledger.assert_claim(f"Final answer: {submitted['answer']}", Status.SPECULATION,
                                                 float(submitted["confidence"]), author="deliver")
            answer, credence = submitted["answer"], final.confidence
            limitations += submitted.get("limitations", [])
            reproduction = {f"step {i + 1}": s for i, s in enumerate(submitted.get("reproduction_steps", []))}
        elif st.hypotheses:
            leader = st.hypotheses[0]
            answer = f"(no final synthesis) leading hypothesis {leader['id']}: {leader['statement']}"
            credence = leader["posterior"]
            reproduction = {}
        else:
            answer, credence, reproduction = "No result was produced.", None, {}
        if st.stop_reason:
            limitations.append(f"run ended because: {st.stop_reason}")
        if self.genome.verification == "none":
            limitations.append("conclusions were not independently verified (verification disabled)")
        reproduction.update({"run_id": st.run_id, "genome": self.genome.fingerprint(),
                             "model": getattr(self.model, "name", "?"), "run_dir": str(self.run_dir),
                             "resume": f"quanta resume {self.run_dir}"})
        extra = []
        if st.frame:
            extra.append(("Problem frame", "\n".join(
                [f"- **Objective:** {st.frame.get('objective', '')}"]
                + [f"- **Success criterion:** {c}" for c in st.frame.get("success_criteria", [])]
                + [f"- **Unknown:** {u}" for u in st.frame.get("unknowns", [])])))
        if st.hypotheses:
            extra.append(("Competing hypotheses", "\n".join(
                f"- {h['id']}: prior {h['prior']:.2f} -> posterior {h['posterior']:.2f} — {h['statement']}"
                for h in st.hypotheses)))
        if st.plan:
            extra.append(("Plan status", "```\n" + TaskGraph.from_dict(st.plan).summary(60) + "\n```"))
        if st.verification:
            extra.append(("Verification", "\n".join(
                f"- iteration {v['iteration']} ({v['mode']}): supported {v['supported']}, promoted to fact "
                f"{v['promoted']}, refuted {v['refuted']}, insufficient {v['insufficient']}"
                for v in st.verification)))
        if st.human_log:
            extra.append(("Human input", "\n".join(f"- Q: {h['question']} — A: {h['answer']}" for h in st.human_log)))
        if submitted and submitted.get("next_steps"):
            extra.append(("Suggested next steps", "\n".join(f"- {s}" for s in submitted["next_steps"])))
        ok, bad = self.governor.audit.verify()
        report = render_report(
            title="Research report", problem=st.problem, answer=answer, confidence=credence, ledger=self.ledger,
            limitations=limitations, reproduction=reproduction, resources=self.governor.budget.snapshot(),
            audit={"entries": len(self.governor.audit.entries), "head": self.governor.audit.head,
                   "chain_intact": ok}, extra_sections=extra)
        (self.run_dir / "report.md").write_text(report, encoding="utf-8")
        st.result = {"answer": answer, "credence": credence}
        st.done = True
        st.phase = "done"
        self._consolidate_memory(answer, credence)

    def _consolidate_memory(self, answer: str, credence: float | None) -> None:
        if self.memory is None:
            return
        st = self.state
        self.memory.record_episode(st.run_id, "outcome", f"problem: {st.problem[:500]}\nanswer: {answer[:1000]}",
                                   {"credence": credence, "stop_reason": st.stop_reason,
                                    "iterations": st.iteration})
        subject = st.problem[:80]
        for c in self.ledger.by_status(Status.FACT):
            self.memory.add_knowledge(subject, c.statement, status="fact", confidence=c.confidence, verified=True,
                                      run_id=st.run_id, meta={"claim": c.id})
        if st.plan:
            titles = [t["title"] for t in st.plan["tasks"] if t["status"] == "done"]
            if titles:
                self.memory.add_procedure(f"approach: {subject}", " -> ".join(titles)[:2000], run_id=st.run_id,
                                          meta={"credence": credence})
