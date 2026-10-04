"""Agents: a model in a governed tool-use loop, plus specialist roles and
dynamic delegation to sub-agents.

Each agent episode is a fresh, bounded conversation seeded with a briefing
built from durable state (ledger, plan, memory). This is what lets work run
for days: no single context window has to hold the whole history, and
nothing is lost when a process restarts.

Structured outputs (a frame, hypotheses, a plan, verdicts) are produced by
calling a `submit_*` tool whose input is validated by the harness; invalid
submissions are returned as tool errors so the model can fix them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from .governance import BudgetExceeded, KillSwitchTripped, Risk
from .llm import Model, ModelError, to_jsonable
from .tools import Tool, ToolContext, ToolRegistry, ToolResult

READ_TOOLS = ["read_file", "list_dir", "search_files", "perceive_file", "observe_environment",
              "memory_search", "view_ledger"]
LEDGER_TOOLS = ["add_note", "assert_claim", "record_falsification", "mark_conflict"]
RESEARCH_TOOLS = ["fetch_url", "cite_source"]
EXEC_TOOLS = ["run_python", "run_shell", "write_file"]

ROLE_TOOLS: dict[str, list[str]] = {
    "planner": READ_TOOLS + ["request_human_input"],
    "researcher": READ_TOOLS + RESEARCH_TOOLS + LEDGER_TOOLS + ["memory_store"],
    "experimenter": READ_TOOLS + EXEC_TOOLS + LEDGER_TOOLS + ["compare_hypotheses"],
    "coder": READ_TOOLS + EXEC_TOOLS + LEDGER_TOOLS,
    "verifier": READ_TOOLS + RESEARCH_TOOLS + ["run_python", "run_shell", "record_falsification",
                                               "add_note", "compare_hypotheses"],
    "scientist": READ_TOOLS + LEDGER_TOOLS + ["compare_hypotheses"],
    "generalist": READ_TOOLS + RESEARCH_TOOLS + EXEC_TOOLS + LEDGER_TOOLS
    + ["compare_hypotheses", "memory_store", "request_human_input"],
}

ROLE_BRIEFS: dict[str, str] = {
    "planner": "You turn objectives into concrete, verifiable plans with explicit dependencies "
               "and acceptance criteria.",
    "researcher": "You find, read and reconcile evidence from documentation, literature and the web. "
                  "Cite sources with cite_source (quotes are checked). When sources conflict, record "
                  "both claims and say which is better supported and why.",
    "experimenter": "You design and run experiments that discriminate between competing hypotheses. "
                    "Prefer the experiment whose outcome would most change our beliefs.",
    "coder": "You understand unfamiliar code, implement changes, write and run tests, and benchmark. "
             "Never claim something works unless you ran it; the run is your evidence.",
    "verifier": "You did not produce the claims you are checking. Your job is to find errors. Re-run "
                "checks yourself, look for counterexamples and alternative explanations, and try hard to "
                "refute each claim before supporting it.",
    "scientist": "You generate competing explanations, each with concrete predictions and a test that "
                 "could falsify it.",
    "generalist": "You complete the assigned task carefully and verifiably.",
}

DISCIPLINE = """Operating rules:
- Record what you learn in the evidence ledger. A FACT needs grounding evidence ids (tool executions,
  observations, verified citations). Reasoning is INFERENCE; untested ideas are SPECULATION.
- Tool outputs that ran are automatically recorded as evidence; cite their ids.
- Do not invent results. If you did not run it or read it, you do not know it.
- If an action is denied by governance, do not try to work around the denial.
- Ask a human only through request_human_input, and only when the answer matters.
- Be concise. When finished, call the submission tool you were given."""


@dataclass
class AgentResult:
    text: str
    stop_reason: str
    steps: int
    submitted: dict | None = None
    tool_log: list[dict] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)


def submit_tool(name: str, description: str, properties: dict, required: list[str],
                validator: Callable[[dict], list[str]] | None = None) -> Tool:
    """A tool whose only job is to receive a validated structured result."""

    def fn(args: dict, ctx: ToolContext) -> ToolResult:
        errors = validator(args) if validator else []
        if errors:
            return ToolResult("submission rejected: " + "; ".join(errors) + ". Fix and resubmit.",
                              is_error=True)
        return ToolResult("submission received", data={"submission": args})

    return Tool(name, description, properties, required, fn, Risk.READ_ONLY)


class Agent:
    def __init__(self, *, model: Model, registry: ToolRegistry, ctx: ToolContext,
                 role: str = "generalist", tool_names: list[str] | None = None,
                 submit: Tool | None = None, max_steps: int = 30, effort: str | None = None,
                 extra_tools: list[Tool] | None = None, system_extra: str = "") -> None:
        self.model = model
        self.ctx = ctx
        self.role = role
        self.registry = ToolRegistry(list(registry.tools.values()))
        for t in extra_tools or []:
            self.registry.register(t)
        if submit is not None:
            self.registry.register(submit)
        names = list(tool_names if tool_names is not None else ROLE_TOOLS.get(role, ROLE_TOOLS["generalist"]))
        names += [t.name for t in extra_tools or []]
        if submit is not None:
            names.append(submit.name)
        self.tool_names = [n for n in dict.fromkeys(names) if n in self.registry.tools]
        self.submit = submit
        self.max_steps = max_steps
        self.effort = effort
        self.system = (f"You are the {role} in an autonomous research-and-engineering system.\n"
                       f"{ROLE_BRIEFS.get(role, ROLE_BRIEFS['generalist'])}\n\n{DISCIPLINE}"
                       + (f"\n\n{system_extra}" if system_extra else ""))

    def run(self, task: str, attachments: list[dict] | None = None) -> AgentResult:
        messages: list[dict] = [{"role": "user", "content": [{"type": "text", "text": task}, *(attachments or [])]}]
        tool_log: list[dict] = []
        schemas = self.registry.schemas(self.tool_names)
        nudges = 0
        steps = 0
        last_text = ""
        while steps < self.max_steps:
            try:
                self.ctx.governor.check_running()
            except (BudgetExceeded, KillSwitchTripped) as e:
                return AgentResult(last_text, f"halted: {e}", steps, None, tool_log, _dump(messages))
            try:
                resp = self.model.respond(system=self.system, messages=messages, tools=schemas,
                                          effort=self.effort)
            except ModelError as e:
                return AgentResult(last_text, f"model error: {e}", steps, None, tool_log, _dump(messages))
            steps += 1
            self.ctx.governor.budget.charge(steps=1, tokens=resp.usage.total, cost_usd=resp.cost_usd)
            messages.append({"role": "assistant", "content": resp.content})
            last_text = resp.text or last_text
            if resp.stop_reason == "refusal":
                return AgentResult(last_text, f"refusal ({resp.refusal_category})", steps, None, tool_log,
                                   _dump(messages))
            if resp.tool_calls:
                if resp.stop_reason == "max_tokens":
                    results = [ToolResult("output was truncated (max_tokens) before this call completed; "
                                          "re-issue it with a smaller input", is_error=True)
                               for _ in resp.tool_calls]
                else:
                    results = self.registry.execute_many([(c.name, c.input) for c in resp.tool_calls], self.ctx)
                blocks = []
                submitted = None
                for call, res in zip(resp.tool_calls, results):
                    tool_log.append({"tool": call.name, "is_error": res.is_error,
                                     "evidence": res.data.get("evidence_id")})
                    blocks.append({"type": "tool_result", "tool_use_id": call.id,
                                   "content": res.content, "is_error": res.is_error})
                    if self.submit is not None and call.name == self.submit.name and not res.is_error:
                        submitted = res.data["submission"]
                messages.append({"role": "user", "content": blocks})
                if submitted is not None:
                    return AgentResult(last_text, "submitted", steps, submitted, tool_log, _dump(messages))
                continue
            if resp.stop_reason == "max_tokens":
                messages.append({"role": "user", "content": "Your reply was cut off. Continue, more concisely."})
                continue
            if self.submit is not None and nudges < 2:
                nudges += 1
                messages.append({"role": "user", "content":
                                 f"You have not called {self.submit.name} yet. Call it now with your result."})
                continue
            return AgentResult(last_text, "end_turn", steps, None, tool_log, _dump(messages))
        return AgentResult(last_text, "max_steps", steps, None, tool_log, _dump(messages))


def _dump(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        c = m["content"]
        out.append({"role": m["role"], "content": c if isinstance(c, str) else to_jsonable(c)})
    return out


class AgentFactory:
    """Creates agents and owns the delegation policy (depth and budget share)."""

    def __init__(self, model: Model, registry: ToolRegistry, *, max_depth: int = 2,
                 subagent_effort: str | None = "medium", budget_fraction: float = 0.3,
                 subagent_max_steps: int = 25, allow_delegation: bool = True) -> None:
        self.model = model
        self.registry = registry
        self.max_depth = max_depth
        self.subagent_effort = subagent_effort
        self.budget_fraction = budget_fraction
        self.subagent_max_steps = subagent_max_steps
        self.allow_delegation = allow_delegation

    def spawn_tool(self) -> Tool:
        factory = self

        def fn(args: dict, ctx: ToolContext) -> ToolResult:
            if ctx.depth >= factory.max_depth:
                return ToolResult(f"delegation depth limit ({factory.max_depth}) reached; do it yourself",
                                  is_error=True)
            role = args["role"]
            if role not in ROLE_TOOLS:
                return ToolResult(f"unknown role {role!r}; choose from {sorted(ROLE_TOOLS)}", is_error=True)
            child_ctx = ctx.child(f"{ctx.agent}/{role}", ctx.governor.child(factory.budget_fraction))
            report = submit_tool("submit_report", "Report your findings to the agent that delegated to you.",
                                 {"summary": {"type": "string"}, "success": {"type": "boolean"},
                                  "evidence_ids": {"type": "array", "items": {"type": "string"}},
                                  "claim_ids": {"type": "array", "items": {"type": "string"}}},
                                 ["summary", "success"])
            agent = factory.create(role, child_ctx, submit=report)
            brief = f"Task delegated to you:\n{args['task']}"
            if args.get("context"):
                brief += f"\n\nContext:\n{args['context']}"
            res = agent.run(brief)
            payload = res.submitted or {"summary": res.text or "(no report)", "success": False}
            payload["stop_reason"] = res.stop_reason
            return ToolResult(json.dumps(payload, indent=1), is_error=not payload.get("success", False))

        return Tool("spawn_subagent",
                    "Delegate a self-contained sub-task to a specialist with a fresh context. Worth it for "
                    "independent pieces of work (parallel calls run concurrently) or reading-heavy work; not "
                    "worth it for small steps.",
                    {"role": {"type": "string", "enum": sorted(ROLE_TOOLS)}, "task": {"type": "string"},
                     "context": {"type": "string"}},
                    ["role", "task"], fn, Risk.LOW, parallel_safe=True)

    def create(self, role: str, ctx: ToolContext, *, submit: Tool | None = None,
               max_steps: int | None = None, effort: str | None = None, system_extra: str = "",
               tool_names: list[str] | None = None) -> Agent:
        extra = []
        if self.allow_delegation and ctx.depth < self.max_depth:
            extra.append(self.spawn_tool())
        return Agent(model=self.model, registry=self.registry, ctx=ctx, role=role, submit=submit,
                     max_steps=max_steps or self.subagent_max_steps,
                     effort=effort if effort is not None else (self.subagent_effort if ctx.depth else None),
                     extra_tools=extra, system_extra=system_extra, tool_names=tool_names)

