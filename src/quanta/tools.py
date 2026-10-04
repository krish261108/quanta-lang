"""Tools the agents act through, all funnelled through one governed chokepoint.

`ToolRegistry.execute` validates arguments, classifies risk, asks the
`Governor` for authorization (HIGH/CRITICAL need a human or a
pre-authorization), runs the tool, and records executions and observations
as *grounding evidence* in the ledger. That last point is the
anti-hallucination backbone: agents cannot create observation/execution
evidence themselves - only the harness does, when something actually ran.
Likewise agents may store memories but cannot mark them verified.
"""
from __future__ import annotations

import fnmatch
import hashlib
import html
import json
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .epistemics import Ledger, LedgerError
from .governance import Action, Governor, Risk
from .memory import MemoryStore
from .perception import observe_environment, perceive
from .uncertainty import should_ask_human

MAX_OUTPUT_CHARS = 20_000


@dataclass
class ToolResult:
    content: str | list[dict]
    is_error: bool = False
    data: dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(b.get("text", f"[{b.get('type')}]") for b in self.content)


class HumanChannel:
    """How agents reach a person. `available=False` means fully autonomous."""

    def __init__(self, ask_fn: Callable[[str], str] | None = None) -> None:
        self.ask_fn = ask_fn
        self.log: list[tuple[str, str]] = []

    @property
    def available(self) -> bool:
        return self.ask_fn is not None

    def ask(self, question: str) -> str:
        if self.ask_fn is None:
            raise RuntimeError("no human available")
        answer = self.ask_fn(question)
        self.log.append((question, answer))
        return answer


@dataclass
class ToolContext:
    workspace: Path
    governor: Governor
    ledger: Ledger
    memory: MemoryStore | None = None
    run_id: str = "run"
    agent: str = "agent"
    human: HumanChannel = field(default_factory=HumanChannel)
    depth: int = 0
    extras: dict[str, Any] = field(default_factory=dict)
    fetched: dict[str, str] = field(default_factory=dict)   # url -> text, for citation checks

    def child(self, agent: str, governor: Governor) -> "ToolContext":
        return ToolContext(self.workspace, governor, self.ledger, self.memory, self.run_id, agent,
                           self.human, self.depth + 1, self.extras, self.fetched)

    def resolve(self, path: str) -> Path:
        """Resolve a path and refuse anything outside the workspace."""
        p = (self.workspace / path).resolve()
        ws = self.workspace.resolve()
        if p != ws and ws not in p.parents:
            raise PermissionError(f"path {path!r} is outside the workspace")
        return p


@dataclass
class Tool:
    name: str
    description: str
    properties: dict[str, dict]
    required: list[str]
    fn: Callable[[dict, ToolContext], ToolResult | str]
    risk: Risk | Callable[[dict], Risk] = Risk.LOW
    parallel_safe: bool = False
    evidence_kind: str | None = None   # record successful runs as this ledger evidence kind

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": {"type": "object", "properties": self.properties,
                                 "required": self.required, "additionalProperties": False}}

    def classify(self, args: dict) -> Risk:
        return self.risk(args) if callable(self.risk) else self.risk


_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
               "array": list, "object": dict}


def validate_args(tool: Tool, args: dict) -> list[str]:
    if not isinstance(args, dict):
        return ["arguments must be an object"]
    errors = [f"missing required argument {r!r}" for r in tool.required if r not in args]
    for k, v in args.items():
        spec = tool.properties.get(k)
        if spec is None:
            errors.append(f"unexpected argument {k!r}")
            continue
        t = spec.get("type")
        py = _JSON_TYPES.get(t)
        if py is not None and (not isinstance(v, py) or (t in ("integer", "number") and isinstance(v, bool))):
            errors.append(f"argument {k!r} must be of type {t}")
        if "enum" in spec and v not in spec["enum"]:
            errors.append(f"argument {k!r} must be one of {spec['enum']}")
    return errors


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self.tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self.tools)

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        chosen = self.tools.values() if names is None else [self.tools[n] for n in names if n in self.tools]
        return [t.schema() for t in chosen]

    def execute(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(f"unknown tool {name!r}; available: {', '.join(self.tools)}", is_error=True)
        errors = validate_args(tool, args)
        if errors:
            return ToolResult("invalid arguments: " + "; ".join(errors), is_error=True)
        risk = tool.classify(args)
        decision = ctx.governor.authorize(Action(name, args, risk, f"{ctx.agent} calls {name}", ctx.agent))
        if not decision.allowed:
            return ToolResult(f"DENIED by governance ({decision.approver}): {decision.reason}. "
                              "Do not retry the same action; choose a lower-risk alternative or "
                              "request human input.", is_error=True, data={"denied": True})
        try:
            result = tool.fn(args, ctx)
            if isinstance(result, str):
                result = ToolResult(result)
        except Exception as e:  # tool failures are information, not crashes
            result = ToolResult(f"{type(e).__name__}: {e}", is_error=True)
        if tool.evidence_kind and not result.data.get("no_evidence"):
            text = result.text()
            ev = ctx.ledger.add_evidence(
                tool.evidence_kind, f"{name}({_short(args)}){' [error]' if result.is_error else ''}",
                {"tool": name, "args": args, "is_error": result.is_error,
                 "output_sha256": hashlib.sha256(text.encode()).hexdigest(), "output_head": text[:600],
                 **{k: v for k, v in result.data.items() if k in ("exit_code", "url")}})
            result.data["evidence_id"] = ev.id
            note = f"\n[recorded as evidence {ev.id}]"
            if isinstance(result.content, str):
                result.content += note
            else:
                result.content = [*result.content, {"type": "text", "text": note}]
        ctx.governor.audit.append("tool_result", tool=name, actor=ctx.agent, is_error=result.is_error,
                                  evidence=result.data.get("evidence_id"))
        return result

    def execute_many(self, calls: list[tuple[str, dict]], ctx: ToolContext) -> list[ToolResult]:
        """Run a batch of calls; concurrently only if every call is parallel-safe."""
        if len(calls) > 1 and all(self.tools.get(n) and self.tools[n].parallel_safe for n, _ in calls):
            with ThreadPoolExecutor(max_workers=min(8, len(calls))) as ex:
                return list(ex.map(lambda c: self.execute(c[0], c[1], ctx), calls))
        return [self.execute(n, a, ctx) for n, a in calls]


def _short(args: dict, n: int = 120) -> str:
    s = json.dumps(args, default=str)
    return s if len(s) <= n else s[: n - 3] + "..."


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n[... {len(text) - limit} characters omitted ...]\n" + text[-half:]


# ---------------------------------------------------------------------------
# Shell risk classification (heuristic; run inside a container/VM regardless)
# ---------------------------------------------------------------------------

_CRITICAL_PATTERNS = [
    r"\brm\s+-[a-zA-Z]*[rR][a-zA-Z]*\s+(-[a-zA-Z]+\s+)*(/|~|\$HOME|\*)(\s|$)", r"\bmkfs", r"\bdd\s+if=",
    r"\b(shutdown|reboot|halt|poweroff)\b", r":\(\)\s*\{", r"\bchmod\s+-R\s+777\s+/",
    r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba|z)?sh\b", r"\bgit\s+push\b.*(--force|-f\b)",
    r"\bkubectl\s+delete\b", r"\bterraform\s+(apply|destroy)\b", r"\bDROP\s+(TABLE|DATABASE)\b",
    r"\bgh\s+repo\s+delete\b",
]
_HIGH_PATTERNS = [
    r"\brm\b", r"\bgit\s+(push|reset\s+--hard|clean|rebase|commit\s+--amend|filter-branch)\b",
    r"\b(curl|wget|ssh|scp|rsync|ftp|telnet|nc)\b",
    r"\b(pip3?|npm|yarn|pnpm|apt(-get)?|brew|cargo|gem|conda)\s+(install|add|remove|uninstall|publish)\b",
    r"\bsudo\b", r"\bdocker\b", r"\bkubectl\b", r"\bmv\b", r"\bch(mod|own)\b", r"\bkill(all)?\b",
    r">\s*/", r"\bsystemctl\b", r"\bcrontab\b", r"\bgh\s+(pr|release|issue)\s+(create|merge|close|edit)\b",
    r"\b(aws|gcloud|az)\s", r"\btwine\b", r"\bsendmail\b",
]
_READ_ONLY_COMMANDS = {"ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "pwd", "echo", "which",
                       "file", "stat", "du", "df", "diff", "tree", "date", "whoami", "uname"}
_READ_ONLY_GIT = {"status", "log", "diff", "show", "branch", "rev-parse", "ls-files", "blame"}


def classify_shell(command: str) -> Risk:
    for pat in _CRITICAL_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return Risk.CRITICAL
    for pat in _HIGH_PATTERNS:
        if re.search(pat, command):
            return Risk.HIGH
    if re.search(r"[;&|`>]|\$\(", command):
        return Risk.LOW
    try:
        words = shlex.split(command)
    except ValueError:
        return Risk.LOW
    if not words:
        return Risk.READ_ONLY
    cmd = words[0]
    if cmd == "find":
        return Risk.LOW if ("-delete" in words or "-exec" in words) else Risk.READ_ONLY
    if cmd in _READ_ONLY_COMMANDS:
        return Risk.READ_ONLY
    if cmd == "git" and len(words) > 1 and words[1] in _READ_ONLY_GIT:
        return Risk.READ_ONLY
    return Risk.LOW


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

def _read_file(args, ctx):
    p = ctx.resolve(args["path"])
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    offset, limit = int(args.get("offset", 0)), int(args.get("limit", 400))
    chunk = lines[offset: offset + limit]
    body = "\n".join(f"{i + offset + 1:6d}\t{l}" for i, l in enumerate(chunk))
    more = f"\n[{len(lines) - offset - len(chunk)} more lines]" if offset + limit < len(lines) else ""
    return _truncate(body) + more


def _write_file(args, ctx):
    p = ctx.resolve(args["path"])
    backup = ""
    if p.exists():
        rel = p.relative_to(ctx.workspace.resolve())
        b = ctx.workspace / ".quanta" / "backups" / f"{time.time_ns()}" / rel
        b.parent.mkdir(parents=True, exist_ok=True)
        b.write_bytes(p.read_bytes())
        backup = f" (previous version backed up to {b.relative_to(ctx.workspace)})"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(args["content"], encoding="utf-8")
    return f"wrote {len(args['content'])} characters to {args['path']}{backup}"


def _list_dir(args, ctx):
    root = ctx.resolve(args.get("path", "."))
    depth = int(args.get("depth", 2))
    out = []
    base_depth = len(root.parts)
    for p in sorted(root.rglob("*")):
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        if len(p.parts) - base_depth > depth:
            continue
        out.append(str(p.relative_to(root)) + ("/" if p.is_dir() else ""))
        if len(out) >= 500:
            out.append("[... truncated]")
            break
    return "\n".join(out) or "(empty)"


def _search_files(args, ctx):
    root = ctx.resolve(args.get("path", "."))
    pattern = re.compile(args["pattern"])
    glob = args.get("glob", "*")
    hits = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or not fnmatch.fnmatch(p.name, glob):
            continue
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"{p.relative_to(root)}:{i}: {line.strip()[:200]}")
                    if len(hits) >= 200:
                        return "\n".join(hits) + "\n[... truncated]"
        except OSError:
            continue
    return "\n".join(hits) or "no matches"


def _run_subprocess(cmd, ctx, timeout, shell=False) -> ToolResult:
    try:
        r = subprocess.run(cmd, cwd=ctx.workspace, capture_output=True, text=True,
                           timeout=timeout, shell=shell)
        out = _truncate(f"exit_code={r.returncode}\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
        return ToolResult(out, is_error=r.returncode != 0, data={"exit_code": r.returncode})
    except subprocess.TimeoutExpired:
        return ToolResult(f"timed out after {timeout}s", is_error=True, data={"exit_code": -1})


def _run_shell(args, ctx):
    return _run_subprocess(args["command"], ctx, int(args.get("timeout", 120)), shell=True)


def _run_python(args, ctx):
    tmp = ctx.workspace / ".quanta" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    script = tmp / f"snippet_{time.time_ns()}.py"
    script.write_text(args["code"], encoding="utf-8")
    return _run_subprocess([sys.executable, str(script)], ctx, int(args.get("timeout", 120)))


_TAG = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.S | re.I)


def _html_to_text(raw: str) -> str:
    return re.sub(r"\n\s*\n+", "\n\n", html.unescape(_TAG.sub(" ", raw))).strip()


def _fetch(url: str, timeout: int = 30) -> str:
    if not re.match(r"^https?://", url):
        raise ValueError("only http(s) URLs are supported")
    req = urllib.request.Request(url, headers={"User-Agent": "quanta-research-agent/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(2_000_000).decode(r.headers.get_content_charset() or "utf-8", errors="replace")
        ctype = r.headers.get("Content-Type", "")
    return _html_to_text(raw) if "html" in ctype else raw


def _fetch_url(args, ctx):
    text = _fetch(args["url"])
    ctx.fetched[args["url"]] = text
    return ToolResult(_truncate(text, 30_000), data={"url": args["url"]})


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _cite_source(args, ctx):
    """Record a citation, verifying the quote actually appears at the URL."""
    url, quote = args["url"], args["quote"]
    text = ctx.fetched.get(url)
    if text is None:
        try:
            text = _fetch(url)
            ctx.fetched[url] = text
        except (urllib.error.URLError, ValueError, OSError, TimeoutError) as e:
            text = None
            reason = f"could not fetch to verify ({type(e).__name__})"
    if text is not None and _norm(quote) in _norm(text):
        ev = ctx.ledger.add_evidence("source", f"verified quote from {url}", {"url": url, "quote": quote})
        return ToolResult(f"quote verified at source; evidence {ev.id} (kind=source)",
                          data={"evidence_id": ev.id, "no_evidence": True})
    if text is not None:
        reason = "quote not found in the fetched page"
    ev = ctx.ledger.add_evidence("note", f"UNVERIFIED citation of {url}: {reason}",
                                 {"url": url, "quote": quote})
    return ToolResult(f"citation NOT verified ({reason}); recorded as note {ev.id}, which cannot "
                      "ground a fact.", data={"evidence_id": ev.id, "no_evidence": True})


def _memory_search(args, ctx):
    if ctx.memory is None:
        return "no long-term memory attached"
    kinds = args.get("kinds") or ["knowledge", "procedure", "episode"]
    items = ctx.memory.search(args["query"], kinds=kinds, limit=int(args.get("limit", 8)))
    if not items:
        return "no matching memories"
    return "\n".join(f"[{it.kind}#{it.id} {'verified' if it.verified else 'unverified'}"
                     f"{f' {it.successes}W/{it.failures}L' if it.kind == 'procedure' else ''}] "
                     f"{it.subject}: {it.content[:500]}" for it in items)


def _memory_store(args, ctx):
    if ctx.memory is None:
        return ToolResult("no long-term memory attached", is_error=True)
    if args["kind"] == "procedure":
        mid = ctx.memory.add_procedure(args["subject"], args["content"], run_id=ctx.run_id,
                                       meta={"author": ctx.agent})
    else:
        mid = ctx.memory.add_knowledge(args["subject"], args["content"], status=args.get("status", "speculation"),
                                       confidence=float(args.get("confidence", 0.5)), verified=False,
                                       run_id=ctx.run_id, meta={"author": ctx.agent})
    return f"stored as unverified {args['kind']} #{mid} (only verified outcomes change future behaviour)"


def _add_note(args, ctx):
    kind = args.get("kind", "note")
    ev = ctx.ledger.add_evidence(kind, args["summary"], {"detail": args.get("detail", ""), "author": ctx.agent})
    return f"recorded {kind} evidence {ev.id}"


def _assert_claim(args, ctx):
    try:
        c = ctx.ledger.assert_claim(args["statement"], args["status"], float(args["confidence"]),
                                    evidence=args.get("evidence_ids", []), depends_on=args.get("depends_on", []),
                                    author=ctx.agent)
    except (LedgerError, ValueError) as e:
        return ToolResult(f"claim rejected: {e}", is_error=True)
    notes = f" notes: {'; '.join(c.notes)}" if c.notes else ""
    return f"claim {c.id} recorded as {c.status.value} (p={c.confidence:.2f}).{notes}"


def _record_falsification(args, ctx):
    try:
        c = ctx.ledger.record_falsification(args["claim_id"], args["test"], args["outcome"],
                                            args.get("evidence_ids", []))
    except LedgerError as e:
        return ToolResult(str(e), is_error=True)
    return f"falsification recorded on {c.id}: {args['outcome']} (refuted={c.refuted})"


def _mark_conflict(args, ctx):
    try:
        ctx.ledger.mark_conflict(args["claim_a"], args["claim_b"])
    except LedgerError as e:
        return ToolResult(str(e), is_error=True)
    if args.get("note"):
        ctx.ledger.add_evidence("note", f"conflict {args['claim_a']} vs {args['claim_b']}: {args['note']}")
    return f"conflict recorded between {args['claim_a']} and {args['claim_b']}; it stays open until one is refuted"


def _ledger_view(args, ctx):
    return ctx.ledger.summary(int(args.get("max_items", 40)))


def _request_human(args, ctx):
    decision = should_ask_human(float(args.get("confidence", 0.5)), float(args.get("impact", 0.5)),
                                human_available=ctx.human.available,
                                irreversible=bool(args.get("irreversible", False)))
    if not decision.ask:
        return (f"Not asking a human: {decision.reason}. Proceed with your best judgment and record "
                "the assumption as a speculation claim.")
    answer = ctx.human.ask(f"{args['question']}\n(context: {args.get('context', '')})")
    ev = ctx.ledger.add_evidence("source", f"human answer to: {args['question'][:200]}",
                                 {"question": args["question"], "answer": answer})
    return f"Human answered (evidence {ev.id}): {answer}"


def _perceive(args, ctx):
    p = perceive(ctx.resolve(args["path"]))
    header = {"type": "text", "text": f"[{p.kind}] {p.summary}"
                                      + ("" if p.perceived else " (NOT perceived)")}
    return ToolResult([header, *p.blocks], data={"perceived": p.perceived})


def _observe_env(args, ctx):
    return json.dumps(observe_environment(ctx.workspace), indent=1)


def _compare_hypotheses(args, ctx):
    from .science.discovery import posterior_from_fits
    from .science.hypotheses import ExpressionHypothesis, FitError, FormulaError
    import math as _m
    data = args["data"]
    xs = [float(p[0]) for p in data]
    ys = [float(p[1]) for p in data]
    fits, errors = {}, []
    for h in args["hypotheses"]:
        try:
            hyp = ExpressionHypothesis(h["name"], h["formula"], tuple(h.get("params", [])))
            fits[h["name"]] = hyp.fit(xs, ys, robust=bool(args.get("robust", False)))
        except (FormulaError, FitError, KeyError, ValueError) as e:
            errors.append(f"{h.get('name')}: {e}")
    if not fits:
        return ToolResult("no hypothesis could be fitted: " + "; ".join(errors), is_error=True)
    prior = {n: _m.log(1 / len(fits)) for n in fits}
    try:
        post = posterior_from_fits(fits, prior)
    except FitError as e:
        return ToolResult(f"{e} (too few data points for these hypotheses?)", is_error=True)
    lo, hi = min(xs), max(xs)
    cands = [lo + (hi - lo) * i / 200 for i in range(201)]

    def spread(x):
        preds = [(post.get(n, 0), f.predict(x)) for n, f in fits.items() if n in post]
        preds = [(p, v) for p, v in preds if _m.isfinite(v)]
        if not preds:
            return 0.0
        m = sum(p * v for p, v in preds)
        return sum(p * (v - m) ** 2 for p, v in preds)

    x_best = max(cands, key=spread)
    report = {"posterior": {k: round(v, 4) for k, v in post.items()},
              "fits": {n: {"params": {k: round(v, 6) for k, v in f.params.items()},
                           "bic": round(f.bic, 3), "rss": f.rss} for n, f in fits.items()},
              "most_discriminating_x": x_best, "errors": errors}
    return json.dumps(report, indent=1)


def builtin_tools() -> list[Tool]:
    S, I, N, B, A = ({"type": "string"}, {"type": "integer"}, {"type": "number"},
                     {"type": "boolean"}, {"type": "array"})
    return [
        Tool("read_file", "Read a text file in the workspace (line-numbered).",
             {"path": S, "offset": I, "limit": I}, ["path"], _read_file, Risk.READ_ONLY, True, "observation"),
        Tool("write_file", "Create or overwrite a file in the workspace. Previous versions are backed up.",
             {"path": S, "content": S}, ["path", "content"], _write_file, Risk.LOW),
        Tool("list_dir", "List files under a workspace directory.", {"path": S, "depth": I}, [],
             _list_dir, Risk.READ_ONLY, True),
        Tool("search_files", "Regex search over files in the workspace.",
             {"pattern": S, "path": S, "glob": S}, ["pattern"], _search_files, Risk.READ_ONLY, True, "observation"),
        Tool("run_shell", "Run a shell command in the workspace. Risky commands need human approval.",
             {"command": S, "timeout": I}, ["command"], _run_shell,
             lambda a: classify_shell(a.get("command", "")), False, "execution"),
        Tool("run_python", "Run a Python snippet in a subprocess (cwd = workspace).",
             {"code": S, "timeout": I}, ["code"], _run_python, Risk.LOW, False, "execution"),
        Tool("fetch_url", "Fetch a web page or document over HTTP(S) and return its text.",
             {"url": S}, ["url"], _fetch_url, Risk.LOW, True, "source"),
        Tool("cite_source", "Record a citation. The quote is checked against the page text; only "
             "verified quotes become source evidence that can ground facts.",
             {"url": S, "quote": S}, ["url", "quote"], _cite_source, Risk.LOW, True),
        Tool("memory_search", "Search long-term memory (knowledge, procedures, past episodes).",
             {"query": S, "kinds": {**A, "items": S}, "limit": I}, ["query"], _memory_search, Risk.READ_ONLY, True),
        Tool("memory_store", "Store a lesson (procedure) or knowledge item in long-term memory (unverified).",
             {"kind": {"type": "string", "enum": ["knowledge", "procedure"]}, "subject": S, "content": S,
              "status": {"type": "string", "enum": ["fact", "inference", "speculation"]}, "confidence": N},
             ["kind", "subject", "content"], _memory_store, Risk.LOW),
        Tool("add_note", "Record reasoning (derivation) or a note in the evidence ledger. These cannot "
             "ground facts; executions, observations and verified sources do.",
             {"summary": S, "detail": S, "kind": {"type": "string", "enum": ["note", "derivation"]}},
             ["summary"], _add_note, Risk.LOW),
        Tool("assert_claim", "Record a claim with epistemic status. FACT requires grounding evidence ids; "
             "INFERENCE requires premises or evidence; anything else is stored as SPECULATION.",
             {"statement": S, "status": {"type": "string", "enum": ["fact", "inference", "speculation"]},
              "confidence": N, "evidence_ids": {**A, "items": S}, "depends_on": {**A, "items": S}},
             ["statement", "status", "confidence"], _assert_claim, Risk.LOW),
        Tool("record_falsification", "Record the outcome of a test designed to refute a claim.",
             {"claim_id": S, "test": S, "outcome": {"type": "string", "enum": ["survived", "refuted", "inconclusive"]},
              "evidence_ids": {**A, "items": S}}, ["claim_id", "test", "outcome"], _record_falsification, Risk.LOW),
        Tool("mark_conflict", "Record that two claims contradict each other (e.g. sources disagree). "
             "Unresolved conflicts are reported until one claim is refuted.",
             {"claim_a": S, "claim_b": S, "note": S}, ["claim_a", "claim_b"], _mark_conflict, Risk.LOW),
        Tool("view_ledger", "Show current facts, inferences, speculation, refutations and conflicts.",
             {"max_items": I}, [], _ledger_view, Risk.READ_ONLY, True),
        Tool("request_human_input", "Ask a human. Only asks when expected loss from guessing exceeds the cost "
             "of interrupting; otherwise tells you to proceed and record the assumption.",
             {"question": S, "context": S, "confidence": N, "impact": N, "irreversible": B},
             ["question", "confidence", "impact"], _request_human, Risk.READ_ONLY),
        Tool("perceive_file", "Look at a file of any type: text, code, image, PDF, audio or video.",
             {"path": S}, ["path"], _perceive, Risk.READ_ONLY, True, "observation"),
        Tool("observe_environment", "Snapshot of the OS, workspace files, git state and available tools.",
             {}, [], _observe_env, Risk.READ_ONLY, True, "observation"),
        Tool("compare_hypotheses", "Fit competing quantitative hypotheses (formulas in x with free "
             "parameters) to data, compare them by BIC, and suggest the most discriminating next x.",
             {"data": {**A, "items": {"type": "array", "items": N}},
              "hypotheses": {**A, "items": {"type": "object", "properties": {
                  "name": S, "formula": S, "params": {**A, "items": S}}, "required": ["name", "formula"]}},
              "robust": B}, ["data", "hypotheses"], _compare_hypotheses, Risk.READ_ONLY, True, "derivation"),
    ]


def default_registry() -> ToolRegistry:
    return ToolRegistry(builtin_tools())
