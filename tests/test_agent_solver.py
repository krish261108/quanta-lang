import re

from quanta.agent import Agent, AgentFactory, submit_tool
from quanta.epistemics import Ledger, Status
from quanta.genome import Genome
from quanta.governance import AuditLog, Budget, Governor, KillSwitch
from quanta.llm import ScriptedModel
from quanta.memory import MemoryStore
from quanta.solver import ResearchLoop, RunState
from quanta.tools import HumanChannel, ToolContext, default_registry


def convo_text(messages) -> str:
    parts = []
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            parts.append(c)
            continue
        for b in c:
            if b.get("type") == "text":
                parts.append(b["text"])
            elif b.get("type") == "tool_result":
                cc = b["content"]
                parts.append(cc if isinstance(cc, str) else " ".join(x.get("text", "") for x in cc))
    return "\n".join(parts)


def make_ctx(tmp_path, **kw):
    return ToolContext(tmp_path, kw.pop("governor", Governor()), Ledger(), MemoryStore(), **kw)


def test_agent_tool_loop_and_submission(tmp_path):
    model = ScriptedModel([
        {"tool_calls": [("run_python", {"code": "print(1+1)"})]},
        {"text": "done"},                                   # forgets to submit -> nudged
        {"tool_calls": [("submit_answer", {"answer": "2"})]},
    ])
    tool = submit_tool("submit_answer", "answer", {"answer": {"type": "string"}}, ["answer"])
    ctx = make_ctx(tmp_path)
    res = Agent(model=model, registry=default_registry(), ctx=ctx, role="coder", submit=tool).run("compute 1+1")
    assert res.submitted == {"answer": "2"} and res.stop_reason == "submitted"
    assert res.tool_log[0]["evidence"] in ctx.ledger.evidence
    # history is append-only: the second call saw the first assistant turn verbatim
    assert model.calls[1]["messages"][1]["content"][0]["type"] == "tool_use"
    assert ctx.governor.budget.steps == 3


def test_agent_rejects_invalid_submission_then_accepts(tmp_path):
    tool = submit_tool("submit_n", "n", {"n": {"type": "number"}}, ["n"],
                       lambda a: [] if a["n"] > 0 else ["n must be positive"])
    model = ScriptedModel([{"tool_calls": [("submit_n", {"n": -1})]}, {"tool_calls": [("submit_n", {"n": 3})]}])
    res = Agent(model=model, registry=default_registry(), ctx=make_ctx(tmp_path), submit=tool).run("x")
    assert res.submitted == {"n": 3}


def test_agent_halts_on_budget_and_refusal(tmp_path):
    model = ScriptedModel(lambda s, m, t: {"tool_calls": [("list_dir", {})]})
    ctx = make_ctx(tmp_path, governor=Governor(budget=Budget(max_steps=3)))
    res = Agent(model=model, registry=default_registry(), ctx=ctx).run("loop forever")
    assert res.stop_reason.startswith("halted") and res.steps == 3
    model = ScriptedModel([{"stop_reason": "refusal", "refusal_category": "cyber"}])
    res = Agent(model=model, registry=default_registry(), ctx=make_ctx(tmp_path)).run("x")
    assert res.stop_reason.startswith("refusal")


def test_kill_switch_stops_agent(tmp_path):
    stop = tmp_path / "STOP"
    model = ScriptedModel(lambda s, m, t: (stop.write_text("x"), {"tool_calls": [("list_dir", {})]})[1])
    ctx = make_ctx(tmp_path, governor=Governor(kill_switch=KillSwitch(stop)))
    res = Agent(model=model, registry=default_registry(), ctx=ctx).run("x")
    assert "kill switch" in res.stop_reason or "stop file" in res.stop_reason
    assert res.steps == 1


def test_delegation_to_subagent(tmp_path):
    def brain(system, messages, tools):
        if "You are the coder" in system:
            return {"tool_calls": [("submit_report", {"summary": "child computed 42", "success": True})]}
        text = convo_text(messages)
        if "child computed" not in text:
            return {"tool_calls": [("spawn_subagent", {"role": "coder", "task": "compute 6*7"}),
                                   ("spawn_subagent", {"role": "coder", "task": "compute 7*6"})]}
        return {"tool_calls": [("submit_answer", {"answer": "42"})]}

    model = ScriptedModel(brain)
    factory = AgentFactory(model, default_registry(), max_depth=1)
    ctx = make_ctx(tmp_path)
    tool = submit_tool("submit_answer", "a", {"answer": {"type": "string"}}, ["answer"])
    res = factory.create("generalist", ctx, submit=tool).run("delegate it")
    assert res.submitted == {"answer": "42"}
    coder_calls = [c for c in model.calls if "You are the coder" in c["system"]]
    assert len(coder_calls) == 2
    assert "spawn_subagent" not in coder_calls[0]["tools"]      # depth limit
    assert coder_calls[0]["effort"] == "medium"                 # sub-agents run at lower effort


def research_brain(system, messages, tools):
    names = {t["name"] for t in tools}
    text = convo_text(messages)
    evid = re.findall(r"recorded as evidence (E\d+)", text)
    if "submit_frame" in names:
        return {"tool_calls": [("submit_frame", {
            "objective": "determine x", "success_criteria": ["x verified by execution"],
            "unknowns": ["value of x"], "assumptions": ["x is an integer"],
            "clarifying_questions": [
                {"question": "Is x positive?", "impact": 0.9, "confidence_without_answer": 0.3},
                {"question": "Use tabs?", "impact": 0.05, "confidence_without_answer": 0.5}]})]}
    if "submit_research" in names:
        return {"tool_calls": [("submit_research", {"findings": [{"statement": "x is defined as 2+2"}]})]}
    if "submit_hypotheses" in names:
        return {"tool_calls": [("submit_hypotheses", {"hypotheses": [
            {"id": "H1", "statement": "x = 4", "predictions": ["2+2 prints 4"], "falsification_test": "run 2+2",
             "prior": 0.5},
            {"id": "H2", "statement": "x = 5", "predictions": ["2+2 prints 5"], "falsification_test": "run 2+2",
             "prior": 0.3},
            {"id": "H3", "statement": "x = 3", "predictions": ["2+2 prints 3"], "falsification_test": "run 2+2",
             "prior": 0.2}]})]}
    if "submit_plan" in names:
        return {"tool_calls": [("submit_plan", {"tasks": [
            {"id": "t1", "title": "compute 2+2", "description": "run it", "deps": [],
             "acceptance": "printed result", "role": "experimenter"}]})]}
    if "submit_task_result" in names:
        if not evid:
            return {"tool_calls": [("run_python", {"code": "print(2+2)"})]}
        claim = re.findall(r"claim (C\d+) recorded", text)
        if not claim:
            return {"tool_calls": [("assert_claim", {"statement": "2+2 evaluates to 4", "status": "fact",
                                                     "confidence": 0.99, "evidence_ids": [evid[0]]})]}
        return {"tool_calls": [("submit_task_result", {"success": True, "summary": "2+2 = 4",
                                                       "evidence_ids": evid, "claim_ids": claim})]}
    if "submit_update" in names:
        return {"tool_calls": [("submit_update", {"updates": [
            {"hypothesis_id": "H1", "likelihood_ratio": 50, "justification": "execution printed 4"},
            {"hypothesis_id": "H2", "likelihood_ratio": 0.02, "justification": "contradicted"},
            {"hypothesis_id": "H3", "likelihood_ratio": 0.02, "justification": "contradicted"}]})]}
    if "submit_verdicts" in names:
        if not evid:
            return {"tool_calls": [("run_python", {"code": "assert 2 + 2 == 4; print('ok')"})]}
        ids = re.findall(r"### (C\d+)", text)
        return {"tool_calls": [("submit_verdicts", {"verdicts": [
            {"claim_id": c, "verdict": "supported", "reason": "re-ran", "evidence_ids": [evid[-1]]} for c in ids]})]}
    if "submit_result" in names:
        facts = re.findall(r"- (C\d+) \(p=[0-9.]+\): 2\+2 evaluates to 4", text)
        return {"tool_calls": [("submit_result", {
            "answer": "x = 4", "confidence": 0.97, "key_claim_ids": facts,
            "limitations": ["toy problem"], "reproduction_steps": ["python -c 'print(2+2)'"],
            "next_steps": ["none"]})]}
    return {"text": "nothing to do"}


def test_research_loop_end_to_end(tmp_path):
    model = ScriptedModel(research_brain)
    mem = MemoryStore()
    asked = []
    loop = ResearchLoop("What is x, where x is defined as 2+2?", model=model, run_dir=tmp_path / "run",
                        memory=mem, human=HumanChannel(lambda q: asked.append(q) or "yes"),
                        genome=Genome(min_hypotheses=3, delegate=False))
    report_path = loop.run()
    st = loop.state
    assert st.done and st.result["answer"] == "x = 4"
    assert asked == ["Is x positive?"]                          # low-impact question was not asked
    assert st.hypotheses[0]["id"] == "H1" and st.hypotheses[0]["posterior"] > 0.99
    assert st.verification and st.verification[0]["supported"]
    led = loop.ledger
    h1 = led.get(st.hypotheses[0]["claim_id"])
    assert h1.status is Status.FACT                             # promoted by grounded verification
    assert any("Assumed without asking: Use tabs?" in c.statement for c in led.claims.values())
    report = report_path.read_text()
    for heading in ("## Result", "## Established facts", "## Inferences", "## Speculation",
                    "## Falsification record", "## Limitations", "## Reproducibility", "## Audit"):
        assert heading in report
    assert "2+2 evaluates to 4" in report and "chain_intact: True" in report
    assert mem.stats().get("knowledge_verified", 0) >= 1
    assert (tmp_path / "run" / "audit.jsonl").exists()


def test_research_loop_resumes_after_interruption(tmp_path):
    run_dir = tmp_path / "run"
    first = ResearchLoop("What is x, where x is defined as 2+2?", model=ScriptedModel(research_brain),
                         run_dir=run_dir, genome=Genome(delegate=False, research_depth=0))
    first._phase_frame()
    first._phase_hypothesize()
    first._checkpoint()                                         # simulate a crash after two phases
    assert RunState.load(run_dir / "state.json").phase == "plan"
    resumed = ResearchLoop.resume(run_dir, model=ScriptedModel(research_brain), genome=Genome(delegate=False))
    assert resumed.state.run_id == first.state.run_id
    assert resumed.governor.budget.steps == first.governor.budget.steps   # spending carried over
    resumed.run()
    assert resumed.state.done and resumed.state.result["answer"] == "x = 4"


def test_research_loop_halts_gracefully_on_budget(tmp_path):
    gov = Governor(budget=Budget(max_steps=4), audit=AuditLog(tmp_path / "a.jsonl"))
    loop = ResearchLoop("What is x?", model=ScriptedModel(research_brain), run_dir=tmp_path / "run",
                        governor=gov, genome=Genome(delegate=False))
    report = loop.run().read_text()
    assert loop.state.done
    assert "budget exhausted" in report


def test_large_task_is_decomposed_hierarchically(tmp_path):
    def brain(system, messages, tools):
        names = {t["name"] for t in tools}
        first = messages[0]["content"][0]["text"]
        if "submit_task_result" in names and "# Your task: t1 -" in first:
            return {"tool_calls": [("submit_task_result", {
                "success": False, "summary": "too big; splitting",
                "subtasks": [{"id": "a", "title": "compute 2+2", "description": "run it", "acceptance": "printed",
                              "role": "experimenter"},
                             {"id": "b", "title": "double-check", "description": "run again", "deps": ["a"],
                              "acceptance": "printed", "role": "experimenter"}]})]}
        return research_brain(system, messages, tools)

    loop = ResearchLoop("What is x, where x is defined as 2+2?", model=ScriptedModel(brain), run_dir=tmp_path / "r",
                        genome=Genome(delegate=False, research_depth=0))
    loop.run()
    tasks = loop.state.plan["tasks"]
    status = {t["id"]: t["status"] for t in tasks}
    assert status == {"t1": "refined", "t1.a": "done", "t1.b": "done"}
    assert loop.state.done and loop.state.result["answer"] == "x = 4"


def test_resume_keeps_the_runs_genome(tmp_path):
    run_dir = tmp_path / "run"
    g = Genome(min_hypotheses=2, delegate=False, research_depth=0)
    ResearchLoop("q", model=ScriptedModel(research_brain), run_dir=run_dir, genome=g)._checkpoint()
    resumed = ResearchLoop.resume(run_dir, model=ScriptedModel(research_brain))
    assert resumed.genome == g


def test_anthropic_request_shape():
    from types import SimpleNamespace
    from quanta.llm import AnthropicModel

    sent = []

    class FakeMessages:
        def create(self, **params):
            sent.append(params)
            block = SimpleNamespace(type="text", text="hi")
            usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0,
                                    cache_creation_input_tokens=0)
            stop = "pause_turn" if len(sent) == 1 else "end_turn"
            return SimpleNamespace(content=[block], stop_reason=stop, usage=usage, model="claude-opus-5-5",
                                   stop_details=None)

    client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages()))
    m = AnthropicModel(client=client, web_search=False, web_fetch=False)
    r = m.respond(system="s", messages=[{"role": "user", "content": "q"}], tools=[], effort="low")
    first = sent[0]
    assert first["model"] == "claude-opus-5-5" and first["thinking"] == {"type": "adaptive"}
    assert first["output_config"] == {"effort": "low"} and "tools" not in first
    assert first["fallbacks"] == "default" and first["betas"] == ["server-side-fallback-2026-07-01"]
    assert "budget_tokens" not in str(first) and "tool_choice" not in first
    # pause_turn: the partial assistant turn is appended and the request resent
    assert sent[1]["messages"][-1]["role"] == "assistant"
    assert r.text == "hihi" and r.usage.input_tokens == 20 and r.cost_usd > 0
