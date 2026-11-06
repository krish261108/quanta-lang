"""Independent verification of conclusions.

The verifier gets a fresh context containing only the claims and the
evidence cited for them - not the reasoning that produced them - and is
asked to refute them. Its verdicts change the ledger mechanically:

* supported, citing grounding evidence  -> promoted to FACT via a
  `verification` evidence record that points at the grounding
* supported, without grounding           -> stays as is, noted
* refuted                                -> recorded falsification; refutation
  propagates to every dependent claim
* insufficient                           -> confidence reduced
"""
from __future__ import annotations

from .agent import AgentFactory, submit_tool
from .epistemics import GROUNDING_KINDS, Claim, Ledger, LedgerError, Status
from .tools import ToolContext

MODE_INSTRUCTIONS = {
    "self": "Re-check each claim from scratch as if you had never seen it.",
    "independent": "Check each claim independently. Re-run computations and re-read sources yourself.",
    "adversarial": "For each claim, design and run the test most likely to refute it. Look for "
                   "counterexamples, confounders, alternative explanations and numerical errors. Support a "
                   "claim only if it survives a genuine attempt to break it.",
}


def briefing(ledger: Ledger, claims: list[Claim]) -> str:
    lines = []
    for c in claims:
        lines.append(f"### {c.id} [{c.status.value}, p={c.confidence:.2f}] {c.statement}")
        for eid in c.evidence_ids[:12]:
            ev = ledger.evidence.get(eid)
            if ev:
                lines.append(f"  - evidence {eid} ({ev.kind}): {ev.summary[:300]}")
        if c.depends_on:
            lines.append(f"  - depends on: {', '.join(c.depends_on)}")
    return "\n".join(lines)


def verdict_tool(ledger: Ledger, claim_ids: set[str]):
    def validate(args: dict) -> list[str]:
        errs = []
        for v in args.get("verdicts", []):
            if v.get("claim_id") not in claim_ids:
                errs.append(f"unknown claim id {v.get('claim_id')!r}")
            if v.get("verdict") not in ("supported", "refuted", "insufficient"):
                errs.append(f"bad verdict {v.get('verdict')!r}")
            for e in v.get("evidence_ids", []):
                if e not in ledger.evidence:
                    errs.append(f"unknown evidence id {e!r}")
        if not args.get("verdicts"):
            errs.append("at least one verdict is required")
        return errs

    return submit_tool(
        "submit_verdicts", "Submit one verdict per claim you checked.",
        {"verdicts": {"type": "array", "items": {"type": "object", "properties": {
            "claim_id": {"type": "string"},
            "verdict": {"type": "string", "enum": ["supported", "refuted", "insufficient"]},
            "reason": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["claim_id", "verdict", "reason"]}}},
        ["verdicts"], validate)


def apply_verdicts(ledger: Ledger, verdicts: list[dict], verifier: str) -> dict:
    summary = {"supported": [], "promoted": [], "refuted": [], "insufficient": []}
    for v in verdicts:
        cid, verdict = v["claim_id"], v["verdict"]
        reason = v.get("reason", "")[:500]
        cited = [e for e in v.get("evidence_ids", []) if e in ledger.evidence]
        grounding = [e for e in cited if ledger.evidence[e].kind in GROUNDING_KINDS]
        claim = ledger.get(cid)
        if verdict == "refuted":
            ledger.record_falsification(cid, f"independent verification ({verifier}): {reason}", "refuted", cited)
            summary["refuted"].append(cid)
        elif verdict == "supported":
            summary["supported"].append(cid)
            ledger.record_falsification(cid, f"independent verification ({verifier}): {reason}", "survived", cited)
            if grounding and not claim.refuted and claim.status is not Status.FACT:
                ev = ledger.add_evidence("verification", f"{verifier} supported {cid}: {reason}",
                                         {"basis": grounding, "verifier": verifier})
                try:
                    ledger.promote_to_fact(cid, ev.id)
                    summary["promoted"].append(cid)
                except LedgerError:
                    pass
            elif not grounding:
                claim.notes.append(f"supported by {verifier} without grounding evidence; status unchanged")
        else:
            ledger.update_confidence(cid, claim.confidence * 0.7, f"{verifier}: insufficient evidence - {reason}")
            summary["insufficient"].append(cid)
    return summary


def review(factory: AgentFactory, ctx: ToolContext, claims: list[Claim], *, mode: str = "independent",
           max_steps: int = 30) -> tuple[dict, dict | None]:
    """Run a fresh-context verifier over `claims`; returns (summary, raw submission)."""
    if not claims:
        return {"supported": [], "promoted": [], "refuted": [], "insufficient": []}, None
    tool = verdict_tool(ctx.ledger, {c.id for c in claims})
    agent = factory.create("verifier", ctx, submit=tool, max_steps=max_steps,
                           system_extra=MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["independent"]))
    res = agent.run("Verify the following claims. You only see the claims and their cited evidence, not "
                    "the reasoning behind them.\n\n" + briefing(ctx.ledger, claims)
                    + "\n\nWhen you support a claim, cite the evidence ids (including any you created by "
                      "re-running checks) that ground it.")
    if res.submitted is None:
        return {"supported": [], "promoted": [], "refuted": [], "insufficient": [],
                "error": f"verifier did not submit ({res.stop_reason})"}, None
    return apply_verdicts(ctx.ledger, res.submitted["verdicts"], f"verifier[{mode}]"), res.submitted
