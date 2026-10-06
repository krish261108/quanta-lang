"""The deliverable: result, evidence, limitations, reproducibility, and an
explicit separation of established facts, inferences and speculation.

The fact/inference/speculation sections are generated from the ledger, not
from model prose, so the separation reflects what was actually grounded.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

from .epistemics import Ledger, Status


def code_version(path: str | Path | None = None) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=path or Path(__file__).parent,
                           capture_output=True, text=True, timeout=10)
        sha = r.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=path or Path(__file__).parent,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{sha}{'+dirty' if dirty else ''}" if sha else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _evidence_refs(ledger: Ledger, ids: list[str], limit: int = 4) -> str:
    refs = []
    for eid in ids[:limit]:
        ev = ledger.evidence.get(eid)
        if ev:
            refs.append(f"{eid} ({ev.kind}: {ev.summary[:80]})")
    more = f", +{len(ids) - limit} more" if len(ids) > limit else ""
    return "; ".join(refs) + more


def render_report(*, title: str, problem: str, answer: str, confidence: float | None, ledger: Ledger,
                  limitations: list[str], reproduction: dict, resources: dict | None = None,
                  audit: dict | None = None, extra_sections: list[tuple[str, str]] | None = None,
                  max_items: int = 60) -> str:
    out = [f"# {title}", "", f"**Problem.** {problem}", "", "## Result", "", answer]
    if confidence is not None:
        out += ["", f"**Credence:** {confidence:.2f} (stated probability that the answer is correct, capped "
                    "by the weakest claim it rests on; calibration is only measured where ground truth exists)"]

    facts = sorted(ledger.by_status(Status.FACT), key=lambda c: -c.confidence)[:max_items]
    infs = sorted(ledger.by_status(Status.INFERENCE), key=lambda c: -c.confidence)[:max_items]
    specs = sorted(ledger.by_status(Status.SPECULATION), key=lambda c: -c.confidence)[:max_items]

    out += ["", "## Established facts", "",
            "_Claims grounded in direct observation, executed checks, verified sources or independent "
            "verification._", ""]
    out += [f"- **{c.id}** {c.statement}  \n  evidence: {_evidence_refs(ledger, c.evidence_ids)}" for c in facts] \
        or ["- (none)"]
    out += ["", "## Inferences", "", "_Derived from facts by reasoning or statistics; can be wrong if a "
                                     "premise or the method is wrong._", ""]
    for c in infs:
        basis = []
        if c.depends_on:
            basis.append("from " + ", ".join(c.depends_on))
        if c.evidence_ids:
            basis.append("evidence " + _evidence_refs(ledger, c.evidence_ids, 3))
        flag = " **[needs review]**" if c.needs_review else ""
        out.append(f"- **{c.id}** (p={c.confidence:.2f}){flag} {c.statement}  \n  {'; '.join(basis)}")
    if not infs:
        out.append("- (none)")
    out += ["", "## Speculation and open hypotheses", "", "_Not established. Listed so they are not "
                                                         "mistaken for findings._", ""]
    out += [f"- **{c.id}** (p={c.confidence:.2f}) {c.statement}" for c in specs] or ["- (none)"]

    tested = [c for c in ledger.claims.values() if c.falsifications]
    out += ["", "## Falsification record", ""]
    if tested:
        for c in tested:
            for f in c.falsifications:
                out.append(f"- {c.id} — {f.outcome.upper()}: {f.test}")
    else:
        out.append("- No falsification tests were run.")
    refuted = ledger.refuted()
    if refuted:
        out += ["", "## Refuted claims", ""] + [f"- {c.id}: {c.statement}" for c in refuted]
    conflicts = ledger.unresolved_conflicts()
    if conflicts:
        out += ["", "## Unresolved conflicts", ""] + [f"- {a.id} ({a.statement}) vs {b.id} ({b.statement})"
                                                      for a, b in conflicts]
    for heading, body in extra_sections or []:
        out += ["", f"## {heading}", "", body]
    out += ["", "## Limitations", ""] + ([f"- {l}" for l in limitations] or ["- (none stated)"])
    repro = {"code_version": code_version(), "python": sys.version.split()[0],
             "platform": platform.platform(), **reproduction}
    out += ["", "## Reproducibility", ""] + [f"- **{k}:** `{v}`" for k, v in repro.items()]
    if resources:
        out += ["", "## Resource use", ""] + [f"- {k}: {v}" for k, v in resources.items()]
    if audit:
        out += ["", "## Audit", ""] + [f"- {k}: {v}" for k, v in audit.items()]
    return "\n".join(out) + "\n"
