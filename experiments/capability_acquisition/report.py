"""Write RESULTS.md from the result files alone (nothing typed in by hand except the
deviations list below, which records changes made after pre-registration).

    python experiments/capability_acquisition/report.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

DEVIATIONS = [
    ("D1", "A session restart killed both acquisition runs mid-round (replication 0 in round 3, "
           "replication 1 in round 2). Each was replayed from the start with the same code and seed; "
           "complete evaluation records of the interrupted attempt were reused only when they covered "
           "exactly the same tasks (`EvaluationService.records`, seed-by-seed check). This touched the "
           "protected evaluator after registration. The replay's decisions are compared with the "
           "interrupted attempt's (`replay_check`)."),
    ("D2", "Exploratory analyses X1-X3 (EXPLORATORY_PLAN.md) were added during the acquisition phase, "
           "before any report-only split was evaluated. They cannot change a pre-registered verdict."),
]


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def fmt_ci(d, key="mean_delta"):
    if not d:
        return "n/a"
    return f"{d[key]:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"


def verdict(v) -> str:
    if v is True:
        return "supported"
    if v is False:
        return "not supported"
    return "exploratory"


def review_rows(rep_dir: Path) -> list[str]:
    rows = []
    by_sha = {}
    regfile = rep_dir / "acquire" / "registry.jsonl"
    if regfile.exists():
        for line in regfile.read_text().splitlines():
            e = json.loads(line)
            if e["event"] == "propose":
                art = e["record"]["artifact"]
                by_sha[art.get("sha256")] = (e["capability_id"], art.get("structure"))
    for f in sorted((rep_dir / "acquire" / "reviewer" / "reviews").glob("*.json"),
                    key=lambda p: json.loads(p.read_text())["block"]):
        d = json.loads(f.read_text())
        g = d["gates"]
        cid, structure = by_sha.get(d["artifact_sha"], ("?", "?"))

        def cell(name):
            v = g.get(name)
            if not v:
                return "–"
            val = v["value"]
            mark = "✅" if v["passed"] else "❌"
            if isinstance(val, dict) and "mean_delta" in val:
                return f"{mark} {val['mean_delta']:+.4f} [{val['ci_low']:+.4f}]"
            return mark
        rows.append(f"| {d['block']} | {cid} `{structure}` | "
                    f"{'**ACTIVE**' if d['approved'] else 'rejected'} | {cell('heldout')} | "
                    f"{cell('familiar_regression')} | {cell('transfer')} | {cell('validation')} | "
                    f"{cell('reproducible')} |")
    return rows


def rep_section(rep: int) -> list[str]:
    d = RESULTS / f"rep{rep}"
    an = load(d / "analysis.json")
    acq = load(d / "acquire" / "summary.json")
    rnd = load(d / "random" / "summary.json")
    out = [f"## Replication {rep}", ""]
    if acq:
        rep_ = acq["report"]
        out += [f"Acquisition: {len(rep_['rounds'])} rounds, {rep_['reviews']} reviews, "
                f"{len(acq['active'])} ACTIVE, {len(rep_['rejected'])} rejected; "
                f"{acq['proposer_seen_seeds']} task seeds seen by the proposer (all train/validation).", ""]
        for r in rep_["rounds"]:
            out.append(f"* round {r['round']}: {r['failures']} self-detected failures "
                       f"({', '.join(f'{k} {v}' for k, v in r['failure_modes'].items())}); researched "
                       f"{r.get('researched')}, kept {r.get('found')}; candidates "
                       f"{', '.join('`' + c + '`' for c in r.get('candidates', []))}; reviews {r.get('reviews')}; "
                       f"accepted {r.get('accepted')}" + (f"; retry resolved {r['retry']['resolved']} of "
                                                         f"{r['retry']['self_detected_failures_before']}"
                                                         if r.get('retry') else ""))
        out += ["", "ACTIVE capabilities: " + (", ".join(f"`{a['capability_id']}` `{a['structure']}`"
                                                        for a in acq["active"]) or "none"), ""]
        dc = rep_.get("diagnosis_consistency") or {}
        out += [f"Diagnosis consistency (F8): {dc.get('consistent')}/{dc.get('checked')} research findings "
                f"matched the diagnosed failure mode (rate {dc.get('rate')}).", ""]
        rc = acq.get("replay_check") or []
        for c in rc:
            out.append(f"Replay check (D1): the replay reproduced the interrupted attempt's "
                       f"{c['decisions_before_interruption']} decisions exactly: **{c['identical_prefix']}**.")
        out += ["", "| block | capability | decision | held-out gain | familiar regression | transfer gain | "
                    "validation | reproducible |", "|---|---|---|---|---|---|---|---|"]
        out += review_rows(d)
        out.append("")
    if rnd:
        out += [f"Random-proposal control (F1): {len(rnd['report']['rejected'])} rejected, "
                f"{len(rnd['active'])} ACTIVE"
                + (": " + ", ".join(f"`{a['structure']}`" for a in rnd["active"]) if rnd["active"] else "") + ".",
                ""]
    if not an:
        out += ["(final analysis not available yet)", ""]
        return out
    h = an["hypotheses"]
    out += ["### Pre-registered hypotheses", "", "| | verdict |", "|---|---|"]
    for k, v in h.items():
        out.append(f"| {k} | {verdict(v) if not isinstance(v, dict) else 'exploratory: ' + fmt_ci(v)} |")
    out += ["", "### Final evaluation (report-only splits), difference vs C0 (95% CI)", ""]
    splits = ["test", "test_familiar", "test_regime", "test_compose", "sealed", "external"]
    out += ["| condition | " + " | ".join(splits) + " |", "|---|" + "---|" * len(splits)]
    for c, by in an["vs_C0"].items():
        out.append(f"| {c} | " + " | ".join(fmt_ci(by.get(s)) for s in splits) + " |")
    out += ["", "Absolute scores (C0): " + ", ".join(
        f"{s} {an['descriptive']['C0_S0'][s].get('score', an['descriptive']['C0_S0'][s].get('score_ext')):.4f}"
        for s in splits if s in an["descriptive"]["C0_S0"]), ""]
    out += ["Recovery rate, C0 → C2 (`test` split): "
            f"{an['descriptive']['C0_S0']['test']['recovered']} → "
            f"{an['descriptive'].get('C2_S0_acquired', {}).get('test', {}).get('recovered')}", ""]
    if an["pairs"]:
        out += ["### Contrasts", "", "| contrast | " + " | ".join(splits) + " |", "|---|" + "---|" * len(splits)]
        for k, by in an["pairs"].items():
            out.append(f"| {k} | " + " | ".join(fmt_ci(by.get(s)) for s in splits) + " |")
        out.append("")
    if an.get("memorization"):
        m = an["memorization"]
        out += [f"Memorization (F2): effect on the training tasks the capabilities came from "
                f"{fmt_ci(m['train_tasks_capabilities_came_from'])} vs fresh test tasks "
                f"{fmt_ci(m['fresh_test_tasks'])}.", ""]
    fe = an.get("fresh_environment")
    if fe:
        out += [f"Fresh environment (F4): cold re-evaluation of {fe.get('n')} test tasks in a new interpreter "
                f"reproduced per-task scores exactly: **{fe.get('identical_per_task_scores')}**.", ""]
    ex = load(d / "exploratory.json")
    if ex:
        out += ["### Exploratory (not pre-registered decision rules)", ""]
        out += ["X2, familiar-gate power: " + "; ".join(
            f"review {r['review'].split('-')[-1]}: per-task sd {r['per_task_sd']}, n needed "
            f"{r['n_needed_for_80pct_power']}" for r in ex["X2_gate_power"]), ""]
        if ex["X1"]:
            out += ["X1, rejected-but-promising candidates on the report-only splits (vs C0):", "",
                    "| configuration | " + " | ".join(splits) + " |", "|---|" + "---|" * len(splits)]
            for k, by in ex["X1"].items():
                out.append(f"| {k} | " + " | ".join(fmt_ci(by.get(s)) for s in splits) + " |")
            out.append("")
        for k, by in ex.get("X3", {}).items():
            fam = by.get("test_familiar")
            if fam:
                out.append(f"X3, {k} on familiar tasks: components {fam['components']}, answers changed "
                           f"{fam['answers_changed']}, extra measurements {fam['mean_extra_measurements']}.")
        out.append("")
    return out


def main() -> int:
    lines = ["# Capability acquisition: results", "",
             "Generated by `report.py` from `results/rep*/`. Design: [PREREGISTRATION.md](PREREGISTRATION.md); "
             "exploratory plan: [EXPLORATORY_PLAN.md](EXPLORATORY_PLAN.md).", ""]
    for rep in (0, 1):
        if (RESULTS / f"rep{rep}").exists():
            lines += rep_section(rep)
    lines += ["## Deviations from the pre-registration", ""]
    lines += [f"* **{k}.** {v}" for k, v in DEVIATIONS]
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
