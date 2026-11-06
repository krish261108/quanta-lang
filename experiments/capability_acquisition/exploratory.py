"""Exploratory analyses X1-X3 (EXPLORATORY_PLAN.md). Run after `run.py --phase final`.

    python experiments/capability_acquisition/exploratory.py --replication 0

They cannot change any pre-registered verdict; results go to results/rep<r>/exploratory.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

from quanta.bench import acquisition as A  # noqa: E402
from quanta.bench import external as EXT  # noqa: E402
from quanta.capabilities.registry import Registry  # noqa: E402


def promising(out: Path) -> list[dict]:
    """Rejected candidates whose reviewer held-out and transfer gates both passed."""
    reg = Registry(out / "acquire" / "registry.jsonl")
    keep = []
    for rec in reg.records.values():
        gates = (rec.review or {}).get("gates") or {}
        if rec.status == "REJECTED" and gates.get("heldout", {}).get("passed") and gates.get("transfer", {}).get("passed"):
            keep.append(rec)
    return keep


def gate_power(out: Path) -> list[dict]:
    rows = []
    for f in sorted((out / "acquire" / "reviewer" / "reviews").glob("*.json")):
        d = json.loads(f.read_text())
        fam = d["gates"].get("familiar_regression", {}).get("value")
        if not isinstance(fam, dict) or not fam.get("n"):
            continue
        n, mean = fam["n"], fam["mean_delta"]
        sd = (fam["ci_high"] - fam["ci_low"]) / (2 * 1.96) * math.sqrt(n)
        margin = -0.01
        need = None if mean <= margin else math.ceil(((1.96 + 0.84) * sd / (mean - margin)) ** 2)
        rows.append({"review": d["review_id"], "n_familiar": n, "mean_delta": mean, "ci": [fam["ci_low"], fam["ci_high"]],
                     "per_task_sd": round(sd, 4), "n_needed_for_80pct_power": need,
                     "passed": d["gates"]["familiar_regression"]["passed"]})
    return rows


def decompose(base: list[dict], cand: list[dict]) -> dict:
    comp = {"claim": [], "interp": [], "extrap": [], "cost": []}
    for a, b in zip(base, cand):
        comp["claim"].append(A.W_CLAIM * ((1 - b["brier"]) - (1 - a["brier"])))
        comp["interp"].append(A.W_INTERP * (math.exp(-b["interp_nrmse"]) - math.exp(-a["interp_nrmse"])))
        comp["extrap"].append(A.W_EXTRAP * (math.exp(-b["extrap_nrmse"]) - math.exp(-a["extrap_nrmse"])))
        comp["cost"].append(-A.W_COST * (b["cost"] - a["cost"]))
    changed = [(a["answer"], b["answer"]) for a, b in zip(base, cand) if a["answer"] != b["answer"]]
    return {"components": {k: round(statistics.mean(v), 5) for k, v in comp.items()},
            "answers_changed": len(changed), "top_changes": Counter(f"{x}->{y}" for x, y in changed).most_common(8),
            "mean_extra_measurements": round(statistics.mean((b.get("measurements") or b["budget"])
                                                             - (a.get("measurements") or a["budget"])
                                                             for a, b in zip(base, cand)), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replication", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    rep = args.replication
    out = run.HERE / "results" / f"rep{rep}"
    final = out / "final"
    base = {s: run.read_jsonl(final / run.BASE / f"{s}.jsonl") for s in list(run.FINAL_SPLITS) + ["external"]}
    res = {"replication": rep, "note": "EXPLORATORY: not part of the pre-registered decision rules",
           "X2_gate_power": gate_power(out), "X1": {}, "X3": {}}
    cands = promising(out)
    res["X1_candidates"] = [{"capability_id": r.capability_id, "structure": r.artifact.get("structure")} for r in cands]
    configs = {f"S0+{r.capability_id}": run.cfg(run.S0, [r.artifact]) for r in cands}
    if len(cands) > 1:
        configs["S0+all_promising"] = run.cfg(run.S0, [r.artifact for r in cands])
    ext_tasks = EXT.tasks(rep)
    lists = {s: run.final_tasks(s, n, rep)[0] for s, n in run.FINAL_SPLITS.items()}
    for name, config in configs.items():
        recs = {}
        for s in list(run.FINAL_SPLITS) + ["external"]:
            path = out / "exploratory" / name / f"{s}.jsonl"
            if path.exists():
                recs[s] = run.read_jsonl(path)
            else:
                recs[s] = (EXT.evaluate(config, replication=rep, workers=args.workers, task_list=ext_tasks)
                           if s == "external" else
                           A.evaluate(config, s, len(lists[s]), replication=rep, workers=args.workers,
                                      task_list=lists[s]))
                run.write_jsonl(path, recs[s])
            run.log(f"exploratory {name} {s}: {len(recs[s])}")
        res["X1"][name] = run.contrast(base, recs)
        res["X3"][name] = {s: decompose(base[s], recs[s]) for s in ("test_familiar", "test") if s in recs}
    run.dump_json(out / "exploratory.json", res)
    print(json.dumps({k: res[k] for k in ("X1_candidates", "X2_gate_power")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
