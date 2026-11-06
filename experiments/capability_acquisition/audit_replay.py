"""Audit of the replayed acquisition phases (deviation D1) and of the leakage gate's blind
spot (independent integrity review, findings 2 and 6).

    python experiments/capability_acquisition/audit_replay.py --replication 0 [--sample 40]

1. Reused record files: for every evaluation record file that the replay took over from
   the interrupted attempt, rebuild the exact configuration from its hash, re-run a random
   sample of its tasks from scratch, and require identical per-task scores.
2. Information flow: every held-out (gate, gate_transfer) record file must belong to a
   block the reviewer consumed, and no report-only split may appear among the acquisition
   phase's evaluations at all.
Writes results/rep<r>/<phase>/replay_audit.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

from quanta.bench import acquisition as A  # noqa: E402
from quanta.bench.review import config_hash  # noqa: E402

NAME = re.compile(r"^(?P<split>[a-z_]+)-r(?P<rep>\d+)-b(?P<block>\d+)-n(?P<n>\d+)-(?P<hash>[0-9a-f]{16})\.jsonl$")


def candidate_configs(phase_dir: Path) -> dict[str, dict]:
    """Every configuration the loop can have evaluated: S0 plus each prefix of the accepted
    capabilities, with or without each proposed artifact appended."""
    proposed, accepted = [], []
    for line in (phase_dir / "registry.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["event"] == "propose":
            proposed.append(e["record"]["artifact"])
        elif e["event"] == "status" and e["status"] == "ACTIVE":
            accepted.append(e["capability_id"])
    by_id = {a["capability_id"]: a for a in proposed}
    out = {}
    for k in range(len(accepted) + 1):
        base = [by_id[c] for c in accepted[:k]]
        for extra in [None, *proposed]:
            caps = base + ([extra] if extra is not None else [])
            cfg = run.cfg(run.S0, caps)
            out[config_hash(cfg)] = cfg
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replication", type=int, default=0)
    ap.add_argument("--phase", default="acquire", choices=["acquire", "random"])
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    rep = args.replication
    d = run.HERE / "results" / f"rep{rep}" / args.phase
    stale_dirs = sorted(d.parent.glob(f"{args.phase}.interrupted-*"))
    stale_names = {f.name for s in stale_dirs for f in (s / "evals").glob("*.jsonl")}
    configs = candidate_configs(d)
    rng = random.Random(1234 + rep)
    reused, flow = [], []
    reviewed_blocks = {json.loads(f.read_text())["block"] for f in (d / "reviewer" / "reviews").glob("*.json")}
    for f in sorted((d / "evals").glob("*.jsonl")):
        m = NAME.match(f.name)
        if not m:
            if not f.name.endswith("-rerun.jsonl"):
                flow.append({"file": f.name, "problem": "unrecognized record file"})
            continue
        split, block, n = m["split"], int(m["block"]), int(m["n"])
        if split in A.REPORT_ONLY_SPLITS:
            flow.append({"file": f.name, "problem": "report-only split evaluated during acquisition"})
        if split in ("gate", "gate_transfer") and block not in reviewed_blocks:
            flow.append({"file": f.name, "problem": f"held-out block {block} not consumed by any review"})
        if f.name not in stale_names:
            continue
        cfg = configs.get(m["hash"])
        if cfg is None:
            reused.append({"file": f.name, "checked": 0, "problem": "configuration not reconstructable"})
            continue
        stored = run.read_jsonl(f)
        idx = sorted(rng.sample(range(len(stored)), min(args.sample, len(stored))))
        tasks = [A.make_task(split, block * n + i, rep) for i in idx]
        fresh = A.evaluate(cfg, split, len(tasks), replication=rep, workers=args.workers, task_list=tasks)
        same = [stored[i]["score"] for i in idx] == [r["score"] for r in fresh]
        reused.append({"file": f.name, "checked": len(idx), "identical_scores": same})
        run.log(f"replay audit {f.name}: {len(idx)} tasks recomputed, identical={same}")
    res = {"replication": rep, "phase": args.phase, "interrupted_attempts": [s.name for s in stale_dirs],
           "reused_files": reused, "all_reused_identical": all(r.get("identical_scores") for r in reused),
           "information_flow_problems": flow, "reviewed_blocks": sorted(reviewed_blocks)}
    run.dump_json(d / "replay_audit.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "reused_files"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
