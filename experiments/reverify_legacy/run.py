"""Re-verify the recorded self-improvement results with the committed code.

The numbers in experiments/ANALYSIS.md and the README (held-out test, 400 tasks:
initial 0.7225 -> final 0.7618) were produced by a code version that was never
committed: running the committed code on the same 400 test tasks gives different
per-task results. This script recomputes the parts of that analysis that do not
need a new self-improvement run (initial vs final genome, and the two ablations)
so the record can state numbers that anyone can reproduce.

    python experiments/reverify_legacy/run.py [--jobs 2]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quanta.bench.harness import run_suite  # noqa: E402
from quanta.genome import Genome  # noqa: E402
from quanta.stats import paired_bootstrap  # noqa: E402

RECORDED = json.loads((ROOT / "experiments" / "self_improvement" / "summary.json").read_text())


def cmp(a, b) -> dict:
    c = paired_bootstrap([r["score"] for r in a.results], [r["score"] for r in b.results], n_boot=4000, seed=1)
    return {"mean_delta": round(c.mean_delta, 5), "ci": [round(c.ci_low, 5), round(c.ci_high, 5)],
            "wins": c.wins, "losses": c.losses}


def brief(res) -> dict:
    d = res.diagnostics
    keys = ("mean_score", "accuracy", "brier", "ece", "overconfident_error_rate", "mean_cost",
            "accuracy_by_category")
    return {k: (round(d[k], 4) if isinstance(d[k], float) else d[k]) for k in keys}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    initial = Genome.from_dict(RECORDED["initial"])
    final = Genome.from_dict(RECORDED["final"])
    runs = {
        "initial": initial,
        "final": final,
        "final, likelihood reverted to gaussian": final.mutate(likelihood="gaussian"),
        "final, falsification_rounds reverted to 0": final.mutate(falsification_rounds=0),
    }
    res = {name: run_suite(g, "test", 400, jobs=args.jobs) for name, g in runs.items()}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = {
        "git_commit": commit,
        "recorded": {"initial": {k: RECORDED["test"]["initial"][k] for k in ("mean_score", "accuracy")},
                     "final": {k: RECORDED["test"]["final"][k] for k in ("mean_score", "accuracy")},
                     "delta": RECORDED["test"]["comparison"]["mean_delta"]},
        "recomputed": {name: brief(r) for name, r in res.items()},
        "final_vs_initial": cmp(res["initial"], res["final"]),
        "ablations_vs_final": {name: cmp(res["final"], r) for name, r in res.items() if name.startswith("final,")},
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
