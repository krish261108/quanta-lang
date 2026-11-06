"""Attempts to falsify the self-improvement result, plus supporting analyses.

Run after `quanta improve --run-dir experiments/self_improvement`:

    PYTHONPATH=src python experiments/analyze.py

Produces experiments/analysis.json and experiments/ANALYSIS.md:

1. Ablations: revert each adopted change individually (on the held-out test
   split). If reverting a change does not hurt, that change was not needed.
2. Control: the same improvement loop with *random* proposals instead of
   diagnosis-driven ones, at the same budget. If random search does as well,
   the diagnostic reasoning is not what produced the gain.
3. Null check: the gate applied to a genome against itself must reject.
4. Calibration: reliability bins before and after.
5. Learning: score as a function of verified experience in memory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from quanta.bench.harness import build_experience_memory, prior_counts_from_memory, run_suite
from quanta.genome import Genome
from quanta.improve import Evaluator, GateConfig, RandomProposer, SelfImprover, gate
from quanta.stats import paired_bootstrap
from quanta.uncertainty import Calibration

HERE = Path(__file__).parent
N_TEST = 400
JOBS = 4


def cmp_dict(base, cand):
    c = paired_bootstrap(base.scores, cand.scores, seed=7)
    return {"mean_base": round(base.mean_score, 4), "mean_cand": round(cand.mean_score, 4),
            "delta": round(c.mean_delta, 4), "ci": [round(c.ci_low, 4), round(c.ci_high, 4)],
            "wins": c.wins, "losses": c.losses, "sign_p": c.sign_p}


def reliability(results):
    cal = Calibration()
    for r in results:
        cal.add(r["credence"], r["correct"])
    return [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}
            for row in cal.reliability(5)]


def main() -> int:
    run = HERE / "self_improvement"
    initial = Genome.load(run / "genomes" / "v0.json")
    final = Genome.load(run / "genomes" / "final.json")
    out: dict = {"initial": initial.fingerprint(), "final": final.fingerprint(), "diff": initial.diff(final)}

    test_i = run_suite(initial, "test", N_TEST, jobs=JOBS)
    test_f = run_suite(final, "test", N_TEST, jobs=JOBS)
    out["test"] = cmp_dict(test_i, test_f)
    out["test_diagnostics"] = {"initial": test_i.diagnostics, "final": test_f.diagnostics}

    # 1. ablations
    ablations = {}
    for name, (v0, v1) in initial.diff(final).items():
        reverted = final.mutate(**{name: v0})
        res = run_suite(reverted, "test", N_TEST, jobs=JOBS)
        ablations[f"{name}: {v1} -> {v0}"] = cmp_dict(test_f, res)
        print("ablation", name, ablations[f"{name}: {v1} -> {v0}"], flush=True)
    out["ablations"] = ablations

    # 2. random-search control at the same budget
    summary = json.loads((run / "summary.json").read_text())
    ctrl_dir = HERE / "random_search_control"
    imp = SelfImprover(Evaluator(jobs=JOBS), [RandomProposer()], run_dir=ctrl_dir, generations=8,
                       candidates_per_generation=6, n_selection=200, n_confirmation=200, n_test=N_TEST, seed=0)
    ctrl = imp.run(initial)
    out["random_search_control"] = {
        "adopted": ctrl.adopted, "generations_run": ctrl.generations_run, "stop_reason": ctrl.stop_reason,
        "test_vs_initial": cmp_dict(ctrl.test_initial, ctrl.test_final),
        "diagnostic_final_vs_random_final": cmp_dict(ctrl.test_final, test_f),
    }
    print("random control", out["random_search_control"]["test_vs_initial"], flush=True)

    # 3. null check (A/A)
    a = run_suite(final, "confirmation", 120, jobs=JOBS, offset=5000)
    b = run_suite(final, "confirmation", 120, jobs=JOBS, offset=5000)
    out["null_check"] = gate(a, b, GateConfig()).to_dict()

    # 4. calibration
    out["reliability"] = {"initial": reliability(test_i.results), "final": reliability(test_f.results)}

    # 5. learning curve (final genome with learned priors on vs. amount of experience)
    curve = []
    learner = final.mutate(use_learned_priors=True)
    no_memory = run_suite(final.mutate(use_learned_priors=False), "test", N_TEST, jobs=JOBS)
    for k in (0, 10, 30, 100, 300):
        with build_experience_memory(k) as mem:
            counts = dict(prior_counts_from_memory(mem))
        res = run_suite(learner, "test", N_TEST, jobs=JOBS, prior_counts=counts)
        curve.append({"experience": k, "mean_score": round(res.mean_score, 4),
                      "accuracy": round(res.diagnostics["accuracy"], 4),
                      "vs_no_memory": cmp_dict(no_memory, res)})
        print("learning", curve[-1], flush=True)
    out["learning_curve"] = curve
    out["summary_from_run"] = {k: summary[k] for k in ("adopted", "stop_reason", "generations_run")}

    (HERE / "analysis.json").write_text(json.dumps(out, indent=1, default=str))
    write_markdown(out)
    return 0


def write_markdown(o: dict) -> None:
    t = o["test"]
    di, df = o["test_diagnostics"]["initial"], o["test_diagnostics"]["final"]
    lines = ["# Analysis of the self-improvement run", "",
             "Generated by `experiments/analyze.py`. All numbers are on the held-out test split "
             f"({N_TEST} tasks) unless noted; deltas are paired (same tasks), CIs are 95% bootstrap.", "",
             "## Headline", "",
             f"- Mean score: {t['mean_base']:.4f} -> {t['mean_cand']:.4f} (Δ {t['delta']:+.4f}, CI "
             f"[{t['ci'][0]:+.4f}, {t['ci'][1]:+.4f}], {t['wins']} wins / {t['losses']} losses)",
             f"- Accuracy: {di['accuracy']:.3f} -> {df['accuracy']:.3f}",
             f"- Brier score: {di['brier']:.3f} -> {df['brier']:.3f}; ECE: {di['ece']:.3f} -> {df['ece']:.3f}",
             f"- Confidently wrong (credence > 0.8): {di['overconfident_error_rate']:.3f} -> "
             f"{df['overconfident_error_rate']:.3f}",
             f"- Mean experiment cost (fraction of budget): {di['mean_cost']:.3f} -> {df['mean_cost']:.3f}",
             f"- Accuracy by category: {json.dumps({k: round(v, 3) for k, v in di['accuracy_by_category'].items()})}"
             f" -> {json.dumps({k: round(v, 3) for k, v in df['accuracy_by_category'].items()})}",
             "", "## Ablations (revert one adopted change; negative Δ = the change was helping)", "",
             "| reverted change | Δ score vs final | 95% CI |", "|---|---|---|"]
    for name, c in o["ablations"].items():
        lines.append(f"| `{name}` | {c['delta']:+.4f} | [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] |")
    rc = o["random_search_control"]
    lines += ["", "## Control: random proposals instead of diagnosis", "",
              f"- Random search adopted {len(rc['adopted'])} change(s) in {rc['generations_run']} generation(s) "
              f"({rc['stop_reason']}).",
              f"- Random-search final vs initial: Δ {rc['test_vs_initial']['delta']:+.4f} "
              f"(CI [{rc['test_vs_initial']['ci'][0]:+.4f}, {rc['test_vs_initial']['ci'][1]:+.4f}])",
              f"- Diagnostic final vs random-search final: Δ {rc['diagnostic_final_vs_random_final']['delta']:+.4f} "
              f"(CI [{rc['diagnostic_final_vs_random_final']['ci'][0]:+.4f}, "
              f"{rc['diagnostic_final_vs_random_final']['ci'][1]:+.4f}])",
              "", "## Null check", "",
              f"- Gate on identical runs: accept={o['null_check']['accept']} ({'; '.join(o['null_check']['reasons'])})",
              "", "## Calibration (reliability bins: mean credence vs. accuracy)", ""]
    for which in ("initial", "final"):
        lines.append(f"**{which}**: " + "; ".join(
            f"[{r['bin'][0]:.1f}-{r['bin'][1]:.1f}] n={r['n']} conf={r['mean_confidence']:.2f} acc={r['accuracy']:.2f}"
            for r in o["reliability"][which]))
        lines.append("")
    lines += ["## Learning from verified experience", "",
              "| verified past tasks in memory | mean score | accuracy | Δ vs. no memory | 95% CI |",
              "|---|---|---|---|---|"]
    for row in o["learning_curve"]:
        v = row["vs_no_memory"]
        lines.append(f"| {row['experience']} | {row['mean_score']:.4f} | {row['accuracy']:.3f} | {v['delta']:+.4f} | "
                     f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] |")
    (HERE / "ANALYSIS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
