"""Capability-acquisition experiment (see PREREGISTRATION.md in this directory).

    python experiments/capability_acquisition/run.py --replication 0 [--phase all] [--workers 4]

Phases (each writes under results/rep<r>/ and is skipped when its output already
exists, so an interrupted run resumes where it stopped):

  acquire   the acquisition loop on S0 (train/validation for selection; the
            independent reviewer on fresh gate/transfer blocks)
  random    the random-proposal control through the identical pipeline
  final     every condition on the report-only splits (never used for any decision)
  fresh     re-evaluate the acquired configuration in a new interpreter from the
            registry file alone; per-task scores must equal the final phase's
  analyze   pre-registered comparisons, ablations and falsification checks -> analysis.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quanta.bench import acquisition as A  # noqa: E402
from quanta.bench import external as EXT  # noqa: E402
from quanta.bench.review import GATES, GATES_SHA, EvaluationService, Reviewer, config_hash  # noqa: E402
from quanta.capabilities.acquire import AcquisitionLoop, AcquisitionSettings, RandomAcquisitionLoop  # noqa: E402
from quanta.capabilities.artifacts import build_artifact  # noqa: E402
from quanta.capabilities.registry import Registry  # noqa: E402
from quanta.genome import Genome  # noqa: E402
from quanta.memory import MemoryStore  # noqa: E402
from quanta.science.synth import X, Model, prod, unary  # noqa: E402
from quanta.science.toolkit import artifact_hash  # noqa: E402
from quanta.stats import paired_bootstrap  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered settings (PREREGISTRATION.md). Not to be edited after the first run.
# ---------------------------------------------------------------------------
S0 = Genome(falsification_rounds=2, likelihood="student_t", flexible_baseline=True, pool_equivalent=True)
S0_CONSTRUCT = Genome(falsification_rounds=2, likelihood="student_t", flexible_baseline=True, pool_equivalent=True,
                      construct=True)
SETTINGS = AcquisitionSettings(n_train=200, n_val=800, k_min=2, max_rounds=6, max_reviews_per_round=2,
                               max_candidates_per_round=4, max_research_tasks=80, synth_max_size=3, cv_ratio=0.8,
                               patience=2)
FIRST_BLOCK = {"acquire": 1, "random": 40}      # disjoint ranges of never-used gate blocks
FINAL_SPLITS = {"test": 1200, "test_familiar": 600, "test_regime": 600, "test_compose": 400, "sealed": 600}
SEED = 20261106

# Designer-built structures for the five gap classes (the ORACLE condition), with
# typical parameter values of each class as starting values (designer knowledge).
ORACLE = {
    "logistic": (Model((unary("I", unary("E", X)),)), [[-1.5, 5.5e-4]]),
    "gaussian_bump": (Model((unary("E", unary("Q", X)),)), [[-5.0, -0.5]]),
    "lorentzian": (Model((unary("I", unary("Q", X)),)), [[-5.0, 1.0]]),
    "damped_oscillation": (Model((prod(unary("E", X), unary("S", X)),)), [[-0.25, 2.5, 0.0]]),
    "linear_sin": (Model((unary("S", X), X)), [[2.0, 0.0]]),
}
PRIMARY = "C2_S0_acquired"
BASE = "C0_S0"
# v2 hands the loop a proposer-only view of the evaluation service (v1: the service itself).
PROPOSER_VIEW = False


def cfg(genome: Genome, caps: list[dict]) -> dict:
    return {"genome": genome.to_dict(), "capabilities": caps}


def oracle_artifacts() -> list[dict]:
    return [build_artifact(f"oracle-{k}", 1, m, inits, name=f"oracle {k}", description="designer-specified structure")
            for k, (m, inits) in ORACLE.items()]


def strip_inits(art: dict) -> dict:
    """The same structure without the starting values taken from training tasks."""
    a = json.loads(json.dumps(art))
    a["capability_id"] = a["capability_id"] + "-noinit"
    for t in a["terms"]:
        t["inits"] = []
    a["sha256"] = artifact_hash(a)
    return a


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def write_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({k: v for k, v in r.items() if k != "observations"}, default=str) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    """Per-task records; falls back to the committed gzip copy (see pack_results.py)."""
    if not path.exists() and path.with_name(path.name + ".gz").exists():
        import gzip
        text = gzip.decompress(path.with_name(path.name + ".gz").read_bytes()).decode("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_acquire(out: Path, rep: int, workers: int, *, control: bool) -> dict:
    name = "random" if control else "acquire"
    d = out / name
    summary_path = d / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    stale = None
    if (d / "events.jsonl").exists():
        # A loop cannot be resumed mid-round. Every step is deterministic, so the run is
        # replayed from the start; complete evaluation records of the interrupted attempt are
        # reused (bench/review.py checks they cover exactly the same tasks), and the replay's
        # decisions are compared with the interrupted attempt's afterwards.
        stale = d.with_name(f"{name}.interrupted-{int(time.time())}")
        d.rename(stale)
        log(f"{name}: previous attempt was interrupted; kept it at {stale.name} and replaying")
    d.mkdir(parents=True, exist_ok=True)
    if stale is not None and (stale / "evals").exists():
        import shutil
        shutil.copytree(stale / "evals", d / "evals", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("train-*", "*-rerun.jsonl"))
    service = EvaluationService(d / "evals", replication=rep, workers=workers)
    reviewer = Reviewer(service, d / "reviewer", workers=workers, first_block=FIRST_BLOCK[name])
    registry = Registry(d / "registry.jsonl")
    memory = MemoryStore(str(d / "memory.db"))
    cls = RandomAcquisitionLoop if control else AcquisitionLoop
    proposer = service.proposer_view() if PROPOSER_VIEW else service
    loop = cls(proposer, reviewer, registry, cfg(S0, []), log_path=d / "events.jsonl", memory=memory,
               settings=SETTINGS, seed=SEED + 1000 * rep + (7 if control else 0), run_id=f"{name}-rep{rep}")
    t0 = time.time()
    report = loop.run()
    mem = {"stats": memory.stats(),
           "procedures": [{"subject": m.subject, "content": m.content, "verified": m.verified,
                           "successes": m.successes} for m in memory.recent(kind="procedure", limit=50)],
           "knowledge": [{"subject": m.subject, "content": m.content, "verified": m.verified}
                         for m in memory.recent(kind="knowledge", limit=100)]}
    memory.close()
    summary = {
        "phase": name, "replication": rep, "seconds": round(time.time() - t0, 1), "report": report,
        "gates": GATES, "gates_sha": GATES_SHA, "reviews_used_blocks": reviewer.state,
        "active": [{"capability_id": r.capability_id, "structure": r.artifact.get("structure"),
                    "sha256": r.artifact.get("sha256"), "confidence": r.confidence,
                    "heldout": r.heldout_evidence, "transfer": r.transfer_evidence, "retry": r.retry_evidence}
                   for r in registry.by_status("ACTIVE")],
        "registry": [asdict(r) for r in registry.records.values()],
        "proposer_seen_seeds": len(service.seen_seeds), "unreviewed_seeds": len(getattr(service, "unreviewed_seeds", ())),
        "memory": mem,
        "replay_check": replay_check(d, sorted(d.parent.glob(f"{name}.interrupted-*"))),
    }
    dump_json(summary_path, summary)
    return summary


def _decisions(events_path: Path) -> list[tuple]:
    keep = ("design", "implement", "validate", "review", "retry")
    out = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e["event"] in keep:
            out.append((e["event"], e.get("capability"), e.get("structure"), e.get("approved"),
                        e.get("mean_delta"), e.get("sha256"),
                        json.dumps([c["structure"] for c in e.get("candidates", [])])))
    return out


def replay_check(d: Path, interrupted: list[Path]) -> list[dict]:
    """Did the replay take exactly the decisions the interrupted attempts had taken so far?"""
    new = _decisions(d / "events.jsonl")
    checks = []
    for old_dir in interrupted:
        old = _decisions(old_dir / "events.jsonl")
        checks.append({"attempt": old_dir.name, "decisions_before_interruption": len(old),
                       "identical_prefix": new[:len(old)] == old})
    return checks


def active_artifacts(out: Path, name: str) -> list[dict]:
    path = out / name / "registry.jsonl"
    return Registry(path).active_artifacts() if path.exists() else []


def conditions(out: Path) -> dict[str, dict]:
    acquired = active_artifacts(out, "acquire")
    rand = active_artifacts(out, "random")
    # cheap and primary conditions first; per-task construction is ~30x slower
    conds = {BASE: cfg(S0, []), PRIMARY: cfg(S0, acquired), "Crand_S0_random_control": cfg(S0, rand),
             "Coracle_S0_designer": cfg(S0, oracle_artifacts())}
    if acquired:
        conds["C2noinit_S0_acquired_without_inits"] = cfg(S0, [strip_inits(a) for a in acquired])
    conds["C1_S0_construct"] = cfg(S0_CONSTRUCT, [])
    conds["C3_S0_acquired_construct"] = cfg(S0_CONSTRUCT, acquired)
    return conds


def provenance() -> dict:
    """Hashes of the report-only test material actually used."""
    import hashlib
    sealed = ROOT / "src" / "quanta" / "bench" / "sealed.py"
    return {"sealed_py_sha256": hashlib.sha256(sealed.read_bytes()).hexdigest() if sealed.exists() else None,
            "sealed_classes": len(A.sealed_classes()) if sealed.exists() else 0,
            "external_datasets": {name: sha for name, sha in EXT.DATASETS}}


def final_tasks(split: str, n: int, rep: int):
    tasks, skipped = [], []
    for i in range(n):
        try:
            tasks.append(A.make_task(split, i, rep))
        except RuntimeError as e:            # an unidentifiable draw (possible for sealed classes): recorded
            skipped.append({"index": i, "error": str(e)})
    return tasks, skipped


def phase_final(out: Path, rep: int, workers: int) -> dict:
    d = out / "final"
    summary_path = d / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    conds = conditions(out)
    ext_tasks = EXT.tasks(rep)
    task_lists = {s: final_tasks(s, n, rep) for s, n in FINAL_SPLITS.items()}
    for s, (_, skipped) in task_lists.items():
        if skipped:
            dump_json(d / f"{s}.skipped.json", skipped)
    done: dict = defaultdict(dict)
    for cname, config in conds.items():
        for split in list(FINAL_SPLITS) + ["external"]:
            path = d / cname / f"{split}.jsonl"
            if path.exists():
                done[cname][split] = len(read_jsonl(path))
                continue
            t0 = time.time()
            if split == "external":
                recs = EXT.evaluate(config, replication=rep, workers=workers, task_list=ext_tasks)
            else:
                tl = task_lists[split][0]
                recs = A.evaluate(config, split, len(tl), replication=rep, workers=workers, task_list=tl)
            write_jsonl(path, recs)
            done[cname][split] = len(recs)
            log(f"final {cname} {split}: {len(recs)} tasks in {time.time() - t0:.0f}s")
    summary = {"conditions": list(conds), "config_hashes": {k: config_hash(v) for k, v in conds.items()},
               "provenance": provenance(),
               "n": done, "skipped": {s: len(v[1]) for s, v in task_lists.items()},
               "capabilities": {k: [a["capability_id"] for a in v["capabilities"]] for k, v in conds.items()}}
    dump_json(summary_path, summary)
    return summary


def phase_fresh(out: Path, rep: int) -> dict:
    """Cold re-evaluation in a new interpreter (fresh imports, no caches, a different
    worker count) from the registry file alone."""
    d = out / "fresh"
    summary_path = d / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    d.mkdir(parents=True, exist_ok=True)
    (d / "genome.json").write_text(json.dumps(S0.to_dict()))
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        "from quanta.bench import acquisition as A\n"
        "from quanta.capabilities.registry import Registry\n"
        f"caps = Registry({str(out / 'acquire' / 'registry.jsonl')!r}).active_artifacts()\n"
        f"genome = json.load(open({str(d / 'genome.json')!r}))\n"
        f"tasks = [A.make_task('test', i, {rep}) for i in range(200)]\n"
        f"recs = A.evaluate({{'genome': genome, 'capabilities': caps}}, 'test', 200, replication={rep},"
        " workers=3, task_list=tasks)\n"
        "print(json.dumps([[r['task_id'], r['score']] for r in recs]))\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=7200)
    if r.returncode != 0:
        summary = {"ok": False, "error": r.stderr[-2000:]}
    else:
        fresh = json.loads(r.stdout.strip().splitlines()[-1])
        main_recs = read_jsonl(out / "final" / PRIMARY / "test.jsonl")[:200]
        same = [[m["task_id"], m["score"]] for m in main_recs] == fresh
        summary = {"ok": True, "identical_per_task_scores": same, "n": len(fresh),
                   "python": platform.python_version()}
    dump_json(summary_path, summary)
    return summary


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compare(base: list[dict], cand: list[dict], key: str = "score") -> dict:
    if [r["task_id"] for r in base] != [r["task_id"] for r in cand]:
        raise ValueError("unpaired")
    vb = [float(r[key]) for r in base]
    vc = [float(r[key]) for r in cand]
    c = paired_bootstrap(vb, vc, n_boot=GATES["n_boot"], confidence=GATES["confidence"], seed=3)
    return {"n": c.n, "base": round(statistics.mean(vb), 5), "cand": round(statistics.mean(vc), 5),
            "mean_delta": round(c.mean_delta, 5), "ci_low": round(c.ci_low, 5), "ci_high": round(c.ci_high, 5),
            "wins": c.wins, "losses": c.losses}


def _num(v) -> float:
    return 1.0 if v is True else 0.0 if v is False else float(v)


def describe(recs: list[dict]) -> dict:
    out = {"n": len(recs), "score": round(statistics.mean(r["score"] for r in recs), 5),
           "recovered": round(statistics.mean(_num(r["recovered"]) for r in recs), 4),
           "confident_wrong": round(statistics.mean(
               1.0 if (r["claimed"] and r["p_claim"] >= 0.9 and not r["recovered"]) else 0.0 for r in recs), 4),
           "median_interp_nrmse": round(statistics.median(r["interp_nrmse"] for r in recs), 4),
           "median_extrap_nrmse": round(statistics.median(r["extrap_nrmse"] for r in recs), 4),
           "mean_measurements": round(statistics.mean(r.get("measurements") or r["budget"] for r in recs), 2),
           "errors": sum(1 for r in recs if r.get("error")),
           "answer_kinds": dict(Counter(r.get("answer_kind") for r in recs))}
    # score components (to see whether a change works through predictions or through the claim term)
    out["components"] = {
        "claim": round(statistics.mean(A.W_CLAIM * (1 - r["brier"]) for r in recs), 5),
        "interp": round(statistics.mean(A.W_INTERP * math.exp(-r["interp_nrmse"]) for r in recs), 5),
        "extrap": round(statistics.mean(A.W_EXTRAP * math.exp(-r["extrap_nrmse"]) for r in recs), 5),
        "cost": round(statistics.mean(-A.W_COST * r["cost"] for r in recs), 5)}
    return out


def describe_ext(recs: list[dict]) -> dict:
    return {"n": len(recs), "score_ext": round(statistics.mean(r["score_ext"] for r in recs), 5),
            "median_interp_nrmse": round(statistics.median(r["interp_nrmse"] for r in recs), 4),
            "median_extrap_nrmse": round(statistics.median(r["extrap_nrmse"] for r in recs
                                                           if r["extrap_nrmse"] is not None), 4),
            "errors": sum(1 for r in recs if r.get("error")),
            "answer_kinds": dict(Counter(r.get("answer_kind") for r in recs))}


def by_family(base: list[dict], cand: list[dict]) -> dict:
    groups: dict = defaultdict(lambda: ([], []))
    for a, b in zip(base, cand):
        groups[a["family"]][0].append(a)
        groups[a["family"]][1].append(b)
    out = {}
    for fam, (ga, gb) in sorted(groups.items()):
        out[fam] = {"n": len(ga),
                    "delta": round(statistics.mean(b["score"] - a["score"] for a, b in zip(ga, gb)), 4),
                    "recovered_base": round(statistics.mean(_num(r["recovered"]) for r in ga), 3),
                    "recovered_cand": round(statistics.mean(_num(r["recovered"]) for r in gb), 3),
                    "answers_cand": dict(Counter(r["answer"] for r in gb).most_common(3))}
    return out


def contrast(a_recs: dict, b_recs: dict) -> dict:
    out = {}
    for s in a_recs:
        if s not in b_recs:
            continue
        if s == "external":
            out[s] = EXT.cluster_bootstrap(a_recs[s], b_recs[s], n_boot=GATES["n_boot"], seed=5)
        else:
            out[s] = compare(a_recs[s], b_recs[s])
            out[s]["recovered"] = compare(a_recs[s], b_recs[s], key="recovered")
            out[s]["extrap_nrmse_capped"] = compare(
                [{**r, "x": min(r["extrap_nrmse"], 10.0)} for r in a_recs[s]],
                [{**r, "x": min(r["extrap_nrmse"], 10.0)} for r in b_recs[s]], key="x")
    return out


def diagnosis_consistency(acq_dir: Path) -> dict:
    """F8, corrected after the independent review: capability building blocks in a
    research finding are expanded into their own structure before matching, and each
    task counts once (its first kept finding). The loop's own figure is also reported."""
    import re
    from quanta.capabilities.acquire import MODE_FEATURES
    blocks = {}
    reg = acq_dir / "registry.jsonl"
    if reg.exists():
        for line in reg.read_text().splitlines():
            e = json.loads(line)
            if e["event"] == "propose":
                for i, t in enumerate(e["record"]["artifact"]["terms"]):
                    blocks[f"{e['capability_id']}.t{i}"] = t.get("structure", "")
    pattern = re.compile(r"<([^<>]+)>")

    def expand(key: str) -> str:
        for _ in range(5):
            new = pattern.sub(lambda m: blocks.get(m.group(1), m.group(0)), key)
            if new == key:
                break
            key = new
        return key

    seen, hits, total = set(), 0, 0
    per_mode: dict = defaultdict(lambda: [0, 0])
    for line in (acq_dir / "events.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e.get("event") != "research" or not e.get("kept") or e["task"] in seen:
            continue
        check = MODE_FEATURES.get(e["mode"])
        if check is None:
            continue
        seen.add(e["task"])
        ok = bool(check(expand(e["kept"])))
        hits += ok
        total += 1
        per_mode[e["mode"]][0] += ok
        per_mode[e["mode"]][1] += 1
    return {"checked_tasks": total, "consistent": hits, "rate": round(hits / total, 3) if total else None,
            "per_mode": {m: {"consistent": a, "checked": b} for m, (a, b) in per_mode.items()}}


def hypotheses(res: dict) -> dict:
    """The pre-registered decision rules (PREREGISTRATION.md, section 5)."""
    v = res["vs_C0"].get(PRIMARY, {})
    acquired = res["acquisition"]["active"]
    rand_active = res["random_control"]["active"]
    h = {}
    h["H1_acquires_and_generalizes"] = bool(acquired) and v.get("test", {}).get("ci_low", -1) > 0
    h["H2_transfers_to_regime_shift"] = bool(acquired) and v.get("test_regime", {}).get("ci_low", -1) > 0
    h["H3_no_familiar_regression"] = (not acquired) or v.get("test_familiar", {}).get("ci_low", -1) > -0.01
    h["H4_compositional_transfer"] = bool(acquired) and v.get("test_compose", {}).get("ci_low", -1) > 0
    vr = res["pairs"].get(f"{PRIMARY} - Crand_S0_random_control", {}).get("test", {})
    h["H5_beats_random_control"] = (bool(acquired) and len(acquired) > len(rand_active)
                                    and (vr.get("ci_low", -1) > 0 if rand_active else h["H1_acquires_and_generalizes"]))
    h["E1_sealed_novel_classes"] = v.get("sealed")
    h["E2_external_real_data"] = v.get("external")
    return h


def phase_analyze(out: Path, rep: int) -> dict:
    final = out / "final"
    conds = json.loads((final / "summary.json").read_text())["conditions"]
    splits = list(FINAL_SPLITS) + ["external"]
    def have(path: Path) -> bool:                    # raw file, or the committed gzip copy
        return path.exists() or path.with_name(path.name + ".gz").exists()

    recs = {c: {s: read_jsonl(final / c / f"{s}.jsonl") for s in splits if have(final / c / f"{s}.jsonl")}
            for c in conds}
    res: dict = {"replication": rep, "descriptive": {}, "vs_C0": {}, "pairs": {}, "by_family": {}}
    for c in conds:
        res["descriptive"][c] = {s: (describe_ext(r) if s == "external" else describe(r)) for s, r in recs[c].items()}
    for c in conds:
        if c != BASE:
            res["vs_C0"][c] = contrast(recs[BASE], recs[c])
            res["by_family"][c] = {s: by_family(recs[BASE][s], recs[c][s]) for s in recs[c] if s != "external"}
    # each entry is "<minuend> - <subtrahend>" and holds minuend minus subtrahend
    for minuend, subtrahend in [(PRIMARY, "Crand_S0_random_control"), ("Coracle_S0_designer", PRIMARY),
                                ("C2noinit_S0_acquired_without_inits", PRIMARY),
                                ("C3_S0_acquired_construct", "C1_S0_construct")]:
        if minuend in recs and subtrahend in recs:
            res["pairs"][f"{minuend} - {subtrahend}"] = contrast(recs[subtrahend], recs[minuend])

    # Memorization: effect on the very training tasks the capabilities came from vs fresh test tasks.
    acquired = active_artifacts(out, "acquire")
    if acquired:
        mem_dir = out / "analysis_evals"
        svc = EvaluationService(mem_dir, replication=rep, workers=4)
        tr0 = svc.records(cfg(S0, []), "train", SETTINGS.n_train)
        tr2 = svc.records(cfg(S0, acquired), "train", SETTINGS.n_train)
        res["memorization"] = {"train_tasks_capabilities_came_from": compare(tr0, tr2),
                               "fresh_test_tasks": res["vs_C0"][PRIMARY]["test"]}
    res["acquisition"] = json.loads((out / "acquire" / "summary.json").read_text())
    res["random_control"] = json.loads((out / "random" / "summary.json").read_text())
    for k in ("acquisition", "random_control"):        # keep analysis.json readable; full copies stay in summary.json
        res[k] = {kk: res[k][kk] for kk in ("active", "report", "seconds", "proposer_seen_seeds", "memory")}
    res["diagnosis_consistency_corrected"] = diagnosis_consistency(out / "acquire")
    fresh = out / "fresh" / "summary.json"
    res["fresh_environment"] = json.loads(fresh.read_text()) if fresh.exists() else None
    res["hypotheses"] = hypotheses(res)
    dump_json(out / "analysis.json", res)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replication", type=int, default=0)
    ap.add_argument("--phase", default="all", choices=["all", "acquire", "random", "final", "fresh", "analyze"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = args.replication
    out = Path(args.out) if args.out else HERE / "results" / f"rep{rep}"
    out.mkdir(parents=True, exist_ok=True)
    meta = out / "run_meta.json"
    if not meta.exists():
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip()
        dump_json(meta, {"git_commit": commit, "src_dirty": bool(dirty), "python": platform.python_version(),
                         "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gates_sha": GATES_SHA,
                         "gates": GATES, "s0": S0.to_dict(), "settings": asdict(SETTINGS),
                         "first_block": FIRST_BLOCK, "final_splits": FINAL_SPLITS, "seed": SEED,
                         "workers": args.workers})
    phases = ["acquire", "random", "final", "fresh", "analyze"] if args.phase == "all" else [args.phase]
    for ph in phases:
        log(f"phase {ph} (replication {rep})")
        t0 = time.time()
        if ph == "acquire":
            s = phase_acquire(out, rep, args.workers, control=False)
            log(f"acquire: active={[a['structure'] for a in s['active']]}")
        elif ph == "random":
            s = phase_acquire(out, rep, args.workers, control=True)
            log(f"random: active={[a['structure'] for a in s['active']]}")
        elif ph == "final":
            phase_final(out, rep, args.workers)
        elif ph == "fresh":
            log(f"fresh: {phase_fresh(out, rep)}")
        elif ph == "analyze":
            res = phase_analyze(out, rep)
            log(f"hypotheses: {json.dumps(res['hypotheses'], default=str)[:2000]}")
        log(f"phase {ph} done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
