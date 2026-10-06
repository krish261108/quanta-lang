"""Run the discovery loop over a split of benchmark tasks.

Splits use disjoint seed ranges:

* selection    - used by the self-improver to compare candidates
* confirmation - fresh tasks used to replicate an apparent improvement
* test         - held out; never used for any decision, only for reporting
* experience   - "past tasks" whose verified outcomes populate memory

PROTECTED: the self-improver may not modify this package.

Can be run as a module (this is how sandboxed candidates are evaluated):

    python -m quanta.bench.harness --genome g.json --split selection --n 60 --out r.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..genome import Genome
from ..memory import MemoryStore
from ..science.discovery import DiscoveryLoop
from ..uncertainty import Calibration
from .tasks import DOMAIN_LABEL, generate_task, grade

SPLITS = {"selection": 10_000, "confirmation": 20_000, "test": 30_000, "experience": 40_000}


def split_seeds(split: str, n: int, offset: int = 0) -> list[int]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")
    base = SPLITS[split] + offset
    return list(range(base, base + n))


def build_experience_memory(n: int, path: str = ":memory:") -> MemoryStore:
    """Record the verified outcomes of `n` past tasks in long-term memory."""
    mem = MemoryStore(path)
    for seed in split_seeds("experience", n):
        task = generate_task(seed)
        mem.add_knowledge(DOMAIN_LABEL, f"verified generating law: {task.expected_answer}",
                          status="fact", confidence=1.0, verified=True,
                          meta={"family": task.expected_answer, "seed": seed})
    return mem


def prior_counts_from_memory(mem: MemoryStore) -> Counter:
    return mem.count_by_meta("knowledge", DOMAIN_LABEL, "family", verified_only=True)


def run_task(genome: Genome, seed: int, prior_counts: dict | None = None) -> dict:
    task = generate_task(seed)
    loop = DiscoveryLoop(genome, budget=task.budget, seed=seed, prior_counts=prior_counts)
    t0 = time.perf_counter()
    res = loop.run(task.system())
    elapsed = time.perf_counter() - t0
    g = grade(task, res.answer, res.credence, res.predict, res.n_experiments)
    return {
        "seed": seed, "truth": task.family, "expected": task.expected_answer,
        "category": task.category, "outliers": task.outlier_rate > 0,
        "answer": res.answer, "stop_reason": res.stop_reason, "expanded": res.expanded,
        "n_experiments": res.n_experiments,
        "falsifications": len(res.falsifications),
        "refutations": sum(1 for f in res.falsifications if f["outcome"] == "refuted"),
        "asked_human": res.human_request is not None,
        "seconds": round(elapsed, 3), **g,
    }


def _worker(args) -> dict:
    genome_dict, seed, prior = args
    return run_task(Genome.from_dict(genome_dict), seed, prior)


@dataclass
class SuiteResult:
    genome: Genome
    split: str
    results: list[dict]
    diagnostics: dict = field(default_factory=dict)

    @property
    def scores(self) -> list[float]:
        return [r["score"] for r in self.results]

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores)

    def to_dict(self) -> dict:
        return {"genome": self.genome.to_dict(), "fingerprint": self.genome.fingerprint(),
                "split": self.split, "mean_score": self.mean_score,
                "diagnostics": self.diagnostics, "results": self.results}


def diagnose(results: list[dict]) -> dict:
    """Aggregate failure analysis used to propose improvements."""
    n = len(results)
    cal = Calibration()
    for r in results:
        cal.add(r["credence"], r["correct"])

    def acc(rows):
        return sum(r["correct"] for r in rows) / len(rows) if rows else float("nan")

    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    wrong = [r for r in results if not r["correct"]]
    confusions = Counter(f"{r['expected']}->{r['answer']}" for r in wrong)
    return {
        "n": n,
        "mean_score": sum(r["score"] for r in results) / n,
        "accuracy": cal.accuracy(),
        "mean_credence": cal.mean_confidence(),
        "brier": cal.brier(),
        "ece": cal.ece(),
        "overconfident_error_rate": sum(1 for r in wrong if r["credence"] > 0.8) / n,
        "underconfident_correct_rate": sum(1 for r in results if r["correct"] and r["credence"] < 0.5) / n,
        "mean_cost": sum(r["cost"] for r in results) / n,
        "mean_nrmse": sum(min(r["nrmse"], 10) for r in results) / n,
        "budget_exhausted_rate": sum(1 for r in results if r["stop_reason"] == "budget exhausted") / n,
        "accuracy_by_category": {k: acc(v) for k, v in sorted(by_cat.items())},
        "count_by_category": {k: len(v) for k, v in sorted(by_cat.items())},
        "top_confusions": confusions.most_common(6),
        "ask_rate": sum(1 for r in results if r["asked_human"]) / n,
    }


def run_suite(genome: Genome, split: str = "selection", n: int = 60, *, jobs: int = 1,
              offset: int = 0, experience_n: int = 60, prior_counts: dict | None = None) -> SuiteResult:
    seeds = split_seeds(split, n, offset)
    if genome.use_learned_priors and prior_counts is None:
        with build_experience_memory(experience_n) as mem:
            prior_counts = dict(prior_counts_from_memory(mem))
    args = [(genome.to_dict(), s, prior_counts) for s in seeds]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_worker, args, chunksize=max(1, len(args) // (jobs * 4))))
    else:
        results = [_worker(a) for a in args]
    return SuiteResult(genome, split, results, diagnose(results))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the discovery benchmark for a genome.")
    ap.add_argument("--genome", help="genome JSON (default: baseline)")
    ap.add_argument("--split", default="selection", choices=sorted(SPLITS))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", help="write full results JSON here")
    args = ap.parse_args(argv)
    genome = Genome.load(args.genome) if args.genome else Genome()
    res = run_suite(genome, args.split, args.n, jobs=args.jobs, offset=args.offset)
    payload = res.to_dict()
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    d = res.diagnostics
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v)
                      for k, v in d.items()}, default=str, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
