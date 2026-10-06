"""The evaluation service (what the acquisition loop may see) and the independent
reviewer (who decides whether a capability becomes ACTIVE).

PROTECTED. Neither the acquisition loop nor a capability can modify this module,
the gates below, or the tasks they are computed on.

Information flow
----------------
The proposer (the acquisition loop) may:
* solve training tasks and receive its own outputs and observations, never grades;
* compare two configurations on the validation split (per-task scores allowed:
  selection happens here).

Only the reviewer touches held-out data. Every review uses a fresh, never-reused
block of the gate and transfer splits, so held-out tasks cannot be selected on
across attempts. The reviewer does not receive the proposer's reasoning: it gets
the base configuration, the candidate configuration, the artifact, a pointer to
validation records stored by the service, and the list of seeds the proposer has
seen. It recomputes every statistic from raw records itself.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..stats import paired_bootstrap
from .acquisition import (DECISION_SPLITS, REPORT_ONLY_SPLITS, SPLITS, evaluate, make_task)
from .policy import PolicyError, check_source

GATES = {
    "confidence": 0.95,
    "n_boot": 4000,
    "validation_ci_low": 0.0,            # paired score gain on the proposer's validation run
    "heldout_ci_low": 0.0,               # paired score gain on a fresh gate block (mixed tasks)
    "familiar_regression_ci_low": -0.01, # no meaningful loss on the familiar tasks of that block
    "transfer_ci_low": 0.0,              # paired score gain on a fresh block of transfer tasks
    "n_gate": 150,
    "n_transfer": 100,
}
GATES_SHA = hashlib.sha256(json.dumps(GATES, sort_keys=True).encode()).hexdigest()


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _artifact_hash(artifact: dict) -> str:
    canonical = json.dumps(artifact.get("terms", []), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _block_tasks(split: str, n: int, block: int, replication: int):
    return [make_task(split, block * n + i, replication) for i in range(n)]


class EvaluationService:
    def __init__(self, log_dir: str | Path, *, replication: int = 0, workers: int = 4) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.replication = replication
        self.workers = workers
        self._cache: dict[tuple, list[dict]] = {}
        self.seen_seeds: set[int] = set()
        self.validations: dict[str, dict] = {}

    def records(self, config: dict, split: str, n: int, block: int = 0, *, with_observations=False,
                workers: int | None = None, use_cache: bool = True) -> list[dict]:
        key = (config_hash(config), split, n, block, with_observations)
        if use_cache and key in self._cache:
            return self._cache[key]
        recs = evaluate(config, split, n, replication=self.replication, workers=workers or self.workers,
                        with_observations=with_observations,
                        task_list=_block_tasks(split, n, block, self.replication))
        path = self.log_dir / f"{split}-r{self.replication}-b{block}-{key[0]}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps({k: v for k, v in r.items() if k != "observations"}, default=str) + "\n")
        if use_cache:
            self._cache[key] = recs
        return recs

    # -- the proposer's view --------------------------------------------------
    def solve_training(self, config: dict, n: int, block: int = 0) -> list[dict]:
        recs = self.records(config, "train", n, block, with_observations=True)
        self.seen_seeds.update(r["seed"] for r in recs)
        hidden = {"score", "interp_nrmse", "extrap_nrmse", "recovered", "brier", "p_claim", "family", "kind",
                  "outliers", "claimed", "cost"}
        return [{k: v for k, v in r.items() if k not in hidden} for r in recs]

    def validate(self, base: dict, cand: dict, n: int, block: int = 0) -> dict:
        rb = self.records(base, "validation", n, block)
        rc = self.records(cand, "validation", n, block)
        self.seen_seeds.update(r["seed"] for r in rb)
        cmp = paired_bootstrap([r["score"] for r in rb], [r["score"] for r in rc],
                               n_boot=GATES["n_boot"], confidence=GATES["confidence"], seed=7)
        vid = f"val-{config_hash(base)}-{config_hash(cand)}-b{block}"
        self.validations[vid] = {"base": rb, "cand": rc}
        return {"validation_id": vid, "mean_delta": cmp.mean_delta, "ci_low": cmp.ci_low, "ci_high": cmp.ci_high,
                "per_task": [{"task_id": a["task_id"], "base": a["score"], "cand": b["score"]}
                             for a, b in zip(rb, rc)]}


@dataclass
class ReviewDecision:
    review_id: str
    approved: bool
    gates: dict
    reasons: list[str]
    artifact_sha: str
    gates_sha: str
    block: int
    seconds: float
    recomputed: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Reviewer:
    def __init__(self, service: EvaluationService, log_dir: str | Path, *, workers: int = 4) -> None:
        self.service = service
        self.log_dir = Path(log_dir)
        (self.log_dir / "reviews").mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self._state_path = self.log_dir / "reviewer_state.json"
        self.state = json.loads(self._state_path.read_text()) if self._state_path.exists() else {"next_block": 0}

    def _next_block(self) -> int:
        b = self.state["next_block"]
        self.state["next_block"] = b + 1
        self._state_path.write_text(json.dumps(self.state))
        return b

    def review(self, base: dict, cand: dict, artifact: dict, validation_id: str,
               proposer_seen_seeds: set[int], unit_test_runner=None) -> ReviewDecision:
        t0 = time.time()
        gates: dict[str, dict] = {}
        reasons: list[str] = []
        block = self._next_block()
        rid = f"review-r{self.service.replication}-b{block}-{_artifact_hash(artifact)[:10]}"

        def gate(name: str, passed: bool, value, threshold) -> None:
            gates[name] = {"passed": bool(passed), "value": value, "threshold": threshold}
            if not passed:
                reasons.append(f"{name}: {value} (threshold {threshold})")

        # R0 integrity: the candidate is the base plus exactly this artifact, which passes the policy.
        sha_ok = artifact.get("sha256") == _artifact_hash(artifact)
        policy_ok = True
        try:
            for t in artifact.get("terms", []):
                check_source(f"lambda x, p: {t['source']}")
        except (PolicyError, KeyError, TypeError):
            policy_ok = False
        diff_ok = (cand.get("genome") == base.get("genome")
                   and cand.get("capabilities", [])[:-1] == base.get("capabilities", [])
                   and cand.get("capabilities", [])[-1:] == [artifact])
        gate("integrity", sha_ok and policy_ok and diff_ok,
             {"hash": sha_ok, "policy": policy_ok, "only_this_artifact_added": diff_ok}, "all true")

        # R1 unit tests, re-run independently (not the proposer's report).
        if unit_test_runner is not None:
            ut = unit_test_runner(artifact)
            gate("unit_tests", ut.get("passed", False), {k: ut.get(k) for k in ("recovery", "finite",
                                                                                 "deterministic", "runtime_s")},
                 "passed")

        # R7 leakage: the proposer saw only decision-split training/validation seeds.
        allowed = {s for s in DECISION_SPLITS if s in ("train", "validation")}
        ranges = [(SPLITS[s] + 1_000_000 * self.service.replication,
                   SPLITS[s] + 1_000_000 * self.service.replication + 10_000) for s in allowed]
        stray = [s for s in proposer_seen_seeds if not any(lo <= s < hi for lo, hi in ranges)]
        gate("no_leakage", not stray, {"seeds_outside_train_validation": len(stray)}, 0)

        if not all(g["passed"] for g in gates.values()):
            return self._finish(rid, False, gates, reasons, artifact, block, t0, {})

        # R2 validation, recomputed from the service's stored records.
        val = self.service.validations.get(validation_id)
        if val is None:
            gate("validation", False, "validation records not found", "present")
            return self._finish(rid, False, gates, reasons, artifact, block, t0, {})
        cv = self._paired(val["base"], val["cand"])
        gate("validation", cv["ci_low"] > GATES["validation_ci_low"], cv, f"ci_low > {GATES['validation_ci_low']}")

        # R3/R4 held-out: a fresh gate block.
        rb = self.service.records(base, "gate", GATES["n_gate"], block)
        rc = self.service.records(cand, "gate", GATES["n_gate"], block)
        ch = self._paired(rb, rc)
        gate("heldout", ch["ci_low"] > GATES["heldout_ci_low"], ch, f"ci_low > {GATES['heldout_ci_low']}")
        fam_b = [r for r in rb if r["kind"] == "familiar"]
        fam_c = [r for r in rc if r["kind"] == "familiar"]
        cf = self._paired(fam_b, fam_c)
        gate("familiar_regression", cf["ci_low"] > GATES["familiar_regression_ci_low"], cf,
             f"ci_low > {GATES['familiar_regression_ci_low']}")

        # R5 transfer: a fresh block of regime-shifted gap tasks.
        tb = self.service.records(base, "gate_transfer", GATES["n_transfer"], block)
        tc = self.service.records(cand, "gate_transfer", GATES["n_transfer"], block)
        ct = self._paired(tb, tc)
        gate("transfer", ct["ci_low"] > GATES["transfer_ci_low"], ct, f"ci_low > {GATES['transfer_ci_low']}")

        # R6 reproducibility: same block, fresh workers, different worker count.
        rc2 = self.service.records(cand, "gate", GATES["n_gate"], block, workers=max(1, self.workers - 1),
                                   use_cache=False)
        same = [a["score"] for a in rc] == [b["score"] for b in rc2]
        gate("reproducible", same, {"identical_per_task_scores": same}, True)

        recomputed = {"validation": cv, "heldout": ch, "familiar": cf, "transfer": ct,
                      "heldout_by_kind": self._by_kind(rb, rc)}
        approved = all(g["passed"] for g in gates.values())
        return self._finish(rid, approved, gates, reasons, artifact, block, t0, recomputed)

    @staticmethod
    def _paired(base: list[dict], cand: list[dict]) -> dict:
        if [r["seed"] for r in base] != [r["seed"] for r in cand]:
            raise ValueError("unpaired records")
        if not base:
            return {"n": 0, "mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        c = paired_bootstrap([r["score"] for r in base], [r["score"] for r in cand],
                             n_boot=GATES["n_boot"], confidence=GATES["confidence"], seed=11)
        return {"n": c.n, "mean_delta": round(c.mean_delta, 5), "ci_low": round(c.ci_low, 5),
                "ci_high": round(c.ci_high, 5), "wins": c.wins, "losses": c.losses,
                "recovered_base": sum(r["recovered"] for r in base) / len(base),
                "recovered_cand": sum(r["recovered"] for r in cand) / len(cand)}

    def _by_kind(self, base, cand) -> dict:
        out = {}
        for kind in sorted({r["kind"] for r in base}):
            b = [r for r in base if r["kind"] == kind]
            c = [r for r in cand if r["kind"] == kind]
            out[kind] = self._paired(b, c)
        return out

    def _finish(self, rid, approved, gates, reasons, artifact, block, t0, recomputed) -> ReviewDecision:
        d = ReviewDecision(rid, approved, gates, reasons or ["all gates passed"], _artifact_hash(artifact),
                           GATES_SHA, block, round(time.time() - t0, 1), recomputed)
        (self.log_dir / "reviews" / f"{rid}.json").write_text(json.dumps(d.to_dict(), indent=1, default=str))
        return d


def assert_report_only(split: str) -> None:
    if split not in REPORT_ONLY_SPLITS:
        raise ValueError(f"{split} is not a report-only split")
