"""The capability-acquisition loop.

    SOLVE -> EVALUATE (self-assessed) -> DIAGNOSE -> CAPABILITY GAP -> ACQUISITION HYPOTHESES
    -> RESEARCH (program search on each failed task's own observations; a law is kept only
       if it predicts held-out observations of that task better than the flexible description)
    -> DESIGN (structures that recur across failed tasks become capability candidates)
    -> IMPLEMENT (code artifact) -> UNIT TEST (fresh subprocess)
    -> GENERALIZATION TEST (validation split, paired)
    -> HELD-OUT + TRANSFER + REGRESSION + REPRODUCIBILITY + LEAKAGE (independent reviewer)
    -> ACCEPT / REJECT -> STORE (registry + memory, failures included) -> RETRY -> MEASURE

What is designer-provided and what is acquired, stated plainly: the expression
language, the search procedure, the failure detectors and the gates are written
by the designers. Which structures to build, from which failures, and whether
they generalize is decided by this loop and the reviewer. No capability, law name
or task label is given to it; it never sees the evaluator's grades on training
tasks, and it never touches held-out data.
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..memory import MemoryStore
from ..science.hypotheses import FitError, FlexibleHypothesis
from ..science.synth import (Abstraction, Model, Shape, UNARY, X, base_shapes, fit_model, prod, synthesize,
                             unary)
from ..science.toolkit import load_artifact
from .artifacts import build_artifact, run_unit_tests, static_check
from .diagnosis import diagnose
from .registry import CapabilityRecord, Registry

# Post-hoc only: which structural features a diagnosis predicts. Used to measure
# diagnosis accuracy after the fact; never used to steer the search.
MODE_FEATURES = {
    "modulated_oscillation": lambda k: "P(" in k and "S(" in k,
    "added_oscillation": lambda k: "S(" in k and k.count("+c*") >= 2,
    "localized_feature": lambda k: "E(Q(" in k or "I(Q(" in k,
    "transition": lambda k: "I(E(" in k,
}


@dataclass
class AcquisitionSettings:
    n_train: int = 150
    n_val: int = 120
    k_min: int = 2
    max_rounds: int = 6
    max_reviews_per_round: int = 2
    max_candidates_per_round: int = 4
    max_research_tasks: int = 60
    synth_max_size: int = 3
    cv_ratio: float = 0.8
    patience: int = 2


@dataclass
class Finding:
    task_id: str
    mode: str
    model: Model
    params: list[float]
    cv_ratio: float


class AcquisitionLoop:
    def __init__(self, service, reviewer, registry: Registry, base_config: dict, *, log_path: str | Path,
                 memory: MemoryStore | None = None, settings: AcquisitionSettings | None = None,
                 seed: int = 0, run_id: str = "acq") -> None:
        self.service = service
        self.reviewer = reviewer
        self.registry = registry
        self.base_config = base_config
        self.memory = memory
        self.s = settings or AcquisitionSettings()
        self.rng = random.Random(seed)
        self.run_id = run_id
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.attempted: set[str] = set()
        self.report: dict = {"rounds": [], "reviews": 0, "accepted": [], "rejected": []}

    # -- bookkeeping ---------------------------------------------------------
    def _log(self, event: str, **data) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "run": self.run_id, "event": event, **data}, default=str) + "\n")

    def config(self) -> dict:
        return {"genome": self.base_config["genome"],
                "capabilities": list(self.base_config.get("capabilities", [])) + self.registry.active_artifacts()}

    def _abstractions(self, config: dict) -> list[Abstraction]:
        blocks = []
        for art in config["capabilities"]:
            _, b = load_artifact(art)
            blocks.extend(b)
        return blocks

    # -- main loop -------------------------------------------------------------
    def run(self) -> dict:
        t0 = time.time()
        config = self.config()
        outs = self.service.solve_training(config, self.s.n_train)
        self._log("solve", n=len(outs), config_capabilities=len(config["capabilities"]))
        no_accept = 0
        for rnd in range(1, self.s.max_rounds + 1):
            if no_accept >= self.s.patience:
                self._log("stop", reason=f"no acceptance in {no_accept} consecutive rounds")
                break
            rinfo = {"round": rnd}
            active_ids = [a["capability_id"] for a in config["capabilities"]]
            diags = [d for d in (diagnose(o["task_id"], o, active_ids) for o in outs) if d is not None]
            rinfo["failures"] = len(diags)
            rinfo["failure_modes"] = Counter(d.mode for d in diags)
            for d in diags:
                self._log("diagnosis", round=rnd, **d.to_dict())
            if not diags:
                self._log("stop", reason="no self-detected failures")
                break
            hyps = self._gap_hypotheses(diags)
            rinfo["acquisition_hypotheses"] = hyps
            self._log("acquisition_hypotheses", round=rnd, hypotheses=hyps)

            findings = self._research(diags, outs, config, rnd)
            rinfo["researched"] = len(findings["researched"])
            rinfo["found"] = len(findings["found"])
            candidates = self._design(findings["found"], config)
            rinfo["candidates"] = [c["structure"] for c in candidates]
            self._log("design", round=rnd, candidates=[{k: v for k, v in c.items() if k != "model"}
                                                       for c in candidates])
            accepted = None
            reviewed = 0
            scored = []
            for cand in candidates[: self.s.max_candidates_per_round]:
                rec = self._implement(cand, diags, rnd)
                if rec is None:
                    continue
                val = self._validate(rec, config)
                if val is not None:
                    scored.append((val["mean_delta"], rec, val))
            scored.sort(key=lambda t: -t[0])
            for delta, rec, val in scored:
                if delta <= 0:
                    self._reject(rec, f"no validation gain (mean delta {delta:+.4f})", stage="validation")
                    continue
                if reviewed >= self.s.max_reviews_per_round or accepted is not None:
                    self._reject(rec, "not reviewed this round (review budget); remains unproven",
                                 stage="budget", final=False)
                    continue
                reviewed += 1
                decision = self._review(rec, config, val)
                if decision.approved:
                    accepted = rec
                    config = self.config()
                else:
                    pass
            rinfo["reviews"] = reviewed
            if accepted is not None:
                no_accept = 0
                before = sum(1 for d in diags)
                outs = self.service.solve_training(config, self.s.n_train)
                after = [d for d in (diagnose(o["task_id"], o, [a["capability_id"] for a in config["capabilities"]])
                                     for o in outs) if d is not None]
                retry = {"self_detected_failures_before": before, "after": len(after),
                         "resolved": before - len(after)}
                self.registry.add_evidence(accepted.capability_id, accepted.version, "retry_evidence", retry)
                rinfo["retry"] = retry
                self._log("retry", round=rnd, capability=accepted.capability_id, **retry)
            else:
                no_accept += 1
            rinfo["accepted"] = accepted.capability_id if accepted else None
            self.report["rounds"].append(rinfo)
        self.report["seconds"] = round(time.time() - t0, 1)
        self.report["registry"] = self.registry.summary()
        self.report["diagnosis_consistency"] = self._diagnosis_consistency()
        self._log("done", **{k: v for k, v in self.report.items() if k != "rounds"})
        return self.report

    # -- steps -----------------------------------------------------------------
    @staticmethod
    def _gap_hypotheses(diags) -> list[dict]:
        by = defaultdict(list)
        for d in diags:
            by[d.mode].append(d)
        hyps = []
        for mode, ds in by.items():
            conf = sum(d.confidence for d in ds) / len(ds)
            hyps.append({"mode": mode, "failures": len(ds), "mean_confidence": round(conf, 3),
                         "hypothesis": ds[0].missing_capability_hypothesis,
                         "priority": round(len(ds) * conf, 3),
                         "experiment": ds[0].proposed_next_experiment})
        return sorted(hyps, key=lambda h: -h["priority"])

    def _research(self, diags, outs, config, rnd) -> dict:
        """Program search on each failed task's observations (most promising modes first)."""
        by_id = {o["task_id"]: o for o in outs}
        order = sorted(diags, key=lambda d: -d.confidence)
        prio = {h["mode"]: i for i, h in enumerate(self._gap_hypotheses(diags))}
        order.sort(key=lambda d: prio[d.mode])
        library = self._abstractions(config)
        robust = config["genome"].get("likelihood") == "student_t"
        researched, found = [], []
        for d in order[: self.s.max_research_tasks]:
            obs = by_id[d.task].get("observations") or []
            xs = [o[0] for o in obs]
            ys = [o[1] for o in obs]
            researched.append(d.task)
            try:
                cands = synthesize(xs, ys, library=library, max_size=self.s.synth_max_size, robust=robust,
                                   seed=self.rng.randrange(1 << 30), top_k=5)
            except Exception as e:  # noqa: BLE001 - research failures are data
                self._log("research_error", round=rnd, task=d.task, error=str(e))
                continue
            best = None
            for c in cands:
                if c.model.size == 0:
                    continue
                ratio = self._cv_ratio(c.model, c.fit.params, xs, ys, robust)
                if ratio is not None and ratio <= self.s.cv_ratio:
                    best = Finding(d.task, d.mode, c.model, list(c.fit.params), ratio)
                    break
            self._log("research", round=rnd, task=d.task, mode=d.mode,
                      top=[c.model.key for c in cands[:3]],
                      kept=best.model.key if best else None, cv_ratio=best.cv_ratio if best else None)
            if best:
                found.append(best)
        return {"researched": researched, "found": found}

    @staticmethod
    def _cv_ratio(model: Model, params, xs, ys, robust) -> float | None:
        """Held-out MSE of the law relative to the flexible description (fit on 2/3, test on 1/3)."""
        idx = sorted(range(len(xs)), key=lambda i: xs[i])
        test = set(idx[1::3])
        tr = [i for i in range(len(xs)) if i not in test]
        te = sorted(test)
        if len(tr) < model.n_params + 2 or len(te) < 3:
            return None
        xtr, ytr = [xs[i] for i in tr], [ys[i] for i in tr]
        try:
            f = fit_model(model, xtr, ytr, init=params, robust=robust, max_starts=2)
            flex = FlexibleHypothesis().fit(xtr, ytr, robust=robust)
        except (FitError, ValueError, ZeroDivisionError, OverflowError):
            return None
        pf = f.predictor()

        def mse(pred):
            vals = [(pred(xs[i]) - ys[i]) ** 2 for i in te]
            vals = [v if math.isfinite(v) else 1e30 for v in vals]
            return sorted(vals)[len(vals) // 2] if robust else sum(vals) / len(vals)

        m_law, m_flex = mse(pf), mse(flex.predict)
        return m_law / m_flex if m_flex > 0 else None

    def _design(self, found: list[Finding], config) -> list[dict]:
        groups = defaultdict(list)
        for f in found:
            groups[f.model.key].append(f)
        existing = {t.get("structure") for a in config["capabilities"] for t in a["terms"]}
        existing |= {a.get("structure") for a in config["capabilities"]}
        cands = []
        for key, fs in groups.items():
            if len(fs) < self.s.k_min or key in self.attempted or key in existing:
                continue
            fs.sort(key=lambda f: f.cv_ratio)
            modes = Counter(f.mode for f in fs)
            cands.append({"structure": key, "model": fs[0].model, "support": len(fs),
                          "tasks": [f.task_id for f in fs], "modes": dict(modes),
                          "inits": [f.params for f in fs[:5]],
                          "mean_cv_ratio": round(sum(f.cv_ratio for f in fs) / len(fs), 4)})
        cands.sort(key=lambda c: (-c["support"], c["mean_cv_ratio"]))
        return cands

    def _implement(self, cand: dict, diags, rnd) -> CapabilityRecord | None:
        self.attempted.add(cand["structure"])
        cid = self.registry.new_id()
        model: Model = cand["model"]
        deps = sorted({s.lib.name.split(".")[0] for sh in model.shapes for s in sh.subshapes() if s.op == "LIB"})
        sources = [sh.source(0)[0] for sh in model.shapes]
        top_mode = max(cand["modes"], key=cand["modes"].get)
        art = build_artifact(cid, 1, model, cand["inits"], name=f"law shape {cand['structure']}",
                             description=(f"Hypothesis family y = c0 + sum_i c_i * term_i with terms {sources}. "
                                          f"Constructed by program search from {cand['support']} failed training "
                                          f"tasks (diagnosed modes: {cand['modes']})."))
        diag_example = next((d.to_dict() for d in diags if d.task == cand["tasks"][0]), {})
        rec = CapabilityRecord(
            capability_id=cid, version=1, name=art["name"], description=art["description"],
            origin=f"acquired:{self.run_id}:round{rnd}",
            triggering_failure={"task_ids": cand["tasks"], "modes": cand["modes"], "example": diag_example},
            hypothesis=next((d.missing_capability_hypothesis for d in diags if d.mode == top_mode), ""),
            artifact=art, prerequisites=["program-search research tool", "flexible-baseline failure detector"],
            dependencies=deps,
            limitations=["validated only on 1-D noisy law-discovery tasks"])
        self.registry.propose(rec)
        errs = static_check(art)
        tests = run_unit_tests(art) if not errs else {"passed": False, "static_errors": errs}
        tests["static_errors"] = errs
        self.registry.add_evidence(cid, 1, "tests", tests)
        self._log("implement", round=rnd, capability=cid, structure=cand["structure"], tests=tests,
                  support=cand["support"], sha256=art["sha256"])
        if not tests.get("passed"):
            self._reject(rec, f"unit tests failed: {tests.get('error') or tests}", stage="unit_tests")
            return None
        return rec

    def _validate(self, rec: CapabilityRecord, config) -> dict | None:
        cand = {"genome": config["genome"], "capabilities": config["capabilities"] + [rec.artifact]}
        val = self.service.validate(config, cand, self.s.n_val, block=0)
        summary = {k: val[k] for k in ("validation_id", "mean_delta", "ci_low", "ci_high")}
        self.registry.add_evidence(rec.capability_id, rec.version, "selection_evidence", summary)
        self._log("validate", capability=rec.capability_id, **summary)
        return val

    def _review(self, rec: CapabilityRecord, config, val):
        cand = {"genome": config["genome"], "capabilities": config["capabilities"] + [rec.artifact]}
        decision = self.reviewer.review(config, cand, rec.artifact, val["validation_id"],
                                        set(self.service.seen_seeds), unit_test_runner=run_unit_tests)
        self.report["reviews"] += 1
        d = decision.to_dict()
        self.registry.add_evidence(rec.capability_id, rec.version, "review",
                                   {"review_id": d["review_id"], "approved": d["approved"], "gates": d["gates"],
                                    "reasons": d["reasons"], "gates_sha": d["gates_sha"]})
        rc = d.get("recomputed", {})
        if rc.get("heldout"):
            self.registry.add_evidence(rec.capability_id, rec.version, "heldout_evidence",
                                       {"gate": rc["heldout"], "familiar": rc.get("familiar"),
                                        "by_kind": rc.get("heldout_by_kind")})
        if rc.get("transfer"):
            self.registry.add_evidence(rec.capability_id, rec.version, "transfer_evidence", rc["transfer"])
        self._log("review", capability=rec.capability_id, approved=d["approved"], reasons=d["reasons"],
                  review_id=d["review_id"])
        if decision.approved:
            conf = 1.0 - 0.5 * (1 - min(1.0, rc["heldout"]["ci_low"] / max(1e-9, rc["heldout"]["mean_delta"])))
            self.registry.set_status(rec.capability_id, rec.version, "ACTIVE", "approved by independent review",
                                     review={"review_id": d["review_id"], "approved": True},
                                     confidence=round(max(0.5, min(0.99, conf)), 3))
            self.report["accepted"].append(rec.capability_id)
            self._remember(rec, True, d)
        else:
            self._reject(rec, "; ".join(d["reasons"]), stage="review", review=d)
        return decision

    def _reject(self, rec: CapabilityRecord, reason: str, *, stage: str, review: dict | None = None,
                final: bool = True) -> None:
        if final:
            self.registry.set_status(rec.capability_id, rec.version, "REJECTED", f"{stage}: {reason}",
                                     review=review, limitations=[f"rejected at {stage}"])
            self.report["rejected"].append({"capability": rec.capability_id, "stage": stage,
                                            "structure": rec.artifact.get("structure"), "reason": reason[:300]})
            self._remember(rec, False, {"reasons": [reason], "stage": stage})
        else:
            self._log("deferred", capability=rec.capability_id, reason=reason)

    def _remember(self, rec: CapabilityRecord, success: bool, decision: dict) -> None:
        if self.memory is None:
            return
        structure = rec.artifact.get("structure")
        modes = rec.triggering_failure.get("modes")
        if success:
            pid = self.memory.add_procedure(
                f"capability {rec.capability_id}: {structure}",
                f"{rec.description} Motivating hypothesis: {rec.hypothesis} "
                f"Accepted by review {decision.get('review_id')}.",
                meta={"capability_id": rec.capability_id, "modes": modes}, run_id=self.run_id)
            self.memory.record_outcome(pid, True)      # verified by the independent reviewer
        else:
            self.memory.add_knowledge(
                "capability acquisition",
                f"Candidate {structure} (from failure modes {modes}) was rejected at "
                f"{decision.get('stage', 'review')}: {'; '.join(decision.get('reasons', []))[:500]}",
                status="fact", confidence=1.0, verified=True, run_id=self.run_id,
                meta={"capability_id": rec.capability_id, "structure": structure, "outcome": "rejected"})

    def _diagnosis_consistency(self) -> dict:
        """Did the diagnosed failure mode predict the structure the research found?"""
        hits, total = 0, 0
        per_mode = defaultdict(lambda: [0, 0])
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e.get("event") != "research" or not e.get("kept") or e.get("run") != self.run_id:
                continue
            check = MODE_FEATURES.get(e["mode"])
            if check is None:
                continue
            ok = check(e["kept"])
            hits += ok
            total += 1
            per_mode[e["mode"]][0] += ok
            per_mode[e["mode"]][1] += 1
        return {"checked": total, "consistent": hits,
                "rate": round(hits / total, 3) if total else None,
                "per_mode": {m: {"consistent": a, "checked": b} for m, (a, b) in per_mode.items()}}


# ---------------------------------------------------------------------------
# Control: random capabilities of matched size, through the same pipeline
# ---------------------------------------------------------------------------

def random_shape(size: int, rng: random.Random) -> Shape:
    if size == 0:
        return X
    if size >= 2 and rng.random() < 0.35:
        left = rng.randint(0, size - 1)
        return prod(random_shape(left, rng), random_shape(size - 1 - left, rng))
    return unary(rng.choice(UNARY), random_shape(size - 1, rng))


def random_model(size: int, rng: random.Random) -> Model:
    if size >= 1 and rng.random() < 0.3:
        a = rng.randint(0, size)
        return Model((random_shape(a, rng), random_shape(max(0, size - a), rng) if size - a else
                      rng.choice(base_shapes()[1:6])))
    return Model((random_shape(max(1, size), rng),))
