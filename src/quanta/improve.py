"""Benchmark-gated self-improvement.

    diagnose -> propose -> evaluate (paired, same tasks) -> statistical gate
             -> replicate on fresh tasks -> authorize -> adopt
    ... until improvements stop coming (patience), then report on a held-out
    split that played no part in any decision.

Safeguards against fooling ourselves:

* Paired evaluation: candidate and champion run on identical tasks/seeds.
* Gate: the lower bound of a bootstrap CI on the mean paired difference must
  exceed `min_effect`, and no task category may regress beyond a tolerance.
* Replication: anything that passes the gate on the selection split must pass
  it again on *fresh* confirmation tasks (new ones every generation), which
  counters selection bias from trying many candidates.
* Held-out test split: used only for the final report.

Safeguards against the system weakening its own oversight:

* Code patches may only touch MUTABLE_PATHS; PROTECTED_PATHS (governance,
  this file, the benchmark/grader, ledger rules, tool risk classification,
  the genome bounds and the invariant tests) can never be modified, and
  their hashes are re-checked in the sandbox.
* Every patch candidate must pass the governance invariant tests inside its
  sandbox before it is benchmarked.
* Adopting a genome (bounded, reversible) is a LOW-risk action; adopting a
  code patch is HIGH-risk and needs human authorization through the Governor.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .bench.harness import SuiteResult, diagnose, run_suite
from .epistemics import Ledger, Status
from .genome import SPACE, Genome, GenomeError, clamp_to_space
from .governance import Action, Governor, Risk
from .report import render_report
from .stats import PairedComparison, paired_bootstrap

REPO_ROOT = Path(__file__).resolve().parents[2]

PROTECTED_PATHS = (
    "src/quanta/governance.py", "src/quanta/improve.py", "src/quanta/bench/", "src/quanta/epistemics.py",
    "src/quanta/tools.py", "src/quanta/genome.py", "src/quanta/stats.py",
    "tests/test_governance_invariants.py",
)
MUTABLE_PATHS = ("src/quanta/science/",)
DISCOVERY_PARAMS = ("n_initial", "design", "n_candidates", "explore_prob", "likelihood", "t_dof",
                    "stop_posterior", "min_experiments", "patience", "falsification_rounds", "temperature",
                    "flexible_baseline", "expand_on_inadequacy", "use_learned_priors", "prior_pseudocount")


@dataclass
class Candidate:
    id: str
    genome: Genome
    parent: str | None = None
    patch: dict[str, str] | None = None
    rationale: str = ""
    proposer: str = "initial"

    def key(self) -> str:
        h = hashlib.sha256(self.genome.fingerprint().encode())
        for path, content in sorted((self.patch or {}).items()):
            h.update(path.encode() + b"\0" + content.encode())
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Proposers
# ---------------------------------------------------------------------------

class Proposer(Protocol):
    name: str

    def propose(self, genome: Genome, diagnostics: dict, history: list[dict], rng: random.Random,
                n: int) -> list[tuple[dict, str, dict | None]]: ...


def active_params(g: Genome) -> list[str]:
    """Discovery parameters that can change behaviour given the rest of the genome
    (changing an inert one wastes an evaluation and can only add noise)."""
    inert = set()
    if not g.use_learned_priors:
        inert.add("prior_pseudocount")
    if g.likelihood != "student_t":
        inert.add("t_dof")
    if not g.flexible_baseline:
        inert.add("expand_on_inadequacy")
    if g.design != "disagreement":
        inert.add("explore_prob")
    if g.design == "random" and g.falsification_rounds == 0:
        inert.add("n_candidates")
    return [p for p in DISCOVERY_PARAMS if p not in inert]


def _random_mutation(genome: Genome, rng: random.Random, params=None) -> tuple[dict, str]:
    name = rng.choice(params or active_params(genome))
    spec = SPACE[name]
    cur = getattr(genome, name)
    if spec.kind == "bool":
        val = not cur
    elif spec.kind == "choice":
        val = rng.choice([c for c in spec.choices if c != cur])
    elif spec.kind == "int":
        step = max(1, round((spec.high - spec.low) * 0.15))
        val = clamp_to_space(name, cur + rng.choice([-step, step]))
    else:
        val = clamp_to_space(name, cur * math.exp(rng.gauss(0, 0.3)) if cur else rng.uniform(spec.low, spec.high))
        val = round(val, 4)
    return {name: val}, f"exploration: random local change to {name}"


class DiagnosticProposer:
    """Reads failure diagnostics and proposes targeted strategy changes.

    The rules encode general research methodology (test before you trust,
    be robust to bad measurements, notice when your hypotheses are all
    wrong, match confidence to accuracy); they know nothing about which
    laws the benchmark contains.
    """

    name = "diagnostic"

    def propose(self, genome, diagnostics, history, rng, n):
        d, g = diagnostics, genome
        acc = d.get("accuracy_by_category", {})
        clean = acc.get("clean", d["accuracy"])
        out: list[tuple[dict, str, dict | None]] = []
        if d["overconfident_error_rate"] > 0.05:
            if g.falsification_rounds < 8:
                out.append(({"falsification_rounds": min(8, g.falsification_rounds + 2)},
                            f"{d['overconfident_error_rate']:.0%} of answers were confidently wrong: test the "
                            "leading hypothesis adversarially before accepting it", None))
            if g.stop_posterior < 0.99:
                out.append(({"stop_posterior": round(min(0.995, g.stop_posterior + (1 - g.stop_posterior) / 2), 4)},
                            "confident errors: require stronger evidence before stopping", None))
        gap = d["mean_credence"] - d["accuracy"]
        if d["ece"] > 0.08 and gap > 0.03:
            out.append(({"temperature": round(clamp_to_space("temperature", g.temperature * 1.3), 4)},
                        f"stated credence exceeds accuracy by {gap:.2f}: temper the posterior", None))
        if d["ece"] > 0.08 and gap < -0.03:
            out.append(({"temperature": round(clamp_to_space("temperature", g.temperature / 1.3), 4)},
                        f"accuracy exceeds stated credence by {-gap:.2f}: sharpen the posterior", None))
        if "outliers" in acc and acc["outliers"] < clean - 0.05 and g.likelihood == "gaussian":
            out.append(({"likelihood": "student_t"},
                        f"accuracy with corrupted measurements ({acc['outliers']:.2f}) trails clean data "
                        f"({clean:.2f}): use a heavy-tailed noise model", None))
        outside = acc.get("outside-initial-space")
        if outside is not None and outside < 0.5 and not g.flexible_baseline:
            out.append(({"flexible_baseline": True, "expand_on_inadequacy": True},
                        f"accuracy on laws outside the initial hypothesis space is {outside:.2f}: compete named "
                        "laws against a flexible baseline and expand the space when they all fail", None))
            out.append(({"flexible_baseline": True, "expand_on_inadequacy": True, "likelihood": "student_t"},
                        "as above; a flexible curve can also absorb outliers, so pair it with a robust noise "
                        "model", None))
        if d["budget_exhausted_rate"] > 0.3 and g.design != "disagreement":
            out.append(({"design": "disagreement"},
                        f"{d['budget_exhausted_rate']:.0%} of runs exhausted the budget: run the experiments "
                        "where competing hypotheses disagree most", None))
        if d["mean_cost"] > 0.6 and d["overconfident_error_rate"] < 0.05 and g.patience == 0:
            out.append(({"patience": 4}, "high cost with few confident errors: stop when progress stalls", None))
        if not g.use_learned_priors:
            out.append(({"use_learned_priors": True}, "use verified outcomes of past tasks as priors", None))
        # Near misses: changes that helped on average but did not clear the gate
        # may be complementary; try them together.
        near = sorted((h for h in history[-2 * max(n, 1):]
                       if h.get("changes") and not h.get("passed_gate", True) and h.get("mean_delta", 0) > 0),
                      key=lambda h: -h["mean_delta"])[:3]
        combos = []
        for i in range(len(near)):
            for j in range(i + 1, len(near)):
                a, b = near[i]["changes"], near[j]["changes"]
                if set(a) & set(b):
                    continue
                combos.append(({**a, **b}, f"combine two near-miss changes (mean Δ {near[i]['mean_delta']:+.4f} "
                                           f"and {near[j]['mean_delta']:+.4f}) that may be complementary", None))
        out = out[:3] + combos + out[3:]
        # De-duplicate by resulting genome: a change that failed under an earlier
        # champion is worth retrying under a new one (interaction effects).
        tried = {h["genome"] for h in history} | {genome.fingerprint()}

        def novel(changes: dict) -> bool:
            try:
                fp = genome.mutate(**changes).fingerprint()
            except GenomeError:
                return False
            if fp in tried:
                return False
            tried.add(fp)
            return True

        fresh = [p for p in out if novel(p[0])]
        for _ in range(50 * n):
            if len(fresh) >= n:
                break
            changes, why = _random_mutation(genome, rng)
            if novel(changes):
                fresh.append((changes, why, None))
        return fresh[:n]


class RandomProposer:
    name = "random"

    def propose(self, genome, diagnostics, history, rng, n):
        return [(*_random_mutation(genome, rng), None) for _ in range(n)]


class LLMProposer:
    """A model reads the diagnostics and lineage and proposes changes (and,
    optionally, code patches confined to MUTABLE_PATHS)."""

    name = "llm"

    def __init__(self, model, *, allow_patches: bool = False, repo_root: Path = REPO_ROOT) -> None:
        self.model = model
        self.allow_patches = allow_patches
        self.repo_root = repo_root

    def propose(self, genome, diagnostics, history, rng, n):
        from .agent import Agent, submit_tool
        from .tools import ToolContext, ToolRegistry

        space = {k: {"kind": v.kind, "low": v.low, "high": v.high, "choices": v.choices, "doc": v.doc}
                 for k, v in SPACE.items() if k in DISCOVERY_PARAMS}

        def validate(a):
            errs = []
            for p in a.get("proposals", []):
                for k, v in (p.get("changes") or {}).items():
                    if k not in space:
                        errs.append(f"{k} is not a tunable parameter")
                        continue
                    try:
                        clamp_to_space(k, v)
                    except (GenomeError, TypeError, ValueError) as e:
                        errs.append(str(e))
                for path in (p.get("patch") or {}):
                    if not self.allow_patches:
                        errs.append("code patches are not enabled")
                    elif not patch_path_allowed(path):
                        errs.append(f"{path} is not a mutable path")
            if not a.get("proposals"):
                errs.append("submit at least one proposal")
            return errs

        props = {"changes": {"type": "object"}, "rationale": {"type": "string"}}
        if self.allow_patches:
            props["patch"] = {"type": "object"}
        tool = submit_tool("submit_proposals", "Submit improvement proposals.", {
            "proposals": {"type": "array", "items": {"type": "object", "properties": props,
                                                     "required": ["changes", "rationale"]}}},
            ["proposals"], validate)
        with tempfile.TemporaryDirectory() as tmp:
            ctx = ToolContext(Path(tmp), Governor(), Ledger())
            agent = Agent(model=self.model, registry=ToolRegistry(), ctx=ctx, role="scientist", submit=tool,
                          tool_names=[], max_steps=6)
            recent = history[-15:]
            res = agent.run(
                "You are improving the strategy of an automated scientific-discovery loop.\n\n"
                f"Tunable parameters:\n{json.dumps(space, indent=1)}\n\nCurrent genome:\n"
                f"{json.dumps({k: getattr(genome, k) for k in space}, indent=1)}\n\n"
                f"Failure diagnostics on the selection split:\n{json.dumps(diagnostics, indent=1, default=str)}\n\n"
                f"Previously tried changes and their measured effect:\n{json.dumps(recent, indent=1, default=str)}\n\n"
                f"Propose {n} changes most likely to raise the mean score. Explain the causal reasoning for each. "
                "Prefer changes that address the largest diagnosed failure mode. Do not repeat failed changes.")
        out = []
        for p in (res.submitted or {}).get("proposals", [])[:n]:
            changes = {k: clamp_to_space(k, v) for k, v in p["changes"].items()}
            out.append((changes, p["rationale"], p.get("patch") if self.allow_patches else None))
        return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def patch_path_allowed(path: str) -> bool:
    norm = os.path.normpath(path).replace(os.sep, "/")
    if norm.startswith("../") or norm.startswith("/"):
        return False
    if any(norm == p.rstrip("/") or norm.startswith(p) for p in PROTECTED_PATHS):
        return False
    return any(norm.startswith(p) for p in MUTABLE_PATHS)


def _hash_paths(root: Path, paths) -> dict[str, str]:
    out = {}
    for p in paths:
        full = root / p
        files = sorted(full.rglob("*.py")) if full.is_dir() else ([full] if full.exists() else [])
        for f in files:
            out[str(f.relative_to(root))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


class EvaluationError(RuntimeError):
    pass


class Evaluator:
    def __init__(self, *, jobs: int = 4, repo_root: Path = REPO_ROOT, timeout: int = 1800,
                 experience_n: int = 60) -> None:
        self.jobs = jobs
        self.repo_root = repo_root
        self.timeout = timeout
        self.experience_n = experience_n
        self._cache: dict[tuple, SuiteResult] = {}

    def evaluate(self, cand: Candidate, split: str, n: int, offset: int = 0) -> SuiteResult:
        key = (cand.key(), split, n, offset)
        if key not in self._cache:
            if cand.patch:
                self._cache[key] = self._evaluate_sandboxed(cand, split, n, offset)
            else:
                self._cache[key] = run_suite(cand.genome, split, n, jobs=self.jobs, offset=offset,
                                             experience_n=self.experience_n)
        return self._cache[key]

    def _evaluate_sandboxed(self, cand: Candidate, split: str, n: int, offset: int) -> SuiteResult:
        for path in cand.patch:
            if not patch_path_allowed(path):
                raise EvaluationError(f"patch touches a non-mutable path: {path}")
        with tempfile.TemporaryDirectory(prefix="quanta-sandbox-") as tmp:
            sb = Path(tmp)
            shutil.copytree(self.repo_root / "src", sb / "src", ignore=shutil.ignore_patterns("__pycache__"))
            (sb / "tests").mkdir()
            shutil.copy2(self.repo_root / "tests" / "test_governance_invariants.py", sb / "tests")
            before = _hash_paths(sb, PROTECTED_PATHS)
            for path, content in cand.patch.items():
                target = sb / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            if _hash_paths(sb, PROTECTED_PATHS) != before:
                raise EvaluationError("protected files changed in the sandbox")
            env = {**os.environ, "PYTHONPATH": str(sb / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
            inv = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                  "tests/test_governance_invariants.py"], cwd=sb, env=env,
                                 capture_output=True, text=True, timeout=self.timeout)
            if inv.returncode != 0:
                raise EvaluationError("governance invariants failed in sandbox:\n" + inv.stdout[-2000:])
            gpath, rpath = sb / "genome.json", sb / "result.json"
            cand.genome.save(gpath)
            run = subprocess.run([sys.executable, "-m", "quanta.bench.harness", "--genome", str(gpath),
                                  "--split", split, "--n", str(n), "--offset", str(offset),
                                  "--jobs", str(self.jobs), "--out", str(rpath)],
                                 cwd=sb, env=env, capture_output=True, text=True, timeout=self.timeout)
            if run.returncode != 0 or not rpath.exists():
                raise EvaluationError("benchmark failed in sandbox:\n" + (run.stderr or run.stdout)[-3000:])
            data = json.loads(rpath.read_text())
        results = data["results"]
        return SuiteResult(cand.genome, split, results, diagnose(results))


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

@dataclass
class GateConfig:
    min_effect: float = 0.0
    confidence: float = 0.95
    max_category_regression: float = 0.10
    min_category_size: int = 8
    n_boot: int = 4000


@dataclass
class GateDecision:
    accept: bool
    comparison: PairedComparison
    reasons: list[str]

    def to_dict(self) -> dict:
        return {"accept": self.accept, "reasons": self.reasons, **self.comparison.to_dict()}


def gate(base: SuiteResult, cand: SuiteResult, cfg: GateConfig = GateConfig(), seed: int = 0) -> GateDecision:
    if [r["seed"] for r in base.results] != [r["seed"] for r in cand.results]:
        raise ValueError("gate requires paired results on identical tasks")
    cmp = paired_bootstrap(base.scores, cand.scores, n_boot=cfg.n_boot, confidence=cfg.confidence, seed=seed)
    reasons = []
    if cmp.ci_low <= cfg.min_effect:
        reasons.append(f"CI lower bound {cmp.ci_low:+.4f} does not exceed {cfg.min_effect:+.4f}")
    counts = base.diagnostics.get("count_by_category", {})
    for cat, a in base.diagnostics.get("accuracy_by_category", {}).items():
        b = cand.diagnostics.get("accuracy_by_category", {}).get(cat, a)
        if counts.get(cat, 0) >= cfg.min_category_size and b < a - cfg.max_category_regression:
            reasons.append(f"accuracy on '{cat}' regressed {a:.2f} -> {b:.2f}")
    return GateDecision(not reasons, cmp, reasons or ["significant improvement, no category regression"])


# ---------------------------------------------------------------------------
# The improvement loop
# ---------------------------------------------------------------------------

@dataclass
class ImprovementResult:
    initial: Genome
    final: Genome
    generations_run: int
    adopted: list[dict]
    test_initial: SuiteResult
    test_final: SuiteResult
    test_comparison: PairedComparison
    stop_reason: str
    run_dir: Path
    lineage: list[dict] = field(default_factory=list)


class SelfImprover:
    def __init__(self, evaluator: Evaluator, proposers: list, *, run_dir: str | Path,
                 gate_config: GateConfig | None = None, generations: int = 8, candidates_per_generation: int = 6,
                 confirm_top: int = 2, patience: int = 3, n_selection: int = 200, n_confirmation: int = 200,
                 n_test: int = 400, seed: int = 0, governor: Governor | None = None) -> None:
        self.evaluator = evaluator
        self.proposers = proposers
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "genomes").mkdir(exist_ok=True)
        self.cfg = gate_config or GateConfig()
        self.generations = generations
        self.k = candidates_per_generation
        self.confirm_top = confirm_top
        self.patience = patience
        self.n_sel, self.n_conf, self.n_test = n_selection, n_confirmation, n_test
        self.rng = random.Random(seed)
        self.seed = seed
        self.governor = governor or Governor()
        self.lineage: list[dict] = []
        self.history: list[dict] = []

    def _record(self, entry: dict) -> None:
        entry = {"ts": time.time(), **entry}
        self.lineage.append(entry)
        with (self.run_dir / "lineage.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def _authorize(self, cand: Candidate) -> bool:
        risk = Risk.HIGH if cand.patch else Risk.LOW
        what = "adopt code patch to " + ", ".join(cand.patch) if cand.patch else "adopt strategy genome"
        decision = self.governor.authorize(Action("self_improve.adopt", {"candidate": cand.id,
                                                                          "diff": cand.rationale[:300]},
                                                  risk, what, "self-improver"))
        return decision.allowed

    def run(self, initial: Genome) -> ImprovementResult:
        start = Candidate("v0", initial)
        champion = start
        base = self.evaluator.evaluate(champion, "selection", self.n_sel)
        initial.save(self.run_dir / "genomes" / "v0.json")
        self._record({"event": "baseline", "candidate": "v0", "mean_score": base.mean_score,
                      "diagnostics": base.diagnostics})
        adopted, stall, version, stop = [], 0, 0, "generation limit reached"
        gen = 0
        for gen in range(1, self.generations + 1):
            proposals: list[Candidate] = []
            seen = {champion.key()}
            for prop in self.proposers:
                for changes, why, patch in prop.propose(champion.genome, base.diagnostics, self.history,
                                                        self.rng, self.k):
                    try:
                        genome = champion.genome.mutate(**changes)
                    except GenomeError:
                        continue
                    c = Candidate(f"g{gen}c{len(proposals) + 1}", genome, champion.id, patch, why, prop.name)
                    if c.key() not in seen:
                        seen.add(c.key())
                        proposals.append(c)
            proposals = proposals[: self.k]
            scored = []
            for c in proposals:
                try:
                    res = self.evaluator.evaluate(c, "selection", self.n_sel)
                except EvaluationError as e:
                    self._record({"event": "rejected", "generation": gen, "candidate": c.id, "reason": str(e)[:1000],
                                  "rationale": c.rationale, "proposer": c.proposer})
                    continue
                dec = gate(base, res, self.cfg, seed=self.seed + gen)
                changes = champion.genome.diff(c.genome)
                self.history.append({"changes": {k: v[1] for k, v in changes.items()},
                                     "genome": c.genome.fingerprint(),
                                     "mean_delta": round(dec.comparison.mean_delta, 4),
                                     "passed_gate": dec.accept})
                self._record({"event": "evaluated", "generation": gen, "candidate": c.id, "parent": champion.id,
                              "proposer": c.proposer, "rationale": c.rationale, "changes": changes,
                              "patched": sorted(c.patch or {}), "mean_score": res.mean_score,
                              "selection_gate": dec.to_dict()})
                scored.append((c, res, dec))
            winners = sorted((s for s in scored if s[2].accept), key=lambda s: -s[2].comparison.mean_delta)
            promoted = None
            for c, res, dec in winners[: self.confirm_top]:
                offset = (gen - 1) * self.n_conf
                cb = self.evaluator.evaluate(champion, "confirmation", self.n_conf, offset)
                cc = self.evaluator.evaluate(c, "confirmation", self.n_conf, offset)
                dec2 = gate(cb, cc, self.cfg, seed=self.seed + 1000 + gen)
                self._record({"event": "confirmation", "generation": gen, "candidate": c.id,
                              "confirmation_gate": dec2.to_dict(), "offset": offset})
                if not dec2.accept:
                    continue
                if not self._authorize(c):
                    self._record({"event": "not_authorized", "generation": gen, "candidate": c.id})
                    continue
                promoted = (c, res, dec, dec2)
                break
            if promoted:
                c, res, dec, dec2 = promoted
                version += 1
                c.genome.save(self.run_dir / "genomes" / f"v{version}.json")
                adopted.append({"version": version, "candidate": c.id, "generation": gen,
                                "changes": champion.genome.diff(c.genome), "rationale": c.rationale,
                                "selection_delta": dec.comparison.mean_delta,
                                "confirmation_delta": dec2.comparison.mean_delta,
                                "confirmation_ci": [dec2.comparison.ci_low, dec2.comparison.ci_high]})
                self._record({"event": "adopted", "generation": gen, "candidate": c.id, "version": version})
                champion, base, stall = c, res, 0
            else:
                stall += 1
                if stall >= self.patience:
                    stop = f"diminishing returns: no confirmed improvement in {stall} consecutive generations"
                    break
        test_i = self.evaluator.evaluate(start, "test", self.n_test)
        test_f = self.evaluator.evaluate(champion, "test", self.n_test)
        cmp = paired_bootstrap(test_i.scores, test_f.scores, seed=self.seed + 99)
        result = ImprovementResult(initial, champion.genome, gen, adopted, test_i, test_f, cmp, stop,
                                   self.run_dir, self.lineage)
        champion.genome.save(self.run_dir / "genomes" / "final.json")
        self._write_report(result)
        return result

    # -- reporting ---------------------------------------------------------------
    def _write_report(self, r: ImprovementResult) -> None:
        led = Ledger()
        di, df = r.test_initial.diagnostics, r.test_final.diagnostics
        ev_i = led.add_evidence("execution", f"held-out test run of initial genome ({r.initial.fingerprint()})",
                                {"mean_score": r.test_initial.mean_score, "n": len(r.test_initial.results)})
        ev_f = led.add_evidence("execution", f"held-out test run of final genome ({r.final.fingerprint()})",
                                {"mean_score": r.test_final.mean_score, "n": len(r.test_final.results)})
        c = r.test_comparison
        f1 = led.assert_claim(f"On {c.n} held-out tasks the initial genome scored {r.test_initial.mean_score:.4f} "
                              f"and the final genome {r.test_final.mean_score:.4f} (paired difference "
                              f"{c.mean_delta:+.4f}; {c.wins} wins / {c.losses} losses / {c.ties} ties).",
                              Status.FACT, 1.0, evidence=[ev_i.id, ev_f.id], author="measurement")
        f2 = led.assert_claim(f"Accuracy {di['accuracy']:.3f} -> {df['accuracy']:.3f}; Brier {di['brier']:.3f} -> "
                              f"{df['brier']:.3f}; ECE {di['ece']:.3f} -> {df['ece']:.3f}; confidently-wrong rate "
                              f"{di['overconfident_error_rate']:.3f} -> {df['overconfident_error_rate']:.3f}; mean "
                              f"cost {di['mean_cost']:.3f} -> {df['mean_cost']:.3f}.",
                              Status.FACT, 1.0, evidence=[ev_i.id, ev_f.id], author="measurement")
        boot = led.add_evidence("derivation", "paired bootstrap on held-out per-task scores", c.to_dict())
        conf = 0.95 if c.ci_low > 0 else 0.5
        if c.wins >= c.losses:
            direction = (f"It also scored higher on more tasks than it scored lower ({c.wins} vs {c.losses}; "
                         f"two-sided sign test p={c.sign_p:.2g}).")
        else:
            direction = (f"However, it scored LOWER on more tasks than it scored higher ({c.losses} vs {c.wins}; "
                         f"two-sided sign test p={c.sign_p:.2g}): the mean gain comes from large improvements on a "
                         "minority of tasks while most tasks pay a small cost.")
        led.assert_claim(f"The final genome has a higher mean score than the initial one on this task distribution "
                         f"(95% CI on the mean improvement: [{c.ci_low:+.4f}, {c.ci_high:+.4f}]). {direction}",
                         Status.INFERENCE, conf, evidence=[boot.id], depends_on=[f1.id, f2.id], author="analysis")
        for a in r.adopted:
            led.assert_claim(f"Adopted change v{a['version']} ({a['changes']}) helped: it passed the gate on the "
                             f"selection split (delta {a['selection_delta']:+.4f}) and replicated on fresh tasks "
                             f"(delta {a['confirmation_delta']:+.4f}). Rationale: {a['rationale']}",
                             Status.INFERENCE, 0.9, depends_on=[f1.id], author="analysis")
        led.assert_claim("The same strategy changes (adversarial falsification, robust likelihoods, a flexible "
                         "'none of the above' baseline, calibrated credence) would improve LLM-driven open-ended "
                         "research. Not tested here.", Status.SPECULATION, 0.4, author="analysis")
        rejected = sum(1 for e in r.lineage if e["event"] == "evaluated" and not e["selection_gate"]["accept"])
        failed_conf = sum(1 for e in r.lineage if e["event"] == "confirmation" and not e["confirmation_gate"]["accept"])
        lines = [f"- v{a['version']} (generation {a['generation']}): `{a['changes']}` — {a['rationale']} "
                 f"(selection Δ {a['selection_delta']:+.4f}, confirmation Δ {a['confirmation_delta']:+.4f}, "
                 f"CI [{a['confirmation_ci'][0]:+.4f}, {a['confirmation_ci'][1]:+.4f}])" for a in r.adopted]
        text = render_report(
            title="Self-improvement report", problem="Improve the strategy of the hypothesis-driven discovery "
            "loop, keeping a change only if it demonstrably performs better.",
            answer=(f"{len(r.adopted)} change(s) adopted over {r.generations_run} generation(s). Held-out mean "
                    f"score {r.test_initial.mean_score:.4f} -> {r.test_final.mean_score:.4f} "
                    f"(Δ {c.mean_delta:+.4f}, 95% CI [{c.ci_low:+.4f}, {c.ci_high:+.4f}])."),
            confidence=None, ledger=led,
            limitations=[
                "The benchmark is a controlled micro-world (1-D black-box law discovery); gains are measured "
                "only on its task distribution.",
                "Changes are to strategy parameters (and optionally sandboxed code in mutable paths); the "
                "improver cannot change the grader, the gate or governance, so it cannot discover improvements "
                "that would require changing them.",
                f"{rejected} candidate(s) failed the selection gate and {failed_conf} failed replication; "
                "a stricter or looser gate would change what is adopted.",
                "Proposals come from fixed diagnostic rules plus random exploration unless an LLM proposer is "
                "used; the search is local and greedy.",
                f"Stopped because: {r.stop_reason}."]
            + ([f"Per-task wins/losses on the held-out split are {c.wins}/{c.losses}: the improvement is in the "
                "mean (the pre-declared objective), not in the typical task."] if c.wins < c.losses else []),
            reproduction={"command": "python -m quanta.cli improve --run-dir <dir>  (see docs/SELF_IMPROVEMENT.md)",
                          "seed": self.seed, "n_selection": self.n_sel, "n_confirmation": self.n_conf,
                          "n_test": self.n_test, "initial_genome": r.initial.fingerprint(),
                          "final_genome": r.final.fingerprint(),
                          "gate": json.dumps(self.cfg.__dict__)},
            extra_sections=[("Adopted changes", "\n".join(lines) or "- none"),
                            ("Final genome changes vs. initial", "```\n" + json.dumps(
                                r.initial.diff(r.final), indent=1, default=str) + "\n```")])
        (self.run_dir / "report.md").write_text(text, encoding="utf-8")
        summary = {"initial": r.initial.to_dict(), "final": r.final.to_dict(), "adopted": r.adopted,
                   "stop_reason": r.stop_reason, "generations_run": r.generations_run,
                   "test": {"initial": di, "final": df, "comparison": c.to_dict()}}
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
