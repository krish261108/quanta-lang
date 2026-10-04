"""The hypothesis-driven experimentation loop, fully automated.

Given an experimental system that can be queried (each query costs budget),
the loop:

1. FRAME      - lists the unknowns and the competing hypotheses (as ledger
                speculation, with priors).
2. RECALL     - optionally sets priors from *verified* past outcomes in memory.
3. EXPERIMENT - space-filling initial design, then adaptive experiments
                chosen where the competing hypotheses disagree most.
4. ANALYZE    - fits every hypothesis, compares them by BIC, forms a
                (tempered) posterior.
5. REVISE     - if an unnamed flexible curve beats every named law, the named
                hypothesis space is inadequate: expand it.
6. FALSIFY    - before accepting a leader, runs experiments designed to
                break it (where its best rivals disagree with it most).
7. STOP       - when confident and the leader survived falsification, when
                progress stalls (diminishing returns), or when out of budget.
8. DELIVER    - answer + calibrated credence + full evidence trail; asks for
                human input when credence is too low to act on.

Every strategic choice above is a genome parameter, which is what the
self-improver tunes.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ..epistemics import Ledger, Status
from ..genome import Genome
from ..uncertainty import DiminishingReturns
from .hypotheses import (CORE_FAMILIES, EXTENDED_FAMILIES, FLEXIBLE, Fit, FitError,
                         make_hypotheses)


class ExperimentalSystem(Protocol):
    domain: tuple[float, float]

    def measure(self, x: float) -> float: ...


@dataclass
class Observation:
    x: float
    y: float
    purpose: str
    evidence_id: str


@dataclass
class DiscoveryResult:
    answer: str
    credence: float
    posterior: dict[str, float]
    best_fit: Fit
    observations: list[Observation]
    falsifications: list[dict]
    expanded: bool
    stop_reason: str
    trace: list[dict]
    ledger: Ledger
    human_request: str | None = None
    hypotheses_considered: list[str] = field(default_factory=list)

    @property
    def n_experiments(self) -> int:
        return len(self.observations)

    def predict(self, x: float) -> float:
        return self.best_fit.predict(x)


def posterior_from_fits(fits: dict[str, Fit], log_prior: dict[str, float],
                        temperature: float = 1.0) -> dict[str, float]:
    """p(h | data) ∝ prior(h) · exp(-BIC(h)/2), tempered by `temperature`."""
    scores = {}
    for name, fit in fits.items():
        bic = fit.bic
        if math.isfinite(bic):
            scores[name] = (log_prior[name] - 0.5 * bic) / temperature
    if not scores:
        raise FitError("no hypothesis could be fitted")
    m = max(scores.values())
    z = {k: math.exp(v - m) for k, v in scores.items()}
    total = sum(z.values())
    return {k: v / total for k, v in sorted(z.items(), key=lambda kv: -kv[1])}


def entropy(p: dict[str, float]) -> float:
    return -sum(v * math.log(v) for v in p.values() if v > 0)


class DiscoveryLoop:
    def __init__(
        self,
        genome: Genome,
        *,
        budget: int = 30,
        seed: int = 0,
        prior_counts: Counter | dict | None = None,
        ledger: Ledger | None = None,
        extra_hypotheses: Sequence = (),
    ) -> None:
        self.g = genome
        self.budget = budget
        self.rng = random.Random(seed)
        self.prior_counts = Counter(prior_counts or {})
        self.ledger = ledger or Ledger()
        self.extra = list(extra_hypotheses)

    # -- priors ------------------------------------------------------------
    def _log_prior(self, names: Sequence[str]) -> dict[str, float]:
        if self.g.use_learned_priors and self.prior_counts:
            w = {n: self.prior_counts.get(n, 0) + self.g.prior_pseudocount for n in names}
        else:
            w = {n: 1.0 for n in names}
        total = sum(w.values())
        return {n: math.log(w[n] / total) for n in names}

    # -- analysis ------------------------------------------------------------
    def _evaluate(self, hyps: dict, obs: list[Observation], full: bool):
        xs = [o.x for o in obs]
        ys = [o.y for o in obs]
        robust = self.g.likelihood == "student_t"
        fits = {}
        for name, h in hyps.items():
            try:
                fits[name] = h.fit(xs, ys, robust=robust, nu=self.g.t_dof, warm=not full)
            except (FitError, OverflowError, ValueError, ZeroDivisionError):
                continue
        post = posterior_from_fits(fits, self._log_prior(list(fits)), self.g.temperature)
        return fits, post

    # -- experiment design ----------------------------------------------------
    def _space_filling(self, obs, lo, hi) -> float:
        xs = [o.x for o in obs]
        cands = [self.rng.uniform(lo, hi) for _ in range(self.g.n_candidates)] + [lo, hi]
        return max(cands, key=lambda c: min((abs(c - x) for x in xs), default=1.0))

    def _next_x(self, fits, post, obs, lo, hi) -> tuple[float, str]:
        g = self.g
        if g.design == "random":
            return self.rng.uniform(lo, hi), "random"
        if g.design == "space_filling" or (g.explore_prob and self.rng.random() < g.explore_prob):
            return self._space_filling(obs, lo, hi), "space-filling"
        # disagreement: maximize posterior-weighted variance of predictions
        active = [(n, p) for n, p in post.items() if p >= 1e-4]
        if len(active) < 2:
            return self._space_filling(obs, lo, hi), "space-filling (consensus)"
        xs = [o.x for o in obs]
        best_x, best_s = None, -1.0
        for _ in range(g.n_candidates):
            c = self.rng.uniform(lo, hi)
            preds = [(p, fits[n].predict(c)) for n, p in active]
            preds = [(p, v) for p, v in preds if math.isfinite(v)]
            if len(preds) < 2:
                continue
            wsum = sum(p for p, _ in preds)
            m = sum(p * v for p, v in preds) / wsum
            s = sum(p * (v - m) ** 2 for p, v in preds) / wsum
            s += 1e-12 * min(abs(c - x) for x in xs)   # tie-break toward unexplored inputs
            if s > best_s:
                best_x, best_s = c, s
        if best_x is None:
            return self._space_filling(obs, lo, hi), "space-filling (fallback)"
        return best_x, "discriminate"

    def _falsification_x(self, leader, fits, post, obs, lo, hi) -> float:
        rivals = [(n, p) for n, p in post.items() if n != leader and p > 0]
        total = sum(p for _, p in rivals)
        if not rivals or total <= 0:
            return self._space_filling(obs, lo, hi)
        cands = [self.rng.uniform(lo, hi) for _ in range(self.g.n_candidates)] + [lo, hi]
        lf = fits[leader]

        def score(c):
            lv = lf.predict(c)
            s = 0.0
            for n, p in rivals:
                v = fits[n].predict(c)
                if math.isfinite(v) and math.isfinite(lv):
                    s += (p / total) * abs(lv - v)
            return s

        return max(cands, key=score)

    # -- main loop -------------------------------------------------------------
    def run(self, system: ExperimentalSystem) -> DiscoveryResult:
        g, led = self.g, self.ledger
        lo, hi = system.domain
        names = list(CORE_FAMILIES) + ([FLEXIBLE] if g.flexible_baseline else [])
        hyps = {h.name: h for h in make_hypotheses(names)}
        for h in self.extra:
            hyps[h.name] = h

        # FRAME: unknowns and competing hypotheses, recorded as speculation.
        led.assert_claim("Unknowns: the generating law, its parameters, the noise level, "
                         "and whether some measurements are corrupted.", Status.SPECULATION, 1.0,
                         author="frame")
        prior = {n: math.exp(v) for n, v in self._log_prior(list(hyps)).items()}
        hyp_claims = {n: led.assert_claim(f"The system follows the '{n}' law.", Status.SPECULATION,
                                          prior[n], author="hypothesize").id for n in hyps}
        if g.use_learned_priors and self.prior_counts:
            ev = led.add_evidence("note", "priors from verified outcomes in memory",
                                  {"counts": dict(self.prior_counts)})
            led.assert_claim("Hypothesis priors reflect verified past outcomes in this domain.",
                             Status.INFERENCE, 0.9, evidence=[ev.id], author="recall")

        obs: list[Observation] = []

        def measure(x: float, purpose: str) -> None:
            y = system.measure(x)
            ev = led.add_evidence("observation", f"measured y({x:.4g}) = {y:.6g}",
                                  {"x": x, "y": y, "purpose": purpose})
            obs.append(Observation(x, y, purpose, ev.id))

        n0 = max(1, min(g.n_initial, self.budget))
        for i in range(n0):
            measure(lo + (hi - lo) * (i / (n0 - 1) if n0 > 1 else 0.5), "initial")

        falsify_left = g.falsification_rounds
        falsifications: list[dict] = []
        pending: tuple[str, float] | None = None
        expanded = False
        stall = DiminishingReturns(patience=g.patience, min_delta=0.005) if g.patience else None
        trace: list[dict] = []
        stop_reason = "budget exhausted"

        while True:
            fits, post = self._evaluate(hyps, obs, full=(len(obs) % 5 == 0 or len(trace) == 0))
            leader = next(iter(post))
            trace.append({"n": len(obs), "leader": leader, "p": round(post[leader], 4),
                          "entropy": round(entropy(post), 4),
                          "top": [(k, round(v, 4)) for k, v in list(post.items())[:3]],
                          "last": obs[-1].purpose})

            if pending is not None:
                tested, x_t = pending
                outcome = "survived" if leader == tested else "refuted"
                falsifications.append({"hypothesis": tested, "x": x_t, "outcome": outcome,
                                       "new_leader": leader, "leader_p": post[leader]})
                led.record_falsification(hyp_claims[tested],
                                         f"prediction at x={x_t:.4g} where rivals disagree most",
                                         outcome, [obs[-1].evidence_id])
                pending = None

            # REVISE: named laws inadequate -> widen the hypothesis space once.
            if (g.expand_on_inadequacy and not expanded and leader == FLEXIBLE
                    and post[leader] >= 0.5 and len(obs) >= g.min_experiments):
                expanded = True
                for h in make_hypotheses(EXTENDED_FAMILIES):
                    hyps[h.name] = h
                    hyp_claims[h.name] = led.assert_claim(
                        f"The system follows the '{h.name}' law.", Status.SPECULATION, 0.5,
                        author="revise").id
                ev = led.add_evidence("derivation", "flexible baseline beat every named law",
                                      {"posterior": post})
                led.assert_claim("The initial hypothesis space was inadequate; expanded it with "
                                 + ", ".join(EXTENDED_FAMILIES), Status.INFERENCE, post[leader],
                                 evidence=[ev.id], author="revise")
                continue

            n = len(obs)
            if n >= self.budget:
                stop_reason = "budget exhausted"
                break
            if post[leader] >= g.stop_posterior and n >= g.min_experiments:
                if falsify_left > 0:
                    falsify_left -= 1
                    x = self._falsification_x(leader, fits, post, obs, lo, hi)
                    pending = (leader, x)
                    measure(x, "falsify")
                    continue
                stop_reason = "confident and survived falsification" if falsifications else "confident"
                break
            if stall is not None and stall.update(post[leader]) and n >= g.min_experiments:
                stop_reason = "diminishing returns"
                break
            x, purpose = self._next_x(fits, post, obs, lo, hi)
            measure(x, purpose)

        leader = next(iter(post))
        credence = post[leader]
        best = fits[leader]
        for name, cid in hyp_claims.items():
            if name in post:
                led.update_confidence(cid, post[name], "posterior after experiments")
        n_falsify = sum(1 for o in obs if o.purpose == "falsify")
        xs_all = [o.x for o in obs]
        led.assert_claim(
            f"Recorded {len(obs)} measurements over x in [{min(xs_all):.3g}, {max(xs_all):.3g}]"
            + (f", {n_falsify} of them designed to falsify the leading hypothesis" if n_falsify else "") + ".",
            Status.FACT, 1.0, evidence=[o.evidence_id for o in obs], author="experiment")
        fit_ev = led.add_evidence("derivation", f"BIC model comparison over {len(post)} hypotheses",
                                  {"posterior": post, "bic": {k: f.bic for k, f in fits.items()}})
        answer_text = ("none of the named laws fits; best description: " + best.description
                       if leader == FLEXIBLE else f"'{leader}' law: {best.description}")
        led.assert_claim(f"Conclusion: {answer_text}", Status.INFERENCE, credence,
                         evidence=[fit_ev.id] + [o.evidence_id for o in obs], author="deliver")

        human_request = None
        if credence < g.ask_below:
            rivals = ", ".join(f"{k} ({v:.2f})" for k, v in list(post.items())[:3])
            x_next = self._falsification_x(leader, fits, post, obs, lo, hi)
            human_request = (f"Inconclusive after {len(obs)} experiments (credence {credence:.2f}). "
                             f"Leading hypotheses: {rivals}. The most informative next experiment is "
                             f"x={x_next:.3g}. Domain knowledge or more budget would help.")
        return DiscoveryResult(
            answer=leader, credence=credence, posterior=post, best_fit=best, observations=obs,
            falsifications=falsifications, expanded=expanded, stop_reason=stop_reason, trace=trace,
            ledger=led, human_request=human_request, hypotheses_considered=list(hyps),
        )
