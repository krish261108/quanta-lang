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
from .synth import FREQ_TRIALS, SynthHypothesis, synthesize

# Named families whose fit searches a frequency grid (look-elsewhere effect).
FREQ_FAMILIES = ("sinusoid", "linear_sin")


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
    constructed: list[str] = field(default_factory=list)
    inadequacy_detected: bool = False
    answer_kind: str = "named"          # named | capability | constructed | none

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
        library: Sequence = (),
        predict_x: Sequence[float] | None = None,
        separate_streams: bool = False,
    ) -> None:
        self.g = genome
        self.budget = budget
        self.rng = random.Random(seed)
        # Common random numbers: with separate streams, the design draws do not depend on
        # how many candidate sets falsification or construction consumed, so two solver
        # configurations on the same task measure at the same random inputs. The default
        # (one shared stream) reproduces earlier recorded results exactly.
        self.rng_aux = random.Random(seed ^ 0x5F3759DF) if separate_streams else self.rng
        self.prior_counts = Counter(prior_counts or {})
        self.ledger = ledger or Ledger()
        self.extra = list(extra_hypotheses)
        self.library = list(library)       # learned abstractions usable as building blocks
        self.predict_x = list(predict_x) if predict_x else None   # where predictions will be asked for
        self._offset: dict[str, float] = {}

    # -- priors ------------------------------------------------------------
    def _log_prior(self, names: Sequence[str]) -> dict[str, float]:
        if self.g.use_learned_priors and self.prior_counts:
            w = {n: self.prior_counts.get(n, 0) + self.g.prior_pseudocount for n in names}
        else:
            w = {n: 1.0 for n in names}
        total = sum(w.values())
        # Structure costs: constructed programs pay for the size of the space searched,
        # and (with freq_penalty) frequency-searched families pay a look-elsewhere charge.
        return {n: math.log(w[n] / total) + self._offset.get(n, 0.0) for n in names}

    def _register(self, h) -> None:
        offset = getattr(h, "log_prior_offset", 0.0)
        if self.g.freq_penalty and h.name in FREQ_FAMILIES:
            offset -= math.log(FREQ_TRIALS)
        if offset:
            self._offset[h.name] = offset

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

    POOL_TOL = 0.5    # predictions "agree" if their RMS difference is below half the noise level

    def _support(self, leader: str, fits: dict, post: dict, obs: list[Observation], lo: float,
                 hi: float) -> float:
        """Credence that the leader's *predictions* are right: the posterior mass of all
        hypotheses whose predictions at the requested points agree with the leader's
        (within POOL_TOL of the leader's robust noise scale). Without pooling this is
        just the leader's posterior."""
        if not self.g.pool_equivalent:
            return post[leader]
        lf = fits[leader]
        res = sorted(abs(o.y - lf.predict(o.x)) for o in obs)
        res = [r for r in res if math.isfinite(r)]
        if not res:
            return post[leader]
        sigma = 1.4826 * res[len(res) // 2]
        if sigma <= 0:
            return post[leader]
        grid = self.predict_x or [lo + (hi - lo) * i / 24 for i in range(25)]
        ref = [lf.predict(x) for x in grid]
        mass = 0.0
        for name, p in post.items():
            if name == leader:
                mass += p
                continue
            if p < 1e-6:
                continue
            se, ok = 0.0, True
            for x, r in zip(grid, ref):
                v = fits[name].predict(x)
                if not (math.isfinite(v) and math.isfinite(r)):
                    ok = False
                    break
                se += (v - r) ** 2
            if ok and math.sqrt(se / len(grid)) <= self.POOL_TOL * sigma:
                mass += p
        return min(1.0, mass)

    # -- experiment design ----------------------------------------------------
    def _space_filling(self, obs, lo, hi, rng: random.Random | None = None) -> float:
        xs = [o.x for o in obs]
        rng = rng or self.rng
        cands = [rng.uniform(lo, hi) for _ in range(self.g.n_candidates)] + [lo, hi]
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
            return self._space_filling(obs, lo, hi, self.rng_aux)
        cands = [self.rng_aux.uniform(lo, hi) for _ in range(self.g.n_candidates)] + [lo, hi]
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
        for h in hyps.values():
            self._register(h)

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
        constructions, next_construct_n, constructed, last_construct_n = 0, 0, [], -1
        inadequacy = False
        stall = DiminishingReturns(patience=g.patience, min_delta=0.005) if g.patience else None
        trace: list[dict] = []
        stop_reason = "budget exhausted"

        while True:
            fits, post = self._evaluate(hyps, obs, full=(len(obs) % 5 == 0 or len(trace) == 0))
            leader = next(iter(post))
            support = self._support(leader, fits, post, obs, lo, hi)
            trace.append({"n": len(obs), "leader": leader, "p": round(post[leader], 4),
                          "support": round(support, 4),
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
            if leader == FLEXIBLE and post[leader] >= 0.5 and len(obs) >= g.min_experiments:
                inadequacy = True
            if (g.expand_on_inadequacy and not expanded and leader == FLEXIBLE
                    and post[leader] >= 0.5 and len(obs) >= g.min_experiments):
                expanded = True
                for h in make_hypotheses(EXTENDED_FAMILIES):
                    hyps[h.name] = h
                    self._register(h)
                    hyp_claims[h.name] = led.assert_claim(
                        f"The system follows the '{h.name}' law.", Status.SPECULATION, 0.5,
                        author="revise").id
                ev = led.add_evidence("derivation", "flexible baseline beat every named law",
                                      {"posterior": post})
                led.assert_claim("The initial hypothesis space was inadequate; expanded it with "
                                 + ", ".join(EXTENDED_FAMILIES), Status.INFERENCE, post[leader],
                                 evidence=[ev.id], author="revise")
                continue

            # CONSTRUCT: no available law explains the data -> build candidate laws.
            if (g.construct and constructions < 2 and leader == FLEXIBLE and post[leader] >= 0.5
                    and len(obs) >= max(g.min_experiments, next_construct_n)):
                constructions += 1
                next_construct_n = len(obs) + 8
                last_construct_n = len(obs)
                added = self._construct(obs, hyps)
                if added:
                    ev = led.add_evidence("derivation", f"program synthesis proposed {len(added)} new laws",
                                          {"programs": [h.name for h in added]})
                    for h in added:
                        hyps[h.name] = h
                        self._register(h)
                        constructed.append(h.name)
                        hyp_claims[h.name] = led.assert_claim(
                            f"The system follows the constructed law {h.name}.", Status.SPECULATION, 0.1,
                            evidence=[ev.id], author="construct").id
                    continue

            n = len(obs)
            if n >= self.budget:
                stop_reason = "budget exhausted"
                break
            # A law constructed from the data must also survive fresh data before it is accepted.
            fresh_ok = last_construct_n < 0 or n >= last_construct_n + g.construct_holdout
            if support >= g.stop_posterior and n >= g.min_experiments and fresh_ok:
                if falsify_left > 0:
                    falsify_left -= 1
                    x = self._falsification_x(leader, fits, post, obs, lo, hi)
                    pending = (leader, x)
                    measure(x, "falsify")
                    continue
                stop_reason = "confident and survived falsification" if falsifications else "confident"
                break
            if stall is not None and stall.update(support) and n >= g.min_experiments:
                stop_reason = "diminishing returns"
                break
            x, purpose = self._next_x(fits, post, obs, lo, hi)
            measure(x, purpose)

        leader = next(iter(post))
        credence = self._support(leader, fits, post, obs, lo, hi)
        best = fits[leader]
        lh = hyps.get(leader)
        if leader == FLEXIBLE:
            kind = "none"
        else:
            kind = getattr(lh, "origin", "named")
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
            constructed=constructed, inadequacy_detected=inadequacy, answer_kind=kind,
        )

    def _construct(self, obs: list[Observation], hyps: dict) -> list:
        """Search the program space for laws that explain the observations; return the
        best few as new competing hypotheses (each carrying its structure cost)."""
        xs = [o.x for o in obs]
        ys = [o.y for o in obs]
        cands = synthesize(xs, ys, library=self.library, max_size=self.g.construct_max_size,
                           robust=self.g.likelihood == "student_t", nu=self.g.t_dof,
                           seed=self.rng_aux.randrange(1 << 30), top_k=4)
        out = []
        for c in cands:
            if c.model.size == 0:          # c + c*x duplicates the named linear law
                continue
            h = SynthHypothesis(c.model, init=c.fit.params,
                                log_prior_offset=c.model.log_prior(len(self.library)))
            if h.name not in hyps:
                out.append(h)
            if len(out) == 3:
                break
        return out
