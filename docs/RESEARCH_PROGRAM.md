# Research program

From a controlled self-improvement experiment toward a rigorously evaluated system that
can acquire capabilities it was not given. This document records the audit, the
diagnosis of the original benchmark's 0% category, the evaluation design, the
architectures for autonomy and safe self-modification, a plan for validating the
language-model layer, and a prioritized roadmap. Every milestone states what would
falsify it.

**What Quanta is not.** It is not AGI or ASI, not conscious, and not generally
intelligent. Its measured abilities are confined to small, mostly simulated law-discovery
problems. Nothing below should be read as progress toward general intelligence; the
honest framing is *how to tell real self-improvement from apparent self-improvement*.

## 1. Where Quanta stands

### Self-improvement levels

| Level | Meaning | Quanta |
|---|---|---|
| L1 parameters | tune settings of a fixed strategy | **Done and measured** (`improve.py`; two adopted changes, replicated, ablated; random-search control) |
| L2 strategy invention | invent new hypotheses, procedures or tools | **Narrow, under test**: capability acquisition invents new hypothesis families inside a designer-written expression language (`capabilities/`) |
| L3 capability / architecture | redesign its own learning machinery | **Not attempted.** Deliberately out of scope: the evaluator, gates and search procedure are outside the system's authority |

### Capability map (summary; the full table is in [ARCHITECTURE.md](ARCHITECTURE.md))

| Area | Status |
|---|---|
| Experiment design, falsification, calibrated "none" answers (discovery loop) | measured |
| Self-improvement of strategy parameters | measured |
| Capability acquisition (new hypothesis families from failures) | implemented; pre-registered experiment in progress |
| Evidence ledger, governance, memory, planning | implemented and tested; the audit found real holes (section 3) |
| Language-model research loop (`quanta solve`) | plumbing tested with scripted models only; **never run against a real model** |
| Perception, long horizons, collaboration | interfaces tested; quality unmeasured |

## 2. Evidence table

FACT = demonstrated by code plus a passing test or a recorded experiment.
INFERENCE = follows from facts but was not directly tested. HYPOTHESIS = untested.

| Claim | Status | Evidence / limit |
|---|---|---|
| Strategy self-improvement improved the held-out score | FACT | +0.039, CI [+0.025, +0.054], 400 test tasks (`experiments/ANALYSIS.md`) |
| The gain came from fewer confident errors, not better typical answers | FACT | 157 wins vs 243 losses per task; confidently-wrong 21.5% → 14.7% |
| Diagnosis-driven search beats random search at equal budget | FACT (one run each) | +0.014, CI [+0.004, +0.023]; a single pair of runs, so weak |
| Learned priors from verified experience help | FACT, with a caveat | +0.009 with 100+ past tasks, but "verified" outcomes came from the grader's ground truth (oracle labels); in real research no such oracle exists |
| The original system could not solve outside-space laws | FACT, structural | 0% because "none" and the extended families were unreachable in the adopted configuration (section 5) |
| The ledger keeps unsupported claims from being called facts | INFERENCE, weaker than stated | it checks the evidence *type* only; a failed file read or a trivial execution grounds an unrelated FACT |
| Governance cannot be weakened by the agent | INFERENCE, false in one case | auto-approved `run_python` can rewrite the run's own state, ledger and audit log and delete the STOP file |
| The audit log is tamper-evident | FACT for in-place edits only | dropping trailing entries or re-chaining the whole log goes undetected |
| Capability acquisition finds structures matching the hidden gap laws | INFERENCE (pilot) | smoke runs found exp(−a(x+b)²), 1/(e^{ax}+b), sin(·)+x shapes without labels |
| Acquired capabilities generalize, transfer and beat random proposals | HYPOTHESIS | the pre-registered experiment tests exactly this |
| The language-model loop does useful research | HYPOTHESIS | never run against a real model |

## 3. Audit findings (core substrate and governance)

Found by an independent audit agent with reproductions; status as of this commit.
Fixes that touch solver-imported modules are deferred until the running experiment's
phases complete, so that the pre-registered run uses one code version.

| Finding | Severity | Status |
|---|---|---|
| Failed tool calls (missing file, failed fetch) count as grounding evidence for FACT | high | **fixed**: a failed call is recorded as a note (`tests/test_hardening.py`) |
| Inference cap applied only at assertion; `update_confidence` ignores it; lowering a premise does not re-cap dependents | medium | open |
| `update_confidence(NaN)` sets 1.0, even on refuted claims; refuting needs no evidence | medium | open |
| Promotion to FACT ignores `needs_review` (a claim whose premise was refuted can be promoted) | medium | open |
| `Ledger.from_dict` re-checks nothing; a stale counter makes new ids overwrite old claims | medium | open |
| Ledger reads are not locked (1,281 "dictionary changed size" errors in a concurrency probe) | medium | open |
| The auto-approval clamp runs only in `__init__`; assigning the attribute later auto-approves CRITICAL | high (in-process only) | **fixed**: the ceiling is enforced on every assignment and read from a private copy |
| A shell prefix rule (`--allow-shell-prefix pytest`) also pre-authorizes `pytest && git push ...` | high | **fixed**: prefix rules match whole words and refuse any shell control syntax |
| The kill switch does not latch: deleting STOP re-enables authorization | medium | **fixed**: it latches until an explicit operator reset |
| `charge()` accepts negative amounts; wall-clock budget resets on resume | medium | negative charges **fixed**; the resume reset is open |
| Audit log: truncation and full re-chaining verify as intact; a raising approver logs nothing; args cut at 2,000 chars | medium | **fixed while the writer is alive** (any modification of the file trips the kill switch; approver failures are logged; the full arguments are hashed). A same-user process can still forge a consistent log after the writer exits |
| Auto-approved `run_python` can rewrite the run's own oversight files | high | open; needs an OS sandbox or approval |
| Memory search returns nothing for non-ASCII queries | low | open |

## 4. Bottlenecks

1. **The hypothesis space is fixed.** Everything outside ten families is either "none" or
   wrong. This is the bottleneck capability acquisition attacks.
2. **Small-sample instability.** At about ten measurements the flexible description becomes
   assessable with two residual degrees of freedom and can win with credence near 1, so
   answers flip on small perturbations. This inflates per-task variance and makes every
   comparison need large samples.
3. **Model comparison is BIC-based.** It over-rewards flexible models at small n and
   under-counts searched parameters (the look-elsewhere effect for frequencies).
4. **The language-model layer is unvalidated.** All claims about open-ended research depend on it.
5. **Oversight is in-process.** The strongest guarantees need an OS-level boundary.

## 5. The 0% category

**Diagnosis (code).** The original benchmark files linear+sine as `linear_sin` (an extended
family) and gaussian bump / damped oscillation as `none`. The adopted champion had
`flexible_baseline=False` and `expand_on_inadequacy=False`: "none" could not be answered and
`linear_sin` was never considered. 0% was therefore guaranteed by the configuration, not
measured as a limit of the method.

**Experiment.** `experiments/zero_category/` evaluates a fixed ladder on the original
held-out test split (400 tasks, 55 of them outside the initial space; no selection among
the configurations). Full table: [experiments/zero_category/RESULTS.md](../experiments/zero_category/RESULTS.md).

| configuration | score | outside-space | linear+sine | bump | damped osc. | clean |
|---|---|---|---|---|---|---|
| L0 champion | 0.798 | 0% | 0% | 0% | 0% | 90.8% |
| L1 + flexible baseline | 0.763 | 38% | 0% | 94% | 29% | 79.2% |
| L2 + expansion on inadequacy | 0.787 | 67% | 94% | 94% | 24% | 76.4% |
| L3 + look-elsewhere correction | 0.786 | 67% | 78% | 94% | 38% | 80.0% |
| L4 + predictive pooling | 0.769 | 65% | 78% | 88% | 38% | 72.8% |

Findings:
* The 0% category is fixable: the flexible baseline plus expansion reaches 67%.
* The look-elsewhere correction does what it is for: it removes false periodic
  detections (clean accuracy 76.4% → 80.0%, damped oscillation 24% → 38%), at the cost of
  some true linear+sine detections (94% → 78%). Its net effect on the score is nil
  (−0.0008, CI [−0.011, +0.008]).
* No configuration beats the champion's overall score: L2 and L3 are −0.011 and −0.012
  (intervals include zero), L1 is −0.035 (CI [−0.058, −0.012]). Gains on the outside-space
  category are paid for on clean tasks, where the flexible description now competes.
* Predictive pooling hurts on this benchmark (−0.017 vs L3; confidently wrong 14.8% →
  24.0%) because this benchmark grades the *label*, not the predictions: pooling raises
  credence when two laws predict alike (e.g. quadratic vs cubic) even when the label is
  wrong. The capability-acquisition benchmark grades predictions, where pooling helped
  (dev pilot: +0.007 on familiar tasks).

## 6. Generalization benchmark

`bench/acquisition.py`, `bench/external.py`, `bench/sealed.py`:

| | Tests | Why |
|---|---|---|
| A familiar | the ten core families | regression: new capabilities must not break what worked |
| B gap | five law types the solver cannot express, never named | whether a missing capability can be acquired |
| C-regime | the gap laws with fewer measurements, more outliers, shifted parameters | transfer to a new context |
| C-compose | laws containing a gap law as a component | compositional reuse |
| D-sealed | classes written blind by an independent agent | novelty the designers could not tailor to |
| D-external | 29 real datasets chosen by a fixed rule | the real world, with no ground-truth law |

Extrapolation to [10, 13] is scored, which separates understanding a law from fitting a curve.

## 7. Self-improvement experiment

Capability acquisition, pre-registered:
[experiments/capability_acquisition/PREREGISTRATION.md](../experiments/capability_acquisition/PREREGISTRATION.md);
mechanism and results: [CAPABILITY_ACQUISITION.md](CAPABILITY_ACQUISITION.md).

## 8. Autonomy architecture (designed, not built)

A daemon that runs the acquisition loop continuously would need, in this order:
1. a task source that is not the evaluator (otherwise it optimizes the benchmark);
2. a budgeted scheduler with the existing governor (steps, tokens, money, wall-clock);
3. the registry as its only write path to capabilities, with ACTIVE requiring review;
4. drift monitoring: re-run the reviewer's gates on fresh blocks periodically and DEPRECATE
   capabilities whose held-out effect decays;
5. a human checkpoint for anything outside the expression language (code, tools, data access);
6. a kill switch that latches and lives outside the process.

## 9. Safe self-modification architecture

| Surface | Who may change it |
|---|---|
| strategy parameters (genome) | the self-improver, through the statistical gate and replication |
| capabilities (expression artifacts) | the acquisition loop, through the independent reviewer |
| solver code | humans (sandboxed patch evaluation exists but no model-written patch was evaluated) |
| evaluator, splits, gates, score, policy, governance | humans only; protected paths and frozen copies |

Rollback: capabilities are versioned and never overwritten; removing a capability is a
DEPRECATED event; the toolkit is rebuilt from ACTIVE records, so rolling back is
re-evaluating without the deprecated record.

## 10. Validating the language-model layer

Stages, each with a pass criterion fixed in advance:
1. request-shape tests against the real API (no reasoning claims);
2. the model drives `compare_hypotheses` on the discovery benchmark; compare with the automated loop at equal measurement budget;
3. the model proposes capability candidates (formulas) inside the same reviewer; compare acceptance and held-out gain with the program search and the random control;
4. small, checkable open-ended tasks with known answers (replicating a published curve fit, finding a planted bug) with a verifier from a different model;
5. only then, open-ended problems, reported with the ledger's FACT/INFERENCE/SPECULATION split.

## 11. Validation progression

1. unit tests of each component;
2. in-distribution synthetic tasks (development split only);
3. held-out synthetic tasks (fresh seeds, gate blocks);
4. transfer (regime shift, composition);
5. blind novel classes written by an independent author;
6. real data chosen by a fixed rule;
7. independent replication (new tasks, fresh state, ideally another person or agent running the code).

A claim is stated at the highest stage it has passed, and no higher.

## 12. Metrics

* **score**: 0.4 (1 − (p_claim − recovered)²) + 0.3 exp(−interp_nrmse) + 0.3 exp(−extrap_nrmse) − 0.1 cost
* **recovered**: interp_nrmse ≤ 0.15 and extrap_nrmse ≤ 0.15 (NRMSE relative to the truth's spread)
* **confidently wrong**: a law claimed with credence ≥ 0.9 that was not recovered
* **acceptance rate**: ACTIVE / proposed capabilities, for the loop and for the random control
* **held-out and transfer gain**: paired score differences on fresh blocks, 95% bootstrap intervals
* **retry resolution**: self-detected training failures resolved after adopting a capability
* **diagnosis consistency**: share of research findings whose structure matches the diagnosed failure mode
* **real-data score**: 0.5 exp(−interp_nrmse) + 0.5 exp(−extrap_nrmse) on held-out points, dataset-clustered intervals

## 13. Roadmap

| # | Capability added | Why it matters | Measurement | Falsifier | Success / failure |
|---|---|---|---|---|---|
| 1 | Capability acquisition (this experiment) | the first step from L1 to L2 | pre-registered H1–H5 | random control does as well; gain vanishes on replication or real data | H1 and H5 hold in both replications / any of them fails |
| 2 | Small-sample-correct model comparison (marginal likelihood over the noise scale) | removes the ~10-measurement flip, cuts variance | flip rate; per-task variance; held-out score | no reduction in flip rate | flips halve without a score loss / they do not |
| 3 | Capability maintenance (drift checks, DEPRECATED) | an accumulating library must not rot | held-out effect of each ACTIVE capability over time | a stale capability keeps passing | decayed capabilities are deprecated / are not |
| 4 | Fix the audit's high-severity findings | the safety claims must be true | each finding's reproduction now fails | any reproduction still succeeds | all reproductions fail / any succeeds |
| 5 | Model-proposed capabilities inside the same reviewer | tests whether a language model adds anything over search | acceptance rate and held-out gain vs program search vs random | the model does no better than random | better than both / no better than random |
| 6 | Multi-input laws and real experimental data | leaves the 1-D toy world | the same gates on 2–3 input tasks and real datasets | gains do not transfer | pre-registered gains replicate / they do not |
| 7 | OS-level sandbox for any executed code | required before evaluating model-written code | escape attempts in a red-team suite | any escape | none / any |

## 14. Research standard

Reconstructed from the requirements recorded for this work (the original wording was not
preserved). Every claimed result must answer:

1. What exactly is the claim, and at which validation stage was it tested?
2. What was the baseline, and was it run on the same tasks (paired)?
3. Which data were used for decisions, and which only for reporting?
4. Could held-out information have leaked into any decision?
5. How large is the effect, with what interval, on how many tasks?
6. Was the analysis fixed before the data were seen?
7. Would random search, or chance, explain it?
8. Does it survive a fresh environment and a replication on new tasks?
9. Does it transfer outside the training distribution?
10. Does it break anything that worked?
11. Is it a scoring artifact (which score component moved)?
12. Is it memorization (train vs test, with and without learned parameter values)?
13. What failed, and is every failed attempt still on record?
14. What would falsify it next, and what is the next experiment?

[CAPABILITY_ACQUISITION.md](CAPABILITY_ACQUISITION.md) answers these for the
capability-acquisition result.
