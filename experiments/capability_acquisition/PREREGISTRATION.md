# Pre-registration: can Quanta acquire capabilities it was not given?

Registered before the first full run. The runner (`run.py`) and this document are
committed together, before any result exists. Anything changed afterwards is listed
under *Deviations* in the results document, together with the reason; runs made
before a change are kept, not discarded.

## 1. Question

When Quanta fails at tasks that its starting toolkit cannot express, can it

1. detect the failure itself (without the evaluator's grades),
2. work out what is missing,
3. build a reusable capability (a code artifact) that fixes the failure,
4. and have that capability **pass an independent reviewer** and then **improve
   performance on fresh tasks it never saw, including transferred, compositional,
   blind-novel and real-world tasks**,

and does it do this better than random proposals pushed through the same pipeline?

## 2. System under test

* **S0** (the starting solver): `Genome(falsification_rounds=2, likelihood="student_t",
  flexible_baseline=True, pool_equivalent=True)`, ten named law families
  (`CORE_FAMILIES`) and a flexible description that can answer "none of my laws fits".
* **The acquisition loop** (`src/quanta/capabilities/acquire.py`) with
  `AcquisitionSettings(n_train=200, n_val=800, k_min=2, max_rounds=6,
  max_reviews_per_round=2, max_candidates_per_round=4, max_research_tasks=80,
  synth_max_size=3, cv_ratio=0.8, patience=2)`, seed 20261106 (+1000 per replication).

What is designer-provided and what is acquired, stated plainly. The designers wrote the
expression language (`science/synth.py`: x, exp, log, sin, 1/(·+b), (·+b)², products,
library leaves), the program search, the failure detectors, the diagnosis rules and the
reviewer. Which structures to build, from which failures, and whether to keep them is
decided by the loop and the reviewer. No capability, law name or task label is given to
the loop. It never sees grades on training tasks, and it never touches held-out data.

## 3. Tasks, splits and information flow

Tasks are hidden 1-D laws, measured with noise (and outliers on some tasks) at inputs the
solver chooses in [0.1, 10]. Predictions are scored on [0.1, 10] and on [10, 13]
(extrapolation). Score (fixed before any acquisition run, `bench/acquisition.py`):

    score = 0.4 (1 - (p_claim - recovered)^2) + 0.3 exp(-interp_nrmse)
            + 0.3 exp(-extrap_nrmse) - 0.1 measurements/budget
    recovered = interp_nrmse <= 0.15 and extrap_nrmse <= 0.15

| Kind | Content |
|---|---|
| A familiar | the ten core families |
| B gap | damped oscillation, gaussian bump, linear+sine, logistic, lorentzian (never named to the loop) |
| C-regime | gap laws with 20 measurements, 15% outliers, shifted parameter ranges |
| C-compose | laws containing a gap law as a component (wave packet, double bump, bump + wave) |
| D-sealed | law classes written blind by an independent agent (`bench/sealed.py`; count and sha256 recorded by the run); the experimenters use only the module's API and do not read it before the final evaluation |
| D-external | 29 real datasets from R's `datasets` package, chosen by a fixed rule (`bench/external.py`) |

Seed ranges are disjoint per split (100,000 seeds each; replication r adds 10,000,000·r).

| Split | Used by | For |
|---|---|---|
| train (200) | the loop | solving and self-diagnosis; outputs and observations only, no grades |
| validation (800, block 0) | the loop | selecting candidates; paired score differences visible |
| gate (1200 per review) | the reviewer only | held-out gain and familiar regression; a fresh block per review |
| gate_transfer (400 per review) | the reviewer only | transfer gain; a fresh block per review |
| test (1200), test_familiar (600), test_regime (600), test_compose (400), sealed (600), external (87) | final report only | never used for any decision |

Reviewer blocks: the acquisition run uses gate blocks 1, 2, ...; the random control uses 40, 41, ....

## 4. Conditions (final evaluation)

| Condition | Configuration |
|---|---|
| C0 | S0 |
| C1 | S0 + per-task construction (program search inside each task) |
| C2 (primary) | S0 + the capabilities the loop got ACTIVE |
| C3 | S0 + construction + the acquired capabilities |
| Crand | S0 + the capabilities the random-proposal control got ACTIVE |
| Coracle | S0 + designer-built artifacts with the exact structures of the five gap classes |
| C2noinit | C2 with the starting values that came from training tasks removed |

## 5. Hypotheses and decision rules

All intervals are 95% paired bootstrap intervals of the per-task score difference
(4,000 resamples). Real-data intervals resample whole datasets.

| | Claim | Supported if |
|---|---|---|
| H1 | Acquisition generalizes | at least one capability ACTIVE and C2 - C0 on `test` has ci_low > 0 |
| H2 | It transfers across regimes | C2 - C0 on `test_regime` has ci_low > 0 |
| H3 | It does not break what worked | C2 - C0 on `test_familiar` has ci_low > -0.01 |
| H4 | It composes | C2 - C0 on `test_compose` has ci_low > 0 |
| H5 | The diagnosis and research steps matter | the loop gets more capabilities ACTIVE than the random control, and C2 - Crand on `test` has ci_low > 0 (if Crand acquires nothing, H5 reduces to H1) |
| E1 | Blind novel classes (exploratory) | reported with its interval; no decision rule |
| E2 | Real data (exploratory) | reported with its interval; no decision rule |

H1 is the primary hypothesis. If no capability becomes ACTIVE, H1, H2, H4 and H5 are not
supported, and H3 holds trivially.

## 6. Reviewer gates (fixed; `bench/review.py`, GATES_SHA recorded in every run)

A capability becomes ACTIVE only if all of these pass, on data the loop never saw:

* integrity: artifact hash and code policy hold; the candidate equals the base plus exactly this artifact
* unit tests, re-run independently in a fresh subprocess
* no leakage: every seed the proposer saw lies in train/validation
* validation, recomputed from the stored records: ci_low > 0
* held-out gain on a fresh gate block of 1,200 tasks: ci_low > 0
* familiar regression on the familiar tasks of that block: ci_low > -0.01
* transfer gain on a fresh block of 400 regime-shifted tasks: ci_low > 0
* reproducible: 200 tasks re-run in fresh workers give identical per-task scores

The sample sizes come from a power analysis on a development pilot (dev split only):
per-task paired score differences had sd of about 0.13 (mixed), 0.06–0.10 (familiar) and
0.18 (gap). The thresholds were set before the pilot and were not changed.

## 7. Falsification tests

| | Threatening explanation | Test |
|---|---|---|
| F1 | Random search would do as well | Crand: random structures (sizes 1–3) with starting values fitted to randomly chosen failed training tasks, through the identical pipeline, at least as many proposals per round |
| F2 | Memorization of training tasks | the effect of the acquired capabilities on the training tasks they came from vs fresh test tasks; a large train–test gap means memorization |
| F3 | The "capability" is memorized parameter values, not structure | C2noinit vs C2 |
| F4 | The result depends on the process state | a cold re-evaluation from the registry file in a new interpreter with a different worker count must reproduce per-task scores exactly |
| F5 | The result is a fluke of one run | a full replication (replication 1: new tasks in every split, new registry, new reviewer state) |
| F6 | A scoring artifact | report score components (claim, interpolation, extrapolation, cost), recovery rates and capped extrapolation error, not just the score |
| F7 | Leakage | the no-leakage gate, the count of seeds the proposer saw, and an independent code review of the information flow |
| F8 | The diagnosis is decorative | post-hoc agreement between the diagnosed failure mode and the structure research found |

## 8. Known limitations at registration

* The domain is tiny: one input, one output, simulated laws plus a small set of real series.
* The expression language and search are designer-written; capabilities are new
  hypothesis families inside that language (a narrow form of library learning).
* S0 sometimes flips its answer with credence near 1 at about ten measurements (the
  flexible description becomes assessable there with two residual degrees of freedom).
  This inflates per-task variance; the gate sizes were chosen with it.
* The LLM research layer of Quanta is not part of this experiment.
* Bootstrap intervals over many comparisons are not corrected for multiplicity; the
  decision rules above are the only confirmatory tests.

## 9. What would count as failure

Any of: no capability passes review; capabilities pass review but C2 does not beat C0 on
`test`; the random control does as well; the gain vanishes on replication; the gain is
confined to the training tasks; the per-task scores cannot be reproduced. Each of these
will be reported as the result, not worked around.
