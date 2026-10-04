# Self-improvement: protocol and results

## What is being improved

The **genome** (`src/quanta/genome.py`): the bounded set of strategy parameters that controls
how the research loop works. For the automated discovery loop these are:
- how many space-filling measurements to start with;
- how to choose experiments (random, space-filling, or where hypotheses disagree);
- the noise model (Gaussian, or robust Student-t);
- when to stop;
- how many adversarial falsification tests to run before accepting a leader;
- posterior tempering, for calibration;
- whether named laws must compete against an unnamed flexible curve, and whether to widen
  the hypothesis space when they all lose;
- whether to use priors learned from verified past outcomes.

The improver can also evaluate **code patches**, confined to `src/quanta/science/` and
benchmarked in a sandbox (see [SAFETY.md](SAFETY.md)). The recorded runs below used only
genome changes.

## Benchmark

`src/quanta/bench/`: each task hides a law y = f(x) on x ∈ [0.1, 10] behind a noisy
instrument; the system gets 30 measurements.

* 13 law types, drawn with realistic skew (simple relations are common).
* 30% of instruments produce gross errors on 10% of readings.
* 6% of laws need the extended hypothesis space (linear + sinusoid), and 10% lie outside
  every hypothesis family the system has. For those the correct answer is "none of the named
  laws".
* Tasks are rejection-sampled for **identifiability**: noise is capped at one third of the
  distance between the true law and its best mimic among laws with at most as many
  parameters. So "the correct answer" is well defined.
* Score per task = 0.6 × (1 − Brier(credence, correct)) + 0.4 × exp(−prediction NRMSE)
  − 0.1 × (measurements used / budget). The Brier term is a proper scoring rule:
  confidently wrong answers cost more than honest uncertainty. These weights were fixed
  before any improvement run.

Splits use disjoint seed ranges:

| split | used for |
|---|---|
| selection | comparing candidates against the current champion (paired, same tasks) |
| confirmation | replicating an apparent improvement on fresh tasks (a new block every generation) |
| test | final report only; never used for any decision |
| experience | "past tasks" whose verified outcomes populate memory |

## Protocol

```
evaluate champion on selection split
repeat for up to 8 generations:
    diagnose failures (accuracy by category, calibration, confidently-wrong rate, cost, confusions)
    propose up to 6 candidates (diagnosis rules + near-miss combinations + random exploration)
    for each candidate: evaluate on the same selection tasks; gate
    for the best (up to 2) that pass: re-evaluate champion and candidate on fresh confirmation
        tasks; gate again
    first candidate that passes both gates and is authorized by the governor is adopted
    stop after 3 consecutive generations without an adoption (diminishing returns)
report initial vs final on the held-out test split
```

**Gate:** the lower bound of a 95% paired percentile-bootstrap CI on the mean score difference
must be above 0, and accuracy in any task category with at least 8 tasks may not drop by more
than 0.10.

## Run 1 (`experiments/run1_self_improvement/`)

Selection and confirmation splits of 120 tasks; test split of 400.

* Adopted one change: **2 adversarial falsification rounds** before accepting a leading
  hypothesis (selection Δ +0.034, replicated Δ +0.016).
* Held-out: mean score 0.7225 → 0.7517 (Δ +0.029, 95% CI [+0.017, +0.043]); accuracy
  0.710 → 0.745; Brier 0.244 → 0.199; ECE 0.242 → 0.176; confidently-wrong rate 0.215 → 0.158;
  cost 0.558 → 0.659.
* Stopped after 4 generations (3 without improvement).

Reading the run-1 lineage critically showed three problems with the **improver**:

1. **Misleading report.** Per task, the final genome won 83, lost 188 and tied 129.
   Falsification costs a few measurements on every task, but prevents a confident wrong
   answer on a minority. The mean (the declared objective) improved, but the typical task got
   slightly worse. The run-1 report quoted the sign test (p = 1.6e-10) as if it supported the
   improvement, when it actually measures the opposite direction. The report generator now
   states the direction of the win/loss balance. Run 1's report is kept unedited as a record.
2. **Wasted evaluations.** 4 of 24 candidates changed parameters that had no effect in the
   current configuration (e.g. `t_dof` while the likelihood was Gaussian), giving Δ exactly 0.
   The improver now only mutates parameters that are active.
3. **Low power.** Several candidates had positive mean effects with CIs straddling zero.
   Run 2 uses 200-task selection and confirmation splits, and the proposer now tries
   combinations of near-miss changes.

Run 1 also exposed a weakness in the **science engine** that is left unfixed (see Next steps).

These improver changes were driven by the selection-split lineage and by the reporting flaw,
not by test-split numbers. The test split was, however, looked at once (in run 1) before run 2.
Strictly speaking it is therefore no longer pristine for run 2.

## Run 2 (`experiments/self_improvement/`)

Code at commit `41d2aee` (the benchmark, science engine and improver have not changed since,
apart from the wording of one speculation line in the report generator). Selection and
confirmation splits of 200 tasks; test split of 400.

| generation | outcome |
|---|---|
| 1 | **adopted** `falsification_rounds 0 → 2`: selection Δ +0.026 [+0.010, +0.044], replicated Δ +0.017 [+0.004, +0.032] |
| 2 | **adopted** `likelihood gaussian → student_t`: selection Δ +0.014 [+0.002, +0.028], replicated Δ +0.013 [+0.001, +0.026]. `flexible_baseline + expand_on_inadequacy` passed selection (Δ +0.034 [+0.002, +0.067]) but **failed replication** (Δ +0.024 [−0.004, +0.054]) |
| 3 | nothing passed; all three near-miss combinations were neutral or negative |
| 4 | `use_learned_priors` passed selection (Δ +0.005 [+0.002, +0.009]) but **failed replication** (Δ +0.001 [−0.003, +0.005]) |
| 5 | nothing passed → stopped: 3 generations without a confirmed improvement |

30 candidates were evaluated. 4 passed the selection gate, and 2 of those replicated.

**Held-out test (400 tasks, never used for decisions):**

| metric | initial | final |
|---|---|---|
| mean score | 0.7225 | **0.7618** (Δ +0.039, 95% CI [+0.025, +0.054]) |
| accuracy | 0.710 | 0.752 |
| Brier score (lower is better) | 0.244 | 0.189 |
| expected calibration error | 0.242 | 0.177 |
| confidently wrong (credence > 0.8) | 21.5% | 14.7% |
| accuracy, clean instruments | 0.844 | 0.880 |
| accuracy, faulty instruments | 0.768 | 0.853 |
| accuracy, law outside initial hypothesis space | 0.000 | 0.000 |
| measurements used (fraction of budget) | 0.558 | 0.647 |

Per task, the final genome scores higher on 157 tasks and lower on 243 (sign test p = 2e-5).
Extra falsification measurements cost a little on most tasks; avoiding confident errors gains
a lot on a minority. The improvement is in the mean score, the pre-declared objective. It is
not in the typical task.

**What did not improve.** The 14% of tasks whose law lies outside the initial hypothesis space
are still never answered correctly. The change that addresses them (flexible baseline +
expansion) helps those tasks but costs accuracy on clean tasks, through the look-elsewhere
effect described below, and it did not survive replication. This is the largest remaining
failure mode.

Attempts to falsify these conclusions (ablations, a random-search control, a null check, the
learning curve) are in [experiments/ANALYSIS.md](../experiments/ANALYSIS.md) and summarized below.

### What is established (measured on the 400 held-out tasks)

* The final genome has a higher mean score: Δ +0.039, 95% CI [+0.025, +0.054].
* **Ablations:** both adopted changes are needed. Reverting falsification costs Δ −0.035
  [−0.049, −0.021]; reverting the robust likelihood costs Δ −0.010 [−0.020, −0.000].
* **Null check:** the gate rejects a comparison of a genome with itself.
* **Random-search control** (same loop and budget, random proposals): it also improved the
  system, Δ +0.026 [+0.015, +0.037], adopting 1 falsification round and learned priors. The
  diagnosis-driven final genome beat the random-search final genome by Δ +0.014
  [+0.004, +0.023].
* **Calibration improved but is still poor.** Answers given with credence above 0.8 (301 of
  400) have mean credence 0.96 and are correct 80% of the time (before: 0.96 vs 74%).
* **Learning curve** for priors learned from verified past outcomes: 10 or 30 past tasks
  *hurt* (Δ −0.006, CI excludes 0); 100 and 300 past tasks help (Δ +0.009 [+0.005, +0.015],
  +0.008 [+0.002, +0.015]).

### What is inferred

* Most of the gain comes from adversarial falsification. It turns confident errors into
  either corrections or honest uncertainty: the confidently-wrong rate fell from 21.5% to
  14.7%, and the ablation shows it is the larger contributor.
* The robust likelihood helps through faulty instruments: accuracy on those tasks rose from
  0.768 to 0.853.
* Diagnosis-driven proposals beat random search at equal budget here. This rests on one run of
  each. The CI covers task sampling, not the randomness of the search itself, and random search
  also found the most valuable change (falsification) in its first generation, because this
  strategy space is small.
* Learned priors need on the order of 100 verified outcomes before they help. With fewer, the
  frequency estimates are noisy enough to steer the posterior the wrong way (the Dirichlet
  pseudocount of 2 is too weak). This also explains why `use_learned_priors`, which uses 60
  past tasks, replicated in the control run but not in run 2: its effect is marginal at that
  sample size.

### Speculation (not tested)

* That these strategy lessons carry over to the LLM-driven research loop.
* Why robust likelihood + flexible baseline interacts badly (see Next steps).
* That correcting the look-elsewhere penalty would let the flexible baseline survive
  replication and solve the out-of-space category.

## Next steps the evidence points to

* **Look-elsewhere effect in periodic hypotheses.** Once the hypothesis space is widened, linear
  laws are sometimes misidentified as linear + sinusoid. The sinusoid's frequency is found by a
  58-point search, which buys more freedom to fit noise than BIC's one-parameter penalty
  assumes. A principled fix is to penalize the frequency by the effective number of independent
  frequencies searched. This is a code change in `src/quanta/science/`, the kind of candidate
  the sandboxed patch path exists for.
* **Robust likelihood with a flexible baseline.** Combining Student-t noise with the flexible
  spline baseline made results worse in both runs, with many named laws answered as "none".
  The cause is not established. *Speculation:* robust down-weighting shrinks the spline's scale
  estimate more than the named laws', inflating its likelihood.
* **Calibration is entangled with stopping.** Posterior tempering lowers stated credence, but the
  stop rule reads the same tempered posterior, so tempering also delays stopping and spends
  measurements. That is plausibly why no temperature change ever passed the gate, even though
  the system stays overconfident. Separate the decision posterior from the reported credence,
  and calibrate the latter post hoc from verified outcomes in memory (e.g. isotonic
  regression). That would turn LEARN into better KNOW-WHEN-IT-DOESN'T-KNOW.
* **LLM proposer and code patches.** Implemented and tested with a scripted model, not yet run
  against Claude.

## Reproduce

```bash
pip install -e .
PYTHONPATH=src python -m quanta.cli improve --run-dir experiments/self_improvement \
    --generations 8 --candidates 6 --n-selection 200 --n-confirmation 200 --n-test 400 --jobs 4 --seed 0
PYTHONPATH=src python experiments/analyze.py
```

Everything is deterministic given the seed; `--jobs` does not change results.
