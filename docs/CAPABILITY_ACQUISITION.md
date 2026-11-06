# Capability acquisition

Can Quanta acquire a capability its designers did not give it, and can we show that
without fooling ourselves? This document describes the mechanism, the evaluation that is
meant to keep it honest, and the results. The pre-registered design is
[experiments/capability_acquisition/PREREGISTRATION.md](../experiments/capability_acquisition/PREREGISTRATION.md);
results are in [Results](#results).

## What counts as a capability

A capability is a **reusable, versioned code artifact** that changes what the solver can
represent, together with the evidence for keeping it. Concretely it is a new hypothesis
family: a law shape `y = c0 + c1·t1(x; p) [+ c2·t2(x; p)]` whose terms `t_i` are generated
expressions in a small language (x, exp, log, sin, 1/(·+b), (·+b)², products, and earlier
capabilities as building blocks). Any task can use it; it is not tied to the task it came from.

Every capability record (`src/quanta/capabilities/registry.py`) carries:

| Field | Content |
|---|---|
| description, origin | what it is; which run and round produced it |
| triggering failure | the failed training tasks, their diagnosed failure modes, an example diagnosis |
| hypothesis | the missing-capability hypothesis it answers |
| artifact | the generated source per term, parameter counts, starting values, sha256 |
| prerequisites, dependencies | the research tools it needs; earlier capabilities it builds on |
| tests | independent unit tests (policy, load, finiteness, self-recovery, determinism, runtime) |
| selection evidence | the paired validation comparison the loop used to choose it |
| held-out evidence, transfer evidence | the reviewer's results on fresh gate and transfer blocks |
| review | the reviewer's decision, every gate's value and threshold, the gates' sha256 |
| retry evidence | self-detected failures on the training tasks before and after adoption |
| limitations, confidence, status, status history | EXPERIMENTAL, ACTIVE, REJECTED or DEPRECATED, with reasons |

The registry is an append-only event log: nothing is overwritten, and the current view is
a replay of the log. ACTIVE requires an approving review id; REJECTED and DEPRECATED are final.

## The loop

```
SOLVE training tasks (outputs and observations only; no grades)
  -> EVALUATE itself: no law fit / low credence / a confident law with significant residual structure
  -> DIAGNOSE each failure (a hypothesis, with evidence and confidence)
  -> GAP HYPOTHESES, ranked by how many failures they would explain
  -> RESEARCH: program search on each failed task's own observations; keep a law only if it
     predicts that task's held-out observations better than the flexible description
  -> DESIGN: structures that recur across >= 2 failed tasks become candidates
  -> IMPLEMENT the artifact -> UNIT TEST in a fresh subprocess
  -> GENERALIZATION TEST on the validation split (paired); review only if ci_low > 0
  -> INDEPENDENT REVIEW: held-out gain, familiar regression, transfer, reproducibility,
     leakage, integrity, on blocks the loop never sees
  -> ACCEPT (ACTIVE) or REJECT -> STORE (registry and memory, failures included)
  -> RETRY the training tasks with the new toolkit -> MEASURE what it resolved
```

A diagnosis has the fields TASK, CURRENT APPROACH, OBSERVED FAILURE, LIKELY FAILURE MODE,
KNOWN CAPABILITIES USED, MISSING CAPABILITY HYPOTHESIS, EVIDENCE, CONFIDENCE and PROPOSED
NEXT EXPERIMENT (`capabilities/diagnosis.py`). It is recorded as a hypothesis, and afterwards
checked against what research found (diagnosis consistency).

## What is designer-provided and what is acquired

Stated plainly, because this is where claims of "self-improvement" usually go wrong.

| Designer-provided | Acquired by the loop |
|---|---|
| the expression language and the program search | which structures to build |
| the failure detectors and the diagnosis rules | from which failures, with which starting values |
| the unit tests, the reviewer, the gates, the score | whether a structure survives review |
| the starting toolkit (ten law families, a flexible description) | the capabilities added to it |

The loop is never told a law's name, a task's label or a grade. It cannot express anything
outside the expression language. This is a narrow form of library learning (related to
DreamCoder-style abstraction learning), not open-ended self-modification.

## Keeping it honest

The reviewer and the evaluation code are outside the loop's authority.

* **Process isolation.** The solver, with whatever capabilities it is being tested with,
  runs in worker subprocesses. The parent process holds the tasks (truth lives in a
  closure), answers measurement requests, enforces the budget (its count is authoritative)
  and grades. Workers receive measurements, the points to predict, the budget, a task
  identifier and a solver seed, never parameters or truth. (An independent review pointed
  out that the identifier and the seed, which is an invertible function of the task seed,
  would let a *malicious* worker running as the same user regenerate its task by importing
  the evaluator. The solver and the artifacts do not do this, but the isolation is only as
  strong as that; see *Residual risks*.)
* **Frozen scoring code.** The fitting code the task generator and grader depend on is a
  protected copy (`bench/frozen.py`), and the reviewer checks artifacts against a frozen
  copy of the code policy (`bench/policy.py`).
* **Code policy.** Artifact source must be a single arithmetic expression over `x`,
  parameters `p[i]` and four helper functions; anything else is rejected before loading.
* **Information flow.** The loop sees training outputs (no grades) and paired validation
  scores. Only the reviewer touches the gate and transfer splits, each review on a fresh,
  never-reused block. The test, sealed and real-data splits are used only for the final report.
* **Leakage gate.** Every seed the proposer has seen must lie in train/validation.
* **Pre-registration.** Hypotheses, decision rules, sample sizes and controls were committed
  before the first run.
* **Controls.** A random-proposal control through the identical pipeline; a designer-built
  oracle; an ablation that strips training-derived starting values; a cold re-evaluation in a
  fresh interpreter; a full replication on new tasks.
* **Blind and external tests.** Law classes written by an independent agent that the
  experimenters did not read; 29 real datasets chosen by a fixed rule.

### Residual risks

* Task identity is recoverable inside a worker: the task identifier names the split and
  index, and the solver seed is an invertible function of the task seed, so code in a worker
  that imported the evaluator could regenerate the hidden law. Nothing in this experiment's
  solver or artifacts does so (artifacts are arithmetic expressions; the solver is designer
  code). Closing it needs opaque job identifiers, a keyed seed derivation and an OS-level
  boundary; changing the first two would change every random stream, so they are scheduled
  for after the pre-registered runs.
* Workers run as the same OS user. Capability artifacts cannot do anything but arithmetic,
  and the solver itself is designer code, so this is acceptable here. Evaluating
  model-written *code* (rather than expressions) would need an OS-level sandbox.
* In-process guards such as the registry's ACTIVE rule are an audit trail, not a security
  boundary against code running in the same process.
* The reviewer's "validation" gate re-checks the proposer's own selection data, so it is
  selection-biased by construction. The held-out and transfer gates are the real tests.

## Results

Pending the pre-registered runs (see the experiment directory); this section is filled in
from `results/rep*/analysis.json` and nothing else.
