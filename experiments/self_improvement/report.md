# Self-improvement report

**Problem.** Improve the strategy of the hypothesis-driven discovery loop, keeping a change only if it demonstrably performs better.

## Result

2 change(s) adopted over 5 generation(s). Held-out mean score 0.7225 -> 0.7618 (Δ +0.0393, 95% CI [+0.0254, +0.0540]).

## Established facts

_Claims grounded in direct observation, executed checks, verified sources or independent verification._

- **C3** On 400 held-out tasks the initial genome scored 0.7225 and the final genome 0.7618 (paired difference +0.0393; 157 wins / 243 losses / 0 ties).  
  evidence: E1 (execution: held-out test run of initial genome (4b3aedaa9a43)); E2 (execution: held-out test run of final genome (73a490250f5e))
- **C4** Accuracy 0.710 -> 0.752; Brier 0.244 -> 0.189; ECE 0.242 -> 0.177; confidently-wrong rate 0.215 -> 0.147; mean cost 0.558 -> 0.647.  
  evidence: E1 (execution: held-out test run of initial genome (4b3aedaa9a43)); E2 (execution: held-out test run of final genome (73a490250f5e))

## Inferences

_Derived from facts by reasoning or statistics; can be wrong if a premise or the method is wrong._

- **C6** (p=0.95) The final genome has a higher mean score than the initial one on this task distribution (95% CI on the mean improvement: [+0.0254, +0.0540]). However, it scored LOWER on more tasks than it scored higher (243 vs 157; two-sided sign test p=2e-05): the mean gain comes from large improvements on a minority of tasks while most tasks pay a small cost.  
  from C3, C4; evidence E5 (derivation: paired bootstrap on held-out per-task scores)
- **C7** (p=0.90) Adopted change v1 ({'falsification_rounds': (0, 2)}) helped: it passed the gate on the selection split (delta +0.0259) and replicated on fresh tasks (delta +0.0165). Rationale: 16% of answers were confidently wrong: test the leading hypothesis adversarially before accepting it  
  from C3
- **C8** (p=0.90) Adopted change v2 ({'likelihood': ('gaussian', 'student_t')}) helped: it passed the gate on the selection split (delta +0.0136) and replicated on fresh tasks (delta +0.0134). Rationale: accuracy with corrupted measurements (0.83) trails clean data (0.94): use a heavy-tailed noise model  
  from C3

## Speculation and open hypotheses

_Not established. Listed so they are not mistaken for findings._

- **C9** (p=0.40) The same strategy changes (adversarial falsification, robust likelihoods, a flexible 'none of the above' baseline, calibrated credence) would improve LLM-driven open-ended research. Not tested here.

## Falsification record

- No falsification tests were run.

## Adopted changes

- v1 (generation 1): `{'falsification_rounds': (0, 2)}` — 16% of answers were confidently wrong: test the leading hypothesis adversarially before accepting it (selection Δ +0.0259, confirmation Δ +0.0165, CI [+0.0043, +0.0315])
- v2 (generation 2): `{'likelihood': ('gaussian', 'student_t')}` — accuracy with corrupted measurements (0.83) trails clean data (0.94): use a heavy-tailed noise model (selection Δ +0.0136, confirmation Δ +0.0134, CI [+0.0013, +0.0255])

## Final genome changes vs. initial

```
{
 "likelihood": [
  "gaussian",
  "student_t"
 ],
 "falsification_rounds": [
  0,
  2
 ]
}
```

## Limitations

- The benchmark is a controlled micro-world (1-D black-box law discovery); gains are measured only on its task distribution.
- Changes are to strategy parameters (and optionally sandboxed code in mutable paths); the improver cannot change the grader, the gate or governance, so it cannot discover improvements that would require changing them.
- 26 candidate(s) failed the selection gate and 2 failed replication; a stricter or looser gate would change what is adopted.
- Proposals come from fixed diagnostic rules plus random exploration unless an LLM proposer is used; the search is local and greedy.
- Stopped because: diminishing returns: no confirmed improvement in 3 consecutive generations.
- Per-task wins/losses on the held-out split are 157/243: the improvement is in the mean (the pre-declared objective), not in the typical task.

## Reproducibility

- **code_version:** `41d2aee+dirty`
- **python:** `3.11.15`
- **platform:** `Linux-6.18.44-fc-v64-x86_64-with-glibc2.39`
- **command:** `python -m quanta.cli improve --run-dir <dir>  (see docs/SELF_IMPROVEMENT.md)`
- **seed:** `0`
- **n_selection:** `200`
- **n_confirmation:** `200`
- **n_test:** `400`
- **initial_genome:** `4b3aedaa9a43`
- **final_genome:** `73a490250f5e`
- **gate:** `{"min_effect": 0.0, "confidence": 0.95, "max_category_regression": 0.1, "min_category_size": 8, "n_boot": 4000}`
