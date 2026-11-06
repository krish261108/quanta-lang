# Self-improvement report

**Problem.** Improve the strategy of the hypothesis-driven discovery loop, keeping a change only if it demonstrably performs better.

## Result

2 change(s) adopted over 6 generation(s). Held-out mean score 0.7225 -> 0.7482 (Δ +0.0257, 95% CI [+0.0157, +0.0378]).

## Established facts

_Claims grounded in direct observation, executed checks, verified sources or independent verification._

- **C3** On 400 held-out tasks the initial genome scored 0.7225 and the final genome 0.7482 (paired difference +0.0257; 156 wins / 195 losses / 49 ties).  
  evidence: E1 (execution: held-out test run of initial genome (4b3aedaa9a43)); E2 (execution: held-out test run of final genome (710569be9fc5))
- **C4** Accuracy 0.710 -> 0.743; Brier 0.244 -> 0.209; ECE 0.242 -> 0.203; confidently-wrong rate 0.215 -> 0.177; mean cost 0.558 -> 0.621.  
  evidence: E1 (execution: held-out test run of initial genome (4b3aedaa9a43)); E2 (execution: held-out test run of final genome (710569be9fc5))

## Inferences

_Derived from facts by reasoning or statistics; can be wrong if a premise or the method is wrong._

- **C6** (p=0.95) The final genome has a higher mean score than the initial one on this task distribution (95% CI on the mean improvement: [+0.0157, +0.0378]). However, it scored LOWER on more tasks than it scored higher (195 vs 156; two-sided sign test p=0.042): the mean gain comes from large improvements on a minority of tasks while most tasks pay a small cost.  
  from C3, C4; evidence E5 (derivation: paired bootstrap on held-out per-task scores)
- **C7** (p=0.90) Adopted change v1 ({'falsification_rounds': (0, 1)}) helped: it passed the gate on the selection split (delta +0.0230) and replicated on fresh tasks (delta +0.0112). Rationale: exploration: random local change to falsification_rounds  
  from C3
- **C8** (p=0.90) Adopted change v2 ({'use_learned_priors': (False, True)}) helped: it passed the gate on the selection split (delta +0.0042) and replicated on fresh tasks (delta +0.0051). Rationale: exploration: random local change to use_learned_priors  
  from C3

## Speculation and open hypotheses

_Not established. Listed so they are not mistaken for findings._

- **C9** (p=0.40) The adopted strategy changes (falsification_rounds, use_learned_priors) would also improve LLM-driven open-ended research. Not tested here.

## Falsification record

- No falsification tests were run.

## Adopted changes

- v1 (generation 1): `{'falsification_rounds': (0, 1)}` — exploration: random local change to falsification_rounds (selection Δ +0.0230, confirmation Δ +0.0112, CI [+0.0015, +0.0238])
- v2 (generation 3): `{'use_learned_priors': (False, True)}` — exploration: random local change to use_learned_priors (selection Δ +0.0042, confirmation Δ +0.0051, CI [+0.0005, +0.0109])

## Final genome changes vs. initial

```
{
 "falsification_rounds": [
  0,
  1
 ],
 "use_learned_priors": [
  false,
  true
 ]
}
```

## Limitations

- The benchmark is a controlled micro-world (1-D black-box law discovery); gains are measured only on its task distribution.
- Changes are to strategy parameters (and optionally sandboxed code in mutable paths); the improver cannot change the grader, the gate or governance, so it cannot discover improvements that would require changing them.
- 29 candidate(s) failed the selection gate and 0 failed replication; a stricter or looser gate would change what is adopted.
- Proposals come from fixed diagnostic rules plus random exploration unless an LLM proposer is used; the search is local and greedy.
- Stopped because: diminishing returns: no confirmed improvement in 3 consecutive generations.
- Per-task wins/losses on the held-out split are 156/195: the improvement is in the mean (the pre-declared objective), not in the typical task.

## Reproducibility

- **code_version:** `7effa84+dirty`
- **python:** `3.11.15`
- **platform:** `Linux-6.18.44-fc-v64-x86_64-with-glibc2.39`
- **command:** `python -m quanta.cli improve --run-dir <dir>  (see docs/SELF_IMPROVEMENT.md)`
- **seed:** `0`
- **n_selection:** `200`
- **n_confirmation:** `200`
- **n_test:** `400`
- **initial_genome:** `4b3aedaa9a43`
- **final_genome:** `710569be9fc5`
- **gate:** `{"min_effect": 0.0, "confidence": 0.95, "max_category_regression": 0.1, "min_category_size": 8, "n_boot": 4000}`
