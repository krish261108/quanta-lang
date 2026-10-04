# Quanta

A governed, self-improving harness for long-horizon research and engineering problems.

You give it a problem with no prescribed solution. It frames the problem, researches what is
known, forms competing hypotheses, plans, builds and runs experiments, tries to falsify its
own conclusions, and iterates until more effort stops paying off. It then delivers a report
that separates **established facts**, **inferences** and **speculation**, with evidence,
limitations and reproduction steps. Separately, it can improve its own strategy, and it
keeps a change only when that change measurably performs better on tasks it has never seen.

**What this is not.** It is not a general intelligence. Open-ended reasoning, reading,
coding and perception come from the underlying model (Claude). Quanta supplies what a model
alone does poorly over long horizons:
- durable memory and state;
- explicit competing hypotheses and discriminating experiments;
- an evidence ledger that will not let an unsupported claim be called a fact;
- independent verification;
- calibrated uncertainty;
- hard limits on what can happen without a human;
- an improvement loop that cannot weaken its own oversight.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) maps each requirement to what implements it and
how far it has been verified, including what is **not** implemented.

## The loop

```
frame → research → hypothesize → plan → execute → update → verify → decide → deliver
                        ▲                                              │
                        └────────────── revise (next iteration) ◄──────┘
```

It runs in two forms:

* **`quanta solve`**: the open-ended loop driven by Claude, with specialist sub-agents,
  governed tools (shell, Python, files, web), checked citations, and a fresh-context verifier.
  Every phase and every task is checkpointed; `quanta resume` continues after a crash.
* **`quanta discover`**: the same methodology fully automated, for systems that can be queried
  experimentally. Hypotheses are executable models. Experiments go where the hypotheses
  disagree most. Leaders are attacked before they are accepted. An unnamed flexible curve
  competes against the named laws, so "none of these" is a possible answer. Credences are
  calibrated. This form needs no API key, which is what makes the methodology and its
  self-improvement **measurable**.

## Results

Self-improvement on the discovery benchmark (a controlled micro-world: identify a hidden law
behind a noisy, sometimes faulty instrument, within 30 measurements). All numbers are on a
held-out test split that played no part in any decision. Paired 95% bootstrap CIs. Details,
caveats and the full lineage are in [docs/SELF_IMPROVEMENT.md](docs/SELF_IMPROVEMENT.md) and
[experiments/](experiments/).

| held-out test, 400 tasks | initial strategy | after self-improvement |
|---|---|---|
| mean score | 0.7225 | **0.7618** (Δ +0.039, CI [+0.025, +0.054]) |
| accuracy | 0.710 | 0.752 |
| confidently wrong (credence > 0.8) | 21.5% | 14.7% |
| Brier score (lower is better) | 0.244 | 0.189 |
| accuracy with faulty instruments | 0.768 | 0.853 |
| accuracy when the law is outside the initial hypothesis space | 0% | **0% (unsolved)** |

* **Adopted:** two adversarial falsification tests before accepting a leading hypothesis, and a
  robust (Student-t) noise model. Each passed a paired bootstrap gate and replicated on fresh
  tasks. Ablations show both are needed.
* **Rejected after looking good once:** a flexible "none of the above" baseline and learned
  priors. Both passed the first gate and failed replication.
* **Controls:**
  - The gate rejects a null (A/A) comparison.
  - Random search at the same budget also improves the system (+0.026), but less than
    diagnosis-driven search (difference +0.014, CI [+0.004, +0.023]; one run each).
* **Learning:** priors from 100+ verified past tasks help (+0.009); priors from 10–30 hurt (−0.006).
* **Honest caveats:**
  - Most tasks pay a small cost in extra measurements (157 wins vs 243 losses per task); the
    gain comes from avoided confident errors.
  - Calibration is better but still overconfident: credence 0.96 at 80% accuracy.
  - An earlier run's report misused a sign test; that was caught and fixed, and both runs are
    kept.

## Quick start

```bash
pip install -e '.[dev]'            # core: standard library only
pytest -q                          # 119 tests, ~30 s

# Offline: no API key needed
quanta discover --seed 30002       # one discovery run, with full report
quanta bench --n 200               # evaluate the default strategy
quanta improve --run-dir runs/improve       # benchmark-gated self-improvement
quanta learn-curve                 # score vs. verified experience in memory

# Open-ended research with Claude
pip install -e '.[llm]'
export ANTHROPIC_API_KEY=...
quanta solve "Why does our nightly ETL job take 3x longer on Mondays? Repo is in ./etl" \
    --workspace ./etl --human --approve interactive --max-cost 10
touch runs/latest/STOP             # halt at the next step
quanta resume runs/latest          # continue later
```

`--approve interactive` asks you before any high-risk action (installing packages,
deleting files, network writes, pushing). The default denies them. `--allow-shell-prefix
"pytest"` pre-authorizes a specific command family. Critical actions can never be
pre-authorized.

## Repository layout

```
src/quanta/
  epistemics.py   evidence ledger: fact / inference / speculation, enforced
  governance.py   risk tiers, approvals, budgets, kill switch, audit log
  memory.py       episodic / semantic / procedural memory (SQLite + FTS5)
  planning.py     long-horizon task graphs
  uncertainty.py  calibration, when to ask a human, diminishing returns
  science/        executable hypotheses and the automated discovery loop
  bench/          benchmark tasks, grader, harness   (protected from self-modification)
  llm.py          Claude backend + scripted test model
  tools.py        governed tools; executions become evidence
  perception.py   text, code, documents, images, PDFs, video frames, environment
  agent.py        tool-use loop, roles, sub-agent delegation
  verify.py       independent verification
  solver.py       the open-ended research loop
  improve.py      benchmark-gated self-improvement
  genome.py       the bounded strategy space self-improvement may tune
docs/             architecture and capability map, safety model, self-improvement results
experiments/      recorded runs, lineage, analysis (ablations, controls, learning curve)
```

## Limitations

* The measured results come from one controlled task family (1-D black-box law discovery).
  They show the methodology and the improvement process work there; they do not show
  general research ability.
* The Claude-driven research loop is tested end to end with a scripted model only. Its
  real-world quality has not been measured here (no API key was available while building it).
* Audio content, GUI/browser automation and multi-day runs are not implemented or not tested.
  See the capability map.
* Shell risk classification is heuristic. Run the agent in a container or VM. See
  [docs/SAFETY.md](docs/SAFETY.md).
