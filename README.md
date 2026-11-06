# Quanta

A governed research harness with a measured self-improvement loop, and a pre-registered
test of whether it can acquire capabilities its designers did not give it.

You give it a problem with no prescribed solution. It frames the problem, researches what is
known, forms competing hypotheses, plans, builds and runs experiments, tries to falsify its
own conclusions, and iterates until more effort stops paying off. It then delivers a report
that separates **established facts**, **inferences** and **speculation**, with evidence,
limitations and reproduction steps. Separately, it can improve its own strategy, keeping a
change only when that change measurably performs better on tasks it has never seen, and it
can try to build new capabilities from its own failures, keeping one only if an
independent reviewer confirms it helps on fresh and different tasks.

**What this is not.** It is not AGI or ASI, and it is not generally intelligent. Its measured
abilities are confined to small, mostly simulated law-discovery problems. Open-ended
reasoning, reading, coding and perception would come from the underlying language model,
and that path has **never been run against a real model**. Quanta supplies structure a
model alone handles poorly over long horizons:
- durable memory and state;
- explicit competing hypotheses and discriminating experiments;
- an evidence ledger that requires grounding evidence before a claim is called a fact (it
  checks the *kind* of evidence, not its relevance);
- independent verification;
- calibrated uncertainty;
- limits on what can happen without a human (in-process guards: run it in a container for
  real isolation; see [docs/SAFETY.md](docs/SAFETY.md));
- improvement loops that are not allowed to change the evaluator or the oversight code.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) maps each requirement to what implements it and
how far it has been verified. [docs/RESEARCH_PROGRAM.md](docs/RESEARCH_PROGRAM.md) records
the audit, the evidence table and a falsifiable roadmap.

## The loop

```
frame → research → hypothesize → plan → execute → update → verify → decide → deliver
                        ▲                                              │
                        └────────────── revise (next iteration) ◄──────┘
```

It runs in two forms:

* **`quanta solve`**: the open-ended loop driven by a language model, with specialist
  sub-agents, governed tools (shell, Python, files, web), checked citations, and a
  fresh-context verifier. Checkpointed; `quanta resume` continues after a crash.
  Plumbing tested with scripted models only.
* **`quanta discover`**: the same methodology fully automated, for systems that can be queried
  experimentally. Hypotheses are executable models. Experiments go where the hypotheses
  disagree most. Leaders are attacked before they are accepted. An unnamed flexible curve
  competes against the named laws, so "none of these" is a possible answer. This form needs
  no API key, which is what makes the methodology and its self-improvement **measurable**.

## Results

### Strategy self-improvement (L1)

The discovery benchmark is a controlled micro-world: identify a hidden law behind a noisy,
sometimes faulty instrument within 30 measurements. Held-out test split (400 tasks) that
played no part in any decision; paired 95% bootstrap intervals.

**Correction.** The numbers first published here were produced by a code version that was
never committed, and they do not reproduce from the repository. Re-running the same
comparisons with the committed code
([experiments/reverify_legacy](experiments/reverify_legacy/results.json)) shifts both
strategies up by about 0.034. The improvement and both ablations reproduce:

| held-out test, 400 tasks | initial strategy | after self-improvement |
|---|---|---|
| mean score | 0.756 | **0.798** (Δ +0.042, CI [+0.027, +0.057]; first published: +0.039) |
| accuracy | 0.738 | 0.785 |
| confidently wrong (credence > 0.8) | 17.3% | 10.0% |
| Brier score (lower is better) | 0.217 | 0.161 |
| accuracy with faulty instruments | 0.800 | 0.916 |
| accuracy when the law is outside the initial hypothesis space | 0% | 0% |

* **Adopted:** two adversarial falsification tests before accepting a leading hypothesis
  (reverting it: −0.039, CI [−0.054, −0.025]) and a robust Student-t noise model
  (reverting it: −0.011, CI [−0.021, −0.0001]).
* **Not yet re-verified with the committed code:** the random-search control (published:
  random search also improves, by less than diagnosis-driven search, +0.014, CI [+0.004,
  +0.023], one run each) and the learning-from-experience curve. The "verified"
  experience behind learned priors comes from the grader's ground truth, i.e. oracle labels.
* **Caveat:** most tasks pay a small cost in extra measurements (166 wins vs 234 losses per
  task); the gain comes from avoided confident errors.

### The 0% category

The 0% above was structural: the adopted strategy could not answer "none" and never
considered the extended families. A pre-specified ladder on the same test split
([experiments/zero_category](experiments/zero_category/RESULTS.md)) shows the flexible
baseline plus expansion reaches **67%** on outside-space laws, and the look-elsewhere
correction removes false periodic detections. No configuration beats the champion's overall
score: the gains outside the space are paid for on clean tasks.

### Capability acquisition (L2, narrow)

Can Quanta build a capability it was not given from its own failures, and does that
capability survive an independent reviewer and help on fresh, transferred, blind-novel and
real-data tasks, better than random proposals? Pre-registered in
[experiments/capability_acquisition/PREREGISTRATION.md](experiments/capability_acquisition/PREREGISTRATION.md);
mechanism and results in [docs/CAPABILITY_ACQUISITION.md](docs/CAPABILITY_ACQUISITION.md).

## Quick start

```bash
pip install -e '.[dev]'            # core: standard library only
pytest -q                          # ~150 tests, ~30 s

# Offline: no API key needed
quanta discover --seed 50002       # one discovery run, with full report
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
  science/        executable hypotheses, the discovery loop, program search (synth.py),
                  capability artifacts (toolkit.py), the isolated solver worker
  capabilities/   capability registry, gap diagnosis, artifact tests, the acquisition loop
  bench/          benchmarks, graders, isolation, the independent reviewer, real-data and
                  blind test suites   (protected from self-modification)
  llm.py          Claude backend + scripted test model
  tools.py        governed tools; executions become evidence
  perception.py   text, code, documents, images, PDFs, video frames, environment
  agent.py        tool-use loop, roles, sub-agent delegation
  verify.py       independent verification
  solver.py       the open-ended research loop
  improve.py      benchmark-gated self-improvement
  genome.py       the bounded strategy space self-improvement may tune
docs/             architecture and capability map, safety model, self-improvement and
                  capability-acquisition results, research program
experiments/      recorded runs, lineage, analysis, pre-registered experiments, re-verification
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
