# Architecture

Quanta is a harness, not a mind, and not a general intelligence. A language model (Claude,
via `quanta.llm.AnthropicModel`) would supply open-ended reasoning, reading, writing and
coding; that path is tested only with scripted models. Deterministic code around it supplies
what models do badly on their own over long horizons: durable state, explicit competing
hypotheses, discriminating experiments, an evidence ledger that requires grounding evidence
before a claim is called a fact, independent verification, calibrated uncertainty, in-process
limits on what may be done without a human, a way to improve its own strategy that only keeps
changes that measurably help, and a way to acquire new hypothesis families from its failures
that only keeps those an independent reviewer approves.

## The loop

```
            ┌──────────── revise (next iteration) ◄────────────┐
            ▼                                                  │
 frame → research → hypothesize → plan → execute → update → verify → decide ──► deliver
   │         │           │          │        │         │        │                  │
   │   cite_source   ≥k competing  DAG of   sub-agents  Bayes   fresh-context   report from
   │   (quotes are   falsifiable   tasks w/ per task,   from    verifier tries   the ledger:
   │    checked)     hypotheses    accept.  retries,    LRs,    to refute;      facts /
   │                               criteria decompose,  tied to grounded         inferences /
 ask a human only                           replan      ledger  support →        speculation,
 if expected loss                                               FACT             limitations,
 > cost of asking                                                                reproducibility
```

Two implementations of the same loop exist:

| | `quanta.solver.ResearchLoop` | `quanta.science.DiscoveryLoop` |
|---|---|---|
| Problems | open-ended, stated in natural language | black-box systems that can be queried |
| Hypotheses | written by the model, tracked as ledger claims | executable models (built-in law families, model-proposed formulas, a flexible baseline) |
| Experiments | tasks run by specialist sub-agents with tools | chosen by the harness where hypotheses disagree most |
| Belief update | model-stated likelihood ratios, Bayes in code | BIC model comparison, tempered posterior |
| Needs an LLM | yes | no |
| Measured | no (plumbing tested with a scripted model) | yes (benchmark, self-improvement, learning curve) |

The discovery loop is where the methodology can be measured. The research loop is where
it is meant to be used.

## Modules

| Module | Responsibility |
|---|---|
| `epistemics.py` | Evidence ledger. Facts need grounding evidence (observation, execution, verified source, verification). Inferences need premises and cannot be more confident than their weakest premise. Refutation propagates to dependents. |
| `governance.py` | Single authorization chokepoint: risk tiers, approvers, budgets, kill switch, hash-chained audit log. See [SAFETY.md](SAFETY.md). |
| `memory.py` | SQLite + FTS5 long-term memory: episodic, semantic, procedural. Agents can write memories but cannot mark them verified. |
| `planning.py` | Task DAGs: validation, readiness, bounded retries, failure propagation, hierarchical refinement, JSON checkpoints. Tested at 3,000 tasks. |
| `uncertainty.py` | Calibration (Brier, log score, ECE, reliability), expected-loss rule for asking humans, diminishing-returns stopping, self-consistency. |
| `science/hypotheses.py` | Hypotheses as fittable models; weighted/robust least squares; safe formula compiler (AST whitelist) + Nelder-Mead for model-proposed laws. |
| `science/discovery.py` | Automated frame → recall → experiment → analyze → revise → falsify → stop → deliver loop. |
| `bench/` | Identifiable law-discovery tasks, proper-scoring grader, parallel harness, data splits; the capability-acquisition evaluator (`acquisition.py`), process isolation (`isolation.py`), the evaluation service and independent reviewer (`review.py`), real-data and blind suites (`external.py`, `sealed.py`), frozen scoring code (`frozen.py`, `policy.py`). Protected from self-modification. |
| `science/synth.py` | Program search over a small expression language (variable projection fits, MDL score, beam search); learned abstractions as building blocks. |
| `science/toolkit.py`, `science/worker.py` | Capability artifacts as hypothesis families; the isolated solver worker and its JSON-lines protocol. |
| `capabilities/` | Versioned append-only capability registry, structured gap diagnosis, artifact building and subprocess unit tests, the acquisition loop and its random-proposal control. |
| `llm.py` | `AnthropicModel` (adaptive thinking, explicit effort, server-side web search/fetch, prompt caching, refusal fallbacks, `pause_turn` handling) and `ScriptedModel` for tests. History is append-only. |
| `tools.py` | Governed tools: files (workspace-confined, backed-up writes), shell (risk-classified), Python, HTTP fetch, citation checking, memory, ledger, human input, perception, hypothesis comparison. Executions become grounding evidence automatically. |
| `perception.py` | Text/code/docs, images and PDFs (as model content blocks), video key frames (ffmpeg), audio metadata (+ optional transcriber), environment snapshot. Reports explicitly when a modality was not perceived. |
| `agent.py` | Tool-use loop, specialist roles, structured submissions, sub-agent delegation with depth and budget limits, parallel execution. |
| `verify.py` | Fresh-context verifier sees claims and evidence, not reasoning; verdicts change the ledger mechanically. |
| `solver.py` | The open-ended research loop with checkpoint/resume. |
| `report.py` | Deliverable generated from the ledger. |
| `genome.py` | The bounded strategy space self-improvement may change. Contains no governance fields. |
| `improve.py` | Propose → paired evaluation → statistical gate → replication → authorization → adopt; held-out reporting. |

## Capability map

Status legend:
**Measured**: implemented and evaluated quantitatively.
**Tested**: implemented, covered by deterministic tests (LLM paths use a scripted model).
**Model-dependent**: the harness provides the interface and structure; quality comes from the model and has not been measured here.
**Partial / not implemented**: stated plainly.

| Requirement | Mechanism | Status |
|---|---|---|
| **Perceive** | `perception.py`; `perceive_file` and `observe_environment` tools; images/PDFs passed to the model as content blocks; video as key frames | Text/code/docs/images/PDF/environment and video key frames (ffmpeg): tested. **Audio content: not implemented** (metadata only unless a transcriber is supplied). Understanding quality: model-dependent. |
| **Remember** | `MemoryStore`: episodic/semantic/procedural, FTS5 retrieval, persists across processes; durable run state (`state.json`, `ledger.json`) | Tested. No forgetting/consolidation policy beyond "only verified items drive behaviour"; not tested at very large scale. |
| **Reason** | Model reasoning (adaptive thinking) + explicit structure: competing hypotheses, Bayes in code, BIC model comparison, `run_python` for computation | Structure tested; reasoning quality model-dependent. No dedicated causal-inference engine (experiments in the discovery loop are interventions by construction). |
| **Research** | Server-side web search/fetch; `fetch_url`; `cite_source` verifies quotes against page text; `mark_conflict` + unresolved-conflict reporting | Citation checking and conflict tracking tested; search quality model-dependent. |
| **Plan** | `TaskGraph`: DAG validation, retries, blocking, refinement; planner role; decomposition of oversized tasks; replanning on failure | Tested (incl. 3,000-node plans and decomposition). |
| **Act** | Governed tools: shell, Python, files, HTTP; sub-agents | Tested. **No GUI/browser automation** (Claude's computer-use toolset is not wired in). Databases/APIs only via shell/Python. |
| **Code** | Coder role: read/search/write/run/test tools; execution evidence | Tool plumbing tested; coding ability model-dependent. |
| **Experiment** | Discovery loop: discriminating designs, falsification, expansion; `compare_hypotheses` tool lets the model use the same machinery | **Measured** on the benchmark. |
| **Collaborate** | `spawn_subagent`: roles, fresh contexts, depth limit, budget share charged to parent, parallel execution | Tested with a scripted model. Whether delegation helps is model-dependent and unmeasured. |
| **Verify** | Fresh-context verifier; promotion to fact requires grounding evidence; falsification record | Tested. |
| **Self-correct** | Refutation propagation; falsification rounds; hypothesis-space expansion when all named hypotheses fail; retry-with-error, decomposition, replanning; anti-fabrication rules (evidence only from real executions; unverified quotes cannot ground facts) | Discovery-side: measured. Agent-side: tested. Not a general hallucination detector. |
| **Learn** | Verified outcomes stored in memory set future priors; procedures with win/loss records; acquired capabilities and rejected candidates stored in memory | Learning curve measured with an earlier code version and not re-verified; its "verified" outcomes are the grader's ground truth (oracle labels). |
| **Self-improve** | `improve.py`: diagnosis-driven and LLM proposals, sandboxed code patches limited to mutable paths, paired bootstrap gate, replication, held-out test, protected surfaces | Strategy level **measured** (`docs/SELF_IMPROVEMENT.md`; re-verified with the committed code). Code-patch path tested; no LLM-written patches were evaluated. |
| **Acquire capabilities** | `capabilities/`: self-assessed failure, diagnosis, program search, artifact, unit tests, validation, independent reviewer on fresh held-out and transfer blocks, registry | Pre-registered experiment (`experiments/capability_acquisition/`); see `docs/CAPABILITY_ACQUISITION.md` for results. |
| **Generalize** | The flexible "none of the above" baseline, per-task program search, acquired capabilities | Measured on transfer (regime shift, composition), blind novel classes and 29 real datasets in the capability-acquisition experiment; all 1-D. |
| **Long horizons** | Episode-based contexts seeded from durable state; checkpoint after every phase and task; `quanta resume`; budgets in steps/tokens/dollars/hours | Resume tested. Not tested on multi-day runs. |
| **Know when it doesn't know** | Calibrated credence (tempering), proper scoring, "none" answers, human-input requests below a credence threshold, expected-loss rule for questions | **Measured** (Brier/ECE, confidently-wrong rate). |
| **Preserve human control** | Governor invariants, protected surfaces, code adoption needs approval; the capability reviewer is outside the acquisition loop's authority | Tested, including that the self-improver cannot patch its own oversight. In-process only: an audit reproduced bypasses (several now fixed, `tests/test_hardening.py`); `run_python` can still tamper with run files without an OS sandbox. |

## Long-horizon design

No single context window holds a run. Each phase and each task is a fresh, bounded agent
episode whose briefing is rebuilt from durable state (problem frame, hypotheses with
posteriors, plan status, ledger summary, relevant memories, budget). State is written
after every phase and every task, so a crash costs at most one task. Conversation history
inside an episode is append-only: assistant content, thinking blocks included, is passed
back unchanged.

## Model usage

Defaults: the model configured in `quanta.llm`, adaptive thinking, `effort=high` for orchestration and
`medium` for sub-agents (both genome parameters), server-side web search/fetch
(`web_search_20260209`, `web_fetch_20260209`), prompt caching on the stable system prefix,
and server-side refusal fallbacks (`fallbacks="default"`). The verifier can be a different
model (`--verifier-model`) for more independent checking.
