# Safety and human control

The design goal is that the system can work autonomously for long periods without being
able to make irreversible or high-impact decisions on its own, and without being able to
weaken the mechanisms that enforce this, including while it is improving itself.

## One chokepoint

Every tool call goes through `ToolRegistry.execute`, which:

1. validates the arguments,
2. classifies the action's risk (`READ_ONLY`, `LOW`, `HIGH`, `CRITICAL`),
3. asks the `Governor` for authorization,
4. runs the tool only if authorized,
5. records the result in the hash-chained audit log, and executions/observations in the ledger.

| Risk | Meaning | Who authorizes |
|---|---|---|
| `READ_ONLY` | observes only | automatic |
| `LOW` | local, reversible side effects inside the workspace (file writes are backed up) | automatic |
| `HIGH` | hard to reverse, external side effects, spends money, installs software, network writes | a human (interactive approver) or an explicit pre-authorization rule |
| `CRITICAL` | irreversible or affecting systems/people outside the sandbox (`rm -rf /`, force-push, `terraform destroy`, piping downloads into a shell) | a human, every time; cannot be pre-authorized |

## Invariants

Enforced in `governance.py` and tested in `tests/test_governance_invariants.py`:

1. HIGH and CRITICAL actions are never auto-approved. The auto-approval ceiling is clamped
   to LOW in code; no configuration, genome or model output can raise it.
2. When the kill switch is tripped (programmatically, or by creating `RUN_DIR/STOP`),
   nothing further is authorized and agents halt at the next step.
3. When any budget (steps, tokens, dollars, wall-clock) is exhausted, nothing further is
   authorized. Sub-agent spending is charged to the parent budget.
4. Every authorization decision is appended to a hash-chained audit log; `quanta audit
   verify FILE` detects any after-the-fact edit.
5. With no human available, the default approver denies.

## Self-improvement cannot weaken oversight

* The genome (the strategy space that may be tuned automatically) contains no governance
  settings, and unknown fields are rejected when a genome is loaded.
* Code patches may only touch `MUTABLE_PATHS` (`src/quanta/science/`). These are always
  off-limits: governance, the self-improver itself, the benchmark and grader, the ledger
  rules, tool risk classification, the genome bounds, statistics, and the invariant tests.
  Path traversal is rejected, and protected-file hashes are re-checked in the sandbox
  after patching.
* Every patch candidate must pass the governance invariant tests inside its sandbox
  before it is benchmarked.
* Adopting a genome change is a LOW-risk action: bounded, reversible, audited. Adopting a
  code patch is HIGH-risk and requires human authorization.

## Epistemic safety

* Agents cannot create observation or execution evidence; only the harness does, when
  something actually ran. A claim labelled FACT without grounding evidence is
  automatically downgraded.
* Citations are checked: a quote that does not appear in the fetched source is recorded
  as an unverified note, which cannot ground a fact.
* Agents may write to long-term memory but cannot mark memories verified, so unverified
  beliefs cannot steer future behaviour.
* Perception reports explicitly when a modality was not perceived (e.g. audio without a
  transcriber), so claims cannot be grounded on content nobody looked at.

## Asking for help

Agents ask a human only through `request_human_input`, which applies an expected-loss
rule: ask when (1 − confidence) × impact (doubled if irreversible) exceeds the cost of
interrupting. Otherwise the agent proceeds and the assumption is recorded as speculation,
so it appears in the final report.

## Known limits

* Shell risk classification is a heuristic over command strings. It catches common
  destructive and external-effect patterns, but a determined process (for example,
  Python code that deletes files) can act outside what the classifier sees. Run the
  harness inside a container or VM. The classifier is defense in depth, not a sandbox.
* `run_python` executes arbitrary code in the workspace at LOW risk. That is reasonable
  inside an isolated environment and unreasonable on a personal machine.
* Approvals are only as good as the human reading them; requests show the tool, the risk
  tier and the arguments.
