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
   to LOW on every assignment, from a private copy of the constant; no configuration, genome
   or model output can raise it.
2. When the kill switch is tripped (programmatically, or by creating `RUN_DIR/STOP`),
   nothing further is authorized and agents halt at the next step. The switch latches:
   deleting the stop file does not re-enable the run; only an explicit operator reset does.
3. When any budget (steps, tokens, dollars, wall-clock) is exhausted, nothing further is
   authorized. Sub-agent spending is charged to the parent budget. Negative charges are refused.
4. Every authorization decision, including one where the approver itself fails, is appended
   to a hash-chained audit log with a hash of the full arguments. `quanta audit verify FILE`
   detects in-place edits. While the governor is running, any truncation, deletion or
   rewrite of the file trips the kill switch.
5. With no human available, the default approver denies.
6. Shell pre-authorizations (`--allow-shell-prefix`) match a whole command word and refuse
   any command containing shell control syntax (`;`, `&&`, `|`, backticks, `$(`, redirects,
   newlines).

All of these are in-process guards. They hold against the agent's *tools*, not against code
running in the same process or as the same OS user, which can rebind anything. An
independent audit (see [RESEARCH_PROGRAM.md](RESEARCH_PROGRAM.md)) found and reproduced
several bypasses; the ones fixed are covered by `tests/test_hardening.py`.

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
  something actually ran successfully (a failed call is recorded as a note). A claim
  labelled FACT without grounding evidence is automatically downgraded. The ledger checks
  the *kind* of evidence, not whether it is relevant to the claim: a successful but
  unrelated execution can still ground a FACT. Treat FACT as "grounded in something that
  happened", not as "true".
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
  inside an isolated environment and unreasonable on a personal machine. The audit showed
  such code can also rewrite the run's own state, ledger and audit files and delete the stop
  file; the live-log check and the latching kill switch catch some of this during a run,
  but only an OS-level sandbox (separate user, read-only oversight files) closes it.
* Self-modification of capabilities (see [CAPABILITY_ACQUISITION.md](CAPABILITY_ACQUISITION.md))
  is limited to arithmetic expressions checked by a static policy and evaluated in worker
  subprocesses. Evaluating model-written code would need an OS-level sandbox first.
* Approvals are only as good as the human reading them; requests show the tool, the risk
  tier and the arguments.
