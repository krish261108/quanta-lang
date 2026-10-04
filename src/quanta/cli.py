"""Command-line interface.

    quanta solve "problem statement"      open-ended research loop (needs an LLM backend)
    quanta resume RUN_DIR                 continue an interrupted run
    quanta discover --seed N              run the automated discovery loop on one benchmark task
    quanta bench                          evaluate a genome on a benchmark split
    quanta improve                        benchmark-gated self-improvement
    quanta learn-curve                    performance vs. amount of verified experience in memory
    quanta memory search|stats            inspect long-term memory
    quanta audit verify FILE              check an audit log's hash chain
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .genome import Genome
from .governance import (AllowlistApprover, AllowRule, AuditLog, Budget, DenyAllApprover, Governor,
                         InteractiveApprover, KillSwitch, Risk)

DEFAULT_MEMORY = Path.home() / ".quanta" / "memory.db"


def _governor(args, run_dir: Path) -> Governor:
    if args.approve == "interactive":
        fallback = InteractiveApprover()
    else:
        fallback = DenyAllApprover()
    rules = [AllowRule("run_shell", Risk.HIGH, match=(lambda a, p=prefix: a.get("command", "").startswith(p)),
                       note=f"shell commands starting with {prefix!r}") for prefix in args.allow_shell_prefix]
    approver = AllowlistApprover(rules, fallback) if rules else fallback
    return Governor(approver=approver,
                    budget=Budget(max_steps=args.max_steps, max_cost_usd=args.max_cost,
                                  max_seconds=args.max_hours * 3600 if args.max_hours else None),
                    audit=AuditLog(run_dir / "audit.jsonl"), kill_switch=KillSwitch(run_dir / "STOP"))


def _loop_kwargs(args, run_dir: Path) -> dict:
    from .llm import AnthropicModel
    from .memory import MemoryStore
    from .tools import HumanChannel

    model = AnthropicModel(args.model, effort=args.effort, web_search=not args.no_web, web_fetch=not args.no_web)
    verifier = AnthropicModel(args.verifier_model, effort=args.effort, web_search=not args.no_web,
                              web_fetch=not args.no_web) if args.verifier_model else None
    human = HumanChannel(lambda q: input(f"\n[question from the agent] {q}\n> ")) if args.human else HumanChannel()
    return {"model": model, "verifier_model": verifier, "governor": _governor(args, run_dir),
            "memory": MemoryStore(args.memory), "human": human,
            "genome": Genome.load(args.genome) if args.genome else Genome(),
            "workspace": args.workspace}


def cmd_solve(args) -> int:
    from .solver import ResearchLoop
    run_dir = Path(args.run_dir)
    loop = ResearchLoop(args.problem, run_dir=run_dir, **_loop_kwargs(args, run_dir))
    print(f"run {loop.state.run_id} in {run_dir} (create {run_dir / 'STOP'} to halt)", file=sys.stderr)
    report = loop.run()
    print(report.read_text())
    return 0


def cmd_resume(args) -> int:
    from .solver import ResearchLoop
    run_dir = Path(args.run_dir)
    loop = ResearchLoop.resume(run_dir, **_loop_kwargs(args, run_dir))
    print(loop.run().read_text())
    return 0


def cmd_discover(args) -> int:
    from .bench.tasks import generate_task, grade
    from .report import render_report
    from .science.discovery import DiscoveryLoop

    genome = Genome.load(args.genome) if args.genome else Genome()
    task = generate_task(args.seed)
    res = DiscoveryLoop(genome, budget=task.budget, seed=args.seed).run(task.system())
    g = grade(task, res.answer, res.credence, res.predict, res.n_experiments)
    answer = (f"**{res.answer}** — {res.best_fit.description}" if res.answer != "none"
              else f"**none of the named laws** — best description: {res.best_fit.description}")
    trace = "\n".join(f"| {t['n']} | {t['last']} | {t['leader']} | {t['p']:.3f} | {t['entropy']:.3f} |"
                      for t in res.trace)
    extra = [("Experiment trace", "| n | last experiment | leader | p(leader) | entropy |\n|---|---|---|---|---|\n"
              + trace),
             ("Posterior", "\n".join(f"- {k}: {v:.4f}" for k, v in list(res.posterior.items())[:6])),
             ("Grading (ground truth revealed after the run)",
              f"- truth: {task.family} {tuple(round(p, 4) for p in task.params)}; expected answer "
              f"'{task.expected_answer}'\n- correct: {g['correct']}; score {g['score']:.4f}; prediction NRMSE "
              f"{g['nrmse']:.4f}")]
    if res.human_request:
        extra.insert(0, ("Request for human input", res.human_request))
    print(render_report(
        title=f"Discovery report (task seed {args.seed})",
        problem="Identify the law y = f(x) behind a noisy instrument on x in [0.1, 10] using at most "
                f"{task.budget} measurements.",
        answer=answer, confidence=res.credence, ledger=res.ledger,
        limitations=[f"stopped because: {res.stop_reason}",
                     "hypothesis space: " + ", ".join(res.hypotheses_considered)],
        reproduction={"command": f"quanta discover --seed {args.seed}" + (f" --genome {args.genome}" if args.genome else ""),
                      "genome": genome.fingerprint()},
        extra_sections=extra, max_items=15))
    return 0


def cmd_bench(args) -> int:
    from .bench.harness import run_suite
    genome = Genome.load(args.genome) if args.genome else Genome()
    res = run_suite(genome, args.split, args.n, jobs=args.jobs)
    if args.out:
        Path(args.out).write_text(json.dumps(res.to_dict(), indent=1, default=str))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v)
                      for k, v in res.diagnostics.items()}, indent=1, default=str))
    return 0


def cmd_improve(args) -> int:
    from .improve import DiagnosticProposer, Evaluator, LLMProposer, SelfImprover
    proposers = []
    if args.proposer in ("diagnostic", "both"):
        proposers.append(DiagnosticProposer())
    if args.proposer in ("llm", "both"):
        from .llm import AnthropicModel
        proposers.append(LLMProposer(AnthropicModel(args.model, effort=args.effort, web_search=False,
                                                    web_fetch=False)))
    run_dir = Path(args.run_dir)
    imp = SelfImprover(Evaluator(jobs=args.jobs), proposers, run_dir=run_dir, generations=args.generations,
                       candidates_per_generation=args.candidates, patience=args.patience,
                       n_selection=args.n_selection, n_confirmation=args.n_confirmation, n_test=args.n_test,
                       seed=args.seed, governor=Governor(audit=AuditLog(run_dir / "audit.jsonl")))
    imp.run(Genome.load(args.genome) if args.genome else Genome())
    print((run_dir / "report.md").read_text())
    return 0


def cmd_learn_curve(args) -> int:
    from .bench.harness import build_experience_memory, prior_counts_from_memory, run_suite
    genome = (Genome.load(args.genome) if args.genome else Genome()).mutate(use_learned_priors=True)
    rows = []
    for k in [int(x) for x in args.experience.split(",")]:
        with build_experience_memory(k) as mem:
            counts = dict(prior_counts_from_memory(mem))
        res = run_suite(genome, args.split, args.n, jobs=args.jobs, prior_counts=counts)
        d = res.diagnostics
        rows.append({"experience": k, "mean_score": round(d["mean_score"], 4), "accuracy": round(d["accuracy"], 4),
                     "brier": round(d["brier"], 4), "mean_cost": round(d["mean_cost"], 4)})
        print(json.dumps(rows[-1]))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


def cmd_memory(args) -> int:
    from .memory import MemoryStore
    with MemoryStore(args.memory) as mem:
        if args.action == "stats":
            print(json.dumps(mem.stats(), indent=1))
        else:
            for it in mem.search(" ".join(args.query), limit=args.limit):
                print(f"[{it.kind}#{it.id} {'verified' if it.verified else 'unverified'}] {it.subject}: {it.content[:300]}")
    return 0


def cmd_audit(args) -> int:
    ok, bad = AuditLog(args.file).verify()
    print("audit chain intact" if ok else f"audit chain BROKEN at entry {bad}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="quanta", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def llm_opts(p):
        p.add_argument("--model", default="claude-opus-5-5")
        p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])

    def run_opts(p):
        llm_opts(p)
        p.add_argument("--verifier-model", help="use a different model for independent verification")
        p.add_argument("--workspace", help="directory the agents may read/write (default: RUN_DIR/workspace)")
        p.add_argument("--genome", help="strategy genome JSON")
        p.add_argument("--memory", default=str(DEFAULT_MEMORY))
        p.add_argument("--max-steps", type=int, default=400)
        p.add_argument("--max-cost", type=float, default=25.0, help="USD")
        p.add_argument("--max-hours", type=float, default=None)
        p.add_argument("--approve", choices=["deny", "interactive"], default="deny",
                       help="who authorizes high-risk actions (default: nobody -> denied)")
        p.add_argument("--allow-shell-prefix", action="append", default=[],
                       help="pre-authorize high-risk shell commands with this prefix (repeatable)")
        p.add_argument("--human", action="store_true", help="a human is available to answer questions")
        p.add_argument("--no-web", action="store_true", help="disable server-side web search/fetch")

    p = sub.add_parser("solve", help="run the open-ended research loop on a problem")
    p.add_argument("problem")
    p.add_argument("--run-dir", default="runs/latest")
    run_opts(p)
    p.set_defaults(fn=cmd_solve)

    p = sub.add_parser("resume", help="resume an interrupted research run")
    p.add_argument("run_dir")
    run_opts(p)
    p.set_defaults(fn=cmd_resume)

    p = sub.add_parser("discover", help="automated discovery on one benchmark task (offline)")
    p.add_argument("--seed", type=int, default=30_000)
    p.add_argument("--genome")
    p.set_defaults(fn=cmd_discover)

    p = sub.add_parser("bench", help="evaluate a genome on a benchmark split (offline)")
    p.add_argument("--genome")
    p.add_argument("--split", default="selection", choices=["selection", "confirmation", "test", "experience"])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("improve", help="benchmark-gated self-improvement (offline unless --proposer llm)")
    llm_opts(p)
    p.add_argument("--run-dir", default="runs/improve")
    p.add_argument("--genome", help="initial genome (default: baseline)")
    p.add_argument("--proposer", choices=["diagnostic", "llm", "both"], default="diagnostic")
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--candidates", type=int, default=6)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--n-selection", type=int, default=200)
    p.add_argument("--n-confirmation", type=int, default=200)
    p.add_argument("--n-test", type=int, default=400)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_improve)

    p = sub.add_parser("learn-curve", help="score vs. verified experience in memory (offline)")
    p.add_argument("--genome")
    p.add_argument("--experience", default="0,10,30,100,300")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_learn_curve)

    p = sub.add_parser("memory", help="inspect long-term memory")
    p.add_argument("action", choices=["search", "stats"])
    p.add_argument("query", nargs="*")
    p.add_argument("--memory", default=str(DEFAULT_MEMORY))
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(fn=cmd_memory)

    p = sub.add_parser("audit", help="verify an audit log")
    p.add_argument("action", choices=["verify"])
    p.add_argument("file")
    p.set_defaults(fn=cmd_audit)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
