"""Isolated solver worker (the untrusted side of evaluation).

Speaks a line-delimited JSON protocol on stdin/stdout with the trusted evaluator
(`quanta.bench.isolation`):

    -> {"cmd": "configure", "config": {...}}          once
    <- {"ok": true}
    -> {"cmd": "solve", "task_id": "...", "domain": [lo, hi], "budget": n, "seed": s, "eval_x": [...]}
    <- {"cmd": "measure", "x": 3.2}                    any number of times
    -> {"y": 1.234}            or {"error": "..."}
    <- {"cmd": "result", "task_id": "...", ...}

The worker never receives a task's seed, parameters or truth: only measurements.
"""
from __future__ import annotations

import json
import sys
import traceback

from .toolkit import Toolkit


class RemoteInstrument:
    def __init__(self, domain, out, inp) -> None:
        self.domain = tuple(domain)
        self._out = out
        self._in = inp

    def measure(self, x: float) -> float:
        self._out.write(json.dumps({"cmd": "measure", "x": float(x)}) + "\n")
        self._out.flush()
        reply = json.loads(self._in.readline())
        if "error" in reply:
            raise RuntimeError(f"instrument refused: {reply['error']}")
        return float(reply["y"])


def main() -> int:
    out = sys.stdout
    sys.stdout = sys.stderr          # stray prints must not corrupt the protocol
    inp = sys.stdin
    toolkit = None
    for line in inp:
        if not line.strip():
            continue
        msg = json.loads(line)
        cmd = msg.get("cmd")
        if cmd == "configure":
            try:
                toolkit = Toolkit.from_config(msg["config"])
                toolkit._loaded()        # validate artifacts up front
                out.write(json.dumps({"ok": True}) + "\n")
            except Exception as e:      # report, don't die: the evaluator records the failure
                out.write(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}) + "\n")
            out.flush()
        elif cmd == "solve":
            try:
                if toolkit is None:
                    raise RuntimeError("not configured")
                inst = RemoteInstrument(msg["domain"], out, inp)
                res = toolkit.solve(inst, budget=int(msg["budget"]), seed=int(msg["seed"]),
                                    eval_x=msg["eval_x"])
                res.update({"cmd": "result", "task_id": msg["task_id"], "error": None})
            except Exception as e:
                res = {"cmd": "result", "task_id": msg.get("task_id"), "error": f"{type(e).__name__}: {e}",
                       "trace": traceback.format_exc()[-2000:]}
            out.write(json.dumps(res) + "\n")
            out.flush()
        elif cmd == "exit":
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
