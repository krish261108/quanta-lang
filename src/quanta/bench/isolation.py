"""Trusted, process-isolated evaluation.

PROTECTED. The solver (with whatever capabilities it is being evaluated with)
runs in worker subprocesses (`python -m quanta.science.worker`). This process
keeps the tasks, answers measurement requests, enforces the measurement budget
(its own count is authoritative), and grades. Workers never receive seeds,
parameters or truth; they cannot call or modify the grader, which lives here.

Residual risk, stated plainly: workers run as the same OS user, so code that is
allowed to execute arbitrary Python in a worker could read files. Capability
artifacts cannot (they are policy-checked arithmetic), and the solver itself is
designer code. Evaluating LLM-written code would additionally need an OS-level
sandbox; see docs/CAPABILITY_ACQUISITION.md.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SRC = str(Path(__file__).resolve().parents[2])


@dataclass
class Job:
    task_id: str
    instrument: Any       # has .domain and .measure(x); lives only in this process
    budget: int
    seed: int
    eval_x: tuple[float, ...]


class _Worker:
    def __init__(self, config: dict, pythonpath: str) -> None:
        env = {**os.environ, "PYTHONPATH": pythonpath, "PYTHONDONTWRITEBYTECODE": "1"}
        self.proc = subprocess.Popen([sys.executable, "-m", "quanta.science.worker"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env,
                                     bufsize=1)
        reply = self._call({"cmd": "configure", "config": config})
        if not reply.get("ok"):
            self.close()
            raise RuntimeError(f"worker rejected configuration: {reply.get('error')}")

    def _send(self, msg: dict) -> None:
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _recv(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("worker exited")
        return json.loads(line)

    def _call(self, msg: dict) -> dict:
        self._send(msg)
        return self._recv()

    def run(self, job: Job, timeout: float) -> dict:
        timer = threading.Timer(timeout, self.proc.kill)
        timer.start()
        used = 0
        t0 = time.perf_counter()
        try:
            self._send({"cmd": "solve", "task_id": job.task_id, "domain": list(job.instrument.domain),
                        "budget": job.budget, "seed": job.seed, "eval_x": list(job.eval_x)})
            while True:
                msg = self._recv()
                if msg.get("cmd") == "measure":
                    if used >= job.budget:
                        self._send({"error": "measurement budget exhausted"})
                        continue
                    used += 1
                    self._send({"y": job.instrument.measure(float(msg["x"]))})
                elif msg.get("cmd") == "result":
                    msg["measurements_used"] = used          # authoritative count
                    msg["seconds"] = round(time.perf_counter() - t0, 3)
                    return msg
                else:
                    raise RuntimeError(f"protocol violation: {msg!r:.200}")
        finally:
            timer.cancel()

    def close(self) -> None:
        try:
            self._send({"cmd": "exit"})
        except (OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def run_isolated(config: dict, jobs: list[Job], *, workers: int = 4, pythonpath: str = SRC,
                 timeout: float = 600.0) -> list[dict]:
    """Solve every job in isolated workers; results are returned in job order.
    A job's result depends only on (config, job), never on scheduling."""
    results: list[dict | None] = [None] * len(jobs)
    q: queue.Queue = queue.Queue()
    for i, j in enumerate(jobs):
        q.put((i, j))

    def loop() -> None:
        w = None
        while True:
            try:
                i, job = q.get_nowait()
            except queue.Empty:
                break
            try:
                if w is None:
                    w = _Worker(config, pythonpath)
                results[i] = w.run(job, timeout)
            except Exception as e:  # crash, timeout or protocol violation: record and restart
                results[i] = {"task_id": job.task_id, "error": f"{type(e).__name__}: {e}",
                              "measurements_used": None}
                if w is not None:
                    w.close()
                w = None
        if w is not None:
            w.close()

    threads = [threading.Thread(target=loop, daemon=True) for _ in range(max(1, min(workers, len(jobs))))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [r if r is not None else {"error": "not run"} for r in results]
