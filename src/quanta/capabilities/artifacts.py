"""Turning a researched structure into a capability artifact, and unit-testing it.

The artifact is data plus generated source code (one expression per term over `x`
and parameters `p[i]`). It is what gets static-checked, unit-tested, reviewed,
hashed into the registry, and loaded by isolated solver workers.

Unit tests run in a fresh subprocess (`python -m quanta.capabilities.artifacts`)
so that a broken artifact cannot affect the caller:

* policy        source passes the static code policy
* load          the artifact loads as a hypothesis family
* finite        finite output on [0.1, 13] at its typical parameters
* recovery      fitting recovers its own form from noisy synthetic data (>= 70%)
* deterministic two fits of the same data agree exactly
* runtime       mean fit time below a bound
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from ..science.synth import Model
from ..science.toolkit import ARTIFACT_INTERFACE, ArtifactError, artifact_hash, load_artifact

SRC = str(Path(__file__).resolve().parents[2])
RECOVERY_THRESHOLD = 0.7
MAX_FIT_SECONDS = 0.5


def build_artifact(capability_id: str, version: int, model: Model, inits: list[list[float]], *,
                   name: str, description: str) -> dict:
    """Generate the artifact for `model`; `inits` are full parameter vectors observed
    when the structure was found (split per term)."""
    terms, offset = [], 0
    for shape in model.shapes:
        src, n = shape.source(0)
        term_inits = [list(v[offset: offset + n]) for v in inits if len(v) >= offset + n][:5]
        terms.append({"source": src, "n_params": n, "n_freq": shape.n_freq, "structure": shape.expand_key(),
                      "inits": term_inits})
        offset += n
    art = {"capability_id": capability_id, "version": version, "interface": ARTIFACT_INTERFACE,
           "name": name, "description": description, "structure": model.key, "terms": terms}
    art["sha256"] = artifact_hash(art)
    return art


def static_check(artifact: dict) -> list[str]:
    try:
        load_artifact(artifact)
    except (ArtifactError, ValueError) as e:
        return [str(e)]
    return []


def _self_tests(artifact: dict, trials: int = 10, seed: int = 0) -> dict:
    out = {"policy": False, "load": False, "finite": False, "recovery": 0.0, "deterministic": False,
           "runtime_s": None}
    try:
        family, blocks = load_artifact(artifact)
        out["policy"] = out["load"] = True
    except Exception as e:  # noqa: BLE001 - report any failure as a failed test
        out["error"] = f"{type(e).__name__}: {e}"
        out["passed"] = False
        return out
    rng = random.Random(seed)
    inits = []
    for b in blocks:
        inits.append([list(i) for i in b.inits] or [[0.5] * b.n_params])
    grid = [0.1 + 12.9 * i / 49 for i in range(50)]

    def term_values(params_per_term, x):
        from ..science.synth import compiled
        return [compiled(b.source)(x, p) for b, p in zip(blocks, params_per_term)]

    finite = 0
    total = 0
    for k in range(max(len(i) for i in inits)):
        params = [inits[t][k % len(inits[t])] for t in range(len(blocks))]
        for x in grid:
            total += 1
            vals = term_values(params, x)
            finite += all(math.isfinite(v) and abs(v) < 1e12 for v in vals)
    out["finite"] = finite / total >= 0.95
    successes, times = 0, []
    dense = [0.1 + 9.9 * i / 59 for i in range(60)]
    for t in range(trials):
        params = [[v * (1 + rng.uniform(-0.1, 0.1)) for v in inits[j][t % len(inits[j])]]
                  for j in range(len(blocks))]
        coefs = [rng.uniform(-1, 1)] + [rng.choice((-1, 1)) * rng.uniform(1, 3) for _ in blocks]

        def truth(x):
            return coefs[0] + sum(c * v for c, v in zip(coefs[1:], term_values(params, x)))

        try:
            truths = [truth(x) for x in dense]
        except (OverflowError, ValueError, ZeroDivisionError):
            continue
        sd = (sum((v - sum(truths) / len(truths)) ** 2 for v in truths) / len(truths)) ** 0.5
        if not math.isfinite(sd) or sd < 1e-6:
            continue
        xs = [rng.uniform(0.1, 10) for _ in range(25)]
        ys = [truth(x) + rng.gauss(0, 0.02 * sd) for x in xs]
        fam, _ = load_artifact(artifact)
        t0 = time.perf_counter()
        try:
            fit = fam.fit(xs, ys)
        except Exception:  # noqa: BLE001
            continue
        times.append(time.perf_counter() - t0)
        err = (sum((fit.predict(x) - v) ** 2 for x, v in zip(dense, truths)) / len(dense)) ** 0.5 / sd
        successes += err < 0.1
        if t == 0:
            fam2, _ = load_artifact(artifact)
            fit2 = fam2.fit(xs, ys)
            out["deterministic"] = all(fit.predict(x) == fit2.predict(x) for x in dense[::7])
    out["recovery"] = successes / trials
    out["runtime_s"] = round(sum(times) / len(times), 4) if times else None
    out["passed"] = bool(out["policy"] and out["load"] and out["finite"] and out["deterministic"]
                         and out["recovery"] >= RECOVERY_THRESHOLD
                         and out["runtime_s"] is not None and out["runtime_s"] <= MAX_FIT_SECONDS)
    return out


def run_unit_tests(artifact: dict, *, trials: int = 10, seed: int = 0, timeout: float = 300.0) -> dict:
    """Run the artifact's unit tests in a fresh subprocess."""
    env = {**os.environ, "PYTHONPATH": SRC, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        r = subprocess.run([sys.executable, "-m", "quanta.capabilities.artifacts", str(trials), str(seed)],
                           input=json.dumps(artifact), capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "unit tests timed out"}
    if r.returncode != 0:
        return {"passed": False, "error": (r.stderr or r.stdout)[-1500:]}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"passed": False, "error": "unit tests produced no result"}


if __name__ == "__main__":
    art = json.loads(sys.stdin.read())
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(json.dumps(_self_tests(art, trials, seed), default=str))
