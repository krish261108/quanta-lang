"""Development comparison (dev split only, never reported as a result): model evidence
(BIC vs g-prior) x capability gating (off vs on), with a fixed set of three capability
artifacts taken from the first v1 acquisition attempt.

    python experiments/evidence_gprior/dev_2x2.py <registry.jsonl> [n] [workers]
"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from quanta.bench.acquisition import evaluate  # noqa: E402
from quanta.capabilities.registry import Registry  # noqa: E402
from quanta.genome import Genome  # noqa: E402
from quanta.stats import paired_bootstrap  # noqa: E402

registry, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 400
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
reg = Registry(registry)
caps = [reg.get(c).artifact for c in ("cap-0003", "cap-0004", "cap-0005")]
base = dict(falsification_rounds=2, likelihood="student_t", flexible_baseline=True, pool_equivalent=True)
configs = {}
for ev in ("bic", "gprior"):
    configs[f"{ev}"] = {"genome": Genome(**base, evidence=ev).to_dict(), "capabilities": []}
    configs[f"{ev}+caps"] = {"genome": Genome(**base, evidence=ev).to_dict(), "capabilities": caps}
    configs[f"{ev}+caps_gated"] = {"genome": Genome(**base, evidence=ev, gate_capabilities=True).to_dict(),
                                   "capabilities": caps}
out = {}
for name, cfg in configs.items():
    t0 = time.time()
    out[name] = evaluate(cfg, "dev", n, workers=workers)
    print(name, round(time.time() - t0), "s", flush=True)
json.dump(out, open(Path(__file__).with_name(f"dev_2x2_{n}.json"), "w"), default=str)


def rec(rs):
    return round(statistics.mean(1.0 if r["recovered"] else 0.0 for r in rs), 3)


def cw(rs):
    return round(statistics.mean(1.0 if (r["claimed"] and r["p_claim"] >= 0.9 and not r["recovered"]) else 0.0
                                 for r in rs), 3)


for ref in ("bic", "gprior"):
    for name in configs:
        if name == ref or (ref == "gprior" and not name.startswith("gprior")):
            continue
        for kind in ("familiar", "gap", None):
            x = [r for r in out[ref] if kind is None or r["kind"] == kind]
            y = [r for r in out[name] if kind is None or r["kind"] == kind]
            c = paired_bootstrap([r["score"] for r in x], [r["score"] for r in y], n_boot=2000, seed=1)
            sd = statistics.pstdev([b["score"] - a["score"] for a, b in zip(x, y)])
            print(f"{name:18s} vs {ref:6s} {str(kind):8s} n={c.n:3d} d={c.mean_delta:+.4f} "
                  f"[{c.ci_low:+.4f},{c.ci_high:+.4f}] sd={sd:.3f} rec {rec(x)}->{rec(y)} confwrong {cw(x)}->{cw(y)} "
                  f"meas {statistics.mean(r['measurements'] for r in x):.1f}->"
                  f"{statistics.mean(r['measurements'] for r in y):.1f}")
