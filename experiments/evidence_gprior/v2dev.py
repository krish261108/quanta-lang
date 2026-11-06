import sys, json, statistics, time
W = sys.argv[1]; N = int(sys.argv[2])
sys.path.insert(0, W + "/src")
from quanta.genome import Genome
from quanta.bench.acquisition import evaluate
from quanta.capabilities.registry import Registry
from quanta.stats import paired_bootstrap
reg = Registry("/home/user/quanta-lang/experiments/capability_acquisition/results/rep0/acquire.interrupted-1793982844/registry.jsonl")
caps = [reg.get(c).artifact for c in ("cap-0003", "cap-0004", "cap-0005")]
base = dict(falsification_rounds=2, likelihood="student_t", flexible_baseline=True, pool_equivalent=True)
configs = {
    "S0": {"genome": Genome(**base).to_dict(), "capabilities": []},
    "S0+caps": {"genome": Genome(**base).to_dict(), "capabilities": caps},
    "S0+caps_gated": {"genome": Genome(**base, gate_capabilities=True).to_dict(), "capabilities": caps},
    "S0gp+caps": {"genome": Genome(**base, evidence="gprior").to_dict(), "capabilities": caps},
    "S0gp+caps_gated": {"genome": Genome(**base, evidence="gprior", gate_capabilities=True).to_dict(), "capabilities": caps},
}
out = {}
for name, cfg in configs.items():
    t0 = time.time()
    out[name] = evaluate(cfg, "dev", N, workers=2)
    print(name, round(time.time() - t0), "s", flush=True)
json.dump(out, open(f"v2dev_{N}.json", "w"), default=str)
b = out["S0"]
for name in list(configs)[1:]:
    c = out[name]
    for kind in ("familiar", "gap", None):
        x = [r for r in b if kind is None or r["kind"] == kind]; y = [r for r in c if kind is None or r["kind"] == kind]
        ci = paired_bootstrap([r["score"] for r in x], [r["score"] for r in y], n_boot=2000, seed=1)
        rec = lambda rs: round(statistics.mean(1.0 if r["recovered"] else 0.0 for r in rs), 3)
        print(f"{name:16s} {str(kind):8s} n={ci.n:3d} d={ci.mean_delta:+.4f} [{ci.ci_low:+.4f},{ci.ci_high:+.4f}] rec {rec(x)}->{rec(y)}")
