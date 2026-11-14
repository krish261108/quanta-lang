import sys, json, statistics, time
W = sys.argv[1]; N = int(sys.argv[2]); split = sys.argv[3] if len(sys.argv) > 3 else "dev"
sys.path.insert(0, W + "/src")
from quanta.genome import Genome
from quanta.science.toolkit import Toolkit
from quanta.bench.acquisition import evaluate
from quanta.stats import paired_bootstrap
base = dict(falsification_rounds=2, likelihood="student_t", flexible_baseline=True, pool_equivalent=True)
out = {}
for name, g in (("bic", Genome(**base)), ("gprior", Genome(**base, evidence="gprior"))):
    t0 = time.time()
    out[name] = evaluate(Toolkit(g, []).to_config(), split, N, workers=2)
    print(name, round(time.time() - t0), "s", flush=True)
json.dump(out, open(f"dev_{split}_{N}.json", "w"), default=str)
a, b = out["bic"], out["gprior"]
for kind in ("familiar", "gap", None):
    x = [r for r in a if kind is None or r["kind"] == kind]; y = [r for r in b if kind is None or r["kind"] == kind]
    c = paired_bootstrap([r["score"] for r in x], [r["score"] for r in y], n_boot=2000, seed=1)
    def m(rs, f): return round(statistics.mean(f(r) for r in rs), 4)
    cw = lambda r: 1.0 if (r["claimed"] and r["p_claim"] >= 0.9 and not r["recovered"]) else 0.0
    print(f"{str(kind):8s} n={c.n} score {m(x, lambda r: r['score'])}->{m(y, lambda r: r['score'])} d={c.mean_delta:+.4f} [{c.ci_low:+.4f},{c.ci_high:+.4f}]"
          f" rec {m(x, lambda r: r['recovered'])}->{m(y, lambda r: r['recovered'])} confwrong {m(x, cw)}->{m(y, cw)}"
          f" meas {m(x, lambda r: r['measurements'])}->{m(y, lambda r: r['measurements'])}"
          f" none {m(x, lambda r: r['answer']=='none')}->{m(y, lambda r: r['answer']=='none')}")
