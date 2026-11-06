"""Category D, part 2: real datasets nobody on this project chose.

PROTECTED, report-only. Never used for any decision by the acquisition loop or
the reviewer.

Selection rule (applied mechanically to the Rdatasets index on 2026-10-06, not by
judgement): every item of R's `datasets` package with exactly two columns, both
numeric, and 15 to 2000 rows. The first column is x, the second y (for time
series: time and value). That gave 30 items; `co2` returned HTTP 404 and is
excluded. The remaining 29 are pinned below with the sha256 of the exact file;
a changed file is an error, not a silent substitution.

Tasks. Each dataset's x range is mapped linearly onto [0.1, 13]. Points mapped
above 10 are the extrapolation holdout (the solver can never measure there).
Of the rest, a random 20% (at least 3) are the interpolation holdout; the others
form the measurable pool. A measurement request at x returns the nearest unused
pool point and reports its actual x. Each dataset gives three tasks with
different interpolation holdouts.

Metrics. There is no ground-truth law, so there is no "recovered" or claim
calibration here: only prediction error on held-out points,
    interp_nrmse = RMSE(holdout inside the measured range) / sd(y)
    extrap_nrmse = RMSE(points beyond the measured range) / sd(y)
    score_ext    = 0.5 exp(-interp_nrmse) + 0.5 exp(-extrap_nrmse)
Comparisons between solver configurations are paired by task and resampled by
dataset (tasks from one dataset are not independent).
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/{item}.csv"
DATASETS: tuple[tuple[str, str], ...] = (
    ("airmiles", "097cfc50332da524d4596d2d44075f3210bc3fd94db91549548aa4d5b9d9bc40"),
    ("AirPassengers", "bdb98adbd418a6de6842a742e0602f363c3b841c26677e7e61a1e055e9509bd8"),
    ("austres", "676629939eb52bae0ab9ce1cf6f094d0628eb79c8ee3627decbd550a97ece587"),
    ("BJsales", "209810170b714bf0da43a79fa2bb679b1aea6edd76881a7c3c6123590e019f80"),
    ("BJsales.lead", "9188fe385aca6c1ec87fd5ddcd07ca1470761681a6c1b85a23e49cf4032983fd"),
    ("cars", "08e75516e5fd1377db316a3ed790545c942dcf905c69f49d0c6f16873a4a6242"),
    ("discoveries", "02bb14a10cd308044c8b0427ce6ced50e06b1423b9624cd027473c1c4e7ffc8b"),
    ("faithful", "5043db1e2c51c8e8fd67e0868c768ae589770cc76ad0ac0c5b7afd1fca31fc57"),
    ("fdeaths", "c083d7adc92229e249a64b3f5f896daf087851c076c4c57633181dd048bcd19c"),
    ("freeny.y", "e4691024b94231e93b44dd3302e231d76e4db3033aed487ba051a64769aac118"),
    ("JohnsonJohnson", "8d84846f400a24fd6bed71d80c5a89b9cda2e47702ef58c03610a538ca9157e7"),
    ("LakeHuron", "7fe634970d5b28128ade898c3932dbc4015fecbeb6dabc443efaf88a2fd037d8"),
    ("ldeaths", "01b316c6e5c2302f7b27270d615a9a9cbf0e849f09a590f06cbb530e5cef00a2"),
    ("lh", "d394f481a27ed1b0ba1ad2c1cd69d6938eaa2e879fd63a369c20dbd0e1630037"),
    ("lynx", "0766d4043eb8a20ac83afefff1c78063beff3891571c59e8e38d72503ca86cf1"),
    ("mdeaths", "9caa553d91173696f77c49462d7ee1042ced51ab10039ad51e8a6354ceb0be27"),
    ("nhtemp", "7178c1f0b23989b5a1d8b5764528c18fe3c00961907eaa2ee40635f4724a7ddb"),
    ("Nile", "d0452bea38c61e796a4eeb950bf91d20fb5c7f13d5822eadf5990fe54f9c8d07"),
    ("nottem", "634e2e374c5d59c1332e6e6de207d691cb4ebdcd007f7f0e2993ac2a19c90473"),
    ("presidents", "e69d938cafd0ba2892da000605844af43f6506f2022c054944466de5de3bc5fc"),
    ("pressure", "5ef371dc3f05836aab2f46025c5cc7574a460afbc6d31d875d73194ac1f157b5"),
    ("state.center", "39c799803bd6fca7c240cd9b2e31c9ca6ae6520f6db34a4fa9517fb6c9febe87"),
    ("sunspot.year", "f6d6b62da79feac7408cd876efff2bf9c9853fd6b76b59ed3717a51d997764f9"),
    ("UKDriverDeaths", "5ea8754082b2a4649966f6f639dc8a06fef2a1e2f7e3a41a372501986369179f"),
    ("UKgas", "69ec013fbeeaf956a42603aaf8f68d943a3aab5f90b6ed89c5de2ed02eba59ff"),
    ("USAccDeaths", "7eb8927b29341a5abe4420a278e4e21100af1a14212d700a0f296f7462377b29"),
    ("uspop", "3a79f8e6ee20b4b73d967a682c8ec3670894bd7b97d808d04d2099b9d590d0db"),
    ("women", "ff95dfaa417b2da885d08a1e9844a73ccf398263ac29b823e227437e590b361c"),
    ("WWWusage", "245774828aa5bc64292f29db15434004b7472022b73b81a3d5d1c02f55c42e08"),
)
EXCLUDED = {"co2": "HTTP 404 at selection time"}
TASKS_PER_DATASET = 3
SEED_BASE = 2_000_000
REPLICATION_STRIDE = 10_000_000
MEASURE_MAX = 10.0
MAP_RANGE = (0.1, 13.0)
HOLDOUT_FRACTION = 0.2
MAX_BUDGET = 30


class DataError(RuntimeError):
    pass


def cache_dir() -> Path:
    return Path(os.environ.get("QUANTA_DATA_CACHE", Path.home() / ".cache" / "quanta" / "rdatasets"))


def fetch(item: str, sha256: str) -> bytes:
    """The exact pinned file (from the cache if present)."""
    path = cache_dir() / f"{item}.csv"
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(BASE_URL.format(item=item), timeout=60) as r:
            data = r.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise DataError(f"{item}: sha256 {got} does not match the pinned {sha256}")
    return data


def parse(data: bytes) -> list[tuple[float, float]]:
    """(x, y) pairs from the last two columns; rows with missing values are dropped."""
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
    out = []
    for r in rows[1:]:
        if len(r) < 2:
            continue
        try:
            x, y = float(r[-2]), float(r[-1])
        except ValueError:
            continue
        if math.isfinite(x) and math.isfinite(y):
            out.append((x, y))
    return out


@dataclass
class ExternalTask:
    task_id: str
    dataset: str
    seed: int
    budget: int
    pool: list[tuple[float, float]]          # mapped (x, y), measurable
    interp: list[tuple[float, float]]        # mapped (x, y), held out inside [0.1, 10]
    extrap: list[tuple[float, float]]        # mapped (x, y), beyond 10
    sd_y: float

    @property
    def eval_x(self) -> tuple[float, ...]:
        return tuple(x for x, _ in self.interp) + tuple(x for x, _ in self.extrap)


def make_tasks(name: str, points: list[tuple[float, float]], *, replication: int = 0,
               index0: int = 0) -> list[ExternalTask]:
    xs = [x for x, _ in points]
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        raise DataError(f"{name}: x does not vary")
    a, b = MAP_RANGE
    mapped = sorted((a + (b - a) * (x - lo) / (hi - lo), y) for x, y in points)
    inside = [p for p in mapped if p[0] <= MEASURE_MAX]
    extrap = [p for p in mapped if p[0] > MEASURE_MAX]
    if len(inside) < 8:
        raise DataError(f"{name}: too few points inside the measurable range")
    ys = [y for _, y in mapped]
    m = sum(ys) / len(ys)
    sd = math.sqrt(sum((y - m) ** 2 for y in ys) / len(ys)) or 1.0
    tasks = []
    for k in range(TASKS_PER_DATASET):
        seed = SEED_BASE + index0 + k + REPLICATION_STRIDE * replication
        rng = random.Random(seed)
        n_hold = max(3, round(HOLDOUT_FRACTION * len(inside)))
        hold_idx = set(rng.sample(range(len(inside)), min(n_hold, len(inside) - 4)))
        interp = [p for i, p in enumerate(inside) if i in hold_idx]
        pool = [p for i, p in enumerate(inside) if i not in hold_idx]
        tasks.append(ExternalTask(f"external:{replication}:{name}:{k}", name, seed, min(MAX_BUDGET, len(pool)),
                                  pool, interp, extrap, sd))
    return tasks


def tasks(replication: int = 0) -> list[ExternalTask]:
    out = []
    for i, (name, sha) in enumerate(DATASETS):
        out.extend(make_tasks(name, parse(fetch(name, sha)), replication=replication,
                              index0=TASKS_PER_DATASET * i))
    return out


class PoolInstrument:
    """Observational data: a request at x returns the nearest unused pool point
    (and, through `measure_at`, the input actually measured)."""

    def __init__(self, task: ExternalTask) -> None:
        self.domain = (MAP_RANGE[0], MEASURE_MAX)
        pool = list(task.pool)
        used = [False] * len(pool)

        def measure_at(x: float) -> tuple[float, float]:
            free = [i for i in range(len(pool)) if not used[i]] or list(range(len(pool)))
            i = min(free, key=lambda j: (abs(pool[j][0] - x), j))
            used[i] = True
            return pool[i]

        self.measure_at = measure_at

    def measure(self, x: float) -> float:
        return self.measure_at(x)[1]


def grade(task: ExternalTask, pred: list[float]) -> dict:
    def nrmse(points, preds):
        if not points:
            return None
        cap = (10 * task.sd_y) ** 2
        se = 0.0
        for (_, y), p in zip(points, preds):
            err = (p - y) if (p is not None and math.isfinite(p)) else 10 * task.sd_y
            se += min(err * err, cap)
        return math.sqrt(se / len(points)) / task.sd_y

    k = len(task.interp)
    interp = nrmse(task.interp, pred[:k])
    extrap = nrmse(task.extrap, pred[k:])
    parts = [v for v in (interp, extrap) if v is not None]
    score = sum(math.exp(-v) for v in parts) / len(parts)
    return {"interp_nrmse": interp, "extrap_nrmse": extrap, "score_ext": score}


def evaluate(config: dict, *, replication: int = 0, workers: int = 4, timeout: float = 600.0,
             task_list: list[ExternalTask] | None = None) -> list[dict]:
    from .acquisition import solver_seed
    from .isolation import Job, run_isolated
    ts = task_list if task_list is not None else tasks(replication)
    jobs = [Job(t.task_id, PoolInstrument(t), t.budget, solver_seed(t.seed), t.eval_x) for t in ts]
    outs = run_isolated(config, jobs, workers=workers, timeout=timeout)
    records = []
    for t, o in zip(ts, outs):
        rec = {"task_id": t.task_id, "dataset": t.dataset, "seed": t.seed, "replication": replication,
               "kind": "external", "budget": t.budget, "n_interp": len(t.interp), "n_extrap": len(t.extrap)}
        if o.get("error"):
            rec.update(grade(t, [float("nan")] * len(t.eval_x)), answer="error", answer_kind="error",
                       credence=0.0, error=o["error"])
        else:
            rec.update(grade(t, o["pred"]), answer=o["answer"], answer_kind=o["kind"], credence=o["credence"],
                       error=None, measurements=o["measurements_used"], description=o.get("description", ""))
        records.append(rec)
    return records


def cluster_bootstrap(base: list[dict], cand: list[dict], key: str = "score_ext", *, n_boot: int = 4000,
                      confidence: float = 0.95, seed: int = 0) -> dict:
    """Paired difference (cand - base) of `key`, resampling whole datasets."""
    if [r["task_id"] for r in base] != [r["task_id"] for r in cand]:
        raise ValueError("unpaired records")
    by: dict[str, list[float]] = {}
    for a, b in zip(base, cand):
        if a[key] is None or b[key] is None:
            continue
        by.setdefault(a["dataset"], []).append(b[key] - a[key])
    groups = list(by.values())
    if not groups:
        return {"n_datasets": 0, "n_tasks": 0, "mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "wins": 0, "losses": 0}
    mean = sum(sum(g) for g in groups) / sum(len(g) for g in groups)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        stats.append(sum(sum(g) for g in sample) / sum(len(g) for g in sample))
    stats.sort()
    lo_i = int((1 - confidence) / 2 * n_boot)
    hi_i = min(n_boot - 1, int((1 + confidence) / 2 * n_boot))
    wins = sum(1 for g in groups for d in g if d > 1e-12)
    losses = sum(1 for g in groups for d in g if d < -1e-12)
    return {"n_datasets": len(groups), "n_tasks": sum(len(g) for g in groups), "mean_delta": mean,
            "ci_low": stats[lo_i], "ci_high": stats[hi_i], "wins": wins, "losses": losses}
