import math
import random

import pytest

from quanta.bench.harness import run_suite, run_task
from quanta.bench.tasks import (GRID, OUT_OF_LIBRARY, closest_rival_nrmse, generate_task, grade)
from quanta.genome import Genome
from quanta.science.discovery import DiscoveryLoop, posterior_from_fits
from quanta.science.hypotheses import (FAMILIES, ExpressionHypothesis, FamilyHypothesis,
                                       FlexibleHypothesis, FormulaError, compile_formula, solve_wls)
from quanta.epistemics import Status


def test_solve_wls_exact():
    xs = [0.5 * i for i in range(10)]
    rows = [(1.0, x, x * x) for x in xs]
    ys = [2 - 3 * x + 0.5 * x * x for x in xs]
    coef = solve_wls(rows, ys)
    assert coef == pytest.approx([2, -3, 0.5], abs=1e-6)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_each_family_recovers_its_own_noise_free_data(name):
    examples = {
        "constant": (3.0,), "linear": (1.0, 2.0), "quadratic": (1.0, -1.0, 0.3),
        "cubic": (1.0, -1.0, 0.3, -0.04), "logarithmic": (1.0, 2.0), "exponential": (2.0, 0.3),
        "power": (1.5, 0.6), "saturating": (4.0, 2.0), "sinusoid": (1.0, 2.0, 1.7, 0.4),
        "step": (1.0, 3.0, 5.05), "linear_sin": (1.0, 0.5, 1.5, 2.2, 0.3),
    }
    from quanta.bench.tasks import truth_value
    xs = [0.1 + 9.9 * i / 39 for i in range(40)]
    ys = [truth_value(name, examples[name], x) for x in xs]
    fit = FamilyHypothesis(name).fit(xs, ys)
    sd = math.sqrt(sum((y - sum(ys) / len(ys)) ** 2 for y in ys) / len(ys)) or 1.0
    assert math.sqrt(fit.rss / len(xs)) / sd < 1e-3


def test_too_little_data_cannot_be_assessed():
    xs, ys = [1.0, 2.0, 3.0, 4.0], [1.0, 2.1, 2.9, 4.2]
    fit = FlexibleHypothesis().fit(xs, ys)
    assert fit.bic == math.inf
    post = posterior_from_fits({"linear": FamilyHypothesis("linear").fit(xs, ys), "none": fit},
                               {"linear": math.log(0.5), "none": math.log(0.5)})
    assert "none" not in post and post["linear"] == pytest.approx(1.0)


def test_formula_whitelist_blocks_code_execution():
    for bad in ["__import__('os').system('true')", "x.__class__", "(lambda: 1)()", "open('f')",
                "[i for i in range(3)]", "a if x else b"]:
        with pytest.raises(FormulaError):
            compile_formula(bad, ["a", "b"])
    f = compile_formula("a * exp(-b * x) + sin(pi * x)", ["a", "b"])
    assert f(1.0, [2.0, 0.0]) == pytest.approx(2.0 + math.sin(math.pi))


def test_expression_hypothesis_fits_novel_law():
    rng = random.Random(0)
    xs = [0.2 * i + 0.1 for i in range(40)]
    ys = [3.0 * math.exp(-0.5 * x) * math.cos(2.0 * x) + rng.gauss(0, 0.01) for x in xs]
    h = ExpressionHypothesis("damped", "a*exp(-b*x)*cos(w*x)", ("a", "b", "w"), restarts=12)
    fit = h.fit(xs, ys)
    assert fit.params["a"] == pytest.approx(3.0, rel=0.05)
    assert fit.params["w"] == pytest.approx(2.0, rel=0.05)


def test_task_generation_is_deterministic_and_identifiable():
    for seed in range(10_000, 10_030):
        t = generate_task(seed)
        assert generate_task(seed) == t
        noise_frac = t.noise_sd / t.truth_sd
        if t.family != "constant":
            assert closest_rival_nrmse(t.family, t.params) >= 3 * noise_frac - 1e-9
        if t.family in OUT_OF_LIBRARY:
            assert t.expected_answer == "none"


def test_grading_is_a_proper_score():
    t = generate_task(10_000)

    def perfect(x):
        return t.truth(x)

    right_sure = grade(t, t.expected_answer, 0.95, perfect, 10)
    wrong_sure = grade(t, "bogus", 0.95, perfect, 10)
    wrong_unsure = grade(t, "bogus", 0.3, perfect, 10)
    assert right_sure["score"] > wrong_unsure["score"] > wrong_sure["score"]
    cheap = grade(t, t.expected_answer, 0.95, perfect, 10)
    costly = grade(t, t.expected_answer, 0.95, perfect, 30)
    assert cheap["score"] > costly["score"]


def _find_seed(family: str) -> int:
    for seed in range(10_000, 12_000):
        t = generate_task(seed)
        if t.family == family and t.outlier_rate == 0:
            return seed
    raise AssertionError(family)


def test_discovery_finds_clean_law_and_records_evidence():
    seed = _find_seed("sinusoid")
    task = generate_task(seed)
    g = Genome(design="disagreement", falsification_rounds=2)
    res = DiscoveryLoop(g, budget=30, seed=seed).run(task.system())
    assert res.answer == "sinusoid"
    assert res.credence > 0.9
    assert len(res.falsifications) >= 1
    led = res.ledger
    observations = [e for e in led.evidence.values() if e.kind == "observation"]
    assert len(observations) == res.n_experiments
    conclusion = [c for c in led.claims.values() if c.statement.startswith("Conclusion")][0]
    assert conclusion.status is Status.INFERENCE
    assert conclusion.confidence == pytest.approx(res.credence)
    hyp = [c for c in led.claims.values() if "'sinusoid' law" in c.statement][0]
    assert hyp.falsifications and hyp.falsifications[0].outcome in ("survived", "refuted")


def test_flexible_baseline_says_none_for_unknown_law():
    seed = _find_seed("gaussian_bump")
    task = generate_task(seed)
    g = Genome(design="disagreement", flexible_baseline=True, min_experiments=12)
    res = DiscoveryLoop(g, budget=30, seed=seed).run(task.system())
    assert res.answer == "none"
    without = DiscoveryLoop(Genome(design="disagreement"), budget=30, seed=seed).run(task.system())
    assert without.answer != "none"


def test_expansion_revises_hypothesis_space():
    g = Genome(design="disagreement", flexible_baseline=True, expand_on_inadequacy=True,
               likelihood="student_t", min_experiments=12)
    seeds = []
    for s in range(10_000, 12_000):
        t = generate_task(s)
        if t.family == "linear_sin" and t.outlier_rate == 0:
            seeds.append(s)
            if len(seeds) == 5:
                break
    solved = 0
    for seed in seeds:
        res = DiscoveryLoop(g, budget=30, seed=seed).run(generate_task(seed).system())
        if res.expanded:
            assert "linear_sin" in res.hypotheses_considered
            assert any("inadequate" in c.statement for c in res.ledger.claims.values())
        solved += res.answer == "linear_sin"
    assert solved >= 3


def test_low_credence_requests_human_input():
    seed = _find_seed("constant")
    res = DiscoveryLoop(Genome(ask_below=0.95), budget=10, seed=seed).run(generate_task(seed).system())
    if res.credence < 0.95:
        assert res.human_request and "most informative next experiment" in res.human_request


def test_suite_is_deterministic_across_process_counts():
    g = Genome(design="disagreement")
    a = run_suite(g, "selection", 8, jobs=1)
    b = run_suite(g, "selection", 8, jobs=2)
    assert a.scores == b.scores
    assert set(a.diagnostics) >= {"accuracy", "ece", "accuracy_by_category", "top_confusions"}
    r = run_task(g, 10_003)
    assert len(GRID) == 101 and 0 <= r["credence"] <= 1
