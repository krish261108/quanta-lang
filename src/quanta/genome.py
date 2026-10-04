"""The genome: everything about the agent's *strategy* that self-improvement
is allowed to change, with explicit bounds.

Deliberately absent: anything about governance (approval ceilings, budgets,
kill switch, audit). Those live in `governance.py`, which the self-improver
treats as a protected surface. A genome that tries to carry extra fields is
rejected at load time.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParamSpec:
    kind: str                       # "int" | "float" | "choice" | "bool" | "text"
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] = ()
    doc: str = ""


SPACE: dict[str, ParamSpec] = {
    # --- scientific discovery strategy -------------------------------------
    "n_initial": ParamSpec("int", 3, 15, doc="space-filling measurements before adaptive design"),
    "design": ParamSpec("choice", choices=("random", "space_filling", "disagreement"),
                        doc="how the next experiment is chosen"),
    "n_candidates": ParamSpec("int", 5, 200, doc="candidate inputs scored per adaptive step"),
    "explore_prob": ParamSpec("float", 0.0, 0.5, doc="probability of a space-filling step instead"),
    "likelihood": ParamSpec("choice", choices=("gaussian", "student_t"),
                            doc="noise model; student_t is robust to outliers"),
    "t_dof": ParamSpec("float", 1.5, 30.0, doc="degrees of freedom for student_t"),
    "stop_posterior": ParamSpec("float", 0.6, 0.999, doc="posterior needed to stop"),
    "min_experiments": ParamSpec("int", 3, 30, doc="never stop before this many measurements"),
    "patience": ParamSpec("int", 0, 15, doc="stop if leader posterior stalls this long (0=off)"),
    "falsification_rounds": ParamSpec("int", 0, 8, doc="adversarial tests of the leader before stopping"),
    "temperature": ParamSpec("float", 0.5, 4.0, doc="posterior tempering for calibrated credence"),
    "flexible_baseline": ParamSpec("bool", doc="compete named laws against an unnamed flexible curve"),
    "expand_on_inadequacy": ParamSpec("bool", doc="add extended hypothesis families when named laws fail"),
    "use_learned_priors": ParamSpec("bool", doc="priors from verified past outcomes in memory"),
    "prior_pseudocount": ParamSpec("float", 0.1, 20.0, doc="Dirichlet smoothing for learned priors"),
    "ask_below": ParamSpec("float", 0.0, 0.95, doc="request human input below this credence"),
    # --- LLM research-loop strategy ----------------------------------------
    "min_hypotheses": ParamSpec("int", 1, 8, doc="competing hypotheses required before planning"),
    "research_depth": ParamSpec("int", 0, 3, doc="0 skips literature/documentation research"),
    "verification": ParamSpec("choice", choices=("none", "self", "independent", "adversarial"),
                              doc="who checks conclusions"),
    "max_iterations": ParamSpec("int", 1, 12, doc="outer hypothesize-test-revise iterations"),
    "main_effort": ParamSpec("choice", choices=("low", "medium", "high", "xhigh", "max")),
    "subagent_effort": ParamSpec("choice", choices=("low", "medium", "high", "xhigh", "max")),
    "delegate": ParamSpec("bool", doc="allow spawning specialized sub-agents"),
    "phase_notes": ParamSpec("text", doc="extra per-phase instructions (JSON object phase->text)"),
}


class GenomeError(ValueError):
    pass


@dataclass(frozen=True)
class Genome:
    n_initial: int = 6
    design: str = "random"
    n_candidates: int = 40
    explore_prob: float = 0.0
    likelihood: str = "gaussian"
    t_dof: float = 4.0
    stop_posterior: float = 0.95
    min_experiments: int = 8
    patience: int = 0
    falsification_rounds: int = 0
    temperature: float = 1.0
    flexible_baseline: bool = False
    expand_on_inadequacy: bool = False
    use_learned_priors: bool = False
    prior_pseudocount: float = 2.0
    ask_below: float = 0.5
    min_hypotheses: int = 3
    research_depth: int = 1
    verification: str = "independent"
    max_iterations: int = 3
    main_effort: str = "high"
    subagent_effort: str = "medium"
    delegate: bool = True
    phase_notes: str = "{}"

    def __post_init__(self) -> None:
        errors = self.validate()
        if errors:
            raise GenomeError("; ".join(errors))

    def validate(self) -> list[str]:
        errors = []
        for f in fields(self):
            spec = SPACE[f.name]
            v = getattr(self, f.name)
            if spec.kind == "int":
                if not isinstance(v, int) or isinstance(v, bool) or not spec.low <= v <= spec.high:
                    errors.append(f"{f.name}={v!r} must be an int in [{spec.low}, {spec.high}]")
            elif spec.kind == "float":
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not spec.low <= v <= spec.high:
                    errors.append(f"{f.name}={v!r} must be a number in [{spec.low}, {spec.high}]")
            elif spec.kind == "choice":
                if v not in spec.choices:
                    errors.append(f"{f.name}={v!r} must be one of {spec.choices}")
            elif spec.kind == "bool":
                if not isinstance(v, bool):
                    errors.append(f"{f.name}={v!r} must be a bool")
            elif spec.kind == "text":
                try:
                    obj = json.loads(v)
                    if not isinstance(obj, dict) or not all(isinstance(x, str) for x in obj.values()):
                        raise ValueError
                except (ValueError, TypeError):
                    errors.append(f"{f.name} must be a JSON object of strings")
        return errors

    # -- (de)serialization ------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Genome":
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise GenomeError(f"unknown genome fields (not part of the mutable strategy space): {sorted(unknown)}")
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Genome":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    # -- variation ---------------------------------------------------------
    def mutate(self, **changes: Any) -> "Genome":
        unknown = set(changes) - {f.name for f in fields(self)}
        if unknown:
            raise GenomeError(f"unknown genome fields: {sorted(unknown)}")
        return replace(self, **changes)

    def diff(self, other: "Genome") -> dict[str, tuple[Any, Any]]:
        a, b = self.to_dict(), other.to_dict()
        return {k: (a[k], b[k]) for k in a if a[k] != b[k]}

    def phase_note(self, phase: str) -> str:
        return json.loads(self.phase_notes).get(phase, "")


def clamp_to_space(name: str, value: Any) -> Any:
    """Coerce a proposed value into the legal range of parameter `name`."""
    spec = SPACE[name]
    if spec.kind == "int":
        return int(min(spec.high, max(spec.low, round(float(value)))))
    if spec.kind == "float":
        return float(min(spec.high, max(spec.low, float(value))))
    if spec.kind == "bool":
        return bool(value)
    if spec.kind == "choice":
        if value not in spec.choices:
            raise GenomeError(f"{value!r} is not a legal value for {name}")
        return value
    return value
