"""Knowing when you don't know.

* `Calibration` measures whether stated confidences match observed accuracy
  (Brier score, log score, expected calibration error). A system that says
  "90%" should be right about 90% of the time.
* `should_ask_human` decides when human input is worth the interruption:
  only when the expected loss avoided by asking exceeds the cost of asking.
* `DiminishingReturns` decides when further iteration is not worth it.
* `agreement_confidence` turns independent re-derivations into a confidence
  estimate (self-consistency).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Hashable, Sequence


@dataclass
class Calibration:
    records: list[tuple[float, bool]] = field(default_factory=list)

    def add(self, confidence: float, correct: bool) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        self.records.append((confidence, bool(correct)))

    def __len__(self) -> int:
        return len(self.records)

    def brier(self) -> float:
        if not self.records:
            return float("nan")
        return sum((c - float(y)) ** 2 for c, y in self.records) / len(self.records)

    def log_score(self, eps: float = 1e-6) -> float:
        """Mean log-probability assigned to what actually happened (higher is better)."""
        if not self.records:
            return float("nan")
        total = 0.0
        for c, y in self.records:
            p = min(1 - eps, max(eps, c if y else 1 - c))
            total += math.log(p)
        return total / len(self.records)

    def accuracy(self) -> float:
        if not self.records:
            return float("nan")
        return sum(1 for _, y in self.records if y) / len(self.records)

    def mean_confidence(self) -> float:
        if not self.records:
            return float("nan")
        return sum(c for c, _ in self.records) / len(self.records)

    def reliability(self, n_bins: int = 10) -> list[dict]:
        bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
        for c, y in self.records:
            bins[min(n_bins - 1, int(c * n_bins))].append((c, y))
        out = []
        for i, b in enumerate(bins):
            if b:
                out.append({"bin": (i / n_bins, (i + 1) / n_bins), "n": len(b),
                            "mean_confidence": sum(c for c, _ in b) / len(b),
                            "accuracy": sum(1 for _, y in b if y) / len(b)})
        return out

    def ece(self, n_bins: int = 10) -> float:
        """Expected calibration error: weighted |confidence - accuracy| over bins."""
        if not self.records:
            return float("nan")
        n = len(self.records)
        return sum(r["n"] / n * abs(r["mean_confidence"] - r["accuracy"]) for r in self.reliability(n_bins))

    def summary(self) -> dict:
        return {"n": len(self), "accuracy": self.accuracy(), "mean_confidence": self.mean_confidence(),
                "brier": self.brier(), "log_score": self.log_score(), "ece": self.ece()}


@dataclass(frozen=True)
class AskDecision:
    ask: bool
    expected_loss_if_proceed: float
    reason: str


def should_ask_human(
    confidence: float,
    impact: float,
    *,
    ask_cost: float = 0.1,
    human_available: bool = True,
    irreversible: bool = False,
) -> AskDecision:
    """Ask only when the expected loss of proceeding on our best guess exceeds
    the cost of interrupting a human.

    `impact` is the loss (0..1 scale relative to the task) if the best guess
    is wrong. Irreversible decisions double the effective impact: they cannot
    be corrected later, so being wrong costs more.
    """
    effective_impact = impact * (2.0 if irreversible else 1.0)
    expected_loss = (1.0 - confidence) * effective_impact
    if not human_available:
        return AskDecision(False, expected_loss, "no human available: proceed and record the assumption")
    if expected_loss > ask_cost:
        return AskDecision(True, expected_loss,
                           f"expected loss {expected_loss:.2f} exceeds cost of asking {ask_cost:.2f}")
    return AskDecision(False, expected_loss,
                       f"expected loss {expected_loss:.2f} below cost of asking {ask_cost:.2f}")


@dataclass
class DiminishingReturns:
    """Stop iterating once the best progress signal has not improved by at
    least `min_delta` over the last `patience` updates."""
    patience: int = 3
    min_delta: float = 1e-3
    history: list[float] = field(default_factory=list)

    def update(self, value: float) -> bool:
        """Record a progress value; return True if we should stop."""
        self.history.append(value)
        return self.should_stop()

    def should_stop(self) -> bool:
        if len(self.history) <= self.patience:
            return False
        best_before = max(self.history[: -self.patience])
        best_recent = max(self.history[-self.patience:])
        return best_recent - best_before < self.min_delta


def agreement_confidence(answers: Sequence[Hashable]) -> tuple[Hashable | None, float]:
    """Majority answer and its Laplace-smoothed agreement rate."""
    if not answers:
        return None, 0.0
    counts = Counter(answers)
    best, k = counts.most_common(1)[0]
    return best, (k + 1) / (len(answers) + 2)
