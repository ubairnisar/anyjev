"""Result objects returned by the Judge."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OrderScore:
    """Scores from one option order (one request)."""

    order: list[str]                 # option keys in the order they were shown (A, B, ...)
    probs: dict[str, float]          # option key -> probability
    sampled_token: str
    missing: list[str] = field(default_factory=list)  # option keys absent from top-k


@dataclass
class Result:
    """The outcome of one judgement.

    label:       the decided option key (threshold-aware).
    probs:       option key -> probability, averaged over all orders.
    confidence:  probability of the decided label.
    order_gap:   largest disagreement between orders for any option (0 = perfectly consistent).
    uncertain:   True if the result should not be trusted blindly; see ``reasons``.
    """

    task: str
    label: str
    probs: dict[str, float]
    confidence: float
    order_gap: float
    uncertain: bool
    reasons: list[str]
    positive: str | None = None
    threshold: float | None = None
    per_order: list[OrderScore] = field(default_factory=list)
    input_id: str | None = None
    error: str | None = None

    @property
    def is_positive(self) -> bool:
        """For tasks with ``positive`` set: did the label land on it?"""
        if self.positive is None:
            raise AttributeError("is_positive needs a task with positive=<option key>.")
        return self.label == self.positive

    @property
    def p_positive(self) -> float:
        if self.positive is None:
            raise AttributeError("p_positive needs a task with positive=<option key>.")
        return self.probs[self.positive]

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        probs = ", ".join(f"{k}={v:.3f}" for k, v in sorted(self.probs.items(), key=lambda kv: -kv[1]))
        flag = " UNCERTAIN" if self.uncertain else ""
        who = f"{self.input_id}: " if self.input_id else ""
        if self.error:
            return f"<Result {who}ERROR {self.error}>"
        return f"<Result {who}{self.label} ({probs}) gap={self.order_gap:.3f}{flag}>"

    @classmethod
    def failed(cls, task: str, error: BaseException, input_id: str | None = None) -> "Result":
        return cls(task=task, label="", probs={}, confidence=0.0, order_gap=0.0,
                   uncertain=True, reasons=["error"], input_id=input_id,
                   error=f"{type(error).__name__}: {error}")
