"""Pick a decision threshold from labelled scores."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class Point:
    threshold: float
    recall: float
    precision: float
    false_positive_rate: float


@dataclass
class Calibration:
    """Chosen threshold plus the metrics it achieves on the calibration set."""

    threshold: float
    recall: float
    precision: float
    false_positive_rate: float
    n_positive: int
    n_negative: int
    target_recall: float
    curve: list[Point] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"threshold           {self.threshold:.4f}",
            f"recall              {self.recall:.3f}   (target {self.target_recall:.3f})",
            f"precision           {self.precision:.3f}",
            f"false positive rate {self.false_positive_rate:.3f}",
            f"samples             {self.n_positive} positive / {self.n_negative} negative",
            "",
            "threshold  recall  precision  fpr",
        ]
        for p in self.curve:
            lines.append(f"{p.threshold:9.3f}  {p.recall:6.3f}  {p.precision:9.3f}  {p.false_positive_rate:.3f}")
        return "\n".join(lines)


def _metrics(scores: Sequence[float], truth: Sequence[bool], thr: float) -> Point:
    tp = sum(1 for s, t in zip(scores, truth) if t and s >= thr)
    fp = sum(1 for s, t in zip(scores, truth) if not t and s >= thr)
    pos = sum(truth)
    neg = len(truth) - pos
    return Point(
        threshold=thr,
        recall=tp / pos if pos else 1.0,
        precision=tp / (tp + fp) if (tp + fp) else 1.0,
        false_positive_rate=fp / neg if neg else 0.0,
    )


def calibrate_threshold(
    scores: Sequence[float], truth: Sequence[bool], *, target_recall: float = 1.0
) -> Calibration:
    """Highest threshold whose recall on positives is at least ``target_recall``.

    The highest such threshold gives the fewest false positives for that recall.
    """
    if len(scores) != len(truth):
        raise ValueError("scores and truth must have the same length.")
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1].")
    positives = sorted(s for s, t in zip(scores, truth) if t)
    if not positives:
        raise ValueError("Calibration needs at least one positive example.")

    # We may miss at most k positives; the threshold sits at the (k)-th lowest positive score.
    k = math.floor((1.0 - target_recall) * len(positives) + 1e-9)
    thr = positives[min(k, len(positives) - 1)]
    best = _metrics(scores, truth, thr)

    grid = sorted({round(x, 2) for x in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)} | {thr})
    return Calibration(
        threshold=thr, recall=best.recall, precision=best.precision,
        false_positive_rate=best.false_positive_rate,
        n_positive=len(positives), n_negative=len(truth) - len(positives),
        target_recall=target_recall,
        curve=[_metrics(scores, truth, t) for t in grid],
    )
