"""Deterministic judge with injectable accuracy and biases, for offline runs and metric tests."""

import hashlib
from collections import Counter
from typing import Literal

from judgecheck.metrics import length_side
from judgecheck.results import Order, Position, to_position
from judgecheck.schema import Comparison, Verdict

_ORDER: tuple[Verdict, ...] = ("a", "b", "tie")


class MockJudge:
    """Follows the experts' majority verdict, with seeded deviations.

    Per comparison, in order of precedence: with probability ``position_bias`` it picks
    whichever answer is shown first; with probability ``verbosity_bias`` it picks the
    longer answer (when the lengths differ); otherwise it follows the majority expert
    verdict with probability ``accuracy`` and picks another verdict if not. Only the
    position bias depends on display order, so a bias-free mock is perfectly consistent.
    ``failure_rate`` returns unusable responses.
    """

    def __init__(
        self,
        name: str,
        *,
        seed: int = 0,
        accuracy: float = 0.7,
        position_bias: float = 0.0,
        verbosity_bias: float = 0.0,
        failure_rate: float = 0.0,
        min_length_ratio: float = 1.2,
    ) -> None:
        for label, value in (
            ("accuracy", accuracy),
            ("position_bias", position_bias),
            ("verbosity_bias", verbosity_bias),
            ("failure_rate", failure_rate),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be between 0 and 1")
        self.name = name
        self.seed = seed
        self.accuracy = accuracy
        self.position_bias = position_bias
        self.verbosity_bias = verbosity_bias
        self.failure_rate = failure_rate
        self.min_length_ratio = min_length_ratio

    def _uniform(self, comparison_id: str, tag: str) -> float:
        digest = hashlib.sha256(f"{self.seed}|{comparison_id}|{tag}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    def judge(self, comparison: Comparison, shown_first: Literal["a", "b"]) -> Position | None:
        order: Order = "ab" if shown_first == "a" else "ba"
        if self._uniform(comparison.id, f"fail|{order}") < self.failure_rate:
            return None
        if self._uniform(comparison.id, f"position|{order}") < self.position_bias:
            return "first"
        longer = length_side(comparison, self.min_length_ratio)
        if longer is not None and self._uniform(comparison.id, "verbosity") < self.verbosity_bias:
            return to_position(longer, order)
        truth = _majority(comparison)
        if self._uniform(comparison.id, "accuracy") < self.accuracy:
            return to_position(truth, order)
        others = [verdict for verdict in _ORDER if verdict != truth]
        return to_position(others[int(self._uniform(comparison.id, "alt") * 2)], order)


def _majority(comparison: Comparison) -> Verdict:
    counts = Counter(vote.verdict for vote in comparison.votes)
    return max(_ORDER, key=lambda verdict: (counts[verdict], -_ORDER.index(verdict)))
