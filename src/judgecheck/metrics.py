"""Pure agreement and bias statistics with seeded bootstrap intervals.

Every statistic returns ``None`` when its denominator is empty. A missing value means
"undefined", never zero. Resampling is by comparison, because votes on one comparison
are not independent.
"""

import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from itertools import combinations
from typing import TypeVar

from pydantic import BaseModel, ConfigDict

from judgecheck.schema import Comparison, Verdict

T = TypeVar("T")
Stat = Callable[[Sequence[T]], float | None]

Unit = tuple[Verdict, tuple[Verdict, ...]]
"""A judge verdict and every expert vote on the same comparison."""


class Estimate(BaseModel):
    """A statistic with a percentile bootstrap interval over ``n`` comparisons."""

    model_config = ConfigDict(frozen=True)

    value: float | None
    low: float | None
    high: float | None
    n: int


def estimate(units: Sequence[T], stat: Stat[T], iterations: int, seed: int) -> Estimate:
    """Evaluate ``stat`` on ``units`` and bootstrap a 95% interval by resampling units."""
    value = stat(units)
    if value is None or iterations <= 0:
        return Estimate(value=value, low=None, high=None, n=len(units))
    rng = random.Random(seed)
    count = len(units)
    draws = []
    for _ in range(iterations):
        draw = stat([units[rng.randrange(count)] for _ in range(count)])
        if draw is not None:
            draws.append(draw)
    if not draws:
        return Estimate(value=value, low=None, high=None, n=count)
    draws.sort()
    low = draws[math.floor(0.025 * (len(draws) - 1))]
    high = draws[math.ceil(0.975 * (len(draws) - 1))]
    return Estimate(value=value, low=low, high=high, n=count)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


# --- agreement -------------------------------------------------------------------------


def judge_agreement(include_ties: bool) -> Stat[Unit]:
    """Judge-to-expert agreement: for each comparison, the share of expert votes matching.

    Averaged over comparisons. With ``include_ties=False`` (MT-Bench's "S2") tie votes
    and tie verdicts are dropped, and comparisons left without a usable pair are skipped.
    """

    def stat(units: Sequence[Unit]) -> float | None:
        scores = []
        for verdict, votes in units:
            if not include_ties:
                if verdict == "tie":
                    continue
                votes = tuple(vote for vote in votes if vote != "tie")
            if votes:
                scores.append(sum(vote == verdict for vote in votes) / len(votes))
        return _mean(scores)

    return stat


def human_agreement(include_ties: bool) -> Stat[tuple[Verdict, ...]]:
    """Expert-to-expert agreement over distinct vote pairs on each comparison."""

    def stat(units: Sequence[tuple[Verdict, ...]]) -> float | None:
        scores = []
        for votes in units:
            if not include_ties:
                votes = tuple(vote for vote in votes if vote != "tie")
            pairs = list(combinations(votes, 2))
            if pairs:
                scores.append(sum(left == right for left, right in pairs) / len(pairs))
        return _mean(scores)

    return stat


def kappa(units: Sequence[Unit]) -> float | None:
    """Cohen's kappa between judge verdicts and expert votes over a, b and tie.

    Pools every (verdict, vote) pair, so comparisons with more votes weigh more.
    Undefined when there are no pairs or chance agreement is already perfect.
    """
    pairs = [(verdict, vote) for verdict, votes in units for vote in votes]
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(left == right for left, right in pairs) / total
    judge_counts = Counter(left for left, _ in pairs)
    human_counts = Counter(right for _, right in pairs)
    chance = sum(judge_counts[label] * human_counts[label] for label in judge_counts) / total**2
    if chance >= 1.0:
        return None
    return (observed - chance) / (1.0 - chance)


# --- position bias ---------------------------------------------------------------------

OrderPair = tuple[Verdict, Verdict]
"""Canonical verdicts for the ``ab`` and ``ba`` display orders of one comparison."""


def _rate(predicate: Callable[[OrderPair], bool]) -> Stat[OrderPair]:
    def stat(units: Sequence[OrderPair]) -> float | None:
        return sum(map(predicate, units)) / len(units) if units else None

    return stat


consistent_rate = _rate(lambda pair: pair[0] == pair[1])
"""Share of comparisons where both display orders give the same verdict."""

first_biased_rate = _rate(lambda pair: pair[0] == "a" and pair[1] == "b")
"""Share where the judge picked whichever answer was shown first, in both orders."""

second_biased_rate = _rate(lambda pair: pair[0] == "b" and pair[1] == "a")
"""Share where the judge picked whichever answer was shown second, in both orders."""

other_inconsistent_rate = _rate(
    lambda pair: pair[0] != pair[1] and {pair[0], pair[1]} != {"a", "b"}
)
"""Share that disagree across orders because one order was a tie."""


def first_position_rate(units: Sequence[OrderPair]) -> float | None:
    """Among decisive verdicts, the share that picked the answer displayed first (0.5 is fair)."""
    first = decisive = 0
    for ab, ba in units:
        if ab != "tie":
            decisive += 1
            first += ab == "a"
        if ba != "tie":
            decisive += 1
            first += ba == "b"
    return first / decisive if decisive else None


# --- verbosity bias --------------------------------------------------------------------


def length_side(comparison: Comparison, min_ratio: float) -> Verdict | None:
    """The model with the clearly longer answer in words, or ``None`` if similar in length."""
    words_a, words_b = len(comparison.answer_a.split()), len(comparison.answer_b.split())
    if min(words_a, words_b) == 0:
        return None
    if words_a >= words_b * min_ratio:
        return "a"
    if words_b >= words_a * min_ratio:
        return "b"
    return None


LongerUnit = tuple[tuple[bool, ...], tuple[bool, ...]]
"""Per comparison with a clearly longer answer: did each decisive judge verdict / expert
vote pick the longer answer?"""


def judge_longer_rate(units: Sequence[LongerUnit]) -> float | None:
    """Share of decisive judge verdicts that picked the longer answer."""
    picks = [pick for judge, _ in units for pick in judge]
    return _mean([float(pick) for pick in picks])


def human_longer_rate(units: Sequence[LongerUnit]) -> float | None:
    """Share of decisive expert votes that picked the longer answer."""
    picks = [pick for _, human in units for pick in human]
    return _mean([float(pick) for pick in picks])


def longer_gap(units: Sequence[LongerUnit]) -> float | None:
    """Judge longer-pick rate minus the experts' rate on the same comparisons."""
    judge, human = judge_longer_rate(units), human_longer_rate(units)
    return None if judge is None or human is None else judge - human
