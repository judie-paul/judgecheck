import pytest

from judgecheck import metrics
from judgecheck.metrics import estimate
from judgecheck.results import JudgeRecord, to_canonical, to_position
from judgecheck.schema import Comparison

from .helpers import comparison


def test_agreement_with_and_without_ties() -> None:
    units: list[metrics.Unit] = [("a", ("a", "b")), ("tie", ("tie",))]
    assert metrics.judge_agreement(True)(units) == pytest.approx(0.75)
    # S2 drops the tie verdict, then the tie vote of nothing: only the first unit remains.
    assert metrics.judge_agreement(False)(units) == pytest.approx(0.5)


def test_agreement_is_undefined_when_nothing_is_comparable() -> None:
    assert metrics.judge_agreement(True)([]) is None
    assert metrics.judge_agreement(False)([("a", ("tie",))]) is None
    assert metrics.judge_agreement(False)([("tie", ("a",))]) is None


def test_human_agreement_uses_distinct_pairs() -> None:
    votes = [("a", "a", "b"), ("a", "b")]
    # first comparison: pairs aa, ab, ab -> 1/3; second: ab -> 0.
    assert metrics.human_agreement(True)(votes) == pytest.approx(1 / 6)
    assert metrics.human_agreement(True)([("a",)]) is None


def test_human_agreement_drops_ties_for_s2() -> None:
    votes = [("a", "a", "tie")]
    assert metrics.human_agreement(True)(votes) == pytest.approx(1 / 3)
    assert metrics.human_agreement(False)(votes) == pytest.approx(1.0)


def test_kappa_matches_hand_calculation() -> None:
    units: list[metrics.Unit] = [("a", ("a",)), ("a", ("b",)), ("b", ("b",)), ("b", ("b",))]
    # observed 3/4; chance (2*1 + 2*3) / 16 = 1/2 -> (0.75 - 0.5) / 0.5
    assert metrics.kappa(units) == pytest.approx(0.5)


def test_kappa_perfect_and_undefined_cases() -> None:
    perfect: list[metrics.Unit] = [("a", ("a",)), ("b", ("b",)), ("tie", ("tie",))]
    assert metrics.kappa(perfect) == pytest.approx(1.0)
    assert metrics.kappa([]) is None
    assert metrics.kappa([("a", ("a",)), ("a", ("a",))]) is None  # chance agreement is 1


def test_position_rates() -> None:
    pairs: list[metrics.OrderPair] = [("a", "a"), ("a", "b"), ("b", "a"), ("a", "tie")]
    assert metrics.consistent_rate(pairs) == 0.25
    assert metrics.first_biased_rate(pairs) == 0.25
    assert metrics.second_biased_rate(pairs) == 0.25
    assert metrics.other_inconsistent_rate(pairs) == 0.25
    # picks of the first-displayed answer: 1 of 2, 2 of 2, 0 of 2, 1 of 1.
    assert metrics.first_position_rate(pairs) == pytest.approx(4 / 7)


def test_position_rates_are_undefined_without_data() -> None:
    assert metrics.consistent_rate([]) is None
    assert metrics.first_position_rate([("tie", "tie")]) is None


def test_length_side_needs_a_clear_difference() -> None:
    long_answer, short_answer = " ".join(["w"] * 30), " ".join(["w"] * 10)
    assert metrics.length_side(comparison("c", long_answer, short_answer), 1.2) == "a"
    assert metrics.length_side(comparison("c", short_answer, long_answer), 1.2) == "b"
    assert metrics.length_side(comparison("c", long_answer, " ".join(["w"] * 28)), 1.2) is None
    assert metrics.length_side(comparison("c", "", long_answer), 1.2) is None


def test_longer_rates_and_gap() -> None:
    units: list[metrics.LongerUnit] = [((True, True), (True, False)), ((False,), (False,))]
    assert metrics.judge_longer_rate(units) == pytest.approx(2 / 3)
    assert metrics.human_longer_rate(units) == pytest.approx(1 / 3)
    assert metrics.longer_gap(units) == pytest.approx(1 / 3)
    assert metrics.longer_gap([((), (True,))]) is None


def test_bootstrap_is_seeded_and_brackets_the_estimate() -> None:
    units = [float(i % 2) for i in range(60)]

    def mean(values: list[float] | tuple[float, ...]) -> float | None:
        return sum(values) / len(values) if values else None

    first = estimate(units, mean, 200, seed=3)
    assert first == estimate(units, mean, 200, seed=3)
    assert first != estimate(units, mean, 200, seed=4)
    assert first.value == 0.5 and first.n == 60
    assert first.low is not None and first.high is not None
    assert first.low < 0.5 < first.high


def test_estimate_edge_cases() -> None:
    def mean(values: list[float] | tuple[float, ...]) -> float | None:
        return sum(values) / len(values) if values else None

    empty = estimate([], mean, 100, seed=1)
    assert (empty.value, empty.low, empty.high, empty.n) == (None, None, None, 0)
    no_interval = estimate([1.0, 2.0], mean, 0, seed=1)
    assert no_interval.value == 1.5 and no_interval.low is None


def test_display_positions_round_trip() -> None:
    for order in ("ab", "ba"):
        for verdict in ("a", "b", "tie"):
            assert to_canonical(to_position(verdict, order), order) == verdict  # type: ignore[arg-type]
    assert to_canonical("first", "ba") == "b"
    assert to_canonical("second", "ba") == "a"


def test_consensus_requires_both_orders_and_ties_on_disagreement() -> None:
    def record(ab: str | None, ba: str | None) -> JudgeRecord:
        return JudgeRecord(comparison_id="c", ab=ab, ba=ba)  # type: ignore[arg-type]

    assert record("a", "a").consensus == "a"
    assert record("a", "b").consensus == "tie"
    assert record("a", None).consensus is None
    assert isinstance(comparison("c", "x", "y"), Comparison)
