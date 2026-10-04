"""Group raw expert rows into canonical comparisons."""

from collections import defaultdict
from collections.abc import Iterable

from judgecheck.schema import (
    Comparison,
    ExpertVote,
    Message,
    RawJudgment,
    Verdict,
    category_for,
)

_FLIP: dict[Verdict, Verdict] = {"a": "b", "b": "a", "tie": "tie"}


def flip(verdict: Verdict) -> Verdict:
    """Swap a verdict between the two positions."""
    return _FLIP[verdict]


def comparison_id(question_id: int, turn: int, model_a: str, model_b: str) -> str:
    """Stable identifier for a canonically ordered comparison."""
    first, second = sorted((model_a, model_b))
    return f"q{question_id}-t{turn}-{first}-vs-{second}"


def normalize(rows: Iterable[RawJudgment]) -> list[Comparison]:
    """Return one comparison per (question, model pair, turn), sorted by id.

    Rows that present the same pair in the opposite order are merged, and their
    votes are flipped into canonical order. Conversations must agree across rows.
    """
    grouped: dict[str, list[RawJudgment]] = defaultdict(list)
    for row in rows:
        if row.model_a == row.model_b:
            raise ValueError(f"question {row.question_id} compares {row.model_a} with itself")
        grouped[comparison_id(row.question_id, row.turn, row.model_a, row.model_b)].append(row)
    return [_build(key, group) for key, group in sorted(grouped.items())]


def _build(key: str, rows: list[RawJudgment]) -> Comparison:
    first = rows[0]
    swapped = first.model_a > first.model_b
    conv_a, conv_b = (
        (first.conversation_b, first.conversation_a)
        if swapped
        else (first.conversation_a, first.conversation_b)
    )
    conv_a, conv_b = _truncate(conv_a, first.turn), _truncate(conv_b, first.turn)
    votes = []
    for row in rows:
        row_swapped = row.model_a > row.model_b
        row_a, row_b = (
            (row.conversation_b, row.conversation_a)
            if row_swapped
            else (row.conversation_a, row.conversation_b)
        )
        if _truncate(row_a, row.turn) != conv_a or _truncate(row_b, row.turn) != conv_b:
            raise ValueError(f"{key}: rows disagree on the conversation text")
        verdict: Verdict = {"model_a": "a", "model_b": "b", "tie": "tie"}[row.winner]  # type: ignore[assignment]
        votes.append(
            ExpertVote(
                judge=row.judge,
                verdict=flip(verdict) if row_swapped else verdict,
                shown_swapped=row_swapped,
            )
        )
    model_a, model_b = sorted((first.model_a, first.model_b))
    return Comparison(
        id=key,
        question_id=first.question_id,
        category=category_for(first.question_id),
        turn=first.turn,
        model_a=model_a,
        model_b=model_b,
        conversation_a=conv_a,
        conversation_b=conv_b,
        votes=sorted(votes, key=lambda vote: (vote.judge, vote.verdict, vote.shown_swapped)),
    )


def _truncate(conversation: list[Message], turn: int) -> list[Message]:
    """Keep the messages up to and including the judged turn's answer."""
    if len(conversation) < 2 * turn:
        raise ValueError(f"conversation is too short for turn {turn}")
    return list(conversation[: 2 * turn])
