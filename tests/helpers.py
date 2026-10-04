"""Shared builders for synthetic comparisons."""

import random

from judgecheck.schema import Comparison, Message


def conversation(answer: str) -> list[Message]:
    return [Message(role="user", content="question"), Message(role="assistant", content=answer)]


def comparison(
    comparison_id: str, answer_a: str, answer_b: str, votes: tuple[str, ...] = ("a",)
) -> Comparison:
    return Comparison.model_validate(
        {
            "id": comparison_id,
            "question_id": 81,
            "category": "writing",
            "turn": 1,
            "model_a": "alpha",
            "model_b": "beta",
            "conversation_a": [m.model_dump() for m in conversation(answer_a)],
            "conversation_b": [m.model_dump() for m in conversation(answer_b)],
            "votes": [
                {"judge": f"expert_{i}", "verdict": verdict, "shown_swapped": False}
                for i, verdict in enumerate(votes)
            ],
        }
    )


def synthetic(count: int, seed: int = 7) -> list[Comparison]:
    """Comparisons with random lengths and random expert votes."""
    rng = random.Random(seed)
    result = []
    for index in range(count):
        votes = tuple(rng.choice(("a", "b", "tie")) for _ in range(rng.randint(1, 3)))
        result.append(
            comparison(
                f"c{index:04d}",
                " ".join(["w"] * rng.randint(5, 80)),
                " ".join(["w"] * rng.randint(5, 80)),
                votes,
            )
        )
    return result
