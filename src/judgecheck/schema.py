"""Validated records for MT-Bench pairwise comparisons and expert votes."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Verdict = Literal["a", "b", "tie"]
"""A verdict in canonical order: ``a`` is the alphabetically first model."""

RawWinner = Literal["model_a", "model_b", "tie"]

# MT-Bench assigns ten consecutive question ids to each category, starting at 81.
CATEGORIES = (
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
)


def category_for(question_id: int) -> str:
    """Return the MT-Bench category for a question id in 81-160."""
    index = (question_id - 81) // 10
    if question_id < 81 or index >= len(CATEGORIES):
        raise ValueError(f"question_id {question_id} is outside MT-Bench's 81-160 range")
    return CATEGORIES[index]


class Message(BaseModel):
    """One conversation message."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class RawJudgment(BaseModel):
    """One row of ``lmsys/mt_bench_human_judgments`` exactly as published."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: int
    model_a: str
    model_b: str
    winner: RawWinner
    judge: str
    conversation_a: list[Message]
    conversation_b: list[Message]
    turn: Literal[1, 2]


class ExpertVote(BaseModel):
    """A human verdict, stored in canonical order with the order the expert saw."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    judge: str
    verdict: Verdict
    shown_swapped: bool = Field(description="True when the expert saw model b first")


class Comparison(BaseModel):
    """One judged item: a question turn answered by two models, with every expert vote.

    Conversations are truncated to the judged turn, so a turn-1 comparison never
    exposes the second question to a judge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question_id: int
    category: str
    turn: Literal[1, 2]
    model_a: str
    model_b: str
    conversation_a: list[Message]
    conversation_b: list[Message]
    votes: list[ExpertVote] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> "Comparison":
        if self.model_a >= self.model_b:
            raise ValueError("models must be distinct and in canonical (sorted) order")
        if self.category != category_for(self.question_id):
            raise ValueError("category does not match question_id")
        for conversation in (self.conversation_a, self.conversation_b):
            roles = [message.role for message in conversation]
            if roles != ["user", "assistant"] * self.turn:
                raise ValueError("conversation must alternate user/assistant up to the turn")
        if questions(self.conversation_a) != questions(self.conversation_b):
            raise ValueError("both models must answer the same user messages")
        return self

    @property
    def answer_a(self) -> str:
        """Model a's answer for the judged turn."""
        return self.conversation_a[-1].content

    @property
    def answer_b(self) -> str:
        """Model b's answer for the judged turn."""
        return self.conversation_b[-1].content


def questions(conversation: list[Message]) -> list[str]:
    """Return the user messages of a conversation."""
    return [message.content for message in conversation if message.role == "user"]
