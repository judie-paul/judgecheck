"""Stored judge verdicts, kept in canonical order so runs can be re-scored without re-judging."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from judgecheck.schema import Verdict

Position = Literal["first", "second", "tie"]
"""A verdict as the judge saw it: relative to the order the answers were displayed."""

Order = Literal["ab", "ba"]
"""``ab`` shows model a first; ``ba`` shows model b first."""


def to_canonical(position: Position, order: Order) -> Verdict:
    """Translate a displayed-position verdict into canonical model order."""
    if position == "tie":
        return "tie"
    shown_first: Verdict = "a" if order == "ab" else "b"
    shown_second: Verdict = "b" if order == "ab" else "a"
    return shown_first if position == "first" else shown_second


def to_position(verdict: Verdict, order: Order) -> Position:
    """Translate a canonical verdict into the position it occupied when displayed."""
    if verdict == "tie":
        return "tie"
    first = "a" if order == "ab" else "b"
    return "first" if verdict == first else "second"


class JudgeRecord(BaseModel):
    """One judge's verdicts for one comparison; ``None`` marks an unparseable response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    comparison_id: str
    ab: Verdict | None
    ba: Verdict | None

    @property
    def consensus(self) -> Verdict | None:
        """The shared verdict when both orders agree, a tie when they disagree.

        Undefined (``None``) unless both orders produced a verdict.
        """
        if self.ab is None or self.ba is None:
            return None
        return self.ab if self.ab == self.ba else "tie"
