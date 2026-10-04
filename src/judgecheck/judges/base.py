"""Judge protocol."""

from typing import Literal, Protocol

from judgecheck.results import Position
from judgecheck.schema import Comparison


class Judge(Protocol):
    """Picks the better of two answers, or declares a tie."""

    name: str

    def judge(self, comparison: Comparison, shown_first: Literal["a", "b"]) -> Position | None:
        """Return the verdict by display position, or ``None`` if the response was unusable.

        ``shown_first`` names the model whose answer is displayed first.
        """
        ...
