"""Judge implementations and their construction from configuration."""

from collections.abc import Mapping
from typing import Any

from judgecheck.judges.base import Judge
from judgecheck.judges.mock import MockJudge

__all__ = ["Judge", "MockJudge", "build_judge"]


def build_judge(
    name: str, kind: str, params: Mapping[str, Any], *, seed: int, min_length_ratio: float
) -> Judge:
    """Construct the judge named in a run configuration."""
    if kind == "mock":
        return MockJudge(name, seed=seed, min_length_ratio=min_length_ratio, **params)
    raise ValueError(f"unknown judge type {kind!r}")
