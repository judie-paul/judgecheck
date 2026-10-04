"""Run judges over comparisons in both display orders."""

import json
import random
from collections.abc import Sequence
from pathlib import Path

from judgecheck.judges.base import Judge
from judgecheck.results import JudgeRecord, to_canonical
from judgecheck.schema import Comparison


def select(comparisons: Sequence[Comparison], limit: int | None, seed: int) -> list[Comparison]:
    """Return all comparisons, or a seeded random subset of ``limit``, sorted by id."""
    ordered = sorted(comparisons, key=lambda comparison: comparison.id)
    if limit is None or limit >= len(ordered):
        return ordered
    if limit < 1:
        raise ValueError("limit must be at least 1")
    return sorted(random.Random(seed).sample(ordered, limit), key=lambda c: c.id)


def run_judge(judge: Judge, comparisons: Sequence[Comparison]) -> list[JudgeRecord]:
    """Judge every comparison with model a first, then with model b first."""
    records = []
    for comparison in comparisons:
        first_ab = judge.judge(comparison, "a")
        first_ba = judge.judge(comparison, "b")
        records.append(
            JudgeRecord(
                comparison_id=comparison.id,
                ab=None if first_ab is None else to_canonical(first_ab, "ab"),
                ba=None if first_ba is None else to_canonical(first_ba, "ba"),
            )
        )
    return records


def write_records(records: Sequence[JudgeRecord], path: Path) -> None:
    """Write judge records as sorted-key JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")


def read_records(path: Path) -> list[JudgeRecord]:
    """Read judge records written by :func:`write_records`."""
    with path.open(encoding="utf-8") as handle:
        return [JudgeRecord.model_validate_json(line) for line in handle if line.strip()]
