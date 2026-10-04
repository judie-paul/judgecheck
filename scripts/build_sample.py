"""Rebuild the bundled sample from the pinned dataset revision.

Selects, for each MT-Bench category and turn, the comparison with the most
expert votes (ties broken by id), keeping every vote for those comparisons, so
the sample covers all categories, both turns, reversed presentation order and
multi-expert comparisons. Requires the ``hf`` extra and network access.

    python scripts/build_sample.py
"""

import json
from collections import defaultdict
from pathlib import Path

from judgecheck.config import Settings
from judgecheck.ingest import SAMPLE_RESOURCE, load_hf
from judgecheck.normalize import comparison_id
from judgecheck.schema import RawJudgment, category_for

PER_CATEGORY_TURN = 1
OUT = Path(__file__).resolve().parents[1] / "src" / "judgecheck" / "resources" / SAMPLE_RESOURCE


def main() -> None:
    rows = load_hf(Settings())
    groups: dict[str, list[RawJudgment]] = defaultdict(list)
    for row in rows:
        groups[comparison_id(row.question_id, row.turn, row.model_a, row.model_b)].append(row)
    buckets: dict[tuple[str, int], list[str]] = defaultdict(list)
    for key, group in groups.items():
        buckets[(category_for(group[0].question_id), group[0].turn)].append(key)
    chosen = sorted(
        key
        for keys in buckets.values()
        for key in sorted(keys, key=lambda k: (-len(groups[k]), k))[:PER_CATEGORY_TURN]
    )
    with OUT.open("w", encoding="utf-8") as handle:
        for key in chosen:
            for row in sorted(groups[key], key=lambda r: (r.judge, r.model_a)):
                handle.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")
    votes = sum(len(groups[key]) for key in chosen)
    print(f"{len(chosen)} comparisons, {votes} votes -> {OUT}")


if __name__ == "__main__":
    main()
