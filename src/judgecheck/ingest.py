"""Load MT-Bench expert judgments from the bundled sample, a JSONL file or Hugging Face."""

import json
from collections.abc import Iterable, Iterator
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from judgecheck.config import DATASET_ID, Settings
from judgecheck.schema import Comparison, RawJudgment

SAMPLE_RESOURCE = "mt_bench_human_sample.jsonl"
HF_PARQUET = "data/human-00000-of-00001-25f4910818759289.parquet"


class IngestError(ValueError):
    """Input that cannot be parsed, reported with its location."""


def parse_raw_lines(lines: Iterable[str], source: str) -> list[RawJudgment]:
    """Validate raw dataset rows, one JSON object per non-blank line."""
    rows = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(RawJudgment.model_validate_json(line))
        except ValidationError as error:
            raise IngestError(f"{source}:{number}: {_first_error(error)}") from error
    return rows


def load_sample() -> list[RawJudgment]:
    """Return the bundled, attributed sample of real expert judgments."""
    text = resources.files("judgecheck.resources").joinpath(SAMPLE_RESOURCE).read_text("utf-8")
    return parse_raw_lines(text.splitlines(), SAMPLE_RESOURCE)


def load_jsonl(path: Path) -> list[RawJudgment]:
    """Load raw rows from a local JSONL export of the dataset."""
    with path.open(encoding="utf-8") as handle:
        return parse_raw_lines(handle, str(path))


def load_hf(settings: Settings) -> list[RawJudgment]:
    """Download the pinned ``human`` split. Requires the ``hf`` extra."""
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise IngestError("Install the hf extra: pip install 'judgecheck[hf]'") from error
    path = hf_hub_download(
        DATASET_ID,
        HF_PARQUET,
        repo_type="dataset",
        revision=settings.dataset_revision,
        cache_dir=settings.cache_dir / "hf",
    )
    records = pd.read_parquet(path).to_dict(orient="records")
    return parse_raw_lines((json.dumps(_plain(record)) for record in records), DATASET_ID)


def write_comparisons(comparisons: Iterable[Comparison], path: Path) -> int:
    """Write comparisons as sorted-key JSONL and return the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for comparison in comparisons:
            handle.write(json.dumps(comparison.model_dump(mode="json"), sort_keys=True) + "\n")
            count += 1
    return count


def read_comparisons(path: Path) -> list[Comparison]:
    """Read normalized comparisons written by :func:`write_comparisons`."""
    return list(_iter_comparisons(path))


def _iter_comparisons(path: Path) -> Iterator[Comparison]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield Comparison.model_validate_json(line)
            except ValidationError as error:
                raise IngestError(f"{path}:{number}: {_first_error(error)}") from error


def _plain(value: Any) -> Any:
    """Convert numpy containers and scalars from pandas into JSON-native values."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def _first_error(error: ValidationError) -> str:
    detail = error.errors()[0]
    location = ".".join(str(part) for part in detail["loc"]) or "record"
    return f"{location}: {detail['msg']}"
