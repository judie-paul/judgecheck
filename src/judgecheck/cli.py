"""Command-line interface."""

from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from judgecheck.config import Settings
from judgecheck.ingest import IngestError, load_hf, load_jsonl, load_sample, write_comparisons
from judgecheck.normalize import normalize

app = typer.Typer(help="Measure how far LLM judges agree with MT-Bench experts.")


class Source(StrEnum):
    sample = "sample"
    jsonl = "jsonl"
    hf = "hf"


@app.callback()
def main() -> None:
    """JudgeCheck commands."""


@app.command()
def ingest(
    source: Annotated[Source, typer.Option(help="sample, jsonl or hf")] = Source.sample,
    path: Annotated[Path | None, typer.Option(help="Raw JSONL export, for --source jsonl")] = None,
    out: Annotated[Path, typer.Option(help="Normalized output")] = Path("data/comparisons.jsonl"),
) -> None:
    """Normalize expert judgments into one record per comparison."""
    try:
        if source is Source.jsonl:
            if path is None:
                raise IngestError("--path is required for --source jsonl")
            rows = load_jsonl(path)
        elif source is Source.hf:
            rows = load_hf(Settings())
        else:
            rows = load_sample()
        comparisons = normalize(rows)
    except (IngestError, ValueError, OSError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from error
    write_comparisons(comparisons, out)
    vote_counts = Counter(len(comparison.votes) for comparison in comparisons)
    multi = sum(count for votes, count in vote_counts.items() if votes > 1)
    typer.echo(
        f"{len(rows)} expert votes -> {len(comparisons)} comparisons "
        f"({multi} with more than one vote) written to {out}"
    )


if __name__ == "__main__":
    app()
