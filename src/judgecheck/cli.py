"""Command-line interface."""

import subprocess
import sys
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from judgecheck.config import Settings
from judgecheck.ingest import IngestError, load_hf, load_jsonl, load_sample, write_comparisons
from judgecheck.llm.base import ConfigError
from judgecheck.normalize import normalize
from judgecheck.pipeline import BudgetError, plan_study, run_study, score_study
from judgecheck.runconfig import load_config

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


@app.command(name="app")
def explorer(
    address: Annotated[str, typer.Option(help="Interface to listen on")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on")] = 8501,
) -> None:
    """Open the results explorer (needs the app extra). Reads JUDGECHECK_REPORTS."""
    path = Path(__file__).with_name("app.py")
    command = [
        sys.executable, "-m", "streamlit", "run", str(path),
        f"--server.address={address}", f"--server.port={port}",
        "--server.headless=true", "--browser.gatherUsageStats=false",
    ]  # fmt: skip
    raise typer.Exit(subprocess.run(command, check=False).returncode)


@app.command()
def plan(
    config: Annotated[Path, typer.Option(help="YAML run configuration")] = Path(
        "configs/default.yaml"
    ),
) -> None:
    """Show calls, cache hits and approximate tokens for LLM judges. Calls no model."""
    try:
        rows = plan_study(load_config(config))
    except (IngestError, ConfigError, ValueError, OSError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from error
    if not rows:
        typer.echo("no LLM judges configured; nothing to spend")
    for row in rows:
        kind = "PAID" if row.paid else "free"
        typer.echo(
            f"{row.judge} ({row.provider}/{row.model}, {kind}): {row.calls} calls, "
            f"{row.cached} cached, {row.to_run} to run, ~{row.approx_input_tokens} input tokens, "
            f"up to {row.max_output_tokens} output tokens"
        )


@app.command()
def run(
    config: Annotated[Path, typer.Option(help="YAML run configuration")] = Path(
        "configs/default.yaml"
    ),
) -> None:
    """Run every configured judge, then write JSON and Markdown reports."""
    try:
        report = run_study(load_config(config))
    except (IngestError, ConfigError, BudgetError, ValueError, OSError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"scored {len(report.judges)} judges on {report.comparisons} comparisons")


@app.command()
def report(
    config: Annotated[Path, typer.Option(help="YAML run configuration")] = Path(
        "configs/default.yaml"
    ),
) -> None:
    """Re-score stored judge runs without calling any judge."""
    try:
        study = score_study(load_config(config))
    except (IngestError, ValueError, OSError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"rescored {len(study.judges)} judges on {study.comparisons} comparisons")


if __name__ == "__main__":
    app()
