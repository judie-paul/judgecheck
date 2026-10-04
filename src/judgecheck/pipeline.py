"""End-to-end study: load data, run judges, score, write reports."""

from pathlib import Path

from judgecheck.config import Settings
from judgecheck.ingest import IngestError, load_hf, load_jsonl, load_sample
from judgecheck.judges import build_judge
from judgecheck.normalize import normalize
from judgecheck.report import RunReport, human_report, judge_report, to_markdown
from judgecheck.runconfig import RunConfig
from judgecheck.runner import read_records, run_judge, select, write_records
from judgecheck.schema import Comparison


def load_comparisons(config: RunConfig, settings: Settings) -> list[Comparison]:
    """Load, normalize and select the comparisons named in the configuration."""
    if config.source == "hf":
        rows = load_hf(settings)
    elif config.source == "jsonl":
        if config.path is None:
            raise IngestError("path is required when source is jsonl")
        rows = load_jsonl(config.path)
    else:
        rows = load_sample()
    return select(normalize(rows), config.limit, config.seed)


def run_path(config: RunConfig, name: str) -> Path:
    """Where a judge's stored records live."""
    return config.out_dir / "runs" / f"{name}.jsonl"


def run_study(config: RunConfig, settings: Settings | None = None) -> RunReport:
    """Run every configured judge, store its records, then score and write reports."""
    settings = settings or Settings()
    comparisons = load_comparisons(config, settings)
    for spec in config.judges:
        judge = build_judge(
            spec.name,
            spec.type,
            spec.params,
            seed=config.seed,
            min_length_ratio=config.min_length_ratio,
        )
        write_records(run_judge(judge, comparisons), run_path(config, spec.name))
    return _score(config, settings, comparisons)


def score_study(config: RunConfig, settings: Settings | None = None) -> RunReport:
    """Score previously stored runs and rewrite the reports."""
    settings = settings or Settings()
    return _score(config, settings, load_comparisons(config, settings))


def _score(config: RunConfig, settings: Settings, comparisons: list[Comparison]) -> RunReport:
    judges = []
    for spec in config.judges:
        path = run_path(config, spec.name)
        if not path.exists():
            raise IngestError(f"no stored run for {spec.name!r}; run `judgecheck run` first")
        records = {record.comparison_id: record for record in read_records(path)}
        judges.append(
            judge_report(
                spec.name,
                comparisons,
                records,
                iterations=config.bootstrap,
                seed=config.seed,
                min_length_ratio=config.min_length_ratio,
            )
        )
    report = RunReport(
        dataset_revision=settings.dataset_revision,
        seed=config.seed,
        bootstrap_iterations=config.bootstrap,
        min_length_ratio=config.min_length_ratio,
        comparisons=len(comparisons),
        human=human_report(comparisons, config.bootstrap, config.seed),
        judges=judges,
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "results.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (config.out_dir / "results.md").write_text(to_markdown(report), encoding="utf-8")
    return report
