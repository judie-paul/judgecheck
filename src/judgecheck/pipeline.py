"""End-to-end study: load data, run judges, score, write reports."""

import sys
from pathlib import Path

from pydantic import BaseModel

from judgecheck.config import Settings
from judgecheck.ingest import IngestError, load_hf, load_jsonl, load_sample
from judgecheck.judges import Judge, LLMJudge, build_judge
from judgecheck.normalize import normalize
from judgecheck.report import RunReport, human_report, judge_report, to_markdown
from judgecheck.runconfig import RunConfig
from judgecheck.runner import read_records, run_judge, select, write_records
from judgecheck.schema import Comparison


class BudgetError(RuntimeError):
    """A paid run was refused because it has no budget or would exceed it."""


class PlanRow(BaseModel):
    """What running one LLM judge would cost, before any call is made."""

    judge: str
    provider: str
    model: str
    paid: bool
    calls: int
    cached: int
    to_run: int
    approx_input_tokens: int
    max_output_tokens: int


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


def build_judges(config: RunConfig, settings: Settings) -> list[Judge]:
    """Build every configured judge. Fails early on a missing key or bad parameter."""
    return [
        build_judge(
            spec.name,
            spec.type,
            spec.params,
            seed=config.seed,
            min_length_ratio=config.min_length_ratio,
            cache_dir=settings.cache_dir,
        )
        for spec in config.judges
    ]


def plan_judges(judges: list[Judge], comparisons: list[Comparison]) -> list[PlanRow]:
    """Count calls, cache hits and approximate tokens for the LLM judges."""
    rows = []
    for judge in judges:
        if not isinstance(judge, LLMJudge):
            continue
        calls = cached = input_chars = 0
        for comparison in comparisons:
            for shown_first in ("a", "b"):
                request = judge.request(comparison, shown_first)
                calls += 1
                if judge.is_cached(request):
                    cached += 1
                else:
                    input_chars += len(request.system) + len(request.user)
        to_run = calls - cached
        rows.append(
            PlanRow(
                judge=judge.name,
                provider=judge.backend.provider,
                model=judge.backend.model,
                paid=judge.paid,
                calls=calls,
                cached=cached,
                to_run=to_run,
                approx_input_tokens=input_chars // 4,
                max_output_tokens=to_run * judge.max_output_tokens,
            )
        )
    return rows


def plan_study(config: RunConfig, settings: Settings | None = None) -> list[PlanRow]:
    """Estimate the work a run would do, without calling any model."""
    settings = settings or Settings()
    return plan_judges(build_judges(config, settings), load_comparisons(config, settings))


def check_budget(plan: list[PlanRow], max_paid_calls: int | None) -> None:
    """Refuse to start paid calls that have no budget or exceed it."""
    paid_calls = sum(row.to_run for row in plan if row.paid)
    if paid_calls == 0:
        return
    if max_paid_calls is None:
        raise BudgetError(
            f"{paid_calls} paid calls are needed. Review `judgecheck plan`, then set "
            "max_paid_calls in the config to authorize them."
        )
    if paid_calls > max_paid_calls:
        raise BudgetError(
            f"{paid_calls} paid calls are needed but max_paid_calls is {max_paid_calls}"
        )


def run_study(config: RunConfig, settings: Settings | None = None) -> RunReport:
    """Run every configured judge, store its records, then score and write reports."""
    settings = settings or Settings()
    comparisons = load_comparisons(config, settings)
    judges = build_judges(config, settings)
    check_budget(plan_judges(judges, comparisons), config.max_paid_calls)
    for judge in judges:
        write_records(
            run_judge(judge, comparisons, _progress(judge.name)), run_path(config, judge.name)
        )
    return _score(config, settings, comparisons)


def _progress(name: str):  # type: ignore[no-untyped-def]
    def report(done: int, total: int) -> None:
        if done == total or done % 10 == 0:
            print(f"{name}: {done}/{total}", file=sys.stderr, flush=True)

    return report


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
