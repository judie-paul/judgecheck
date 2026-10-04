"""Score stored judge records against expert votes."""

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel

from judgecheck import metrics
from judgecheck.metrics import Estimate, estimate
from judgecheck.results import JudgeRecord
from judgecheck.schema import Comparison, Verdict

VariantName = Literal["order_ab", "order_ba", "consensus"]
Selector = Callable[[JudgeRecord], Verdict | None]


class VariantMetrics(BaseModel):
    """Agreement with experts for one way of reading the judge's verdicts."""

    agreement_s1: Estimate
    agreement_s2: Estimate
    kappa: Estimate


class PositionMetrics(BaseModel):
    """Order sensitivity over comparisons where both display orders gave a verdict."""

    consistent: Estimate
    first_biased: Estimate
    second_biased: Estimate
    other_inconsistent: Estimate
    first_position_rate: Estimate


class VerbosityMetrics(BaseModel):
    """Longer-answer preference on comparisons whose answers differ clearly in length."""

    judge_longer_rate: Estimate
    human_longer_rate: Estimate
    gap: Estimate


class JudgeReport(BaseModel):
    """Everything measured for one judge."""

    judge: str
    comparisons: int
    unparseable_rate: float | None  # share of calls that failed, were refused or had no verdict
    variants: dict[VariantName, VariantMetrics]
    position: PositionMetrics
    verbosity: VerbosityMetrics


class HumanReport(BaseModel):
    """Expert-to-expert agreement: the ceiling a judge can reasonably be held to."""

    comparisons_with_multiple_votes: int
    agreement_s1: Estimate
    agreement_s2: Estimate


class RunReport(BaseModel):
    """A complete study result."""

    dataset_revision: str
    seed: int
    bootstrap_iterations: int
    min_length_ratio: float
    comparisons: int
    human: HumanReport
    judges: list[JudgeReport]


def human_report(comparisons: Sequence[Comparison], iterations: int, seed: int) -> HumanReport:
    """Measure agreement among experts."""
    votes = [tuple(vote.verdict for vote in comparison.votes) for comparison in comparisons]
    multi = [group for group in votes if len(group) > 1]
    return HumanReport(
        comparisons_with_multiple_votes=len(multi),
        agreement_s1=estimate(multi, metrics.human_agreement(True), iterations, seed),
        agreement_s2=estimate(multi, metrics.human_agreement(False), iterations, seed),
    )


def judge_report(
    name: str,
    comparisons: Sequence[Comparison],
    records: Mapping[str, JudgeRecord],
    *,
    iterations: int,
    seed: int,
    min_length_ratio: float,
) -> JudgeReport:
    """Measure one judge. Comparisons without a record are ignored."""
    judged = [c for c in comparisons if c.id in records]
    calls = 2 * len(judged)
    usable = sum((records[c.id].ab is not None) + (records[c.id].ba is not None) for c in judged)
    selectors: dict[VariantName, Selector] = {
        "order_ab": lambda record: record.ab,
        "order_ba": lambda record: record.ba,
        "consensus": lambda record: record.consensus,
    }
    variants: dict[VariantName, VariantMetrics] = {}
    for variant, select in selectors.items():
        units: list[metrics.Unit] = []
        for comparison in judged:
            verdict = select(records[comparison.id])
            if verdict is not None:
                units.append((verdict, tuple(vote.verdict for vote in comparison.votes)))
        variants[variant] = VariantMetrics(
            agreement_s1=estimate(units, metrics.judge_agreement(True), iterations, seed),
            agreement_s2=estimate(units, metrics.judge_agreement(False), iterations, seed),
            kappa=estimate(units, metrics.kappa, iterations, seed),
        )

    pairs: list[metrics.OrderPair] = []
    for comparison in judged:
        record = records[comparison.id]
        if record.ab is not None and record.ba is not None:
            pairs.append((record.ab, record.ba))
    position = PositionMetrics(
        consistent=estimate(pairs, metrics.consistent_rate, iterations, seed),
        first_biased=estimate(pairs, metrics.first_biased_rate, iterations, seed),
        second_biased=estimate(pairs, metrics.second_biased_rate, iterations, seed),
        other_inconsistent=estimate(pairs, metrics.other_inconsistent_rate, iterations, seed),
        first_position_rate=estimate(pairs, metrics.first_position_rate, iterations, seed),
    )

    longer_units: list[metrics.LongerUnit] = []
    for comparison in judged:
        side = metrics.length_side(comparison, min_length_ratio)
        if side is None:
            continue
        record = records[comparison.id]
        judge_picks = tuple(v == side for v in (record.ab, record.ba) if v in ("a", "b"))
        human_picks = tuple(v.verdict == side for v in comparison.votes if v.verdict != "tie")
        longer_units.append((judge_picks, human_picks))
    verbosity = VerbosityMetrics(
        judge_longer_rate=estimate(longer_units, metrics.judge_longer_rate, iterations, seed),
        human_longer_rate=estimate(longer_units, metrics.human_longer_rate, iterations, seed),
        gap=estimate(longer_units, metrics.longer_gap, iterations, seed),
    )
    return JudgeReport(
        judge=name,
        comparisons=len(judged),
        unparseable_rate=(calls - usable) / calls if calls else None,
        variants=variants,
        position=position,
        verbosity=verbosity,
    )


def _fmt(value: Estimate, digits: int = 3) -> str:
    if value.value is None:
        return "undefined"
    text = f"{value.value:.{digits}f}"
    if value.low is not None and value.high is not None:
        text += f" [{value.low:.{digits}f}, {value.high:.{digits}f}]"
    return text


def to_markdown(report: RunReport) -> str:
    """Render a report as Markdown tables."""
    human = report.human
    lines = [
        "# JudgeCheck results",
        "",
        f"Dataset revision `{report.dataset_revision}`; {report.comparisons} comparisons; "
        f"seed {report.seed}; {report.bootstrap_iterations} bootstrap resamples "
        "(95% percentile intervals, resampled by comparison).",
        "",
        "## Expert agreement ceiling",
        "",
        f"On {human.comparisons_with_multiple_votes} comparisons with more than one expert vote: "
        f"S1 (ties counted) {_fmt(human.agreement_s1)}; S2 (ties dropped) "
        f"{_fmt(human.agreement_s2)}.",
        "",
        "## Agreement with experts (consensus of both display orders)",
        "",
        "| Judge | S1 | S2 | Cohen's kappa | Unusable responses |",
        "| --- | --- | --- | --- | --- |",
    ]
    for judge in report.judges:
        consensus = judge.variants["consensus"]
        unparseable = "n/a" if judge.unparseable_rate is None else f"{judge.unparseable_rate:.1%}"
        lines.append(
            f"| {judge.judge} | {_fmt(consensus.agreement_s1)} | {_fmt(consensus.agreement_s2)} "
            f"| {_fmt(consensus.kappa)} | {unparseable} |"
        )
    lines += [
        "",
        "## Position bias",
        "",
        "| Judge | Consistent | Always first | Always second | First-position pick rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for judge in report.judges:
        p = judge.position
        lines.append(
            f"| {judge.judge} | {_fmt(p.consistent)} | {_fmt(p.first_biased)} "
            f"| {_fmt(p.second_biased)} | {_fmt(p.first_position_rate)} |"
        )
    lines += [
        "",
        "## Verbosity bias",
        "",
        "Share of decisive verdicts that picked the clearly longer answer "
        f"(at least {report.min_length_ratio}x the words).",
        "",
        "| Judge | Judge picks longer | Experts pick longer | Gap |",
        "| --- | --- | --- | --- |",
    ]
    for judge in report.judges:
        v = judge.verbosity
        lines.append(
            f"| {judge.judge} | {_fmt(v.judge_longer_rate)} | {_fmt(v.human_longer_rate)} "
            f"| {_fmt(v.gap)} |"
        )
    lines.append("")
    return "\n".join(lines)
