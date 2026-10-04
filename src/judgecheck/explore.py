"""Data access for the results explorer, kept free of any UI code so it can be tested."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from judgecheck.ingest import read_comparisons
from judgecheck.report import JudgeReport, RunReport
from judgecheck.results import JudgeRecord, Order, to_position
from judgecheck.runner import read_records
from judgecheck.schema import Comparison, Verdict

FILTERS = {
    "all": "All comparisons",
    "disagrees": "Judge consensus differs from the expert majority",
    "inconsistent": "Verdict changes when the answer order is swapped",
    "unusable": "At least one response was unusable",
}


@dataclass(frozen=True)
class Study:
    """One finished run directory."""

    directory: Path
    report: RunReport
    comparisons: dict[str, Comparison]
    records: dict[str, dict[str, JudgeRecord]]


def find_studies(root: Path) -> list[Path]:
    """Directories at or directly under ``root`` that hold a finished report."""
    candidates = [root, *sorted(path for path in root.glob("*") if path.is_dir())]
    return [path for path in candidates if (path / "results.json").exists()]


def load_study(directory: Path) -> Study:
    """Load a report, its comparisons and every judge's stored records."""
    report = RunReport.model_validate_json((directory / "results.json").read_text("utf-8"))
    comparisons = {c.id: c for c in read_comparisons(directory / "comparisons.jsonl")}
    records = {
        judge.judge: {
            r.comparison_id: r for r in read_records(directory / "runs" / f"{judge.judge}.jsonl")
        }
        for judge in report.judges
    }
    return Study(directory, report, comparisons, records)


def majority(comparison: Comparison) -> Verdict:
    """The most common expert verdict; ties between verdicts resolve to a tie."""
    counts = Counter(vote.verdict for vote in comparison.votes).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return "tie"
    return counts[0][0]


def _value(estimate: Any) -> float | None:
    return None if estimate is None else estimate.value


def summary_rows(study: Study) -> list[dict[str, Any]]:
    """One flat row per judge for the overview table and chart."""
    rows = []
    for judge in study.report.judges:
        consensus = judge.variants["consensus"]
        rows.append(
            {
                "judge": judge.judge,
                "comparisons": judge.comparisons,
                "S1": consensus.agreement_s1.value,
                "S1 low": consensus.agreement_s1.low,
                "S1 high": consensus.agreement_s1.high,
                "S2": consensus.agreement_s2.value,
                "S2 low": consensus.agreement_s2.low,
                "S2 high": consensus.agreement_s2.high,
                "kappa": consensus.kappa.value,
                "unusable": judge.unparseable_rate,
            }
        )
    return rows


def bias_rows(study: Study) -> list[dict[str, Any]]:
    """One flat row per judge for the position and verbosity views."""
    rows = []
    for judge in study.report.judges:
        position, verbosity = judge.position, judge.verbosity
        rows.append(
            {
                "judge": judge.judge,
                "order-consistent": position.consistent.value,
                "always first": position.first_biased.value,
                "always second": position.second_biased.value,
                "first-position pick rate": position.first_position_rate.value,
                "first-position pick rate low": position.first_position_rate.low,
                "first-position pick rate high": position.first_position_rate.high,
                "judge picks longer": verbosity.judge_longer_rate.value,
                "experts pick longer": verbosity.human_longer_rate.value,
                "longer gap": verbosity.gap.value,
                "longer gap low": verbosity.gap.low,
                "longer gap high": verbosity.gap.high,
            }
        )
    return rows


def _matches(name: str, comparison: Comparison, record: JudgeRecord) -> bool:
    if name == "disagrees":
        return record.consensus is not None and record.consensus != majority(comparison)
    if name == "inconsistent":
        return record.ab is not None and record.ba is not None and record.ab != record.ba
    if name == "unusable":
        return record.ab is None or record.ba is None
    return True


def drilldown(study: Study, judge: str, filter_name: str) -> list[Comparison]:
    """Comparisons judged by ``judge`` that satisfy a filter, sorted by id."""
    records = study.records[judge]
    return [
        study.comparisons[comparison_id]
        for comparison_id in sorted(records)
        if comparison_id in study.comparisons
        and _matches(filter_name, study.comparisons[comparison_id], records[comparison_id])
    ]


def verdict_label(comparison: Comparison, verdict: Verdict | None, order: Order) -> str:
    """A human-readable verdict, naming the model and where it was displayed."""
    if verdict is None:
        return "unusable response"
    if verdict == "tie":
        return "tie"
    model = comparison.model_a if verdict == "a" else comparison.model_b
    return f"{model} (displayed {to_position(verdict, order)})"


def judge_report_of(study: Study, judge: str) -> JudgeReport:
    """The scored report for one judge."""
    return next(item for item in study.report.judges if item.judge == judge)
