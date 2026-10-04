from pathlib import Path

import pytest
from typer.testing import CliRunner

from judgecheck.cli import app
from judgecheck.ingest import IngestError
from judgecheck.judges import MockJudge, build_judge
from judgecheck.pipeline import run_study, score_study
from judgecheck.report import judge_report
from judgecheck.runconfig import JudgeSpec, RunConfig, load_config
from judgecheck.runner import read_records, run_judge, select, write_records

from .helpers import synthetic

COMPARISONS = synthetic(400)


def score(judge: MockJudge):  # type: ignore[no-untyped-def]
    records = {r.comparison_id: r for r in run_judge(judge, COMPARISONS)}
    return judge_report(
        judge.name, COMPARISONS, records, iterations=50, seed=1, min_length_ratio=1.2
    )


def test_bias_free_mock_is_perfectly_position_consistent() -> None:
    report = score(MockJudge("fair", seed=1, accuracy=0.6))
    assert report.position.consistent.value == 1.0
    assert report.position.first_biased.value == 0.0
    assert report.position.first_position_rate.value == pytest.approx(0.5, abs=0.1)


def test_injected_position_bias_is_recovered() -> None:
    always_first = score(MockJudge("first", seed=1, position_bias=1.0)).position
    assert always_first.first_biased.value == 1.0
    assert always_first.first_position_rate.value == 1.0

    partial = score(MockJudge("partial", seed=1, position_bias=0.3)).position
    assert partial.second_biased.value == 0.0
    assert partial.first_biased.value is not None and 0.05 < partial.first_biased.value < 0.4
    assert partial.first_position_rate.value is not None
    assert partial.first_position_rate.value > 0.55


def test_injected_verbosity_bias_is_recovered() -> None:
    fair = score(MockJudge("fair", seed=1, accuracy=0.7)).verbosity
    verbose = score(MockJudge("verbose", seed=1, accuracy=0.7, verbosity_bias=1.0)).verbosity
    assert verbose.judge_longer_rate.value == 1.0
    assert fair.judge_longer_rate.value is not None
    assert fair.judge_longer_rate.value == pytest.approx(0.5, abs=0.12)
    assert verbose.gap.value is not None and fair.gap.value is not None
    assert verbose.gap.value > fair.gap.value + 0.25


def test_accuracy_controls_agreement_with_experts() -> None:
    sharp = score(MockJudge("sharp", seed=1, accuracy=1.0)).variants["consensus"]
    noisy = score(MockJudge("noisy", seed=1, accuracy=0.2)).variants["consensus"]
    assert sharp.agreement_s1.value is not None and noisy.agreement_s1.value is not None
    assert sharp.agreement_s1.value > noisy.agreement_s1.value + 0.2
    assert sharp.kappa.value is not None and noisy.kappa.value is not None
    assert sharp.kappa.value > 0.3 > noisy.kappa.value


def test_unparseable_responses_are_reported_not_hidden() -> None:
    report = score(MockJudge("broken", seed=1, failure_rate=1.0))
    assert report.unparseable_rate == 1.0
    assert report.variants["consensus"].agreement_s1.value is None
    assert report.position.consistent.value is None
    partial = score(MockJudge("flaky", seed=1, failure_rate=0.3))
    assert partial.unparseable_rate is not None and 0.2 < partial.unparseable_rate < 0.4


def test_mock_is_deterministic_per_seed() -> None:
    first = run_judge(MockJudge("m", seed=5, accuracy=0.5), COMPARISONS[:50])
    assert first == run_judge(MockJudge("m", seed=5, accuracy=0.5), COMPARISONS[:50])
    assert first != run_judge(MockJudge("m", seed=6, accuracy=0.5), COMPARISONS[:50])


@pytest.mark.parametrize("field", ["accuracy", "position_bias", "verbosity_bias", "failure_rate"])
def test_mock_rejects_probabilities_outside_unit_interval(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        MockJudge("m", **{field: 1.5})


def test_select_is_seeded_and_validated() -> None:
    assert select(COMPARISONS, None, 1) == sorted(COMPARISONS, key=lambda c: c.id)
    assert select(COMPARISONS, 25, 1) == select(COMPARISONS, 25, 1)
    assert select(COMPARISONS, 25, 1) != select(COMPARISONS, 25, 2)
    assert len(select(COMPARISONS, 10_000, 1)) == len(COMPARISONS)
    with pytest.raises(ValueError):
        select(COMPARISONS, 0, 1)


def test_records_round_trip(tmp_path: Path) -> None:
    records = run_judge(MockJudge("m", seed=1), COMPARISONS[:5])
    write_records(records, tmp_path / "nested" / "m.jsonl")
    assert read_records(tmp_path / "nested" / "m.jsonl") == records


def config(tmp_path: Path, **extra: object) -> RunConfig:
    return RunConfig.model_validate(
        {
            "out_dir": tmp_path / "reports",
            "bootstrap": 20,
            "judges": [
                {"name": "fair", "type": "mock"},
                {"name": "first", "type": "mock", "params": {"position_bias": 1.0}},
            ],
        }
        | extra
    )


def test_run_study_writes_reports_and_rescoring_needs_no_judge(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    report = run_study(cfg)
    assert [judge.judge for judge in report.judges] == ["fair", "first"]
    assert report.comparisons == 16
    assert report.human.comparisons_with_multiple_votes == 16
    assert (tmp_path / "reports" / "runs" / "fair.jsonl").exists()
    markdown = (tmp_path / "reports" / "results.md").read_text()
    assert "| fair |" in markdown and "Expert agreement ceiling" in markdown
    assert (tmp_path / "reports" / "results.json").exists()
    assert score_study(cfg) == report


def test_scoring_without_a_run_explains_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="judgecheck run"):
        score_study(config(tmp_path))


def test_config_validation(tmp_path: Path) -> None:
    spec = {"name": "x", "type": "mock"}
    with pytest.raises(ValueError, match="unique"):
        RunConfig.model_validate({"judges": [spec, spec]})
    with pytest.raises(ValueError, match="letters"):
        RunConfig.model_validate({"judges": [{"name": "../x", "type": "mock"}]})
    with pytest.raises(ValueError, match="path is required"):
        RunConfig.model_validate({"judges": [spec], "source": "jsonl"})
    with pytest.raises(ValueError):
        RunConfig.model_validate({"judges": [spec], "surprise": 1})
    with pytest.raises(ValueError, match="unknown judge type"):
        build_judge("x", "nonsense", {}, seed=1, min_length_ratio=1.2)
    assert JudgeSpec(name="x", type="mock").params == {}


def test_default_config_loads() -> None:
    assert [j.name for j in load_config(Path("configs/default.yaml")).judges] == [
        "mock-fair",
        "mock-first-biased",
        "mock-verbose",
    ]


def test_cli_run_and_report(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        f"out_dir: {tmp_path / 'out'}\nbootstrap: 10\njudges:\n  - {{name: fair, type: mock}}\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["report", "--config", str(path)])
    assert result.exit_code == 1 and "judgecheck run" in result.output
    result = runner.invoke(app, ["run", "--config", str(path)])
    assert result.exit_code == 0, result.output
    assert "scored 1 judges on 16 comparisons" in result.output
    result = runner.invoke(app, ["report", "--config", str(path)])
    assert result.exit_code == 0 and "rescored 1 judges" in result.output
    result = runner.invoke(app, ["run", "--config", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 1
