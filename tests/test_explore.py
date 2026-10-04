import importlib.util
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from judgecheck.cli import app
from judgecheck.explore import (
    FILTERS,
    bias_rows,
    drilldown,
    find_studies,
    load_study,
    majority,
    summary_rows,
    verdict_label,
)
from judgecheck.pipeline import run_study
from judgecheck.runconfig import RunConfig

from .helpers import comparison

APP = Path(__file__).resolve().parents[1] / "src" / "judgecheck" / "app.py"


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("reports")
    config = RunConfig.model_validate(
        {
            "out_dir": root / "mock",
            "bootstrap": 20,
            "judges": [
                {"name": "fair", "type": "mock", "params": {"accuracy": 0.6}},
                {"name": "first", "type": "mock", "params": {"position_bias": 1.0}},
                {"name": "broken", "type": "mock", "params": {"failure_rate": 0.5}},
            ],
        }
    )
    run_study(config)
    return root


def test_find_and_load_study(reports: Path) -> None:
    assert find_studies(reports) == [reports / "mock"]
    assert find_studies(reports / "missing") == []
    study = load_study(reports / "mock")
    assert len(study.comparisons) == 16
    assert set(study.records) == {"fair", "first", "broken"}


def test_summary_and_bias_rows(reports: Path) -> None:
    study = load_study(reports / "mock")
    rows = {row["judge"]: row for row in summary_rows(study)}
    assert rows["fair"]["comparisons"] == 16 and rows["fair"]["S1"] is not None
    assert rows["broken"]["unusable"] is not None and rows["broken"]["unusable"] > 0.2
    bias = {row["judge"]: row for row in bias_rows(study)}
    assert bias["first"]["always first"] == 1.0
    assert bias["fair"]["order-consistent"] == 1.0


def test_drilldown_filters(reports: Path) -> None:
    study = load_study(reports / "mock")
    assert set(FILTERS) == {"all", "disagrees", "inconsistent", "unusable"}
    assert len(drilldown(study, "fair", "all")) == 16
    assert drilldown(study, "fair", "inconsistent") == []
    # An always-first judge says "a" with model a first and "b" with model b first.
    assert len(drilldown(study, "first", "inconsistent")) == 16
    assert drilldown(study, "broken", "unusable")
    for item in drilldown(study, "fair", "disagrees"):
        record = study.records["fair"][item.id]
        assert record.consensus is not None and record.consensus != majority(item)


def test_majority_resolves_ties_to_a_tie() -> None:
    assert majority(comparison("c", "x", "y", ("a", "a", "b"))) == "a"
    assert majority(comparison("c", "x", "y", ("a", "b"))) == "tie"
    assert majority(comparison("c", "x", "y", ("tie",))) == "tie"


def test_verdict_labels_name_the_model_and_position() -> None:
    item = comparison("c", "x", "y")
    assert verdict_label(item, "a", "ab") == "alpha (displayed first)"
    assert verdict_label(item, "a", "ba") == "alpha (displayed second)"
    assert verdict_label(item, "tie", "ab") == "tie"
    assert verdict_label(item, None, "ab") == "unusable response"


def test_app_renders_headlessly(reports: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("JUDGECHECK_REPORTS", str(reports))
    page = testing.AppTest.from_file(str(APP), default_timeout=60).run()
    assert not page.exception
    assert [tab.label for tab in page.tabs] == [
        "Agreement",
        "Position bias",
        "Verbosity bias",
        "Comparisons",
    ]
    assert page.title[0].value.startswith("JudgeCheck")
    assert page.selectbox[1].options  # judges are listed


def test_app_explains_an_empty_reports_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("JUDGECHECK_REPORTS", str(tmp_path))
    page = testing.AppTest.from_file(str(APP), default_timeout=60).run()
    assert not page.exception
    assert "make pipeline" in page.info[0].value


def test_cli_app_launches_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> Any:
        seen.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = CliRunner().invoke(app, ["app", "--port", "9000"])
    assert result.exit_code == 0
    assert "streamlit" in seen[0] and "--server.port=9000" in seen[0]
    assert seen[0][4].endswith("app.py") and "--server.address=127.0.0.1" in seen[0]


def test_results_script_records_commands_and_settings(reports: Path, tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "results_script", APP.parents[2] / "scripts" / "results.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    study = load_study(reports / "mock")
    page = module.build_page(
        study.report,
        Path("configs/ollama-study.yaml"),
        Path("reports/ollama-study"),
        date(2026, 1, 2),
    )
    assert page.startswith("# Results\n\nGenerated on 2026-01-02")
    assert "python scripts/results.py --run reports/ollama-study" in page
    assert "- `qwen2.5-1.5b-direct`: ollama qwen2.5:1.5b (direct)" in page
    assert "## Expert agreement ceiling" in page and "# JudgeCheck results" not in page
