import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from judgecheck.cli import app
from judgecheck.config import Settings
from judgecheck.ingest import (
    IngestError,
    load_hf,
    load_jsonl,
    load_sample,
    read_comparisons,
    write_comparisons,
)
from judgecheck.normalize import comparison_id, flip, normalize
from judgecheck.schema import Comparison, RawJudgment, category_for


def conversation(answer_1: str, answer_2: str = "second answer") -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": answer_1},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": answer_2},
    ]


def raw(
    model_a: str = "alpha",
    model_b: str = "beta",
    winner: str = "model_a",
    judge: str = "expert_1",
    turn: int = 1,
    question_id: int = 81,
) -> RawJudgment:
    answers = {"alpha": "alpha answer", "beta": "beta answer"}
    return RawJudgment.model_validate(
        {
            "question_id": question_id,
            "model_a": model_a,
            "model_b": model_b,
            "winner": winner,
            "judge": judge,
            "conversation_a": conversation(answers[model_a], f"{model_a} turn 2"),
            "conversation_b": conversation(answers[model_b], f"{model_b} turn 2"),
            "turn": turn,
        }
    )


def test_categories_follow_mt_bench_question_ranges() -> None:
    assert category_for(81) == "writing"
    assert category_for(111) == "math"
    assert category_for(160) == "humanities"
    for invalid in (80, 161):
        with pytest.raises(ValueError):
            category_for(invalid)


def test_turn_one_hides_the_second_question() -> None:
    (comparison,) = normalize([raw(turn=1)])
    assert [m.content for m in comparison.conversation_a] == ["first question", "alpha answer"]
    assert comparison.answer_a == "alpha answer"
    assert comparison.answer_b == "beta answer"


def test_turn_two_keeps_full_context() -> None:
    (comparison,) = normalize([raw(turn=2)])
    assert len(comparison.conversation_a) == 4
    assert comparison.answer_a == "alpha turn 2"
    assert comparison.id == "q81-t2-alpha-vs-beta"


def test_reversed_presentation_merges_and_flips_votes() -> None:
    rows = [
        raw(judge="expert_1", winner="model_a"),
        raw(model_a="beta", model_b="alpha", winner="model_a", judge="expert_2"),
        raw(model_a="beta", model_b="alpha", winner="tie", judge="expert_3"),
    ]
    (comparison,) = normalize(rows)
    assert (comparison.model_a, comparison.model_b) == ("alpha", "beta")
    assert comparison.answer_a == "alpha answer"
    votes = {vote.judge: (vote.verdict, vote.shown_swapped) for vote in comparison.votes}
    assert votes == {
        "expert_1": ("a", False),
        "expert_2": ("b", True),
        "expert_3": ("tie", True),
    }


def test_turns_and_questions_are_separate_comparisons() -> None:
    comparisons = normalize([raw(turn=1), raw(turn=2), raw(question_id=111)])
    assert [c.id for c in comparisons] == [
        "q111-t1-alpha-vs-beta",
        "q81-t1-alpha-vs-beta",
        "q81-t2-alpha-vs-beta",
    ]
    assert comparisons[0].category == "math"


def test_conflicting_conversation_text_is_rejected() -> None:
    other = RawJudgment.model_validate(
        raw(judge="expert_2").model_dump() | {"conversation_a": conversation("edited")}
    )
    with pytest.raises(ValueError, match="disagree"):
        normalize([raw(), other])


def test_self_comparison_is_rejected() -> None:
    with pytest.raises(ValueError, match="itself"):
        normalize([raw(model_b="alpha")])


def test_comparison_rejects_non_canonical_order() -> None:
    (comparison,) = normalize([raw()])
    data = comparison.model_dump() | {"model_a": "zeta"}
    with pytest.raises(ValidationError, match="canonical"):
        Comparison.model_validate(data)


def test_comparison_rejects_mismatched_questions() -> None:
    (comparison,) = normalize([raw()])
    data = comparison.model_dump()
    data["conversation_b"][0]["content"] = "a different question"
    with pytest.raises(ValidationError, match="same user messages"):
        Comparison.model_validate(data)


def test_flip_and_id_are_order_independent() -> None:
    assert [flip(v) for v in ("a", "b", "tie")] == ["b", "a", "tie"]
    assert comparison_id(81, 1, "beta", "alpha") == comparison_id(81, 1, "alpha", "beta")


def test_raw_rows_reject_unknown_winner() -> None:
    with pytest.raises(ValidationError):
        RawJudgment.model_validate(raw().model_dump() | {"winner": "both"})


def test_bundled_sample_is_real_and_normalizes() -> None:
    rows = load_sample()
    comparisons = normalize(rows)
    assert sum(len(c.votes) for c in comparisons) == len(rows)
    assert any(len(c.votes) > 1 for c in comparisons)
    assert any(v.shown_swapped for c in comparisons for v in c.votes)
    assert {c.turn for c in comparisons} == {1, 2}


def test_jsonl_round_trip_is_byte_stable(tmp_path: Path) -> None:
    comparisons = normalize(load_sample())
    first, second = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    assert write_comparisons(comparisons, first) == len(comparisons)
    write_comparisons(read_comparisons(first), second)
    assert first.read_bytes() == second.read_bytes()


def test_local_jsonl_errors_report_line_numbers(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    good = raw().model_dump_json()
    path.write_text(f"{good}\n\n{json.dumps({'question_id': 81})}\n", encoding="utf-8")
    with pytest.raises(IngestError, match=r"raw\.jsonl:3"):
        load_jsonl(path)
    path.write_text(good + "\n", encoding="utf-8")
    assert load_jsonl(path) == [raw()]


def test_invalid_normalized_file_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "comparisons.jsonl"
    path.write_text('{"id": "broken"}\n', encoding="utf-8")
    with pytest.raises(IngestError, match=r"comparisons\.jsonl:1"):
        read_comparisons(path)


def test_cli_ingest_sample(tmp_path: Path) -> None:
    out = tmp_path / "comparisons.jsonl"
    result = CliRunner().invoke(app, ["ingest", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "comparisons" in result.output
    assert read_comparisons(out) == normalize(load_sample())


def test_cli_ingest_jsonl_requires_path(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["ingest", "--source", "jsonl"])
    assert result.exit_code == 1
    assert "--path is required" in result.output


def test_cli_ingest_jsonl(tmp_path: Path) -> None:
    source, out = tmp_path / "raw.jsonl", tmp_path / "out.jsonl"
    source.write_text(raw().model_dump_json() + "\n", encoding="utf-8")
    result = CliRunner().invoke(
        app, ["ingest", "--source", "jsonl", "--path", str(source), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert len(read_comparisons(out)) == 1


def test_settings_pin_the_dataset_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JUDGECHECK_DATASET_REVISION", raising=False)
    assert Settings(_env_file=None).dataset_revision.startswith("f7d2896d")


@pytest.mark.network
def test_pinned_download_contains_the_sample(tmp_path: Path) -> None:
    rows = load_hf(Settings(_env_file=None, cache_dir=tmp_path))
    assert len(rows) == 3355
    downloaded = {row.model_dump_json() for row in rows}
    assert {row.model_dump_json() for row in load_sample()} <= downloaded


def test_hf_loader_converts_parquet_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd = pytest.importorskip("pandas")
    huggingface_hub = pytest.importorskip("huggingface_hub")
    parquet = tmp_path / "human.parquet"
    pd.DataFrame(
        [raw().model_dump(mode="json"), raw(judge="expert_2").model_dump(mode="json")]
    ).to_parquet(parquet)
    calls: list[dict[str, object]] = []

    def fake_download(*args: object, **kwargs: object) -> str:
        calls.append(kwargs)
        return str(parquet)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    rows = load_hf(Settings(_env_file=None, cache_dir=tmp_path))
    assert rows == [raw(), raw(judge="expert_2")]
    assert calls[0]["revision"] == Settings(_env_file=None).dataset_revision
