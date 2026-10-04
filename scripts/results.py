"""Regenerate docs/results.md and docs/results.json from a stored run.

    python scripts/results.py --run reports/ollama-study --config configs/ollama-study.yaml

The published numbers are copied from the stored report, never recomputed by hand, and the
page records the commands and settings that produced them.
"""

import argparse
import shutil
from datetime import date
from pathlib import Path

from judgecheck.report import RunReport, to_markdown
from judgecheck.runconfig import load_config

ROOT = Path(__file__).resolve().parents[1]


def build_page(report: RunReport, config_path: Path, run_dir: Path, today: date) -> str:
    """Wrap the generated tables with how they were produced."""
    config = load_config(config_path)
    judges = "\n".join(
        f"- `{spec.name}`: {spec.type} {spec.params.get('model', '')} "
        f"({spec.params.get('strategy', 'n/a')})".replace("  ", " ")
        for spec in config.judges
    )
    header = f"""# Results

Generated on {today.isoformat()} from `{run_dir.as_posix()}`. Regenerate with:

```bash
judgecheck run --config {config_path.as_posix()}
python scripts/results.py --run {run_dir.as_posix()} --config {config_path.as_posix()}
```

Source: `{config.source}`, {report.comparisons} comparisons
{"(seeded random sample of the pinned split)" if config.limit else "(the full pinned split)"},
seed {config.seed}. Judges:

{judges}

"""
    body = to_markdown(report).split("\n", 1)[1]
    return header + body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run directory with results.json")
    parser.add_argument("--config", type=Path, required=True, help="config that produced the run")
    parser.add_argument("--out", type=Path, default=ROOT / "docs")
    args = parser.parse_args()
    report = RunReport.model_validate_json((args.run / "results.json").read_text("utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.md").write_text(
        build_page(report, args.config, args.run, date.today()), encoding="utf-8"
    )
    shutil.copyfile(args.run / "results.json", args.out / "results.json")
    print(f"wrote {args.out / 'results.md'} and {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
