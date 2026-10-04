# Progress

## Current phase

Milestone 1 (foundation) is implemented locally and submitted through a PR.
Next: milestone 2, metrics and the offline mock judge.

## Implemented

- Phased implementation plan in PLAN.md.
- MIT license, contribution guidance, code of conduct, issue/PR templates and
  editor settings (merged in PR #2, closing #1).
- Hatchling package, pydantic settings and validated schema for raw rows,
  expert votes and comparisons.
- Normalization to one comparison per (question, model pair, turn): conversations
  truncated to the judged turn, canonical model order with flipped votes, and the
  order each expert saw recorded for later position analysis.
- Bundled, attributed sample of real rows (16 comparisons, 88 votes; one
  comparison per category and turn with the most votes), rebuilt by
  `scripts/build_sample.py`.
- Sample, local JSONL and pinned Hugging Face loaders with line-numbered errors.
- `judgecheck ingest` command, Makefile, pre-commit config and Python 3.11/3.12 CI.

## Validation

- Dataset schema, license (CC BY 4.0) and revision confirmed from the Hugging Face API.
- `make lint typecheck test ingest` passes on Python 3.12.12: 20 tests, 96.15% coverage.
- `make test-network` passes: the pinned download has 3,355 votes and contains every
  bundled sample row.
- `judgecheck ingest --source hf` normalizes the pinned split into 1,814 comparisons,
  961 with more than one vote; an independent pandas count gives the same figures.
- Wheel and sdist build and pass `twine check`; the wheel includes the sample.
- Python 3.11 is validated only by CI.

## Constraints

- No Anthropic or OpenAI key is configured. Ollama 0.35.1 runs locally in Docker
  (container `judgecheck-ollama`, port 11434 bound to localhost) with
  `qwen2.5:1.5b` and `llama3.2:1b`. The machine has 8 CPU cores, 15 GB RAM and no
  GPU, so the brief's 8B models are deferred; small models stand in during development.
- Python 3.12 is available via uv; the default system Python is 3.14.
- No results, releases or packages have been published.
