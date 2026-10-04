# Progress

## Current phase

Milestone 4 (results explorer, container, results script) is submitted through a PR.
Milestones 1-3 are merged. A local Ollama study (`configs/ollama-study.yaml`) is running;
milestone 5 publishes its measured results and the write-up.

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

- Metrics: S1/S2 agreement, Cohen's kappa, expert ceiling, position consistency and
  first/second bias, first-position pick rate, longer-answer preference, all with seeded
  bootstrap intervals and explicit undefined results.
- Seeded mock judge (accuracy, position bias, verbosity bias, failure rate) and
  `judgecheck run` / `report` driven by YAML; stored per-judge JSONL runs allow
  re-scoring without re-judging.

- LLM judges for Anthropic (`claude-sonnet-5-5`, no temperature, `between_tools` thinking),
  OpenAI (`gpt-4o-mini`, temperature 0, fixed seed) and Ollama; strict `[[A]]/[[B]]/[[C]]`
  parsing; delimiter-neutralized prompts; atomic response cache; `judgecheck plan`; paid runs
  need a covering `max_paid_calls`; missing keys raise instead of mocking.

- Streamlit explorer (`judgecheck app`) with agreement, position and verbosity views and a
  drill-down; runs now store `comparisons.jsonl` beside the report. Docker image runs as a
  non-root user with a health check; Compose adds an optional Ollama profile.
- `scripts/results.py` regenerates `docs/results.md` and `docs/results.json` with the commands
  and settings that produced them.

## Validation

- Dataset schema, license (CC BY 4.0) and revision confirmed from the Hugging Face API.
- `make lint typecheck test ingest` passes on Python 3.12.12: 20 tests, 96.15% coverage.
- `make test-network` passes: the pinned download has 3,355 votes and contains every
  bundled sample row.
- `judgecheck ingest --source hf` normalizes the pinned split into 1,814 comparisons,
  961 with more than one vote; an independent pandas count gives the same figures.
- Wheel and sdist build and pass `twine check`; the wheel includes the sample.
- Milestone 2: 51 tests pass at 97.9% coverage. Injected biases are recovered: a bias-free
  mock is 100% order-consistent; position_bias=1 gives first-pick rate 1.0; verbosity_bias=1
  gives a longer-pick rate of 1.0.
- Mock run on the full pinned split (1,814 comparisons, seed 42, `source: hf`): expert-to-expert
  agreement is S1 0.672 [0.646, 0.698] and S2 0.828 [0.801, 0.853] over 961 multi-vote
  comparisons, in line with the roughly 63%/81% the MT-Bench paper reports. Mock judges are
  simulations, not model results.
- Milestone 3: 82 tests pass at 97.2% coverage. Provider clients are faked: no network or paid
  call occurs in the default suite. The Anthropic and OpenAI backends have therefore never been
  run against the real APIs (no keys are configured); their request shapes follow the SDK
  documentation and are asserted in tests.
- Live Ollama probe (3 calls per cell, not a result): both models return a parseable verdict on
  the direct strategy (about 5.5 s per call); `llama3.2:1b` ignored the `[[A]]` format on
  rationale and rubric prompts (0/3 parsed, answering "Assistant A's reply is better"), and
  `qwen2.5:1.5b` parsed 2/3 and 3/3. Strict parsing keeps these as unusable responses.
- Milestone 4: 92 tests pass at 97.2% coverage. The explorer's data layer is unit tested and
  the Streamlit script renders headlessly under `streamlit.testing` (the script is excluded
  from coverage because it runs in that runtime). `docker compose build` and
  `docker compose up --wait` succeed locally: the container reports healthy and serves the
  mounted reports.
- Python 3.11 is validated only by CI.

## Constraints

- No Anthropic or OpenAI key is configured. Ollama 0.35.1 runs locally in Docker
  (container `judgecheck-ollama`, port 11434 bound to localhost) with
  `qwen2.5:1.5b` and `llama3.2:1b`. The machine has 8 CPU cores, 15 GB RAM and no
  GPU, so the brief's 8B models are deferred; small models stand in during development.
- Python 3.12 is available via uv; the default system Python is 3.14.
- No results, releases or packages have been published.
