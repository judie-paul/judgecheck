# Changelog

## 0.1.0 - tooling release

- Repository standards, contribution guidance and implementation roadmap.
- Validated MT-Bench schema, turn-aware canonical normalization and pinned loaders.
- Agreement (S1, S2, kappa), position-bias and verbosity-bias metrics with seeded
  bootstrap intervals, and an expert-agreement ceiling.
- LLM judges for Anthropic, OpenAI and Ollama with direct, rationale and rubric strategies,
  strict verdict parsing, an on-disk response cache, a `plan` command and a paid-call budget guard.
- Streamlit results explorer, Dockerfile and Compose with a health check, and
  `scripts/results.py` to regenerate published results with their commands.
- Seeded mock judge, YAML-configured `run` and `report` commands, JSON and Markdown reports.
- Bundled CC BY 4.0 sample, `judgecheck ingest` command, Makefile and CI.

Study results (Ollama, Claude, GPT-4o-mini) are not part of this release.

## Unreleased

- Reduced local study configs (`ollama-study.yaml`, `ollama-strategies.yaml`) sized for CPU-only runs.
