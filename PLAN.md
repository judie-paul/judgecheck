# JudgeCheck implementation plan

## Scope and working agreement

Build the JudgeCheck study described in the portfolio brief incrementally: run
several LLM judges over MT-Bench pairwise comparisons that human experts already
judged, then measure agreement with the experts, position bias and verbosity bias
across prompting strategies. The owner has authorized creating the public
judie-paul/judgecheck repository and delivering work through issues, branches and
pull requests. Real API runs, the published write-up and release tagging are gated
on the owner supplying keys and approving spend.

Data: `lmsys/mt_bench_human_judgments`, `human` split (3,355 expert votes, CC BY 4.0),
pinned to revision `f7d2896d2cc5d80f8b55c2bbc722613555233c25`. Each row stores both
conversation turns; the `turn` field says which turn the expert judged.

## Milestone 1: foundation

- Hatchling package, Python 3.11+, pydantic settings and a canonical schema.
- Normalize raw rows into one comparison per (question, model pair, turn), with the
  conversation truncated to the judged turn and every expert vote attached.
- Canonical model order so the same pair judged in either order aggregates; flip the
  vote when the order is swapped. Keep the shown order for position analysis.
- Committed, attributed fixture sampled from the real dataset; local JSONL loading;
  optional pinned Hugging Face download behind an extra.
- Typer `ingest` command, Makefile, ruff, mypy strict, pytest coverage, Python 3.11/3.12 CI.
- Gate: fixture and download produce identical normalized records; no network in tests.

## Milestone 2: metrics and offline judge

- Vote-level agreement as defined in the MT-Bench paper: with ties (S1) and without
  ties (S2), plus Cohen's kappa. Human-to-human agreement as the reference ceiling.
- Position consistency: judge every comparison in both orders; report consistent,
  first-position-biased and second-position-biased rates.
- Verbosity preference: how often the judge picks the longer answer when lengths
  differ, compared with how often the experts do on the same comparisons.
- Bootstrap confidence intervals; explicit undefined results for empty denominators.
- Seeded mock judge with configurable accuracy, position bias and verbosity bias.
  Tests prove each metric recovers the bias that was injected.
- YAML-configured `run` command writing JSON and Markdown reports.
- Gate: complete offline run on the fixture with no keys.

## Milestone 3: real judges and prompting strategies

- Providers: Claude Sonnet (Anthropic), GPT-4o-mini (OpenAI), Llama 3.1 8B and
  Qwen 2.5 7B through Ollama. Explicit selection; timeouts, bounded retries and limits.
- Strategies: direct verdict, rationale-first and rubric-guided. Every judge already runs in
  both display orders, so order-swapped consensus (inconsistent pairs become ties) is a
  scoring variant of every run rather than a separate strategy.
- Strict verdict parsing; report the unparseable rate instead of hiding it.
- On-disk response cache keyed by provider, model, strategy and prompt hash, so
  reruns and report changes cost nothing. Budget guard: sample limit and dry-run
  cost estimate before any paid call.
- All providers tested with mocked clients. Never call a paid API in default tests.
- Gate: a small, owner-approved real run reproduces from the cache.

## Milestone 4: explorer and delivery

- Streamlit explorer: agreement by judge and strategy, bias charts, and
  per-comparison drill-down showing the expert votes beside the judge's verdicts.
- Dockerfile and Compose (optional Ollama service), `make setup && make pipeline`.
- Results script that regenerates `docs/results.md` from stored runs.

## Milestone 5: study and release

- Owner-approved full or stratified run; document the sample, cost and commands.
- Written analysis in `docs/analysis.md` for Medium/Hashnode publication.
- README with measured results, then tag v1.0.0.

## Non-goals

No fine-tuning, no new human labels, and no claims beyond what was measured.
