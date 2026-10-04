# JudgeCheck

How far can an LLM judge be trusted to grade another LLM?

JudgeCheck runs LLM judges over the MT-Bench pairwise comparisons that human
experts have already judged
([`lmsys/mt_bench_human_judgments`](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments),
about 3,300 expert votes). It measures each judge's agreement with the experts,
position bias (favoring whichever answer is shown first) and verbosity bias
(favoring the longer answer) across prompting strategies.

## Quick start

```bash
make setup            # Python 3.12 virtualenv with development tools
make ingest           # normalize the bundled sample, no network needed
make ingest-hf        # normalize the full pinned dataset (needs the hf extra)
make pipeline         # offline run: mock judges on the sample -> reports/results.md
make app              # results explorer (needs the app extra)
make lint typecheck test
```

Each comparison is one question turn answered by two models, with every expert
vote attached. Model order is canonical, and the order each expert saw is kept
for position-bias analysis.

## What is measured

Every judge sees each comparison twice, once with each answer displayed first.

- **Agreement with experts**: MT-Bench's S1 (ties counted) and S2 (ties dropped), and
  Cohen's kappa, using the consensus of both orders (a tie when the orders disagree).
  The expert-to-expert agreement on the same data is the ceiling to compare against.
- **Position bias**: how often the verdict survives swapping the answers, how often the
  judge picks the first (or second) displayed answer in both orders, and its overall
  first-position pick rate (0.5 is fair).
- **Verbosity bias**: how often the judge picks the clearly longer answer, next to how often
  the experts do on the same comparisons.

All intervals are 95% percentile bootstraps resampled by comparison. Undefined results
are reported as undefined, never as zero. Judges are configured in `configs/*.yaml`;
`make pipeline` runs the offline mock judges, whose known biases verify the metrics
(`tests/test_judges.py` recovers each injected bias).

## Explorer

`make app` (or `judgecheck app`) opens a Streamlit explorer over any finished run in
`reports/`: agreement and bias charts with confidence intervals, and a per-comparison
drill-down that shows the expert votes beside the judge's verdicts in both display orders,
filterable to disagreements with the experts, order-sensitive verdicts and unusable responses.
With Docker, `docker compose up --build` serves it at http://127.0.0.1:8501 from `./reports`;
`docker compose --profile ollama up` also starts a local model server.

## Judges

Judges are chosen explicitly in a YAML config; see `configs/`.

| Type | Models | Cost |
| --- | --- | --- |
| `mock` | seeded simulation with injectable biases | free |
| `ollama` | any local model, for example `qwen2.5:1.5b`, `llama3.2:1b` | free |
| `openai` | `gpt-4o-mini` (default) | paid |
| `anthropic` | `claude-sonnet-5-5` (default) | paid |

Strategies: `direct` (verdict only), `rationale` (short comparison, then verdict) and
`rubric` (criteria-guided comparison, then verdict). Verdicts must be `[[A]]`, `[[B]]` or
`[[C]]` (tie); a response without one counts as **unusable** and is reported, never guessed.
Dataset text is delimited and declared to be data, not instructions.

Responses are cached on disk (`.cache/responses`) by provider, model, settings and prompt, so
repeated or interrupted runs cost nothing extra. Spending is gated: `judgecheck plan` shows
calls, cache hits and approximate tokens without calling any model, and a run with paid judges
refuses to start unless `max_paid_calls` in the config covers the calls still to be made. A
missing API key is an error, never a silent fallback to a mock.

## Status

Early development. See `PLAN.md` for the roadmap and `PROGRESS.md` for what is
implemented and verified. No results are published yet.

## License

Code is MIT licensed. MT-Bench human judgments are released by LMSYS under
CC BY 4.0.
