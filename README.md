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

## Status

Early development. See `PLAN.md` for the roadmap and `PROGRESS.md` for what is
implemented and verified. No results are published yet.

## License

Code is MIT licensed. MT-Bench human judgments are released by LMSYS under
CC BY 4.0.
