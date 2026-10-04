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
make lint typecheck test
```

Each comparison is one question turn answered by two models, with every expert
vote attached. Model order is canonical, and the order each expert saw is kept
for position-bias analysis.

## Status

Early development. See `PLAN.md` for the roadmap and `PROGRESS.md` for what is
implemented and verified. No results are published yet.

## License

Code is MIT licensed. MT-Bench human judgments are released by LMSYS under
CC BY 4.0.
