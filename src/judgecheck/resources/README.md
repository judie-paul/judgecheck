# Bundled MT-Bench sample

`mt_bench_human_sample.jsonl` is an unmodified subset of rows from the `human`
split of [`lmsys/mt_bench_human_judgments`](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
at revision `f7d2896d2cc5d80f8b55c2bbc722613555233c25`, released by LMSYS under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Source: Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*,
NeurIPS 2023 Datasets and Benchmarks.

It lets the project run and test offline. It is rebuilt with
`python scripts/build_sample.py`, which documents the selection rule. Results on
this sample are for smoke testing only, not for publication.
