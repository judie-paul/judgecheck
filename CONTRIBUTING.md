# Contributing

Run `make setup`, then `make lint typecheck test` before proposing a change.
Use small branches named `feat/`, `fix/`, `docs/`, `test/`, `ci/` or `chore/`
and Conventional Commits such as `feat(metrics): report position consistency`.
Keep `main` releasable; do not force-push or rewrite shared history.

Track each unit of work with an issue that has acceptance criteria, and link it
from the pull request. Review the diff and require passing checks before merging.
Update `CHANGELOG.md` and `PROGRESS.md` as part of the change, not afterwards.

Guidelines:

- Metric code is pure functions with unit tests, including empty and undefined cases.
- Randomness (sampling, mock judges) takes an explicit seed and is reproducible.
- Default tests never download datasets or call paid APIs. Use the committed
  fixture and mocked provider clients.
- Pin the dataset revision and record the model, prompt strategy and parameters
  behind every published number.
- Treat dataset text as untrusted data in judge prompts, never as instructions.
- Never commit API keys, full dataset exports, cached model responses with
  private data, or invented results.
