PYTHON ?= .venv/bin/python

.PHONY: app results docker-build docker-up plan setup lint format typecheck test test-network ingest ingest-hf pipeline report clean
setup:
	uv --cache-dir /tmp/judgecheck-uv-cache venv --python 3.12 .venv
	uv --cache-dir /tmp/judgecheck-uv-cache pip install --python $(PYTHON) -e '.[dev,llm,hf,app]'
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .
typecheck:
	$(PYTHON) -m mypy
test:
	$(PYTHON) -m pytest -m 'not network'
test-network:
	$(PYTHON) -m pytest -m network --no-cov
ingest:
	$(PYTHON) -m judgecheck.cli ingest --source sample --out data/comparisons.jsonl
ingest-hf:
	$(PYTHON) -m judgecheck.cli ingest --source hf --out data/comparisons.jsonl
plan:
	$(PYTHON) -m judgecheck.cli plan --config configs/default.yaml
pipeline:
	$(PYTHON) -m judgecheck.cli run --config configs/default.yaml
report:
	$(PYTHON) -m judgecheck.cli report --config configs/default.yaml
app:
	$(PYTHON) -m judgecheck.cli app
results:
	$(PYTHON) scripts/results.py --run reports/ollama-study --config configs/ollama-study.yaml
docker-build:
	docker compose build
docker-up:
	docker compose up --build
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov build dist .coverage
