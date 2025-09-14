SHELL := /bin/bash

.PHONY: setup sample run aggregate publish-logs lint test

setup:
	bash scripts/setup.sh

sample:
	bash scripts/run_sample.sh

run:
	DATASET=$${DATASET:-data/sample_sanitized.jsonl} SUBSET=$${SUBSET:-512} bash scripts/run_evalset.sh

aggregate:
	.venv/bin/python -m robustcbrn.analysis.aggregate --logs $${LOGS_DIR:-logs} --out $${RESULTS_DIR:-artifacts/results} $${CONSENSUS_K:+--k $${CONSENSUS_K}}

publish-logs:
	bash scripts/publish_logs.sh $${LOGS_DIR:-logs} $${OUT_DIR:-site/logs}

download-models:
	bash scripts/download_models.sh

lint:
	.venv/bin/ruff check robustcbrn tests

test:
	.venv/bin/pytest -q

fill-report:
	.venv/bin/python scripts/fill_report.py
