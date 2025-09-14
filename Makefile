SHELL := /bin/bash

.PHONY: setup sample run aggregate publish-logs lint test
.PHONY: data data-list
.PHONY: pipeline pipeline-validate pipeline-setup pipeline-sample pipeline-full
.PHONY: pipeline-aggregate pipeline-figures pipeline-tests pipeline-report pipeline-verify
.PHONY: pipeline-safe

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

# Data management shortcuts (used by integration tests)
data:
	python scripts/fetch_data.py $(DATASET)

data-list:
	python scripts/fetch_data.py --list

fill-report:
	.venv/bin/python scripts/fill_report.py

# New Pipeline Targets
pipeline:
	bash scripts/run_pipeline.sh

pipeline-validate:
	bash scripts/run_pipeline.sh --steps validate

pipeline-setup:
	bash scripts/run_pipeline.sh --steps validate,setup

pipeline-sample:
	bash scripts/run_pipeline.sh --steps validate,setup,discover,sample

pipeline-full:
	bash scripts/run_pipeline.sh --steps validate,setup,discover,sample,full,aggregate,figures,tests,report,verify

pipeline-aggregate:
	bash scripts/run_pipeline.sh --steps aggregate

pipeline-figures:
	bash scripts/run_pipeline.sh --steps figures

pipeline-tests:
	bash scripts/run_pipeline.sh --steps tests

pipeline-report:
	bash scripts/run_pipeline.sh --steps report

pipeline-verify:
	bash scripts/run_pipeline.sh --steps verify

# Safe local pipeline (tiny model, CPU-only)
pipeline-safe:
	MODELS=sshleifer/tiny-gpt2 DEVICE=cpu DTYPE=float32 SEEDS=123 SUBSET_SIZE=8 \
	bash scripts/run_pipeline.sh --steps validate,setup,sample,aggregate,figures,tests,report,verify

# Individual Pipeline Scripts (for advanced users)
validate-platform:
	bash scripts/validate_platform.sh

setup-env:
	bash scripts/setup_env.sh

discover-entry-points:
	bash scripts/discover_entry_points.sh

run-sample-eval:
	bash scripts/run_sample_evaluation.sh

run-full-eval:
	bash scripts/run_full_evaluation.sh

aggregate-results:
	bash scripts/aggregate_results.sh

generate-figures:
	bash scripts/generate_figures.sh

run-tests-security:
	bash scripts/run_tests_and_security.sh

generate-report:
	bash scripts/generate_report.sh

final-verification:
	bash scripts/final_verification.sh
