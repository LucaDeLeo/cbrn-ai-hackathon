# RobustCBRN Eval Docs

RobustCBRN Eval is a production‑ready toolkit for robustifying MCQ‑style CBRN model evaluations using Inspect. It implements choices‑only consensus screening (Deep Ignorance §D.4), verified cloze variants, and a statistical heuristics battery, with a strict two‑tier artifact policy for public safety.

## Docs Index
- Overview: `../overview.md` (high‑level rationale and capabilities)
- Getting Started: `getting-started/usage.md`
- Pipeline Guide: `../scripts/PIPELINE_README.md`
- Architecture: `architecture/architecture.md`
- Evaluation Playbook: `evaluation/robustness-playbook.md`
- Metrics: `evaluation/metrics.md`
- Security & Release Policy: `safety/security-considerations.md`, `safety/release-checklist.md`
- Results & Reports: `results/report.md`, `results/results-template.md`
- Lambda Runbook: `lambda_runbook.md`
- QA Gates & Stories: `qa/gates/`, `stories/`

## Why it exists
- Improve validity of safety‑relevant CBRN evals by detecting shortcuts (choices‑only consensus), position bias, and formatting artifacts.
- Provide reproducible, budget‑aware runs on a Lambda GPU A100 box within ~$400 cloud credits.
- Ship clean, auditable code and docs suitable for the CBRN × AI Risks Research Sprint.

## Quickstart
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- `make setup`
- `make sample`
- `make run`

## Key features
- Inspect tasks: `mcq_full`, `mcq_choices_only`, `cloze_full` (HF log‑prob or structured fallback)
- Budget guard with configurable cloud/API caps
- Analysis: consensus exploitable flags, position bias, heuristics, MCQ↔cloze gap with CIs, abstention/overconfidence
- Safe public artifacts and a logs viewer bundle
