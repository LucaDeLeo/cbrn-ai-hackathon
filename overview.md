## TL;DR

**RobustCBRN Eval** is a Python 3.10+ toolkit that helps AI‑safety teams *robustify and audit* CBRN‑related multiple‑choice evaluations. It detects shortcut exploitation (e.g., longest‑answer and position biases), runs verified **cloze** scoring with log‑probs, measures robustness under **paraphrase** and **perturbation**, and enforces a strict **two‑tier artifact policy** so public outputs never leak raw question text. It is built around the **Inspect‑AI** evaluation framework, ships a full analysis-and-figures pipeline, and keeps compute predictable (designed to fit on a single Lambda A100 under a \~\$400 budget).

---

## Why this exists (problem statement)

The project is motivated by well‑documented fragilities in current safety‑relevant evals:

* Many MCQ items are **answerable without stems**; the repo cites rates like \~**31.81%** for WMDP‑Bio.
* **Heuristic gaming** is common (e.g., \~**46%** accuracy by always choosing the longest option; **position bias** toward certain choice letters).
* **Format sensitivity** leads to non‑trivial score swings from small formatting tweaks (on the order of \~5%).
* **Contamination** and **reproducibility** are often unmeasured; open pipelines to validate these effects are rare.

**Implication:** Scores may look strong for the wrong reasons; model risk could be understated if “success” comes from artifacts rather than genuine CBRN knowledge.

---

## What it delivers (core capabilities)

1. **Choices‑Only Consensus Screen**

   * Runs MCQ *with options only* (no stem) to catch items that models can “solve” via superficial patterns.
   * Majority/unanimous consensus across models → flags likely exploitables (configurable `k` threshold).

2. **Verified Cloze Scoring (log‑prob path)**

   * Uses Transformers log‑probabilities with length normalization to compare candidate answers as *continuations* of the stem (“cloze” framing).
   * Reduces formatting artifacts relative to letter‑pick MCQ.

3. **Robustness Battery**

   * **Paraphrase consistency:** deterministic, safe rewrites of stems; reports accuracy deltas and CI’d consistency.
   * **Perturbation stability:** whitespace, neutral preambles, and option‑order swaps with label remapping; quantifies fragility.
   * **Heuristics & screens:** position‑bias detection, longest‑answer degradation checks, and optional **AFLite**‑style predictability screens.

4. **Confidence‑Aware Evaluation**

   * Thresholded abstention (e.g., t ∈ {0, 0.5, 0.75, 0.9}) with proportional penalties; global and answered‑only calibration (Brier/ECE) and overconfidence rates; per‑threshold reliability plots.

5. **Analysis & Reporting**

   * Aggregates logs → `artifacts/results/summary.json` + CSVs and minimal **matplotlib** figures (e.g., paraphrase consistency, perturbation fragility).
   * Results template (`docs/results/report.md`) plus a script to auto‑fill tables/figures.

6. **Budget & determinism**

   * Cost/throughput heuristics (`robustcbrn/utils/cost.py`) and simple budget guards.
   * Fixed seeds, reproducible pipelines, incremental caching.

---

## Safety posture (two‑tier release policy)

* **Private (internal)**: full logs and raw text may exist for research QA.
* **Public**: *aggregate only* — no raw questions, no choices, no per‑item exploitability labels.
* Enforced by a CI‑run validator (`scripts/validate_release.sh`) that **blocks** releases if forbidden keys/columns are found.
* Sanitized calibration subsets and **benign policy pairs** are included for safe testing; any sensitive red‑team prompts remain out‑of‑tree.

---

## Architecture & data flow

**Data → Tasks → Logs → Aggregate → Artifacts**

* **Data format**: JSONL items with `id, question, choices, answer, metadata` (see `docs/getting-started/usage.md` and `robustcbrn/data/schemas.py`).
* **Tasks (Inspect‑AI)**

  * `mcq_full`: standard letter‑choice MCQ baseline.
  * `mcq_choices_only`: options‑only screen for shortcut detection.
  * `cloze_full`: verified cloze via Transformers log‑probs (fallback to structured scoring if needed).
  * Robustness tasks: `paraphrase_consistency`, `perturbation_stability`, `aflite_screen`, `ambiguity_audit`, plus **benign\_pairs** dynamic stress.
* **Aggregation** (`robustcbrn/analysis/aggregate.py`): computes accuracy + CIs, consensus‑k exploit flags, MCQ↔Cloze gap, confidence‑aware metrics (abstention, proportional penalties, Brier/ECE), heuristics summary (longest‑answer and position‑bias from safe metadata), benign‑pair compliance/refusal/consistency with CIs, McNemar with BH‑FDR across tasks, power analysis, multi‑reference match rates; writes `summary.json`, `all_results.csv`, and figures.
* **Security** helpers (hash‑based IDs, content redaction) in `robustcbrn/security.py` and docs under `docs/safety/`.

---

## Repo layout (high‑value directories & files)

* `robustcbrn/`

  * `tasks/` — MCQ baseline/choices‑only/cloze + robustness tasks.
  * `analysis/` — aggregation, robustness metrics, figures.
  * `statistical/` — position‑bias analysis, bootstrap helpers.
  * `qa/` — ambiguity/filters/paraphrase/perturb modules.
  * `utils/` — determinism, logging, cost heuristics, validation.
  * `cli/` — CLI entry points for loading, analysis, validation.
* `docs/`

  * `overview/brief.md` — the project brief (goals, metrics, roadmap).
  * `architecture/architecture.md` — module overview and dataflow.
  * `evaluation/` — **robustness playbook**, prompts appendix, annotation guide.
  * `safety/` — security considerations & release checklist.
  * `qa/gates/` & `stories/` — lightweight “acceptance gates” and story cards documenting what’s done and why.
  * `results/` — report template + figures references.
* `data/` — **sanitized** samples and benign‑pairs toy set.
* `scripts/` — setup, run, aggregate, validate‑release, etc.
* `tests/` — broad unit/integration coverage for tasks, stats, CLIs.
* CI: `.github/workflows/ci.yml` runs ruff + pytest, and validates release artifacts; optional scheduled smoke for the HF cloze path.

---

## How you run it (quickstart)

1. **Setup (uses `uv`):**

```bash
make setup
```

2. **Sanity check on toy data:**

```bash
make sample
```

3. **Run an evaluation set (subset configurable):**

```bash
make run DATASET=/path/to/eval.jsonl SUBSET=512
```

4. **Aggregate & plot:**

```bash
make aggregate
```

5. **(Optional) Fill the sprint report template:**

```bash
make fill-report
```

6. **Validate public artifacts before you share anything:**

```bash
bash scripts/validate_release.sh
```

> Default runs prefer **local HF models** (e.g., Llama‑3‑8B, Mistral‑7B, Qwen‑2.5‑7B) and target single‑GPU (A100 40/80GB). Seeds and caches make results repeatable.

New: unified pipeline (optional)

```bash
make pipeline           # end‑to‑end orchestrator
make pipeline-validate  # platform checks only
make pipeline-full      # full suite + figures + report + verify
```

## CLI (shortcuts)

- Load preview

```bash
python -m robustcbrn.cli.main load data/sample.jsonl --config configs/default.json
```

- Analyze dataset (longest‑answer baseline + optional tests)

```bash
python -m robustcbrn.cli.main analyze \
  --input data/dataset.jsonl \
  --output artifacts/analysis.json \
  --verbose --max-items 50 \
  --tests position_bias \
  --robust-input data/robust.jsonl \
  --stratify-by data/strata.json
```

- Position bias analysis

```bash
python -m robustcbrn.cli.main position-bias \
  --input data/dataset.jsonl \
  --output artifacts/position_bias.json \
  --verbose
```

- Heuristic degradation (original vs robust)

```bash
python -m robustcbrn.cli.main heuristic-degradation \
  --original data/original.jsonl \
  --robust data/robust.jsonl \
  --output artifacts/degradation.json
```

- Dataset validator (JSONL)

```bash
python -m robustcbrn.cli.validate_dataset --schema both data/eval.jsonl
```

- Aggregator (confidence‑aware + calibration)

```bash
.venv/bin/python -m robustcbrn.analysis.aggregate \
  --logs logs --out artifacts/results \
  --k 2 --confidence-thresholds 0,0.5,0.75,0.9
```

---

## What comes out (key artifacts)

* `artifacts/results/summary.json` — consolidated metrics (accuracy + CIs, consensus‑k exploitables, MCQ↔Cloze gap, robustness, confidence‑aware metrics + calibration).
* `artifacts/results/all_results.csv` — per‑row merged results across tasks/models.
* `artifacts/figs/` — bar/paired‑Δ plots (e.g., paraphrase consistency, perturbation fragility, MCQ↔Cloze), plus calibration plots `calibration_t*.png` and confidence histograms `confidence_hist_t*.png`.
* `docs/results/report.md` — sprint report you populate from artifacts.
* **No** raw questions/choices/per‑item labels in public outputs (CI enforces).

---

## Users & value

* **Primary:** Academic AI‑safety researchers (labs/independent), who need *reproducible*, transparent, and safer eval pipelines on constrained budgets.
* **Secondary:** Evaluation orgs (e.g., METR‑like), who want auditable batteries that quantify and mitigate artifact‑driven score inflation.

**Value props:**

* Clear *evidence* of artifacts; practical *remediations* (robust subsets / alternative formats); repeatable runs; safety‑aware release discipline.

---

## Goals & success metrics (from the brief)

* **Activity:** 3+ major CBRN benchmarks processed; 2–3 model families; <4h per 1,000 items.
* **Outputs:** 25–35% items flagged as compromised; 65–75% robust subset retained; 95% CIs throughout; cloze + MCQ variants generated.
* **Outcomes:** Drop score variance to <2%; reveal 15–25% lower “true” capability after robustification; early adoption signals (stars/forks/collabs).
* **Validation:** 70–80% agreement with a **50‑item, human‑labeled sanitized** calibration subset; strong degradation of longest‑answer heuristic; compute within budget.

---

## MVP scope & what’s explicitly out‑of‑scope

**In MVP:**

* Choices‑only consensus across up to three local models.
* Verified cloze scoring via HF log‑probs.
* Robustness battery (paraphrase, perturbation, AFLite, position bias).
* Confidence‑aware scoring and calibration metrics.
* Multi‑level caching, deterministic runs, and full audit trail.
* Conservative robust split with balance checks.

**Not in MVP:**

* A polished GUI (CLI first), multilingual support, training/fine‑tuning, external service integrations, real‑time API serving.

---

## Risks & open questions (tracked in docs)

* **Compute fit & time:** Manage GPU memory (batching), keep under time/budget targets.
* **Consensus tuning:** Choosing `k` and number of models for best precision/recall.
* **Contamination detection:** Methods without train‑data access.
* **Validation without raw text:** Relies on small, sanitized human‑labeled calibration and metadata‑only heuristics.

---

## Project status, QA gates & CI

* The repository includes **BMAD‑style QA gates** per story (e.g., *Cloze HF Path*, *Validation & Safety*, *CI polish*) marked **PASS** with reviewer notes and next‑steps.
* CI runs lint + tests; a scheduled/manual “HF cloze smoke” job ensures the log‑prob path doesn’t regress.
* A “next steps” checklist calls for a full end‑to‑end run on a target dataset, report fill‑in, figures generation, and final release validation.

---

## Tech & licensing

* **Stack:** Python 3.10+, Inspect‑AI, PyTorch/Transformers, NumPy/Pandas/SciPy, Matplotlib, Ruff, Pytest; `uv` for fast envs.
* **Packaging:** `pyproject.toml` with `robustcbrn-eval` as the package name.
* **License:** MIT.
