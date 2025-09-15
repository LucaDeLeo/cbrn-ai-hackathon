# Data Management

This directory contains dataset management for RobustCBRN Eval.

## Directory Structure

```
data/
├── README.md                 # This file
├── registry.yaml            # Dataset registry with URLs and checksums
├── sample_sanitized.jsonl   # Small sanitized sample for testing
├── benign_pairs_sanitized.jsonl  # Benign pairs for testing
├── raw/                     # Downloaded raw datasets (gitignored)
└── processed/               # Processed datasets in JSONL format (gitignored)
```

## Usage

### List Available Datasets

```bash
make data-list
# or
uv run python scripts/fetch_data.py --list
```

### Download a Dataset

```bash
make data DATASET=wmdp_bio
# or
uv run python scripts/fetch_data.py wmdp_bio
```

### Run Evaluation on Downloaded Dataset

```bash
# After downloading and processing
make run DATASET=data/processed/wmdp_bio/eval.jsonl SUBSET=512
```

### New: MMLU-Pro and HarmBench adapters

- Add direct file URLs in `data/registry.yaml` for:
  - `mmlu_pro` (HF TIGER-Lab/MMLU-Pro; jsonl/parquet/csv)
  - `harmbench` (JSON/JSONL/CSV from the repo)

Then fetch and convert:

```bash
make data DATASET=mmlu_pro     # writes data/processed/mmlu_pro/eval.jsonl
make data DATASET=harmbench    # writes data/processed/harmbench/eval.jsonl
```

Run a 20% sample on each (example with one local HF model):

```bash
export MODELS="mistralai/Mistral-7B-Instruct-v0.3" SEEDS=123 DEVICE=cuda DTYPE=bfloat16 CLOZE_MODE=fallback
for NAME in mmlu_pro harmbench wmdp_chem; do
  DS="data/processed/$NAME/eval.jsonl"; N=$(wc -l < "$DS"); SUBSET=$(( (N*20 + 99)/100 ))
  LOGS_DIR="logs/$NAME" RESULTS_DIR="artifacts/results/$NAME"
  DATASET="$DS" SUBSET="$SUBSET" LOGS_DIR="$LOGS_DIR" RESULTS_DIR="$RESULTS_DIR" \
    bash scripts/run_evalset.sh
done
```

## Adding New Datasets

1. **Update registry.yaml** with dataset information:
   ```yaml
   datasets:
     your_dataset:
       url: "https://example.com/dataset.csv"
       sha256: "PLACEHOLDER_SHA256_TO_BE_COMPUTED"  # Will be computed on first download
       license: "License type"
       unpack: none  # or tar.gz, zip
       process:
         adapter: robustcbrn.data.adapters.your_adapter:convert_function
       safe_to_publish: false
       notes: "Description of dataset"
   ```

2. **Create adapter** (if needed) in `robustcbrn/data/adapters/`:
   - Adapter should convert raw format to JSONL schema
   - Expected schema: `{id, question, choices, answer, metadata}`

3. **Test the download**:
   ```bash
   uv run python scripts/fetch_data.py your_dataset
   ```

## Dataset Schema

All processed datasets follow this JSONL schema:

```json
{
  "id": "unique_identifier",
  "question": "Question text",
  "choices": ["Option A", "Option B", "Option C", "Option D"],
  "answer": "B",  // Letter format (A, B, C, D)
  "metadata": {
    "domain": "optional_domain",
    "category": "optional_category"
  }
}
```

## Security & Safety

- **Never commit raw datasets** to the repository
- **Public releases** must not contain raw question text or choices
- **Use checksums** to ensure data integrity
- **Document licenses** for all datasets

## Authentication

For gated datasets (e.g., HuggingFace):

```bash
export HF_TOKEN=your_token_here
make data DATASET=gated_dataset
```

## Troubleshooting

### Checksum Mismatch
- Dataset may have been updated
- Verify the source and update registry.yaml

### Download Fails
- Check network connection
- For HuggingFace: ensure HF_TOKEN is set
- Verify URL in registry.yaml

### Adapter Errors
- Check CSV/JSON format matches expectations
- Review adapter code in robustcbrn/data/adapters/
- Test with small subset first

## License Information

Each dataset in registry.yaml includes license information. Ensure you comply with all dataset licenses and terms of use before using them for evaluation.
