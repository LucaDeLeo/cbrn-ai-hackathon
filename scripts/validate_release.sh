#!/usr/bin/env bash
set -euo pipefail

# Two-tier artifacts policy enforcement
# - Public artifacts must not contain raw item text or per-item exploit labels

echo "[validate_release] Checking artifacts for policy compliance"

# Resolve repo root to use its virtualenv python regardless of cwd
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"

VIOLATIONS=0

# Parse allowlist of safe CSV columns (case-insensitive), e.g. VALIDATE_SAFE_COLUMNS="id,foo,question"
SAFE_COLUMNS_RAW=${VALIDATE_SAFE_COLUMNS:-}
SAFE_COLUMNS=()
if [ -n "$SAFE_COLUMNS_RAW" ]; then
  IFS=',' read -ra TOKS <<< "$SAFE_COLUMNS_RAW"
  for t in "${TOKS[@]}"; do
    # lowercase and trim
    tt=$(echo "$t" | tr '[:upper:]' '[:lower:]' | sed -E 's/^\s+|\s+$//g')
    if [ -n "$tt" ]; then
      SAFE_COLUMNS+=("$tt")
    fi
  done
fi

is_safe_column() {
  local name=$(echo "$1" | tr '[:upper:]' '[:lower:]')
  for col in "${SAFE_COLUMNS[@]}"; do
    if [ "$col" = "$name" ]; then
      return 0
    fi
  done
  return 1
}

check_forbidden() {
  local path="$1"
  local pattern="$2"
  if grep -R -n -E "$pattern" "$path" > /dev/null 2>&1; then
    echo "[validate_release] Forbidden pattern '$pattern' found under $path"
    VIOLATIONS=1
  fi
}

if [ -d artifacts ]; then
  # Disallow raw text keys commonly used for content
  check_forbidden artifacts '\"question\"\s*:'
  check_forbidden artifacts '\"choices\"\s*:'
  # Disallow per-item exploitability labels
  check_forbidden artifacts 'exploitable\":\s*(true|false|[01])'

  # Additionally, scan CSV headers for forbidden columns that could leak sensitive info
  # - Block per-item 'exploitable' labels in any CSV header
  # - Block 'question' and 'choices' columns in public CSVs
  while IFS= read -r -d '' csv; do
    if [ -f "$csv" ]; then
      header=$(head -n 1 "$csv")
      # Match column names case-insensitively at CSV boundaries
      echo "$header" | grep -qiE '(^|,)[[:space:]]*exploitable[[:space:]]*(,|$)' && {
        echo "[validate_release] Forbidden CSV column 'exploitable' found in $csv"; VIOLATIONS=1; }
      if ! is_safe_column "question"; then
        echo "$header" | grep -qiE '(^|,)[[:space:]]*question[[:space:]]*(,|$)' && {
          echo "[validate_release] Forbidden CSV column 'question' found in $csv"; VIOLATIONS=1; }
      fi
      if ! is_safe_column "choices"; then
        echo "$header" | grep -qiE '(^|,)[[:space:]]*choices[[:space:]]*(,|$)' && {
          echo "[validate_release] Forbidden CSV column 'choices' found in $csv"; VIOLATIONS=1; }
      fi
    fi
  done < <(find artifacts -type f -name "*.csv" -print0)
fi

if [ "$VIOLATIONS" -ne 0 ]; then
  echo "[validate_release] Release violates policy. Remove sensitive fields from public artifacts."
  exit 2
fi

echo "[validate_release] Policy content checks passed"

# Automated QA & label hygiene (Module 5)
DATASET=${DATASET:-data/sample_sanitized.jsonl}
OUT_REPORT=${OUT_REPORT:-artifacts/data_quality_report.csv}
MAX_DUP_FRAC=${MAX_DUP_FRAC:-0.05}
MAX_BAD_LABEL_FRAC=${MAX_BAD_LABEL_FRAC:-0.0}
MAX_CHOICE_DUP_FRAC=${MAX_CHOICE_DUP_FRAC:-0.02}
DUP_HAMMING=${DUP_HAMMING:-3}

echo "[validate_release] Running QA rules on ${DATASET}"
mkdir -p "$(dirname "$OUT_REPORT")"

set +e
PY_OUT=$(PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$VENV_PY" -m robustcbrn.qa.rules \
  --dataset "$DATASET" \
  --out-csv "$OUT_REPORT" \
  --dup-hamming "$DUP_HAMMING" \
  --max-dup-frac "$MAX_DUP_FRAC" \
  --max-bad-label-frac "$MAX_BAD_LABEL_FRAC" \
  --max-choice-dup-frac "$MAX_CHOICE_DUP_FRAC" \
  2>&1)
RC=$?
set -e

echo "$PY_OUT" | tail -n 1 || true

if [ $RC -ne 0 ]; then
  echo "[validate_release] QA thresholds failed. See $OUT_REPORT"
  exit 3
fi

echo "[validate_release] OK"
########################################
# Guard: tracked raw data and large files
########################################
# If we're in a git repo, block tracked raw data under data/raw
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  RAW_TRACKED=$(git ls-files -z -- data/raw 2>/dev/null || true)
  if [ -n "$RAW_TRACKED" ]; then
    echo "[validate_release] Tracked files found under data/raw — this is forbidden."
    VIOLATIONS=1
  fi
  # Warn/fail on large dataset-like files (>5MB)
  LARGE_FOUND=0
  while IFS= read -r -d '' f; do
    if [ -f "$f" ]; then
      size=$(wc -c < "$f" 2>/dev/null || echo 0)
      if [ "$size" -gt $((5*1024*1024)) ]; then
        if [ "$LARGE_FOUND" -eq 0 ]; then
          echo "[validate_release] large dataset-like files detected:"; LARGE_FOUND=1; fi
        echo "  - $f (${size}B)"
        VIOLATIONS=1
      fi
    fi
  done < <(git ls-files -z -- '*.jsonl' '*.csv' '*.parquet' || true)
fi
