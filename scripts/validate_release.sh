#!/usr/bin/env bash
set -euo pipefail

# Two-tier artifacts policy enforcement
# - Public artifacts must not contain raw item text or per-item exploit labels

echo "[validate_release] Checking artifacts for policy compliance"

VIOLATIONS=0

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
      echo "$header" | grep -qiE '(^|,)[[:space:]]*question[[:space:]]*(,|$)' && {
        echo "[validate_release] Forbidden CSV column 'question' found in $csv"; VIOLATIONS=1; }
      echo "$header" | grep -qiE '(^|,)[[:space:]]*choices[[:space:]]*(,|$)' && {
        echo "[validate_release] Forbidden CSV column 'choices' found in $csv"; VIOLATIONS=1; }
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
PY_OUT=$(python -m robustcbrn.qa.rules \
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
