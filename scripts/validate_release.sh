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
fi

if [ "$VIOLATIONS" -ne 0 ]; then
  echo "[validate_release] Release violates policy. Remove sensitive fields from public artifacts."
  exit 2
fi

echo "[validate_release] OK"
