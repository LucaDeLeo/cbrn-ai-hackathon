#!/usr/bin/env bash
set -euo pipefail

LOGS_DIR=${1:-logs}
OUT_DIR=${2:-site/logs}
mkdir -p "$OUT_DIR"

echo "[publish_logs] Bundling Inspect logs viewer"
.venv/bin/inspect view bundle "$LOGS_DIR" --output "$OUT_DIR"

echo "[publish_logs] Bundle created at $OUT_DIR"
echo "Serve locally: python -m http.server --directory site 8000"
echo "Or publish via GitHub Pages by pushing 'site' to 'gh-pages' branch."
