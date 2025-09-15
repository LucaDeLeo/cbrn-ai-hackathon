# Releasing

This project uses simple, manual releases. Keep artifacts sanitized and reproducible.

## Steps
1. Ensure CI is green (lint + tests).
2. Run an end‑to‑end pipeline on a representative dataset:
   - `make pipeline-full` or `bash scripts/run_pipeline.sh --steps validate,setup,discover,full,aggregate,figures,report,verify`
3. Validate public artifacts:
   - `bash scripts/validate_release.sh`
4. Update `CHANGELOG.md` and bump version in `pyproject.toml` if needed.
5. Tag and push: `git tag -a vX.Y.Z -m "Release vX.Y.Z" && git push --tags`.
6. Create a GitHub release; attach sanitized artifacts if applicable.

## Sanitation policy
- No raw questions/choices or per‑item exploitability labels in public outputs.
- CI runs `scripts/validate_release.sh` to enforce this.
- Datasets under `data/raw/` must never be tracked.
