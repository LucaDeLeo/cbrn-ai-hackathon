# Epic 5: Release & Submission

**Goal**: Package and submit a safe, reproducible, and convincing demo aligned with hackathon criteria.

## Story 5.1: Public Bundle Assembly

**Acceptance Criteria:**
1. Assemble sanitized public bundle (bias report, calibration IDs, reproducibility bundle, Prompts Appendix)
2. Include checksums and a MANIFEST with file hashes
3. Bundle passes `docs/release_checklist.md`

## Story 5.2: Reproducibility Bundle & Runbook

**Acceptance Criteria:**
1. Provide runbook with exact commands for sample and full runs
2. Include seeds, env vars, determinism flags, model SHAs/commits
3. Include measured runtime and cost estimates for Lambda GPU

## Story 5.3: Demo Script & Video Outline

**Acceptance Criteria:**
1. 3–5 minute script demonstrating pipeline, key metrics, and safety posture
2. Screenshots or terminal recordings for fallback
3. Link script from README and PRD

## Story 5.4: Judges Quick-Verify Guide

**Acceptance Criteria:**
1. One-page guide instructing judges how to run the sample demo, verify diversity check, and confirm public bundle contents
2. Time-to-verify target: <15 minutes
3. Links to Prompts Appendix and Release Checklist
