# Epic 4: Robust Splitting & Reporting

**Goal**: Generate final outputs that the safety community can actually use - validated question subsets, comprehensive reports, and security-conscious documentation.

## Story 4.0: Balance Preservation Validator

**As a** researcher,
**I want** automated checks for distributional balance in the robust subset,
**so that** filtering doesn’t skew topic/difficulty composition.

**Acceptance Criteria:**
1. Compare pre/post distributions across topic/difficulty strata
2. Flag large deviations (configurable thresholds) in the report
3. Offer optional rebalancing (conservative removal per stratum)
4. Include balance summary tables (IDs only, no content)

## Story 4.1: Intelligent Question Filtering

**As a** researcher,
**I want** multiple filtering strategies,
**so that** I can choose the appropriate trade-off for my use case.

**Acceptance Criteria:**
1. Conservative mode (high confidence, fewer removals)
2. Balanced mode (default, 25-35% removal target)
3. Aggressive mode (remove anything suspicious)
4. Preserve topic/difficulty balance in filtered sets
5. Generate both "robust" and "excluded" question lists; publish excluded IDs only for the sanitized calibration subset using the public salt; keep full excluded lists internal
6. Include filtering rationale for each removed question

## Story 4.2: Comprehensive Report Generation

**As a** user,
**I want** detailed analysis reports,
**so that** I can understand and trust the robustification results.

**Acceptance Criteria:**
1. Executive summary with key statistics and confidence intervals
2. Detailed methodology section explaining each technique
3. Results broken down by detection method
4. Comparison table: original vs robust benchmark scores
5. Markdown format for readability, JSON for machine parsing
6. Include timestamp, version, configuration, and reproducibility info
7. Include a Prompts Appendix with exact templates, rendering logic, and versioned hashes; redact any sensitive content per policy (see `docs/prompts_appendix.md`)
8. Provide a Reproducibility Bundle: commands, seeds, package versions, model SHAs, and artifact checksums
9. Release bundle must satisfy the `docs/release_checklist.md`

## Story 4.3: Security & Safety Measures

**As a** safety researcher,
**I want** responsible handling of sensitive content,
**so that** results don't create new info-hazards.

**Acceptance Criteria:**
1. No plaintext question content in public outputs
2. SHA-256 hashes for question identification (two salts: private internal salt; public fixed salt for sanitized subset)
3. Aggregate statistics only in public outputs (no per-item details outside sanitized subset)
4. Security considerations appendix documenting limitations
5. Optional local-only mode for maximum security
6. Redaction helpers for sensitive patterns
7. Artifacts Release Policy implemented with automated redaction checks on publish
8. Public release must pass the `docs/release_checklist.md`

## Story 4.5: Safety‑Relevance Validation (Sanitized Subset)

**As a** safety researcher,
**I want** a small, sanitized conceptual vs procedural subset,
**so that** I can demonstrate that robustified scoring suppresses spurious gains more on procedural-like items without exposing sensitive content.

**Acceptance Criteria:**
1. Construct a 50-item non-actionable, sanitized subset partitioned into conceptual vs procedural categories with labeling protocol documented
2. Run the full pipeline on this subset; report robust vs original score deltas with 95% CIs and topic breakdowns
3. Publish only hashed IDs using a public salt, aggregate metrics, and safe exemplar snippets if applicable
4. Include caveats that results are a proxy for safety-relevance, not a deployment clearance

## Story 4.4: Demo & Documentation Package

**As a** judge,
**I want** clear demonstration of the tool's capabilities,
**so that** I can evaluate its effectiveness and usability.

**Acceptance Criteria:**
1. One-line command that runs full pipeline on sample data
2. README with installation, usage, and methodology
3. Example output on WMDP-Bio subset (sanitized)
4. Performance benchmarks (time, memory, cost)
5. Video script prepared (even if not recorded)
6. Submission report following hackathon template
7. Include link to `docs/release_checklist.md` and `docs/prompts_appendix.md` in repository documentation

## Story 4.6: Release Gate Automation & Checklist

**As a** maintainer,
**I want** an automated release gate,
**so that** public artifacts are safe and consistent with the checklist.

**Acceptance Criteria:**
1. Implement release gate task that runs redaction checks and assembles public bundle
2. Validate ID remapping (private salt → public salt for sanitized subset)
3. Ensure bundle contains bias_report.public.json, calibration IDs, Prompts Appendix, reproducibility bundle
4. Gate fails if any policy violation (plaintext detection, missing files)

## Story 4.7: Audit Log & Run Manifest

**As a** maintainer,
**I want** a comprehensive audit log and manifest,
**so that** every decision and artifact is traceable for judges and future runs.

**Acceptance Criteria:**
1. Audit log includes timestamps, seeds, model families, config hash, and key decisions
2. Run manifest lists artifacts with checksums and paths (internal/public)
3. Manifest included in both internal and public bundles (redacted fields in public)
4. CLI flag `--manifest` writes manifest to results directory
