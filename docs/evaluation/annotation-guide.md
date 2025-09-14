<!-- canonical path: docs/evaluation/annotation-guide.md -->
Ambiguity/Unanswerable Adjudication Guide

Scope
- This guide specifies a small, safety-aware, human-in-the-loop process to validate the Ambiguity/Unanswerable module on a sanitized calibration subset only.
- Do not annotate raw or hazardous items. Only use the sanitized set distributed under `data/sample_sanitized.jsonl` or an equivalent pre-approved subset.

Definitions
- clean: The item has a single resolvable correct answer. Reasonable readers agree.
- ambiguous: Multiple choices are plausibly correct due to wording, duplicates, or contradictions in options.
- unanswerable: The stem or options do not allow a unique answer (e.g., “all of the above/none of the above” without clarity, boolean-like options that require the stem, missing information).

Annotator Protocol (3 annotators)
- Materials: CSV with columns `id,choices,metadata` where choices are scrubbed of sensitive text. Provide the full stem only if approved as sanitized; otherwise, omit the stem and use metadata.
- Each annotator independently labels each item with one of {clean, ambiguous, unanswerable} and may add comma-separated reason codes (e.g., duplicate_choices, near_duplicate, contradictory_options, numeric_too_close, boolean_like_requires_stem).
- Tie-breaks: Use majority vote. If all three disagree, escalate for adjudication by a coordinator who records the final label and the adjudication rationale.

Output Schema
- Per-item fields (private workspace only): `id,label,reason_codes,annotator_id,notes`
- Public aggregates only: counts and rates, no stems or free-text notes.

Quality Targets
- Agreement: target ≥70–80% agreement with the module’s heuristic labels on the sanitized calibration set.
- Drift check: spot check a random 10% sample in future sprints.

Safety Notes
- Do not include stems or free-form rationales in public artifacts. Public reports must contain only item IDs and aggregate counts.
- Follow the project’s info-hazard guidance in `hackathon-context/mainpage.md`.

Operational Tips
- Prefer structured reason codes over free text when possible.
- If in doubt between ambiguous vs unanswerable: choose unanswerable when the item cannot be resolved without information not present in the options.
