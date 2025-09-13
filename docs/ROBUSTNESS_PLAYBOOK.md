# Robustness Playbook

This playbook documents the safe, minimal procedures we use to stress-test models in RobustCBRN Eval without introducing hazardous content into public artifacts. It complements docs/brief.md and hackathon-context/mainpage.md.

## Dynamic Stress (Benign Pairs)

- Purpose: Check whether models both refuse clearly prohibited requests and comply with clearly benign safety-policy queries.
- Dataset: `data/benign_pairs_sanitized.jsonl` contains tiny, non‑operational pairs. Each record provides:
  - `prompt_safe`: a benign question (e.g., safe disposal policies, recycling guidance)
  - `prompt_unsafe`: a generic, non‑specific prohibited request (no operational details)
- Task: `robustcbrn.tasks.benign_policy_pairs:benign_policy_pairs` converts each pair into two MCQ items with choices `["Refuse", "Comply"]` and logs Inspect‑native results with `variant` set to `safe` or `unsafe`.
- Metrics (informal):
  - Compliance on `safe` variants (higher is better)
  - Refusal on `unsafe` variants (higher is better)
  - Per‑pair consistency (both correct within a pair)
- Safety: No hazardous strings are included; public logs contain IDs, predictions, and safe metadata only.

## Red‑Team Protocol (Private Pathway)

This repository does not include sensitive prompts or outputs. For post‑sprint private work with qualified reviewers, the following protocol applies:

- Roles: lead reviewer, subject‑matter reviewer, safety officer.
- Checklist:
  - Define scope and exit criteria; set API/time budgets up front.
  - Work from hashed IDs and metadata only; never commit stems.
  - Start with sanitized calibrations; escalate to private review for any sensitive content.
  - Log only minimal, non‑operational descriptors (e.g., codes, flags); avoid raw text.
  - Apply two‑tier release review per `scripts/validate_release.sh` and hackathon info‑hazard guidance.
- Escalation: Any content that could plausibly be dual‑use or operational must be stripped from artifacts and handled off‑repo under an approved private review channel.

## Running the Benign Pair Task

- Ensure `inspect-ai` is installed (via project setup) and run:
  - `inspect eval robustcbrn.tasks.benign_policy_pairs:benign_policy_pairs --arg dataset_path=data/benign_pairs_sanitized.jsonl --model <provider/model> --log-dir logs`
- Or use the convenience script `scripts/run_robustness_suite.sh` (see script header for env vars).

## Release Safety

- Public artifacts must not contain raw stems, choices, or per‑item exploit labels.
- `scripts/validate_release.sh` enforces the two‑tier policy and scans for forbidden fields. Only IDs, high‑level aggregates, and safe metadata are published.

