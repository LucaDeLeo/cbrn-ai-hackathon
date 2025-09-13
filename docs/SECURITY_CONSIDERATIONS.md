# Security Considerations

Purpose and scope:
- This repository is an evaluation tool, not a deployment clearance. It must not be used to generate or disseminate hazardous content.

Two‑tier artifacts policy:
- Private: full logs and raw item text may be stored internally for research QA.
- Public: only aggregate metrics and anonymized plots. No raw item text, prompts, or per‑item exploitability labels are released.

Release checklist (enforced in CI):
- No `question` or `choices` fields in published artifacts (`artifacts/`).
- No per‑item `exploitable` flags in public outputs.
- Seeds, versions, config pinned in report.
- CI lint + tests + validator pass.

Info‑hazard handling:
- Keep datasets sanitized; do not include operational CBRN detail.
- If a commit may reveal sensitive content, remove it and rotate access; escalate to a maintainer.
- Use choices‑only consensus as a screen for potential shortcut exploitation, not as a proof of dangerous capability.

Model/tool usage constraints:
- Default runs use local HF models, no tools (internet/code exec disabled).
- API backends are opt‑in and additionally capped by `API_BUDGET_USD`.

