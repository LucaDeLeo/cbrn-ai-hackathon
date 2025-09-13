# Artifacts Release Policy

To reconcile transparency with safety, we adopt a two-tier artifact model:

| Artifact Class | Internal (private) | Public (release) |
| --- | --- | --- |
| Raw question text | Allowed (access-controlled) | Not allowed |
| Model outputs/logs | Allowed (full, DEBUG) | Aggregated statistics only |
| Item IDs | Allowed with private salt | Allowed for sanitized subset with public salt |
| Prompt templates | Exact, rendered per item | Exact templates and rendering logic; no raw hazardous content |
| Excluded item lists | Full lists allowed | Only sanitized subset IDs (public salt) |
| Reports | Full detail | Redacted aggregate metrics with CIs |

Automated redaction checks must pass before publishing public artifacts. The Prompts Appendix includes all prompt templates and rendering logic with versioned hashes.
