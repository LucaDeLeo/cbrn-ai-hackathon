# GitHub Branch Protection Setup

Purpose: Define and validate branch protection rules for the `main` branch to satisfy Story 1.0 AC3.

## Required Settings (Checklist)

- [ ] Require a pull request before merging
  - [ ] Require at least 1 approving review
  - [ ] Dismiss stale pull request approvals when new commits are pushed
- [ ] Require status checks to pass before merging
  - [ ] Require branches to be up to date before merging
  - [ ] Required checks: `ci` (or your workflow job names)
- [ ] Include administrators
- [ ] Restrict who can push to matching branches (optional; recommended: maintainers/admins only)
- [ ] Require signed commits (optional)

## Validation Steps

1. Navigate: Repository → Settings → Branches → Branch protection rules → Edit `main`.
2. Apply the Required Settings checklist above.
3. Save changes and capture evidence:
   - Screenshot of the rule configuration, or
   - Link to a policy-as-code file (if using codeowners/branch rules via API), or
   - GitHub CLI output (see below).
4. Record validation in Story 1.0 → Dev Agent Record → Completion Notes (with date/time and link/screenshot reference).

### GitHub CLI (Optional)

You can also view current rules using the GitHub CLI and the REST API:

```
gh api \
  -H "Accept: application/vnd.github+json" \
  repos/:owner/:repo/branches/main/protection

# Example for this repository:
gh api \
  -H "Accept: application/vnd.github+json" \
  repos/LucaDeLeo/cbrn-ai-hackathon/branches/main/protection | tee docs/qa/branch-protection-main.json
```

Replace `:owner/:repo` accordingly. Confirm fields:
- `required_pull_request_reviews`
- `required_status_checks`
- `enforce_admins`
- `restrictions`

## Evidence

- Date validated: <YYYY-MM-DD>
- Evidence link/screenshot: <URL or path>
- Notes: <context>

Suggested evidence artifact (JSON): `docs/qa/branch-protection-main.json`

## References

- Epic 1 → Story 1.0 (AC3)
- This repository’s CI workflow job names to set as required checks
