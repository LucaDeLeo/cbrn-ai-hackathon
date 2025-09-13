#!/usr/bin/env bash
# Configure GitHub branch protection rules for `main` to satisfy Story 1.0 AC3
# Requires: GitHub CLI (`gh`) authenticated with repo:admin scope

set -euo pipefail

echo "🔐 Configuring branch protection for 'main'..."

# Check dependencies
if ! command -v gh >/dev/null 2>&1; then
  echo "❌ GitHub CLI 'gh' not found. Install from https://cli.github.com/"
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "❌ 'gh' is not authenticated. Run: gh auth login"
  exit 1
fi

# Derive owner/repo from git remote
remote_url=$(git remote get-url origin 2>/dev/null || true)
if [[ -z "${remote_url}" ]]; then
  echo "❌ Could not determine git remote 'origin'. Set a remote and retry."
  exit 1
fi

# Normalize remote URL to https form
if [[ "${remote_url}" =~ ^git@github.com:(.*)/(.*)\.git$ ]]; then
  owner="${BASH_REMATCH[1]}"; repo="${BASH_REMATCH[2]}"
elif [[ "${remote_url}" =~ ^https://github.com/(.*)/(.*)\.git$ ]]; then
  owner="${BASH_REMATCH[1]}"; repo="${BASH_REMATCH[2]}"
elif [[ "${remote_url}" =~ ^https://github.com/(.*)/(.*)$ ]]; then
  owner="${BASH_REMATCH[1]}"; repo="${BASH_REMATCH[2]}"
else
  echo "❌ Unrecognized remote URL: ${remote_url}"
  exit 1
fi

echo "📦 Repository: ${owner}/${repo}"

# Required status checks contexts (customize as your CI job names)
contexts=("ci")

# Build JSON payload
payload=$(cat <<JSON
{
  "required_status_checks": {
    "strict": true,
    "contexts": [$(printf '"%s",' "${contexts[@]}" | sed 's/,$//')]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1
  },
  "restrictions": null
}
JSON
)

echo "🛠️ Applying branch protection..."
gh api \
  -X PUT \
  -H "Accept: application/vnd.github+json" \
  "repos/${owner}/${repo}/branches/main/protection" \
  -f required_status_checks.strict=true \
  $(for c in "${contexts[@]}"; do printf -- "-f required_status_checks.contexts[]=%s " "$c"; done) \
  -f enforce_admins=true \
  -f required_pull_request_reviews.dismiss_stale_reviews=true \
  -f required_pull_request_reviews.require_code_owner_reviews=false \
  -f required_pull_request_reviews.required_approving_review_count=1 \
  -F restrictions=

echo "📖 Fetching and saving protection settings to docs/qa/branch-protection-main.json..."
mkdir -p docs/qa
if command -v jq >/dev/null 2>&1; then
  gh api -H "Accept: application/vnd.github+json" \
    "repos/${owner}/${repo}/branches/main/protection" | jq . > docs/qa/branch-protection-main.json
else
  gh api -H "Accept: application/vnd.github+json" \
    "repos/${owner}/${repo}/branches/main/protection" | tee docs/qa/branch-protection-main.json >/dev/null
fi

echo "✅ Branch protection configured. Review docs/qa/branch-protection-main.json for evidence."
