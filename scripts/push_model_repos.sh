#!/usr/bin/env bash
# Create pouya-codes/aim-inference-worker-* on GitHub (if missing) and push
# the seeded local repos. Requires GITHUB_TOKEN with repo scope.
#
# Until a GitHub PAT is available, the same trees are published as Azure Repos
# under aimlabdev/cpathportal (aim-inference-worker-{pdac,hovernet,ec,deepliif,template})
# and pipelines worker-* already point at those.
#
#   GITHUB_TOKEN=ghp_... scripts/push_model_repos.sh [/path/to/seeded]
set -euo pipefail

OUT="${1:-${HOME}/code/aim-inference-workers}"
ORG="${GITHUB_ORG:-pouya-codes}"
API="https://api.github.com"

[[ -n "${GITHUB_TOKEN:-}" ]] || {
  echo "Set GITHUB_TOKEN to a GitHub PAT with 'repo' scope" >&2
  exit 1
}

auth=(-H "Authorization: Bearer ${GITHUB_TOKEN}" -H "Accept: application/vnd.github+json")

create_repo() {
  local name="$1" is_template="${2:-false}"
  local code
  code=$(curl -s -o /tmp/gh-repo.json -w "%{http_code}" "${auth[@]}" "${API}/repos/${ORG}/${name}")
  if [[ "$code" == "200" ]]; then
    echo "    ${ORG}/${name} exists"
    return 0
  fi
  echo "    creating ${ORG}/${name}"
  # is_template is bash true/false — map to Python bool (not bare false/true names).
  local body
  body=$(IS_TEMPLATE="$is_template" NAME="$name" python3 - <<'PY'
import json, os
print(json.dumps({
    "name": os.environ["NAME"],
    "private": True,
    "description": "AIM inference worker",
    "auto_init": False,
    "is_template": os.environ["IS_TEMPLATE"].lower() in ("1", "true", "yes"),
}))
PY
)
  curl -sS "${auth[@]}" -X POST "${API}/user/repos" \
    -H "Content-Type: application/json" \
    -d "$body" | python3 -c "import json,sys; d=json.load(sys.stdin); print('    id', d.get('full_name') or d)"
}

# The token goes only into the one-off push URL, never into .git/config.
push_repo() {
  local repo="$1" dir="${OUT}/$1"
  git -C "$dir" remote remove origin 2>/dev/null || true
  git -C "$dir" remote add origin "https://github.com/${ORG}/${repo}.git"
  git -C "$dir" push "https://x-access-token:${GITHUB_TOKEN}@github.com/${ORG}/${repo}.git" HEAD:main
}

for name in pdac hovernet ec deepliif; do
  repo="aim-inference-worker-${name}"
  echo "==> ${repo}"
  create_repo "$repo" false
  push_repo "$repo"
done

echo "==> aim-inference-worker-template"
create_repo aim-inference-worker-template true
push_repo aim-inference-worker-template

echo "done"
