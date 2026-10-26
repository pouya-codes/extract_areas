#!/usr/bin/env bash
# Create the aim-inference namespace and the inference-secrets Secret from
# deploy/compose.env (HF_TOKEN, GATEWAY_API_TOKEN), so the same tokens are
# used whether the stack runs under compose or k3s.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ENV_FILE="${1:-$REPO_DIR/deploy/compose.env}"
[[ -f "$ENV_FILE" ]] || { echo "env file not found: $ENV_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a
kubectl create namespace aim-inference --dry-run=client -o yaml | kubectl apply -f -
kubectl -n aim-inference create secret generic inference-secrets \
    --from-literal=HF_TOKEN="${HF_TOKEN:-}" \
    --from-literal=GATEWAY_API_TOKEN="${GATEWAY_API_TOKEN:-}" \
    --dry-run=client -o yaml | kubectl apply -f -
echo "secret inference-secrets updated in namespace aim-inference"
