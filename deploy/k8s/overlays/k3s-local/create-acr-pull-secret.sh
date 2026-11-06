#!/usr/bin/env bash
# Create (or refresh) the Docker registry pull secret that lets worker pods
# pull from aimlabacr.azurecr.io. Credentials come from an ACR token
# (see deploy/k8s/README.md); never commit the password.
#
#   export ACR_USERNAME=k3s-pull
#   export ACR_PASSWORD=...   # from: az acr token create ...
#   deploy/k8s/overlays/k3s-local/create-acr-pull-secret.sh
#
# Or read from files written by the bootstrap:
#   ~/.config/aimlabacr-k3s-pull.{username,password}
set -euo pipefail

NAMESPACE="${NAMESPACE:-aim-inference}"
SERVER="${ACR_LOGIN_SERVER:-aimlabacr.azurecr.io}"
SECRET_NAME="${ACR_PULL_SECRET:-acr-pull}"

if [[ -z "${ACR_USERNAME:-}" && -f "${HOME}/.config/aimlabacr-k3s-pull.username" ]]; then
  ACR_USERNAME="$(cat "${HOME}/.config/aimlabacr-k3s-pull.username")"
fi
if [[ -z "${ACR_PASSWORD:-}" && -f "${HOME}/.config/aimlabacr-k3s-pull.password" ]]; then
  ACR_PASSWORD="$(cat "${HOME}/.config/aimlabacr-k3s-pull.password")"
fi
[[ -n "${ACR_USERNAME:-}" && -n "${ACR_PASSWORD:-}" ]] || {
  echo "ACR_USERNAME and ACR_PASSWORD are required (or ~/.config/aimlabacr-k3s-pull.*)" >&2
  exit 1
}

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$NAMESPACE" create secret docker-registry "$SECRET_NAME" \
  --docker-server="$SERVER" \
  --docker-username="$ACR_USERNAME" \
  --docker-password="$ACR_PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f -
echo "secret/$SECRET_NAME updated in namespace $NAMESPACE (server=$SERVER)"
