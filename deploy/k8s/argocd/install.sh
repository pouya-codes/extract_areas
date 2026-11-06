#!/usr/bin/env bash
# Install (or refresh) Argo CD + Argo CD Image Updater on this k3s cluster,
# then register the aim-inference Application.
#
#   deploy/k8s/argocd/install.sh
#
# Optional env:
#   ARGOCD_VERSION=v3.6.1
#   IMAGE_UPDATER_VERSION=v1.4.0
#   GITHUB_TOKEN=...   # contents:write PAT for Image Updater git write-back
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# Image Updater v0.16.x uses annotation-based config (matches application.yaml).
# v1.x moved to Helm/CRDs and dropped manifests/install.yaml.
ARGOCD_VERSION="${ARGOCD_VERSION:-v3.6.1}"
IMAGE_UPDATER_VERSION="${IMAGE_UPDATER_VERSION:-v0.16.0}"

echo "==> 1/5 namespace argocd"
kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -

echo "==> 2/5 Argo CD ${ARGOCD_VERSION}"
# Server-side apply: the Applicationset CRD exceeds the client-side annotation size limit.
kubectl apply --server-side --force-conflicts -n argocd -f \
  "https://raw.githubusercontent.com/argoproj/argo-cd/${ARGOCD_VERSION}/manifests/install.yaml"
kubectl -n argocd rollout status deploy/argocd-server --timeout=300s
kubectl -n argocd rollout status deploy/argocd-repo-server --timeout=300s
kubectl -n argocd rollout status statefulset/argocd-application-controller --timeout=300s

echo "==> 3/5 Argo CD Image Updater ${IMAGE_UPDATER_VERSION}"
kubectl apply --server-side --force-conflicts -n argocd -f \
  "https://raw.githubusercontent.com/argoproj-labs/argocd-image-updater/${IMAGE_UPDATER_VERSION}/manifests/install.yaml"
kubectl -n argocd rollout status deploy/argocd-image-updater --timeout=300s

echo "==> 4/5 ACR credentials for Image Updater"
# Reuse the same pull token the workers use; Image Updater lists tags via the registry API.
if [[ -f "${HOME}/.config/aimlabacr-k3s-pull.username" && -f "${HOME}/.config/aimlabacr-k3s-pull.password" ]]; then
  USER="$(cat "${HOME}/.config/aimlabacr-k3s-pull.username")"
  PASS="$(cat "${HOME}/.config/aimlabacr-k3s-pull.password")"
  kubectl -n argocd create secret generic acr-credentials \
    --from-literal=cred="aimlabacr.azurecr.io:${USER}:${PASS}" \
    --dry-run=client -o yaml | kubectl apply -f -
  kubectl -n argocd create configmap argocd-image-updater-config \
    --from-literal=registries.conf="$(cat <<'EOF'
registries:
- name: aimlabacr
  api_url: https://aimlabacr.azurecr.io
  prefix: aimlabacr.azurecr.io
  credentials: secret:argocd/acr-credentials#cred
EOF
)" \
    --dry-run=client -o yaml | kubectl apply -f -
  kubectl -n argocd rollout restart deploy/argocd-image-updater
  kubectl -n argocd rollout status deploy/argocd-image-updater --timeout=180s
else
  echo "    WARN: ~/.config/aimlabacr-k3s-pull.* missing; Image Updater cannot list ACR yet" >&2
fi

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  echo "    configuring git write-back credentials"
  kubectl -n argocd create secret generic git-creds \
    --from-literal=username=git \
    --from-literal=password="${GITHUB_TOKEN}" \
    --dry-run=client -o yaml | kubectl apply -f -
  # Repository credentials for both Argo CD and Image Updater write-back.
  kubectl -n argocd create secret generic repo-extract-areas \
    --from-literal=type=git \
    --from-literal=url=https://github.com/pouya-codes/extract_areas.git \
    --from-literal=password="${GITHUB_TOKEN}" \
    --from-literal=username=git \
    --dry-run=client -o yaml | kubectl apply -f -
  kubectl -n argocd label secret repo-extract-areas \
    argocd.argoproj.io/secret-type=repository --overwrite
fi

echo "==> 5/5 Application"
# Overlay path does not need git submodules (DeepLIIF/HoVer-Net use SSH remotes).
kubectl -n argocd patch configmap argocd-cmd-params-cm --type merge \
  -p '{"data":{"reposerver.enable.git.submodule":"false"}}' >/dev/null
kubectl -n argocd set env deploy/argocd-repo-server ARGOCD_REPO_SERVER_ENABLE_GIT_SUBMODULE=false >/dev/null
kubectl apply -k "${REPO_DIR}/deploy/k8s/argocd"

echo
echo "Argo CD admin password:"
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d
echo
echo "UI (port-forward): kubectl -n argocd port-forward svc/argocd-server 8081:443"
echo "done"
