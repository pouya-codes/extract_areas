#!/usr/bin/env bash
# One-time Azure DevOps bootstrap for the model-worker CI pipeline.
#
# Prerequisites:
#   * az login (subscription Azure credit)
#   * An Azure DevOps organization already created in the browser at
#     https://dev.azure.com (CLI cannot create the first org for a new MSA).
#   * A PAT with scopes: Agent Pools (read/manage), Build (read/execute),
#     Code (read), Service Connections (read/write/query). Export it as
#     AZURE_DEVOPS_EXT_PAT before running.
#
#   export AZURE_DEVOPS_EXT_PAT=...
#   export ADO_ORG=https://dev.azure.com/aimlabdev
#   deploy/azure-devops/bootstrap.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ADO_ORG="${ADO_ORG:-https://dev.azure.com/aimlabdev}"
ADO_PROJECT="${ADO_PROJECT:-cpathportal}"
POOL_NAME="${POOL_NAME:-k3s-host}"
GITHUB_REPO="${GITHUB_REPO:-pouya-codes/extract_areas}"
SUBSCRIPTION_ID="${SUBSCRIPTION_ID:-$(az account show --query id -o tsv)}"
ACR_ID="$(az acr show -n aimlabacr -g rg-aim-inference --query id -o tsv)"

[[ -n "${AZURE_DEVOPS_EXT_PAT:-}" ]] || {
  echo "Set AZURE_DEVOPS_EXT_PAT to a DevOps PAT, and create the org at ${ADO_ORG} first." >&2
  exit 1
}

az devops configure --defaults organization="$ADO_ORG" project="$ADO_PROJECT"

echo "==> 1/5 project ${ADO_PROJECT}"
if ! az devops project show --project "$ADO_PROJECT" &>/dev/null; then
  az devops project create --name "$ADO_PROJECT" --visibility private -o table
else
  echo "    already exists"
fi

echo "==> 2/5 agent pool ${POOL_NAME}"
if ! az pipelines pool list --query "[?name=='${POOL_NAME}'].id" -o tsv | grep -q .; then
  # Agent pools API: create via REST (az pipelines pool has no create).
  az devops invoke --area distributedtask --resource pools \
    --http-method POST --api-version 7.1 \
    --route-parameters "project=$(az devops project show -p "$ADO_PROJECT" --query id -o tsv)" \
    --in-file <(cat <<EOF
{"name":"${POOL_NAME}","autoProvision":true,"autoUpdate":true,"autoSize":true}
EOF
) -o json >/tmp/pool.json || {
    # Fallback: organization-level pool via UI instructions
    echo "    Could not create pool via API; create an agent pool named '${POOL_NAME}' in" >&2
    echo "    ${ADO_ORG}/${ADO_PROJECT}/_settings/agentqueues and re-run." >&2
  }
else
  echo "    already exists"
fi

echo "==> 3/5 Azure Resource Manager service connection aim-inference-acr"
# Prefer the AcrPush SP created during ACR bootstrap. Password goes in env
# (az devops does not accept --azure-rm-service-principal-key on the CLI).
SP_FILE="${HOME}/.config/aimlabacr-push-sp.json"
if [[ -f "$SP_FILE" ]]; then
  if az devops service-endpoint list --query "[?name=='aim-inference-acr'].id" -o tsv | grep -q .; then
    echo "    already exists"
  else
    APP_ID="$(python3 -c "import json;print(json.load(open('$SP_FILE'))['appId'])")"
    TENANT="$(python3 -c "import json;print(json.load(open('$SP_FILE'))['tenant'])")"
    export AZURE_DEVOPS_EXT_AZURE_RM_SERVICE_PRINCIPAL_KEY
    AZURE_DEVOPS_EXT_AZURE_RM_SERVICE_PRINCIPAL_KEY="$(python3 -c "import json;print(json.load(open('$SP_FILE'))['password'])")"
    az devops service-endpoint azurerm create \
      --name aim-inference-acr \
      --azure-rm-service-principal-id "$APP_ID" \
      --azure-rm-subscription-id "$SUBSCRIPTION_ID" \
      --azure-rm-subscription-name "$(az account show --query name -o tsv)" \
      --azure-rm-tenant-id "$TENANT" \
      -o table
    EP_ID="$(az devops service-endpoint list --query "[?name=='aim-inference-acr'].id" -o tsv)"
    az devops service-endpoint update --id "$EP_ID" --enable-for-all true -o none 2>/dev/null || true
    unset AZURE_DEVOPS_EXT_AZURE_RM_SERVICE_PRINCIPAL_KEY
  fi
else
  echo "    missing $SP_FILE; create the ARM service connection in the UI" >&2
fi

echo "==> 4/5 pipeline from azure-pipelines.yml (GitHub)"
# Requires a GitHub service connection named github-extract-areas (create in UI
# with a GitHub PAT that can read pouya-codes/extract_areas).
if az pipelines show --name model-workers &>/dev/null; then
  echo "    pipeline model-workers already exists"
else
  az pipelines create \
    --name model-workers \
    --repository "$GITHUB_REPO" \
    --repository-type github \
    --branch main \
    --yml-path azure-pipelines.yml \
    --skip-first-run true \
    --service-connection github-extract-areas \
    -o table || {
      echo "    Create a GitHub service connection named github-extract-areas, then re-run." >&2
    }
fi

echo "==> 5/5 self-hosted agent on this host"
bash "${REPO_DIR}/deploy/azure-devops/install-agent.sh"

echo "done"
echo "Queue a run: az pipelines run --name model-workers"
