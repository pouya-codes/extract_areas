# Azure DevOps bootstrap (model workers)

The pipeline definition lives at the repo root: [`azure-pipelines.yml`](../../azure-pipelines.yml).

## Already provisioned on Azure

| Resource | Name |
|----------|------|
| Resource group | `rg-aim-inference` (eastus2) |
| Container Registry | `aimlabacr` → `aimlabacr.azurecr.io` |
| AcrPush service principal | saved at `~/.config/aimlabacr-push-sp.json` (host only) |
| ACR pull token | `k3s-pull` → `~/.config/aimlabacr-k3s-pull.{username,password}` |

## You must do once in the browser

1. Open https://dev.azure.com and create an organization (suggested name: **`aimlab`**). Azure CLI cannot create the first org for a Microsoft account that has never used DevOps.
2. Create a PAT with: Agent Pools (Read & manage), Build (Read & execute), Code (Read), Service Connections (Read, query & manage).
3. In the new project (created by the script below, or manually as **`extract-areas`**), add a **GitHub** service connection named **`github-extract-areas`** that can read `pouya-codes/extract_areas`.

```bash
export AZURE_DEVOPS_EXT_PAT=...
export ADO_ORG=https://dev.azure.com/aimlab   # or your org URL
deploy/azure-devops/bootstrap.sh
```

That creates the project, ARM service connection `aim-inference-acr`, pipeline `model-workers`, agent pool `k3s-host`, and installs the self-hosted agent as a systemd user service on this host.

## Queue a build

```bash
az pipelines run --name model-workers --org "$ADO_ORG" --project extract-areas
```

Images land at:

```
aimlabacr.azurecr.io/aimlab/aim-inference-worker-{pdac,hovernet,ec,deepliif}:<sha>
aimlabacr.azurecr.io/aimlab/aim-inference-worker-{pdac,hovernet,ec,deepliif}:main
```
