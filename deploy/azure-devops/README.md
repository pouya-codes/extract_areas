# Azure DevOps bootstrap (model workers)

The pipeline definition lives at the repo root: [`azure-pipelines.yml`](../../azure-pipelines.yml).

## Already provisioned on Azure

| Resource | Name |
|----------|------|
| Resource group | `rg-aim-inference` (eastus2) |
| Container Registry | `aimlabacr` → `aimlabacr.azurecr.io` |
| AcrPush service principal | saved at `~/.config/aimlabacr-push-sp.json` (host only) |
| ACR pull token | `k3s-pull` → `~/.config/aimlabacr-k3s-pull.{username,password}` |

## Defaults on this host

| Setting | Value |
|---------|-------|
| Organization | `https://dev.azure.com/aimlabdev` |
| Project | `cpathportal` |
| Agent pool | `k3s-host` |

## You must do once in the browser

1. Create a PAT with: Agent Pools (Read & manage), Build (Read & execute), Code (Read), Service Connections (Read, query & manage).
2. In project **`cpathportal`**, add a **GitHub** service connection named **`github-extract-areas`** that can read `pouya-codes/extract_areas`.

```bash
export AZURE_DEVOPS_EXT_PAT=...
export ADO_ORG=https://dev.azure.com/aimlabdev
# ADO_PROJECT defaults to cpathportal
deploy/azure-devops/bootstrap.sh
```

That creates the project (if needed), ARM service connection `aim-inference-acr`, pipeline `model-workers` (since renamed `worker-base` in ADO), agent pool `k3s-host`, and installs the self-hosted agent as a systemd user service on this host.

## Queue a build

```bash
az pipelines run --id 1 --org "$ADO_ORG" --project cpathportal
```

Images land at:

```
aimlabacr.azurecr.io/aimlab/aim-inference-worker-{pdac,hovernet,ec,deepliif}:<sha>
aimlabacr.azurecr.io/aimlab/aim-inference-worker-{pdac,hovernet,ec,deepliif}:main
```
