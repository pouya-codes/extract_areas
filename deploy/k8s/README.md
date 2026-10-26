# Kubernetes manifests

Kustomize layout:

| Path | Purpose |
|------|---------|
| `base/` | Namespace, Redis, gateway (Deployment/Service/HPA/PDB), tissue-api, one Deployment + KEDA `ScaledObject` per model, shared config |
| `overlays/prod-gpu/` | GPU node pool selectors, NFS volumes, registry image tags, higher scaling ceilings |
| `overlays/k3s-local/` | The current single-node k3s host: hostPath volumes, HF cache, NodePort gateway (30080) and tissue-api (30800) |
| `overlays/dev-cpu/` | No GPUs: gateway + Redis + stub worker for smoke tests |
| `addons/` | Pinned KEDA and NVIDIA device plugin manifests (cluster-admin) |

## Local k3s, step by step

```bash
make images                                   # build images with Docker
make k8s-addons                               # KEDA + device plugin (once; cluster-admin)
make k8s-import-images                        # docker save | k3s ctr images import
deploy/k8s/overlays/k3s-local/create-secret.sh  # namespace + tokens from deploy/compose.env
make k8s-apply-local                          # kubectl apply -k overlays/k3s-local
kubectl -n aim-inference get deploy,scaledobject,pods
curl -H "Authorization: Bearer $GATEWAY_API_TOKEN" http://127.0.0.1:30080/v1/models
```

Stop the compose stack first (`make compose-down`): both would claim the GPUs.

## Any other cluster

```bash
kubectl kustomize deploy/k8s/overlays/prod-gpu | less      # inspect
kubectl apply -k deploy/k8s/overlays/prod-gpu
```

Prerequisites: the NVIDIA device plugin (so pods can request
`nvidia.com/gpu`) and KEDA (`ScaledObject`); both are in `addons/`.
