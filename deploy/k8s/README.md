# Kubernetes manifests

Kustomize layout:

| Path | Purpose |
|------|---------|
| `base/` | Namespace, Redis, gateway (Deployment/Service/HPA/PDB), one Deployment + KEDA `ScaledObject` per model, shared config |
| `overlays/prod-gpu/` | GPU node pool selectors, NFS volumes, registry image tags, higher scaling ceilings |
| `overlays/k3s-local/` | The current single-node k3s host: hostPath volumes, NodePort gateway |
| `overlays/dev-cpu/` | No GPUs: gateway + Redis + stub worker for smoke tests |

```bash
kubectl kustomize deploy/k8s/overlays/prod-gpu | less      # inspect
kubectl apply -k deploy/k8s/overlays/prod-gpu
kubectl -n aim-inference get scaledobjects,hpa,deploy
```

Cluster prerequisites: the NVIDIA device plugin (so pods can request
`nvidia.com/gpu`) and KEDA (`ScaledObject`). Both install with a single
`kubectl apply`/`helm install`; see `docs/MICROSERVICES_ARCHITECTURE.md`.
