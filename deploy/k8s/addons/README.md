# Cluster add-ons

Pinned copies of the two cluster-level components the inference platform
needs, so a deployment is reproducible without fetching from the internet
at apply time.

| File | What | Upstream |
|---|---|---|
| `keda-2.16.1.yaml` | KEDA operator + metrics adapter (namespace `keda`) | https://github.com/kedacore/keda/releases/tag/v2.16.1 |
| `nvidia-device-plugin.yaml` | NVIDIA device plugin DaemonSet (kube-system) with 2-way time-slicing | https://github.com/NVIDIA/k8s-device-plugin v0.17.1 |

```bash
kubectl apply --server-side -f deploy/k8s/addons/keda-2.16.1.yaml
kubectl apply -f deploy/k8s/addons/nvidia-device-plugin.yaml
kubectl describe node | grep -A1 'nvidia.com/gpu'   # capacity shows up within a minute
```

The device plugin needs the NVIDIA container runtime on every GPU node
(`nvidia-container-toolkit` package; k3s picks it up on restart and creates
the `nvidia` RuntimeClass).
