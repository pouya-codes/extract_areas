#!/usr/bin/env bash
# Move this host from the Docker Compose stack to k3s with autoscaling.
# Needs cluster-admin (installs KEDA + the NVIDIA device plugin) and sudo.
#
#   deploy/k8s/overlays/k3s-local/switch-from-compose.sh
#
# Steps: add-ons -> images -> namespace+secret -> stop compose -> apply ->
# wait for workers -> repoint AIMViewer (config.yaml) -> restart OMERO.web.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
AIM_CONFIG="${AIM_CONFIG:-/home/ubuntu/code/aimviewer/aimviewer/config.yaml}"
OMERO_BIN="${OMERO_BIN:-/home/ubuntu/prog/omero/venv_server/bin/omero}"
cd "$REPO_DIR"

echo "==> 1/7 cluster add-ons (KEDA, NVIDIA device plugin)"
kubectl apply --server-side -f deploy/k8s/addons/keda-2.16.1.yaml >/dev/null
kubectl apply -f deploy/k8s/addons/nvidia-device-plugin.yaml >/dev/null
kubectl -n keda rollout status deploy/keda-operator --timeout=300s
kubectl -n kube-system rollout status ds/nvidia-device-plugin-daemonset --timeout=300s
for i in $(seq 1 30); do
    cap=$(kubectl get node -o jsonpath='{.items[0].status.capacity.nvidia\.com/gpu}')
    [[ -n "$cap" && "$cap" != "0" ]] && break; sleep 5
done
echo "    nvidia.com/gpu capacity: ${cap:-none}"

echo "==> 2/7 images into k3s containerd (skips ones already present)"
for t in gateway worker-hovernet worker-ec worker-pdac worker-deepliif tissue-api; do
    if ! sudo k3s ctr images ls -q | grep -q "aimlab/aim-inference-$t:dev"; then
        sudo docker save "aimlab/aim-inference-$t:dev" | sudo k3s ctr images import - >/dev/null
    fi
done

echo "==> 3/7 namespace + secret"
deploy/k8s/overlays/k3s-local/create-secret.sh

echo "==> 4/7 stop the compose stack (frees the GPUs; waits for in-flight regions)"
docker compose --env-file deploy/compose.env -f deploy/docker-compose.yml down

echo "==> 5/7 apply"
kubectl apply -k deploy/k8s/overlays/k3s-local
kubectl -n aim-inference rollout status deploy/inference-gateway --timeout=300s
for d in worker-deepliif worker-hovernet worker-ec-cancer tissue-api; do
    kubectl -n aim-inference rollout status deploy/$d --timeout=900s
done

echo "==> 6/7 smoke"
TOKEN=$(grep '^GATEWAY_API_TOKEN=' deploy/compose.env | cut -d= -f2-)
curl -fsS -H "Authorization: Bearer $TOKEN" http://127.0.0.1:30080/v1/models | python3 -c "import sys,json;print('   ', [(m['name'], m['replicas']) for m in json.load(sys.stdin)['models']])"
curl -fsS -o /dev/null http://127.0.0.1:30800/health && echo "    tissue-api ok"

echo "==> 7/7 repoint AIMViewer"
cp "$AIM_CONFIG" "$AIM_CONFIG.bak-pre-k3s"
sed -i -e 's#URL: http://127.0.0.1:8090#URL: http://127.0.0.1:30080#' \
       -e 's#general: http://127.0.0.1:8010#general: http://127.0.0.1:30800#' "$AIM_CONFIG"
"$OMERO_BIN" web restart
echo "done: AIMViewer -> gateway :30080, tissue-api :30800; KEDA scales workers on queue depth"
