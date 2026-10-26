#!/usr/bin/env bash
# Build (and optionally push) every service image.
#
#   docker/build.sh                      # build all, tag :dev
#   TAG=1.2.0 REGISTRY=ghcr.io/aimlab docker/build.sh --push
#   TARGETS="gateway worker-stub" docker/build.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="${REGISTRY:-aimlab}"
TAG="${TAG:-dev}"
TARGETS="${TARGETS:-gateway worker-stub worker-hovernet worker-ec worker-pdac worker-deepliif}"
PUSH=0
[[ "${1:-}" == "--push" ]] && PUSH=1

cd "$REPO_DIR"
# DeepLIIF and HoVer-Net are git submodules; an empty directory means they
# were never checked out and the image would be built without the model code.
for sub in module/DeepLiff module/hovernet; do
    if [[ -z "$(ls -A "$sub" 2>/dev/null)" ]]; then
        echo "submodule $sub is empty; run: git submodule update --init --recursive" >&2
        exit 1
    fi
done

for target in $TARGETS; do
    image="${REGISTRY}/aim-inference-${target}:${TAG}"
    echo "==> building ${image}"
    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile --target "$target" -t "$image" \
        --build-arg WORKER_IMAGE="$image" .
    if [[ $PUSH -eq 1 ]]; then
        docker push "$image"
    fi
done
echo "done"
