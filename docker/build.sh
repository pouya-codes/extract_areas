#!/usr/bin/env bash
# Build (and optionally push) service images from this monorepo.
#
# Canonical per-model images are built from the aim-inference-worker-* repos
# (FROM aimlabacr.azurecr.io/aimlab/aim-inference-worker-base:main). Prefer
# those for ACR publishes. Local worker-* targets here remain for offline
# monorepo builds during the transition.
#
#   docker/build.sh                      # build all, tag :dev
#   TARGETS=worker-base TAG=main REGISTRY=aimlabacr.azurecr.io/aimlab docker/build.sh --push
#   TARGETS="gateway worker-stub" docker/build.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="${REGISTRY:-aimlab}"
TAG="${TAG:-dev}"
TARGETS="${TARGETS:-gateway worker-stub worker-base worker-hovernet worker-ec worker-pdac worker-deepliif tissue-api}"
PUSH=0
[[ "${1:-}" == "--push" ]] && PUSH=1

cd "$REPO_DIR"
# DeepLIIF and HoVer-Net are git submodules; an empty directory means they
# were never checked out and the image would be built without the model code.
# Only monorepo hovernet/deepliif stages need those submodules. Per-model
# GitHub repos vendor the code; worker-base never needs them.
need_submodules=0
for target in $TARGETS; do
    case "$target" in worker-hovernet) need_hover=1; need_submodules=1 ;;
                      worker-deepliif) need_deep=1; need_submodules=1 ;;
    esac
done
if [[ "$need_submodules" -eq 1 ]]; then
    [[ -n "${need_hover:-}" && -z "$(ls -A module/hovernet 2>/dev/null)" ]] && {
        echo "submodule module/hovernet is empty; run: git submodule update --init --recursive" >&2
        exit 1
    }
    [[ -n "${need_deep:-}" && -z "$(ls -A module/DeepLiff 2>/dev/null)" ]] && {
        echo "submodule module/DeepLiff is empty; run: git submodule update --init --recursive" >&2
        exit 1
    }
fi

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
