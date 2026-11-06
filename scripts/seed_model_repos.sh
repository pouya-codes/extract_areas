#!/usr/bin/env bash
# Seed local checkouts of the per-model GitHub repos (and the template).
#
#   scripts/seed_model_repos.sh [/path/to/output]
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${HOME}/code/aim-inference-workers}"
BASE_IMAGE="${BASE_IMAGE:-aimlabacr.azurecr.io/aimlab/aim-inference-worker-base:main}"
ORG="${GITHUB_ORG:-pouya-codes}"

write_pipeline() {
  local dest="$1" image_suffix="$2"
  cat >"$dest" <<EOF
# Build this model worker FROM the shared base image and push to ACR.
trigger:
  batch: true
  branches:
    include:
      - main
pr: none

variables:
  ACR_NAME: aimlabacr
  REGISTRY: aimlabacr.azurecr.io/aimlab
  IMAGE_NAME: aim-inference-worker-${image_suffix}
  azureSubscription: aim-inference-acr
  BASE_IMAGE: ${BASE_IMAGE}

pool:
  name: k3s-host

steps:
  - checkout: self
    submodules: recursive
    fetchDepth: 1

  - task: AzureCLI@2
    displayName: az acr login + docker build/push
    inputs:
      azureSubscription: \$(azureSubscription)
      scriptType: bash
      scriptLocation: inlineScript
      inlineScript: |
        set -euo pipefail
        SHORT_SHA="\$(echo '\$(Build.SourceVersion)' | cut -c1-12)"
        # Set DOCKER_CONFIG *before* az acr login so the login lands where docker push reads.
        DOCKER_CONFIG="\${AGENT_TEMPDIRECTORY}/docker-config"
        mkdir -p "\$DOCKER_CONFIG"
        export DOCKER_CONFIG
        az acr login --name "\$(ACR_NAME)"
        IMAGE="\$(REGISTRY)/\$(IMAGE_NAME):\${SHORT_SHA}"
        MAIN="\$(REGISTRY)/\$(IMAGE_NAME):main"
        docker build -t "\$IMAGE" --build-arg BASE_IMAGE="\$(BASE_IMAGE)" .
        docker tag "\$IMAGE" "\$MAIN"
        docker push "\$IMAGE"
        docker push "\$MAIN"
        echo "Pushed \$IMAGE and \$MAIN"
EOF
}

git_commit() {
  local dir="$1" msg="$2"
  git -C "$dir" init -b main >/dev/null
  git -C "$dir" add -A
  git -C "$dir" -c user.email="ci@cpathportal.local" -c user.name="cPathPortal CI" commit -m "$msg" >/dev/null
}

write_readme() {
  local dest="$1" name="$2" model_env="$3"
  cat >"$dest" <<EOF
# aim-inference-worker-${name}

Region-processing worker for **${model_env}**. Built \`FROM\` the shared
base image \`${BASE_IMAGE}\` published by
[\`extract_areas\`](https://github.com/${ORG}/extract_areas).

## Build locally

\`\`\`bash
az acr login --name aimlabacr
docker build -t aimlabacr.azurecr.io/aimlab/aim-inference-worker-${name}:dev \\
  --build-arg BASE_IMAGE=${BASE_IMAGE} .
\`\`\`

Weights are not in this image; on k3s they mount from \`/data/models\`.

## CI/CD

Push to \`main\` runs Azure Pipelines on pool \`k3s-host\` and pushes \`:sha\`
and \`:main\` to ACR. Argo CD Image Updater rolls out the new digest.
EOF
}

seed_model() {
  local name="$1" model_env="$2" req="$3" model_py="$4"
  local dir="${OUT}/aim-inference-worker-${name}"
  echo "==> ${dir}"
  rm -rf "$dir"
  mkdir -p "$dir/models"
  cp "$SRC/worker/requirements/${req}" "$dir/requirements.txt"
  cp "$SRC/models/${model_py}" "$dir/models/"
  printf '' >"$dir/models/__init__.py"
  printf '%s\n' '__pycache__/' '*.pyc' '.venv/' '.DS_Store' >"$dir/.gitignore"

  case "$name" in
    pdac)
      mkdir -p "$dir/module"
      cp -a "$SRC/module/pdac_inference" "$dir/module/"
      cat >"$dir/Dockerfile" <<EOF
ARG BASE_IMAGE=${BASE_IMAGE}
FROM \${BASE_IMAGE}
USER root
COPY requirements.txt /tmp/model.txt
RUN pip install -r /tmp/model.txt
COPY models/ /app/models/
COPY module/ /app/module/
USER 10001
ENV MODEL_NAME=pdac HF_HOME=/tmp/hf
EOF
      ;;
    hovernet)
      mkdir -p "$dir/module"
      cp -a "$SRC/module/hovernet" "$dir/module/"
      # Drop docs/samples/media — not needed at inference time and bloated CI.
      rm -rf "$dir/module/hovernet/docs" "$dir/module/hovernet/examples" \
        "$dir/module/hovernet/diagram.png" "$dir/module/hovernet/seg.gif" \
        "$dir/module/hovernet/CHANGELOG.md"
      find "$dir/module/hovernet" -type f \( -name '*.gif' -o -name '*.ipynb' \) -delete || true
      cat >"$dir/Dockerfile" <<EOF
ARG BASE_IMAGE=${BASE_IMAGE}
FROM \${BASE_IMAGE}
USER root
COPY requirements.txt /tmp/model.txt
RUN pip install -r /tmp/model.txt
COPY models/ /app/models/
COPY module/ /app/module/
USER 10001
ENV MODEL_NAME=hovernet
EOF
      ;;
    ec)
      cat >"$dir/Dockerfile" <<EOF
ARG BASE_IMAGE=${BASE_IMAGE}
FROM \${BASE_IMAGE}
USER root
COPY requirements.txt /tmp/model.txt
RUN pip install -r /tmp/model.txt
COPY models/ /app/models/
USER 10001
ENV MODEL_NAME=ec_cancer
EOF
      ;;
    deepliif)
      mkdir -p "$dir/module"
      cp -a "$SRC/module/DeepLiff" "$dir/module/"
      rm -rf "$dir/module/DeepLiff/Datasets" "$dir/module/DeepLiff/Sample_Large_Tissues" \
        "$dir/module/DeepLiff/docs" "$dir/module/DeepLiff/images" \
        "$dir/module/DeepLiff/ImageJ_Plugin" "$dir/module/DeepLiff/assets" \
        "$dir/module/DeepLiff/DeepLIIF_Statistics" "$dir/module/DeepLiff/Registration" \
        "$dir/module/DeepLiff/Scripts"
      find "$dir/module/DeepLiff" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
      cat >"$dir/Dockerfile" <<EOF
ARG BASE_IMAGE=${BASE_IMAGE}
FROM \${BASE_IMAGE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-11-jdk-headless \\
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
COPY requirements.txt /tmp/model.txt
RUN pip install --no-build-isolation -r /tmp/model.txt
COPY models/ /app/models/
COPY module/ /app/module/
USER 10001
ENV MODEL_NAME=deepliif
EOF
      ;;
  esac

  write_pipeline "$dir/azure-pipelines.yml" "$name"
  write_readme "$dir/README.md" "$name" "$model_env"
  git_commit "$dir" "Initial ${model_env} worker from extract_areas"
}

mkdir -p "$OUT"
seed_model pdac pdac pdac.txt pdac_model.py
seed_model hovernet hovernet hovernet.txt hovernet_model.py
seed_model ec ec_cancer ec.txt ec_cancer_model.py
seed_model deepliif deepliif deepliif.txt deepliif_model.py

# Template
T="${OUT}/aim-inference-worker-template"
echo "==> ${T}"
rm -rf "$T"
mkdir -p "$T/models"
cp "$SRC/worker/templates/model_template.py" "$T/models/MODEL_NAME_model.py"
cat >"$T/requirements.txt" <<'EOF'
# Model-specific pip deps beyond the shared worker-base image.
EOF
cat >"$T/Dockerfile" <<EOF
ARG BASE_IMAGE=${BASE_IMAGE}
ARG MODEL_NAME=my_model
FROM \${BASE_IMAGE}
USER root
COPY requirements.txt /tmp/model.txt
RUN pip install -r /tmp/model.txt
COPY models/ /app/models/
USER 10001
ENV MODEL_NAME=\${MODEL_NAME}
EOF
cat >"$T/azure-pipelines.yml" <<EOF
# After "Use this template", set ModelName to the snake_case model id.
trigger:
  batch: true
  branches:
    include: [main]
pr: none

parameters:
  - name: ModelName
    type: string
    default: my_model

variables:
  ACR_NAME: aimlabacr
  REGISTRY: aimlabacr.azurecr.io/aimlab
  IMAGE_NAME: aim-inference-worker-\${{ parameters.ModelName }}
  azureSubscription: aim-inference-acr
  BASE_IMAGE: ${BASE_IMAGE}

pool:
  name: k3s-host

steps:
  - checkout: self
    fetchDepth: 1
  - task: AzureCLI@2
    displayName: build and push
    inputs:
      azureSubscription: \$(azureSubscription)
      scriptType: bash
      scriptLocation: inlineScript
      inlineScript: |
        set -euo pipefail
        SHORT_SHA="\$(echo '\$(Build.SourceVersion)' | cut -c1-12)"
        DOCKER_CONFIG="\${AGENT_TEMPDIRECTORY}/docker-config"
        mkdir -p "\$DOCKER_CONFIG" && export DOCKER_CONFIG
        az acr login --name "\$(ACR_NAME)"
        IMAGE="\$(REGISTRY)/\$(IMAGE_NAME):\${SHORT_SHA}"
        MAIN="\$(REGISTRY)/\$(IMAGE_NAME):main"
        docker build -t "\$IMAGE" --build-arg BASE_IMAGE="\$(BASE_IMAGE)" \\
          --build-arg MODEL_NAME="\${{ parameters.ModelName }}" .
        docker tag "\$IMAGE" "\$MAIN"
        docker push "\$IMAGE" && docker push "\$MAIN"
EOF
cat >"$T/README.md" <<EOF
# aim-inference-worker-template

GitHub **template** repository for a new AIM inference model worker.

1. Use this template → name it \`aim-inference-worker-<name>\`.
2. Rename \`models/MODEL_NAME_model.py\` and fill in the TODOs (same contract as
   extract_areas \`worker/templates/model_template.py\`).
3. Edit \`requirements.txt\` and the pipeline \`ModelName\` parameter.
4. In **extract_areas**, add \`deploy/k8s/base/workers/worker-<name>.yaml\` and
   Image Updater annotations on the Argo CD Application.
5. Push \`main\` → Azure Pipeline builds \`FROM ${BASE_IMAGE}\`.
EOF
printf '%s\n' '__pycache__/' '*.pyc' '.venv/' '.DS_Store' >"$T/.gitignore"
git_commit "$T" "Initial worker template"

echo
echo "Seeded:"
ls -1 "$OUT"
echo "Push with: GITHUB_TOKEN=... ${SRC}/scripts/push_model_repos.sh"
