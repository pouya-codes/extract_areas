# Developer shortcuts for the inference platform.
PY ?= python
REGISTRY ?= aimlab
TAG ?= dev

.PHONY: test lint images images-push compose-dev compose-up compose-down compose-ps k8s-build k8s-apply-local k8s-import-images k8s-addons describe new-model

test:            ## unit + loop tests (no GPU, in-memory Redis)
	$(PY) -m pytest tests -q

images:          ## build every service image
	REGISTRY=$(REGISTRY) TAG=$(TAG) docker/build.sh

images-push:     ## build and push every service image
	REGISTRY=$(REGISTRY) TAG=$(TAG) docker/build.sh --push

compose-dev:     ## CPU-only stack: redis + gateway + stub worker
	docker compose -f deploy/docker-compose.dev.yml up --build

compose-up:      ## full GPU stack on one host (deploy/compose.env holds tokens + GPU pins)
	docker compose --env-file deploy/compose.env -f deploy/docker-compose.yml up -d

compose-down:    ## stop the compose stack (waits for in-flight regions)
	docker compose --env-file deploy/compose.env -f deploy/docker-compose.yml down

compose-ps:      ## health of the compose stack
	docker compose --env-file deploy/compose.env -f deploy/docker-compose.yml ps

k8s-build:       ## render every overlay (sanity check)
	kubectl kustomize deploy/k8s/overlays/prod-gpu  > /dev/null
	kubectl kustomize deploy/k8s/overlays/dev-cpu   > /dev/null
	kubectl kustomize deploy/k8s/overlays/k3s-local > /dev/null
	@echo "all overlays render"

k8s-addons:      ## cluster prerequisites: KEDA + NVIDIA device plugin (cluster-admin)
	kubectl apply --server-side -f deploy/k8s/addons/keda-2.16.1.yaml
	kubectl apply -f deploy/k8s/addons/nvidia-device-plugin.yaml

k8s-import-images: ## load locally built images into k3s containerd (no registry needed)
	for t in gateway worker-stub worker-hovernet worker-ec worker-pdac worker-deepliif tissue-api; do \
	  docker save $(REGISTRY)/aim-inference-$$t:$(TAG) | sudo k3s ctr images import - ; done

k8s-apply-local: ## deploy to the local k3s cluster (after k8s-addons, k8s-import-images, create-secret.sh)
	kubectl apply -k deploy/k8s/overlays/k3s-local

new-model:       ## scaffold a model: make new-model NAME=tumor_grader
	python scripts/new_model.py $(NAME)

gateway:         ## run the gateway locally (needs Redis on localhost)
	$(PY) -m gateway

worker:          ## run one worker locally: make worker MODEL=hovernet
	MODEL_NAME=$(MODEL) $(PY) -m worker.main
