# Developer shortcuts for the inference platform.
PY ?= python
REGISTRY ?= aimlab
TAG ?= dev

.PHONY: test lint images images-push compose-dev compose-up k8s-build k8s-apply-local describe

test:            ## unit + loop tests (no GPU, in-memory Redis)
	$(PY) -m pytest tests -q

images:          ## build every service image
	REGISTRY=$(REGISTRY) TAG=$(TAG) docker/build.sh

images-push:     ## build and push every service image
	REGISTRY=$(REGISTRY) TAG=$(TAG) docker/build.sh --push

compose-dev:     ## CPU-only stack: redis + gateway + stub worker
	docker compose -f deploy/docker-compose.dev.yml up --build

compose-up:      ## full GPU stack on one host
	docker compose -f deploy/docker-compose.yml up -d --build

k8s-build:       ## render every overlay (sanity check)
	kubectl kustomize deploy/k8s/overlays/prod-gpu  > /dev/null
	kubectl kustomize deploy/k8s/overlays/dev-cpu   > /dev/null
	kubectl kustomize deploy/k8s/overlays/k3s-local > /dev/null
	@echo "all overlays render"

k8s-apply-local: ## deploy to the local k3s cluster
	kubectl apply -k deploy/k8s/overlays/k3s-local

gateway:         ## run the gateway locally (needs Redis on localhost)
	$(PY) -m gateway

worker:          ## run one worker locally: make worker MODEL=hovernet
	MODEL_NAME=$(MODEL) $(PY) -m worker.main
