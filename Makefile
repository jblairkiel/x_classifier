# tl-ai-detector operations. Run `make help`.
# Production uses docker-compose.yml; LOCAL=1 adds docker-compose.local.yml (https://localhost:8443).

SHELL := /bin/bash
COMPOSE_FILES := -f docker-compose.yml $(if $(LOCAL),-f docker-compose.local.yml)
COMPOSE := docker compose $(COMPOSE_FILES)
GPU_COMPOSE := docker compose -f docker-compose.yml -f docker-compose.gpu.yml
BACKUP_DIR ?= backups
STAMP := $(shell date -u +%Y%m%d-%H%M%S)
SESSION ?= secrets/x_storage_state.json

.PHONY: help build up down restart logs ps backup restore push-session push-model hash-password train train-gpu eval \
        classify features status resume test local-up local-down

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Build web, worker and proxy images
	$(COMPOSE) build web worker proxy

up: ## Start the stack (detached)
	@test -f .env || (echo "missing .env -- cp .env.example .env and fill it in" && exit 1)
	$(COMPOSE) up -d web worker proxy

down: ## Stop the stack (volumes are kept)
	$(COMPOSE) down

restart: ## Restart web + worker
	$(COMPOSE) restart web worker

logs: ## Follow logs (S=service to filter, e.g. make logs S=worker)
	$(COMPOSE) logs -f --tail=200 $(S)

ps: ## Container status + health
	$(COMPOSE) ps

status: ## Scraper state, counts, recent runs
	$(COMPOSE) exec worker tl status

resume: ## Clear a paused/error scraper state
	$(COMPOSE) exec worker tl resume

push-session: ## Copy a local storage-state file into the secrets volume (SESSION=path)
	@test -f "$(SESSION)" || (echo "no session file at $(SESSION); run 'tl login' locally first" && exit 1)
	$(COMPOSE) --profile ops run --rm -T ops install-session < "$(SESSION)"
	@echo "Done. Resume the scraper from the dashboard (or: make resume)."

hash-password: ## Generate ADMIN_PASSWORD_HASH (prompts for the password)
	$(COMPOSE) --profile ops run --rm ops hash-password

backup: ## Write backups/tl-<timestamp>.tar.gz (DB + model checkpoints)
	@mkdir -p $(BACKUP_DIR)
	$(COMPOSE) --profile ops run --rm -T ops backup - > $(BACKUP_DIR)/tl-$(STAMP).tar.gz
	@ls -lh $(BACKUP_DIR)/tl-$(STAMP).tar.gz

restore: ## Restore FILE=backups/tl-....tar.gz (stops web+worker while restoring)
	@test -n "$(FILE)" || (echo "usage: make restore FILE=backups/tl-....tar.gz" && exit 1)
	$(COMPOSE) stop web worker
	$(COMPOSE) --profile ops run --rm -T ops restore --yes - < "$(FILE)"
	$(COMPOSE) start web worker

train: ## Train on CPU in the opt-in train container (ARGS="--epochs 2 ...")
	$(COMPOSE) --profile train run --rm train train $(ARGS)

train-gpu: ## Train on an NVIDIA GPU (needs NVIDIA Container Toolkit)
	$(GPU_COMPOSE) --profile train run --rm train train --device cuda $(ARGS)

push-model: ## Copy a locally trained checkpoint into the models volume (VERSION=v..., MODELS=./models)
	@test -n "$(VERSION)" && test -f "$(or $(MODELS),models)/$(VERSION)/COMPLETE" || \
	  (echo "usage: make push-model VERSION=v... [MODELS=./models]  (dir must contain COMPLETE)" && exit 1)
	tar -C "$(or $(MODELS),models)" -cf - "$(VERSION)" | \
	  $(COMPOSE) --profile ops run --rm -T --entrypoint sh ops -c 'tar -C /models -xf - && ls /models'
	@echo "Done. With MODEL_VERSION=latest the worker uses the newest checkpoint; press Reclassify on the Model page."

eval: ## Evaluate a checkpoint (ARGS="--version v...")
	$(COMPOSE) --profile train run --rm train eval $(ARGS)

classify: ## Score unscored tweets with the active model (in the worker, CPU)
	$(COMPOSE) exec worker tl classify

features: ## Recompute account botlike scores
	$(COMPOSE) exec worker tl features

test: ## Run the test suite locally (needs `pip install -e .[web,scraper,dev]`)
	python -m pytest -q

local-up: ## Start the stack on https://localhost:8443 (self-signed)
	$(MAKE) up LOCAL=1

local-down: ## Stop the local stack
	$(MAKE) down LOCAL=1
