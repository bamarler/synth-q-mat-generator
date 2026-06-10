# Shell configuration for git bash compatibility
ifeq ($(OS),Windows_NT)
    SHELL := C:/Program Files/Git/bin/bash.exe
    ifeq ($(wildcard $(SHELL)),)
        SHELL := C:/Program Files/Git/usr/bin/bash.exe
    endif
    export PATH := C:/Program Files/Git/usr/bin:$(PATH)
else
    SHELL := /bin/bash
endif
.SHELLFLAGS := -euo pipefail -c

.PHONY: help check-deps setup sync pull push download-data download-models \
        setup-generator gen-smoke \
        baseline train eval viz hpc-train hpc-status mlflow-ui \
        lint format test clean

# Colors for output
GREEN  := \033[0;32m
RED    := \033[0;31m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
CYAN   := \033[0;36m
NC     := \033[0m
CHECKMARK := [OK]
CROSSMARK := [NO]

# Project settings
PROJECT_NAME  := synth-q-mat-generator
PYTHON_VER    := 3.13
# Local DVC remote until an S3 bucket exists. To move to S3 later (after
# creating a bucket + filling AWS keys in .env):
#   uv run dvc remote add -d -f storage s3://<bucket>/dvc
DVC_REMOTE    := .dvcstore
UV_CACHE_DIR  ?= .uv-cache
UV_RUN        = UV_CACHE_DIR=$(UV_CACHE_DIR) uv run
MATTERGEN_VENV := .venv-mattergen

help: ## Show this help message
	@printf "$(CYAN)====================================================\n$(NC)"
	@printf "$(CYAN)  synth-q-mat-generator - Development Commands\n$(NC)"
	@printf "$(CYAN)====================================================\n\n$(NC)"
	@awk 'BEGIN {FS = ":.*##"} \
		/^[a-zA-Z_-]+:.*?##/ { \
			printf "  $(GREEN)%-18s$(NC) %s\n", $$1, $$2 \
		}' $(MAKEFILE_LIST)
	@printf "\n"

check-deps: ## Verify python, uv, dvc, aws cli, and CUDA are available
	@printf "$(BLUE)Checking dependencies...\n$(NC)\n"
	@command -v python >/dev/null 2>&1 && \
		python -c "import sys; exit(0 if sys.version_info[:2]==(3,13) else 1)" 2>/dev/null && \
		printf "  $(GREEN)$(CHECKMARK)$(NC) Python:   $$(python --version)\n" || \
		printf "  $(RED)$(CROSSMARK)$(NC) Python:   Not 3.13 (found: $$(python --version 2>/dev/null || echo 'not installed'))\n"
	@command -v uv >/dev/null 2>&1 && \
		printf "  $(GREEN)$(CHECKMARK)$(NC) uv:       $$(uv --version)\n" || \
		printf "  $(RED)$(CROSSMARK)$(NC) uv:       Not found (install: curl -LsSf https://astral.sh/uv/install.sh | sh)\n"
	@$(UV_RUN) dvc --version >/dev/null 2>&1 && \
		printf "  $(GREEN)$(CHECKMARK)$(NC) dvc:      $$($(UV_RUN) dvc --version)\n" || \
		printf "  $(YELLOW)[WARN]$(NC) dvc:      Not found (run 'make sync')\n"
	@command -v aws >/dev/null 2>&1 && \
		printf "  $(GREEN)$(CHECKMARK)$(NC) aws cli:  $$(aws --version 2>&1 | cut -d' ' -f1)\n" || \
		printf "  $(YELLOW)[WARN]$(NC) aws cli:  Not found (needed for S3; dvc[s3] provides a fallback)\n"
	@$(UV_RUN) python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && \
		printf "  $(GREEN)$(CHECKMARK)$(NC) CUDA:     available ($$($(UV_RUN) python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null))\n" || \
		printf "  $(YELLOW)[WARN]$(NC) CUDA:     not available (CPU-only is fine for non-training steps)\n"
	@printf "\n"

setup: ## One-time setup: sync deps, init DVC, configure local remote
	@printf "$(BLUE)Setting up $(PROJECT_NAME)...\n$(NC)"
	@$(MAKE) sync
	@if [ ! -d .dvc ]; then \
		$(UV_RUN) dvc init; \
		$(UV_RUN) dvc remote add -d storage $(DVC_REMOTE); \
		printf "  $(GREEN)$(CHECKMARK)$(NC) DVC initialized with local remote $(DVC_REMOTE)\n"; \
		printf "  $(YELLOW)[NOTE]$(NC)  Local remote for now; swap to S3 later (see Makefile DVC_REMOTE comment)\n"; \
	else \
		printf "  $(GREEN)$(CHECKMARK)$(NC) DVC already initialized\n"; \
	fi
	@if [ ! -f .env ]; then cp .env.example .env; \
		printf "  $(YELLOW)[ACTION]$(NC) Created .env from template - add your MP_API_KEY\n"; fi
	@printf "\n"

sync: ## Install/sync all dependencies with uv (incl. dev extras)
	@printf "$(BLUE)Syncing dependencies...\n$(NC)"
	@UV_CACHE_DIR=$(UV_CACHE_DIR) uv sync --extra dev
	@printf "  $(GREEN)$(CHECKMARK)$(NC) Dependencies synced\n\n"

pull: ## Restore data/ and models/ from the DVC remote (dvc pull)
	@printf "$(BLUE)Pulling artifacts from remote '$(DVC_REMOTE)'...\n$(NC)"
	@$(UV_RUN) dvc pull
	@printf "  $(GREEN)$(CHECKMARK)$(NC) Artifacts restored\n\n"

push: ## Upload tracked data/ and models/ to the DVC remote (dvc push)
	@printf "$(BLUE)Pushing artifacts to remote '$(DVC_REMOTE)'...\n$(NC)"
	@$(UV_RUN) dvc push
	@printf "  $(GREEN)$(CHECKMARK)$(NC) Artifacts uploaded\n\n"

download-data: ## Download source databases (TQC, JARVIS-DFT, Materials Project)
	@printf "$(BLUE)Downloading source databases...\n$(NC)\n"
	@$(UV_RUN) python scripts/download_data.py $(ARGS)

download-models: ## Fetch + smoke-load the pretrained reward model (M3GNet)
	@printf "$(BLUE)Downloading reward models...\n$(NC)\n"
	@$(UV_RUN) python scripts/download_models.py $(ARGS)

setup-generator: ## Create the isolated MatterGen env + fetch a base checkpoint
	@printf "$(BLUE)Setting up MatterGen (isolated env $(MATTERGEN_VENV))...\n$(NC)\n"
	@bash scripts/setup_mattergen.sh

gen-smoke: ## Smoke-test MatterGen: generate a couple of structures
	@printf "$(BLUE)MatterGen smoke test...\n$(NC)\n"
	@if [ ! -d $(MATTERGEN_VENV) ]; then \
		printf "  $(RED)$(CROSSMARK)$(NC) $(MATTERGEN_VENV) missing - run 'make setup-generator' first\n"; exit 1; fi
	@VENV_DIR=$(MATTERGEN_VENV) bash scripts/gen_smoke.sh

baseline: ## Evaluate unconditional generation baseline
	@printf "$(BLUE)Running baseline evaluation...\n$(NC)\n"
	@$(UV_RUN) python -m synth_q_mat.eval.baseline $(ARGS)

train: ## Train the MatInvent PPO policy (override config via ARGS)
	@printf "$(BLUE)Starting RL training...\n$(NC)\n"
	@$(UV_RUN) python -m synth_q_mat.rl.train $(ARGS)

eval: ## Evaluate / rank generated candidates
	@printf "$(BLUE)Running candidate evaluation...\n$(NC)\n"
	@$(UV_RUN) python -m synth_q_mat.eval.evaluate $(ARGS)

viz: ## Generate figures (Pareto fronts, band structures, training curves)
	@printf "$(BLUE)Generating visualizations...\n$(NC)\n"
	@$(UV_RUN) python -m synth_q_mat.viz.generate $(ARGS)

hpc-train: ## Submit RL training as a SLURM job on the HPC cluster
	@printf "$(BLUE)Submitting SLURM training job...\n$(NC)\n"
	@sbatch scripts/hpc/train.sbatch

hpc-status: ## Show your SLURM queue
	@squeue -u $(USER)

mlflow-ui: ## Launch the MLflow tracking UI (http://localhost:5000)
	@printf "$(BLUE)Starting MLflow UI on http://localhost:5000 ...\n$(NC)\n"
	@$(UV_RUN) mlflow ui --host 0.0.0.0 --port 5000

lint: ## Check linting and formatting with Ruff
	@printf "$(BLUE)Checking linting...\n$(NC)"
	@$(UV_RUN) ruff check src/ tests/ scripts/
	@printf "  $(GREEN)$(CHECKMARK)$(NC) Linting passed\n"
	@printf "$(BLUE)Checking formatting...\n$(NC)"
	@$(UV_RUN) ruff format --check src/ tests/ scripts/
	@printf "  $(GREEN)$(CHECKMARK)$(NC) Formatting passed\n\n"

format: ## Auto-fix linting and format code with Ruff
	@printf "$(BLUE)Fixing linting...\n$(NC)"
	@$(UV_RUN) ruff check --fix src/ tests/ scripts/
	@printf "$(BLUE)Formatting code...\n$(NC)"
	@$(UV_RUN) ruff format src/ tests/ scripts/
	@printf "  $(GREEN)$(CHECKMARK)$(NC) Format complete\n\n"

test: ## Run unit tests
	@printf "$(BLUE)Running tests...\n$(NC)\n"
	@$(UV_RUN) pytest

clean: ## Remove caches and temporary outputs
	@printf "$(YELLOW)Cleaning caches...\n$(NC)"
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name '*.py[co]' -delete 2>/dev/null || true
	@rm -rf results/tmp 2>/dev/null || true
	@printf "  $(GREEN)$(CHECKMARK)$(NC) Clean\n\n"
