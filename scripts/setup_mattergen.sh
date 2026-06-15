#!/usr/bin/env bash
# Set up MatterGen in an ISOLATED environment.
#
# MatterGen pins torch==2.2.1+cu118 / numpy<2.0 / Python 3.10, which conflict
# with the main env (torch 2.6+cu124, numpy 2.x, py3.13). So it lives in its own
# venv and is driven via its CLI (mattergen-generate) as a subprocess — the same
# way the RL loop will invoke the generator.
#
# Run on a machine with a WORKING GPU. Idempotent: re-running updates the clone.
#
#   make setup-generator      # or: bash scripts/setup_mattergen.sh
set -euo pipefail

VENV_DIR="${MATTERGEN_VENV:-.venv-mattergen}"
SRC_DIR="${VENV_DIR}-src"          # the cloned mattergen repo
REPO="https://github.com/microsoft/mattergen.git"

green() { printf "\033[0;32m%s\033[0m\n" "$1"; }
yellow() { printf "\033[0;33m%s\033[0m\n" "$1"; }

# --- prerequisites -------------------------------------------------------
command -v uv >/dev/null 2>&1 || { echo "uv not found (https://astral.sh/uv)"; exit 1; }
if ! command -v git-lfs >/dev/null 2>&1; then
    yellow "[warn] git-lfs not found — checkpoints download via huggingface_hub instead,"
    yellow "       but installing it is recommended: sudo apt install git-lfs && git lfs install"
fi

# --- clone / update MatterGen -------------------------------------------
if [ -d "${SRC_DIR}/.git" ]; then
    green "[mattergen] updating existing clone in ${SRC_DIR}"
    git -C "${SRC_DIR}" pull --ff-only
else
    green "[mattergen] cloning ${REPO} -> ${SRC_DIR}"
    git clone "${REPO}" "${SRC_DIR}"
fi

# --- isolated env --------------------------------------------------------
green "[mattergen] creating isolated env ${VENV_DIR} (Python 3.10)"
uv venv "${VENV_DIR}" --python 3.10
green "[mattergen] installing MatterGen (editable) into ${VENV_DIR}"
VIRTUAL_ENV="${VENV_DIR}" uv pip install -e "${SRC_DIR}"

green "[mattergen] done."
echo "  Smoke-test generation with:   make gen-smoke"
echo "  Base checkpoint downloads from HuggingFace on first generate (--pretrained-name=mattergen_base)."
