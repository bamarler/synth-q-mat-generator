#!/usr/bin/env bash
# Smoke-test MatterGen: generate a couple of structures from the base model in
# the isolated env. Needs a working GPU. Run via `make gen-smoke`.
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv-mattergen}"
OUT="results/mattergen_smoke"

if [ ! -x "${VENV_DIR}/bin/mattergen-generate" ]; then
    echo "mattergen-generate not found in ${VENV_DIR} — run 'make setup-generator' first."
    exit 1
fi

mkdir -p "${OUT}"
echo "[gen-smoke] generating 2 structures into ${OUT} (base model)..."
"${VENV_DIR}/bin/mattergen-generate" "${OUT}" \
    --pretrained-name=mattergen_base \
    --batch_size=2 \
    --num_batches=1

echo "[gen-smoke] outputs:"
ls -la "${OUT}"
