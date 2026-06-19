# Working on synth-q-mat-generator

How to set up, run, and resume this project across machines. For *what* the
project is, see [README.md](README.md); for coding standards and design
rationale, see [CLAUDE.md](CLAUDE.md).

Everything routes through the **Makefile** (`make help`) and through **DVC** so
every run is versioned and backed up to S3. The guiding idea: you should be able
to clone on any machine, pull, and pick up exactly where you left off.

## Prerequisites

- [uv](https://astral.sh/uv) and Python 3.13
- An S3 access key for the `synth-q-mat-artifacts` bucket (for `dvc pull`/`push`)
- A free [Materials Project](https://next-gen.materialsproject.org/api) API key
  (only needed for `make download-data`)

## First-time setup

```bash
make setup        # sync deps (incl. dev tools), init DVC, create .env from template
make sync         # (re)install deps — note: bare `uv sync` drops dev tools; use make sync
uv run pre-commit install   # enable the lint/format/docstring hooks on commit
```

Then fill in `.env` (copied from `.env.example`) — see **Credentials** below —
and sanity-check the toolchain:

```bash
make check-deps   # python / uv / dvc / aws / CUDA
make test
```

## Credentials

One file, `.env` (gitignored), holds every secret:

```bash
MP_API_KEY=...                 # Materials Project (make download-data only)
AWS_ACCESS_KEY_ID=...          # S3: boto3 reads these for BOTH dvc[s3] and aws CLI
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
MLFLOW_TRACKING_URI=...        # optional; defaults to http://127.0.0.1:5000
```

Setting `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in `.env` is all DVC needs
for `dvc push`/`pull` — no per-machine `.dvc/config.local` step. (That legacy
path still works if you prefer: `uv run dvc remote modify --local storage
access_key_id <key>`.) A `local-fallback` remote (`.dvcstore/`) exists for
offline work: `uv run dvc push -r local-fallback`.

## The DVC + Hydra workflow

Config lives in `conf/` as Hydra **config groups** (`reward/`, `train/`,
`composition/`, `models/`, plus top-level `seed`/`paths`/`logging` in
`conf/config.yaml`). DVC's Hydra integration composes `conf/` into a single
`params.yaml` that the pipeline reads.

There are two ways to run, and the distinction matters:

### Reproducible runs — `dvc repro` (the default)

```bash
make repro         # whole pipeline: baseline -> train -> eval -> viz
make train         # just the train stage (and anything upstream it needs)
make baseline / make eval / make viz
```

These run the **committed `params.yaml`**, skip stages whose inputs haven't
changed, and are reproducible on any machine. This is what you commit and what
HPC jobs run.

### Exploration & sweeps — `dvc exp run` (overrides)

```bash
make exp ARGS="reward.w_topology=0.5"                  # one tracked experiment
make exp ARGS="train=smoke"                            # swap a whole config group
make sweep ARGS="reward.w_topology=0.3,0.4,0.5"        # grid (Hydra Choice/Range)
uv run dvc exp show                                    # compare results
```

`dvc exp run` **recomposes and rewrites the working `params.yaml`** each time, so
it is *not* a reliable record of a past run. To promote a winning experiment to
the reproducible path: bake the values into `conf/` (or `dvc exp apply <name>`),
then commit `conf/` + `params.yaml` + `dvc.lock` together.

> **Staleness guard:** because `dvc exp run` mutates `params.yaml`, `make repro`/
> `make train` print a warning (`check-params`) if `params.yaml` differs from
> HEAD. Run `git checkout -- params.yaml` to get back the committed config.

With `exp.auto_push = true`, every experiment is pushed to S3 as it finishes, so
nothing lives only on one machine. Disable with `dvc config exp.auto_push false`.

## What's tracked where

- **git:** code, `conf/`, `params.yaml`, `dvc.lock`, and the small `cache:false`
  metrics in `results/metrics/*.json` (legible without a `dvc pull`).
- **S3 (DVC cache):** `models/checkpoints/` (persisted across re-runs/SLURM
  timeouts), `results/figures/`, `results/candidates.json`, and `data/raw`.
- **Not tracked:** pretrained weights (M3GNet, MatterGen base) re-fetch
  reproducibly from their package+handle.

## Resuming on a fresh machine

```bash
git pull                 # code + conf/ + params.yaml + dvc.lock
make sync                # deps
make pull                # data + checkpoints from S3 (needs AWS creds in .env)
make setup-generator     # MatterGen's isolated venv — DVC can't restore this
make repro               # reproduce the committed pipeline
```

The two things DVC can **not** restore are S3 credentials and the MatterGen
venv — provision both, and everything else comes back via `git pull` + `make
pull`.

## Models — why three environments

The 2026 materials-ML ecosystem is split across incompatible dependency stacks,
so each model runs where it can:

- **M3GNet (stability)** — `matgl`, lives in the main env; fetched on demand by
  `make download-models`.
- **Topology (spillage) predictor** — no DGL-free pretrained model exists, so
  it's trained on JARVIS-DFT `spillage` labels in the research phase (matgl or
  MACE-MP). ALIGNN was dropped: it needs DGL, which has no modern torch/Python
  builds.
- **MatterGen (generator)** — pins `torch==2.2.1+cu118` / `numpy<2.0`, which
  can't co-exist with the main env, so it gets its own Python 3.10 venv
  (`.venv-mattergen`) and is driven via its CLI as a subprocess:

  ```bash
  make setup-generator   # clone + install into .venv-mattergen
  make gen-smoke         # generate a couple of structures (needs a working GPU)
  ```

## Downloading source data

```bash
make download-data ARGS="--source jarvis"      # JARVIS-DFT (spillage labels)
make download-data ARGS="--source mp"          # Materials Project (needs MP_API_KEY)
uv run dvc add data/raw && make push           # version the raw dump
```

## Experiment tracking (MLflow)

```bash
make mlflow-ui     # http://localhost:5000 on the workstation
```

HPC jobs log to it by pointing `MLFLOW_TRACKING_URI` in `.env` at the
workstation's address. Every run links to its git commit.

## HPC

`scripts/hpc/*.sbatch` are **generic SLURM templates** — partition names, module
loads, and the VASP launch command are TBD until cluster access is established.
`train.sbatch` does `dvc pull → dvc repro train → dvc push` and fails fast if the
MatterGen venv is missing. After a job, commit + push `dvc.lock` and
`results/metrics/` so other machines can resume.

## Before committing

`make lint` and `make test`. Pre-commit runs Ruff, the standard hygiene hooks,
and `scripts/check_docstrings.py` (no module docstrings, snippet-length
docstrings). Coding standards are in [CLAUDE.md](CLAUDE.md).
