# CLAUDE.md

Context for AI assistants working in this repo.

## What this is

`synth-q-mat-generator` — a 10-week research project (PEAK Summit Fellowship,
Northeastern, with Prof. Yan) to discover **topologically constrained quantum
materials** with reinforcement learning. A MatInvent-style PPO policy fine-tunes
a pretrained crystal generator (**MatterGen**) against a multi-objective reward
that combines:

- **topology** — spin-orbit spillage from a regressor trained on JARVIS labels
- **dos_fermi** — density of states at the Fermi level (superconductive tendency)
- **formation** — formation energy / stability from M3GNet
- **composition** — light-element (Li–Si) preference to keep DFT cheap

Top candidates are validated with DFT (VASP) and phonon calculations (Quantum
ESPRESSO) on an HPC cluster. Full proposal + timeline in `references/*.pdf`.

## Key decisions (and why)

- **Python 3.13** — not 3.14 (free-threading still experimental, partial ML
  support); 3.13 has full stack support and the maintainer uses it.
- **uv** for packaging; torch index is platform-conditional (CUDA on Linux,
  CPU elsewhere) — see `[tool.uv.sources]` in `pyproject.toml`.
- **Generator = MatterGen** (not DiffCSP, which pins Python 3.8/torch 1.9).
  MatterGen pins torch 2.2.1+cu118 / numpy<2.0, incompatible with the main env,
  so it lives in an isolated Python 3.10 venv (`.venv-mattergen`) and is driven
  via its CLI. Deviation from the proposal's DiffCSP — flagged for Prof. Yan.
- **Topology predictor = DGL-free spillage regressor**, trained on JARVIS
  spillage labels (matgl or MACE-MP) in the research phase. ALIGNN was dropped:
  it needs DGL, which has no modern torch/Python builds.
- **DVC** for data/RL-checkpoint versioning; default remote is
  **s3://synth-q-mat-artifacts/dvc** (single-purpose IAM key scoped to that
  bucket; per-machine credentials in gitignored `.dvc/config.local`, AWS keys
  also in `.env` for the aws CLI). `.dvcstore` kept as `local-fallback` remote.
  Pretrained reward/generator weights re-fetch from their package+handle, so
  they are NOT DVC-tracked.
- **MLflow** (self-hosted) for experiment tracking; runs link to git commits.
- **MIT** license.
- Viz: matplotlib/seaborn for publication, plotly for interactive.
- **Julia evaluated and rejected** — generative/RL/GNN crystal work is entirely
  PyTorch; Julia has no equivalent and interop adds friction. Stay Python.

## Working in this repo

- Everything is driven through the **Makefile** — `make help` lists targets.
  Common: `make setup`, `make sync`, `make pull`/`push`, `make train`,
  `make baseline`, `make eval`, `make viz`, `make test`, `make lint`,
  `make format`, `make mlflow-ui`.
- Pass config overrides via `ARGS`, Hydra-style:
  `make train ARGS="reward.w_topology=0.5 train.total_steps=200000"`.
- Config lives in `conf/` (Hydra config groups), loaded by
  `src/synth_q_mat/config.py` as a plain dict — it reads `params.yaml` if present
  (DVC composes `conf/` into it), else composes `conf/` directly. Override
  Hydra-style: `reward.w_topology=0.5`.
- Source entrypoints are module-runnable: `python -m synth_q_mat.rl.train`, etc.
- Run `make lint` and `make test` before committing. Ruff config and the test
  suite are in `pyproject.toml`.

## Coding style

Lazy means efficient, not careless. Follow YAGNI: don't build abstractions,
config, or flexibility until something needs them. Stdlib over a dependency;
the platform's feature over hand-rolled code; the shortest version that works.

Comments earn their place — most code shouldn't need them:

- **No module/file-level docstrings.** None. Documentation lives in classes and
  functions, not in a banner at the top of the file. (Enforced by a pre-commit
  hook: `scripts/check_docstrings.py`.)
- **Class/method docstrings are snippets, not essays** — one line on what it
  does; name params/returns only when the signature doesn't already make them
  obvious. Keep them short (the hook caps length).
- **Inline comments are rare and intentional** — reserve them for a genuinely
  non-obvious line. When a line implements a formula or algorithm, say what it
  computes and cite the source (paper, textbook, or named method), e.g.
  `# spin-orbit spillage, Eq. 2 of Liu et al. 2019`.

If an explanation is longer than the code it describes, cut the explanation.

## Current state

Scaffold stage. The entrypoints (`rl/train.py`, `eval/baseline.py`,
`eval/evaluate.py`, `viz/generate.py`) load config and print TODOs — the real
pipeline gets filled in over weeks 1–9. The one piece of real, tested logic is
the reward in `src/synth_q_mat/rl/reward.py` (see `tests/test_reward.py`).

## Gotchas

- **Three model environments by necessity** (incompatible dep stacks):
  M3GNet/matgl in the main env; MatterGen in `.venv-mattergen` (run
  `make setup-generator`); the spillage predictor trained later (matgl/MACE).
- **Local GPU needs attention** — `nvidia-smi` reported a driver/library version
  mismatch (NVML 580.159), so CUDA can't init until the box is rebooted (or the
  `nvidia` kernel modules reloaded). Heavy compute is meant for other machines
  anyway.
- **MP_API_KEY** (free, materialsproject.org) is required for `make download-data`
  Materials Project pulls; put it in `.env`. JARVIS/TQC need no key.
- **HPC details are TBD** — the professor's private cluster, not Northeastern
  Discovery. `scripts/hpc/*.sbatch` are generic SLURM templates; partition
  names, module loads, and the VASP launch command need adjusting once access
  is established.
- Keep `data/`, `models/`, `results/` out of git (already in `.gitignore`); use
  `dvc add` + `make push`.
