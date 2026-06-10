# synth-q-mat-generator

**Synthesis of topologically constrained quantum materials via reinforcement learning.**

RL-guided discovery of crystal structures that jointly satisfy three criteria
most pipelines optimize one at a time:

1. **Topological protection** — quantum states protected by symmetry
2. **Superconducting tendency** — high density of states at the Fermi level
3. **Synthesizability** — low formation energy, makeable in a lab

A MatInvent-style PPO policy fine-tunes a pretrained crystal generator
(**MatterGen**) against a multi-objective reward built from a topology predictor
(a spin-orbit **spillage** regressor) and a stability evaluator (**M3GNet**),
then promising candidates are validated with DFT (VASP) and phonon calculations
(Quantum ESPRESSO) on an HPC cluster.

> PEAK Summit Fellowship project (Northeastern University) with Prof. Yan.
> See `references/` for the full proposal and timeline.

## Stack

| Concern | Choice |
| --- | --- |
| Language | Python 3.13 |
| Package manager | [uv](https://docs.astral.sh/uv/) (CUDA vs CPU torch selected by platform) |
| ML | PyTorch, stable-baselines3 (PPO), gymnasium |
| Materials | pymatgen, ASE, matgl (M3GNet), jarvis-tools, mp-api |
| Generator | MatterGen — isolated env (`scripts/setup_mattergen.sh`) |
| Topology predictor | spillage regressor trained on JARVIS (matgl/MACE; DGL-free) |
| Experiment tracking | MLflow (self-hosted) |
| Data / model versioning | DVC (local remote now; S3 later) |
| Visualization | matplotlib + seaborn (publication), plotly (interactive) |
| Lint / format | Ruff |
| License | MIT |

## Quickstart

```bash
# Prerequisites: uv (https://astral.sh/uv) and Python 3.13
make setup            # sync deps, init DVC + local remote, create .env
# edit .env: add MP_API_KEY (free, https://next-gen.materialsproject.org/api)
make check-deps       # confirm python/uv/dvc/aws/CUDA
make test             # run unit tests
make download-models  # fetch + smoke-load M3GNet (stability)
make download-data ARGS="--source jarvis"   # raw databases (MP needs MP_API_KEY)

# Crystal generator (isolated env — needs a working GPU):
make setup-generator  # clone + install MatterGen into .venv-mattergen
make gen-smoke        # generate a couple of structures to verify
```

Run `make help` for all commands.

### Models — why three different setups

The 2026 materials-ML ecosystem is fragmented across incompatible dependency
stacks, so each model is provisioned where it can actually run:

- **M3GNet (stability)** — `matgl`, PyTorch-Geometric based, lives in the main
  env. Fetched on demand by `make download-models`.
- **Topology (spillage) predictor** — no DGL-free pretrained model exists, so
  it's trained on JARVIS-DFT `spillage` labels in the research phase (matgl or
  MACE-MP). ALIGNN was dropped: it needs DGL, which has no modern torch/Python
  builds.
- **MatterGen (generator)** — pins `torch==2.2.1+cu118` / `numpy<2.0`, which
  can't co-exist with the main env, so it gets its own Python 3.10 venv and is
  driven via its CLI.

## Cross-machine workflow

This project runs across a local Linux+NVIDIA workstation, the professor's HPC
cluster, and possibly CPU-only machines. **Code** travels via git; **data and
RL checkpoints** travel via DVC so any machine restores identical state. The DVC
remote is a local directory (`.dvcstore`) for now; swap to S3 later (see the
`DVC_REMOTE` comment in the Makefile).

```bash
# On any machine:
git pull && make pull        # latest code + artifacts

# After producing new data/checkpoints:
dvc add models/checkpoints   # or data/raw, etc.
make push                    # upload contents to the DVC remote
git add models/checkpoints.dvc && git commit -m "..." && git push
```

Pretrained reward models (M3GNet) and MatterGen checkpoints re-fetch
reproducibly from their package/handle, so they are **not** DVC-tracked — only
downloaded data and our own RL checkpoints are.

MLflow runs on the workstation (`make mlflow-ui`); HPC jobs log to it by setting
`MLFLOW_TRACKING_URI` in `.env`. Every run is linked to its git commit.

## Layout

```
configs/        Hydra/YAML experiment configs (reward weights, hyperparams)
data/           DVC-tracked: raw + processed databases (gitignored)
models/         pretrained manifest + MatterGen checkpoint + RL checkpoints (gitignored)
results/        experiment outputs, DFT results (gitignored)
scripts/        data/model download, MatterGen setup, HPC SLURM templates
src/synth_q_mat/
  data/         dataset loaders + light-element filtering
  models/       wrappers for the generator + reward models
  rl/           PPO pipeline + multi-objective reward
  eval/         baseline + Pareto-front analysis
  viz/          publication and interactive figures
tests/          unit tests
```

## License

MIT — see [LICENSE](LICENSE).
