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
| Data / model versioning | DVC → S3 (`synth-q-mat-artifacts`) |
| Visualization | matplotlib + seaborn (publication), plotly (interactive) |
| Lint / format | Ruff |
| License | MIT |

## Quickstart

```bash
# Prerequisites: uv (https://astral.sh/uv) and Python 3.13
make setup            # sync deps, init DVC, create .env from template
# edit .env: add AWS S3 creds + MP_API_KEY (see CONTRIBUTING.md)
make check-deps       # confirm python/uv/dvc/aws/CUDA
make test             # run unit tests
make repro            # run the pipeline: baseline -> train -> eval -> viz
```

`make help` lists all commands. **[CONTRIBUTING.md](CONTRIBUTING.md)** is the
full how-to: setup, credentials, the DVC + Hydra workflow, sweeps (`make exp` /
`make sweep`), cross-machine resume, the MatterGen generator, and HPC.

## Layout

```
conf/           Hydra config groups (reward, train, composition, models)
dvc.yaml        pipeline stages (baseline -> train -> eval -> viz); params.yaml is composed
data/           DVC-tracked: raw + processed databases (gitignored)
models/         pretrained manifest + MatterGen checkpoint + RL checkpoints (gitignored)
results/        experiment outputs, DFT results (gitignored; metrics/ kept in git)
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
