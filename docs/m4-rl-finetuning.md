# M4 — RL fine-tuning of MatterGen (research + port plan, DEFERRED)

Status: **deferred**. The warmup currently stops at the gradient-free baseline
harness (M3). This doc captures everything needed to resume M4 cold.

## What M4 is

Fine-tune the MatterGen generator with RL so it produces **more stable**
(lower energy-above-hull) **light-element** crystals — the first real
"train the generator toward a property" step. Single-objective (stability)
warmup; the multi-objective topology/dos reward comes later.

## Where the warmup stands now (M0–M3, all committed)

- **M0** reward split into `src/synth_q_mat/rl/reward/` (weights/stability/combine).
- **M1** `EnergyAboveHullEvaluator` — M3GNet-Eform + pymatgen hull. **Key
  deviation from the original plan:** the hull is built from M3GNet-Eform of MP
  structures (`src/synth_q_mat/data/hull_cache.py` → `data/processed/hull_eform.parquet`),
  NOT MP DFT energies — mixing M3GNet candidates against an MP-DFT hull gave
  >1 eV/atom errors. Self-consistent after the fix.
- **M2** `MatterGenGenerator` — CLI subprocess to `.venv-mattergen`. Made a
  hot-swappable factory (`build_generator`, `models.generator`).
- **M3** baseline harness (`eval/baseline.py`): conditional **light** generation
  → light gate → M3GNet+hull score → reward → `candidates.json` + metrics →
  MLflow → seaborn figure. **Deviation:** uses MatterGen's pretrained
  `chemical_system` conditional model, sampling a light chemsys per batch
  (noble gases excluded), because base MatterGen emits ~0% light-only structures.
  Verified working: 12 light structures across Li-Mg/Li-O/Na-Si, rewards 0–0.24.

These pieces stay. M4 adds the in-process RL loop; the M1 evaluator + M2 CLI
adapter remain for the gradient-free baseline/eval/analysis paths.

## The method (MatInvent — our template paper)

- Paper: *Accelerating inverse materials design …with RL* (MatInvent),
  arXiv:2511.03112 (2025). It is the proposal's RL template.
- **Open-source**: <https://github.com/schwallergroup/matinvent> ·
  checkpoints <https://huggingface.co/jwchen25/MatInvent>. Hydra-based,
  `python main.py pipeline=mat_invent model=mattergen reward=<name>`.
- Method: reframe MatterGen's T-step denoising as an MDP; policy gradient with
  **reward-weighted KL** to the frozen prior (paper Eq. 11). ~60–120 iterations,
  ~1,000–1,900 property evals — 26–378× less data than MatterGen's own
  conditional fine-tuning.
- Implementation nuance (matters for the port): the loss does **not** store
  denoising trajectories. `ft_step` re-noises each retained structure at every
  timestep and computes a reward-weighted score-matching loss + an **L2-KL**
  surrogate between agent and frozen-prior score predictions. Cost ≈
  `timesteps(1000) × epochs(3)` score-model passes per retained sample — an
  overnight/HPC job, not interactive.

## Decision: PORT the RL core into our repo (not use as external tool)

Vendor MatInvent's generator-agnostic loop + the MatterGen-specific pieces into
`src/`, wire to our Hydra/DVC/MLflow.

### Portable core (file map from MatInvent recon)

| MatInvent file | Class/fn | Role |
|---|---|---|
| `pipeline/base.py` | `ReinL` | RL base: model_suite, reward, sampler, replay, `reward_step` |
| `pipeline/mat_invent.py` | `MatInvent(ReinL)` | the loop: `run_rl`→`rl_step`→ sample/reward/topk/replay/`ft_step` |
| `memory/replay_buffer.py` | `ReplayBuffer` | top-K replay (dedup by composition) |
| `memory/ltm.py` | `LongTimeMem` | all-time memory + diversity filter |
| `pipeline/filters/opt_filter.py` | `OptFilter` | SUN filter (validity/novel/unique/stable) |
| `models/suite/base.py` | `ModelSuite` | generator interface the loop depends on |
| `models/suite/mattergen.py` | — | MatterGen ckpt load, ChemGraph, collate |
| `models/mattergen/pl_module.py` | `MatterGenModule` | **`add_noise`, `calc_sample_loss`, `calc_kl_reg`** — the seam |
| `models/mattergen/{loss,sample,dataset}.py` | — | SampleLoss, PC sampler, finetune dataset |
| `rewards/reward.py` | `Reward` | reward aggregator (scale + reduce) |
| `rewards/calculators/base.py` | `Calculator` | **our reward plugs in here** |
| `main.py` | — | Hydra entrypoint, `instantiate(cfg.pipeline, …)` |

DiffCSP is wired as a second `ModelSuite` (`models/suite/diffcsp.py`) — a clean
reference for what a generator port looks like against the same loop.

### Critical architectural implications

1. **The generator must be importable in-process** — gradients flow through
   `diffusion_module.model` in `ft_step`. Our M2 CLI-subprocess adapter does NOT
   serve the RL path. The RL loop runs in a py3.10 / torch==2.2.1+cu118 /
   numpy<2 env with MatterGen importable. The reward, not the generator, is what
   gets pushed to a subprocess if it ever conflicts.
2. **Env: extend `.venv-mattergen`, do not duplicate.** It already has the
   py3.10 + torch 2.2.1+cu118 + MatterGen stack. Add `mattersim` + the PyG stack
   (`torch_geometric`/`scatter`/`sparse`/`cluster`) + MatterGen's
   `MetricsEvaluator` reference dataset. Incremental ≈ **1–2 GB** vs ~5–7 GB for
   a standalone env (avoids duplicating ~2.5–3 GB of torch+CUDA). MatInvent's
   `env.yml` proves mattergen + mattersim + PyG coexist. (Disk was tight — check
   free space first.)
3. **Reward source DECISION (locked): MatterSim in-env.** Use the
   energy-above-hull MatInvent already computes in-env — MatterSim relaxation +
   MatterGen `MetricsEvaluator` against the MP2020-corrected reference. No
   cross-env boundary; a more "official" Ehull than our M3GNet-hull. Our M1
   M3GNet evaluator stays for the gradient-free baseline/eval analysis in the
   main env. NOTE: in MatInvent, stability is a pre-scoring **SUN filter**, not a
   reward — to use Ehull as the gradient signal, add a `Calculator` that wraps
   MatterSim/MetricsEvaluator and returns per-structure Ehull (target:
   descending), or reward = `formation_to_score(Ehull)`.
4. **Light-element constraint:** condition on the `mattergen_chemical_system`
   model (the loop's `MatterGenSampler` supports `properties_to_condition_on`)
   with a sampled light chemsys, plus the SUN/validity filter. Reuse our
   `sample_light_chemsys` (`rl/reward/composition.py`).
5. The three model-coupled methods (`add_noise`, `calc_sample_loss`,
   `calc_kl_reg`) are MatterGen-specific — **reuse MatInvent's**; only reimplement
   if swapping the generator.

### Reward interface to implement against

```python
# rewards/calculators/base.py
class Calculator:
    def calc(self, samples: tuple[list[Structure], str], label: str) -> np.ndarray:
        # one value per structure; NaN => that sample is dropped
```
`Reward.scoring((structures, xyz_path)) -> (rewards, prop_dict, failed_mask)`;
applies `linear_scaling` per target/min/max then reduces (mean/min/weight). For
multi-objective later: `reduce: weight` with per-prop `weight:`.

### Config knobs (Hydra)

- `base.yaml`: `rl_epoch: 120` (iterations), `eval_size: 16` (scored/loop),
  `sample_cfg.filter` = SUN `OptFilter`.
- `pipeline/mat_invent.yaml`: `topk_ratio: 0.5`; `replay` (buffer 100, sample 10,
  cutoff 0.1); `div_filter` (tol 3, buff 6); `finetune_cfg` (accum_steps 50,
  epochs 3, **`sigma: 0.025` = KL weight α/β**).
- `model/mattergen.yaml`: `model_name`; `sample_cfg.batch_size = eval_size*4`
  (oversample, then filter+cap); `finetune_cfg` (timesteps 1000, lr 1e-5).

### Env gotchas

- Hard pins from MatterGen: numpy<2, torch==2.2.1+cu118, py3.10. `matgl` likely
  can't coexist in-process → that's why the RL reward uses MatterSim, not our
  M3GNet evaluator. fairchem (eSEN) lives in its own conda env, bridged by a
  `conda run` + extxyz-file subprocess (the pattern to copy if a reward ever
  needs an incompatible dep).

## Resume checklist

1. Confirm free disk; extend `.venv-mattergen` with `mattersim` + PyG stack.
2. Re-clone MatInvent to a stable path (the recon clone went to `/dev/shm`,
   RAM-backed — gone on reboot): `git clone https://github.com/schwallergroup/matinvent`.
3. Vendor the portable core (table above) into `src/synth_q_mat/rl/` + the
   MatterGen suite; adapt to our Hydra config (`conf/`) and MLflow logging.
4. Write a MatterSim-Ehull `Calculator`; wire reward → `formation_to_score`.
5. Light: `mattergen_chemical_system` + `sample_light_chemsys` + SUN filter.
6. `rl/train.py` drives the loop; add a `train` DVC stage; checkpoints → MLflow.
7. Small local smoke (few iterations, eval_size small) → confirm mean reward /
   on-hull fraction trend up vs the M3 baseline; then a full overnight/HPC run.

## Relevant papers to add to `references/`

- **arXiv:2504.13048** — *Design Topological Materials by Reinforcement
  Fine-Tuned Generative Model* — directly this project's end goal.
- **arXiv:2504.02367** — CrystalFormer-RL (RL fine-tuning, different generator).
- **arXiv:2511.03112** — MatInvent (already templated here).
