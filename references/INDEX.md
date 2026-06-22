
# Reference index

Master index of the paper corpus. Grep this (or `tags.yaml`) by facet first, then open `text/<key>.md`; open the PDF in `papers/` only for figures or equations.

Status: `adopted` = in the pipeline · `rejected` = evaluated and dropped · `candidate` = under consideration · `context` = background/cautionary. Reward terms: `topology`, `superconductivity` (DOS at E_F), `stability` (formation energy), `composition` (light-element).

## Generators
- `zeni2025mattergen` [adopted] — Diffusion crystal generator; the pretrained policy backbone this project fine-tunes. (MatterGen)
- `jiao2023diffcsp` [rejected] — Joint lattice+coordinate diffusion generator; rejected (old torch/Python pins, weak property conditioning). (DiffCSP)

## RL methods & theory
- `chen2025matinvent` [adopted-template] — The methodological template: RL fine-tuning of a crystal diffusion model with a reward-weighted KL objective. (MatInvent)
- `schulman2017ppo` [candidate] — PPO; the default policy-optimization choice, treated as one option not a settled default.
- `haarnoja2018sac` [candidate] — Off-policy actor-critic; alternative when DFT reward evaluations are expensive (replay reuse). (SAC)
- `bengio2021gflownet` [candidate] — Sample proportional to reward for diverse candidate batches rather than one optimum. (GFlowNets)
- `black2023ddpo` [adopted-lineage] — Denoising as an MDP + policy-gradient RL; direct template for RL-tuning a diffusion generator. (DDPO)
- `fan2023dpok` [adopted-lineage] — KL-regularized reward-maximizing diffusion fine-tuning; cited by MatInvent. (DPOK)
- `olivecrona2017reinvent` [lineage] — Prior-anchored RL fine-tuning of a generative model; ancestor of MatInvent's KL anchoring. (REINVENT)
- `roijers2013morl` [context] — MORL survey; framing for combining the four reward terms (weighted sum vs Pareto/Chebyshev).
- `sutton2018rl` [context] — Canonical RL reference; root anchor for the methods lineage. (Sutton & Barto)

## Predictors & potentials (reward models)
- `chen2022m3gnet` [adopted] — Fast formation-energy / stability reward and geometry relaxer. (M3GNet) — reward: stability
- `batatia2022mace` [candidate] — Equivariant MLIP; candidate backbone for the spillage regressor. (MACE) — reward: topology
- `choudhary2021alignn` [rejected] — Line-graph property predictor; rejected (needs DGL, no modern builds). (ALIGNN) — reward: topology
- `miksch2021mlsimulations` [context] — Tutorial review on building ML interatomic potentials; methodology for training the spillage regressor.
- `choudhary2019spillage` [adopted-concept] — Defines spin-orbit spillage as a cheap band-inversion topology proxy. — reward: topology
- `choudhary2021magnetic_spillage` [adopted-concept] — Spillage + ML screening of ~40k materials; how to train a spillage classifier on JARVIS. — reward: topology

## Quantum-materials physics
- `bradlyn2017tqc` [context-validation] — Framework linking band topology to bonding/symmetry; for validating topological candidates. (TQC) — reward: topology
- `zhang2009topological` [context] — Establishes the Bi2Se3 family of 3D topological insulators; lab-grown workhorse compounds. — reward: topology
- `lutchyn2018majorana` [context] — How Majorana/topological-superconductor platforms are actually grown; synthesizability reality check. — reward: superconductivity

## Synthesizability & real-world discovery
- `cao2025crystalformer` [candidate-baseline] — Closest analogue: RL fine-tuning of a crystal generator with stability + property rewards. (CrystalFormer-RL) — reward: stability
- `jang2020synthesizability` [candidate] — Learned crystal-likeness score (CLscore) via PU learning; synthesizability reward beyond hull. — reward: stability
- `aykol2018thermolimit` [context] — Amorphous limit as a thermodynamic ceiling on metastable synthesis; justifies the stability reward. — reward: stability
- `bartel2022stability` [context] — Systematizes DFT stability; warns thermodynamic stability is an imperfect synthesizability proxy. — reward: stability
- `merchant2023gnome` [context] — ~381k claimed newly-stable structures via GNN stability + substitution. (GNoME) — reward: stability
- `szymanski2023alab` [context] — Robotic synthesis of GNoME targets; note the Jan 2026 Nature correction on novelty. (A-Lab)
- `cheetham2024gnom_critique` [context-cautionary] — Critique of GNoME novelty/credibility; trivial superstructures, implausible compositions.
- `leeman2024challenges` [context-cautionary] — Critique of A-Lab: false ordering on disordered phases; no genuinely new materials demonstrated.
- `cheetham2025digitalage` [context] — Provenance of new inorganic compounds; computational predictions are candidates, not discoveries.

## Data sources
- `choudhary2020jarvis` [adopted] — Source of spillage labels and the topological subset for the topology regressor. (JARVIS-DFT) — reward: topology
- `jain2013materialsproject` [adopted] — Convex-hull / stability reference and a MatterGen training source. (Materials Project) — reward: stability

## Simulation & DFT tooling
- `ong2013pymatgen` [adopted] — Structure handling, phase-diagram/hull analysis, DFT I/O glue. (pymatgen)
- `larsen2017ase` [adopted] — Uniform calculator interface driving relaxations and DFT orchestration. (ASE)
- `giannozzi2009qe` [adopted] — Phonon / dynamical-stability calculations on top candidates. (Quantum ESPRESSO)

---

Not in this corpus (intentionally): VASP method papers (Kresse & Furthmüller / Kresse & Joubert) and the DFT foundations (Hohenberg-Kohn, Kohn-Sham). Add them under `tooling` / a `dft-roots` group if you want the validation stack fully cited.
