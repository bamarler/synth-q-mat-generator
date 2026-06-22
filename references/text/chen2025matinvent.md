---
key: chen2025matinvent
title: Accelerating inverse materials design using generative diffusion models with
  reinforcement learning
year: 2025
primary:
- rl
role:
- rl-algorithm
- generator-ft
status: adopted-template
reward_term:
- multi
domain:
- ml
tags:
- matinvent
- ppo
- kl-regularized
- diffusion-rl
- multi-objective
- 378x
summary: 'The methodological template: RL fine-tuning of a crystal diffusion model
  with a reward-weighted KL objective.'
---

# Accelerating inverse materials design using generative diffusion models with reinforcement learning

Junwu Chen1,2\*, Jeff Guo1,2, Edvin Fako1,2 and Philippe Schwaller1,2\*

<sup>1</sup>*Laboratory of Artificial Chemical Intelligence (LIAC), Institute of Chemical Sciences and Engineering, Ecole Polytechnique Fed´ erale de Lausanne (EPFL), Lausanne, Switzerland ´* <sup>2</sup>*National Centre of Competence in Research (NCCR) Catalysis, Ecole Polytechnique Fed´ erale de ´ Lausanne (EPFL), Lausanne, Switzerland*

<sup>\*</sup>*Corresponding authors. Email: junwu.chen@epfl.ch, philippe.schwaller@epfl.ch*

# ABSTRACT

Diffusion models promise to accelerate material design by directly generating novel structures with desired properties, but existing approaches typically require expensive and substantial labeled data (>10,000) and lack adaptability. Here we present MatInvent, a general and efficient reinforcement learning workflow that optimizes diffusion models for goal-directed crystal generation. For single-objective designs, MatInvent rapidly converges to target values within 60 iterations (∼ 1,000 property evaluations) across electronic, magnetic, mechanical, thermal, and physicochemical properties. Furthermore, MatInvent achieves robust optimization in design tasks with multiple conflicting properties, successfully proposing low-supply-chain-risk magnets and high-κ dielectrics. Compared to state-of-the-art methods, MatInvent exhibits superior generation performance under specified property constraints while dramatically reducing the demand for property computation by up to 378-fold. Compatible with diverse diffusion model architectures and property constraints, MatInvent could offer broad applicability in materials discovery.

Keywords: reinforcement learning, diffusion generative model, inverse materials design

# 1 Introduction

The development of novel functional materials is pivotal for accelerating scientific progress in various fields such as catalysis, microelectronics, and renewable energy [\[1–](#page-20-0)[3\]](#page-20-1). The key objective is to identify property-optimal candidates within an enormous design space. Existing methods include iterative experimental trial-and-error [\[4,](#page-20-2) [5\]](#page-20-3) and high-throughput screening [\[6,](#page-20-4) [7\]](#page-20-5). However, the brute-force screening of all possible materials is prohibitive, and is a challenge circumvented through expert-defined search spaces. Although this approach has offered success in discovering novel materials, manually constraining the search space could introduce negative bias.

Recently, generative models [\[8](#page-20-6)[–12\]](#page-21-0), particularly diffusion models [\[13–](#page-21-1)[17\]](#page-21-2), have emerged as promising frameworks for generating novel and theoretically stable inorganic crystal structures spanning the entire periodic table. Several methods, such as conditional generation by classifier-free guidance [\[14\]](#page-21-3), have been proposed to steer diffusion models toward generating materials with targeted properties [\[18–](#page-22-0)[20\]](#page-22-1). Nevertheless, these methods require substantial preexisting labeled data for model fine-tuning, limiting their generalizability and flexibility across diverse inverse design tasks.

Reinforcement learning (RL) provides a framework for optimizing generative models by iterative exploration of complex problem spaces based on feedback rewards, with potential to improve generation quality and controllability [\[21](#page-22-2)[–23\]](#page-22-3). This approach decouples learning from dense annotations by leveraging sparse or indirect reward signals, requiring substantially fewer labeled data compared to supervised fine-tuning [\[22,](#page-22-4) [23\]](#page-22-3). Notably, RL has become a mainstream strategy for optimizing SMILES-based language models to accomplish goal-directed molecular generation [\[24–](#page-22-5)[28\]](#page-23-0). Although several RL approaches have been proposed for crystal structure prediction [\[29,](#page-23-1) [30\]](#page-23-2) and composition generation of metal oxides [\[31\]](#page-23-3), RL frameworks for optimizing diffusion models in inorganic materials design remain scarce [\[32](#page-23-4)[–34\]](#page-23-5).

This work proposes MatInvent, a versatile and efficient RL workflow for optimizing pretrained diffusion models toward objective-driven crystal generation. By framing denoising generation as a multi-step decision-making problem, MatInvent leverages policy optimization with reward-weighted Kullback–Leibler (KL) regularization, including experience replay and diversity filters to enhance sample efficiency and diversity. For single-objective optimization, MatInvent demonstrates remarkable performance and flexibility across various material design tasks encompassing electronic, magnetic, mechanical, physicochemical, thermal, and synthesizability properties. Compared to conditional generation of MatterGen, our RL approach substantially reduces the requirement for labeled data while exhibiting enhanced generative performance under target property constraints. Furthermore, MatInvent achieves robust optimization in design tasks with multiple competing objectives, successfully designing magnets with low supply-chain risk and high-κ gate dielectrics. This versatility makes our approach highly appealing to researchers in materials science, chemistry, and catalysis.

# 2 Results

### 2.1 Reinforcement learning pipeline

MatInvent is an RL workflow designed for goal-directed generation of crystalline materials (Fig. [1\)](#page-4-0). In the pipeline, the diffusion model acts as the RL agent that generates novel 3D crystal structures through a T-step reverse denoising process on atomic types, atomic coordinates, and lattice matrices [\[13,](#page-21-1) [14\]](#page-21-3). The denoising process of the diffusion model can be reframed as a T-step Markov decision process (MDP) [\[22,](#page-22-4) [23\]](#page-22-3) for our online RL algorithms (see Methods). We denote the diffusion model before RL fine-tuning as the prior, which was pre-trained on large-scale unlabeled datasets of crystal structures (e.g., Alex-MP [\[14\]](#page-21-3)) and can generate diverse crystalline materials spanning over 80 elements. In each RL iteration, the diffusion model randomly generates a batch of m crystal structures. The generated structures undergo geometry optimization using universal ML interatomic potentials (MLIP) [\[35\]](#page-23-6) and their energy above hull (Ehull) is calculated. Only crystal structures that are thermodynamically Stable (Ehull < 0.1 eV/atom), Unique, and Novel (SUN) [\[14\]](#page-21-3) are retained after filtering, in which n samples are randomly selected for property evaluation and assigned corresponding rewards. The material properties and rewards can be obtained through theoretical simulations, ML predictions, and empirical calculations. The top k samples ranked by reward are used to fine-tune the diffusion model based on policy optimization with reward-weighted KL regularization (see Methods). The KL regularizer between the pre-trained and fine-tuned models is incorporated into the RL objective function to prevent reward overfitting while preserving the material knowledge acquired during pre-training [\[22\]](#page-22-4). Moreover, experience replay and the diversity filter are respectively employed to improve optimization efficiency and sample diversity of RL process, enabling faster convergence toward the target while generating novel and diverse crystal structures. Experience replay is used to improve the stability and efficiency of

<span id="page-4-0"></span>Fig. 1: MatInvent workflow for goal-directed material generation. (a) The schematic overview of MatInvent methodology. In each reinforcement learning (RL) iteration, the diffusion model acts as the RL agent to generate a batch of 3D crystal structures, which are subsequently geometrically optimized using machine learning potentials. Only valid, Stable, Unique, and Novel (SUN) structures are retained after filtering, proceeding to target property evaluation and reward assignment. High-reward samples are then used to fine-tune the diffusion model by policy optimization with reward-weighted Kullback–Leibler (KL) regularization, aided by experience replay and diversity filter to enhance sample efficiency and diversity. (b) The impact of geometry optimization (opt) and SUN filtering before property evaluation on the SUN ratio of generated structures during the RL process targeting a density of 18.0 g/cm<sup>3</sup>. (c) The effect of experience replay on the optimization efficiency of RL process targeting a density of 18.0 g/cm<sup>3</sup>. (d) The role of diversity filter in the composition diversity of generated structures during the RL process with a target density of 18.0 g/cm<sup>3</sup>.

learning by storing past high-reward crystals in a replay buffer and reusing them during RL fine-tuning [\[27,](#page-23-7) [36\]](#page-23-8). The diversity filter imposes a linear penalty on the reward of non-unique crystal structures based on the number of previous occurrences [\[37,](#page-24-0) [38\]](#page-24-1). Specifically, crystals with the same structure or composition as previously generated samples are assigned reduced rewards and subsequently removed from the replay buffer, thereby encouraging the diffusion model to explore unseen material space. Notably, MatInvent is a general-purpose RL workflow that is compatible with different diffusion model architectures (Supplementary Information section [C.1\)](#page-36-0). Unless otherwise specified, all experiments in this work use the MatterGen [\[14\]](#page-21-3) framework as the diffusion model in the RL process.

To investigate the importance of individual components in our RL workflow, ablation studies were conducted using the material design task with a target density of 18.0 g/cm<sup>3</sup> (Supplementary Information section [C.2\)](#page-38-0). Two metrics, the SUN ratio and composition diversity ratio, are defined to evaluate the generation quality and diversity of material structures from diffusion models (Supplementary Information section [B\)](#page-35-0). As shown in Fig. [1b](#page-4-0) and Supplementary Information section [C.2.1,](#page-38-1) MLIP-based geometry optimization and SUN filtering prior to property evaluation improve both the SUN ratio and composition diversity of the generated structures during the RL process. As depicted in Fig. [1c](#page-4-0) and Supplementary Information section [C.2.2,](#page-39-0) experience replay enhances the RL optimization efficiency, enabling faster convergence to the target property value with fewer property evaluations. This is particularly important for material properties that have expensive evaluation costs. Moreover, the diversity filter can encourage diffusion models to explore different chemical systems and achieve a higher diversity ratio of chemical compositions during the RL process (Fig. [1d](#page-4-0) and Supplementary Information section [C.2.3\)](#page-40-0). This facilitates the design of diverse crystal structures with target properties and prevents the RL optimization from stagnating in local minima.

# 2.2 Single property optimization

In numerous applications, such as energy storage, superconductivity, and electronic devices, the primary demand lies in designing novel materials with targeted or enhanced properties. Mat-Invent was evaluated on different inverse design tasks for single property optimization. These tasks encompass various properties of inorganic materials, including electronic, magnetic, mechanical, thermal, physicochemical, and synthesizability characteristics. The property values and corresponding rewards are derived from density functional theory (DFT) calculations (Fig.

<span id="page-6-0"></span>**Fig. 2: MatInvent performance on single property optimization.** The optimization curves (left) for reinforcement learning (RL) and visualizations of some generated crystal structures (right) on different inverse design tasks with a single target property: (a) band gap equal to 3.0 eV; (b) magnetic density higher than  $0.2 \, \text{Å}^{-3}$ ; (c) specific heat capacity exceeding  $1.5 \, \text{J/g/K}$ ; (d) minimal co-incident area (MCIA) below 80 Å<sup>2</sup> on the Si(100) substrate; (e) bulk modulus of 300 GPa; (f) total dielectric constants exceeding 80; (g) synthesizability score higher than 0.9; and (h) Herfindahl–Hirschman index (HHI) score below 1250. Ten repeat experiments were performed for tasks c–h, while three for tasks a and b. The curves show the mean of repeated experiments while the shading represents standard deviation.

[2](#page-6-0) a and b), MLIP simulations (Fig. [2](#page-6-0) c and d), or ML prediction models (Fig. [2](#page-6-0) e, f, and g). The first task (Fig. [2a](#page-6-0)) aims to generate materials with a target band gap of 3.0 eV, a key property for light-emitting devices [\[39\]](#page-24-2), photocatalysis [\[40\]](#page-24-3), and wide-bandgap semiconductor [\[41\]](#page-24-4). In the second task (Fig. [2b](#page-6-0)), the goal is to design materials with magnetic densities higher than 0.2 A˚ −3 , a prerequisite for permanent magnets [\[14\]](#page-21-3). The third task (Fig. [2c](#page-6-0)) involves generating novel inorganic compounds with specific heat capacities exceeding 1.5 J/g/K, which is crucial for thermal energy storage and high-temperature protection materials [\[42\]](#page-24-5). The fourth task (Fig. [2d](#page-6-0)) focuses on designing novel crystal structures with strong epitaxial matching to the commercially dominant Si(100) substrate, requiring a minimal co-incident area (MCIA) below 80 A˚ 2 [\[43\]](#page-24-6). A lower MCIA indicates a higher degree of matching between the thin-film material and the substrate, which is crucial for material synthesis techniques such as chemical vapor deposition and sputtering [\[43\]](#page-24-6). The fifth task (Fig. [2e](#page-6-0)) targets the generation of materials with a high bulk modulus of 300 GPa, an essential property for superhard and aerospace materials [\[44\]](#page-24-7). The sixth task (Fig. [2f](#page-6-0)) focuses on discovering new materials with high total dielectric constants exceeding 80, important for applications such as electronic devices and supercapacitors [\[45,](#page-24-8) [46\]](#page-25-0). The seventh task (Fig. [2g](#page-6-0)) investigates the generation of materials with high synthesizability scores based on feedback from the ML model [\[47\]](#page-25-1), aiming to design novel and experimentally synthesizable materials. Finally, the eighth task (Fig. [2h](#page-6-0)) is to design new materials with low supply chain risk, requiring a Herfindahl–Hirschman index (HHI) score [\[48\]](#page-25-2) below 1250 directly computed through PyMatGen [\[49\]](#page-25-3). Further details on the RL experiments and reward calculation are provided in Supplementary Information sections [E.1](#page-52-0) and [E.2,](#page-53-0) while the methods for material property evaluation are described in Supplementary Information section [D.](#page-42-0)

As shown in Fig. [2a](#page-6-0)–h, across all tasks, the average property values of the generated materials progressively approach the target values with successive RL iterations. Remarkably, within 60 iterations and ∼1000 property evaluation calls, the average property values converge to their targets for most tasks. Six more single-objective design tasks were also explored (Supplementary Information section [E.3\)](#page-54-0), involving shear modulus, Young's modulus, Pugh ratio, formation energy, crustal abundance, and price. Moreover, the property distributions of the SUN structures generated by the RL fine-tuned model displayed a clear shift, and became more concentrated around the target values compared with the pre-trained model (Supplementary Information section [E.4\)](#page-56-0). This confirms that MatInvent can optimize diffusion models and steer their generative distribution toward regions of materials with desired properties. As depicted in Supplementary Information section [E.5,](#page-57-0) most RL fine-tuned models exhibited higher SUN ratios (> 45 %) relative to the initial pretrained model (38.7 %), which can be attributed to MLIP-based structure optimization and SUN filtering prior to property evaluation. All results demonstrate that MatInvent is an efficient and general RL framework for diffusion models in single-property inverse design tasks.

We further compared MatInvent with the state-of-the-art conditional generation method (MatterGen [\[14\]](#page-21-3)), on two specific tasks: targeting materials with bandgaps of 3.0 eV and magnetic densities exceeding 0.2 A˚ −3 . For a fair comparison, all RL experiments used the same unconditional MatterGen model pre-trained on Alex-MP-20 dataset as the initial model [\[14\]](#page-21-3), and more details are in Supplementary Information section [E.6.](#page-58-0) For MatterGen's conditional generation, the pre-trained model with adapter modules undergoes fine-tuning on pre-existing and DFT labeled datasets, subsequently applying classifier-free guidance to steer crystal generation toward the desired objectives [\[14\]](#page-21-3). As illustrated in Fig. [3a](#page-9-0), MatterGen's conditional generation method requires 42,000 and 605,000 DFT-labeled data points for fine-tuning on the two tasks [\[14\]](#page-21-3), respectively, whereas MatInvent needs only 1,600 DFT calculations to obtain rewards for 100 RL iterations. MatInvent substantially reduces the expensive DFT computational costs required for model fine-tuning by factors of 26 and 378 on the two tasks, respectively (Fig. [3a](#page-9-0)). Moreover, the RL-finetuned model demonstrates approximately twice the SUN ratio compared to the conditional generation of MatterGen (Fig. [3b](#page-9-0)). In Fig. [3c](#page-9-0) and d, the property distributions of SUN structures generated by the RL-finetuned model are more concentrated around target values in both tasks, compared to those from MatterGen's conditional generation. We also evaluated the performance of MatInvent against MatterGen's conditional generation in discovering SUN structures that satisfy stringent property requirements under limited DFT calculation budgets. As shown in Fig. [3e](#page-9-0), the RL-finetuned model identified 27 SUN structures with magnetic densities exceeding 0.2 A˚ −3 within a budget of 250 DFT property calculations, outperforming MatterGen conditional generation (23 structures). Figure [3f](#page-9-0) reveals that the RL-finetuned model discovered 43 SUN structures with band gaps of 3.0 ± 0.1 eV after 250 DFT property calculations, substantially surpassing the conditional generation of MatterGen (11 structures). It is worth noting that tasks with narrow property range constraints present comparable challenges to those with extreme target properties. All results demonstrate that MatInvent achieves improved goal-directed crystal generation performance, while significantly reducing DFT computational burden. This arises because RL directly optimizes the diffusion model to maximize rewards, concentrating generation in high-reward regions, whereas conditional generation requires learning the complete conditional probability distribution over the target property.

<span id="page-9-0"></span>Fig. 3: Comparison between conditional generation and reinforcement learning. (a) Number of DFT-labeled data used for model fine-tuning in the MatInvent workflow and conditional generation of MatterGen across two inverse design tasks. (b) SUN ratios of generated structures from MatterGen conditional generation and RL-finetuned diffusion model following the MatInvent workflow. Probability density distributions of property values of SUN structures generated by RL-finetuned diffusion models and MatterGen's conditional generation, respectively, for inverse design targets of (c) magnetic density higher than 0.2 A˚ −3 and (d) band gap of 3.0 eV. Number of SUN structures satisfying property requirements discovered by MatterGen conditional generation and RL-finetuned diffusion models within 250 DFT property calculations, for targets of (e) magnetic density higher than 0.2 A˚ −3 and (f) band gap of 3 ± 0.1 eV.

# 2.3 Multiple property optimization

Most material design problems require finding structures that satisfy multiple property constraints. Two tasks were designed to evaluate the performance of MatInvent in the simultaneous

<span id="page-10-0"></span>Fig. 4: Designing permanent magnets with low supply chain risk. (a) Property distribution of SUN structures generated during the initial (0–20 loops) and final (100–120 loops) stages of RL process. (b) Mean values of target properties of SUN structures generated in each RL iteration. (c) Amount of DFT-labeled data used for model fine-tuning (left) and SUN ratios of generated structures (right) for MatterGen conditional generation and MatInvent workflow. (d) Number of SUN structures satisfying property requirements found by MatterGen conditional generation and RL-finetuned diffusion models within 200 DFT property calculations, for targets with magnetic density above 0.2 A˚ −3 and HHI score below 1500. (e) Visualizations of some SUN structures generated by RL-finetuned diffusion models, along with their chemical formula, space group, energy above hull (Ehull), magnetic density, HHI score, and synthesizability score.

optimization of multiple material properties. The first task focuses on designing novel permanent magnets with low supply chain risk, aiming to avoid the utilization of rare-earth elements [\[14\]](#page-21-3). This task can be formulated as satisfying two property constraints: (1) magnetic density higher than 0.2 A˚ −3 , and (2) Herfindahl–Hirschman index (HHI) score below 1500. An HHI score below 1500 is considered indicative of low supply chain risk [\[48\]](#page-25-2). In the RL experiments, the DFT method was employed to determine the magnetic densities of the generated structures, and PyMatGen [\[49\]](#page-25-3) was utilized to compute their HHI scores. The minimum between the scaled values of magnetic density and HHI score served as the reward for each sample during the online RL process, thereby facilitating the simultaneous optimization of both target properties (Supplementary Information section [F.2\)](#page-60-0). As illustrated in Fig. [4b](#page-10-0), the average values of both properties for the generated SUN structures gradually approached the target region with successive RL iterations, ultimately converging near the desired values after 100 iterations. In contrast to the initial phase of RL (loops 0-20), the distribution of both properties for SUN structures generated during loops 100-120 exhibited a pronounced shift and became narrowly concentrated around the target values, as demonstrated in Fig. [4a](#page-10-0). These findings demonstrate that MatInvent can iteratively optimize the diffusion model and its generation distribution for two competing material properties.

As depicted in Fig. [4c](#page-10-0), MatInvent required only 1,920 DFT property calculations for 120 RL iterations, representing a 315-fold reduction compared to the 605,000 DFT-labeled data points used for fine-tuning in MatterGen's conditional generation [\[14\]](#page-21-3). Moreover, we compared the performance between the RL-finetuned diffusion model and MatterGen's conditional generation in discovering SUN structures that meet two property requirements under limited DFT computation budgets (Supplementary Information section [F.3\)](#page-62-0). As shown in Fig. [4d](#page-10-0), the RLfinetuned diffusion model identified 14 SUN structures satisfying both property requirements under 200 DFT property calculations, outperforming MatterGen's conditional generation (11 SUN structures). Of the 14 SUN structures found by MatInvent, 78.6 % (n=11) exhibit MLpredicted synthesizability scores above 0.5, indicating potential experimental feasibility [\[47\]](#page-25-1). Some of these structures are presented in Fig. [4e](#page-10-0). Overall, all results establish that MatInvent is highly efficient for inverse design tasks with multiple objectives, achieving superior crystal generation performance, while drastically reducing DFT computational costs.

The second task aims to design novel high-κ dielectrics, critical components in numerous microelectronic devices, including central processing units (CPU), dynamic random-access

<span id="page-12-0"></span>Fig. 5: Designing novel high-κ dielectrics. (a) Property distribution of SUN structures generated during the initial (0–20 loops) and final (220–240 loops) stages of RL process. (b) Evolution of Pareto fronts across RL iterations for two conflicting material properties: dielectric constant and band gap. (c) DFT-calculated property distribution of SUN structures generated by the RL-finetuned diffusion model, which were ranked and selected based on ML predictions. (d) Number of SUN structures satisfying property requirements found by RL-finetuned diffusion models within 200 DFT property calculations, for objectives of band gap (Eg) exceeding 3.0 eV, total dielectric constant (εtotal ) surpassing 30, and figure of merit (FoM) higher than 210. (e) Distribution of DFT-computed figure of merit for generated structures by the pre-trained and RL-finetuned diffusion models. (f) Visualizations of some SUN structures generated by RLfinetuned diffusion models, along with their chemical formula, space group, energy above hull (Ehull), synthesizability score, Eg, εtotal , and FoM.

memory (DRAM), and high-frequency antennas [\[45,](#page-24-8) [46\]](#page-25-0). Their performance depends on an intricate balance between a high dielectric constant and a wide bandgap, two inversely correlated characteristics that rarely co-exist within a single material [\[45,](#page-24-8) [46\]](#page-25-0). Moreover, a high figure of merit (FoM) is desirable to suppress tunneling current, while one top experimentally reported high-κ dielectric t-HfO<sup>2</sup> exhibits an FoM of approximately 210 [\[46\]](#page-25-0). Consequently, the second task can be formulated with three optimization objectives: band gap (Eg) exceeding 3.0 eV, total dielectric constant (εtotal ) surpassing 30, and FoM higher than 210 [\[46\]](#page-25-0). Given the computational expense of DFT property evaluation, ML models were employed to predict Eg, εtotal , and corresponding FoM (= E<sup>g</sup> × εtotal) of the crystal structures generated during the RL process (Supplementary Information section [F.2\)](#page-60-0). As illustrated in Fig. [5a](#page-12-0), the structures generated during early-stage RL (0-20 iterations) fall into two primary categories: wide E<sup>g</sup> but low εtotal , and narrow E<sup>g</sup> but high εtotal . After 220 RL iterations, the property distributions of generated SUN structures exhibited a pronounced shift toward the target region with high E<sup>g</sup> and εtotal , compared to the initial phase of RL (0-20 iterations). Correspondingly, the Pareto frontier was progressively optimized during the RL iterations (Fig. [5b](#page-12-0)), continuously advancing toward the region of high E<sup>g</sup> and high εtotal . These results demonstrate that MatInvent can achieve Pareto optimization for two conflicting material properties. Subsequently, more crystal structures were generated by the RL-finetuned diffusion model and ranked according to MLpredicted FoM values, from which 200 structures were selected for DFT validation. As depicted in the Fig. [5c](#page-12-0), over 95 % of the structures exhibit a DFT-calculated E<sup>g</sup> exceeding 3.0 eV, which benefits from RL finetuning and the accurate predictive model for Eg. In contrast, only about 20 % of the structures possess a DFT-calculated εtotal higher than 30, potentially resulting from the poor εtotal prediction accuracy of ML model (Supplementary Information section [D.3.5\)](#page-48-0). Despite the inevitable errors in ML predictions, the DFT property distribution after RL finetuning shows a significant shift compared to the pre-trained model (Fig. [5e](#page-12-0)). Most structures exhibit DFT-calculated FoM exceeding 120, with four structures achieving FoM values greater than 400. Within a budget of 200 DFT property evaluations, MatInvent successfully identified 17 SUN structures that satisfy all three target criteria (Fig. [5d](#page-12-0)), highlighting its superior performance in Pareto optimization. All results demonstrate that our RL framework is capable of accomplishing challenging inverse design tasks involving multiple conflicting properties, even when using computationally less expensive rewards with limited accuracy.

# 3 Discussion

MatInvent is a versatile and efficient RL workflow that can tailor the generation of pre-trained diffusion models towards novel material structures with desired properties. This workflow implements policy optimization with reward-weighted KL regularization, experience replay, and diversity filters to ensure efficient optimization and diverse sampling. Across various singleobjective design tasks spanning from electronic, magnetic, mechanical, thermal, and physicochemical properties, to synthesizability, MatInvent demonstrates excellent optimization performance, with fast convergence to target values within approximately 60 iterations (1000 property evaluations). Moreover, MatInvent exhibits robust optimization capabilities in design tasks involving multiple conflicting objectives, even with low-precision rewards. Compared to conditional generation approaches that require substantial labeled data for target properties, Mat-Invent achieves enhanced generative performance in both single- and multi-objective inverse design while dramatically reducing the demand for property assessment.

Despite these strengths, there are several promising directions to enhance MatInvent. The current RL workflow relies on external property evaluators such as ML prediction models and DFT calculations, which may introduce noise and potential biases into the reward signal. Future extensions could incorporate uncertainty-aware or differentiable property predictors to provide informative gradients and enhance learning robustness [\[50\]](#page-25-4). Moreover, real-world materials design tasks frequently involve more than five objectives with varying degrees of importance. Techniques such as curriculum learning [\[51,](#page-25-5) [52\]](#page-25-6), Pareto set learning [\[53\]](#page-25-7), and preferenceconditioned policies [\[54\]](#page-25-8) are worth exploring to enhance MatInvent's performance in multiobjective optimization. Furthermore, material synthesis information could be integrated into the RL framework, such as precursor availability, synthetic route constraints, and synthesizability criteria, which is important for experimental validation and autonomous laboratories.

MatInvent establishes a promising paradigm for inverse material design. Its versatility enables extension to diverse material classes by employing different diffusion models, including perovskites, metal-organic frameworks [\[55,](#page-26-0) [56\]](#page-26-1), and two-dimensional materials [\[57\]](#page-26-2). The framework can be further adapted to various practical applications, such as catalysis, superconductivity [\[58\]](#page-26-3), and quantum computing [\[19\]](#page-22-6), using carefully designed RL rewards and property evaluation methods. Moreover, the integration of MatInvent into automated laboratories could offer a compelling avenue for achieving closed-loop material discovery. This general and efficient workflow is poised to attract widespread attention in the material research community.

# 4 Methods

### 4.1 Representation of crystal structures

The periodic structure of crystals arises from the repeating arrangement of atoms in 3D space, and the simplest repeating unit is defined as the unit cell. A unit cell with N atoms can be described by M = (A, X, L), where A = [a1, a2, . . . , a<sup>N</sup> ] ∈ R <sup>h</sup>×<sup>N</sup> represents the one-hot encoding of atom types, X = [x1, x2, . . . , x<sup>N</sup> ] ∈ R <sup>3</sup>×<sup>N</sup> symbolizes atoms' Cartesian coordinates, and L = [l1,l2,l3] ∈ R 3×3 expresses the crystal lattice matrix. The volume of a unit cell V = | det(L)| must be non-zero, meaning that L is invertible. Based on periodic boundary conditions, the atomic positions within the unit cell can also be described using fractional coordinates F = L <sup>−</sup><sup>1</sup>X = [f<sup>1</sup> , f<sup>2</sup> , . . . , f <sup>N</sup> ] ∈ [0, 1)<sup>3</sup>×<sup>N</sup> , which are widely used in crystallography and crystal generation. Thus, a unit cell with N atoms can also be described by M = (A,F, L), and the infinite crystal structure can be represented as

<span id="page-15-1"></span>
$$\left\{ (\boldsymbol{a}_{i}', \boldsymbol{f}_{i}') \mid \boldsymbol{a}_{i}' = \boldsymbol{a}_{i}, \boldsymbol{f}_{i}' = \boldsymbol{f}_{i} + \boldsymbol{L}\boldsymbol{k}\boldsymbol{1}_{N}, \forall \boldsymbol{k} \in \mathbb{Z}^{3} \right\}$$
(1)

where elements of k express integer translations of the lattice and 1 is a 1 × n matrix of ones to emulate broadcasting.

# 4.2 Diffusion models of crystal generation

This part provides a methodological overview of diffusion models for *de novo* crystal structure generation. The general algorithmic formulation of such models is detailed in Supplementary Information section [A.1.](#page-29-0) Implementation details for specific model architectures, including MatterGen [\[14\]](#page-21-3), can be found in their original references.

The diffusion models involve two Markov chains: a forward noising process on atom types, atomic fractional coordinates and lattice matrix, and a reverse denoising process learned by a graph neural network (GNN). For the data distribution q<sup>0</sup> of 3D crystal structures, M<sup>0</sup> ∼ q<sup>0</sup> (M0). The diffusion model approximates q<sup>0</sup> with a parameterized (θ) GNN by denoising process in the form of p<sup>θ</sup> (M0) = R p<sup>θ</sup> (M0:<sup>T</sup> ) dM1:<sup>T</sup> , where p<sup>θ</sup> (M0:<sup>T</sup> ) is calculated by

<span id="page-15-0"></span>
$$p_{\theta}\left(\mathcal{M}_{0:T}\right) = p_{T}\left(\mathcal{M}_{T}\right) \prod_{t=1}^{T} p_{\theta}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right), \tag{2}$$

and in the timestep t can be described by

$$p_{\theta}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right) = \mathcal{N}\left(\mu_{\theta}\left(\mathcal{M}_{t}, t\right), \sigma_{t}^{2} \mathbf{I}\right), \tag{3}$$

where µ<sup>θ</sup> (M<sup>t</sup> , t) is predicted by GNN.

Based on the approximate posterior q(M1:<sup>T</sup> | M0), the denoising process is the reverse of a forward noising process. In the forward process, Gaussian noises are gradually added to M according to a variance schedule β1, . . . , β<sup>T</sup> :

$$q\left(\mathcal{M}_{1:T} \mid \mathcal{M}_{0}\right) = \prod_{t=1}^{T} q\left(\mathcal{M}_{t} \mid \mathcal{M}_{t-1}\right),$$

$$q\left(\mathcal{M}_{t} \mid \mathcal{M}_{t-1}\right) = \mathcal{N}\left(\sqrt{1 - \beta_{t}} \mathcal{M}_{t-1}, \beta_{t} \boldsymbol{I}\right),$$
(4)

<span id="page-16-0"></span>Training the diffusion model is conducted by maximizing a variational lower bound on the log-likelihood E<sup>q</sup> [log p<sup>θ</sup> (M0)], which is equivalent to optimize the following objective

$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}\{0,T\}, \mathcal{M}_t \sim q(\mathcal{M}_t | \mathcal{M}_0)} \left[ \left\| \tilde{\boldsymbol{\mu}} \left( \mathcal{M}_0, t \right) - \boldsymbol{\mu}_{\theta} \left( \mathcal{M}_t, t \right) \right\|^2 \right]$$
 (5)

where µ˜ is the posterior mean of the forward process.

### 4.3 Reinforcement learning for crystal diffusion models

A Markov decision process (MDP) formalizes sequential decision-making problems. It can be characterized by a tuple (S, A, ρ0, P, R), where S denotes the state space, A represents the action space, ρ<sup>0</sup> is the initial state distribution, P specifies the transition kernel, and R defines the reward function. In every step t, the agent observes a state s<sup>t</sup> ∈ S, selects an action a<sup>t</sup> ∈ A, obtains a reward R(s<sup>t</sup> , at), and transforms into a subsequent state st+1 ∼ P(st+1|s<sup>t</sup> , at). The agent's behavior is determined by its policy π(a|s). As the agent interacts with the MDP, it generates trajectories of states and actions τ = (s0, a0, s1, a1, . . . , s<sup>T</sup> , a<sup>T</sup> ). The goal of reinforcement learning (RL) is to optimize the agent's policy π to maximize the expected cumulative reward JRL(π) over sampled trajectories:

$$\mathcal{J}_{\mathrm{RL}}(\pi) = \mathbb{E}_{\tau \sim p(\tau \mid \pi)} \left[ \sum_{t=0}^{T} R\left(s_{t}, a_{t}\right) \right]$$
 (6)

Our online RL algorithms formulates the denoising process of the diffusion model as a MDP and optimize diffusion models for crystal generation with target properties [\[22,](#page-22-4) [23\]](#page-22-3). Given a crystal diffusion model pθ(M0:<sup>T</sup> ), parameterized by θ and the final reward r(M0) of crystal M<sup>0</sup> involving single or multiple target material properties, the denoising process can be reframed as a T-step MDP:

$$s_{t} = \mathcal{M}_{T-t}, \quad a_{t} = \mathcal{M}_{T-t-1},$$

$$\rho_{0}(s_{0}) = (\mathcal{N}(0, \mathbf{I}), \mathcal{U}(0, 1)), \quad P(s_{t+1} \mid s_{t}, a_{t}) = \delta_{a_{t}},$$

$$\pi(a_{t} \mid s_{t}) = p_{\theta}(\mathcal{M}_{T-t-1} \mid \mathcal{M}_{T-t}),$$

$$R(s_{t}, a_{t}) = \begin{cases} r(s_{t+1}) = r(\mathcal{M}_{0}) & \text{if } t = T - 1, \\ 0 & \text{otherwise} \end{cases}$$
(7)

where δ<sup>y</sup> is the Dirac delta distribution with nonzero density only at y. Sampling the initial state s<sup>0</sup> of a trajectory is similar to the first stateM<sup>T</sup> = (A<sup>T</sup> ,F<sup>T</sup> , L<sup>T</sup> ) of the denoising generation, in which A<sup>T</sup> and L<sup>T</sup> are sampled from N (0, I), and F<sup>T</sup> is sampled from U(0, 1). The cumulative reward of every trajectory is equal to r (M0), because all intermediate rewards are 0, as only the final state M<sup>0</sup> of the denoising process is meaningful for computing crystal properties and rewards. Therefore, optimizing the policy π is equivalent to fine-tuning the diffusion model. The common goal in RL fine-tuning of diffusion models is to maximize the expected reward of the generated crystals:

$$\mathcal{J}_{\mathrm{RL}}(\theta) = \mathbb{E}_{p_{\theta}(\mathcal{M}_{0})} \left[ r\left( \mathcal{M}_{0} \right) \right]. \tag{8}$$

As depicted in Supplementary Information section [A.2,](#page-31-0) the gradient of this objective is

$$\nabla_{\theta} \mathcal{J}_{RL} = \mathbb{E}_{p_{\theta}(\mathcal{M}_{0:T})} \left[ r\left(\mathcal{M}_{0}\right) \sum_{t=1}^{T} \nabla_{\theta} \log p_{\theta} \left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right) \right]. \tag{9}$$

The risk of fine-tuning solely based on rewards related to target properties is that the diffusion model may overfit to the rewards and move too far away from the initial state (pre-trained model) [\[22\]](#page-22-4). To retain the broad material knowledge that the diffusion model has learned from the pre-training dataset for generating reasonable and valid crystal structures, we add the reward-weighted KL between the pre-trained and current fine-tuned models as a regularizer to the objective function according to:

$$\mathbb{E}_{p_{\theta}(\mathcal{M}_{0:T})}\left[\left(\lambda - r\left(\mathcal{M}_{0}\right)\right) \sum_{t=1}^{T} \mathrm{KL}\left(p_{\theta}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right) \| p_{\mathrm{pre}}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right)\right)\right],\tag{10}$$

where λ is a constant slightly larger than the maximum reward and more details are in Supplementary Information section [A.3.](#page-32-0) The reward weight allows the current diffusion model to appropriately move away from the pre-trained model [\[24\]](#page-22-5), thereby encouraging the model to shift its distribution to higher reward regions. Thus, the final gradient to optimize the RL objective is:

$$-\alpha r\left(\mathcal{M}_{0}\right) \sum_{t=1}^{T} \nabla_{\theta} \log p_{\theta} \left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right)$$

$$+\beta \left(\lambda - r\left(\mathcal{M}_{0}\right)\right) \sum_{t=1}^{T} \nabla_{\theta} \operatorname{KL} \left(p_{\theta} \left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right) \| p_{\operatorname{pre}} \left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right)\right)$$

$$(11)$$

where α and β are the weights of reward and KL regularization, respectively.

Experience replay The experience replay [\[27,](#page-23-7) [36\]](#page-23-8) is integrated into MatInvent, which is used to improve the stability and efficiency of RL by storing past high-reward crystals and reusing them during model fine-tuning. It breaks the correlation between consecutive experiences by sampling from a buffer of previous experiences (called the replay buffer) rather than relying only on the most recent experience. Specifically, the size of replay buffer is set to 100. When the number of stored crystal structures exceeded this capacity, only the 100 structures with the highest rewards are retained. In each RL iteration, 10 crystal structures are randomly sampled from the replay buffer and combined with the top 50 % rewarded structures from the current iteration for model fine-tuning. After fine-tuning, the top 50 % rewarded structures are added to the replay buffer, applying a deduplication criterion whereby only the highest-rewarded structure is preserved for each unique chemical composition.

Diversity filter (DF) We draw inspiration for DFs from [\[37,](#page-24-0) [38\]](#page-24-1) with small modifications. In this work. DFs linearly penalize crystals with non-unique chemical compositions based on the number of previous occurrences, which acts as a more lenient version of the unique DF, i.e., directly truncate the reward to 0 [\[37\]](#page-24-0). The reward is transformed according to the number of previous occurrences (Occ) beyond an allowed tolerance (Tol) until a hard threshold is reached, referred to as the buffer (Buff):

Filtered reward =
$$\begin{cases} r\left(\mathcal{M}_{0}\right) \times \frac{\text{Occ- Tol}}{\text{Buff- Tol}} & \text{if Tol < Occ < Buff} \\ r\left(\mathcal{M}_{0}\right) & \text{if Occ } \leq \text{Tol} \\ 0 & \text{if Occ } \geq \text{Buff} \end{cases}$$
 (12)

where Tol is set to 3 and Buff is set to 6. Selective memory purge will be triggered for material structures that remain in the replay buffer but are penalized by the diversity filter, resulting in their removal from the replay buffer [\[27\]](#page-23-7). That is, crystals with the same chemical composition as previously generated samples are assigned reduced rewards and subsequently removed from the replay buffer.

# Data availability

The diffusion models were pre-trained on the open-source Alex-MP [\[14\]](#page-21-3) or MP-20 [\[9,](#page-21-4) [59\]](#page-26-4) datasets. Checkpoint files for the diffusion model and property prediction model are available at Hugging Face <https://huggingface.co/jwchen25/MatInvent>.

# Code availability

The source code for MatInvent is available at GitHub [https://github.com/schwall](https://github.com/schwallergroup/matinvent) [ergroup/matinvent](https://github.com/schwallergroup/matinvent).

# Acknowledgments

J.C., J.G., E.F. and P.S. acknowledge support from the NCCR Catalysis (grant number 225147), a National Centre of Competence in Research funded by the Swiss National Science Foundation. J.G. (PGSD-521528389) acknowledges support from the Natural Sciences and Engineering Research Council of Canada (NSERC).

# Author contributions

J.C. contributed to methodology, model development, coding, writing, visualization, and assessment. J.G. and E.F. contributed to methodology, model design and writing. P.S. contributed to conceptualization, methodology, model design, writing, assessment, funding and project supervision.

# Competing interests

The authors declare no competing interests.

# References

- <span id="page-20-0"></span>[1] Tianyou Mou, Hemanth Somarajan Pillai, Siwen Wang, Mingyu Wan, Xue Han, Neil M Schweitzer, Fanglin Che, and Hongliang Xin. Bridging the complexity gap in computational heterogeneous catalysis with machine learning. *Nature Catalysis*, 6(2):122–136, 2023.
- [2] Zhenpeng Yao, Yanwei Lum, Andrew Johnston, Luis Martin Mejia-Mendoza, Xin Zhou, Yonggang Wen, Alan Aspuru-Guzik, Edward H Sargent, and Zhi Wei Seh. Machine ´ learning for a sustainable energy future. *Nature Reviews Materials*, 8(3):202–215, 2023.
- <span id="page-20-1"></span>[3] Sean Molesky, Zin Lin, Alexander Y Piggott, Weiliang Jin, Jelena Vuckovic, and Alejan- ´ dro W Rodriguez. Inverse design in nanophotonics. *Nature Photonics*, 12(11):659–670, 2018.
- <span id="page-20-2"></span>[4] Paul Raccuglia, Katherine C Elbert, Philip DF Adler, Casey Falk, Malia B Wenny, Aurelio Mollo, Matthias Zeller, Sorelle A Friedler, Joshua Schrier, and Alexander J Norquist. Machine-learning-assisted materials discovery using failed experiments. *Nature*, 533(7601):73–76, 2016.
- <span id="page-20-3"></span>[5] Alex Zunger. Inverse design in search of materials with target functionalities. *Nature Reviews Chemistry*, 2(4):0121, 2018.
- <span id="page-20-4"></span>[6] Nathan J Szymanski, Bernardus Rendy, Yuxing Fei, Rishi E Kumar, Tanjin He, David Milsted, Matthew J McDermott, Max Gallant, Ekin Dogus Cubuk, Amil Merchant, et al. An autonomous laboratory for the accelerated synthesis of novel materials. *Nature*, 624(7990):86–91, 2023.
- <span id="page-20-5"></span>[7] Gary Tom, Stefan P Schmid, Sterling G Baird, Yang Cao, Kourosh Darvish, Han Hao, Stanley Lo, Sergio Pablo-Garc´ıa, Ella M Rajaonson, Marta Skreta, et al. Self-driving laboratories for chemistry and materials science. *Chemical Reviews*, 124(16):9633–9732, 2024.
- <span id="page-20-6"></span>[8] Zhilong Wang and Fengqi You. Leveraging generative models with periodicity-aware, invertible and invariant representations for crystalline materials design. *Nature Computational Science*, pages 1–12, 2025.

- <span id="page-21-4"></span>[9] Tian Xie, Xiang Fu, Octavian-Eugen Ganea, Regina Barzilay, and Tommi Jaakkola. Crystal diffusion variational autoencoder for periodic material generation. In *International Conference on Learning Representations*, 2022.
- [10] Hang Xiao, Rong Li, Xiaoyang Shi, Yan Chen, Liangliang Zhu, Xi Chen, and Lei Wang. An invertible, invariant crystal representation for inverse design of solid-state materials using generative deep learning. *Nature Communications*, 14(1):7027, 2023.
- [11] Benjamin Kurt Miller, Ricky T. Q. Chen, Anuroop Sriram, and Brandon M Wood. FlowMM: Generating materials with riemannian flow matching. In *International Conference on Machine Learning*, 2024.
- <span id="page-21-0"></span>[12] Nate Gruver, Anuroop Sriram, Andrea Madotto, Andrew Gordon Wilson, C Lawrence Zitnick, and Zachary Ulissi. Fine-tuned language models generate stable inorganic materials as text. In *International Conference on Learning Representations*, 2024.
- <span id="page-21-1"></span>[13] Rui Jiao, Wenbing Huang, Peijia Lin, Jiaqi Han, Pin Chen, Yutong Lu, and Yang Liu. Crystal structure prediction by joint equivariant diffusion. *Advances in Neural Information Processing Systems*, 36:17464–17497, 2023.
- <span id="page-21-3"></span>[14] Claudio Zeni, Robert Pinsler, Daniel Zugner, Andrew Fowler, Matthew Horton, Xiang Fu, ¨ Zilong Wang, Aliaksandra Shysheya, Jonathan Crabbe, Shoko Ueda, et al. A generative ´ model for inorganic materials design. *Nature*, pages 1–3, 2025.
- [15] Zhendong Cao, Xiaoshan Luo, Jian Lv, and Lei Wang. Space group informed transformer for crystalline materials generation. *arXiv preprint arXiv:2403.15734*, 2024.
- [16] Sherry Yang, KwangHwan Cho, Amil Merchant, Pieter Abbeel, Dale Schuurmans, Igor Mordatch, and Ekin Dogus Cubuk. Scalable diffusion for materials generation. In *International Conference on Learning Representations*, 2024.
- <span id="page-21-2"></span>[17] Daniel Levy, Siba Smarak Panigrahi, Sekou-Oumar Kaba, Qiang Zhu, Kin Long Kelvin ´ Lee, Mikhail Galkin, Santiago Miret, and Siamak Ravanbakhsh. SymmCD: Symmetrypreserving crystal generation with diffusion models. In *International Conference on Learning Representations*, 2025.

- <span id="page-22-0"></span>[18] Gabe Guo, Tristan Luca Saidi, Maxwell W Terban, Michele Valsecchi, Simon JL Billinge, and Hod Lipson. Ab initio structure solutions from nanocrystalline powder diffraction data via diffusion models. *Nature Materials*, pages 1–9, 2025.
- <span id="page-22-6"></span>[19] Ryotaro Okabe, Mouyang Cheng, Abhijatmedhi Chotrattanapituk, Manasi Mandal, Kiran Mak, Denisse Cordova Carrizales, Nguyen Tuan Hung, Xiang Fu, Bowen Han, Yao Wang, ´ et al. Structural constraint integration in a generative model for the discovery of quantum materials. *Nature Materials*, pages 1–8, 2025.
- <span id="page-22-1"></span>[20] Sourav Mal, Subhankar Mishra, and Prasenjit Sen. Diffcrysgen: A score-based diffusion model for design of diverse inorganic crystalline materials. *arXiv preprint arXiv:2505.07442*, 2025.
- <span id="page-22-2"></span>[21] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-R1 incentivizes reasoning in LLMs through reinforcement learning. *Nature*, 645(8081):633–638, 2025.
- <span id="page-22-4"></span>[22] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. DPOK: Reinforcement learning for fine-tuning text-to-image diffusion models. *Advances in Neural Information Processing Systems*, 36:79858–79885, 2023.
- <span id="page-22-3"></span>[23] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In *International Conference on Learning Representations*, 2024.
- <span id="page-22-5"></span>[24] Marcus Olivecrona, Thomas Blaschke, Ola Engkvist, and Hongming Chen. Molecular denovo design through deep reinforcement learning. *Journal of Cheminformatics*, 9:1–14, 2017.
- [25] Daniel Neil, Marwin Segler, Laura Guasch, Mohamed Ahmed, Dean Plumbley, Matthew Sellwood, and Nathan Brown. Exploring deep recurrent models with reinforcement learning for molecule design. In *International Conference on Learning Representations*, 2018.
- [26] Mariya Popova, Olexandr Isayev, and Alexander Tropsha. Deep reinforcement learning for de novo drug design. *Science Advances*, 4(7):eaap7885, 2018.

- <span id="page-23-7"></span>[27] Jeff Guo and Philippe Schwaller. Augmented memory: Sample-efficient generative molecular design with reinforcement learning. *JACS Au*, 2024.
- <span id="page-23-0"></span>[28] Jeff Guo and Philippe Schwaller. Saturn: Sample-efficient generative molecular design using memory manipulation. *arXiv preprint arXiv:2405.17066*, 2024.
- <span id="page-23-1"></span>[29] Elena Zamaraeva, Christopher M Collins, Dmytro Antypov, Vladimir V Gusev, Rahul Savani, Matthew S Dyer, George R Darling, Igor Potapov, Matthew J Rosseinsky, and Paul G Spirakis. Reinforcement learning in crystal structure prediction. *Digital Discovery*, 2(6):1831–1840, 2023.
- <span id="page-23-2"></span>[30] Prashant Govindarajan, Santiago Miret, Jarrid Rector-Brooks, Mariano Phielipp, Janarthanan Rajendran, and Sarath Chandar. Learning conditional policies for crystal design using offline reinforcement learning. *Digital Discovery*, 3(4):769–785, 2024.
- <span id="page-23-3"></span>[31] Christopher Karpovich, Elton Pan, and Elsa A Olivetti. Deep reinforcement learning for inverse inorganic materials design. *npj Computational Materials*, 10(1):287, 2024.
- <span id="page-23-4"></span>[32] Junwu Chen, Jeff Guo, and Philippe Schwaller. Matinvent: Reinforcement learning for 3d crystal diffusion generation. In *ICLR 2025 AI for Accelerated Materials Design (AI4Mat) Workshop*, 2025.
- [33] Prashant Govindarajan, Mathieu Reymond, Antoine Clavaud, Mariano Phielipp, Santiago Miret, and Sarath Chandar. CrystalGym: A new benchmark for materials discovery using reinforcement learning. *arXiv preprint arXiv:2509.23156*, 2025.
- <span id="page-23-5"></span>[34] Zhendong Cao and Lei Wang. Crystalformer-RL: Reinforcement fine-tuning for materials design. *arXiv preprint arXiv:2504.02367*, 2025.
- <span id="page-23-6"></span>[35] Han Yang, Chenxi Hu, Yichi Zhou, Xixian Liu, Yu Shi, Jielan Li, Guanzhi Li, Zekun Chen, Shuizhou Chen, Claudio Zeni, et al. MatterSim: A deep learning atomistic model across elements, temperatures and pressures. *arXiv preprint arXiv:2405.04967*, 2024.
- <span id="page-23-8"></span>[36] Long-Ji Lin. Self-improving reactive agents based on reinforcement learning, planning and teaching. *Machine learning*, 8:293–321, 1992.

- <span id="page-24-0"></span>[37] Thomas Blaschke, Ola Engkvist, Jurgen Bajorath, and Hongming Chen. Memory-assisted ¨ reinforcement learning for diverse molecular de novo design. *Journal of Cheminformatics*, 12(1):68, 2020.
- <span id="page-24-1"></span>[38] Morgan Thomas, Noel M O'Boyle, Andreas Bender, and Chris De Graaf. Augmented hillclimb increases reinforcement learning efficiency for language-based de novo molecule generation. *Journal of Cheminformatics*, 14(1):68, 2022.
- <span id="page-24-2"></span>[39] Andrew M Smith and Shuming Nie. Semiconductor nanocrystals: structure, properties, and band gap engineering. *Accounts of chemical research*, 43(2):190–200, 2010.
- <span id="page-24-3"></span>[40] Jiehui Xu, Wenzhong Wang, Songmei Sun, and Lu Wang. Enhancing visible-lightinduced photocatalytic activity by coupling with wide-band-gap semiconductor: A case study on bi2wo6/tio2. *Applied Catalysis B: Environmental*, 111:126–132, 2012.
- <span id="page-24-4"></span>[41] Yuhao Zhang, Dong Dong, Qiang Li, Richard Zhang, Florin Udrea, and Han Wang. Widebandgap semiconductors and power electronics as pathways to carbon neutrality. *Nature Reviews Electrical Engineering*, 2(3):155–172, 2025.
- <span id="page-24-5"></span>[42] Joshua Ojih, Uche Onyekpe, Alejandro Rodriguez, Jianjun Hu, Chengxiao Peng, and Ming Hu. Machine learning accelerated discovery of promising thermal energy storage materials with high heat capacity. *ACS Applied Materials & Interfaces*, 14(38):43277– 43289, 2022.
- <span id="page-24-6"></span>[43] Hong Ding, Shyam S Dwaraknath, Lauren Garten, Paul Ndione, David Ginley, and Kristin A Persson. Computational approach for epitaxial polymorph stabilization through substrate selection. *ACS Applied Materials & Interfaces*, 8(20):13086–13093, 2016.
- <span id="page-24-7"></span>[44] Aria Mansouri Tehrani, Anton O Oliynyk, Marcus Parry, Zeshan Rizvi, Samantha Couper, Feng Lin, Lowell Miyagi, Taylor D Sparks, and Jakoah Brgoch. Machine learning directed search for ultraincompressible, superhard materials. *Journal of the American Chemical Society*, 140(31):9844–9853, 2018.
- <span id="page-24-8"></span>[45] Janosh Riebesell, Todd Wesley Surta, Rhys Edward Andrew Goodall, Michael William Gaultois, et al. Discovery of high-performance dielectric materials with machinelearning-guided search. *Cell Reports Physical Science*, 5(10), 2024.

- <span id="page-25-0"></span>[46] Kanghoon Yim, Youn Yong, Joohee Lee, Kyuhyun Lee, Ho-Hyun Nahm, Jiho Yoo, Chanhee Lee, Cheol Seong Hwang, and Seungwu Han. Novel high-κ dielectrics for nextgeneration electronic devices screened by automated ab initio calculations. *NPG Asia Materials*, 7(6):e190–e190, 2015.
- <span id="page-25-1"></span>[47] Jidon Jang, Juhwan Noh, Lan Zhou, Geun Ho Gu, John M Gregoire, and Yousung Jung. Synthesizability of materials stoichiometry using semi-supervised learning. *Matter*, 7(6):2294–2312, 2024.
- <span id="page-25-2"></span>[48] Michael W Gaultois, Taylor D Sparks, Christopher KH Borg, Ram Seshadri, William D Bonificio, and David R Clarke. Data-driven review of thermoelectric materials: performance and resource considerations. *Chemistry of Materials*, 25(15):2911–2920, 2013.
- <span id="page-25-3"></span>[49] Shyue Ping Ong, William Davidson Richards, Anubhav Jain, Geoffroy Hautier, Michael Kocher, Shreyas Cholia, Dan Gunter, Vincent L Chevrier, Kristin A Persson, and Gerbrand Ceder. Python materials genomics (pymatgen): A robust, open-source python library for materials analysis. *Computational Materials Science*, 68:314–319, 2013.
- <span id="page-25-4"></span>[50] Masatoshi Uehara, Yulai Zhao, Kevin Black, Ehsan Hajiramezanali, Gabriele Scalia, Nathaniel Lee Diamant, Alex M Tseng, Sergey Levine, and Tommaso Biancalani. Feedback efficient online fine-tuning of diffusion models. In *International Conference on Machine Learning*, 2024.
- <span id="page-25-5"></span>[51] Yoshua Bengio, Jer´ ome Louradour, Ronan Collobert, and Jason Weston. Curriculum ˆ learning. In *International Conference on Machine Learning*, 2009.
- <span id="page-25-6"></span>[52] Jeff Guo, Vendy Fialkova, Juan Diego Arango, Christian Margreitter, Jon Paul Janet, ´ Kostas Papadopoulos, Ola Engkvist, and Atanas Patronov. Improving de novo molecular design with curriculum learning. *Nature Machine Intelligence*, 4(6):555–563, 2022.
- <span id="page-25-7"></span>[53] Erlong Liu, Yu-Chang Wu, Xiaobin Huang, Chengrui Gao, Ren-Jian Wang, Ke Xue, and Chao Qian. Pareto set learning for multi-objective reinforcement learning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 39, pages 18789–18797, 2025.
- <span id="page-25-8"></span>[54] Yucheng Yang, Tianyi Zhou, Mykola Pechenizkiy, and Meng Fang. Preference controllable reinforcement learning with advanced multi-objective optimization. In *International Conference on Machine Learning*, 2025.

- <span id="page-26-0"></span>[55] Xiang Fu, Tian Xie, Andrew Scott Rosen, Tommi S. Jaakkola, and Jake Allen Smith. MOFDiff: Coarse-grained diffusion for metal-organic framework design. In *International Conference on Learning Representations*, 2024.
- <span id="page-26-1"></span>[56] Junkil Park, Youhan Lee, and Jihan Kim. Multi-modal conditional diffusion model using signed distance functions for metal-organic frameworks generation. *Nature Communications*, 16(1):34, 2025.
- <span id="page-26-2"></span>[57] Shihang Xu, Shibing Chu, Rami Mrad, Zhejun Zhang, Zhelin Li, Runxian Jiao, and Yuanping Chen. Discovery of 2d materials via symmetry-constrained diffusion model. *The Journal of Physical Chemistry C*, 129(14):6794–6802, 2025.
- <span id="page-26-3"></span>[58] Pawan Prakash, Jason B Gibson, Zhongwei Li, Gabriele Di Gianluca, Juan Esquivel, Eric Fuemmeler, Benjamin Geisler, Jung Soo Kim, Adrian Roitberg, Ellad B Tadmor, et al. Guided diffusion for the discovery of new superconductors. *arXiv preprint arXiv:2509.25186*, 2025.
- <span id="page-26-4"></span>[59] Anubhav Jain, Shyue Ping Ong, Geoffroy Hautier, Wei Chen, William Davidson Richards, Stephen Dacek, Shreyas Cholia, Dan Gunter, David Skinner, Gerbrand Ceder, et al. Commentary: The materials project: A materials genome approach to accelerating materials innovation. *APL Materials*, 1(1):011002, 2013.

### Supplementary Information

#### Contents

| A |     | Algorithm details and mathematical proofs                   | S3  |  |  |  |  |
|---|-----|-------------------------------------------------------------|-----|--|--|--|--|
|   | A.1 | Diffusion models of crystal generation                      | S3  |  |  |  |  |
|   | A.2 | Gradient of RL objective function<br>                       | S5  |  |  |  |  |
|   | A.3 | KL regularization<br>                                       | S6  |  |  |  |  |
|   | A.4 | Implementation details<br>                                  | S8  |  |  |  |  |
| B |     | Analysis of generated structures                            |     |  |  |  |  |
|   | B.1 | Stability, uniqueness and novelty (SUN)<br>                 | S9  |  |  |  |  |
|   | B.2 | Diversity ratio                                             | S9  |  |  |  |  |
|   | B.3 | Visualization<br>                                           | S9  |  |  |  |  |
| C |     | Generalizability and ablation study of MatInvent<br>S10     |     |  |  |  |  |
|   | C.1 | Generalizability of MatInvent on different diffusion models | S10 |  |  |  |  |
|   | C.2 | Ablation study                                              | S12 |  |  |  |  |
|   |     | C.2.1<br>Effect of geometry optimization and filter<br>     | S12 |  |  |  |  |
|   |     | C.2.2<br>Effect of experience replay                        | S13 |  |  |  |  |
|   |     | C.2.3<br>Effect of diversity filter                         | S14 |  |  |  |  |
|   |     | C.2.4<br>Effect of the weight of KL regularization<br>      | S15 |  |  |  |  |
| D |     | Material property evaluation                                |     |  |  |  |  |
|   | D.1 | DFT calculations<br>                                        | S16 |  |  |  |  |
|   | D.2 | MLIP-based simulations<br>                                  | S17 |  |  |  |  |
|   |     | D.2.1<br>Heat capacity                                      | S17 |  |  |  |  |
|   |     | D.2.2<br>Minimal co-incident area (MCIA)<br>                | S18 |  |  |  |  |
|   | D.3 | ML prediction models                                        | S19 |  |  |  |  |
|   |     | D.3.1<br>Bulk modulus                                       | S19 |  |  |  |  |
|   |     | D.3.2<br>Shear modulus<br>                                  | S20 |  |  |  |  |
|   |     | D.3.3<br>Young's modulus                                    | S21 |  |  |  |  |
|   |     | D.3.4<br>Pugh ratio                                         | S22 |  |  |  |  |
|   |     | D.3.5<br>Total dielectric constant<br>                      | S22 |  |  |  |  |
|   |     | D.3.6<br>Formation energy                                   | S23 |  |  |  |  |
|   | D.4 | Synthesizability score                                      | S23 |  |  |  |  |

|   | D.5 | HHI score<br>                                                     | S24 |
|---|-----|-------------------------------------------------------------------|-----|
|   | D.6 | Crustal abundance                                                 | S24 |
|   | D.7 | Price                                                             | S25 |
| E |     | Single property optimization                                      | S26 |
|   | E.1 | Experimental details<br>                                          | S26 |
|   | E.2 | Reward calculation<br>                                            | S27 |
|   | E.3 | More tasks                                                        | S28 |
|   | E.4 | Property distributions                                            | S30 |
|   | E.5 | SUN ratio after RL finetuning<br>                                 | S31 |
|   | E.6 | Comparison between MatterGen conditional generation and MatInvent | S32 |
| F |     | Multiple property optimization                                    | S33 |
|   | F.1 | Experimental details<br>                                          | S33 |
|   | F.2 | Reward calculation<br>                                            | S34 |
|   | F.3 | Comparison between MatterGen conditional generation and MatInvent | S36 |

# <span id="page-29-1"></span>A Algorithm details and mathematical proofs

#### <span id="page-29-0"></span>A.1 Diffusion models of crystal generation

This part provides the general algorithmic formulation of diffusion models for *de novo* crystal structure generation. These diffusion models involve two Markov chains: a forward noising process on atom types, atomic fractional coordinates and lattice matrix, and a reverse denoising process learned by a graph neural network.

A unit cell of crystal with N atoms is described byM = (A, X, L), where A = [a1, a2, . . . , a<sup>N</sup> ] ∈ R <sup>h</sup>×<sup>N</sup> represents the one-hot encoding of atom types, X = [x1, x2, . . . , x<sup>N</sup> ] ∈ R <sup>3</sup>×<sup>N</sup> symbolizes atoms' Cartesian coordinates, and L = [l1,l2,l3] ∈ R 3×3 expresses the crystal lattice matrix. Based on periodic boundary conditions, the atomic positions within the unit cell can also be described using fractional coordinates F = L <sup>−</sup>1X = [f<sup>1</sup> , f<sup>2</sup> , . . . , f <sup>N</sup> ] ∈ [0, 1)3×<sup>N</sup> , which are widely used in crystallography and crystal generation. Thus, a unit cell with N atoms can also be described by M = (A,F, L).

Diffusion on lattice L The diffusion on the continuous variable L is based on Denoising Diffusion Probabilistic Model (DDPM) [\[60\]](#page-63-0). Specifically, in the forward process, Gaussian noises are gradually added to L according to a variance schedule β1, . . . , β<sup>T</sup> :

$$q\left(\boldsymbol{L}_{1:T} \mid \boldsymbol{L}_{0}\right) = \prod_{t=1}^{T} q\left(\boldsymbol{L}_{t} \mid \boldsymbol{L}_{t-1}\right),$$

$$q\left(\boldsymbol{L}_{t} \mid \boldsymbol{L}_{t-1}\right) = \mathcal{N}\left(\boldsymbol{L}_{t} \mid \sqrt{1 - \beta_{t}} \boldsymbol{L}_{t-1}, \beta_{t} \boldsymbol{I}\right),$$
(S1)

which can be expressed as the probability conditional on the initial state:

$$q\left(\boldsymbol{L}_{t} \mid \boldsymbol{L}_{0}\right) = \mathcal{N}\left(\boldsymbol{L}_{t} \mid \sqrt{\bar{\alpha}_{t}}\boldsymbol{L}_{0}, (1 - \bar{\alpha}_{t})\boldsymbol{I}\right), \tag{S2}$$

using α<sup>t</sup> = 1 − β<sup>t</sup> and α¯<sup>t</sup> = Q<sup>t</sup> <sup>s</sup>=1 αs.

The reverse process is defined by:

$$p_{\theta}\left(\boldsymbol{L}_{0:T}\right) = p\left(\boldsymbol{L}_{T}\right) \prod_{t=1}^{T} p_{\theta}\left(\boldsymbol{L}_{t-1} \mid \boldsymbol{L}_{t}\right),$$

$$p_{\theta}\left(\boldsymbol{L}_{t-1} \mid \boldsymbol{L}_{t}\right) = \mathcal{N}\left(\boldsymbol{L}_{t-1} \mid \boldsymbol{\mu}_{\theta, \boldsymbol{L}}\left(\mathcal{M}_{t}, t\right), \sigma_{t}^{2} \boldsymbol{I}\right),$$
(S3)

where µθ,<sup>L</sup> (M<sup>t</sup> , t) = √ 1 αt L<sup>t</sup> − <sup>√</sup> βt 1−α¯<sup>t</sup> ˆϵθ,<sup>L</sup> (M<sup>t</sup> , t) and p (L<sup>T</sup> ) = N (0, I). The denoising term ˆϵθ,<sup>L</sup> (M<sup>t</sup> , t) ∈ R 3×3 is predicted by the equivariant graph neural network θ (M<sup>t</sup> , t) = θ (L<sup>t</sup> ,F<sup>t</sup> , A<sup>t</sup> , t).0 For training the denoising model θ, let L<sup>t</sup> = √ α¯tL<sup>0</sup> + √ 1 − α¯tϵ<sup>L</sup> and ϵ<sup>L</sup> ∼ N (0, I) according to Eq. [\(S2\)](#page-15-0). The training objective is denoted as the ℓ<sup>2</sup> loss between ϵ<sup>L</sup> and ˆϵθ,L:

$$\mathcal{L}_{L} = \mathbb{E}_{t \sim \mathcal{U}(1,T)} \left[ \| \boldsymbol{\epsilon}_{L} - \hat{\boldsymbol{\epsilon}}_{\theta,L} \left( \mathcal{M}_{t}, t \right) \|^{2} \right]. \tag{S4}$$

**Diffusion on atom types** A The discrete atom types A can be simply considered as continuous variables in real space  $\mathbb{R}^{h\times N}$ , facilitating the DDPM-based approach for diffusion on atom types, as also shown in [61]. Similar to diffusion on L (Eq. S1-S4), the forward process of A is denoted as

$$q\left(\mathbf{A}_{t} \mid \mathbf{A}_{0}\right) = \mathcal{N}\left(\mathbf{A}_{t} \mid \sqrt{\bar{\alpha}_{t}} \mathbf{A}_{0}, (1 - \bar{\alpha}_{t}) \mathbf{I}\right), \tag{S5}$$

the reverse process is expressed as

$$p_{\theta}\left(\boldsymbol{A}_{t-1} \mid \boldsymbol{A}_{t}\right) = \mathcal{N}\left(\boldsymbol{A}_{t-1} \mid \boldsymbol{\mu}_{\theta \mid \boldsymbol{A}}\left(\mathcal{M}_{t}, t\right), \sigma_{t}^{2} \boldsymbol{I}\right), \tag{S6}$$

and the training objective for diffusion on A is

$$\mathcal{L}_{\mathbf{A}} = \mathbb{E}_{t \sim \mathcal{U}(1,T)} \left[ \left\| \boldsymbol{\epsilon}_{\mathbf{A}} - \hat{\boldsymbol{\epsilon}}_{\theta,\mathbf{A}} \left( \mathcal{M}_{t}, t \right) \right\|^{2} \right]. \tag{S7}$$

**Diffusion on atom positions** F As the domain of fractional coordinates  $[0,1)^{3\times N}$  forms a quotient space  $\mathbb{R}^{3\times N}/\mathbb{Z}^{3\times N}$ , the score matching method [62] with wrapped normal distribution [63] is used to achieve diffusion on F [64]. The forward process is implemented by wrapped normal distribution to maintain periodic translation invariance according to:

$$q(\mathbf{F}_{t} \mid \mathbf{F}_{0}) = \mathcal{N}_{W}(\mathbf{F}_{t} \mid \mathbf{F}_{0}, \sigma_{t}^{2} \mathbf{I}), \quad \mathbf{F}_{t} = w(\mathbf{F}_{0} + \sigma_{t} \epsilon_{\mathbf{F}}), \tag{S8}$$

where  $\epsilon_{\pmb{F}} \sim \mathcal{N}(0, \pmb{I})$  and  $w(\cdot)$  retains the fractional part of the input. The noise scale  $\sigma_t$  obeys the exponential scheduler:  $\sigma_0 = 0$  and  $\sigma_t = \sigma_1 \left(\frac{\sigma_T}{\sigma_1}\right)^{\frac{t-1}{T-1}}$ , if t > 0.

For the reverse process,  $F_T \sim \mathcal{U}(0,1)$  and  $F_0$  are generated using a two-step predictor-corrector sampler method [62, 64] with the denoising term  $\hat{\epsilon}_{\theta,F}(\mathcal{M}_t,t) \in \mathbb{R}^{3\times N}$ :

$$p_{\theta}\left(\boldsymbol{F}_{t-1} \mid \mathcal{M}_{t}\right) = p_{P}\left(\boldsymbol{F}_{t-\frac{1}{2}} \middle| \boldsymbol{L}_{t}, \boldsymbol{F}_{t}, \boldsymbol{A}_{t}\right) p_{C}\left(\boldsymbol{F}_{t-1} \mid \boldsymbol{L}_{t-1}, \boldsymbol{F}_{t-\frac{1}{2}}, \boldsymbol{A}_{t-1}\right), \tag{S9}$$

where  $p_P, p_C$  are the transitions of the predictor and corrector.

The training objective from score matching of F is

$$\mathcal{L}_{F} = \mathbb{E}_{t \sim \mathcal{U}(1,T)} \left[ \lambda_{t} \| \nabla \log q \left( \mathbf{F}_{t} \mid \mathbf{F}_{0} \right) - \hat{\boldsymbol{\epsilon}}_{\theta,F} \left( \mathcal{M}_{t}, t \right) \|^{2} \right]$$
 (S10)

where  $\lambda_t = \mathbb{E}_{\boldsymbol{F}_t}^{-1} \left[ \|\nabla \log q \left( \boldsymbol{F}_t \mid \boldsymbol{F}_0 \right) \|^2 \right]$  is calculated by Monte-Carlo sampling.

#### <span id="page-31-0"></span>A.2 Gradient of RL objective function

A common goal in RL fine-tuning of diffusion models is to maximize the expected reward of the generated crystal structures:

$$\min_{\theta} \mathbb{E}_{p_{\theta}(\mathcal{M}_0)} \left[ -r\left( \mathcal{M}_0 \right) \right] \tag{S11}$$

The gradient of this objective function can be obtained as follows:

$$\nabla_{\theta} \mathbb{E}_{p_{\theta}(\mathcal{M}_{0})} \left[ -r \left( \mathcal{M}_{0} \right) \right] = \nabla_{\theta} \int p_{\theta} \left( \mathcal{M}_{0} \right) r \left( \mathcal{M}_{0} \right) d\mathcal{M}_{0}$$

$$= -\nabla_{\theta} \int \left( \int p_{\theta} \left( \mathcal{M}_{0:T} \right) d\mathcal{M}_{1:T} \right) r \left( \mathcal{M}_{0} \right) d\mathcal{M}_{0}$$

$$= -\int \nabla_{\theta} \log p_{\theta} \left( \mathcal{M}_{0:T} \right) \times r \left( \mathcal{M}_{0} \right) \times p_{\theta} \left( \mathcal{M}_{0:T} \right) d\mathcal{M}_{0:T}$$

$$= -\int \nabla_{\theta} \log \left( p_{T} \left( \mathcal{M}_{T} \right) \prod_{t=1}^{T} p_{\theta} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \right) \times r \left( \mathcal{M}_{0} \right) \times p_{\theta} \left( \mathcal{M}_{0:T} \right) d\mathcal{M}_{0:T}$$

$$= \mathbb{E}_{p_{\theta}(\mathcal{M}_{0:T})} \left[ -r \left( \mathcal{M}_{0} \right) \sum_{t=1}^{T} \nabla_{\theta} \log p_{\theta} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \right].$$
(S12)

#### <span id="page-32-0"></span>A.3 KL regularization

To prevent overfitting to task-specific rewards while preserving material knowledge that the diffusion model has learned from the pre-training dataset for generating reasonable and valid crystal structures, we augment the RL objective function with a reward-weighted KL divergence regularizer between the pre-trained and fine-tuned diffusion models. Unlike the language models in which the KL regularizer is computed over the entire sequence/trajectory (of tokens), in diffusion models, it makes sense to compute it only for the final crystal structures KL p<sup>θ</sup> (M0) ∥ppre (M0) . Unfortunately, pθ(M0) is a marginal and its closed-form is unknown. Thus, it is converted to an upper-bound format. From data processing inequality with the Markov kernel, we have

$$KL\left(p_{\theta}\left(\mathcal{M}_{0}\right)\right) \|p_{\text{pre}}\left(\mathcal{M}_{0}\right)\right) \leq KL\left(p_{\theta}\left(\mathcal{M}_{0:T}\right) \|p_{\text{pre}}\left(\mathcal{M}_{0:T}\right)\right) \tag{S13}$$

where a periodic crystal is described by M = (A, X, L). Using the Markov property of p<sup>θ</sup> and ppre, it can be converted into

$$KL\left(p_{\theta}\left(\mathcal{M}_{0:T}\right) \| p_{\text{pre}}\left(\mathcal{M}_{0:T}\right)\right) = \int p_{\theta}\left(\mathcal{M}_{0:T}\right) \times \log \frac{p_{\theta}\left(\mathcal{M}_{0:T}\right)}{p_{\text{pre}}\left(\mathcal{M}_{0:T}\right)} d\mathcal{M}_{0:T}$$

$$= \int p_{\theta}\left(\mathcal{M}_{0:T}\right) \log \frac{p_{\theta}\left(\mathcal{M}_{T}\right) \prod_{t=1}^{T} p_{\theta}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right)}{p_{\text{pre}}\left(\mathcal{M}_{T}\right) \prod_{t=1}^{T} p_{\text{pre}}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right)} d\mathcal{M}_{0:T}$$

$$= \int p_{\theta}\left(\mathcal{M}_{0:T}\right) \left(\log \frac{p_{\theta}\left(\mathcal{M}_{T}\right)}{p_{\text{pre}}\left(\mathcal{M}_{T}\right)} + \sum_{t=1}^{T} \log \frac{p_{\theta}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right)}{p_{\text{pre}}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right)}\right) d\mathcal{M}_{0:T}$$

$$= \mathbb{E}_{p_{\theta}\left(\mathcal{M}_{0:T}\right)} \left[\sum_{t=1}^{T} \log \frac{p_{\theta}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right)}{p_{\text{pre}}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right)}\right] = \sum_{t=1}^{T} \mathbb{E}_{p_{\theta}\left(\mathcal{M}_{0:t-1} \mid \mathcal{M}_{t:T}\right)} \left[\log \frac{p_{\theta}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right)}{p_{\text{pre}}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right)}\right]$$

$$= \sum_{t=1}^{T} \mathbb{E}_{p_{\theta}\left(\mathcal{M}_{t}\right)} \mathbb{E}_{p_{\theta}\left(\mathcal{M}_{0:t-1} \mid \mathcal{M}_{t}\right)} \left[\log \frac{p_{\theta}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right)}{p_{\text{pre}}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right)}\right] = \sum_{t=1}^{T} \mathbb{E}_{p_{\theta}\left(\mathcal{M}_{t}\right)} \left[KL\left(p_{\theta}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right) \| p_{\text{pre}}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right)\right)\right].$$
(S14)

Finally,

<span id="page-32-1"></span>
$$KL\left(p_{\theta}\left(\mathcal{M}_{0}\right)\right) \|p_{\text{pre}}\left(\mathcal{M}_{0}\right)\right) \leqslant \sum_{t=1}^{T} \mathbb{E}_{p_{\theta}\left(\mathcal{M}_{t}\right)}\left[KL\left(p_{\theta}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right) \|p_{\text{pre}}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right)\right)\right]. \tag{S15}$$

For online fine-tuning, we need to regularize P<sup>T</sup> <sup>t</sup>=1 <sup>E</sup>pθ(Mt) -KL p<sup>θ</sup> (Mt−<sup>1</sup> | Mt) ∥ppre (Mt−<sup>1</sup> | Mt) . By the product rule, we can have the gradient of objective Eq. [S15](#page-32-1)

$$\nabla_{\theta} \sum_{t=1}^{T} \mathbb{E}_{p_{\theta}(\mathcal{M}_{t})} \left[ \text{KL} \left( p_{\theta} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \| p_{\text{pre}} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \right) \right]$$

$$= \sum_{t=1}^{T} \mathbb{E}_{p_{\theta}(\mathcal{M}_{t})} \left[ \nabla_{\theta} \text{KL} \left( p_{\theta} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \| p_{\text{pre}} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \right) \right]$$

$$+ \sum_{t=1}^{T} \mathbb{E}_{p_{\theta}(\mathcal{M}_{t})} \left[ \sum_{t'>t}^{T} \nabla_{\theta} \log p_{\theta} \left( \mathcal{M}_{t'-1} \mid \mathcal{M}_{t'} \right) \cdot \text{KL} \left( p_{\theta} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \| p_{\text{pre}} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \right) \right], \tag{S16}$$

which treats the sum of conditional KL-divergences along the future trajectory as a scalar reward at each step. However, computing these sums is more inefficient than just the first term. Empirically, we find that regularizing only the first term already works well, so that

$$\nabla_{\theta} \sum_{t=1}^{T} \mathbb{E}_{p_{\theta}(\mathcal{M}_{t})} \left[ \text{KL} \left( p_{\theta} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \| p_{\text{pre}} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \right) \right]$$

$$\approx \sum_{t=1}^{T} \mathbb{E}_{p_{\theta}(\mathcal{M}_{t})} \left[ \nabla_{\theta} \text{KL} \left( p_{\theta} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \| p_{\text{pre}} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \right) \right]$$

$$\approx \mathbb{E}_{p_{\theta}(\mathcal{M}_{0:T})} \left[ \sum_{t=1}^{T} \nabla_{\theta} \text{KL} \left( p_{\theta} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \| p_{\text{pre}} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \right) \right]$$
(S17)

And the corresponding

$$\sum_{t=1}^{T} \mathbb{E}_{p_{\theta}(\mathcal{M}_{t})} \left[ \operatorname{KL} \left( p_{\theta} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \| p_{\operatorname{pre}} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \right) \right] \\
\approx \mathbb{E}_{p_{\theta}(\mathcal{M}_{0:T})} \left[ \sum_{t=1}^{T} \operatorname{KL} \left( p_{\theta} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \| p_{\operatorname{pre}} \left( \mathcal{M}_{t-1} \mid \mathcal{M}_{t} \right) \right) \right]$$
(S18)

And the reward-weighted KL regularization between the pre-trained and current fine-tuned models is defined by

$$\mathbb{E}_{p_{\theta}(\mathcal{M}_{0:T})}\left[\left(\lambda - r\left(\mathcal{M}_{0}\right)\right) \sum_{t=1}^{T} \mathrm{KL}\left(p_{\theta}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right) \| p_{\mathrm{pre}}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right)\right)\right],\tag{S19}$$

where λ is a constant slightly larger than the maximum reward. And corresponding gradient:

$$\mathbb{E}_{p_{\theta}(\mathcal{M}_{0:T})}\left[\left(\lambda - r\left(\mathcal{M}_{0}\right)\right) \sum_{t=1}^{T} \nabla_{\theta} \operatorname{KL}\left(p_{\theta}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right) \| p_{\operatorname{pre}}\left(\mathcal{M}_{t-1} \mid \mathcal{M}_{t}\right)\right)\right], \tag{S20}$$

#### <span id="page-34-0"></span>A.4 Implementation details

The diffusion models were built with PyTorch Geometric [\[65\]](#page-63-5) and PyTorch [\[66\]](#page-63-6). The crystal structures were processed using the Atomic Simulation Environment (ASE) package [\[67\]](#page-63-7) and pymatgen [\[68\]](#page-63-8). Matplotlib [\[69\]](#page-64-0) was used to draw the graphs presented in this work. During the RL process, the diffusion model was fine-tuned on a single NVIDIA H100 GPU at float32 precision.

# <span id="page-35-0"></span>B Analysis of generated structures

#### <span id="page-35-1"></span>B.1 Stability, uniqueness and novelty (SUN)

The thermodynamically Stability (Ehull < 0.1 eV/atom), Uniqueness, and Novelty (SUN) [\[70\]](#page-64-1) of each generated structure was evaluated by MatterGen's method using their Alex-MP reference dataset and code [\[70\]](#page-64-1). For DFT evaluation, Ehull was calculated by DFT energy after relaxation. For SUN ratios of different models, Ehull was obtained by MLIP energy after geometry optimization using MatterSim [\[71\]](#page-64-2) due to the high computational cost. To evaluate and compare SUN ratios across different models, each diffusion model generated 1,024 structures, with the SUN ratio defined as the percentage of structures satisfying the SUN criteria.

#### <span id="page-35-2"></span>B.2 Diversity ratio

The uniqueness index in the SUN metric reflects the ratio of unique structures in the generated crystals. However, in experimental studies, chemical composition is also an important focus. Furthermore, the unique composition ratio is often smaller than the unique structure ratio, because structures with different compositions must have different crystal structures.

During RL fine-tuning, the diffusion generative model tends to produce crystals in specific regions with high rewards, leading to reduced sample diversity. Therefore, the diversity ratio metric is defined as the ratio between the number of unique chemical compositions (u) and the number of all crystal structures (s) generated during the RL process:

Diversity ratio =
$$\frac{u}{s}$$
, (S21)

#### <span id="page-35-3"></span>B.3 Visualization

The generated crystal structures were visualized using Crystal Toolkit [\[72\]](#page-64-3) with the default setting. A uniform atomic radius 0.5 A was used, while CrystalNN bonding algorithm [ ˚ [73\]](#page-64-4) was used for chemical bonds. All 3D visualization images show atoms, bonds, unit cell and polyhedra.

# <span id="page-36-1"></span>C Generalizability and ablation study of MatInvent

#### <span id="page-36-0"></span>C.1 Generalizability of MatInvent on different diffusion models

As illustrated in Fig. [S1](#page-36-2) and [S2,](#page-37-0) we evaluted MatInvent across two different diffusion models, DiffCSP [\[64\]](#page-63-4) and EquiCSP [\[74\]](#page-64-5), on four single-property optimization tasks: (1) bulk modulus of 300 GPa; (2) MCIA below 80 A˚ 2 ; (3) HHI score below 1250; and (d) density of 18.0 g/cm<sup>3</sup> . The results demonstrate that MatInvent iteratively optimizes the diffusion models through RL, progressively driving the mean value of the target property of generated structures toward the optimization objective. For most tasks, MatInvent achieves rough convergence within 60 iterations. Thus, MatInvent is a general-purpose RL workflow that is compatible with different diffusion model architectures.

<span id="page-36-2"></span>Fig. S1: The optimization curves of MatInvent workflow using DiffCSP diffusion model on different inverse design tasks with a single target property: (a) bulk modulus of 300 GPa; (b) minimal co-incident area (MCIA) below 80 A˚ 2 ; (c) Herfindahl–Hirschman index (HHI) score below 1250; and (d) density of 18.0 g/cm<sup>3</sup> . The curves represents the average values of the target properties of the generated structures in each RL iteration, while the shading depicts standard deviation.

<span id="page-37-0"></span>Fig. S2: The optimization curves of MatInvent workflow using EquiCSP diffusion model on different inverse design tasks with a single target property: (a) bulk modulus of 300 GPa; (b) minimal co-incident area (MCIA) below 80 A˚ 2 ; (c) Herfindahl–Hirschman index (HHI) score below 1250; and (d) density of 18.0 g/cm<sup>3</sup> . The curves represents the average values of the target properties of the generated structures in each RL iteration, while the shading depicts standard deviation.

#### <span id="page-38-0"></span>C.2 Ablation study

#### <span id="page-38-1"></span>C.2.1 Effect of geometry optimization and filter

As shown in Fig. [S3a](#page-38-2), MLIP-based geometric optimization (opt) and SUN filtering prior to property assessment exert negligible influence on RL optimization efficiency. However, Fig. [S3b](#page-38-2) and c reveal that structure optimization and SUN filtering substantially enhance the compositional diversity and SUN ratio of structures generated by the diffusion model during RL iterations. This enhancement arises because opt and filter remove redundant and unstable structures, thereby promoting the diffusion model to discover unexplored material space. All results demonstrate the essential contribution of opt and filter to RL fine-tuning of diffusion models.

<span id="page-38-2"></span>Fig. S3: The RL optimization curves (a), composition diversity ratios (b), and SUN ratios (c) of generated structures during the MatInvent process with or without MLIP-based geometry optimization (opt) and SUN filter prior to property evaluation, for the target density of 18.0 g/cm<sup>3</sup> . The optimization curves represents the average values of density of the generated structures in each RL iteration.

#### <span id="page-39-0"></span>C.2.2 Effect of experience replay

As shown in Fig. [S4a](#page-39-1), experience replay clearly enhances RL optimization efficiency, enabling convergence to the target value in fewer iterations. Fig. [S4c](#page-39-1) demonstrates that experience replay exerts negligible influence on the SUN ratio of generated structures. However, experience replay will diminish the compositional diversity of crystal structures generated during RL iterations (Fig. [S4b](#page-39-1)). This motivates the adoption of a diversity filter to counteract the reduction in compositional diversity.

<span id="page-39-1"></span>Fig. S4: The RL optimization curves (a), composition diversity ratios (b), and SUN ratios (c) of generated structures during the MatInvent process with or without experience replay, for the target density of 18.0 g/cm<sup>3</sup> . The optimization curves represents the average values of density of the generated structures in each RL iteration.

#### <span id="page-40-0"></span>C.2.3 Effect of diversity filter

As shown in Fig. [S5b](#page-40-1), the diversity filter (DF) substantially enhances the compositional diversity of structures generated during RL iterations. This improvement arises because DF penalizes structures with duplicate compositions relative to previously generated samples by assigning lower rewards, thereby incentivizing the diffusion model to explore new chemical compositions and material space. Notably, Fig. [S5a](#page-40-1) and c demonstrate that DF exerts negligible influence on both RL optimization efficiency and the SUN ratio of generated crystal structures.

<span id="page-40-1"></span>Fig. S5: The RL optimization curves (a), composition diversity ratios (b), and SUN ratios (c) of generated structures during the MatInvent process with or without diversity filter (DF), for the target density of 18.0 g/cm<sup>3</sup> . The optimization curves represents the average values of density of the generated structures in each RL iteration.

#### <span id="page-41-0"></span>C.2.4 Effect of the weight of KL regularization

The risk of fine-tuning solely based on rewards related to target properties is that the diffusion model may overfit to the rewards and move too far away from the initial state (pre-trained model). To retain the broad material knowledge that the diffusion model has learned from the pre-training dataset for generating reasonable and valid crystal structures, we add the reward-weighted KL between the pre-trained and current fine-tuned models as a regularizer to the RL objective function. σ is the weight of KL regularization in the RL objective function. As illustrated in Fig. [S6,](#page-41-1) a large KL regularization weight enhances compositional diversity but impairs RL optimization efficiency. This behavior mirrors the classic exploration–exploitation trade-off, wherein an appropriate weight achieves an optimal balance between compositional diversity and optimization efficiency. In addition, the absence of KL regularization could lead to failure during RL fine-tuning, as the diffusion model deviates excessively from its pre-trained state and consequently fails to generate chemically reasonable crystal structures.

<span id="page-41-1"></span>Fig. S6: The RL optimization curves (a), composition diversity ratios (b), and SUN ratios (c) of generated structures during the MatInvent process with different weights of KL regularization, for the target density of 18.0 g/cm<sup>3</sup> . The optimization curves represents the average values of density of the generated structures in each RL iteration.

### <span id="page-42-0"></span>D Material property evaluation

#### <span id="page-42-1"></span>**D.1 DFT** calculations

Density functional theory (DFT) calculations were conducted using Vienna Ab initio Simulation Package (VASP) [75, 76] with projector augmented wave (PAW) method, accessed via atomate2 [77] and pymatgen [68] software. All computational parameters followed Materials Project [78] protocols, including the Perdew–Burke–Ernzerhof (PBE) functional within the generalized gradient approximation (GGA) [79, 80], and Hubbard U corrections [81]. The workflow for different properties is as follows:

- (1) The total energy and energy above hull were calculated by the DoubleRelaxMaker and StaticMaker classes in the atomate2 software [77] with default settings. Specifically, this workflow includes two back-to-back relaxations and a static calculation.
- (2) The band gaps were calculated by the RelaxBandStructureMaker class in the atomate2 software [77] with default settings. Specifically, this workflow includes two back-to-back relaxations, a static calculation to generate the charge density, a non-self-consistent field calculation on a dense uniform mesh, and a non-self-consistent field calculation on the high-symmetry k-point path to generate the line mode band structure [81].
- (3) The magnetic densities of generated structures were calculated by the <code>DoubleRelaxMaker</code> and <code>StaticMaker</code> classes in the atomate2 software [77] with default settings. Specifically, this workflow includes two back-to-back relaxations and a static calculation. The magnetic density is defined as the total magnetization (magnetic moment) of the simulation unit cell divided by the volume of unit cell.
- (4) The total dielectric constants were calculated by the DoubleRelaxMaker and DielectricMaker classes in the atomate2 software [77] with default settings. Specifically, this workflow includes two back-to-back relaxations and a static calculation using density functional perturbation theory to obtain static and high-frequency (ionic) dielectric constants [82, 83]. Static dielectric constant is electronic contribution to the total dielectric constant. High-frequency (ionic) dielectric constant is ionic contribution to the total dielectric constant. The total dielectric tensor ( $3 \times 3$  matrix) can be computed by the ionic ( $\epsilon^0$ ) and electronic ( $\epsilon^0$ ) contributions:  $\epsilon_{ij} = \epsilon_{ij}^0 + \epsilon_{ij}^\infty$ . The total dielectric constant is the average of the diagonal elements of the total dielectric tensor.

For RL experiments using the DFT property evaluation, sample generation and fine-tuning of the diffusion model are performed on the GPU, while all DFT tasks are sent to the CPU cluster and run concurrently to reduce latency. Each DFT task is assigned a maximum computation time limit of 2 hours, the computed property values are sent back to the GPU cluster for RL fine-tuning, while tasks that time out or fail return a None value. Generated structures with a property value of None are deleted, but these DFT computations are still included in the property evaluation cost.

#### <span id="page-43-0"></span>D.2 MLIP-based simulations

#### <span id="page-43-1"></span>D.2.1 Heat capacity

The specific heat capacities of the generated materials at 300 K were obtained through geometry optimization and phonon calculations using FairChem software (version 1.10.0) [\[84\]](#page-65-6), quacc software [\[85\]](#page-65-7), and pre-trained machine learning potentials eSEN-30M-OAM [\[86,](#page-65-8) [87\]](#page-66-0). Specifically, the calculation workflow can be divided into the following steps:

- (1) runs a relaxation on the unit cell and atoms;
- (2) repeats the unit cell a number of times to make it sufficiently large to capture many interesting vibrational models;
- (3) generatives a number of finite displacement structures by moving each atom of the unit cell a little bit in each direction;
  - (4) running single point calculations on each of (3);
  - (5) gathering all of the calculations and calculating second derivatives (the Hessian matrix);
- (6) calculating the eigenvalues/eigenvectors of the Hessian matrix to find the vibrational modes of the material
  - (7) analyzing the thermodynamic properties of the vibrational modes.

<span id="page-43-2"></span>Fig. S7: The linear correlation between the specific heat capacity calculated based on MLIP and the experimental (a) or DFT (b) results.

As shown in the Fig. [S7](#page-43-2) and Table [S1,](#page-44-1) the specific heat capacity calculated based on MLIP has an excellent linear correlation with the experimental results or DFT results [\[88\]](#page-66-1). This shows that MLIP simulation can be used as a fast and accurate method to calculate the specific heat capacity in RL.

<span id="page-44-1"></span>Table S1: The linear correlation between the specific heat capacity calculated based on MLIP and the experimental or DFT results.

| Formula | Experimental | DFT    | MLIP  |
|---------|--------------|--------|-------|
| KCl     | 0.695        | 0.653  | 0.651 |
| NaCl    | 0.850        | 0.822  | 0.815 |
| ZnS     | 0.469        | 0.465  | 0.458 |
| LiF     | 1.562        | 1.513  | 1.58  |
| ZnO     | 0.495        | 0.501  | 0.491 |
| AlAs    | 0.490        | 0.435  | 0.432 |
| AlN     | 0.819        | 0.802  | 0.757 |
| NaF     | 1.088        | 1.084  | 1.058 |
| PbS     | 0.285        | 0.2028 | 0.203 |
| KI      | 0.313        | 0.297  | 0.297 |
| MgO     | 1.0          | 0.937  | 0.907 |

#### <span id="page-44-0"></span>D.2.2 Minimal co-incident area (MCIA)

Advanced materials synthesis techniques, including Chemical Vapor Deposition (CVD), Molecular Beam Epitaxy (MBE), and sputtering, are widely employed in contemporary materials research. A critical consideration in implementing these techniques is the rational selection of combination of films and substrates. Successful epitaxial growth of heterogeneous interfaces requires multiple factors: the crystallographic properties of both substrate and film materials, preferred cleavage planes, lattice mismatch parameters, and the resulting stress-strain fields at the interface [\[89,](#page-66-2) [90\]](#page-66-3).

The Si(100) substrate serves as the industry standard for semiconductor device fabrication due to its superior electronic properties and processing advantages. It is ideal for metal-oxide-semiconductor fieldeffect transistors (MOSFETs) in modern integrated circuits such as central processing unit (CPU) and graphics processing unit (GPU). Heteroepitaxial growth on Si(100) substrates requires precise control of film thickness and interface quality—critical parameters for device performance.

The minimal co-incident areas (MCIA) between the generated crystal structures (film) and Si(100) substrate were calculated by Zurr & McGill method [\[89,](#page-66-2) [90\]](#page-66-3) using MatterSim [\[71\]](#page-64-2) MLIP and pymatgen [\[68\]](#page-63-8) software. First, symmetry-preserving geometric relaxations of lattice vectors and atomic coordinates were performed on the conventional cells of generated crystal structures using MatterSim [\[71\]](#page-64-2) MLIP. Subsequently, MCIA was calculated from the crystallographic information of the generated structures and Si(100) using the SubstrateAnalyzer class in the pymatgen package [\[68\]](#page-63-8).

#### <span id="page-45-0"></span>D.3 ML prediction models

#### <span id="page-45-2"></span><span id="page-45-1"></span>D.3.1 Bulk modulus

Fig. S8: The linear correlation between the bulk modulus predicted by trained ALIGNN model and DFT results on the test set.

The bulk modulus of a substance measures its resistance to a uniform compression. It is defined as the ratio of the infinitesimal pressure increase to the resulting relative decrease of volume. A higher bulk modulus indicates greater resistance to compression, meaning a larger pressure is required to produce a given volume change.

We trained an ALIGNN model [\[91\]](#page-66-4) to predict the bulk modulus of the generated structures during RL process. First, all materials with 3D structures and DFT Voigt-Reuss-Hill (VRH) average bulk modulus values were extracted from Materials Project database [\[78\]](#page-65-0), which are 12,845 data points in total. The dataset was randomly split into training, validation, and test sets at a ratio of 8:1:1 for the model training. The model uses a periodic 12-nearest-neighbor graph construction method for training and prediction, with a cutoff radius of 8 A for the construction of neighbor list and bonds (edges). The ˚ mean squared error (MSE) loss function was used for model training. The model was trained for 200 epochs using the AdamW optimizer [\[92\]](#page-66-5) with a normalized weight decay of 10−<sup>5</sup> and a batch size of 32. The learning rate schedule follows the one-cycle policy [\[93\]](#page-66-6) with a maximum learning rate of 0.001. The model architecture incorporates initial atom representations (size = 92) derived from the CGCNN framework [\[94\]](#page-66-7), along with 80 initial bond radial basis function (RBF) features and 40 initial bond angle RBF features. The atom, bond, and angle feature embedding layers generate 64-dimensional inputs for subsequent graph convolution layers. The core network architecture comprises 4 ALIGNN layers and 4 graph convolution (GCN) layers, each with a hidden dimension of 256. The final atomlevel representations are aggregated through atom-wise average pooling and subsequently mapped to regression outputs via a single linear transformation layer.

As shown in the Fig. [S8,](#page-45-2) the trained model achieves a mean absolute error (MAE) of 9.57 GPa and a R<sup>2</sup> of 0.935 on the test set, showing a great linear correlation with the DFT-calculated bulk modulus.

#### <span id="page-46-1"></span><span id="page-46-0"></span>D.3.2 Shear modulus

Fig. S9: The linear correlation between the shear modulus predicted by trained ALIGNN model and DFT results on the test set.

In materials science, shear modulus is a measure of the elastic shear stiffness of a material and is

defined as the ratio of shear stress to the shear strain. A higher shear modulus indicates a more rigid material that resists shape changes, while a zero shear modulus signifies a fluid that flows freely. The value is important in fields like structural engineering, material testing, and automotive design, where it helps predict how materials will behave under twisting or shearing forces.

We trained an ALIGNN model [\[91\]](#page-66-4) to predict the shear modulus of the generated structures during RL process. First, all materials with 3D structures and DFT Voigt-Reuss-Hill (VRH) average shear modulus values were extracted from Materials Project database [\[78\]](#page-65-0), which are 12,186 data points in total. The dataset was randomly split into training, validation, and test sets at a ratio of 8:1:1 for the model training. The model uses a periodic 12-nearest-neighbor graph construction method for training and prediction, with a cutoff radius of 8 A for the construction of neighbor list and bonds (edges). The ˚ mean squared error (MSE) loss function was used for model training. The model was trained for 200 epochs using the AdamW optimizer [\[92\]](#page-66-5) with a normalized weight decay of 10−<sup>5</sup> and a batch size of 32. The learning rate schedule follows the one-cycle policy [\[93\]](#page-66-6) with a maximum learning rate of 0.001. The model architecture incorporates initial atom representations (size = 92) derived from the CGCNN framework [\[94\]](#page-66-7), along with 80 initial bond radial basis function (RBF) features and 40 initial bond angle RBF features. The atom, bond, and angle feature embedding layers generate 64-dimensional inputs for subsequent graph convolution layers. The core network architecture comprises 4 ALIGNN layers and 4 graph convolution (GCN) layers, each with a hidden dimension of 256. The final atomlevel representations are aggregated through atom-wise average pooling and subsequently mapped to regression outputs via a single linear transformation layer.

As shown in the Fig. [S9,](#page-46-1) the trained model exhibits a MAE of 9.84 GPa and a R<sup>2</sup> of 0.739 on the test set, showing a good linear correlation with the DFT-calculated shear modulus.

#### <span id="page-47-0"></span>D.3.3 Young's modulus

Young's modulus (E) quantifies the stiffness of an isotropic elastic material, defined as the ratio of uniaxial stress to strain in the elastic regime:

$$E = \frac{\sigma}{\varepsilon} = \frac{F/A}{\Delta L/L_0} \tag{S22}$$

where σ is stress, ε is strain, F is force, A is cross-sectional area, ∆L is elongation, and L<sup>0</sup> is original length.

Young's modulus characterizes the resistance to elastic deformation under tensile or compressive loads. High E values indicate high stiffness and small deformations under load (e.g., diamond: ∼ 1000 GPa, steel: ∼ 200 GPa), suitable for structural applications requiring dimensional stability. Low E values indicate high compliance and large deformations under load (e.g., rubber: about 0.01–0.1 GPa), suitable for flexible or shock-absorbing applications.

Young's modulus can be derived from the bulk modulus (K) and shear modulus (G):

$$E = \frac{9KG}{3K + G} \tag{S23}$$

In this work, the Young's modulus was calculated by ML-predicted bulk and shear modulus (Section [D.3.1](#page-45-1) and [D.3.2\)](#page-46-0).

#### <span id="page-48-1"></span>D.3.4 Pugh ratio

The Pugh ratio is a material science criterion calculated by a material's bulk modulus divided by its shear modulus to predict its ductility or brittleness. Materials with a higher Pugh ratio are more likely to be ductile and tough, while materials with a lower ratio tend to be brittle and prone to fracture. This ratio indicates whether a material is more prone to plastic deformation or fracture. In this work, the Pugh ratio was calculated by ML-predicted bulk and shear modulus (Section [D.3.1](#page-45-1) and [D.3.2\)](#page-46-0).

#### <span id="page-48-2"></span><span id="page-48-0"></span>D.3.5 Total dielectric constant

Fig. S10: The linear correlation between the total dielectric constant predicted by trained ALIGNN model and DFT results on the test set.

We trained an ALIGNN model [\[91\]](#page-66-4) to predict the total dielectric constant of the generated structures during RL process. All crystal structures with DFT-calculated total dielectric constants ranging from 0 to 150 were extracted from Materials Project database [\[78\]](#page-65-0), which are 7,227 data points in total. The dataset was randomly split into training, validation, and test sets at a ratio of 8:1:1 for the model training. The model uses a periodic 12-nearest-neighbor graph construction method for training and prediction, with a cutoff radius of 8 A for the construction of neighbor list and bonds (edges). The ˚ mean squared error (MSE) loss function was used for model training. The model was trained for 200 epochs using the AdamW optimizer [\[92\]](#page-66-5) with a normalized weight decay of 10−<sup>5</sup> and a batch size of 32. The learning rate schedule follows the one-cycle policy [\[93\]](#page-66-6) with a maximum learning rate of 0.001. The model architecture incorporates initial atom representations (size = 92) derived from the CGCNN framework [\[94\]](#page-66-7), along with 80 initial bond radial basis function (RBF) features and 40 initial bond angle RBF features. The atom, bond, and angle feature embedding layers generate 64-dimensional inputs for subsequent graph convolution layers. The core network architecture comprises 4 ALIGNN layers and 4 graph convolution (GCN) layers, each with a hidden dimension of 256. The final atomlevel representations are aggregated through atom-wise average pooling and subsequently mapped to regression outputs via a single linear transformation layer.

As shown in the Fig. [S10,](#page-48-2) the trained model exhibits a MAE of 8.0 and a R<sup>2</sup> of 0.40 on the test set. The unsatisfactory accuracy may be attributed to the small dataset size and the complex structureproperty relationships. Future work should investigate more accurate predictive models [\[95\]](#page-66-8).

#### <span id="page-49-0"></span>D.3.6 Formation energy

Formation energy (unit: eV/atom) is the energy change when one mole of a substance is formed from its constituent elements in their standard states, indicating the material's thermodynamic stability. A negative formation energy signifies that the material is stable and can be formed, while a positive value suggests it is more difficult to form. This parameter is crucial in materials science for designing stable catalysts.

In this work, the formation energy was predicted by ALIGNN model from Ref [\[91\]](#page-66-4) pretrained on data from Materials Project [\[78\]](#page-65-0).

#### <span id="page-49-1"></span>D.4 Synthesizability score

The synthesizability scores of materials were predicted by the model of Jung et al [\[96\]](#page-66-9). The model was trained by positive-unlabeled learning to predict the likelihood of synthesizing inorganic materials for any given elemental stoichiometries. This model shows a true positive rate of 83.4 % for the test dataset and an estimated precision of 83.6 %. The output probability of this model is defined as the synthesizability score, which ranges from 0 to 1. Generally, a score higher than 0.5 indicates that the crystal is likely to be experimentally synthesized.

#### <span id="page-50-0"></span>D.5 HHI score

Herfindahl-Hirschman index (HHI) score based on geological reserves for crystals were calculated by the HHIModel class in the pymatgen package [\[68\]](#page-63-8). In terms of chemical composition, HHI score based on geological and geopolitical data, provides a quantitative measure of resource economic factors for evaluating the supply and demand risk of materials. It is also a measure of how geographically confined or dispersed the elements comprising a compound are. Using the United States Geological Survey (USGS) commodity statistics, the HHI parameter can be calculated as sum squared of market fraction (χi) for a given country, based on their production (HHIP) or geological reserves (HHIR) of each element [\[97,](#page-66-10) [98\]](#page-66-11). Here, for each composition, the weighted average HHI<sup>R</sup> values were calculated using weight fraction of each element in the chemical formula. The U.S. Department of Justice and the Federal Trade Commission define markets as unconcentrated, highly concentrated, or moderately concentrated for a given commodity when HHI scores are below 1500, over 2500, and between these limits, respectively. A lower HHI is desirable, and materials with an HHI score of less than 1500 are considered to have low supply chain risk [\[97,](#page-66-10) [98\]](#page-66-11).

#### <span id="page-50-1"></span>D.6 Crustal abundance

Crustal abundance refers to the concentration of elements of a material in the Earth's crust, typically expressed in parts per million (ppm). Higher crustal abundance indicates greater natural reserves and easier extraction, ensuring long-term supply security. Abundant elements (e.g., Si: ∼ 280,000 ppm, Al: ∼82,000 ppm) are generally less expensive than rare elements (e.g., In: ∼0.05 ppm, rare earth elements: < 100 ppm), directly impacting material production costs. Materials derived from abundant elements are more sustainable for large-scale applications, reducing environmental impact and geopolitical supply risks. Low crustal abundance elements often create supply chain vulnerabilities, particularly for critical technologies (e.g., indium in displays, rare earths in magnets).

The crustal abundance of each element was obtained from the SMACT package, and the crustal abundance of a given material (compound) was calculated using by the weighted average of the crustal abundance (in ppm) based on the mass fraction of each element in the compound:

$$CA_{\text{compound}} = \sum_{i=1}^{n} CA_i \times w_i$$
 (S24)

where CAcompound is the crustal abundance of the compound, CA<sup>i</sup> is the crustal abundance of element i, wi is the mass fraction of element i in the compound, and n is the number of elements in the compound.

#### <span id="page-51-0"></span>D.7 Price

The cost of a material is one factor that must be considered in its industrial production. To this end, elemental prices obtained from market statistics are used to estimate the raw material costs of a given compound. Low costs and prices are desirable and crucial for sustainable large-scale production.

In this work, CostAnalyzer and CostDBElements classes in pymatgen [\[68\]](#page-63-8) package were used to calculate the price/cost of a given material. The price P (unit: USD/kg) was computed based on the mass fraction and price of each element in a compound:

$$P = \sum_{i=1}^{n} P_i \times w_i \tag{S25}$$

where P<sup>i</sup> is the price of element i, w<sup>i</sup> is the mass fraction of element i in the compound, and n is the number of elements in the compound.

### <span id="page-52-1"></span>E Single property optimization

#### <span id="page-52-0"></span>E.1 Experimental details

In each RL experiment, the unconditional MatterGen [\[70\]](#page-64-1) model pre-trained on Alex-MP-20 dataset [\[70\]](#page-64-1) was used as the initial generative model and RL agent. In each RL iteration:

- (1) The diffusion model randomly generates a batch of 64 crystal structures.
- (2) The generated structures undergo geometry optimization using MatterSim-v1.0.0-5M MLIP [\[71\]](#page-64-2). Only crystal structures that are thermodynamically Stable (Ehull < 0.1 eV/atom), Unique, and Novel (SUN) [\[70\]](#page-64-1) are retained. The SUN of each structure was evaluated by Alex-MP reference dataset and MatterGen's code [\[70\]](#page-64-1).
- (3) After filtering, 16 samples are randomly selected for property evaluation and assigned corresponding rewards. Due to calculation failure or timeout, the structure with reward of None will be deleted, and this calculation will also be included in the property evaluation budget.
- (4) The generated structures with rewards are fed into the diversity filter. The diversity filter imposes a linear penalty on the rewards of structures with non-unique compositions based on the number of previous occurrences. The penalized structures will be removed from the replay buffer (selective memory purge).
- (5) The top 50 % structures ranked by reward and 10 structures randomly sampled with the replay buffer of maximum size 100, are used to fine-tune the diffusion model based on policy optimization with reward-weighted Kullback–Leibler (KL) regularization (weight = 0.025). The learning rate of RL fine-tuning is 10−<sup>5</sup> , with a batch size of 16.
- (6) To update the replay buffer (maximum size = 100), the top 50 % structures were added into replay buffer, retaining only the top 100 compositionally unique structures with the highest rewards.

Diversity filter (DF) In this work. DFs linearly penalize non-unique crystal compositions based on the number of previous occurrences. The reward r is transformed according to the number of previous occurrences (Occ) beyond an allowed tolerance (Tol) until a hard threshold is reached, referred to as the buffer (Buff):

$$r' = \begin{cases} r \times \frac{\text{Occ-Tol}}{\text{Buff-Tol}} & \text{if Tol} < \text{Occ} < \text{Buff} \\ \\ r & \text{if Occ} \le \text{Tol} \\ \\ 0 & \text{if Occ} \ge \text{Buff} \end{cases}$$
 (S26)

where Tol is set to 3 and Buff is set to 6.

#### <span id="page-53-0"></span>E.2 Reward calculation

In all RL experiments, the values of the target material properties are scaled to between 0 and 1 and used as RL rewards. The property values are derived from DFT calculations, MLIP simulations, or ML prediction models (Section [D\)](#page-42-0). Higher rewards are the optimization goal of the RL process. For tasks involving maximization of target property (p) or requiring p to exceed a specified threshold τ1, the reward r is calculated according to clipped min-max normalization:

$$r = \operatorname{clip}\left(\frac{p - p_{\min}}{p_{\max} - p_{\min}}, 0, 1\right)$$
 (S27)

where pmax and pmin denote the upper and lower bounds of p, respectively, determined by the physically meaningful range of p and the task objectives. The normalized values are clipped to the range [0, 1]. Normally, pmax should be higher than τ1.

Conversely, for tasks involving minimization of p or requiring p to fall below a given threshold τ2, the reward r is computed by

$$r = \operatorname{clip}\left(\frac{p_{\max} - p}{p_{\max} - p_{\min}}, 0, 1\right). \tag{S28}$$

Normally, pmin should be lower than τ2.

Moreover, for tasks aimed at achieving a specific target value θ of p, the reward r is given by:

$$r = \operatorname{clip}\left(\frac{d_{\max} - |p - \theta|}{d_{\max} - d_{\min}}, 0, 1\right)$$
(S29)

where dmax and dmin denote the upper and lower bounds of the absolute deviation between p and θ, respectively. Normally, dmin is set to 0.

Typically, the MatInvent workflow is configured such that the mean reward at the initial iteration remains below 0.1. The parameters for reward calculation across different tasks are presented as follows:

- band gap equal to 3.0 eV: dmin = 0 and dmax = 2;
- magnetic density higher than 0.2 A˚ −3 : pmin = 0.0 and pmax = 0.25;
- specific heat capacity exceeding 1.5 J/g/K: pmin = 0.25 and pmax = 2.0;
- MCIA below 80 A˚ 2 on the Si(100) substrate: pmin = 0.0 and pmax = 180;
- bulk modulus of 300 GPa: dmin = 0 and dmax = 250;
- total dielectric constants exceeding 80: pmin = 35.0 and pmax = 120.0;
- synthesizability score higher than 0.9: pmin = 0.5 and pmax = 1.0;
- HHI score below 1250: pmin = 750 and pmax = 3250;
- density of 18.0 g/cm<sup>3</sup> : dmin = 0 and dmax = 10.

#### <span id="page-54-0"></span>E.3 More tasks

Moreover, MatInvetn was evaluated on six more inverse design tasks with a single target property: (1) maximizing Young's modulus; (2) shear modulus of 200 GPa; (3) maximizing Pugh ratio; (4) minimizing formation energy; (5) maximizing crustal abundances of elements in materials; and (6) minimizing the price of materials. The description and evaluation methods of target material properties were listed in Section [D.](#page-42-0)

Across all tasks (Fig. [S11\)](#page-55-0), as RL iterations progressed, the average property values of the generated materials continued to move toward the target. Notably, after 50 iterations (approximately 800 property evaluation costs), the average property values were close to convergence. All results demonstrate that our MatInvent is a general and highly efficient RL framework tailored for diffusion models in single property optimization tasks.

<span id="page-55-0"></span>Fig. S11: The optimization curves of MatInvent workflow on different inverse design tasks with a single target property: (a) maximize Young's modulus; (b) shear modulus of 200 GPa; (c) maximize Pugh ratio; (d) minimize formation energy; (e) maximize crustal abundances of elements in materials; and (f) minimize the price of materials. The curves represents the average values of the target properties of the generated structures in each RL iteration, while the shading depicts standard deviation.

#### <span id="page-56-0"></span>**E.4** Property distributions

In all RL experiments, the unconditional MatterGen [70] model pre-trained on Alex-MP-20 dataset was utilized as the initial model of RL process. In each task in Figure 3, the initial model and the RL-finetuned model after 100 iterations generated 1024 structures respectively. For Fig. S12 a and b, all generated structures were relaxed using the MatterSim [71] MLIP and filtered by stability, uniqueness, and novelty. Subsequently, for each model, 150 structures were randomly selected and evaluated by DFT calculations (Section D.1). For Fig. S12 c-i, all generated structures were relaxed using the MatterSim [71] MLIP, and only SUN structures were evaluated for target properties using MLIP simulation, ML prediction and pymatgen [68] (Section D).

<span id="page-56-1"></span>**Fig. S12:** Probability density distributions of property values of SUN structures generated by pretrained and RL-finetuned diffusion models for inverse design targets of (a) band gap equal to 3.0 eV; (b) magnetic density higher than 0.2 Å<sup>-3</sup>; (c) specific heat capacity exceeding 1.5 J/g/K; (d) MCIA below 80 Å<sup>2</sup> on the Si(100) substrate; (e) bulk modulus of 300 GPa; (f) total dielectric constants exceeding 80; (g) synthesizability score higher than 0.9; (h) HHI score below 1250; and (i) density of 18.0 g/cm<sup>3</sup>.

#### <span id="page-57-0"></span>E.5 SUN ratio after RL finetuning

We found that the SUN ratio of MatterGen [70] after conditional generation decreased compared to the unconditional pre-trained model. For example, the SUN ratio of conditional generation at a magnetic density of  $0.2 \text{ Å}^{-3}$  was 13.1 %, a decrease of approximately 25 %. This may be a shortcoming of conditional generation. As illustrated in Fig. S13, most RL fine-tuned models exhibited higher SUN ratios (> 45 %) relative to the initial pretrained model (38.7 %), which can be attributed to MLIP-based structure optimization and SUN filtering prior to property evaluation.

<span id="page-57-1"></span>**Fig. S13:** SUN ratios of 1024 generated structures by RL-finetuned diffusion models for different inverse design targets: band gap equal to 3.0 eV; magnetic density higher than  $0.2 \text{ Å}^{-3}$ ; specific heat capacity exceeding 1.5 J/g/K; minimal co-incident area (MCIA) below 80 Å<sup>2</sup> on the Si(100) substrate; bulk modulus of 300 GPa; total dielectric constants exceeding 80; synthesizability score higher than 0.9; Herfindahl–Hirschman index (HHI) score below 1250; and density of 18.0 g/cm<sup>3</sup>.

#### <span id="page-58-0"></span>E.6 Comparison between MatterGen conditional generation and MatInvent

For a fair comparison, our RL experiments utilized the same unconditional MatterGen model pre-trained on Alex-MP-20 dataset as the initial model [\[70\]](#page-64-1). For the task targeting materials with magnetic densities exceeding 0.2 A˚ −3 , MatterGen generated 4096 samples with their fine-tuned model by conditioning on a magnetic density value of 0.2 A˚ −3 [\[70\]](#page-64-1). Similarly, the RL-finetuned diffusion model after 100 iterations using the MatInvent workflow also generated 4096 structures. All structures were relaxed using the MatterSim [\[71\]](#page-64-2) MLIP and filtered by stability, uniqueness, and novelty. Subsequently, for each model, 250 structures were randomly selected and subjected to DFT evaluation.

For the task targeting materials with bandgaps of 3.0 eV, MatterGen generated 1024 structures with their fine-tuned model by conditioning on a value of 3.0 eV for band gap [\[70\]](#page-64-1). Similarly, the RLfinetuned diffusion model after 100 iterations also generated 1024 structures. All structures were relaxed using the MatterSim [\[71\]](#page-64-2) MLIP and filtered by stability, uniqueness, and novelty. Subsequently, for each model, 250 structures were randomly selected and subjected to DFT evaluation.

# <span id="page-59-0"></span>F Multiple property optimization

#### <span id="page-59-1"></span>F.1 Experimental details

For the task targeting materials with magnetic densities exceeding 0.2 A˚ −3 and HHI score below 1500, MatInvent followed the same hyperparameters and settings as in Section [E.1,](#page-52-0) where the property evaluation size for each RL iteration is 16. The maximum number of RL iterations is set to 120. The DFT method (Section [D.1\)](#page-42-1) was employed to determine the magnetic density of the generated structures, and pymatgen [\[68\]](#page-63-8) package was utilized to compute their HHI scores (Section [D.5\)](#page-50-0).

For the task designing novel high-κ dielectrics, MatInvent uses a similar setup as in Section [E.1,](#page-52-0) but with a property evaluation size of 32 and a sampling size of 128 per RL iteration. The maximum number of RL iterations is set to 240. Due to computational expense of DFT property evaluation, ML models were employed to predict Eg, εtotal , and corresponding FoM of the generated structures during the RL process.

#### <span id="page-60-0"></span>F.2 Reward calculation

In the multiple property optimization, the numerical values of material properties are first scaled to the range of 0 to 1 through the clipped Min-Max normalization (Section [E.2\)](#page-53-0), and subsequently, the scaled values of different properties are combined to calculate the final reward. Additionally, standardization methods could be investigated in future studies as an alternative approach for scaling the numerical values of material properties.

(1) In the task targeting materials with magnetic densities exceeding 0.2 A˚ −3 and HHI score below 1500, the scaled value (sm) of magnetic density (pm) was calculated by

$$s_m = \operatorname{clip}\left(\frac{p_m}{0.25}, 0, 1\right). \tag{S30}$$

And the scaled value (sh) of HHI score (ph) was computed by

$$s_h = \text{clip}\left(\frac{3250 - p_h}{3250 - 750}, 0, 1\right).$$
 (S31)

The final reward r was calculated by

$$r = \min(s_m, s_h). \tag{S32}$$

(2) For the task designing novel high-κ dielectrics, the scaled value (sg) of band gap (Eg) was calculated by

$$s_g = \text{clip}\left(\frac{E_g - 0.5}{3.5 - 0.5}, 0, 1\right).$$
 (S33)

And the scaled value (st) of total dielectric constant (εtotal) was calculated by

$$s_t = \operatorname{clip}\left(\frac{\varepsilon_{\text{total}} - 25}{50 - 25}, 0, 1\right). \tag{S34}$$

And the scaled value (s<sup>f</sup> ) of figure of merit (FoM = E<sup>g</sup> × εtotal) was calculated by

$$s_f = \text{clip}\left(\frac{\text{FoM} - 10}{250 - 10}, 0, 1\right).$$
 (S35)

The final reward r was calculated by

$$r = \alpha s_g + \beta s_t + (1 - \alpha - \beta)s_f \tag{S36}$$

where α = β = 0.1 normally. In addition, α = β = 0 is also an effective parameter for Pareto optimization.

The trained ALIGNN model [\[91\]](#page-66-4) in Section [D.3.5](#page-48-0) was used to predict the total dielectric constants of the generated structures during RL process. In addition, we trained an ALIGNN model [\[91\]](#page-66-4) to predict the band gap of the generated structures during RL process. First, all crystal structures with DFT band gap values were extracted from Materials Project database [\[78\]](#page-65-0), which are 154,839 data points in total. The dataset was randomly split into training, validation, and test sets at a ratio of 8:1:1 for the model training. The model uses a periodic 12-nearest-neighbor graph construction method for training and prediction, with a cutoff radius of 8 A for the construction of neighbor list and bonds (edges). The ˚ mean squared error (MSE) loss function was used for model training. The model was trained for 150 epochs using the AdamW optimizer [\[92\]](#page-66-5) with a normalized weight decay of 10−<sup>5</sup> and a batch size of 64. The learning rate schedule follows the one-cycle policy [\[93\]](#page-66-6) with a maximum learning rate of 0.001. The model architecture incorporates initial atom representations (size = 92) derived from the CGCNN framework [\[94\]](#page-66-7), along with 80 initial bond radial basis function (RBF) features and 40 initial bond angle RBF features. The atom, bond, and angle feature embedding layers generate 64-dimensional inputs for subsequent graph convolution layers. The core network architecture comprises 4 ALIGNN layers and 4 graph convolution (GCN) layers, each with a hidden dimension of 256. The final atomlevel representations are aggregated through atom-wise average pooling and subsequently mapped to regression outputs via a single linear transformation layer.

<span id="page-61-0"></span>As shown in the Fig. [S14,](#page-61-0) the trained model achieves a mean absolute error (MAE) of 0.29 eV and a R<sup>2</sup> of 0.78 on the test set, showing a good linear correlation with the DFT-calculated band gap.

Fig. S14: The linear correlation between the band gaps predicted by trained ALIGNN model and DFT results on the test set.

#### <span id="page-62-0"></span>F.3 Comparison between MatterGen conditional generation and MatInvent

For a fair comparison, our RL experiments utilized the same unconditional MatterGen model pre-trained on Alex-MP-20 dataset as the initial model [\[70\]](#page-64-1). For the task targeting materials with magnetic densities exceeding 0.2 A˚ −3 and HHI score below 1500, MatterGen generated 10,240 structures with their finetuned model by jointly conditioning on a magnetic density value of 0.2 A˚ −3 and an HHI score of 1500 [\[70\]](#page-64-1). Similarly, the RL-finetuned diffusion model after 120 iterations using the MatInvent workflow also generated 10,240 structures. All structures were relaxed using the MatterSim [\[71\]](#page-64-2) MLIP and filtered by stability, uniqueness, and novelty. Subsequently, the HHI scores of all structures were calculated using the pymatgen package [\[68\]](#page-63-8), and structures with HHI scores higher than 1500 were removed. Finally, for each model, 200 structures were randomly selected and subjected to DFT evaluation.

# Supplementary References

- <span id="page-63-0"></span>[60] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In *Conference on Neural Information Processing Systems*, 2020.
- <span id="page-63-1"></span>[61] Emiel Hoogeboom, Vıctor Garcia Satorras, Clement Vignac, and Max Welling. Equivariant dif- ´ fusion for molecule generation in 3d. In *International conference on machine learning*, pages 8867–8887. PMLR, 2022.
- <span id="page-63-2"></span>[62] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In *International Conference on Learning Representations*, 2021.
- <span id="page-63-3"></span>[63] Valentin De Bortoli, Emile Mathieu, MJ Hutchinson, James Thornton, Yee Whye Teh, and Arnaud Doucet. Riemannian score-based generative modelling. In *Conference on Neural Information Processing Systems*, 2022.
- <span id="page-63-4"></span>[64] Rui Jiao, Wenbing Huang, Peijia Lin, Jiaqi Han, Pin Chen, Yutong Lu, and Yang Liu. Crystal structure prediction by joint equivariant diffusion. *Advances in Neural Information Processing Systems*, 36:17464–17497, 2023.
- <span id="page-63-5"></span>[65] Matthias Fey and Jan E. Lenssen. Fast graph representation learning with PyTorch Geometric. In *ICLR Workshop on Representation Learning on Graphs and Manifolds*, 2019.
- <span id="page-63-6"></span>[66] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32, 2019.
- <span id="page-63-7"></span>[67] Ask Hjorth Larsen, Jens Jørgen Mortensen, Jakob Blomqvist, Ivano E Castelli, Rune Christensen, Marcin Dułak, Jesper Friis, Michael N Groves, Bjørk Hammer, Cory Hargus, et al. The atomic simulation environment—a python library for working with atoms. *Journal of Physics: Condensed Matter*, 29(27):273002, 2017.
- <span id="page-63-8"></span>[68] Shyue Ping Ong, William Davidson Richards, Anubhav Jain, Geoffroy Hautier, Michael Kocher, Shreyas Cholia, Dan Gunter, Vincent L Chevrier, Kristin A Persson, and Gerbrand Ceder. Python materials genomics (pymatgen): A robust, open-source python library for materials analysis. *Computational Materials Science*, 68:314–319, 2013.

- <span id="page-64-0"></span>[69] John D Hunter. Matplotlib: A 2d graphics environment. *Computing in science & engineering*, 9(03):90–95, 2007.
- <span id="page-64-1"></span>[70] Claudio Zeni, Robert Pinsler, Daniel Zugner, Andrew Fowler, Matthew Horton, Xiang Fu, Zi- ¨ long Wang, Aliaksandra Shysheya, Jonathan Crabbe, Shoko Ueda, et al. A generative model for ´ inorganic materials design. *Nature*, pages 1–3, 2025.
- <span id="page-64-2"></span>[71] Han Yang, Chenxi Hu, Yichi Zhou, Xixian Liu, Yu Shi, Jielan Li, Guanzhi Li, Zekun Chen, Shuizhou Chen, Claudio Zeni, et al. MatterSim: A deep learning atomistic model across elements, temperatures and pressures. *arXiv preprint arXiv:2405.04967*, 2024.
- <span id="page-64-3"></span>[72] Matthew Horton, Jimmy-Xuan Shen, Jordan Burns, Orion Cohen, Franc¸ois Chabbey, Alex M Ganose, Rishabh Guha, Patrick Huck, Hamming Howard Li, Matthew McDermott, et al. Crystal toolkit: A web app framework to improve usability and accessibility of materials science research algorithms. *arXiv preprint arXiv:2302.06147*, 2023.
- <span id="page-64-4"></span>[73] Hillary Pan, Alex M Ganose, Matthew Horton, Muratahan Aykol, Kristin A Persson, Nils ER Zimmermann, and Anubhav Jain. Benchmarking coordination number prediction algorithms on inorganic crystal structures. *Inorganic chemistry*, 60(3):1590–1603, 2021.
- <span id="page-64-5"></span>[74] Peijia Lin, Pin Chen, Rui Jiao, Qing Mo, Cen Jianhuan, Wenbing Huang, Yang Liu, Dan Huang, and Yutong Lu. Equivariant diffusion for crystal structure prediction. In *International Conference on Machine Learning*, 2024.
- <span id="page-64-6"></span>[75] Georg Kresse and Jurgen Furthm ¨ uller. Efficient iterative schemes for ab initio total-energy calcu- ¨ lations using a plane-wave basis set. *Physical Review B*, 54(16):11169, 1996.
- <span id="page-64-7"></span>[76] Georg Kresse and Jurgen Furthm ¨ uller. Efficiency of ab-initio total energy calculations for metals ¨ and semiconductors using a plane-wave basis set. *Computational Materials Science*, 6(1):15–50, 1996.
- <span id="page-64-8"></span>[77] Alex M. Ganose, Hrushikesh Sahasrabuddhe, Mark Asta, Kevin Beck, Tathagata Biswas, Alexander Bonkowski, Joana Bustamante, Xin Chen, Yuan Chiang, Daryl C. Chrzan, Jacob Clary, Orion A. Cohen, Christina Ertural, Max C. Gallant, Janine George, Sophie Gerits, Rhys E. A. Goodall, Rishabh D. Guha, Geoffroy Hautier, Matthew Horton, T. J. Inizan, Aaron D. Kaplan, Ryan S. Kingsbury, Matthew C. Kuner, Bryant Li, Xavier Linn, Matthew J. McDermott, Rohith Srinivaas Mohanakrishnan, Aakash A. Naik, Jeffrey B. Neaton, Shehan M. Parmar, Kristin A. Persson, Guido Petretto, Thomas A. R. Purcell, Francesco Ricci, Benjamin Rich, Janosh Riebesell, Gian-Marco Rignanese, Andrew S. Rosen, Matthias Scheffler, Jonathan Schmidt, Jimmy-Xuan

- Shen, Andrei Sobolev, Ravishankar Sundararaman, Cooper Tezak, Victor Trinquet, Joel B. Varley, Derek Vigil-Fowler, Duo Wang, David Waroquiers, Mingjian Wen, Han Yang, Hui Zheng, Jiongzhi Zheng, Zhuoying Zhu, and Anubhav Jain. Atomate2: modular workflows for materials science. *Digital Discovery*, 2025.
- <span id="page-65-0"></span>[78] Anubhav Jain, Shyue Ping Ong, Geoffroy Hautier, Wei Chen, William Davidson Richards, Stephen Dacek, Shreyas Cholia, Dan Gunter, David Skinner, Gerbrand Ceder, et al. Commentary: The materials project: A materials genome approach to accelerating materials innovation. *APL Materials*, 1(1):011002, 2013.
- <span id="page-65-1"></span>[79] John P Perdew and Yue Wang. Accurate and simple analytic representation of the electron-gas correlation energy. *Physical Review B*, 45(23):13244, 1992.
- <span id="page-65-2"></span>[80] John P Perdew, Kieron Burke, and Matthias Ernzerhof. Generalized gradient approximation made simple. *Physical Review Letters*, 77(18):3865, 1996.
- <span id="page-65-3"></span>[81] Anubhav Jain, Geoffroy Hautier, Charles J Moore, Shyue Ping Ong, Christopher C Fischer, Tim Mueller, Kristin A Persson, and Gerbrand Ceder. A high-throughput infrastructure for density functional theory calculations. *Computational Materials Science*, 50(8):2295–2310, 2011.
- <span id="page-65-4"></span>[82] Ioannis Petousis, Wei Chen, Geoffroy Hautier, Tanja Graf, Thomas D Schladt, Kristin A Persson, and Fritz B Prinz. Benchmarking density functional perturbation theory to enable highthroughput screening of materials for dielectric constant and refractive index. *Physical Review B*, 93(11):115151, 2016.
- <span id="page-65-5"></span>[83] Ioannis Petousis, David Mrdjenovich, Eric Ballouz, Miao Liu, Donald Winston, Wei Chen, Tanja Graf, Thomas D Schladt, Kristin A Persson, and Fritz B Prinz. High-throughput screening of inorganic compounds for the discovery of novel dielectric and optical materials. *Scientific Data*, 4(1):1–12, 2017.
- <span id="page-65-6"></span>[84] Lowik Chanussot, Abhishek Das, Siddharth Goyal, Thibaut Lavril, Muhammed Shuaibi, Morgane Riviere, Kevin Tran, Javier Heras-Domingo, Caleb Ho, Weihua Hu, et al. Open catalyst 2020 (oc20) dataset and community challenges. *ACS Catalysis*, 11(10):6059–6072, 2021.
- <span id="page-65-7"></span>[85] Andrew Rosen. quacc – the quantum accelerator, May 2025.
- <span id="page-65-8"></span>[86] Luis Barroso-Luque, Muhammed Shuaibi, Xiang Fu, Brandon M Wood, Misko Dzamba, Meng Gao, Ammar Rizvi, C Lawrence Zitnick, and Zachary W Ulissi. Open materials 2024 (omat24) inorganic materials dataset and models. *arXiv preprint arXiv:2410.12771*, 2024.

- <span id="page-66-0"></span>[87] Xiang Fu, Brandon M Wood, Luis Barroso-Luque, Daniel S Levine, Meng Gao, Misko Dzamba, and C Lawrence Zitnick. Learning smooth and expressive interatomic potentials for physical property prediction. *arXiv preprint arXiv:2502.12147*, 2025.
- <span id="page-66-1"></span>[88] Joshua Ojih, Uche Onyekpe, Alejandro Rodriguez, Jianjun Hu, Chengxiao Peng, and Ming Hu. Machine learning accelerated discovery of promising thermal energy storage materials with high heat capacity. *ACS Applied Materials & Interfaces*, 14(38):43277–43289, 2022.
- <span id="page-66-2"></span>[89] Hong Ding, Shyam S Dwaraknath, Lauren Garten, Paul Ndione, David Ginley, and Kristin A Persson. Computational approach for epitaxial polymorph stabilization through substrate selection. *ACS Applied Materials & Interfaces*, 8(20):13086–13093, 2016.
- <span id="page-66-3"></span>[90] A Zur and TC McGill. Lattice match: An application to heteroepitaxy. *Journal of Applied Physics*, 55(2):378–386, 1984.
- <span id="page-66-4"></span>[91] Kamal Choudhary and Brian DeCost. Atomistic line graph neural network for improved materials property predictions. *npj Computational Materials*, 7(1):185, 2021.
- <span id="page-66-5"></span>[92] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*, 2017.
- <span id="page-66-6"></span>[93] Leslie N Smith. A disciplined approach to neural network hyper-parameters: Part 1–learning rate, batch size, momentum, and weight decay. *arXiv preprint arXiv:1803.09820*, 2018.
- <span id="page-66-7"></span>[94] Tian Xie and Jeffrey C Grossman. Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. *Physical Review Letters*, 120(14):145301, 2018.
- <span id="page-66-8"></span>[95] Zetian Mao, Wenwen Li, and Jethro Tan. Dielectric tensor prediction for inorganic materials using latent information from preferred potential. *npj Computational Materials*, 10(1):265, 2024.
- <span id="page-66-9"></span>[96] Jidon Jang, Juhwan Noh, Lan Zhou, Geun Ho Gu, John M Gregoire, and Yousung Jung. Synthesizability of materials stoichiometry using semi-supervised learning. *Matter*, 7(6):2294–2312, 2024.
- <span id="page-66-10"></span>[97] Michael W Gaultois, Taylor D Sparks, Christopher KH Borg, Ram Seshadri, William D Bonificio, and David R Clarke. Data-driven review of thermoelectric materials: performance and resource considerations. *Chemistry of Materials*, 25(15):2911–2920, 2013.
- <span id="page-66-11"></span>[98] Aria Mansouri Tehrani, Leila Ghadbeigi, Jakoah Brgoch, and Taylor D Sparks. Balancing mechanical properties and sustainability in the search for superhard materials. *Integrating materials and manufacturing innovation*, 6(1):1–8, 2017.
