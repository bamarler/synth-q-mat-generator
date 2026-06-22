---
key: cao2025crystalformer
title: 'CrystalFormer-RL: Reinforcement Fine-Tuning for Materials Design'
year: 2025
primary:
- generators
- rl
- synthesis
role:
- generator
- rl-algorithm
status: candidate-baseline
reward_term:
- stability
domain:
- ml
tags:
- crystalformer
- rl-fine-tuning
- energy-above-hull
- conflicting-properties
- baseline
summary: 'Closest published analogue: RL fine-tuning of a crystal generator with stability
  + property rewards. Use as baseline.'
---

### Reinforcement Fine-Tuning for Materials Design

Zhendong Cao1, 2 and Lei Wang1, 3, [∗](#page-0-0)

<sup>1</sup>*Beijing National Laboratory for Condensed Matter Physics and Institute of Physics, Chinese Academy of Sciences, Beijing 100190, China* <sup>2</sup>*School of Physical Sciences, University of Chinese Academy of Sciences, Beijing 100190, China* <sup>3</sup>*Songshan Lake Materials Laboratory, Dongguan, Guangdong 523808, China* (Dated: January 19, 2026)

Reinforcement fine-tuning played an instrumental role in enhancing the instruction-following and reasoning abilities of large language models. In this work, we employ reinforcement fine-tuning for materials design, in which discriminative machine learning models are used to provide rewards to the autoregressive transformerbased materials generative model CrystalFormer [Sci. Bull. 70[, 3522 \(2025\)\]](https://www.sciencedirect.com/science/article/pii/S2095927325009752). By optimizing the reward signals—such as energy above the convex hull and material property figures of merit—reinforcement fine-tuning infuses knowledge from discriminative models into generative models. The resulting model, CrystalFormer-RL, shows enhanced stability in generated crystals and successfully discovers crystals with desirable yet conflicting material properties, such as substantial dielectric constant and band gap simultaneously. Notably, we observe that reinforcement fine-tuning not only enables the property-guided material design but also unlocks property-based material retrieval behavior of pretrained generative model. The present framework opens an exciting gateway to the synergies of the machine learning ecosystem for materials design.

#### I. INTRODUCTION

In materials science, machine learning (ML) techniques are revolutionizing research to enable rapid design and discovery of materials with desired properties. Machine learning models can be broadly categorized into generative and discriminative approaches. Crystal generative models, aim to capture the prior probability distribution *p*(*x*) of stable crystal structures in the chemical space, learning the underlying patterns and correlations in crystal data to generate novel structures. These generative models [\[1–](#page-7-0)[6\]](#page-7-1) capture the underlying distribution of stable materials in the chemical space and enable the direct generation of novel materials. On the other hand, discriminative models, including machine learning interatomic potentials (MLIP) [\[7–](#page-7-2)[13\]](#page-7-3) and property prediction models [\[14,](#page-7-4) [15\]](#page-7-5), focus on modeling the conditional probability *p*(*y*|*x*), where *y* represents material properties given a crystal structure *x*. These models excel at predicting specific properties or behaviors of materials and enable accelerated simulations and property predictions. The integration of these complementary approaches—combining the generative capability of models that capture *p*(*x*) with the predictive power of discriminative models that capture *p*(*y*|*x*)—presents a powerful framework for materials discovery. Further achievements hinge on the flexible combination of discriminative and generative models, enhancing their performance and adapting them to specific application domains. Reinforcement learning (RL) provides a principled way to achieve this synergy, allowing the generative model to be improved and guided by the knowledge embedded in discriminative models.

The synergy of discriminative and generative machine learning models has already shown great success in realworld applications in the past few years. A notable example is classifier-guided generation [\[16,](#page-7-6) [17\]](#page-7-7), where supervised trained discriminative models are leveraged to guide sampling of generative models, eliminating the need for repeated retraining of the generative model. Moreover, in the posttraining phase of large language models (LLM), reinforcement learning from human feedback (RLHF) [\[18,](#page-7-8) [19\]](#page-7-9) is employed to extract signals from supervised trained reward models to instruct generative models. In this context, RL [\[20\]](#page-7-10) serves as a powerful mechanism to integrate unsupervised and supervised learned models. For example, LLM trained via RL have demonstrated improved instruction following [\[18\]](#page-7-8) and reasoning capabilities [\[21\]](#page-7-11), which shows it is an effective way to enhance the performance of generative models.

Inspired by RLHF, we devise an RL approach to finetune the crystal generative model CrystalFormer using the reward provided by discriminative models such as MLIP and property prediction models, see Fig. [1\(](#page-1-0)a). We demonstrate that the approach effectively enhances the stability and desired figures of merit of generated materials. We have released the codes and trained models [\[22\]](#page-7-12). Moreover, we provide instructions on fine-tuning CrystalFormer with customized MLIP or property prediction models for broader applications.

The organization of the paper is as follows: Section [II](#page-0-1) describes the RL algorithm for fine-tuning CrystalFormer. Section [III](#page-2-0) presents the results on improving the stability of generated materials and examples of property-controlled material generation. Section [IV](#page-5-0) discusses the prospects of this work in a broader context.

### <span id="page-0-1"></span>II. METHOD

In the spirit of RLHF [\[18,](#page-7-8) [19\]](#page-7-9), we fine tune pre-trained crystal generative model using reward signal from MLIP or a material property prediction model. The objective function to be maximized reads [\[23\]](#page-7-13),

<span id="page-0-2"></span>
$$\mathcal{L} = \underset{x \sim p_{\theta}(x)}{\mathbb{E}} \left[ r(x) - \tau \ln \left( \frac{p_{\theta}(x)}{p_{\text{base}}(x)} \right) \right], \tag{1}$$

<span id="page-0-0"></span><sup>∗</sup> [wanglei@iphy.ac.cn](mailto:wanglei@iphy.ac.cn)

<span id="page-1-0"></span>Figure 1. (a) The reinforcement fine-tuning workflow. Machine learning interatomic potential or property prediction models provide rewards to the material generated by CrystalFormer. The RL training loop updates the parameters of a pre-trained CrystalFormer to maximize the objective function in Eq. (1). (b) The reinforcement fine-tuned model deviates from the base model to maximize the expected reward with entropy regularization.

where x represents crystalline materials sampled from a policy network  $p_{\theta}(x)$ , r(x) is the reward function which awards preferred materials with high returns,  $\tau$  is the regularization coefficient controls proximity to the base model  $p_{\text{base}}(x)$ . The second term of Eq. (1) is the Kullback-Leibler (KL) divergence between the policy distribution and the base model [24]. Overall, the objective function aims to maximize the expected reward of samples while ensuring that the policy also aligns with the base model.

The combination of both terms in Eq. (1) can also be regarded as a KL divergence between the policy network  $p_{\theta}(x)$  under fine-tuning and the optimal policy [23, 25]

<span id="page-1-1"></span>
$$p_{\theta}^*(x) = \frac{p_{\text{base}}(x)}{Z} \exp\left(\frac{r(x)}{\tau}\right),$$
 (2)

where Z is the normalization factor. The corresponding optimal value of the objective function Eq. (1) will be  $\tau \ln Z$ . Setting the reward to be proportional to the log-likelihood of certain desired property  $r(x) \propto \tau \ln p(y|x)$ , one sees that  $p_{\theta}^*(x) \propto p_{\text{base}}(x)p(y|x)$ . Therefore, Eq. (2) can be viewed as Bayesian inference with a prior distribution  $p_{\text{base}}(x)$  and the likelihood  $\exp(r(x)/\tau)$  [25, 26]. In practice, the pre-trained model not only provides the prior distribution  $p_{\text{base}}(x)$  but also provides parameter initialization for the policy network. Figure 1(b) illustrates the RL fine-tuning process. The variational inference provides an alternative to the Markov chain Monte Carlo (MCMC) sampling carried out in Ref. [5]. Here, larger  $\tau$  corresponds to higher temperature and hence stronger en-

tropy regularization. Such optimization is also equivalent to the variational free energy calculations carried out for statistical mechanics problems [27, 28], where the two terms in the objective function Eq. (1) correspond to expected energy and entropy, respectively. We employ the proximal policy optimization (PPO) [29] algorithm to maximize the expected reward. The implementation details are provided in Appendix C.

In the RLHF paradigm for LLM, a reward model that represents human preferences is used to evaluate the quality of generated samples. Analogously, in computational material science, density functional theory (DFT) is frequently employed to assess the quality of crystalline materials. However, the computational expense associated with DFT renders its direct application as a reward model within the RLHF framework infeasible. To circumvent this limitation, we propose the utilization of an MLIP as a surrogate reward model. Recent advancements in MLIP have yielded models capable of simulating atomic systems with both high accuracy and reduced computational cost, thus providing a viable proxy for DFT calculations [7]. Moreover, universal MLIP [11–13, 30–32], encompassing the entire periodic table of chemical elements, offering the capacity to evaluate a diverse range of materials, making them particularly well-suited for our RL fine-tuning.

RLHF can also be effectively employed for property-guided material generation, wherein the reward model is one or a combination of several property prediction models [33–36]. The fine-tuned generative model is capable of generating materials with desired properties, such as band gap, formation energy, and so on. We choose to carry out RL fine-tuning for the crystal generative model CrystalFormer [5] due to its simplicity, generality, and flexibility.

The CrystalFormer is an autoregressive transformer which models the tokenized representation of crystal structures with explicit knowledge of space group symmetries. The representation includes the space group number, Wyckoff letter, chemical element, and fractional coordinates of each symmetry inequivalent atom, and finally, the lattice parameters [5]. For example, the crystal LaH<sub>10</sub> in the  $Fm\bar{3}m$  (No. 225) space group with the cubic conventional unit cell of the length 5.1Å is represented as the sequence "225-a-La-0-0-c-H-1/4-1/4-f-H-0.375-0.375-0.375-5.1-5.1-5.1-90-90-90", which contains all the necessary information of the crystal. Key to the design of CrystalFormer is the sequential nature of Wyckoff letters which decrease in site symmetry in an alphabet order. In this regard, the CrystalFormer leverages Nature's codebook-the Wyckoff position table–for a multimodal probabilistic modeling. Although the representation appears to lack a sense of spatial geometry, unsupervised pre-training allows the model to ingest solid-state chemistry knowledge by compressing the crystalline material database into the model parameters. Furthermore, the RL phase may allow it to extract the geometry and physics from MLIP and property prediction models trained with supervised approaches.

In the present work, we first train the CrystalFormer on the Alex-20 dataset, which is curated from the Alexandria dataset [37]. More details of the dataset can be found in the

<span id="page-2-1"></span>Figure 2. (a) The average energy above the convex hull of generated materials and (b) the KL divergence between the policy and the base model versus training steps. We set the regularization coefficient  $\tau = 0.1$ .

Appendix B. The resulting pre-trained model then serves as the base model for the RL procedure, where we fine-tune the model using the RL method using the MLIP or crystal property prediction models as the reward model. In the RL stage, the model learns from the reward model on its own samples. We sample crystal structures from CrystalFormer's policy,  $p_{\theta}(x)$ , with the space groups distributed according to the Alex-20 dataset. The samples are then evaluated by the reward model r(x), which assigns a reward based on stableness or properties. These reward signals provide feedback for updating the CrystalFormer's policy to maximize the objective function in Eq. (1), iteratively improving the CrystalFormer's capacity to generate target crystal structures. We stop the fine-tuning process when the loss function converges.

#### <span id="page-2-0"></span>III. APPLICATIONS

# <span id="page-2-3"></span>A. Reinforcement learning from MLIP for improved stability of generated materials

As the first application of the reinforcement fine-tuning of CrystalFormer, we choose the energy above the convex hull as the reward signal, i.e.,  $r(x) = -E_{\text{hull}}(x)$ . The energy above the convex hull is the key metric for assessing the stability of a material, with lower values indicating greater stability. To predict the energy and calculate the energy above the convex hull based on the Alexandria convex hull [37], we utilize the Orb model [13]. This choice allows us to obtain accurate energy predictions and assess the stability of materials relative to the convex hull fast. It is important to emphasize that our approach is not limited to the Orb model; any universal MLIP that can compute energy can serve as the reward model. More accurate and robust MLIP yields better reward signals, enhancing alignment with DFT calculations. Benchmarks on universal MLIP such as [38] provide an interactive leaderboard to select the favorable one for the fine-tuning purpose. We evaluate the performance by measuring the fraction of structures that are stable, unique, and novel (S.U.N.) [3]. The stability is defined as the energy above the convex hull being less than 0.1 eV/atom, uniqueness is defined as the gener-

<span id="page-2-2"></span>Figure 3. The histogram of energy above convex hull for the crystal samples from the pre-trained base model and the ones relaxed by the Orb model [13]. In comparison, reinforcement fine-tuned of the model significantly reduces the energy above convex hull of generated materials. The red dashed line indicates the threshold of 0.1 eV/atom for stable materials. Relaxation ratio from 44.7% in the base model to 57.8%, while RL fine-tuning improves the ratio of stable materials to 73.4% even without relaxation.

ated structure not being identical to any other generated structures, and novelty is defined as the generated structure not being present in the training dataset.

Figure 2(a) shows the expected energy above the convex hull of generated samples, which shows steady improvement during the training process. Figure 2(b) shows the KL divergence between the policy and the pre-trained base model  $\text{KL}(p_{\theta}||p_{\text{base}}) = \mathbb{E}_{x \sim p_{\theta}(x)} \left[ \ln \left( \frac{p_{\theta}(x)}{p_{\text{base}}(x)} \right) \right]$ , which is orders of magnitude smaller than change in the negative log-likelihood during pre-training [5].

Figure 3 shows the histogram of the energy above the convex hull for the pre-trained model and the reinforcement finetuned model. For each model, 1000 structures were sampled, with their space groups drawn from the training data distribution. For the pre-trained model, 44.7% of the generated structures show  $E_{\text{hull}}$  < 0.1 eV/atom. Relaxation improves this ratio to 57.8%. On the other hand, reinforcement fine-tuning with respect to the energy above the convex hull enhances the ratio to 73.4%. Such substantial improvement is due to that the reinforcement fine-tuning reshapes the distribution with a global modification of the Wyckoff sites, fractional coordinates, and lattice parameters while relaxation only modifies the materials locally. Besides reducing the energy above the convex hull, the reinforcement fine-tuning also improves the structure validity metrics compared to the base model [1], see Appendix B.

Figure 4 shows the fraction of generated structures that are stable and are also S.U.N. for 14 space groups spanning the seven crystal systems. We investigate the performance of the model on the same 14 space groups as Ref. [3]. For each space group, we generate 1000 structures and evaluate the quality of samples using the Orb model [13]. The results demonstrate that the reinforcement fine-tuned model generates a higher fraction of stable and S.U.N. structures in most cases, indicating that the RL method can enhance the model's ability to generate S.U.N. structures even though we only award for the stability. Reinforcement fine-tuning with respect to the en-

<span id="page-3-0"></span>Figure 4. Fraction of generated structures that are stable, novel and unique (S.U.N.) for 14 space groups spanning the seven crystal systems. For each space group, two side-by-side bars correspond to the samples from the pre-trained ("Base") and RL fine-tuned ("RL") models. The S.U.N. structures form a subset of stable materials each space group. The shaded regions separate different crystal systems.

ergy above the convex improves the stability across all space group (as it should). However, the improvement of the S.U.N. is impaired by the novelty criteria as there is strong constraints to the structural motifs in these high-symmetry systems. From the figure one also observes that the performance of CrystalFormer is worst for the least symmetric P1 space group where there is minimal symmetry-related inductive bias to be explored. This is in sharp contrast to Ref. [3] which shows the best S.U.N. in the P1 space group, highlighting the distinctive advantage of our method. In fact, for space group with higher symmetry which are more relevant to the natural distribution of inorganic materials [39], we observed that the performance of CrystalFormer is on par or even better than the ones shown in Fig. D8 of Ref. [3]. Lastly, while reinforcement fine-tuning does not affect the generation speed of the model, it increases the rate of generating S.U.N. samples from 3.46 to 4.89 per second.

Figure 5 breaks down the performance enhancement in terms of the S.U.N. ratio coming from enlarging the pretraining dataset and different ways of fine-tuning. For these evaluations, we sample space group following the same distribution of the training dataset and use it as precondition to the CrystalFormer. First of all, one sees that enlarging the pre-training dataset from MP-20 to Alex-20 yields a significant improvement in the S.U.N. ratio, rising from 2.6% to 15.3%. Next, we also tested the improvement brought by supervised fine-tuning (SFT) on the pre-trained model. For SFT, we generate 10,000 crystal samples and relax them using the Orb models. Only those stable samples with  $E_{\rm hull} < 0.1$  eV/atom were used to finetune the pre-trained model. Fig-

<span id="page-3-1"></span>Figure 5. The weighted S.U.N. ratio averaged over all space groups of the CrystalFormer trained on the MP-20 dataset and the Alex-20 dataset compared with the fine-tuned model with SFT and RL approaches. The energy above the convex hull is calculated based on the Alexandria convex hull and the novelty is calculated based on the Alex-20 dataset.

ure 5 shows that SFT yields some improvement over the pretrained model, further increasing the S.U.N. ratio from 15.3% to 18.3%. However, it is the reinforcement fine-tuned model that delivers the most significant increase in the S.U.N. ratio, leading to a final S.U.N. ratio of 21.6%.

# B. Reinforcement learning from the figure of merits for property-guided material generation

Next, we apply the RL method for dielectric material discovery. The interplay between the electronic dielectric constant and the band gap reflects a fundamental constraint in condensed matter systems. Recent works have established that the electronic polarizability originates from virtual interband transitions governed by the quantum metric and inversely scales with the spectral gap [40], while the gap itself is bounded by the dielectric response and electron density [41]. Pursuing materials that simultaneously exhibit a large electronic dielectric constant and a wide band gap thus represents a key challenge in realizing highly polarizable yet robust insulators, where quantum geometry and energetic stability reach a delicate balance. We utilize the pre-trained MEG-Net [15, 42] and AnisoNet [43] models to predict the band gap  $E_g$  and electronic dielectric constant  $\varepsilon_{\rm elec}$  of the generated samples, respectively. MEGNet achieves a mean absolute error (MAE) of 0.314 eV for band gap prediction on the Materials Project dataset [44], while AnisoNet achieves a MAE of 0.297 for electronic dielectric constant prediction on the Materials Project dielectric dataset [44, 45]. It is worth noting both models are trained on the data computed using the Perdew-Burke-Ernzerhof functional [46], which is known to underestimate the band gap. These data also omit corrections such as spin-orbit coupling or excitonic effects, which may be important for certain materials. More accurate band gap prediction model would require training on higher-fidelity datasets, such as those computed using hybrid functionals [47, 48], the GW method [49] or GW-BSE approach [50]. The electronic dielectric constant is a dimensionless number, which is the ratio of the permittivity of a substance to the permittivity of vacuum. The reward function is defined as the figure of merit (FoM),  $r(x) = \varepsilon_{\text{elec}} \cdot E_g$ , which favors materials with both sizeable band gap and permittivity.

Figure 6(a) shows the ML predicted band gap  $E_g$  versus electronic dielectric constant  $\varepsilon_{\rm elec}$  for generated samples in the  $Fm\bar{3}m$  space group (No.225) using the base and finetuned model respectively. The reinforcement fine-tuning significantly improves the FoM, with the majority of generated samples exhibiting higher FoM values compared to the pretrained base model. We observe that the RL method primarily enhances the FoM by increasing the band gap, rather than the dielectric constant. This outcome may be attributed to the limited and relatively low dielectric constant data used to train the dielectric model, potentially causing the model to predict lower dielectric constants. To address this issue, a more robust dielectric prediction model that captures a wider range of dielectric constants could be employed [51]. In contrast, the band gap prediction model demonstrates greater accuracy, resulting in the RL method favoring an increase in the band gap to optimize the FoM.

We first generated 1000 samples and then use Orb model to filter out the unstable materials, followed by DFT verification of the top 50 candidates with the highest FoM in the remaining subset. More computational details are provided in Appendix D, and materials with  $E_{\rm hull} < 0.1 \; {\rm eV/atom}$  are listed in

<span id="page-4-0"></span>Figure 6. (a) The distribution of band gap  $E_g$  and electronic dielectric constant  $\varepsilon_{\text{elec}}$  of material samples generated in the  $Fm\bar{3}m$  space group (No.225). The two properties are predicted by the MEGNet and AnisoNet models, respectively. The dashed line indicates a constant FoM at values  $\varepsilon_{\rm elec} \cdot E_g = 10 \, \rm eV$ . The inset shows average FoM of generated materials versus training steps. Two stars indicate the S.U.N. materials. (b) The histogram of energy above the convex hull predicted by the Orb model [13] for the crystal samples from the pre-trained and the RL fine-tuned model in  $Fm\bar{3}m$  space group (No.225), respectively. The red dashed line indicates the threshold of 0.1 eV/atom and the x-axis is up to 0.4 eV/atom. RL improves the FoM of the generated samples towards high FoM dielectric materials at the cost of only slightly affecting the stability. (c) Two examples of discovered high FoM materials along with their properties verified by DFT calculation. We also show the tokenized material string generated with CrystalFormer (fractional coordinate and lattice parameters omitted).

225-a-Ca-b-Ca-c-Cs-e-F

CsCaF<sub>3</sub>

225-a-Li-b-Lu-c-Cs-e-F

Cs<sub>2</sub>LiLuF<sub>6</sub>

Table I. We observed that many of these samples are fluoride, probably due to fluorine's strong electronegativity increases the ionic character of bonds, so that to ensure a sizeable band gap. Moreover, many of these high FoM samples are composed of Cs elements, which may be due to the relatively large ionic radius of Cesium, leading to higher polarizability and thus an increased total dielectric constant of the material. Figure 6(b) shows the energy above the convex hull for the materials sampled from the base and fine-tuned model respectively, which suggests that the reinforcement fine-tuning does not significantly affect the stability of generated materials. Two examples of high FoM materials discovered by CrystalFormer-RL, Cs<sub>2</sub>LiLuF<sub>6</sub> and CsCaF<sub>3</sub>, are shown in Fig 6(c). Cs<sub>2</sub>LiLuF<sub>6</sub> belongs to the double perovskite [52] structure prototype. CsCaF3 is obtained by substituting Ba with Ca in CsBaF<sub>3</sub>, which is included in the Alex-20 dataset, and it crystallizes in the halide perovskite [52, 53] structure prototype.

Many of the generated samples are present in the Alex-20 dataset which is in line with the S.U.N ratio 5.1% of the base model on space group No.225. This nevertheless demonstrates that the reinforcement fine-tuning procedure can guide the generative process toward discovering candidates within the pretrained dataset with no property label. Notably, two materials in Table I are *absent* from the Alex-20 dataset, highlighting the ability of RL to guide the generative process toward novel samples beyond the pretrained dataset.

We compare the DFT-calculated band gap and electronic dielectric constant of the generated samples with the predictions from the MEGNet and AnisoNet models. As shown in Figure 7, the ML predictions align well with the DFT results, validating the effectiveness of the RL fine-tuning approach for property-guided material generation.

<span id="page-5-1"></span>Figure 7. ML predicted vs. DFT calculated (a) energy above the convex hull  $E_{\rm hull}$ , (b) band gap  $E_g$  and (c) electronic dielectric constant  $\varepsilon_{\rm elec}$  for the discovered dielectric materials. Panels (b) and (c) include only materials with  $E_{\rm hull} < 0.1$  eV/atom according to the DFT calculation. The two stars indicate the same two novel samples as Fig. 6(a).

The CrystalFormer's ability to aid property-guided exploration of crystalline materials has been demonstrated in previous work [5] using the MCMC sampling method from the posterior distribution according to the Bayes rule. Both the MCMC and reinforcement fine-tuning approach work with any machine learning interatomic potential or property prediction models trained separately of the CrystalFormer. The RL approach addresses the long-mixing time or even the ergodicity issue of the MCMC sampling of the posterior distribution with the additional effort of fine-tuning and a variation-

ally approximated posterior.

#### <span id="page-5-0"></span>IV. DISCUSSIONS

Reinforcement fine-tuning of materials generative model is useful for optimizing other materials properties such as the thermoelectric figure of merit, thermal conductivity, and superconducting critical temperature. In addition to a foundational generative model such as the CrystalFormer for the materials prior p(x), the framework requires a reliable ML prediction model for the likelihood p(y|x), which is a research topic with rapid progress both on generalizability and accuracy, see, e.g. [54–56].

It is often easier to discriminate the utility of a given material than to generate a material with the desired util-Reinforcement fine-tuning of a materials generative model using discriminative feedback offers a way to bridge such discrimination-generation gap. RL on the autoregressive transformer such as CrystalFormer [5] is particularly convenient given a variety of post-training techniques available for LLM. Although those RL techniques are also applicable to natural language based materials generative models [4, 57, 58], we believe finetuned CrystalFormer is more advantageous because it natively speaks the language of crystalline materials, which cherishes the fundamental inductive bias of crystalline materials such as space group symmetry and periodicity. Furthermore, the idea of reinforcement finetuning can also be applied to the material generative models based on diffusion [59, 60]. The diffusion model is typically fine-tuned through the incorporation of auxiliary networks to condition the generation on external inputs [61], while RL eliminates the necessity for these additional network components. For example, Refs. [62, 63] employ the RL algorithm to fine-tune the pre-trained text-to-image diffusion model, leading to improved sample quality and diversity. In the mean time, applying RL with KL regularization to the diffusion model is challenging due to the need for precise log-likelihood calculations [64].

It is also interesting to compare active learning and RL in the context of generative materials discovery. Active learning [65, 66] selects crystal samples back to the generative training dataset, which is similar to the SFT approach discussed in Sec. III A. Similar to pretrain, the active learning procedure optimizes the forward KL divergence between the empirical distribution of the training dataset and the generative model. In contrast, RL minimizes the reverse KL divergence [25] between the generative model and the target distribution Eq. (2). There are both empirical and theoretical evidences that the RL approach delivers better performance compared to the active learning approach in the contexts of LLM fine-tuning [67, 68]. Consequently, the primary advantage of RL over active learning is its ability to directly incorporate task-specific reward signals, enabling the model to generate outputs that more effectively meet practical application requirements. Moreover, we suspect the difference in the mode covering versus mode seeking in the active learning versus RL may also be a reason for the difference in the perfor-

<span id="page-6-0"></span>Table I. Dielectric materials discovered by CrystalFormer-RL, ranked according to their figures of merit (FoM): <sup>ε</sup>elec ·*Eg*. The band gap, dielectric constant, and energy above the convex hull are all calculated by DFT. The two highlighted rows are not in the Alex-20 dataset. See details of the samples at https://[zenodo.org](https://zenodo.org/records/17982949)/records/17982949.

| Formula    | (eV) 1<br>Eg | εelec | FoM (eV) | Ehull<br>(eV/atom) |
|------------|--------------|-------|----------|--------------------|
| Cs2LiLuF6  | 7.37         | 2.38  | 17.52    | 0.09               |
| Cs2LiTmF6  | 7.35         | 2.37  | 17.40    | 0.08               |
| Cs2LiHoF6  | 7.32         | 2.37  | 17.39    | 0.09               |
| Cs2LiErF6  | 7.29         | 2.37  | 17.28    | 0.09               |
| Cs2LiTbF6  | 7.21         | 2.36  | 17.05    | 0.09               |
| Cs2LiDyF6  | 7.17         | 2.34  | 16.80    | 0.09               |
| CsCaF3     | 6.94         | 2.32  | 16.10    | 0.06               |
| Cs2LiScF6  | 6.64         | 2.40  | 15.92    | 0.09               |
| Cs2NaTmF6  | 6.98         | 2.27  | 15.84    | 0.05               |
| Cs2NaErF6  | 6.96         | 2.27  | 15.77    | 0.05               |
| Cs2NaLuF6  | 6.93         | 2.26  | 15.63    | 0.05               |
| Cs2NaHoF6  | 6.90         | 2.26  | 15.59    | 0.05               |
| Cs2NaYF6   | 6.89         | 2.25  | 15.48    | 0.03               |
| Cs2LiLuCl6 | 5.34         | 2.90  | 15.48    | 0.01               |
| Rb2LiLuF6  | 7.01         | 2.19  | 15.34    | 0.07               |
| Cs2LiHoCl6 | 5.16         | 2.86  | 14.78    | 0.01               |
| Cs2LiTbCl6 | 5.07         | 2.89  | 14.66    | 0.01               |
| Y2ErF6     | 7.15         | 1.96  | 14.01    | 0.05               |
| Cs2SrMgCl6 | 5.07         | 2.74  | 13.89    | 0.03               |
| Cs2MgPbF6  | 5.29         | 2.60  | 13.77    | 0.08               |
| K2Rb2AlF6  | 6.40         | 2.02  | 12.92    | 0.06               |
| Cs2RbDyCl6 | 4.92         | 2.46  | 12.09    | 0.03               |
| Cs2ErGaCl6 | 3.82         | 3.05  | 11.64    | 0.04               |
| Cs2BaPbI6  | 3.17         | 3.62  | 11.48    | 0.07               |
| Cs2NaAlCl6 | 3.97         | 2.77  | 11.00    | 0.03               |
| Rb2TmGaCl6 | 3.60         | 2.99  | 10.77    | 0.06               |

mance [\[26,](#page-8-1) [69\]](#page-9-11).

Materials generative models are typically trained on equilibrium or near equilibrium structures, which only represent a small fraction of the chemical space. Incorporating nonequilibrium structures into the training process can improve model robustness and enhance the model's capability to generate novel structures. Reinforcement fine-tuning achieves this by using MLIP reward model to penalize the unstable structures generated by the model. Alternatively, the direct preference optimization [\[70\]](#page-9-12) algorithm may be employed to finetune the model on pairs of equilibrium and non-equilibrium structures, which eliminates the necessity of training an explicit reward model for material stability.

Figure [7](#page-5-1) compares the ML-predicted and DFT-calculated results, revealing that while the overall trends are consistent, noticeable discrepancies remain between the predictions and the DFT evaluations. This reveals a fundamental challenge in the application of RL in materials science: what is the verifiable ground truth reward signal. Given the intricacies involved in material synthesis and characterization [\[71\]](#page-9-13), such a challenge is particularly concerning as faulty reward functions [\[72\]](#page-9-14) which fails to align with the final goal will suggest structures with high reward but fail for the final experiment or even intermediate DFT verifications. We believe the design of reward functions that are both actionable and well-aligned with experimental outcomes, with the insights of human experts, is crucial for ensuring the reliability of the present and many related efforts.

Compared to property-guided material generation based on training on labeled data [\[3,](#page-7-17) [73](#page-9-15)[–75\]](#page-9-16), the approaches advertised in [\[5\]](#page-7-15) (MCMC) and the present work (Reinforcement finetuning, also known as variational Bayes) rely on the external discriminative model in a plug-and-play manner. The flexibility and modularity nature of this approach allows seamless integration with many existing and very strong MLIP and ML property prediction models into the materials generative design workflow.

This study focuses on how generative models can be enhanced and specialized through the integration of property prediction models. However, the reverse question: how property prediction models can be improved by leveraging generative models—is also an intriguing question worth future investigation. For example, the Orb models employ a twostage training framework that integrates generative and supervised learning to achieve state-of-the-art performance [\[13\]](#page-7-3). Initially, the model is pre-trained as denoising diffusion models on datasets of stable materials. The pre-trained model then serves as the initialization for an MLIP, which is subsequently fine-tuned in a supervised manner to predict the energy, forces, and stress of materials. This approach exemplifies how generative pre-training can enhance the performance of discriminative models such as MLIP. Extending this idea further, Ref. [\[12\]](#page-7-19) has demonstrated that fine-tuning pretrained MLIP also gives better performance for property prediction models.

#### ACKNOWLEDGMENTS

We thank Han Wang, Linfeng Zhang, Jian Lv, Hongteng Xu, Shigang Ou, Xiaoshan Luo, Mingzhang Yang, Tianping Ying, Ling Lu, Weiluo Ren and Huaxian Jia for useful discussions. This project is supported by the National Natural Science Foundation of China under Grants No. T2225018, No. 92270107, No. 12188101, and No. T2121001, Cross-Disciplinary Key Project of Beijing Natural Science Foundation No. Z250005, National Key Projects for Research and Development of China Grant. No. 2021YFA1400400, and the Strategic Priority Research Program of Chinese Academy of Sciences under Grants No. XDB0500000.

- <span id="page-7-0"></span>[1] T. Xie, X. Fu, O.-E. Ganea, R. Barzilay, and T. Jaakkola, Crystal diffusion variational autoencoder for periodic material generation, (2021), [arXiv:2110.06197 \[cs.LG\].](https://arxiv.org/abs/2110.06197)
- [2] R. Jiao, W. Huang, P. Lin, J. Han, P. Chen, Y. Lu, and Y. Liu, Crystal structure prediction by joint equivariant diffusion, (2023), [arXiv:2309.04475 \[cond-mat.mtrl-sci\].](https://arxiv.org/abs/2309.04475)
- <span id="page-7-17"></span>[3] C. Zeni, R. Pinsler, D. Zugner, A. Fowler, M. Horton, X. Fu, ¨ Z. Wang, A. Shysheya, J. Crabbe, S. Ueda, ´ *et al.*, A generative model for inorganic materials design, Nature 639[, 624 \(2025\).](https://www.nature.com/articles/s41586-025-08628-5)
- <span id="page-7-18"></span>[4] N. Gruver, A. Sriram, A. Madotto, A. G. Wilson, C. L. Zitnick, and Z. Ulissi, Fine-tuned language models generate stable inorganic materials as text, (2024), [arXiv:2402.04379 \[cs.LG\].](https://arxiv.org/abs/2402.04379)
- <span id="page-7-15"></span>[5] Z. Cao, X. Luo, J. Lv, and L. Wang, Space group informed transformer for crystalline materials generation, [\(2024\),](https://arxiv.org/abs/2403.15734) [arXiv:2403.15734 \[cond-mat.mtrl-sci\].](https://arxiv.org/abs/2403.15734)
- <span id="page-7-1"></span>[6] R. Jiao, W. Huang, Y. Liu, D. Zhao, and Y. Liu, Space group constrained crystal generation, (2024), [arXiv:2402.03992](https://arxiv.org/abs/2402.03992) [\[cs.LG\].](https://arxiv.org/abs/2402.03992)
- <span id="page-7-2"></span>[7] O. T. Unke, S. Chmiela, H. E. Sauceda, M. Gastegger, I. Poltavsky, K. T. Schuutt, A. Tkatchenko, and K.-R. M ¨ uller, ¨ Machine learning force fields, [Chemical Reviews](https://pubs.acs.org/doi/10.1021/acs.chemrev.0c01111) 121, 10142 [\(2021\).](https://pubs.acs.org/doi/10.1021/acs.chemrev.0c01111)
- [8] L. Zhang, J. Han, H. Wang, R. Car, and W. E, Deep potential molecular dynamics: A scalable model with the accuracy of quantum mechanics, Phys. Rev. Lett. 120[, 143001 \(2018\).](https://doi.org/10.1103/PhysRevLett.120.143001)
- [9] S. Batzner, A. Musaelian, L. Sun, M. Geiger, J. P. Mailoa, M. Kornbluth, N. Molinari, T. E. Smidt, and B. Kozinsky, E (3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials, [Nature communications](https://www.nature.com/articles/s41467-022-29939-5) 13, 2453 [\(2022\).](https://www.nature.com/articles/s41467-022-29939-5)
- [10] I. Batatia, D. P. Kovacs, G. Simm, C. Ortner, and G. Csanyi, ´ Mace: Higher order equivariant message passing neural networks for fast and accurate force fields, [Advances in neural in](https://proceedings.neurips.cc/paper_files/paper/2022/file/4a36c3c51af11ed9f34615b81edb5bbc-Paper-Conference.pdf)[formation processing systems](https://proceedings.neurips.cc/paper_files/paper/2022/file/4a36c3c51af11ed9f34615b81edb5bbc-Paper-Conference.pdf) 35, 11423 (2022).
- <span id="page-7-16"></span>[11] D. Zhang, H. Bi, F.-Z. Dai, W. Jiang, L. Zhang, and H. Wang, Dpa-1: Pretraining of attention-based deep potential model for molecular simulation, (2022), [arXiv:2208.08236](https://arxiv.org/abs/2208.08236) [\[physics.chem-ph\].](https://arxiv.org/abs/2208.08236)
- <span id="page-7-19"></span>[12] D. Zhang, X. Liu, X. Zhang, C. Zhang, C. Cai, H. Bi, Y. Du, X. Qin, J. Huang, B. Li, Y. Shan, J. Zeng, Y. Zhang, S. Liu, Y. Li, J. Chang, X. Wang, S. Zhou, J. Liu, X. Luo, Z. Wang, W. Jiang, J. Wu, Y. Yang, J. Yang, M. Yang, F.-Q. Gong,

- L. Zhang, M. Shi, F.-Z. Dai, D. M. York, S. Liu, T. Zhu, Z. Zhong, J. Lv, J. Cheng, W. Jia, M. Chen, G. Ke, W. E, L. Zhang, and H. Wang, Dpa-2: Towards a universal large atomic model for molecular and material simulation, (2023), [arXiv:2312.15492 \[physics.chem-ph\].](https://arxiv.org/abs/2312.15492)
- <span id="page-7-3"></span>[13] M. Neumann, J. Gin, B. Rhodes, S. Bennett, Z. Li, H. Choubisa, A. Hussey, and J. Godwin, Orb: A fast, scalable neural network potential, [\(2024\),](https://arxiv.org/abs/2410.22570) [arXiv:2410.22570 \[cond-mat.mtrl-sci\].](https://arxiv.org/abs/2410.22570)
- <span id="page-7-4"></span>[14] T. Xie and J. C. Grossman, Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties, Phys. Rev. Lett. 120[, 145301 \(2018\).](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301)
- <span id="page-7-5"></span>[15] C. Chen, W. Ye, Y. Zuo, C. Zheng, and S. P. Ong, Graph networks as a universal machine learning framework for molecules and crystals, [Chemistry of Materials](https://doi.org/10.1021/acs.chemmater.9b01294) 31, 3564 (2019).
- <span id="page-7-6"></span>[16] S. Dathathri, A. Madotto, J. Lan, J. Hung, E. Frank, P. Molino, J. Yosinski, and R. Liu, Plug and play language models: A simple approach to controlled text generation, (2019), [arXiv:1912.02164 \[cs.CL\].](https://arxiv.org/abs/1912.02164)
- <span id="page-7-7"></span>[17] P. Dhariwal and A. Nichol, Diffusion models beat gans on image synthesis, [\(2021\),](https://arxiv.org/abs/2105.05233) [arXiv:2105.05233 \[cs.LG\].](https://arxiv.org/abs/2105.05233)
- <span id="page-7-8"></span>[18] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, *et al.*, Training language models to follow instructions with human feedback, [Advances in neural information processing systems](https://papers.nips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf) 35[, 27730 \(2022\).](https://papers.nips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf)
- <span id="page-7-9"></span>[19] D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving, Fine-tuning language models from human preferences, [arXiv preprint](https://arxiv.org/abs/1909.08593) [arXiv:1909.08593 \(2019\).](https://arxiv.org/abs/1909.08593)
- <span id="page-7-10"></span>[20] R. S. Sutton and A. G. Barto, *[Reinforcement Learning: An In](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)[troduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)*, 2nd ed. (MIT Press, 2018).
- <span id="page-7-11"></span>[21] DeepSeek AI, Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, [\(2025\),](https://arxiv.org/abs/2501.12948) [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) [\[cs.CL\].](https://arxiv.org/abs/2501.12948)
- <span id="page-7-12"></span>[22] See https://github.com/deepmodeling/[CrystalFormer](https://github.com/deepmodeling/CrystalFormer) for code and model checkpoint.
- <span id="page-7-13"></span>[23] J. Schulman, X. Chen, and P. Abbeel, Equivalence between policy gradients and soft q-learning, [arXiv preprint](https://arxiv.org/abs/1704.06440) [arXiv:1704.06440 \(2017\).](https://arxiv.org/abs/1704.06440)
- <span id="page-7-14"></span>[24] N. Jaques, S. Gu, D. Bahdanau, J. M. Hernandez-Lobato, R. E. ´ Turner, and D. Eck, Sequence tutor: Conservative fine-tuning of sequence generation models with kl-control, in *[International](https://proceedings.mlr.press/v70/jaques17a/jaques17a.pdf)*

- *[Conference on Machine Learning](https://proceedings.mlr.press/v70/jaques17a/jaques17a.pdf)* (PMLR, 2017) pp. 1645– 1654.
- <span id="page-8-0"></span>[25] T. Korbak, E. Perez, and C. L. Buckley, Rl with kl penalties is better viewed as bayesian inference, [\(2022\),](https://arxiv.org/abs/2205.11275) [arXiv:2205.11275](https://arxiv.org/abs/2205.11275) [\[cs.LG\].](https://arxiv.org/abs/2205.11275)
- <span id="page-8-1"></span>[26] S. Levine, Reinforcement learning and control as probabilistic inference: Tutorial and review, [\(2018\),](https://arxiv.org/abs/1805.00909) [arXiv:1805.00909](https://arxiv.org/abs/1805.00909) [\[cs.LG\].](https://arxiv.org/abs/1805.00909)
- <span id="page-8-2"></span>[27] S.-H. Li and L. Wang, Neural network renormalization group, Phys. Rev. Lett. 121[, 260601 \(2018\).](https://doi.org/10.1103/PhysRevLett.121.260601)
- <span id="page-8-3"></span>[28] D. Wu, L. Wang, and P. Zhang, Solving statistical mechanics using variational autoregressive networks, [Phys. Rev. Lett.](https://doi.org/10.1103/PhysRevLett.122.080602) 122, [080602 \(2019\).](https://doi.org/10.1103/PhysRevLett.122.080602)
- <span id="page-8-4"></span>[29] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, Proximal policy optimization algorithms, [\(2017\),](https://arxiv.org/abs/1707.06347) [arXiv:1707.06347 \[cs.LG\].](https://arxiv.org/abs/1707.06347)
- <span id="page-8-5"></span>[30] C. Chen and S. P. Ong, A universal graph deep learning interatomic potential for the periodic table, [Nature Computational](https://doi.org/10.1038/s43588-022-00349-3) Science 2[, 718 \(2022\).](https://doi.org/10.1038/s43588-022-00349-3)
- [31] I. Batatia, P. Benner, Y. Chiang, A. M. Elena, D. P. Kovacs, ´ J. Riebesell, X. R. Advincula, M. Asta, M. Avaylon, W. J. Baldwin, F. Berger, N. Bernstein, A. Bhowmik, S. M. Blau, V. Carare, J. P. Darby, S. De, F. D. Pia, V. L. Deringer, ˘ R. Elijosius, Z. El-Machachi, F. Falcioni, E. Fako, A. C. ˇ Ferrari, A. Genreith-Schriever, J. George, R. E. A. Goodall, C. P. Grey, P. Grigorev, S. Han, W. Handley, H. H. Heenen, K. Hermansson, C. Holm, J. Jaafar, S. Hofmann, K. S. Jakob, H. Jung, V. Kapil, A. D. Kaplan, N. Karimitari, J. R. Kermode, N. Kroupa, J. Kullgren, M. C. Kuner, D. Kuryla, G. Liepuoniute, J. T. Margraf, I.-B. Magdau, A. Michaelides, J. H. Moore, ˘ A. A. Naik, S. P. Niblett, S. W. Norwood, N. O'Neill, C. Ortner, K. A. Persson, K. Reuter, A. S. Rosen, L. L. Schaaf, C. Schran, B. X. Shi, E. Sivonxay, T. K. Stenczel, V. Svahn, C. Sutton, T. D. Swinburne, J. Tilly, C. van der Oord, E. Varga-Umbrich, T. Vegge, M. Vondrak, Y. Wang, W. C. Witt, F. Zills, and ´ G. Csanyi, A foundation model for atomistic materials chem- ´ istry, (2024), [arXiv:2401.00096 \[physics.chem-ph\].](https://arxiv.org/abs/2401.00096)
- <span id="page-8-6"></span>[32] H. Yang, C. Hu, Y. Zhou, X. Liu, Y. Shi, J. Li, G. Li, Z. Chen, S. Chen, C. Zeni, M. Horton, R. Pinsler, A. Fowler, D. Zugner, ¨ T. Xie, J. Smith, L. Sun, Q. Wang, L. Kong, C. Liu, H. Hao, and Z. Lu, Mattersim: A deep learning atomistic model across elements, temperatures and pressures, [\(2024\),](https://arxiv.org/abs/2405.04967) [arXiv:2405.04967](https://arxiv.org/abs/2405.04967) [\[cond-mat.mtrl-sci\].](https://arxiv.org/abs/2405.04967)
- <span id="page-8-7"></span>[33] H. Park, S. Majumdar, X. Zhang, J. Kim, and B. Smit, Inverse design of metal–organic frameworks for direct air capture of co 2 via deep reinforcement learning, [Digital Discovery](https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00010b) 3, 728 [\(2024\).](https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00010b)
- [34] M. Thomas, A. Bou, and G. D. Fabritiis, Reinforce-ing chemical language models in drug design, [\(2025\),](https://arxiv.org/abs/2501.15971) [arXiv:2501.15971](https://arxiv.org/abs/2501.15971) [\[cs.LG\].](https://arxiv.org/abs/2501.15971)
- [35] M. Popova, O. Isayev, and A. Tropsha, Deep reinforcement learning for de novo drug design, Science Advances 4, 10.1126/[sciadv.aap7885](https://doi.org/10.1126/sciadv.aap7885) (2018).
- <span id="page-8-8"></span>[36] E. Mazuz, G. Shtar, B. Shapira, and L. Rokach, Molecule generation using transformers and policy gradient reinforcement learning, [Scientific Reports](https://doi.org/10.1038/s41598-023-35648-w) 13, 8799 (2023).
- <span id="page-8-9"></span>[37] J. Schmidt, T. F. Cerqueira, A. H. Romero, A. Loew, F. Jager, ¨ H.-C. Wang, S. Botti, and M. A. Marques, Improving machinelearning models in materials science through large datasets, [Materials Today Physics](https://doi.org/https://doi.org/10.1016/j.mtphys.2024.101560) 48, 101560 (2024).
- <span id="page-8-10"></span>[38] J. Riebesell, R. E. A. Goodall, P. Benner, Y. Chiang, B. Deng, A. A. Lee, A. Jain, and K. A. Persson, Matbench discovery – a framework to evaluate machine learning crystal stability predictions, (2024), [arXiv:2308.14920 \[cond-mat.mtrl-sci\].](https://arxiv.org/abs/2308.14920)

- <span id="page-8-11"></span>[39] V. S. Urusov and T. N. Nadezhina, Frequency distribution and selection of space groups in inorganic crystal chemistry, [Journal](https://doi.org/10.1007/s10947-009-0186-9) [of Structural Chemistry](https://doi.org/10.1007/s10947-009-0186-9) 50, 22 (2009).
- <span id="page-8-12"></span>[40] I. Komissarov, T. Holder, and R. Queiroz, The quantum geometric origin of capacitance in insulators, [Nature Communications](https://doi.org/10.1038/s41467-024-48808-x) 15[, 4621 \(2024\).](https://doi.org/10.1038/s41467-024-48808-x)
- <span id="page-8-13"></span>[41] Y. Onishi and L. Fu, Universal relation between energy gap and dielectric constant, Phys. Rev. B 110[, 155107 \(2024\).](https://doi.org/10.1103/PhysRevB.110.155107)
- <span id="page-8-14"></span>[42] C. Chen, Y. Zuo, W. Ye, X. Li, and S. P. Ong, Learning properties of ordered and disordered materials from multi-fidelity data, [Nature Computational Science](https://doi.org/10.1038/s43588-020-00002-x) 1, 46 (2021).
- <span id="page-8-15"></span>[43] Y. Lou and A. M. Ganose, Discovery of highly anisotropic dielectric crystals with equivariant graph neural networks, [Fara](https://doi.org/10.1039/D4FD00096J)[day Discuss.](https://doi.org/10.1039/D4FD00096J) 256, 255 (2025).
- <span id="page-8-16"></span>[44] A. Jain, S. P. Ong, G. Hautier, W. Chen, W. D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, and K. A. Persson, Commentary: The Materials Project: A materials genome approach to accelerating materials innovation, [APL](https://doi.org/10.1063/1.4812323) Materials 1[, 011002 \(2013\).](https://doi.org/10.1063/1.4812323)
- <span id="page-8-17"></span>[45] I. Petousis, D. Mrdjenovich, E. Ballouz, M. Liu, D. Winston, W. Chen, T. Graf, T. D. Schladt, K. A. Persson, and F. B. Prinz, High-throughput screening of inorganic compounds for the discovery of novel dielectric and optical materials, [Scientific data](https://www.nature.com/articles/sdata2016134) 4[, 1 \(2017\).](https://www.nature.com/articles/sdata2016134)
- <span id="page-8-18"></span>[46] J. P. Perdew, K. Burke, and M. Ernzerhof, Generalized Gradient Approximation Made Simple, [Phys. Rev. Lett.](https://doi.org/10.1103/physrevlett.77.3865) 77, 3865 (1996).
- <span id="page-8-19"></span>[47] J. Heyd, G. E. Scuseria, and M. Ernzerhof, Hybrid functionals based on a screened coulomb potential, The Journal of chemical physics 118, 8207 (2003).
- <span id="page-8-20"></span>[48] J. Heyd, G. E. Scuseria, and M. Ernzerhof, Erratum: "hybrid functionals based on a screened coulomb potential" [j. chem. phys. 118, 8207 (2003)], [The Journal of Chemical Physics](https://doi.org/10.1063/1.2204597) 124[, 219906 \(2006\),](https://doi.org/10.1063/1.2204597) https://[pubs.aip.org](https://arxiv.org/abs/https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/1.2204597/15387022/219906_1_online.pdf)/aip/jcp/articlepdf/doi/10.1063/[1.2204597](https://arxiv.org/abs/https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/1.2204597/15387022/219906_1_online.pdf)/15387022/219906 1 online.pdf.
- <span id="page-8-21"></span>[49] M. van Schilfgaarde, T. Kotani, and S. Faleev, Quasiparticle self-consistent *gw* theory, Phys. Rev. Lett. 96[, 226402 \(2006\).](https://doi.org/10.1103/PhysRevLett.96.226402)
- <span id="page-8-22"></span>[50] G. Onida, L. Reining, and A. Rubio, Electronic excitations: density-functional versus many-body green's-function approaches, [Rev. Mod. Phys.](https://doi.org/10.1103/RevModPhys.74.601) 74, 601 (2002).
- <span id="page-8-23"></span>[51] S. Falletta, A. Cepellotti, A. Johansson, C. W. Tan, A. Musaelian, C. J. Owen, and B. Kozinsky, Unified differentiable learning of electric response, [\(2024\),](https://arxiv.org/abs/2403.17207) [arXiv:2403.17207](https://arxiv.org/abs/2403.17207) [\[cond-mat.mtrl-sci\].](https://arxiv.org/abs/2403.17207)
- <span id="page-8-24"></span>[52] N. R. Wolf, B. A. Connor, A. H. Slavney, and H. I. Karunadasa, Doubling the stakes: the promise of halide double perovskites, [Angewandte Chemie](https://onlinelibrary.wiley.com/doi/abs/10.1002/anie.202016185) 133, 16400 (2021).
- <span id="page-8-25"></span>[53] Y. Cai, W. Xie, Y. T. Teng, P. C. Harikesh, B. Ghosh, P. Huck, K. A. Persson, N. Mathews, S. G. Mhaisalkar, M. Sherburne, *et al.*, High-throughput computational study of halide double perovskite inorganic compounds, Chemistry of Materials 31, 5392 (2019).
- <span id="page-8-26"></span>[54] B. Pota, P. Ahlawat, G. Cs ´ anyi, and M. Simoncelli, ´ [Ther](https://arxiv.org/abs/2408.00755)[mal conductivity predictions with foundation atomistic models](https://arxiv.org/abs/2408.00755) (2024), [arXiv:2408.00755 \[cond-mat.mtrl-sci\].](https://arxiv.org/abs/2408.00755)
- [55] A. Loew, D. Sun, H.-C. Wang, S. Botti, and M. A. L. Marques, [Universal machine learning interatomic potentials are ready for](https://arxiv.org/abs/2412.16551) [phonons](https://arxiv.org/abs/2412.16551) (2024), [arXiv:2412.16551 \[cond-mat.mtrl-sci\].](https://arxiv.org/abs/2412.16551)
- <span id="page-8-27"></span>[56] X. Fu, B. M. Wood, L. Barroso-Luque, D. S. Levine, M. Gao, M. Dzamba, and C. L. Zitnick, Learning smooth and expressive interatomic potentials for physical property prediction, [arXiv](https://arxiv.org/abs/2502.12147) [preprint arXiv:2502.12147 \(2025\).](https://arxiv.org/abs/2502.12147)
- <span id="page-8-28"></span>[57] D. Flam-Shepherd and A. Aspuru-Guzik, Language models can generate molecules, materials, and protein binding sites directly in three dimensions as xyz, cif, and pdb files, (2023),

- [arXiv:2305.05708 \[cs.LG\].](https://arxiv.org/abs/2305.05708)
- <span id="page-9-0"></span>[58] L. M. Antunes, K. T. Butler, and R. Grau-Crespo, Crystal structure generation with autoregressive large language modeling, [Nature Communications](https://www.nature.com/articles/s41467-024-54639-7) 15, 1 (2024).
- <span id="page-9-1"></span>[59] H. Xu, D. Qian, Z. Liu, Y. Jiang, and J. Wang, Design topological materials by reinforcement fine-tuned generative model, [\(2025\),](https://arxiv.org/abs/2504.13048) [arXiv:2504.13048 \[cond-mat.mtrl-sci\].](https://arxiv.org/abs/2504.13048)
- <span id="page-9-2"></span>[60] J. Chen, J. Guo, and P. Schwaller, Matinvent: Reinforcement learning for 3d crystal diffusion generation, in *[AI for Acceler](https://openreview.net/forum?id=Ovxfri7l5L)[ated Materials Design - ICLR 2025](https://openreview.net/forum?id=Ovxfri7l5L)* (2025).
- <span id="page-9-3"></span>[61] L. Zhang, A. Rao, and M. Agrawala, Adding conditional control to text-to-image diffusion models, [\(2023\),](https://arxiv.org/abs/2302.05543) [arXiv:2302.05543 \[cs.CV\].](https://arxiv.org/abs/2302.05543)
- <span id="page-9-4"></span>[62] Y. Fan and K. Lee, Optimizing ddpm sampling with shortcut fine-tuning, [\(2024\),](https://arxiv.org/abs/2301.13362) [arXiv:2301.13362 \[cs.LG\].](https://arxiv.org/abs/2301.13362)
- <span id="page-9-5"></span>[63] K. Black, M. Janner, Y. Du, I. Kostrikov, and S. Levine, Training diffusion models with reinforcement learning, [\(2024\),](https://arxiv.org/abs/2305.13301) [arXiv:2305.13301 \[cs.LG\].](https://arxiv.org/abs/2305.13301)
- <span id="page-9-6"></span>[64] Y. Fan, O. Watkins, Y. Du, H. Liu, M. Ryu, C. Boutilier, P. Abbeel, M. Ghavamzadeh, K. Lee, and K. Lee, Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models, [\(2023\),](https://arxiv.org/abs/2305.16381) [arXiv:2305.16381 \[cs.LG\].](https://arxiv.org/abs/2305.16381)
- <span id="page-9-7"></span>[65] X.-Q. Han, Z. Ouyang, P.-J. Guo, H. Sun, Z.-F. Gao, and Z.-Y. Lu, Invdesflow: An ai-driven materials inverse design workflow to explore possible high-temperature superconductors, [Chin.](http://iopscience.iop.org/article/10.1088/0256-307X/42/4/047301) [Phys. Lett. \(2025\).](http://iopscience.iop.org/article/10.1088/0256-307X/42/4/047301)
- <span id="page-9-8"></span>[66] Z. Li, S. Liu, B. Ye, D. J. Srolovitz, and T. Wen, [Active learn](https://arxiv.org/abs/2502.16984)[ing for conditional inverse design with crystal generation and](https://arxiv.org/abs/2502.16984) [foundation atomic models](https://arxiv.org/abs/2502.16984) (2025), [arXiv:2502.16984 \[cond](https://arxiv.org/abs/2502.16984)[mat.mtrl-sci\].](https://arxiv.org/abs/2502.16984)
- <span id="page-9-9"></span>[67] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo, Deepseekmath: Pushing the limits of mathematical reasoning in open language models, [\(2024\),](https://arxiv.org/abs/2402.03300) [arXiv:2402.03300 \[cs.CL\].](https://arxiv.org/abs/2402.03300)
- <span id="page-9-10"></span>[68] I. Shenfeld, J. Pari, and P. Agrawal, [Rl's razor: Why online](https://arxiv.org/abs/2509.04259) [reinforcement learning forgets less](https://arxiv.org/abs/2509.04259) (2025), [arXiv:2509.04259](https://arxiv.org/abs/2509.04259) [\[cs.LG\].](https://arxiv.org/abs/2509.04259)
- <span id="page-9-11"></span>[69] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning* (MIT Press, 2016) http://[www.deeplearningbook.org.](http://www.deeplearningbook.org)
- <span id="page-9-12"></span>[70] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn, Direct preference optimization: Your language model is secretly a reward model, [\(2024\),](https://arxiv.org/abs/2305.18290) [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) [\[cs.LG\].](https://arxiv.org/abs/2305.18290)
- <span id="page-9-13"></span>[71] A. K. Cheetham and R. Seshadri, Artificial intelligence driving materials discovery? perspective on the article: Scaling deep learning for materials discovery, [Chemistry of Materials](https://pubs.acs.org/doi/10.1021/acs.chemmater.4c00643) 36[, 3490 \(2024\).](https://pubs.acs.org/doi/10.1021/acs.chemmater.4c00643)
- <span id="page-9-14"></span>[72] J. Clark and D. Amodei, [Faulty reward functions in the wild](https://openai.com/index/faulty-reward-functions/) (2016).
- <span id="page-9-15"></span>[73] R. Gomez-Bombarelli, J. N. Wei, D. Duvenaud, J. M. ´ Hernandez-Lobato, B. S ´ anchez-Lengeling, D. Sheberla, ´ J. Aguilera-Iparraguirre, T. D. Hirzel, R. P. Adams, and A. Aspuru-Guzik, Automatic chemical design using a datadriven continuous representation of molecules, [ACS Central](https://doi.org/10.1021/acscentsci.7b00572) Science 4[, 268 \(2018\),](https://doi.org/10.1021/acscentsci.7b00572) pMID: 29532027.
- [74] C.-Y. Ye, H.-M. Weng, and Q.-S. Wu, Con-cdvae: A method for the conditional generation of crystal structures, [Computational](https://www.sciencedirect.com/science/article/pii/S2950463524000036) [Materials Today](https://www.sciencedirect.com/science/article/pii/S2950463524000036) 1, 100003 (2024).
- <span id="page-9-16"></span>[75] X. Luo, Z. Wang, P. Gao, J. Lv, Y. Wang, C. Chen, and Y. Ma, Deep learning generative model for crystal structure prediction, [npj Computational Materials](https://www.nature.com/articles/s41524-024-01443-y) 10, 254 (2024).
- <span id="page-9-17"></span>[76] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu, Roformer: Enhanced transformer with rotary position embedding, [\(2023\),](https://arxiv.org/abs/2104.09864) [arXiv:2104.09864 \[cs.CL\].](https://arxiv.org/abs/2104.09864)

- <span id="page-9-18"></span>[77] D. P. Kingma and J. Ba, Adam: A method for stochastic optimization, [\(2017\),](https://arxiv.org/abs/1412.6980) [arXiv:1412.6980 \[cs.LG\].](https://arxiv.org/abs/1412.6980)
- <span id="page-9-19"></span>[78] https://[huggingface.co](https://huggingface.co/datasets/zdcao/alex-20)/datasets/zdcao/alex-20.
- <span id="page-9-20"></span>[79] S. Mohamed, M. Rosca, M. Figurnov, and A. Mnih, Monte carlo gradient estimation in machine learning, [Journal of Ma](https://jmlr.org/papers/volume21/19-346/19-346.pdf)[chine Learning Research](https://jmlr.org/papers/volume21/19-346/19-346.pdf) 21, 1 (2020).
- <span id="page-9-21"></span>[80] R. J. Williams, Simple statistical gradient-following algorithms for connectionist reinforcement learning, [Machine learning](https://link.springer.com/article/10.1007/BF00992696) 8, [229 \(1992\).](https://link.springer.com/article/10.1007/BF00992696)
- <span id="page-9-22"></span>[81] Z. Liu, C. Chen, W. Li, P. Qi, T. Pang, C. Du, W. S. Lee, and M. Lin, Understanding r1-zero-like training: A critical perspective, https://github.com/sail-sg/[understand-r1-zero](https://github.com/sail-sg/understand-r1-zero) (2025).
- <span id="page-9-23"></span>[82] B. Deng, P. Zhong, K. Jun, J. Riebesell, K. Han, C. J. Bartel, and G. Ceder, Chgnet: Pretrained universal neural network potential for charge-informed atomistic modeling, [\(2023\),](https://arxiv.org/abs/2302.14231) [arXiv:2302.14231 \[cond-mat.mtrl-sci\].](https://arxiv.org/abs/2302.14231)
- <span id="page-9-24"></span>[83] S. P. Ong, W. D. Richards, A. Jain, G. Hautier, M. Kocher, S. Cholia, D. Gunter, V. L. Chevrier, K. A. Persson, and G. Ceder, Python materials genomics (pymatgen): A robust, open-source python library for materials analysis, [Computa](https://doi.org/https://doi.org/10.1016/j.commatsci.2012.10.028)[tional Materials Science](https://doi.org/https://doi.org/10.1016/j.commatsci.2012.10.028) 68, 314 (2013).
- <span id="page-9-25"></span>[84] P. E. Blochl, Projector augmented-wave method, ¨ [Phys. Rev. B](https://doi.org/10.1103/physrevb.50.17953) 50[, 17953 \(1994\).](https://doi.org/10.1103/physrevb.50.17953)
- <span id="page-9-26"></span>[85] G. Kresse and J. Furthmuller, E ¨ fficient iterative schemes for ab initio total-energy calculations using a plane-wave basis set, Phys. Rev. B 54[, 11169 \(1996\).](https://doi.org/10.1103/physrevb.54.11169)

#### Appendix A: The CrystalFormer model card

The architecture of CrystalFormer is identical to the one described in Ref. [5], except for an update on the positional encoding. We use the rotary positional encoding (RoPE) [76] instead of the absolute positional encoding in the original design. The base model was trained for 5500 epochs with a total batch size of 8000 over eight A100 GPUs using the Adam optimizer [77]. The learning rate was set to 8e - 3.

The reinforcement fine-tuning was performed for 100 - 200 steps, depending on the task. For the reinforcement fine-tuning, we use a total batch size of 1000 over two A100 GPUs and the Adam optimizer with a learning rate of 1e - 4. The pre-training phase required approximately 1000 GPU hours, while the reinforcement fine-tuning phase took around 12 to 24 GPU hours.

For the supervised fine-tuning (SFT) of the CrystalFormer, we generate 10,000 crystal samples and relax them using the Orb models. Only those stable samples with  $E_{\rm hull} < 0.1$  eV/atom were retained for SFT. The SFT was performed for 55 epochs when validation loss stopped improving, with a batch size of 100 using the Adam optimizer with a learning rate of 1e - 5 on a single A100 GPU.

Table S1. A table of hyperparameters used in this work.

| Remarks                        |
|--------------------------------|
|                                |
| 'H' to 'Og', plus padding aton |
| 'a-z'+'A', plus padding atom   |
|                                |
|                                |
|                                |
|                                |
|                                |
|                                |
|                                |
|                                |
|                                |
| No dropout in fine-tuning      |
|                                |
| 2 for FoM reward               |
|                                |
|                                |
| -                              |

#### <span id="page-11-0"></span>Appendix B: Training dataset

The Alexandria dataset [37] was chosen as a starting point because it was the largest openly available DFT dataset of equilibrium and near equilibrium structures (~ 4.5 million materials). Using the same criteria as MatterGen [3] (distances to the convex hull less than 0.1 eV/atom and less than 20 atoms in the unit cell), Alexandria now contains more than 1.3 million materials. 80 % of the data set was used for training and the remaining were divided equally between validation and test datasets. The Alex-20 dataset is available at [78].

Figure S1. Space group distribution of the Alex-20 dataset.

Figure S2. Distribution of elements in Alex-20.

Figure S3. Composition and structure validity of samples generated by the pretrained CrystalFormer using the MP-20 and Alex-20 dataset, as well as reinforcement fine-tuned model using the energy above hull as the reward function.

#### <span id="page-13-0"></span>Appendix C: Details of reinforcement fine-tuning implementation

The proximal policy optimization (PPO) [29] is a simple yet effective policy gradient-based RL algorithm. The algorithm optimizes the following objective function,

$$\mathcal{L}^{\text{PPO}}(\theta) = \underset{x \sim p_{\text{old}}(x)}{\mathbb{E}} \left[ \min(\gamma(\theta)A(x), \text{clip}(\gamma(\theta), 1 - \epsilon, 1 + \epsilon)A(x)) \right], \tag{S1}$$

where  $\gamma(\theta) = \frac{p_{\theta}(x)}{p_{\text{old}}(x)}$  denotes the probability ratio between the policy under optimization and the policy for sampling, A(x) is the advantage function, and  $\epsilon$  is the clipping parameter. The second term,  $\text{clip}(\gamma(\theta), 1 - \epsilon, 1 + \epsilon)$ , modifies the surrogate objective by clipping the probability ratio, which removes the incentive for moving  $\gamma(\theta)$  outside of the interval  $[1 - \epsilon, 1 + \epsilon]$ . Drawing samples from  $p_{\text{old}}(x)$  rather than  $p_{\theta}(x)$  enables sample reuse and allows multiple gradient update steps on the same batch of data, thereby improving sample efficiency without requiring costly re-evaluation of the reward function for each update.

The advantage function  $\mathcal{A}(x)$  is then defined as

$$\mathcal{A}(x) = r(x) - b, (S2)$$

where b represents the baseline. For simplicity, we incorporate the KL regularization term into the advantage function, yielding

$$A(x) = \mathcal{A}(x) - \tau \ln \frac{p_{\theta}(x)}{p_{\text{base}}(x)}.$$
 (S3)

Note that one does not need to compute the gradient of A(x) with respect to network parameters  $\theta$  in the PPO algorithm. Instead, one uses the score function gradient estimate [79] to maximize the objective function  $\mathcal{L}^{PPO}(\theta)$ . For each batch of samples from  $p_{\text{old}}$  and reward signal, we carry out a few gradient ascent steps to update the policy network. The detailed hyperparameters used in the PPO algorithm are listed in Table S1.

The advantage function measures the deviation of the reward r(x) from this baseline, indicating how much better or worse the generated samples is compared to the baseline. In standard PPO implementations, this baseline is typically approximated by a value function, which is trained alongside the policy model. We use the average reward of the generated samples x, i.e.,  $b = \mathbb{E}_{x \sim p_{\text{old}}}[r(x)]$  as the baseline [80]. This approach obviates the need for training a separate value function and is closely related to the group relative policy optimization done right [67, 81].

For the baseline b we use an exponential moving average of the previous rewards, which stabilizes the training and also leverages the past reward information to form a lagged baseline. To be more specific, the baseline is updated as  $b_k = \eta b_{k-1} + (1-\eta)\bar{R}_k$ , where  $\eta = 0.95$  and  $b_0$  is initialized as the average reward of the initial batch  $\bar{R}_0$ . Here,  $\bar{R}_k$  is the average reward of the current batch.

#### <span id="page-14-0"></span>Appendix D: Computational details

#### 1. Machine learning force field

We employ the Orb-v2 [\[13\]](#page-7-3) trained on the MPTraj [\[82\]](#page-9-23) and Alexandria dataset [\[37\]](#page-8-9) to as the reward model for energy prediction for the generated samples. We calculate the energy above the convex hull using the energy predicted by the Orb model in the Materials Project energy correction framework (i.e., MaterialsProject2020Compatibility from pymatgen [\[83\]](#page-9-24)).

#### 2. DFT calculations

The DFT calculations were performed with the Perdew-Burke-Ernzerhof (PBE) exchange-correlation functional [\[46\]](#page-8-18) and allelectron projector-augmented wave method [\[84\]](#page-9-25), as implemented in the VASP code [\[85\]](#page-9-26). All parameters of the calculations including settings of PBE functional, Hubbard U corrections, and ferromagnetic initialization are chosen to be consistent with Materials Project by using of MPRelaxSet function in pymatgen [\[83\]](#page-9-24). All structures containing Yb element are ignored when calculating energy above the hull due to they are unavailable from the Materials Project at the time of writing. The DFPT input files are generated using the MPStaticSet in pymatgen, with a k-point density of 3,000 per reciprocal atom and a plane-wave energy cutoff of 600 eV, consistent with the Materials Project settings [\[45\]](#page-8-17).
