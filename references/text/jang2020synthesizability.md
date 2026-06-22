---
key: jang2020synthesizability
title: Structure-Based Synthesizability Prediction of Crystals Using Partially Supervised
  Learning
year: 2020
primary:
- synthesis
- predictors
role:
- reward-candidate
status: candidate
reward_term:
- stability
domain:
- ml
tags:
- synthesizability
- clscore
- positive-unlabeled
- cgcnn
- beyond-hull
summary: Learned crystal-likeness score (CLscore) via PU learning; candidate synthesizability
  reward beyond energy-above-hull.
---

pubs.acs.org/JACS Article

# Structure-Based Synthesizability Prediction of Crystals Using **Partially Supervised Learning**

Jidon Jang,<sup>†</sup> Geun Ho Gu,<sup>†</sup> Juhwan Noh, Juhwan Kim, and Yousung Jung\*

Cite This: J. Am. Chem. Soc. 2020, 142, 18836-18843

ACCESS

III Metrics & More

Article Recommendations

Supporting Information

ABSTRACT: Predicting the synthesizability of inorganic materials is one of the major challenges in accelerated material discovery. A widely employed approximate approach is to consider the thermodynamic decomposition stability due to its simplicity of computing, but it is notorious for either producing too many candidates or missing important metastable materials. These results, however, are not unexcepted since the synthesizability is a complex phenomenon, and the thermodynamic stability is just one contributor. Here, we suggest a machine-learning model to quantify the probability of synthesis based on the partially supervised learning of materials database. We adapted the positive and unlabeled machine learning (PU learning) by implementing the graph convolutional neural network as a classifier in which the model outputs crystal-likeness scores (CLscore). The model shows 87.4% true positive (CLscore > 0.5) prediction accuracy for the

test set of experimentally reported cases (9356 materials) in the Materials Project. We further validated the model by predicting the synthesizability of newly reported experimental materials in the last 5 years (2015-2019) with an 86.2% true positive rate using the model trained with the database as of the end of year 2014. Our analysis shows that our model captures the structural motif for synthesizability beyond what is possible by E<sub>hull</sub>. We find that 71 materials among the top 100 high-scoring virtual materials have indeed been previously synthesized in the literature. With the proposed data-driven metric of the crystal-likeness score, highthroughput virtual screenings and generative models can benefit significantly by effectively reducing the chemical space that needs to be explored experimentally in the future toward more rational materials design.

# INTRODUCTION

High-performance computational chemistry in the past few decades has been very successful in aiding the experimental discovery of new functional inorganic crystal materials<sup>1-5</sup> via large-scale high-throughput virtual screening (HTVS) based on density functional theory (DFT). In HTVS, one searches through the materials database<sup>6-8</sup> one by one by evaluating their properties of interest until promising candidates are identified. Since these conventional HTVSs are performed on the basis of the existing materials library, in order to go beyond the database as well as to increase the search efficiency in the functional manifold, global optimizations<sup>9-13</sup> have also been used. More recently, these computational approaches are further accelerated by data-driven machine learning (ML) techniques. 14-16 The ML methods accelerate the materials discovery by predicting the material properties with less computational cost than DFT17,18 and by generating new materials with the desired property in an inverse design framework using various flavors of generative models (GMs).  $^{19-24}$

Once the candidate materials are designed either computationally or using data-driven models, the next step is the synthesis. All current approaches for crystal materials design

described above, however, either ignore the synthesis feasibility or estimate the synthesizability simply by considering the thermodynamic formation energies (or energy above the convex hull,  $E_{\text{hull}}$ ). 5,19 Since material synthesis is a complex process based not only on the thermodynamic stability but also on the kinetics, availability of precursors, and experimental conditions,<sup>25</sup> many predicted materials with favorable formation energies are often not synthesized and a great majority of designed materials are thus discarded during the synthesis stage. The opposite is also not uncommon; that is, many useful functional materials are metastable and synthesized even though their formation energies were not on the convex hull minimum.<sup>26</sup> Therefore, a quantitative measure to predict the synthesizability of a material in advance is an urgent tool to reduce unnecessary experimental costs and time for the

Received: July 9, 2020 Published: October 26, 2020

practical acceleration of materials discovery using the aforementioned HTVS and data-driven models. This synthesizability prediction becomes even more important when using GMs. GMs can produce materials with completely different structural motifs, and thus the synthesis feasibility may be lower in general compared to that for materials obtained via the simple substitution of known materials (As shown below, the synthesis probability of GM-produced materials is indeed lower.)

In the area of drug-discovery and molecules, there is a metric called the synthetic accessibility score (SAscore)<sup>27</sup> that assesses the likelihood of synthesis based on the molecular structure. It is a statistical rule-based scoring method which yields low score (easily synthetically accessible) for frequently appearing fragments in the already-synthesized drug-data sets and yields high score (difficult to synthesize) for the presence of nonstandard structural features such as large rings to quantify the synthesizability of a molecule. This metric was comparable to the estimation of synthesizability by experienced chemists. It has been and is still widely used to quickly and roughly estimate the synthesis feasibility of organic molecules in various molecular design research studies. <sup>28–30</sup> However, for crystal structures, there is no such model that assesses the synthesizability.

One of the major challenges to developing a synthesizability model for inorganic crystals is the difficulty in obtaining heuristics to decompose periodic crystal structures in terms of different functional groups or fragments as in molecules. To face this challenge, some attempts to predict synthesizable crystals were made by investigating the existing (synthesizable) crystal structure database. The data-mined structure prediction algorithm (DMSP)<sup>31</sup> suggested many virtual crystal structures of the Materials Project (MP) database by performing statistically probable chemical substitution from existing crystals of the Inorganic Crystal Structure Database (ICSD).<sup>32</sup> This method follows an approach used in the machine translation field<sup>33</sup> and has been quite successful in leading to experimental synthesis for different applications later on.<sup>34–37</sup> Since this method is based on the probability of compositional substitution and not considering any structural motifs or the local environment of atoms, it may cause unrealistic substitutions.

Recently, a crystal structure prediction that can utilize the structural information was reported using a deep neural network (DNN)<sup>38</sup> by predicting the likelihood of the element in local atomic sites. This model could successfully learn chemical knowledge from the existing database and identify the probable specific composition in a given structural template. To predict the overall crystal structure, however, an approximation was made in which the likelihood of synthesizable crystal structure is equal to the product of the likelihood of the residence of the element in all atomic sites. Due to this rule, the likelihood of crystals from large structure templates having many atoms within a large unit cell inordinately diminishes, making the prediction results potentially biased. Another related attempt is a network analysis<sup>39</sup> of stable (ground-state) crystals from their phase diagram to predict the synthesizability of the not-yetsynthesized crystal. The possibility of synthesis was predicted by the sequential training of the network properties along the discovery timeline of crystals in the network. Several materials predicted as synthesizable by this model were indeed synthesized. However, since only the ground-state materials

can generate the network properties by definition, in its present form this model does not consider metastable candidates that might be worthwhile.

In this work, we use a partially supervised classification model to predict the synthesis probability of crystal structures. Popular crystal databases such as ICSD and MP include both experimentally synthesized and DFT-computed virtual (mostly via substitution) crystal structures. Since the synthesizability of virtual crystals is undetermined, these databases have only "positive data" that are already synthesized and "unlabeled data" that are as yet unknown for their synthesizability. To solve this problem, we employ the positive and unlabeled machine learning (PU learning)<sup>40</sup> to classifying the material as either synthesizable or unsynthesizable. This iterative PU learning technique has been successfully applied to various classification tasks in which there are an abundance of unlabeled data with a smaller portion of positive data such as data stream classification, 41 information retrieval, 42 and disease gene identification. 43 Recently, the PU learning implemented with the decision-tree-based classifier was applied to predict the possibility of synthesis for two-dimensional metal carbides and nitrides and their precursors. 44 A key feature of our work that distinguishes itself from the previous approach is that by using graph convolutions to represent the material, the model naturally learns to identify the structural pattern of synthesizable materials during the training, unlike the elemental and computed descriptors used in Frey et al. 44 The use of learnable graph representation also allows the model to be generalizable for all crystal types, not just one class of materials.

### METHODS

**Data Set for Model Training.** The synthesis probability in the present study is trained and tested using the MP database. The MP, accumulated since 2011, is a database of inorganic crystal structures with DFT-calculated properties consisting of almost all elements in the periodic table and is freely accessible through the materials application programming interface (MAPI). A total of 124 515 unique crystal structure data points queried in February 2020 (consisting of 46 781 synthesized crystals associated with the ICSD identifiers and 77 734 theoretically proposed virtual crystals) were used for our PU learning model.

**Graph Convolutional Neural Network.** To devise a general-purpose synthesis probability model, we encode the crystal structure using the graph neural networks (GNN) to represent the crystal. In GNN, atoms and bonds are described as nodes and edges, respectively, and their chemical environments are encoded via convolution or iterative message passing of the neighboring atoms and bonds. These automatically generated material features have been shown to predict various properties (formation energy, band gap, etc.) of materials with high accuracy. In particular, we envision that this structure-based learnable representation is ideally suited to capture the structural pattern of the materials to determine their synthesizability. Detailed network structures are adopted from Xie et al., 45 and their optimized hyperparameters are listed in Table S1.

Positive and Unlabeled Machine Learning Algorithm. The PU learning scheme used in this work (Figure 1) is a variant of the transductive bagging support vector machine developed by Mordelet et al. 46 Frey et al. successfully implemented the decision tree classifier in this algorithm as a natural extension. 44 We adapted the PU learning scheme with the GNN-based classifier proposed by Xie et al. 45 The algorithmic details of our model are as follows: Let P be the positive data set (experimentally synthesized data from MP), U be the unlabeled data set (virtual data from MP), K be the number of positive data, and T be the number of iterations for bagging (bootstrap aggregating). For every iteration, a subsample in U is

<span id="page-2-0"></span>Figure 1. Schematic diagram of PU learning. Green, red, and gray circles express positive, negative, and unlabeled data, respectively.

randomly chosen, denoted as  $\mathbf{U}_n$ where n denotes the nth iteration, and is labeled as negative. The size of this subset  $\mathbf{U}_n$  is chosen to be  $\mathbf{K}$  to ensure a balance for training between truly positive and negative-labeled data. Twenty percent of  $\mathbf{P}$  and  $\mathbf{U}_n$  are set aside as test data for assessing the performance of the classifier. The GNN binary classification model trained using the training sets of  $\mathbf{P}$  and  $\mathbf{U}_n$  is then used to predict the labels of the remaining unlabeled samples ( $\mathbf{U} \setminus \mathbf{U}_n$ ). That is, the classifier predicts the classification score to be close to 1 if the unlabeled sample is likely to be positive-labeled (synthesizable) and close to 0 otherwise. After repeating this  $\mathbf{T}$  times, the final score of each datum point is obtained by averaging the prediction scores from GNN classifiers trained on the subsample in which the datum point is not included. This average value is defined as a crystal-likeness score (CLscore) of between 0 and 1. We use this CLscore to quantify the synthesizability of a given crystal structure.

Out of 124515 Material Project (MP) data points, for every iteration, 46 781 unlabeled data points were randomly selected and labeled as negative data (to balance the 46 781 positive data). Among the total 93 562 labeled data points, 18 712 (20%) data points were set aside as the test set of the GNN classifier with a 1:1 ratio of positive and negative data. The entries of 9356 (20% of the positive data) test positive data points were fixed during all bagging iterations, and their true positive rate were used to assess the PU learning performance. We used T = 100 (i.e., 100 models constructed, which yielded the well converged performance (Figure S1)). In each iteration, the GNN classifier was trained for 50 epochs, providing a converged area under the curve (AUC) (Figure S2b) at reasonable computational costs. We did not use the early stopping method for holdout cross-validation as the model accuracy was similar without the validation set. The CLscores for new crystal candidates are thus the average of 100 predicted values from the 100 classifiers. Source codes used in this PU learning, prediction and their detailed descriptions are available for download from our Github repository (https://github.com/kaist-amsg/Synthesizability-PU-CGCNN).

#### ■ RESULTS AND DISCUSSION

**PU Learning Prediction Results.** Figure 2a shows histogram of the crystal-likeness scores applied to the test

Figure 2. CLscore distribution for the test positive set (experimentally synthesized materials) of the MP database (a) from our main model that uses the truly positive data for training and (b) from the baseline model that uses the randomly labeled positive data for training. (c) and (d) show the predicted CLscore distributions for the virtual materials in the (c) MP and (d) OQMD databases.

positive data in the MP data set. We assessed the model performance by evaluating the true positive rate of the test positive data. If we used the CLscore = 0.5 as a decision boundary for the synthesizability prediction, then 87.4% (8178 out of 9356) of the already synthesized materials are indeed predicted as synthesizable by our model (true positive rate). This high true positive rate shows that the model has indeed learned the structural features of synthesizable materials through iterative classification.

As a baseline model to compare, we also built the PU learning using randomly labeled data. That is, among the total 124 515 MP data points, instead of using the truly positive 46 781 data points for training, we randomly selected 46 781 data points and labeled them as positive and the remaining 77 734 data points were treated as unlabeled. The performance of this baseline model shows that 4251 of the 9356 test positive data points are predicted to be positive (CLscore > 0.5), a true positive rate of 45.4% with the peak at CLscore = 0.5. (Figure 2b). Therefore, this baseline benchmark test clearly demonstrates that, using this arbitrary data set, the model cannot learn and distinguish the synthesizable structure at all and predicted all of the materials in the test set to have a CLscore near 0.5 (i.e., 50:50 random guessing). This result further proves the existence of certain trainable common features in real synthesized crystal structures which can be extracted for our synthesizability prediction model.

In Figure 2c,d, the CLscore distribution for the virtual data without ICSD tags in the MP and the Open Quantum Materials Database (OQMD) is shown. For the MP database, 16 181 of a total of 77 734 virtual materials (20.8%) have CLscore > 0.5. For OQMD, 72 298 out of 599 717 virtual data points (12.1%) show CLscore > 0.5. Interestingly, the higher ratio of the virtual materials with CLscore > 0.5 is observed in MP (20.8%) rather than in OQMD (12.1%). This difference may be due to the difference in the virtual data generation methods between MP and OQMD. In MP, the virtual data are generated using the DMSP algorithm aided by machine-learned probable substitution from the synthesizable data-

base[,31](#page-6-0) while the OQMD database[8](#page-6-0) creates virtual data by calculating predefined prototype structures commonly observed in nature for every combination of elements (excluding the noble gases). We uploaded the CLscores of virtual materials data (i.e., material data without ICSD-id) in MP and OQMD in our Github repository so that those materials with high CLscores can be utilized in various high-throughput virtual screening applications.

Predicting the synthesizability of a material using the simply substituted and yet unrelaxed geometry would be of great value as it bypasses DFT calculations in screening. Thus, we compared the CLscore using initial vs final relaxed structures for the test data in [Figure S3](http://pubs.acs.org/doi/suppl/10.1021/jacs.0c07384/suppl_file/ja0c07384_si_001.pdf) (as available in the MP database). The accuracy of the test positive set is similar between the initial and relaxed structures (87.36 vs 87.41%). For unlabeled data, however, CLscores before and after geometry optimization have larger variations, and 67% of unlabeled data in MP have a higher CLscore after relaxation. This indicates that using a DFT relaxed structure yields a higher possibility of finding more valid and synthesizable candidates.

Model Validation. To validate our model, we sorted the 77 734 virtual data point in MP by a descending order in CLscore and searched the literature to see if any materials among the top 100 synthesizable virtual materials had been synthesized although they have not been documented in the database repository. We find that 71 of these top 100 synthesizable virtual materials have indeed been experimentally synthesized and reported in the scientific papers (12 of them are isostructure with partial occupancy). Among them, the top five crystals are shown in Figure 3 with their MP-id (without ICSD tags), space group, CLscore, and crystal structure visualized by VESTA.[52](#page-7-0) The XRD powder diffraction patterns of these virtual structures match well with those of experimental structures reported in the literature. All of these top 100 materials are tabulated in [Table S2](http://pubs.acs.org/doi/suppl/10.1021/jacs.0c07384/suppl_file/ja0c07384_si_001.pdf), and XRD comparisons for 71 synthesized crystals are demonstrated in [Figure S4.](http://pubs.acs.org/doi/suppl/10.1021/jacs.0c07384/suppl_file/ja0c07384_si_001.pdf)

To further validate our model, we trained the model in the same way described above but using only the positive data that have been created and reported before 2015 in MP to simulate the state of the pre-2015 version of the MP database. Then we tested if this model can correctly predict the synthesizability of the newly synthesized materials between 2015 and 2019. As the cumulative growth of the MP data shows in [Figure S5](http://pubs.acs.org/doi/suppl/10.1021/jacs.0c07384/suppl_file/ja0c07384_si_001.pdf), before 2015, there were 31 833 synthesizable positive data sets, and 935 materials have been newly synthesized and reported since then. As summarized in [Figure 4,](#page-4-0) this model that was trained with the pre-2015 positive data set predicted 806 materials to have CLscore > 0.5, with the true positive rate being 86.2%. This consistent model performance clearly demonstrates that our model can be practically utilized to predict the synthesizable future entries by training with the present version of the database.

CLscore versus Thermodynamic Stability. We next compared the CLscores of virtual materials in MP and their stability in terms of Ehull [\(Figure 5](#page-4-0)), a quantity often used to approximately estimate the synthesizability of a virtually synthesized material. There are two important points to be made.

First, a majority (98.5%) of the materials with a high CLscore (>0.5) are shown to be relatively stable with Ehull < 1 eV/atom. Thus, clearly, the model seems to have learned a property (stability) of materials even though the model has no

Figure 3. Top 5 materials reported in the literature among unlabeled MP data within the top 100 CLscores. (a) Ce2InCu2, (b) Sm(Ge3Pt)4, (c) AuF3, (d) CsAg3Te2, and (e) HgC2(SN)2 are reported in refs [47](#page-7-0)−[51](#page-7-0), respectively. Their mp-ids, space groups, and CLscores are displayed in the parentheses with their visualized primitive structures below. Right insets show the agreement between the computed (MP) and experimental (Exp) X-ray powder diffraction patterns (Cu Kα radiation, λ = 1.54 Å). A complete list of 71 unlabeled data points that were successfully synthesized in the literature, among the top 100 CLscores list is summarized in [Table S2](http://pubs.acs.org/doi/suppl/10.1021/jacs.0c07384/suppl_file/ja0c07384_si_001.pdf) [and Figure S4.](http://pubs.acs.org/doi/suppl/10.1021/jacs.0c07384/suppl_file/ja0c07384_si_001.pdf)

reference to the energy information and utilizes only the structural encoding. In other words, our model learned the structure−property relationships.

Second, however, not all metastable materials with Ehull < 1 eV/atom are predicted to have high CLscores. In fact, 78.6% of the materials with Ehull < 1 eV/atom are predicted to have low CLscores of <0.5, and only 21.4% are predicted to be potentially synthesizable with high CLscores of >0.5. To be

<span id="page-4-0"></span>**Figure 4.** Prediction results of the crystal-likeness score for 935 newly synthesized materials in the last 5 years (2015–2019).

**Figure 5.** Two-dimensional histogram of energy above hull vs a crystal-likeness score of 77 734 unlabeled data points in MP. The inset percentage in each area is the ratio for the total number of unlabeled data points in MP. The right panel shows a magnified plot of the 0-1 eV/atom ( $E_{\rm hull}$ ) range with smaller binning sizes.

more specific, for those virtual materials with  $E_{hull}$  < 1 eV/ atom, the correlation between the CL score and  $E_{\text{hull}}$  is essentially nonexistent ( $R^2 = 0.005$ ). This uncorrelated nature between the synthesizability scores and the stability of materials was similarly observed for positive data as well (Figure S6). In Figure S7, we more systematically compare our synthesizability result with that using the  $E_{\rm hull}$  criteria. If the synthesizability criterion is set to be  $E_{hull}$  < 1 eV, then the successful positive ratio is 98.8% for test positive data as described in Figure S6, but at the same time almost all unlabeled data (95.8%) are also predicted to be synthesizable (meaning almost zero specificity). This is obviously not helpful for the virtual screening process due to the potentially large number of false positives. Conversely, when the criterion value of  $E_{\text{hull}}$  is gradually lowered, the true positive rate becomes smaller than that of our model, and the true positive rate is only 50.4% near  $E_{\text{hull}} = 0$  (convex hull minimum), which is considered to be the most thermodynamically stable phase.

These two points described above lead to an important general perspective that the synthesizability of crystals does not depend solely on the thermodynamic stability for decomposition and may explain why certain relatively unstable materials are synthesized and yet more stable materials are not synthesized.

Similarity Analysis of Crystal Latent Vectors. The process in which our model learns the structural similarity of the synthesized materials in order to predict the synthesizability of unknown compounds can be understood by visualizing the training process of the GNN classifier during every PU learning iteration. For that, we plotted the heat maps

for cosine similarities of crystal hidden feature vectors between the training and test data in MP for the selected training epochs (Figure 6). The receiver operating characteristic

**Figure 6.** Cosine similarities of crystal hidden feature vectors for MP data within one iteration for bagging during PU learning. Blue indicates that the cosine similarity of two vectors is close to 1 and these vectors are similar, while red indicates that these two vectors are different, with a value of 0.

(ROC) curve and convergence of the area under the curve for this iteration are also plotted in Figure S2. Since the current PU classification is learning the frequent crystal structural motifs that appear in already-synthesized compounds iteratively using the graph-based structural encoding, we analyzed the similarity of feature vectors of positive and unlabeled data. In Figure 6, we compare the cosine similarities of test positive (Figure 6a) and test unlabeled data (Figure 6b) relative to the training positive data using hidden feature vectors in the latent space from the final network layer before deciding the CLscore. The material data indices are sorted in ascending order along the axes according to the predicted CLscore. At the initial epoch, the classifier did not distinguish these feature vectors at all. As the training epoch progresses, feature vectors of crystals predicted as synthesizable (high-CLscore over 0.5) become similar (blue-colored) with a synthesizable training data set, while those of low-CLscore crystals become different (red-colored). These results reflect that our GNN classifier successfully identifies the structurally similar features of the training positive data set in the latent space during training. Thus, our model indeed learns the structural similarity of training positive data to discriminate the crystal-like and -unlike data in the latent space within every bagging iteration. In this perspective, misclassified crystals (CLscore < 0.5) in the positive data set may be highly novel structures or outliers from the synthesized data set. Our model validation for the last 5 years (Figure 4) means that candidates having structurally similar features with already-synthesized crystals were actually easier to synthesize, and our model can learn this structuresynthesizability relationship.

**Application to the Generative Model.** As a potential application, we used our model to estimate the synthesizability

<span id="page-5-0"></span>of the materials generated using the generative model. For that, we applied the proposed framework to the 1356 new vanadium oxide (V–O) materials that were generated by sampling the latent space trained with a variational autoencoder (VAE), iMatGen. <sup>19</sup> Our trained PU learning scheme predicts that 4.8% (65 out of 1356) of the newly generated crystal structures may be synthesizable with CLscore > 0.5 (Figure S8). This ratio (4.8%) of synthesizable materials for the generative model is substantially lower than the synthesizable ratios of the virtual data in MP or OQMD (Figure 7). This difference can be

Figure 7. Highly synthesizable ratio of crystals from various sources of databases (MP, OQMD, and iMatGen) predicted by our model. If a CLscore predicted by the model is >0.5, then the corresponding material is defined as a synthesizable entry. Green and gray colors represent already-synthesized and not-yet-synthesized (virtual) crystal data sets, respectively. The CLscores of all virtual materials data (i.e., materials data without experimental ICSD-id) in MP and OQMD can be found in our Github repository for use in high-throughput virtual screening for various applications.

ascribed to the fact that iMatGen samples structural vectors from the latent space to generate new data with completely different structural motifs compared to MP and OQMD, which are basically substitution-based databases of existing materials. These results suggest that the present synthesizability prediction model can be quite helpful in identifying synthesizable materials out of newly found materials from generative models, thus reducing the cost of unsuccessful experimental synthesis of the new findings.

# CONCLUSIONS

A quantitative metric of synthesizability for inorganic crystals, which we denote as the crystal-likeness score (CLscore), is developed on the basis of structural encoding and partially supervised learning. We obtain an 87.4% true positive (CLscore > 0.5) prediction accuracy for the independent test set of experimentally reported cases (9356 materials) in the MP database. As a validation of our model, we demonstrate that the materials experimentally synthesized and reported in the last 5 years between 2015 and 2019 are predicted to be synthesizable (CLscore > 0.5) with a 86.2% true positive rate when the model is trained without these materials using the pre-2015 database. The present model estimates the ratio of synthesizable virtual data in widely used databases such as MP and OQMD to be 20.8 and 12.1%, respectively, which are much higher than 4.8% for one of the generative models,

iMatGen. Among the top 100 high CLscore MP virtual crystals, we find that 71 materials have indeed been synthesized and reported in the literature. Analyzing the similarity of hidden feature vectors revealed that the model successfully extracts the similar features of the synthesizable structures in the latent space for prediction. A further analysis demonstrates that our structure-based learning model captures the motif for synthesizability beyond that possible by  $E_{\rm hull}$ .

Our present approach currently predicts the synthesizability of crystals using only the structural information in the crystal database and can thus be readily applicable for practical applications of various virtual databases and generated materials from GM. However, the external thermodynamic conditions of synthesis (e.g., temperature and pressure) should also be important factors for synthetic accessibility.<sup>53</sup> We anticipate that an improved and more advanced model performance would be obtained by properly considering these external thermodynamic features for training materials together with our structure-based model. In addition, the use of DFT-calculated electron density would also improve the model performance since the electronic structure of the crystal should play a very important role in the synthesis of the crystal but our present structure-based GNN model indirectly captures the electronic structure. Another promising future direction of the current framework is to use explainable artificial intelligence (XAI) to understand which features the machine used to predict the candidate as synthesizable and which features led to the negative decisions. This understanding would eventually lead to a critical step toward the ultimate goal of materials science, namely, understanding and controlling the structure-property (in this case, synthesizability) relationships of materials.

#### ASSOCIATED CONTENT

#### Supporting Information

The Supporting Information is available free of charge at https://pubs.acs.org/doi/10.1021/jacs.0c07384.

Optimized hyperparameters, ROC curve, and AUC convergence for the classifier; top 100 scoring crystals of MP virtual data; XRD comparisons; data growth of MP database; and additional figures for the results of learning and analysis (PDF)

# AUTHOR INFORMATION

# **Corresponding Author**

Yousung Jung — Department of Chemical and Biomolecular Engineering, Korea Advanced Institute of Science and Technology, Daejeon 34141, Republic of Korea; orcid.org/0000-0003-2615-8394; Email: ysjn@kaist.ac.kr

### **Authors**

Jidon Jang — Department of Chemical and Biomolecular Engineering, Korea Advanced Institute of Science and Technology, Daejeon 34141, Republic of Korea

Geun Ho Gu — Department of Chemical and Biomolecular Engineering, Korea Advanced Institute of Science and Technology, Daejeon 34141, Republic of Korea

Juhwan Noh — Department of Chemical and Biomolecular Engineering, Korea Advanced Institute of Science and Technology, Daejeon 34141, Republic of Korea

<span id="page-6-0"></span>Juhwan Kim – Department of Chemical and Biomolecular Engineering, Korea Advanced Institute of Science and Technology, Daejeon 34141, Republic of Korea

Complete contact information is available at: https://pubs.acs.org/10.1021/jacs.0c07384

#### **Author Contributions**

<sup>†</sup>J.J. and G.H.G. contributed equally to this work.

#### Notes

The authors declare no competing financial interest.

# ACKNOWLEDGMENTS

Y.J. acknowledges generous financial support from the National Science Foundation of Korea (NRF-2017R1A2B3010176 and 2019M3D1A1079303) and supercomputing time from KISTI.

# REFERENCES

- (1) Mueller, T.; Hautier, G.; Jain, A.; Ceder, G. Evaluation of Tavorite-Structured Cathode Materials for Lithium-Ion Batteries Using High-Throughput Computing. *Chem. Mater.* **2011**, 23 (17), 3854–3862.
- (2) Hautier, G.; Jain, A.; Ong, S. P.; Kang, B.; Moore, C.; Doe, R.; Ceder, G. Phosphates as Lithium-Ion Battery Cathodes: An Evaluation Based on High-Throughput *ab Initio* Calculations. *Chem. Mater.* **2011**, 23 (15), 3495–3508.
- (3) Gomez-Bombarelli, R.; Aguilera-Iparraguirre, J.; Hirzel, T. D.; Duvenaud, D.; Maclaurin, D.; Blood-Forsythe, M. A.; Chae, H. S.; Einzinger, M.; Ha, D. G.; Wu, T.; Markopoulos, G.; Jeon, S.; Kang, H.; Miyazaki, H.; Numata, M.; Kim, S.; Huang, W.; Hong, S. I.; Baldo, M.; Adams, R. P.; Aspuru-Guzik, A. Design of efficient molecular organic light-emitting diodes by a high-throughput virtual screening and experimental approach. *Nat. Mater.* **2016**, *15* (10), 1120–1127.
- (4) Norskov, J. K.; Bligaard, T.; Rossmeisl, J.; Christensen, C. H. Towards the computational design of solid catalysts. *Nat. Chem.* **2009**, *1* (1), 37–46.
- (5) Singh, A. K.; Montoya, J. H.; Gregoire, J. M.; Persson, K. A. Robust and synthesizable photocatalysts for  $CO_2$  reduction: a data-driven materials discovery. *Nat. Commun.* **2019**, *10* (1), 443.
- (6) Curtarolo, S.; Setyawan, W.; Hart, G. L.; Jahnatek, M.; Chepulskii, R. V.; Taylor, R. H.; Wang, S.; Xue, J.; Yang, K.; Levy, O.; Mehl, M. J.; Stokes, H. T.; Demchenko, D. O.; Morgan, D. AFLOW: an automatic framework for high-throughput materials discovery. *Comput. Mater. Sci.* 2012, 58, 218–226.
- (7) Jain, A.; Ong, S. P.; Hautier, G.; Chen, W.; Richards, W. D.; Dacek, S.; Cholia, S.; Gunter, D.; Skinner, D.; Ceder, G. Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. *APL Mater.* **2013**, *1* (1), 011002.
- (8) Saal, J. E.; Kirklin, S.; Aykol, M.; Meredig, B.; Wolverton, C. Materials Design and Discovery with High-Throughput Density Functional Theory: The Open Quantum Materials Database (OQMD). *JOM* **2013**, *65* (11), 1501–1509.
- (9) Franceschetti, A.; Zunger, A. The inverse band-structure problem of finding an atomic configuration with given electronic properties. *Nature* **1999**, *402* (6757), 60–63.
- (10) Doll, K.; Schön, J.; Jansen, M. Structure prediction based on *ab initio* simulated annealing for boron nitride. *Phys. Rev. B: Condens. Matter Mater. Phys.* **2008**, 78 (14), 144110.
- (11) Amsler, M.; Goedecker, S. Crystal structure prediction using the minima hopping method. J. Chem. Phys. 2010, 133 (22), 224104.
- (12) Glass, C. W.; Oganov, A. R.; Hansen, N. USPEX—Evolutionary crystal structure prediction. *Comput. Phys. Commun.* **2006**, 175 (11–12), 713–720.
- (13) Wang, Y.; Lv, J.; Zhu, L.; Ma, Y. CALYPSO: A method for crystal structure prediction. *Comput. Phys. Commun.* **2012**, *183* (10), 2063–2070.

- (14) Butler, K. T.; Davies, D. W.; Cartwright, H.; Isayev, O.; Walsh, A. Machine learning for molecular and materials science. *Nature* **2018**, 559 (7715), 547–555.
- (15) Meredig, B.; Agrawal, A.; Kirklin, S.; Saal, J. E.; Doak, J. W.; Thompson, A.; Zhang, K.; Choudhary, A.; Wolverton, C. Combinatorial screening for new materials in unconstrained composition space with machine learning. *Phys. Rev. B: Condens. Matter Mater. Phys.* **2014**, 89 (9), 094104.
- (16) Jennings, P. C.; Lysgaard, S.; Hummelshøj, J. S.; Vegge, T.; Bligaard, T. Genetic algorithms for computational materials discovery accelerated by machine learning. NPJ. Comput. Mater. 2019, 5 (1), 46.
- (17) Alberi, K.; Nardelli, M. B.; Zakutayev, A.; Mitas, L.; Curtarolo, S.; Jain, A.; Fornari, M.; Marzari, N.; Takeuchi, I.; Green, M. L. The 2019 materials by design roadmap. *J. Phys. D: Appl. Phys.* 2019, 52 (1), 013001.
- (18) Liu, Y.; Zhao, T.; Ju, W.; Shi, S. Materials discovery and design using machine learning. *J. Materiomics* **2017**, 3 (3), 159–177.
- (19) Noh, J.; Kim, J.; Stein, H. S.; Sanchez-Lengeling, B.; Gregoire, J. M.; Aspuru-Guzik, A.; Jung, Y. Inverse Design of Solid-State Materials via a Continuous Representation. *Matter* **2019**, *1* (5), 1370–1384.
- (20) Hoffmann, J.; Maestrati, L.; Sawada, Y.; Tang, J.; Sellier, J. M.; Bengio, Y., Data-driven approach to encoding and decoding 3-d crystal structures. *arXiv* preprint *arXiv*:1909.00949, 2019.
- (21) Kim, S.; Noh, J.; Gu, G. H.; Aspuru-Guzik, A.; Jung, Y. Generative adversarial networks for crystal structure prediction. *ACS Cent. Sci.* **2020**, *6* (8), 1412–1420.
- (22) Nouira, A.; Sokolovska, N.; Crivello, J.-C., Crystalgan: learning to discover crystallographic structures with generative adversarial networks. *arXiv preprint arXiv:1810.11203*, 2018.
- (23) Kim, B.; Lee, S.; Kim, J. Inverse design of porous materials using artificial neural networks. *Sci. Adv.* **2020**, *6* (1), eaax9324.
- (24) Dong, Y.; Li, D.; Zhang, C.; Wu, C.; Wang, H.; Xin, M.; Cheng, J.; Lin, J., Inverse Structural Design of Graphene/Boron Nitride Hybrids by Regressional GAN. arXiv preprint arXiv:1908.07959, 2019.
- (25) De Yoreo, J. J.; Gilbert, P. U.; Sommerdijk, N. A.; Penn, R. L.; Whitelam, S.; Joester, D.; Zhang, H.; Rimer, J. D.; Navrotsky, A.; Banfield, J. F.; Wallace, A. F.; Michel, F. M.; Meldrum, F. C.; Colfen, H.; Dove, P. M. Crystallization by particle attachment in synthetic, biogenic, and geologic environments. *Science* **2015**, 349 (6247), aaa6760.
- (26) Sun, W.; Dacek, S. T.; Ong, S. P.; Hautier, G.; Jain, A.; Richards, W. D.; Gamst, A. C.; Persson, K. A.; Ceder, G. The thermodynamic scale of inorganic crystalline metastability. *Sci. Adv.* **2016**, *2* (11), e1600225.
- (27) Ertl, P.; Schuffenhauer, A. Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. *J. Cheminf.* **2009**, *1* (1), 8.
- (28) Li, Y.; Zhang, L.; Liu, Z. Multi-objective de novo drug design with conditional graph generative model. *J. Cheminf.* **2018**, *10* (1), 33.
- (29) Guimaraes, G. L.; Sanchez-Lengeling, B.; Outeiral, C.; Farias, P. L. C.; Aspuru-Guzik, A. Objective-reinforced generative adversarial networks (ORGAN) for sequence generation models. *arXiv preprint arXiv:1705.10843*, 2017.
- (30) De Cao, N.; Kipf, T. MolGAN: An implicit generative model for small molecular graphs. arXiv preprint arXiv:1805.11973, 2018.
- (31) Hautier, G.; Fischer, C.; Ehrlacher, V.; Jain, A.; Ceder, G. Data mined ionic substitutions for the discovery of new compounds. *Inorg. Chem.* **2011**, *50* (2), 656–663.
- (32) Bergerhoff, G.; Brown, I.; Allen, F. Crystallographic databases. *International Union of Crystallography, Chester* **1987**, 360, 77–95.
- (33) Brown, P. F.; Pietra, V. J. D.; Pietra, S. A. D.; Mercer, R. L. The mathematics of statistical machine translation: Parameter estimation. *Comput. Linguist.* **1993**, *19* (2), 263–311.
- (34) Jain, A.; Hautier, G.; Moore, C.; Kang, B.; Lee, J.; Chen, H.; Twu, N.; Ceder, G. A Computational Investigation of  $\text{Li}_9M_3(P_2O_7)_3(PO_4)_2(M=V,Mo)$  as Cathodes for Li Ion Batteries. *J. Electrochem. Soc.* **2012**, *159* (5), A622–A633.
- (35) Chen, H.; Hautier, G.; Jain, A.; Moore, C.; Kang, B.; Doe, R.; Wu, L.; Zhu, Y.; Tang, Y.; Ceder, G. Carbonophosphates: A New

- <span id="page-7-0"></span>[Family of Cathode Materials for Li-Ion Batteries Identified Computa](https://dx.doi.org/10.1021/cm203243x)[tionally.](https://dx.doi.org/10.1021/cm203243x) Chem. Mater. 2012, 24 (11), 2009−2016.
- (36) Collins, C.; Dyer, M. S.; Demont, A.; Chater, P. A.; Thomas, M. F.; Darling, G. R.; Claridge, J. B.; Rosseinsky, M. J[. Computational](https://dx.doi.org/10.1039/C3SC52734D) [prediction and experimental confirmation of B-site doping in](https://dx.doi.org/10.1039/C3SC52734D) [YBa2Fe3O8.](https://dx.doi.org/10.1039/C3SC52734D) Chem. Sci. 2014, 5 (4), 1493−1505.
- (37) Mammoottil Abraham, A.; Kammampata, S. P.; Ponnurangam, S.; Thangadurai, V[. Efficient Synthesis and Characterization of Robust](https://dx.doi.org/10.1021/acsami.9b11967) MoS2 [and S Cathode for Advanced Li-S Battery: Combined](https://dx.doi.org/10.1021/acsami.9b11967) [Experimental and Theoretical Studies.](https://dx.doi.org/10.1021/acsami.9b11967) ACS Appl. Mater. Interfaces 2019, 11 (39), 35729−35737.
- (38) Ryan, K.; Lengyel, J.; Shatruk, M[. Crystal Structure Prediction](https://dx.doi.org/10.1021/jacs.8b03913) [via Deep Learning.](https://dx.doi.org/10.1021/jacs.8b03913) J. Am. Chem. Soc. 2018, 140 (32), 10158−10168.
- (39) Aykol, M.; Hegde, V. I.; Hung, L.; Suram, S.; Herring, P.; Wolverton, C.; Hummelshoj, J. S[. Network analysis of synthesizable](https://dx.doi.org/10.1038/s41467-019-10030-5) [materials discovery.](https://dx.doi.org/10.1038/s41467-019-10030-5) Nat. Commun. 2019, 10 (1), 2018.
- (40) Elkan, C.; Noto, K. Learning Classifiers from Only Positive and Unlabeled Data; Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; 2008; pp 213−220.
- (41) Li, X.-L.; Yu, P. S.; Liu, B.; Ng, S.-K. Positive Unlabeled Learning for Data Stream Classification; Proceedings of the 2009 SIAM International Conference on Data Mining; SIAM: 2009; pp 259−270.
- (42) Teng, C.; He, Y.; Ji, D.; Ren, H.; Yang, L.; Xiong, W. Information Retrieval Using PU Learning Based Re-ranking; NTCIR, 2008.
- (43) Mordelet, F.; Vert, J.-P[. ProDiGe: Prioritization Of Disease](https://dx.doi.org/10.1186/1471-2105-12-389) [Genes with multitask machine learning from positive and unlabeled](https://dx.doi.org/10.1186/1471-2105-12-389) [examples.](https://dx.doi.org/10.1186/1471-2105-12-389) BMC Bioinf. 2011, 12 (1), 389.
- (44) Frey, N. C.; Wang, J.; Vega Bellido, G. I.; Anasori, B.; Gogotsi, Y.; Shenoy, V. B. [Prediction of Synthesis of 2D Metal Carbides and](https://dx.doi.org/10.1021/acsnano.8b08014) [Nitrides \(MXenes\) and Their Precursors with Positive and Unlabeled](https://dx.doi.org/10.1021/acsnano.8b08014) [Machine Learning.](https://dx.doi.org/10.1021/acsnano.8b08014) ACS Nano 2019, 13 (3), 3031−3041.
- (45) Xie, T.; Grossman, J. C. [Crystal Graph Convolutional Neural](https://dx.doi.org/10.1103/PhysRevLett.120.145301) [Networks for an Accurate and Interpretable Prediction of Material](https://dx.doi.org/10.1103/PhysRevLett.120.145301) [Properties.](https://dx.doi.org/10.1103/PhysRevLett.120.145301) Phys. Rev. Lett. 2018, 120 (14), 145301.
- (46) Mordelet, F.; Vert, J. P. [A bagging SVM to learn from positive](https://dx.doi.org/10.1016/j.patrec.2013.06.010) [and unlabeled examples.](https://dx.doi.org/10.1016/j.patrec.2013.06.010) Pattern Recognit. Lett. 2014, 37, 201−209.
- (47) Kaczorowski, D.; Rogl, P.; Hiebl, K. [Magnetic behavior in a](https://dx.doi.org/10.1103/PhysRevB.54.9891) [series of cerium ternary intermetallics: Ce2T2In \(T = Ni, Cu, Rh, Pd,](https://dx.doi.org/10.1103/PhysRevB.54.9891) [Pt, and Au\).](https://dx.doi.org/10.1103/PhysRevB.54.9891) Phys. Rev. B: Condens. Matter Mater. Phys. 1996, 54 (14), 9891−9902.
- (48) Gumeniuk, R.; Schöneich, M.; Leithe-Jasper, A.; Schnelle, W.; Nicklas, M.; Rosner, H.; Ormeci, A.; Burkhardt, U.; Schmidt, M.; Schwarz, U.; Ruck, M.; Grin, Y. [High-pressure synthesis and exotic](https://dx.doi.org/10.1088/1367-2630/12/10/103035) [heavy-fermion behaviour of the filled skutterudite SmPt4Ge12.](https://dx.doi.org/10.1088/1367-2630/12/10/103035) New J. Phys. 2010, 12 (10), 103035.
- (49) Einstein, F.; Rao, P.; Trotter, J.; Bartlett, N[. The crystal](https://dx.doi.org/10.1039/j19670000478) [structure of gold trifluoride.](https://dx.doi.org/10.1039/j19670000478) J. Chem. Soc. A 1967, 478−482.
- (50) Li, J.; Guo, H.-Y.; Zhang, X.; Kanatzidis, M. G. [CsAg5Te3: a](https://dx.doi.org/10.1016/0925-8388(94)01359-4) [new metal-rich telluride with a unique tunnel structure.](https://dx.doi.org/10.1016/0925-8388(94)01359-4) J. Alloys Compd. 1995, 218 (1), 1−4.
- (51) Beauchamp, A. L.; Goutier, D. [Structure cristalline et](https://dx.doi.org/10.1139/v72-153) [moleculaire du thiocyanate mercurique.](https://dx.doi.org/10.1139/v72-153) Can. J. Chem. 1972, 50, 977−981.
- (52) Momma, K.; Izumi, F[. VESTA 3 for three-dimensional](https://dx.doi.org/10.1107/S0021889811038970) [visualization of crystal, volumetric and morphology data.](https://dx.doi.org/10.1107/S0021889811038970) J. Appl. Crystallogr. 2011, 44 (6), 1272−1276.
- (53) Aykol, M.; Dwaraknath, S. S.; Sun, W.; Persson, K. A. [Thermodynamic limit for synthesis of metastable inorganic materials.](https://dx.doi.org/10.1126/sciadv.aaq0148) Sci. Adv. 2018, 4 (4), eaaq0148.
