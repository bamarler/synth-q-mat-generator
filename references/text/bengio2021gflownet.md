---
key: bengio2021gflownet
title: Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation
year: 2021
primary:
- rl
role:
- rl-algorithm
status: candidate
reward_term: []
domain:
- ml
tags:
- gflownet
- diversity
- reward-proportional
- mode-coverage
summary: GFlowNets; sample proportional to reward for diverse candidate batches rather
  than a single optimum.
---

# Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation

Emmanuel Bengio1,<sup>2</sup> , Moksh Jain1,<sup>5</sup> , Maksym Korablyov<sup>1</sup> Doina Precup1,2,<sup>4</sup> , Yoshua Bengio1,<sup>3</sup> <sup>1</sup>Mila, <sup>2</sup>McGill University, <sup>3</sup>Université de Montréal, <sup>4</sup>DeepMind, <sup>5</sup>Microsoft

# Abstract

This paper is about the problem of learning a stochastic policy for generating an object (like a molecular graph) from a sequence of actions, such that the probability of generating an object is proportional to a given positive reward for that object. Whereas standard return maximization tends to converge to a single return-maximizing sequence, there are cases where we would like to sample a diverse set of high-return solutions. These arise, for example, in black-box function optimization when few rounds are possible, each with large batches of queries, where the batches should be diverse, e.g., in the design of new molecules. One can also see this as a problem of approximately converting an energy function to a generative distribution. While MCMC methods can achieve that, they are expensive and generally only perform local exploration. Instead, training a generative policy amortizes the cost of search during training and yields to fast generation. Using insights from Temporal Difference learning, we propose GFlowNet, based on a view of the generative process as a flow network, making it possible to handle the tricky case where different trajectories can yield the same final state, e.g., there are many ways to sequentially add atoms to generate some molecular graph. We cast the set of trajectories as a flow and convert the flow consistency equations into a learning objective, akin to the casting of the Bellman equations into Temporal Difference methods. We prove that any global minimum of the proposed objectives yields a policy which samples from the desired distribution, and demonstrate the improved performance and diversity of GFlowNet on a simple domain where there are many modes to the reward function, and on a molecule synthesis task.

# 1 Introduction

The maximization of expected return R in reinforcement learning (RL) is generally achieved by putting all the probability mass of the policy π on the highest-return sequence of actions. In this paper, we study the scenario where our objective is not to generate the single highest-reward sequence of actions but rather to sample a distribution of trajectories whose probability is proportional to a given positive return or reward function. This can be useful in tasks where exploration is important, i.e., we want to sample from the leading modes of the return function. This is equivalent to the problem of turning an energy function into a corresponding generative model, where the object to be generated is obtained via a sequence of actions. By changing the temperature of the energy function (i.e., scaling it multiplicatively) or by taking the power of the return, one can control how selective the generator should be, i.e., only generate from around the highest modes at low temperature or explore more with a higher temperature.

A motivating application for this setup is iterative black-box optimization where the learner has access to an oracle which can compute a reward for a large batch of candidates at each round, e.g., in drug-discovery applications. Diversity of the generated candidates is particularly important when the oracle is itself uncertain, e.g., it may consist of cellular assays which is a cheap proxy for clinical

trials, or it may consist of the result of a docking simulation (estimating how well a candidate small molecule binds to a target protein) which is a proxy for more accurate but more expensive downstream evaluations (like cellular assays or in-vivo assays in mice).

When calling the oracle is expensive (e.g. it involves a biological experiment), a standard way [\(Anger](#page-10-0)[mueller et al.,](#page-10-0) [2020\)](#page-10-0) to apply machine learning in such exploration settings is to take the data already collected from the oracle (say a set of (x, y) pairs where x is a candidate solution an y is a scalar evaluation of x from the oracle) and train a supervised proxy f (viewed as a simulator) which predicts y from x. The function f or a variant of f which incorporates uncertainty about its value, like in Bayesian optimization [\(Srinivas et al.,](#page-12-0) [2010;](#page-12-0) [Negoescu et al.,](#page-11-0) [2011\)](#page-11-0), can then be used as a reward function R to train a generative model or a policy that will produce a batch of candidates for the next experimental assays. Searching for x which maximizes R(x) is not sufficient because we would like to sample for the batch of queries a representative set of x's with high values of R, i.e., around modes of R(x). Note that alternative ways to obtain diversity exist, e.g., with batch Bayesian optimization [\(Kirsch et al.,](#page-11-1) [2019\)](#page-11-1). An advantage of the proposed approach is that the computational cost is linear in the size of the batch (by opposition with methods which compare pairs of candidates, which is at least quadratic). With the possibility of assays of a hundred thousand candidates using synthetic biology, linear scaling would be a great advantage.

In this paper, we thus focus on the specific machine learning problem of turning a given positive reward or return function into a generative policy which samples with a probability proportional to the return. In applications like the one mentioned above, we only apply the reward function after having generated a candidate, i.e., the reward is zero except in a terminal state, and the return is the terminal reward. We are in the so-called episodic setting of RL.

The proposed approach views the probability assigned to an action given a state as the flow associated with a network whose nodes are states, and outgoing edges from that node are deterministic transitions driven by an action (not to be confused with normalizing flows; [Rezende and Mohamed](#page-12-1) [\(2016\)](#page-12-1)). The total flow into the network is the sum of the rewards in the terminal states (i.e., a partition function) and can be shown to be the flow at the root node (or start state). The proposed algorithm is inspired by Bellman updates and converges when the incoming and outgoing flow into and out of each state match. A policy which chooses an action with probability proportional to the outgoing flow corresponding to that action is proven to achieve the desired result, i.e., the probability of sampling a terminal state is proportional to its reward. In addition, we show that the resulting setup is off-policy; it converges to the above solution even if the training trajectories come from a different policy, so long as it has large enough support on the state space.

The main contributions of this paper are as follows:

- We propose GFlowNet, a novel generative method for unnormalized probability distributions based on flow networks and local flow-matching conditions: the flow incoming to a state must match the outgoing flow.
- We prove crucial properties of GFlowNet, including the link between the flow-matching conditions (which many training objectives can provide) and the resulting match of the generated policy with the target reward function. We also prove its offline properties and asymptotic convergence (if the training objective can be minimized). We also demonstrate that previous related work [\(Buesing et al.,](#page-10-1) [2019\)](#page-10-1) which sees the generative process like a tree would fail when there are many action sequences which can lead to the same state.
- We demonstrate on synthetic data the usefulness of departing from seeking one mode of the return, and instead seeking to model the entire distribution and all its modes.
- We successfully apply GFlowNet to a large scale molecule synthesis domain, with comparative experiments against PPO and MCMC methods.

All implementations are available at <https://github.com/bengioe/gflownet>.

# 2 Approximating Flow Network generative models with a TD-like objective

Consider a discrete set X and policy π(a|s) to sequentially build x ∈ X with probability π(x) with

$$\pi(x) \approx \frac{R(x)}{Z} = \frac{R(x)}{\sum_{x' \in \mathcal{X}} R(x')}$$
 (1)

where R(x)>0 is a reward for a terminal state x. This would be useful to sample novel drug-like molecules when given a reward function R that scores molecules based on their chemical properties. Being able to sample from the high modes of R(x) would provide diversity in the batches of generated molecules sent to assays. This is in contrast with the typical RL objective of maximizing return which we have found to often end up focusing around one or very few good molecules. In our context, R(x) is a proxy for the actual values obtained from assays, which means it can be called often and cheaply. R(x) is retrained or fine-tuned each time we acquire new data from the assays.

What method should one use to generate batches sampled from  $\pi(x) \propto R(x)$ ? Let's first think of the state space under which we would operate.

Let  $\mathcal S$  denote the set of states and  $\mathcal X\subset\mathcal S$  denote the set of terminal states. Let  $\mathcal A$  be a finite set, the alphabet,  $\mathcal A(s)\subseteq\mathcal A$  be the set of allowed actions at state s, and let  $\mathcal A^*(s)$  be the set of all sequences of actions allowed after state s. To every action sequence  $\vec a=(a_1,a_2,a_3,...,a_h)$  of  $a_i\in\mathcal A,h\leq H$  corresponds a single x, i.e. the environment is deterministic so we can define a function C mapping a sequence of actions  $\vec a$  to an x. If such a sequence is 'incomplete' we define its reward to be 0. When the correspondence between action sequences and states is **bijective**, a state s is uniquely described by some sequence  $\vec a$ , and we can visualize the generative process as the traversal of a tree from a single root node to a leaf corresponding to the sequence of actions along the way.

However, when this correspondence is **non-injective**, i.e. when multiple action sequences describe the same x, things get trickier. Instead of a tree, we get a directed acyclic graph or DAG (assuming that the sequences must be of finite length, i.e., there are no deterministic cycles), as illustrated in Figure 1. For example, and of interest here, molecules can be seen as graphs, which can be described in multiple orders (canonical representations such as SMILES strings also have this problem: there may be multiple descriptions for the same actual molecule). The standard approach to such a sampling problem is to use iterative MCMC methods (Xie et al., 2021; Grathwohl et al., 2021). Another option is to relax the desire to have  $p(x) \propto R(x)$  and to use non-interative (sequential) RL methods (Gottipati et al., 2020), but these are at high risk of getting stuck in local maxima and of missing modes. Indeed, in our setting, the policy which maximizes the expected return (which is the expected final reward) generates the sequence with the highest return (i.e., a single molecule).

#### 2.1 Flow Networks

In this section we propose the Generative Flow Network framework, or GFlowNet, which enables us to learn policies such that  $p(x) \propto R(x)$  when sampled. We first discuss why existing methods are inadequate, and then show how we can use the metaphor of flows, sinks and sources, to construct adequate policies. We then show that such policies can be learned via a flow-matching objective.

With existing methods in the bijective case, one can think of the sequential generation of one x as an episode in a tree-structured deterministic MDP, where all leaves x are terminal states (with reward R(x)) and the root is initial state  $s_0$ . Interestingly, in such a case one can express the pseudo-value of a state  $\tilde{V}(s)$  as the sum of all the rewards of the descendants of s (Buesing et al., 2019).

In the non-injective case, these methods are inadequate. Constructing  $\pi(\tau) \approx R(\tau)/Z$ , e.g. as per Buesing et al. (2019), MaxEnt RL (Haarnoja et al., 2017), or via an autoregressive method (Nash and Durkan, 2019; Shi et al., 2021) has a particular problem as shown below: if multiple action sequences  $\vec{a}$  (i.e. multiple trajectories  $\tau$ ) lead to a final state x, then a serious bias can be introduced in the generative probabilities. Let us denote  $\vec{a} + \vec{b}$  as the concatenation of the two sequences of actions  $\vec{a}$  and  $\vec{b}$ , and by extension  $s + \vec{b}$  the state reached by applying the actions in  $\vec{b}$  from state s.

<span id="page-2-0"></span>**Proposition 1.** Let  $C: A^* \mapsto \mathcal{S}$  associate each allowed action sequence  $\vec{a} \in A^*$  to a state  $s = C(\vec{a}) \in \mathcal{S}$ . Let  $\tilde{V}: \mathcal{S} \mapsto \mathbf{R}^+$  associate each state  $s \in \mathcal{S}$  to  $\tilde{V}(s) = \sum_{\vec{b} \in A^*(s)} R(s + \vec{b}) > 0$ , where  $A^*(s)$  is the set of allowed continuations from s and  $s + \vec{b}$  denotes the resulting state, i.e.,  $\tilde{V}(s)$  is the sum of the rewards of all the states reachable from s. Consider a policy  $\pi$  which starts from the state corresponding to the empty string  $s_0 = C(\emptyset)$  and chooses from state  $s \in \mathcal{S}$  an allowable action  $a \in \mathcal{A}(s)$  with probability  $\pi(a|s) = \frac{\tilde{V}(s+a)}{\sum_{b \in \mathcal{A}(s)} \tilde{V}(s+b)}$ . Denote  $\pi(\vec{a} = (a_1, \ldots, a_N)) = \prod_{i=1}^N \pi(a_i|C(a_1, \ldots, a_{i-1}))$  and  $\pi(s)$  with  $s \in \mathcal{S}$  the probability of visiting a state s with this policy. The following then obtains:  $(a) \pi(s) = \sum_{\vec{a}_i:C(\vec{a}_i)=s} \pi(\vec{a}_i)$ .

(b) If C is bijective, then  $\pi(s) = \frac{\tilde{V}(s)}{\tilde{V}(s_0)}$  and as a special case for terminal states x,  $\pi(x) = \frac{R(x)}{\sum_{x \in \mathcal{X}} R(x)}$ . (c) If C is non-injective and there are n(x) distinct action sequences  $\vec{a}_i$  s.t.  $C(\vec{a}_i) = x$ , then  $\pi(x) = \frac{n(x)R(x)}{\sum_{x' \in \mathcal{X}} n(x')R(x')}$ .

See Appendix A.1 for the proof. In combinatorial spaces, such as for molecules, where C is non-injective (there are many ways to construct a molecule graph), this can become exponentially bad as trajectory lengths increase. It means that larger molecules would be exponentially more likely to be sampled than smaller ones, just because of the many more paths leading to them. In this scenario, the pseudo-value  $\tilde{V}$  is "misinterpreting" the MDP's structure as a tree, leading to the wrong  $\pi(x)$ .

An alternative is to see the MDP as a **flow network**, that is, leverage the DAG structure of the MDP, and learn a flow F, rather than estimating the pseudo-value  $\tilde{V}$  as a sum of descendant rewards, as elaborated below. We define the flow network as a having a single source, the root node (or initial state)  $s_0$  with in-flow Z, and one sink for each leaf (or terminal state) x with out-flow R(x) > 0. We write T(s,a) = s' to denote that the state-action pair (s,a) leads to state s'. Note that because C is not a bijection, i.e., there are many paths (action sequences) leading to some node, a node can have multiple parents, i.e.  $|\{(s,a) \mid T(s,a) = s'\}| \geq 1$ , except for the root, which has no parent. We write F(s,a) for the flow between node s and node s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s' = T(s,a), s'

<span id="page-3-0"></span>Figure 1: A flow network MDP. Episodes start at source  $s_0$  with flow Z. Like with SMILES strings, there are no cycles. Terminal states are sinks with out-flow R(s). Exemplar state  $s_3$  has parents  $\{(s,a)|T(s,a)=s_3\}=\{(s_1,a_2),(s_2,a_5)\}$  and allowed actions  $\mathcal{A}(s_3)=\{a_4,a_7\}$ .  $s_4$  is a terminal sink state with  $R(s_4)>0$  and only one parent. The goal is to estimate F(s,a) such that the flow equations are satisfied for all states: for each node, incoming flow equals outgoing flow.

To satisfy flow conditions, we require that for any node, the incoming flow equals the outgoing flow, which is the total flow F(s) of node s. Boundary conditions are given by the flow into the terminal nodes x, R(x). Formally, for any node s', we must have that the in-flow

$$F(s') = \sum_{s,a:T(s,a)=s'} F(s,a)$$
 (2)

equals the out-flow

<span id="page-3-2"></span>
$$F(s') = \sum_{a' \in \mathcal{A}(s')} F(s', a'). \tag{3}$$

More concisely, with R(s) = 0 for interior nodes, and  $A(s) = \emptyset$  for leaf (sink/terminal) nodes, we write the following flow consistency equations:

$$\sum_{s,a:T(s,a)=s'} F(s,a) = R(s') + \sum_{a' \in \mathcal{A}(s')} F(s',a'). \tag{4}$$

with F being a flow,  $F(s,a) > 0 \ \forall s,a$  (for this we needed to constrain R(x) to be positive too). One could include in principle nodes and edges with zero flow but it would make it difficult to talk about the logarithm of the flow, as we do below, and such states can always be excluded by the allowed set of actions for their parent states. Let us now show that such a flow correctly produces  $\pi(x) = R(x)/Z$  when the above flow equations are satisfied.

<span id="page-3-1"></span><sup>&</sup>lt;sup>1</sup>In some sense, F(s) and F(s,a) are close to V and Q, RL's value and action-value functions. These effectively inform an agent taking decisions at each step of an MDP to act in a desired way. With some work, we can also show an equivalence between F(s,a) and the "real"  $Q^{\hat{\pi}}$  of some policy  $\hat{\pi}$  in a modified MDP (see A.2).

<span id="page-4-2"></span>**Proposition 2.** Let us define a policy  $\pi$  that generates trajectories starting in state  $s_0$  by sampling actions  $a \in A(s)$  according to

<span id="page-4-0"></span>
$$\pi(a|s) = \frac{F(s,a)}{F(s)} \tag{5}$$

where F(s,a) > 0 is the flow through allowed edge (s,a),  $F(s) = R(s) + \sum_{a \in \mathcal{A}(s)} F(s,a)$  where R(s) = 0 for non-terminal nodes s and F(x) = R(x) > 0 for terminal nodes x, and the flow consistency equation  $\sum_{s,a:T(s,a)=s'} F(s,a) = R(s') + \sum_{a' \in \mathcal{A}(s')} F(s',a')$  is satisfied. Let  $\pi(s)$  denote the probability of visiting state s when starting at  $s_0$  and following  $\pi(\cdot|\cdot)$ . Then

(a)
$$\pi(s) = \frac{F(s)}{F(s_0)}$$

(b)  $F(s_0) = \sum_{x \in \mathcal{X}} R(x)$
(c)  $\pi(x) = \frac{R(x)}{\sum_{x' \in \mathcal{X}} R(x')}$ .

*Proof.* We have  $\pi(s_0)=1$  since we always start in root node  $s_0$ . Note that  $\sum_{x\in\mathcal{X}}\pi(x)=1$  because terminal states are mutually exclusive, but in the case of non-bijective C, we cannot say that  $\sum_{s\in\mathcal{S}}\pi(s)$  equals 1 because the different states are not mutually exclusive in general. This notation is different from the one typically used in RL where  $\pi(s)$  refers to the asymptotic distribution of the Markov chain. Then

$$\pi(s') = \sum_{(a,s):T(s,a)=s'} \pi(a|s)\pi(s)$$
(6)

i.e., using Eq. 5,

$$\pi(s') = \sum_{(a,s):T(s,a)=s'} \frac{F(s,a)}{F(s)} \pi(s). \tag{7}$$

We can now conjecture that the statement

$$\pi(s) = \frac{F(s)}{F(s_0)} \tag{8}$$

is true and prove it by induction. This is trivially true for the root, which is our base statement, since  $\pi(s_0) = 1$ . By induction, we then have that if the statement is true for parents s of s', then

$$\pi(s') = \sum_{s,a:T(s,a)=s'} \frac{F(s,a)}{F(s)} \frac{F(s)}{F(s_0)} = \frac{\sum_{s,a:T(s,a)=s'} F(s,a)}{F(s_0)} = \frac{F(s')}{F(s_0)}$$
(9)

which proves the statement, i.e., the first conclusion (a) of the theorem. We can then apply it to the case of terminal states x, whose flow is fixed to F(x) = R(x) and obtain

<span id="page-4-1"></span>
$$\pi(x) = \frac{R(x)}{F(s_0)}. (10)$$

Noting that  $\sum_{x \in X} \pi(x) = 1$  and summing both sides of Eq. 10 over x we thus obtain (b), i.e.,  $F(s_0) = \sum_{x \in \mathcal{X}} R(x)$ . Plugging this back into Eq. 10, we obtain (c), i.e.,  $\pi(x) = \frac{R(x)}{\sum_{x' \in \mathcal{X}} R(x')}$ .

Thus our choice of  $\pi(a|s)$  satisfies our desiderata: it maps a reward function R to a generative model which generates x with probability  $\pi(x) \propto R(x)$ , whether C is bijective or non-injective (the former being a special case of the latter, and we just provided a proof for the general non-injective case).

### 2.2 Objective Functions for GFlowNet

We can now leverage our RL intuitions to create a learning algorithm out of the above theoretical results. In particular, we propose to approximate the flows F such that the flow consistency equations are respected at convergence with enough capacity in our estimator of F, just like the Bellman equations for temporal-difference (TD) algorithms (Sutton and Barto, 2018). This could yield the following objective for a trajectory  $\tau$ :

$$\tilde{\mathcal{L}}_{\theta}(\tau) = \sum_{s' \in \tau \neq s_0} \left( \sum_{s,a:T(s,a)=s'} F_{\theta}(s,a) - R(s') - \sum_{a' \in \mathcal{A}(s')} F_{\theta}(s',a') \right)^2. \tag{11}$$

One issue from a learning point of view is that the flow will be very large for nodes near the root (early in the trajectory) and tiny for nodes near the leaves (late in the trajectory). In high-dimensional spaces where the cardinality of  $\mathcal X$  is exponential (e.g., in the typical number of actions to form an x), the F(s,a) and F(s) for early states will be exponentially larger than for later states. Since we want F(s,a) to be the output of a neural network, this would lead to serious numerical issues.

To avoid this problem, we define the flow matching objective on a log-scale, where we match not the incoming and outgoing flows but their logarithms, and we train our predictor to estimate  $F_{\theta}^{\log}(s,a) = \log F(s,a)$ , and exponentiate-sum-log the  $F_{\theta}^{\log}$  predictions to compute the loss, yielding the square of a difference of logs:

<span id="page-5-0"></span>
$$\mathcal{L}_{\theta,\epsilon}(\tau) = \sum_{s' \in \tau \neq s_0} \left( \log \left[ \epsilon + \sum_{s,a:T(s,a)=s'} \exp F_{\theta}^{\log}(s,a) \right] - \log \left[ \epsilon + R(s') + \sum_{a' \in \mathcal{A}(s')} \exp F_{\theta}^{\log}(s',a') \right] \right)^2$$
(12)

which gives equal gradient weighing to large and small magnitude predictions. Note that matching the logs of the flows is equivalent to making the ratio of the incoming and outgoing flow closer to 1. To give more weight to errors on large flows and avoid taking the logarithm of a tiny number, we compare  $\log(\epsilon+\mathrm{incoming}\ \mathrm{flow})$  with  $\log(\epsilon+\mathrm{outgoing}\ \mathrm{flow})$ . It does not change the global minimum, which is still when the flow equations are satisfied, but it avoids numerical issues with taking the log of a tiny flow. The hyper-parameter  $\epsilon$  also trades-off how much pressure we put on matching large versus small flows, and in our experiments is set to be close to the smallest value R can take. Since we want to discover the top modes of R, it makes sense to care more for the larger flows. Many other objectives are possible for which flow matching is also a global minimum.

An interesting advantage of such objective functions is that they yield off-policy offline methods. The predicted flows F do not depend on the policy used to sample trajectories (apart from the fact that the samples should sufficiently cover the space of trajectories in order to obtain generalization). This is formalized below, which shows that we can use any broad-support policy to sample training trajectories and still obtain the correct flows and generative model, i.e., training can be off-policy.

**Proposition 3.** Let trajectories  $\tau$  used to train  $F_{\theta}$  be sampled from an exploratory policy P with the same support as the optimal  $\pi$  defined in Eq. 5 for a consistent flow  $F^* \in \mathcal{F}^*$ . A flow is consistent if Eq. 4 is respected. Also assume that  $\exists \theta: F_{\theta} = F^*$ , i.e., we choose a sufficiently rich family of predictors. Let  $\theta^* \in \operatorname{argmin}_{\theta} E_{P(\tau)}[L_{\theta}(\tau)]$  a minimizer of the expected training loss. Let  $L_{\theta}(\tau)$  have the property that when flows are matched it achieves its lowest possible value. First, it can be shown that this property is satisfied for the loss in Eq. 12. Then

$$F_{\theta^*} = F^*, \text{ and } L_{\theta^*}(\tau) = 0 \quad \forall \tau \sim P(\theta),$$
 (13)

i.e., a global optimum of the expected loss provides the correct flows. If  $\pi_{\theta^*}(a|s) = \frac{F_{\theta^*}(s,a)}{\sum_{a' \in \mathcal{A}(s)} F_{\theta^*}(s,a')}$  then we also have

$$\pi_{\theta^*}(x) = \frac{R(x)}{Z}.\tag{14}$$

The proof is in Appendix A.1. Note that, in RL terms, this method is akin to asynchronous dynamic programming (Sutton and Barto, 2018, §4.5), which is an off-policy off-line method which converges provided every state is visited infinitely many times asymptotically.

### 3 Related Work

The objective of training a policy generating states with a probability proportional to rewards was presented by Buesing et al. (2019) but the proposed method only makes sense when there is a bijection between action sequences and states. In contrast, GFlowNet is applicable in the more general setting where many paths can lead to the same state. The objective to sample with probability proportional to a given unnormalized positive function is achieved by many MCMC methods (Grathwohl et al., 2021; Dai et al., 2020). However, when mixing between modes is challenging (e.g., in high-dimensional spaces with well-separated modes occupying a fraction of the total volume) convergence to the target distribution can be extremely slow. In contrast, GFlowNet is not iterative and amortizes the challenge of sampling from such modes through a training procedure which must be sufficiently exploratory.

This sampling problem comes up in molecule generation and has been studied in this context with numerous generative models (Shi et al., 2020; Jin et al., 2020; Luo et al., 2021), MCMC methods (Seff

et al., 2019; Xie et al., 2021), RL (Segler et al., 2017; Cao and Kipf, 2018; Popova et al., 2019; Gottipati et al., 2020; Angermueller et al., 2020) and evolutionary methods (Brown et al., 2004; Jensen, 2019; Swersky et al., 2020). Some of these methods rely on a given set of "positive examples" (high-reward) to train a generative model, thus not taking advantage of the "negative examples" and the continuous nature of the measurements (some examples should be generated more often than others). Others rely on the traditional return maximization objectives of RL, which tends to focus on one or a few dominant modes, as we find in our experiments. Beyond molecules, there are previous works generating data non-greedily through RL (Bachman and Precup, 2015) or energy-based GANs (Dai et al., 2017).

The objective that we formulate in (12) may remind the reader of the objective of control-as-inference's Soft Q-Learning (Haarnoja et al., 2017), with the difference that we include all the parents of a state in the in-flow, whereas Soft Q-Learning only uses the parent contained in the trajectory. Soft Q-Learning induces a different policy, as shown by Proposition 1, one where  $P(\tau) \propto R(\tau)$  rather than  $P(x) \propto R(x)$ . More generally, we only consider deterministic generative settings whereas RL is a more general framework for stochastic environments.

Literature at the intersection of network flow and deep learning is sparse, and is mostly concerned with solving maximum flow problems (Nazemi and Omidi, 2012; Chen and Zhang, 2020) or classification within existing flow networks (Rahul et al., 2017; Pektaş and Acarman, 2019). Finally, the idea of accounting for the search space being a DAG rather than a tree in MCTS, known as transpositions (Childs et al., 2008), also has some links with the proposed method.

# 4 Empirical Results

We first verify that GFlowNet works as advertised on an artificial domain small enough to compute the partition function exactly, and compare its abilities to recover modes compared to standard MCMC and RL methods, with its sampling distribution better matching the normalized reward. We find that GFlowNet (A) converges to  $\pi(x) \propto R(x)$ , (B) requires less samples to achieve some level of performance than MCMC and PPO methods and (C) recovers all the modes and does so faster than MCMC and PPO, both in terms of wall-time and number of states visited and queried. We then test GFlowNet on a large scale domain, which consists in generating small drug molecule graphs, with a reward that estimates their binding affinity to a target protein (see Appendix A.3). We find that GFlowNet finds higher reward and more diverse molecules faster than baselines.

#### 4.1 A (hyper-)grid domain

Consider an MDP where states are the cells of a n-dimensional hypercubic grid of side length H. The agent starts at coordinate  $x=(0,0,\ldots)$  and is only allowed to increase coordinate i with action  $a_i$  (up to H, upon which the episode terminates). A stop action indicates to terminate the trajectory. There are many action sequences that lead to the same coordinate, making this MDP a DAG. The reward for ending the trajectory in x is some R(x)>0. For MCMC methods, in order to have an ergodic chain, we allow the iteration to decrease coordinates as well, and there is no stop action.

We ran experiments with this reward function:

$$R(x) = R_0 + R_1 \prod_i \mathbb{I}(0.25 < |x_i/H - 0.5|) + R_2 \prod_i \mathbb{I}(0.3 < |x_i/H - 0.5| < 0.4)$$

with  $0 < R_0 \ll R_1 < R_2$ , pictured when n=2 on the right. For this choice of R, there are only interesting rewards near the corners of the grid, and there are exactly  $2^n$  modes. We set  $R_1=1/2,\ R_2=2$ . By varying  $R_0$  and setting it closer to 0, we make this problem artificially harder, creating a region of the state space which it is undesirable to explore. To measure the performance of a method, we measure the empirical L1 error  $\mathbb{E}[|p(x)-\pi(x)|]$ .  $p(x)=\frac{R(x)}{Z}$  is known in this domain, and  $\pi$  is estimated by repeated sampling and counting frequencies for each possible x. We also measure the number of modes with at least 1 visit as a function of the number of states visited.

We run the above experiment for  $R_0 \in \{10^{-1}, 10^{-2}, 10^{-3}\}$  with n=4, H=8. In Fig. 2 we see that GFlowNet is robust to  $R_0$  and obtains a low L1 error, while a Metropolis-Hastings-MCMC based method requires exponentially more samples than GFlowNet to achieve some level of L1 error. This is apparent in Fig. 2 (with a log-scale horizontal axis) by comparing the slope of progress of GFlowNet (beyond the initial stage) and that of the MCMC sampler. We also see that MCMC takes much longer to visit each mode *once* as  $R_0$  decreases, while GFlowNet is only slightly affected, with GFlowNet converging to some level of L1 error faster, as per hypothesis (B). This suggests that

GFlowNet is robust to the separation between modes (represented by R<sup>0</sup> being smaller) and thus recovers all the modes much faster than MCMC (again, noting the log-scale of the horizontal axis).

To compare to RL, we run PPO [\(Schulman et al.,](#page-12-12) [2017\)](#page-12-12). To discover all the modes in a reasonable time, we need to set the entropy maximization term much higher (0.5) than usual ( 1). We verify that PPO is not overly regularized by comparing it to a random agent. PPO finds all the modes faster than uniform sampling, but much more slowly than GFlowNet, and is also robust to the choice of R0. This and the previous result validates hypothesis (C). We also run SAC [\(Haarnoja et al.,](#page-11-11) [2018\)](#page-11-11), finding similar or worse results. We provide additional results and discussion in Appendix [A.6.](#page-19-0)

<span id="page-7-0"></span>Figure 2: Hypergrid domain. Changing the task difficulty R<sup>0</sup> to illustrate the advantage of GFlowNet over others. We see that as R<sup>0</sup> gets smaller, MCMC struggles to fit the distribution because it struggles to visit all the modes. PPO also struggles to find all the modes, and requires very large entropy regularization, but is robust to the choice of R0. We plot means over 10 runs for each setting.

### 4.2 Generating small molecules

Here our goal is to generate a diverse set of small molecules that have a high reward. We define a large-scale environment which allows an agent to sequentially generate molecules. This environment is challenging, with up to 10<sup>16</sup> states and between 100 and 2000 actions depending on the state.

We follow the framework of [Jin et al.](#page-11-6) [\(2020\)](#page-11-6) and generate molecules by parts using a predefined vocabulary of building blocks that can be joined together forming a *junction tree* (detailed in [A.3\)](#page-15-0). This is also known as fragment-based drug design [\(Kumar et al.,](#page-11-12) [2012;](#page-11-12) [Xie et al.,](#page-12-2) [2021\)](#page-12-2). Generating such a graph can be described as a sequence of additive edits: given a molecule and constraints of chemical validity, we choose an atom to attach a block to. The action space is thus the product of choosing where to attach a block and choosing which block to attach. There is an extra action to stop the editing sequence. This sequence of edits yields a DAG MDP, as there are multiple action sequences that lead to the same molecule graph, and no edge removal actions, which prevents cycles.

The reward is computed with a pretrained *proxy* model that predicts the binding energy of a molecule to a particular protein target (soluble epoxide hydrolase, sEH, see [A.3\)](#page-15-0). Although computing binding energy is computationally expensive, we can call this proxy cheaply. Note that for realistic drug design, we would need to consider many more quantities such as drug-likeness [\(Bickerton et al.,](#page-10-8) [2012\)](#page-10-8), toxicity, or synthesizability. Our goal here is not solve this problem, and our work situates itself within such a larger project. Instead, we want to show that given a proxy R in the space of molecules, we can quickly match its induced distribution π(x) ∝ R(x) and find many of its modes.

We parameterize the proxy with an MPNN [\(Gilmer et al.,](#page-11-13) [2017\)](#page-11-13) over the atom graph. Our flow predictor F<sup>θ</sup> is parameterized similarly to MARS [\(Xie et al.,](#page-12-2) [2021\)](#page-12-2), with an MPNN, but over the junction tree graph (the graph of blocks), which had better performance. For fairness, this architecture is used for both GFlowNet and the baselines. Complete details can be found in Appendix [A.4.](#page-16-0)

We pretrain the proxy with a semi-curated semi-random dataset of 300k molecules (see [A.4\)](#page-16-0) down to a test MSE of 0.6; molecules are scored according to the docking score [\(Trott and Olson,](#page-12-13) [2010\)](#page-12-13), renormalized so that most scores fall between 0 and 10 (to have R(x) > 0). We plot the dataset's reward distribution in Fig. [3.](#page-8-0) We train all generative models with up to 10<sup>6</sup> molecules. During training, sampling follows exploratory policy P(a|s) which is a mixture between π(a|s) (Eq. [5\)](#page-4-0), used with probability 0.95, and a uniform distribution over allowed actions with probability 0.05.

Experimental results In Fig. [3](#page-8-0) we show the empirical distribution of rewards in two settings; first when we train our model with R(x), then with R(x) β . If GFlowNet learns a reasonable policy π,

<span id="page-8-0"></span>Figure 3: Empirical density of rewards. We verify that GFlowNet is consistent by training it with  $R^{\beta}$ ,  $\beta=4$ , which has the hypothesized effect of shifting the density to the right.

Figure 4: The average reward of the top-k as a function of learning (averaged over 3 runs). Only unique hits are counted. Note the log scale. Our method finds more unique good molecules faster.

this should shift the distribution to the right. This is indeed what we observe. We compare GFlowNet to MARS (Xie et al., 2021), known to work well in the molecule domain, and observe the same shift. Note that GFlowNet finds more high reward molecules than MARS with these  $\beta$  values; this is consistent with the hypothesis that it finds high-reward modes faster (since MARS is an MCMC method, it would eventually converge to the same distribution, but takes more time).

In Fig. 4, we show the average reward of the top-k molecules found so far, without allowing for duplicates (based on SMILES). We compare GFlowNet with MARS, PPO, and JT-VAE (Jin et al., 2020) with Bayesian Optimization. As expected, PPO plateaus after a while; RL tends to be satisfied with good enough trajectories unless it is strongly regularized with exploration mechanisms. For GFlowNet and for MARS, the more molecules are visited, the better they become, with a slow convergence towards the proxy's max reward. Given the same compute time, JT-VAE+BO generates only about  $10^3$  molecules (due to its expensive Gaussian Process) and so does not perform well.

The maximum reward in the proxy's dataset is 10, with only 233 examples above 8. In our best run, we find 2339 unique molecules during training with a score above 8, only 39 of which are in the dataset. We compute the average pairwise Tanimoto similarity for the top 1000 samples: GFlowNet has a mean of  $0.44 \pm 0.01$ , PPO,  $0.62 \pm 0.03$ , and MARS,  $0.59 \pm 0.02$  (mean and std over 3 runs). As expected, our MCMC baseline (MARS) and RL baseline (PPO) find less diverse candidates. We also find that GFlowNet discovers **many** more modes (>1500 with R > 8 vs < 100 for MARS). This is shown in Fig. 5 where we consider a mode to be a Bemis-Murcko scaffold (Bemis and Murcko, 1996), counted for molecules above a certain reward threshold. We provide additional insights into how GFlowNet matches the rewards in Appendix A.7.

<span id="page-8-1"></span>Figure 5: Number of diverse Bemis-Murcko scaffolds found above reward threshold T as a function of the number of molecules seen. Left, T=7.5. Right, T=8.

### 4.3 Multi-Round Experiments

To demonstrate the importance of diverse candidate generation in an active learning setting, we consider a sequential acquisition task. We simulate the setting where there is a limited budget for calls to the true oracle O. We use a proxy M initialized by training on a limited dataset of (x, R(x))

pairs  $D_0$ , where R(x) is the true reward from the oracle. The generative model  $(\pi_\theta)$  is trained to fit to the unnormalized probability function learned by the proxy M. We then sample a batch  $B = \{x_1, x_2, \dots x_k\}$  where  $x_i \sim \pi_\theta$ , which is evaluated with the oracle O. The proxy M is updated with this newly acquired and labeled batch, and the process is repeated for N iterations. We discuss the experimental setting in more detail in Appendix A.5.

<span id="page-9-0"></span>Figure 6: The top-k return (mean over 3 runs) in the 4-D Hyper-grid task with active learning. GFlowNet gets the highest return faster.

Figure 7: The top-k docking reward (mean over 3 runs) in the molecule task with active learning. GFlowNet consistently generates better samples.

Hyper-grid domain We present results for the multi-round task in the 4-D hyper-grid domain in Figure 6. We use a Gaussian Process (Williams and Rasmussen, 1995) as the proxy. We compare the *Top-k Return* for all the methods, which is defined as mean( $top-k(D_i)$ ) — mean( $top-k(D_{i-1})$ ), where  $D_i$  is the dataset of points acquired until step i, and k=10 for this experiment. The initial dataset  $D_0$  ( $|D_0|=512$ ) is the same for all the methods compared. We observe that GFlowNet consistently outperforms the baselines in terms of return over the initial set. We also observe that the mean pairwise L2-distance between the top-k points at the end of the final round is  $0.83\pm0.03$ ,  $0.61\pm0.01$  and  $0.51\pm0.02$  for GFlowNet, MCMC and PPO respectively. This demonstrates the ability of GFlowNet to capture the modes, even in the absence of the true oracle, as well as the importance of capturing this diversity in multi-round settings.

Small Molecules For the molecule discovery task, we initialize an MPNN proxy to predict docking scores from AutoDock (Trott and Olson, 2010), with  $|D_0|=2000$  molecules. At the end of each round we generate 200 molecules which are evaluated with AutoDock and used to update the proxy. Figure 7 shows GFlowNet discovers molecules with significantly higher energies than the initial set  $D_0$ . It also consistently outperforms MARS as well as Random Acquisition. PPO training was unstable and diverged consistently so the numbers are not reported. The mean pairwise Tanimoto similarity in the initial set is 0.60. At the end of the final round, it is  $0.54\pm0.04$  for GFlowNet and  $0.64\pm0.03$  for MARS. This further demonstrates the ability of GFlowNet to generate diverse candidates, which ultimately helps improve the final performance on the task. Similar to the single step setting, we observe that JT-VAE+BO is only able to generate  $10^3$  molecules with similar compute time, and thus performs poorly.

# 5 Discussion & Limitations

In this paper we have introduced a novel TD-like objective for learning a flow for each state and (*state, action*) pair such that policies sampling actions proportional to these flows draw terminal states in proportion to their reward. This can be seen as an alternative approach to turn an energy function into a fast generative model, without the need for an iterative method like that needed with MCMC methods, and with the advantage that when training succeeds, the policy generates a great diversity of samples near the main modes of the target distribution without being slowed by issues of mixing between modes.

**Limitations.** One downside of the proposed method is that, as for TD-based methods, the use of bootstrapping may cause optimization challenges (Kumar et al., 2020; Bengio et al., 2020) and limit its performance. In applications like drug discovery, sampling from the regions surrounding each mode is already an important advantage, but future work should investigate how to combine such a generative approach to local optimization in order to refine the generated samples and approach the local maxima of reward while keeping the batches of candidates diverse.

**Negative Social Impact.** The authors do not foresee negative social impacts of this work specifically.

# Acknowledgments and Disclosure of Funding

This research was enabled in part by computational resources provided by Calcul Québec ([www.](www.calculquebec.ca) [calculquebec.ca](www.calculquebec.ca)) and Compute Canada (<www.computecanada.ca>). All authors are funded by their primary academic institution. We also acknowledge funding from Samsung Electronics Co., Ldt., CIFAR and IBM.

The authors are grateful to Andrei Nica for generating the molecule dataset, to Maria Kadukova for advice on molecular docking, to Harsh Satija for feedback on the paper, as well as to all the members of the Mila Molecule Discovery team for the many research discussions on the challenges we faced.

# Author Contributions

EB and YB contributed to the original idea, and wrote most sections of the paper. YB wrote the proofs of Propositions 1-3, EB the proof of Proposition 4. EB wrote the code and ran experiments for sections 4.1 (hypergrid) and 4.2 (small molecules). MJ wrote the code and ran experiments for section 4.3 (multi-round) and wrote the corresponding results section of the paper. MK wrote the biochemical framework upon which the molecule experiments are built, assisted in debugging and running experiments for section 4.3, implemented mode-counting routines used in 4.2, and wrote the biochemical details of the paper.

MK, DP and YB provided supervision for the project. All authors contributed to proofreading and editing the paper.

# References

- <span id="page-10-0"></span>Christof Angermueller, David Dohan, David Belanger, Ramya Deshpande, Kevin Murphy, and Lucy Colwell. Model-based reinforcement learning for biological sequence design. In *International Conference on Learning Representations*, 2020.
- <span id="page-10-5"></span>Philip Bachman and Doina Precup. Data generation as sequential decision making. *Advances in Neural Information Processing Systems*, 28:3249–3257, 2015.
- <span id="page-10-9"></span>Guy W Bemis and Mark A Murcko. The properties of known drugs. 1. molecular frameworks. *Journal of medicinal chemistry*, 39(15):2887–2893, 1996.
- <span id="page-10-10"></span>Emmanuel Bengio, Joelle Pineau, and Doina Precup. Interference and generalization in temporal difference learning. In *International Conference on Machine Learning*, pages 767–777. PMLR, 2020.
- <span id="page-10-8"></span>G Richard Bickerton, Gaia V Paolini, Jérémy Besnard, Sorel Muresan, and Andrew L Hopkins. Quantifying the chemical beauty of drugs. *Nature chemistry*, 4(2):90–98, 2012.
- <span id="page-10-4"></span>Nathan Brown, Ben McKay, François Gilardoni, and Johann Gasteiger. A graph-based genetic algorithm and its application to the multiobjective evolution of median molecules. *Journal of chemical information and computer sciences*, 44(3):1079–1087, 2004.
- <span id="page-10-1"></span>Lars Buesing, Nicolas Heess, and Theophane Weber. Approximate inference in discrete distributions with monte carlo tree search and value functions, 2019.
- <span id="page-10-3"></span>Nicola De Cao and Thomas Kipf. Molgan: An implicit generative model for small molecular graphs, 2018.
- <span id="page-10-6"></span>Yize Chen and Baosen Zhang. Learning to solve network flow problems via neural decoding. *arXiv preprint arXiv:2002.04091*, 2020.
- <span id="page-10-7"></span>Benjamin E Childs, James H Brodeur, and Levente Kocsis. Transpositions and move groups in monte carlo tree search. In *2008 IEEE Symposium On Computational Intelligence and Games*, pages 389–395. IEEE, 2008.
- <span id="page-10-2"></span>Hanjun Dai, Rishabh Singh, Bo Dai, Charles Sutton, and Dale Schuurmans. Learning discrete energy-based models via auxiliary-variable local exploration. In *Neural Information Processing Systems (NeurIPS)*, 2020.

- <span id="page-11-9"></span>Zihang Dai, Amjad Almahairi, Philip Bachman, Eduard Hovy, and Aaron Courville. Calibrating energy-based generative adversarial networks. *arXiv preprint arXiv:1702.01691*, 2017.
- <span id="page-11-13"></span>Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural message passing for quantum chemistry, 2017.
- <span id="page-11-3"></span>Sai Krishna Gottipati, Boris Sattarov, Sufeng Niu, Yashaswi Pathak, Haoran Wei, Shengchao Liu, Karam M. J. Thomas, Simon Blackburn, Connor W. Coley, Jian Tang, Sarath Chandar, and Yoshua Bengio. Learning to navigate the synthetically accessible chemical space using reinforcement learning, 2020.
- <span id="page-11-2"></span>Will Grathwohl, Kevin Swersky, Milad Hashemi, David Duvenaud, and Chris J. Maddison. Oops i took a gradient: Scalable sampling for discrete distributions, 2021.
- <span id="page-11-4"></span>Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. In *International Conference on Machine Learning*, pages 1352–1361. PMLR, 2017.
- <span id="page-11-11"></span>Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor, 2018.
- <span id="page-11-8"></span>Jan H Jensen. A graph-based genetic algorithm and generative model/monte carlo tree search for the exploration of chemical space. *Chemical science*, 10(12):3567–3572, 2019.
- <span id="page-11-6"></span>Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Chapter 11. junction tree variational autoencoder for molecular graph generation. *Drug Discovery*, page 228–249, 2020. ISSN 2041-3211. doi: 10. 1039/9781788016841-00228. URL <http://dx.doi.org/10.1039/9781788016841-00228>.
- <span id="page-11-1"></span>Andreas Kirsch, Joost van Amersfoort, and Yarin Gal. Batchbald: Efficient and diverse batch acquisition for deep bayesian active learning, 2019.
- <span id="page-11-12"></span>Ashutosh Kumar, A Voet, and KYJ Zhang. Fragment based drug design: from experimental to computational approaches. *Current medicinal chemistry*, 19(30):5128–5147, 2012.
- <span id="page-11-14"></span>Aviral Kumar, Rishabh Agarwal, Dibya Ghosh, and Sergey Levine. Implicit under-parameterization inhibits data-efficient deep reinforcement learning, 2020.
- <span id="page-11-16"></span>Greg Landrum. Rdkit: Open-source cheminformatics. URL <http://www.rdkit.org>.
- <span id="page-11-7"></span>Youzhi Luo, Keqiang Yan, and Shuiwang Ji. Graphdf: A discrete flow model for molecular graph generation, 2021.
- <span id="page-11-5"></span>Charlie Nash and Conor Durkan. Autoregressive energy machines, 2019.
- <span id="page-11-10"></span>Alireza Nazemi and Farahnaz Omidi. A capable neural network model for solving the maximum flow problem. *Journal of Computational and Applied Mathematics*, 236(14):3498–3513, 2012.
- <span id="page-11-0"></span>Diana M. Negoescu, Peter I. Frazier, and Warren B. Powell. The knowledge-gradient algorithm for sequencing experiments in drug discovery. 23(3):346–363, 2011. ISSN 1526-5528. doi: 10.1287/ijoc.1100.0417. URL <https://doi.org/10.1287/ijoc.1100.0417>.
- <span id="page-11-17"></span>Andrew Y Ng, Stuart J Russell, et al. Algorithms for inverse reinforcement learning. In *Icml*, volume 1, page 2, 2000.
- <span id="page-11-15"></span>Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, *Advances in Neural Information Processing Systems 32*, pages 8024–8035. Curran Associates, Inc., 2019. URL [http://papers.neurips.cc/paper/](http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf) [9015-pytorch-an-imperative-style-high-performance-deep-learning-library.](http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf) [pdf](http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf).

- <span id="page-12-11"></span>Abdurrahman Pekta¸s and Tankut Acarman. Deep learning to detect botnet via network flow summaries. *Neural Computing and Applications*, 31(11):8021–8033, 2019.
- <span id="page-12-8"></span>Mariya Popova, Mykhailo Shvets, Junier Oliva, and Olexandr Isayev. Molecularrnn: Generating realistic molecular graphs with optimized properties, 2019.
- <span id="page-12-10"></span>RK Rahul, T Anjali, Vijay Krishna Menon, and KP Soman. Deep learning for network flow analysis and malware classification. In *International Symposium on Security in Computing and Communication*, pages 226–235. Springer, 2017.
- <span id="page-12-1"></span>Danilo Jimenez Rezende and Shakir Mohamed. Variational inference with normalizing flows, 2016.
- <span id="page-12-12"></span>John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017.
- <span id="page-12-6"></span>Ari Seff, Wenda Zhou, Farhan Damani, Abigail Doyle, and Ryan P Adams. Discrete object generation with reversible inductive construction. *arXiv preprint arXiv:1907.08268*, 2019.
- <span id="page-12-7"></span>Marwin H. S. Segler, Thierry Kogej, Christian Tyrchan, and Mark P. Waller. Generating focussed molecule libraries for drug discovery with recurrent neural networks, 2017.
- <span id="page-12-5"></span>Chence Shi, Minkai Xu, Zhaocheng Zhu, Weinan Zhang, Ming Zhang, and Jian Tang. Graphaf: a flow-based autoregressive model for molecular graph generation, 2020.
- <span id="page-12-3"></span>Yunsheng Shi, Zhengjie Huang, Shikun Feng, Hui Zhong, Wenjin Wang, and Yu Sun. Masked label prediction: Unified message passing model for semi-supervised classification, 2021.
- <span id="page-12-0"></span>Niranjan Srinivas, Andreas Krause, S. Kakade, and M. Seeger. Gaussian process optimization in the bandit setting: No regret and experimental design. In *ICML*, 2010.
- <span id="page-12-15"></span>Teague Sterling and John J Irwin. Zinc 15–ligand discovery for everyone. *Journal of chemical information and modeling*, 55(11):2324–2337, 2015.
- <span id="page-12-4"></span>Richard S Sutton and Andrew G Barto. *Reinforcement learning: An introduction*. MIT press, 2018.
- <span id="page-12-9"></span>Kevin Swersky, Yulia Rubanova, David Dohan, and Kevin Murphy. Amortized bayesian optimization over discrete spaces. In *Conference on Uncertainty in Artificial Intelligence*, pages 769–778. PMLR, 2020.
- <span id="page-12-13"></span>Oleg Trott and Arthur J Olson. Autodock vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. *Journal of computational chemistry*, 31(2):455–461, 2010.
- <span id="page-12-14"></span>C. K. Williams and C. Rasmussen. Gaussian processes for regression. In *Neural Information Processing Systems (NeurIPS)*, 1995.
- <span id="page-12-2"></span>Yutong Xie, Chence Shi, Hao Zhou, Yuwei Yang, Weinan Zhang, Yong Yu, and Lei Li. {MARS}: Markov molecular sampling for multi-objective drug discovery. In *International Conference on Learning Representations*, 2021. URL <https://openreview.net/forum?id=kHSu4ebxFXY>.

# A Appendix

All our ML code uses the PyTorch (Paszke et al., 2019) library. We reimplement RL and other baselines. We use the AutoDock Vina (Trott and Olson, 2010) library for binding energy estimation and RDKit (Landrum) for chemistry routines.

Running all the molecule experiments presented in this paper takes an estimated 26 GPU days. We use a cluster with NVidia V100 GPUs. The grid experiments take an estimated 8 CPU days (for a single-core).

All implementations are available at https://github.com/bengioe/gflownet.

#### <span id="page-13-0"></span>A.1 Proofs

**Proposition 1.** Let  $C: \mathcal{A}^* \mapsto \mathcal{S}$  associate each allowed action sequence  $\vec{a} \in \mathcal{A}^*$  to a state  $s = C(\vec{a}) \in \mathcal{S}$ . Let  $\tilde{V}: \mathcal{S} \mapsto \mathbf{R}^+$  associate each state  $s \in \mathcal{S}$  to  $\tilde{V}(s) = \sum_{\vec{b} \in \mathcal{A}^*(s)} R(s + \vec{b}) > 0$ , where  $\mathcal{A}^*(s)$  is the set of allowed continuations from s and  $s + \vec{b}$  denotes the resulting state, i.e.,  $\tilde{V}(s)$  is the sum of the rewards of all the states reachable from s. Consider a policy  $\pi$  which starts from the state corresponding to the empty string  $s_0 = C(\emptyset)$  and chooses from state  $s \in \mathcal{S}$  an allowable action  $a \in \mathcal{A}(s)$  with probability  $\pi(a|s) = \frac{\tilde{V}(s+a)}{\sum_{b \in \mathcal{A}(s)} \tilde{V}(s+b)}$ . Denote  $\pi(\vec{a} = (a_1, \ldots, a_N)) = \prod_{i=1}^N \pi(a_i|C(a_1, \ldots, a_{i-1}))$  and  $\pi(s)$  with  $s \in \mathcal{S}$  the probability of visiting a state s with this policy. The following then obtains:

(a)  $\pi(s) = \sum_{\vec{a}_i:C(\vec{a}_i)=s} \pi(\vec{a}_i)$ .

(b) If C is bijective, then  $\pi(s) = \frac{\tilde{V}(s)}{\tilde{V}(s_0)}$  and as a special case for terminal states s,  $\pi(s) = \frac{R(s)}{\sum_{s \in \mathcal{X}} R(s)}$ .

(c) If C is non-injective and there are n(s) distinct action sequences  $\vec{a}_i$  s.t.  $C(\vec{a}_i) = s$ , then  $\pi(s) = \frac{n(s)R(s)}{\sum_{s' \in \mathcal{X}} n(s')R(s')}$ .

*Proof.* Since s can be reached (from  $s_0$ ) according to any of the action sequences  $\vec{a}_i$  such that  $C(\vec{a}_i) = s$  and they are mutually exclusive and cover all the possible ways of reaching s, the probability that  $\pi$  visits state s is simply  $\sum_{\vec{a}_i:C(\vec{a}_i)=s}\pi(\vec{a}_i)$ , i.e., we obtain (a). If C is bijective, it means that there is only one such action sequence  $\vec{a}=(a_1,\ldots,a_N)$  landing in state s, and the set of action sequences and states forms a tree rooted at  $s_0$ . hence by (a) we get that  $\pi(s)=\pi(\vec{a})$ . First note that because  $\tilde{V}(s)=\sum_{\vec{b}\in\mathcal{A}^*(s)}R(s+\vec{b})$ , i.e.,  $\tilde{V}(s)$  is the sum of the terminal rewards for all the leaves rooted at s, we have that  $\tilde{V}(s)=\sum_{b\in\mathcal{A}(s)}V(s+b)$ . Let us now

prove by induction that  $\pi(s) = \frac{\tilde{V}(s)}{\tilde{V}(s_0)}$ . It is true for  $s = s_0$  since  $\pi(s_0 = 1)$  (i.e., every trajectory includes  $s_0$ ). Assuming it is true for  $s' = C(a_1, \dots, a_{N-1})$ , consider  $s = C(a_1, \dots, a_N)$ :

$$\pi(s) = \pi(a_N|s')\pi(s') = \frac{\tilde{V}(s)}{\sum_{b \in \mathcal{A}(s')} \tilde{V}(s'+b)} \frac{\tilde{V}(s')}{\tilde{V}(s_0)}.$$

Using our above result that  $\tilde{V}(s) = \sum_{b \in \mathcal{A}(s)} \tilde{V}(s+b)$ , we thus obtain a cancellation of  $\tilde{V}(s')$  with  $\sum_{b \in \mathcal{A}(s')} \tilde{V}(s'+b)$  and obtain

<span id="page-13-1"></span>
$$\pi(s) = \frac{\tilde{V}(s)}{\tilde{V}(s_0)},\tag{15}$$

proving that the recursion holds. We already know from the definition of  $\tilde{V}$  that  $\tilde{V}(s_0) = \sum_{x \in \mathcal{X}} R(x)$ , so for the special case of x a terminal state,  $\tilde{V}(x) = R(x)$  and Eq. 15 becomes  $\pi(x) = \frac{R(x)}{\sum_{x' \in \mathcal{X}} R(x')}$ , which finishes to prove (b).

On the other hand, if C is non-injective, the set of paths forms a DAG, and not a tree. Let us transform the DAG into a tree by creating a new state-space (for the tree version) which is the action sequence itself. Note how the same original leaf node x is now repeated n(x) times in the tree (with leaves denoted by action sequences  $\vec{a}_i$ ) if there are n(x) action sequences leading to x in the DAG. With the same definition of  $\tilde{V}$  and  $\pi(a|s)$  but in the tree, we obtain all the results from (b) (which are applicable

because we have a tree), and in particular  $\pi(\vec{a}_i)$  under the tree is proportional to R(x') = R(x). Applying (a), we see that  $\pi(x) \propto n(x)R(x)$ , which proves (c).

**Proposition 3.** Let trajectories  $\tau$  used to train  $F_{\theta}$  be sampled from an exploratory policy P with the same support as the optimal  $\pi$  defined in Eq. 5 for a consistent flow  $F^* \in \mathcal{F}^*$ . A flow is consistent if Eq. 4 is respected. Also assume that  $\exists \theta : F_{\theta} = F^*$ , i.e., we choose a sufficiently rich family of predictors. Let  $\theta^* \in \operatorname{argmin}_{\theta} E_{P(\tau)}[L_{\theta}(\tau)]$  a minimizer of the expected training loss. Let  $L_{\theta}(\tau)$  have the property that when flows are matched it achieves its lowest possible value. First, it can be shown that this property is satisfied for the loss in Eq. 12. Then

$$F_{\theta^*} = F^*, \quad \text{and} \tag{16}$$

$$L_{\theta^*}(\tau) = 0 \quad \forall \tau \sim P(\theta), \tag{17}$$

i.e., a global optimum of the expected loss provides the correct flows. If

$$\pi_{\theta^*}(a|s) = \frac{F_{\theta^*}(s, a)}{\sum_{a' \in \mathcal{A}(s)} F_{\theta^*}(s, a')}$$
(18)

then we also have

$$\pi_{\theta^*}(x) = \frac{R(x)}{Z}.\tag{19}$$

*Proof.* A per-trajectory loss of 0 can be achieved by choosing a  $\theta$  such that  $F_{\theta} = F^*$  (which we assumed was possible), since this makes the incoming flow equal the outgoing flow. Note that there always exists a solution  $F^*$  in the space of allow possible flow functions which satisfies the flow equations (incoming = outgoing) by construction of flow networks with only a constraint on the flow in the terminal nodes (leaves). Since having  $L_{\theta}(\tau)$  equal to 0 for all  $\tau \sim P(\theta)$  makes the expected loss 0, and this is the lowest achievable value (since  $L_{\theta}(\tau) \geq 0 \ \forall \theta$ ), it means that such a  $\theta$  is a global minimizer of the expected loss, and we can denote it  $\theta^*$ . If we optimize F in function space, we can directly set to 0 the gradient of the loss with respect to F(s,a) separately, and find a solution.

Since we have chosen P with support large enough to include all the trajectories leading to a terminal state R(x)>0, it means that  $L_{\theta}(\tau)=0$  for all these trajectories and that  $F_{\theta}=F^*$  for all nodes on these trajectories. We can then apply Proposition 2 (since the flows match everywhere and we have defined the policy correspondingly, as per Eq. 5). We then obtain the conclusion by applying result (c) from Proposition 2.

Note that in the general case, an infinite number of solutions exist. Consider the case where two trajectories are possible, say  $s_0, a_1, s_A, a_2, s_T$  and  $s_0, a_3, s_B, a_4, s_T$ , and both lead to the same terminal state  $s_T$  with reward r. Then a valid solution solves the constrained system of equations  $F(s_A) + F(s_B) = r, F(s_A) > 0, F(s_B) > 0$ , and we see that there is an infinite number of solutions described by one parameter u where  $F(s_A) = u, F(s_B) = r - u$   $u \in [0, r]$ .

#### <span id="page-14-0"></span>A.2 Action-value function equivalence

Here we show that the flow F(s,a) that the proposed method learns can correspond to a "real" action-value function  $\hat{Q}^{\mu}(s,a)$  in an RL sense, for a policy  $\mu$ .

First note that this is in a way trivially true: in inverse RL (Ng et al., 2000) there typically exists an infinite number of solutions to defining  $\hat{R}$  from a policy  $\pi$  such that  $\pi = \arg \max_{\pi_i} V^{\pi_i}(s; \hat{R}) \ \forall s$ , where  $V^{\pi_i}(s; \hat{R})$  is the value function at s for reward function  $\mathbf{R}$ .

More interesting is the case where F(s, a; R) obtained from computing the flow corresponding to R is exactly equal to some  $Q^{\mu}(s, a; \hat{R})$  modulo a multiplicative factor f(s). What are  $\mu$  and  $\hat{R}$ ?

In the bijective case a simple answer exists.

<span id="page-14-1"></span>**Proposition 4.** Let  $\mu$  be the uniform policy such that  $\mu(a|s) = 1/|\mathcal{A}(s)|$ , let  $f(x) = \prod_{t=0}^{n} |\mathcal{A}(s_t)|$  when  $x \equiv (s_0, s_1, ..., s_n)$ , and let  $\hat{R}(x) = R(x)f(s_{n-1})$ , then  $Q^{\mu}(s, a; \hat{R}) = F(s, a; R)f(s)$ .

*Proof.* By definition of the action-value function in terms of the action-value at the next step and by definition of  $\mu$ :

<span id="page-15-1"></span>
$$Q^{\mu}(s, a; \hat{R}) = \hat{R}(s') + \frac{1}{|\mathcal{A}(s')|} \sum_{a' \in \mathcal{A}(s')} Q^{\mu}(s', a'; \hat{R})$$
 (20)

where s' = T(s, a), as the environment is deterministic and has a tree structure.

For some leaf s',  $Q^{\mu}(s, a; \hat{R}) = \hat{R}(s') = R(s')f(s)$ . Again for some leaf s', the flow is F(s, a; R) = R(s'). Thus  $Q^{\mu}(s, a; \hat{R}) = F(s, a; R)f(s)$ . Suppose (20) is true, then by induction for a non-leaf s':

$$Q^{\mu}(s, a; \hat{R}) = \hat{R}(s') + \frac{1}{|\mathcal{A}(s')|} \sum_{a' \in \mathcal{A}(s')} Q^{\mu}(s', a'; \hat{R})$$
 (21)

$$Q^{\mu}(s, a; \hat{R}) = 0 + \frac{1}{|\mathcal{A}(s')|} \sum_{a' \in \mathcal{A}(s')} F(s', a'; R) f(s')$$
 (22)

we know from Eq 4 that

$$F(s, a; R) = \sum_{a' \in \mathcal{A}(s')} F(s', a'; R)$$
(23)

and since  $f(s') = f(s)|\mathcal{A}(s')|$ , we have that:

$$Q^{\mu}(s, a; \hat{R}) = \frac{F(s, a; R)f(s')}{|\mathcal{A}(s')|}$$
(24)

$$= \frac{F(s, a; R)f(s)|\mathcal{A}(s')|}{|\mathcal{A}(s')|}$$
(25)

$$= F(s, a; R)f(s) \tag{26}$$

Thus we have shown that the flow in a bijective case corresponds to the action-value of the uniform policy. This result suggests that the policy evaluation of the uniform policy learns something non-trivial in the tree MDP case. Perhaps such a quantity could be used in other interesting ways.

In the non-injective case, since an infinite number of valid flows exists, it's not clear that such a simple equivalence always exists.

As a particular case, consider the flow F which assigns exactly 0 flow to edges that would induce multiple paths to any node. In other words, consider the flow which induces a tree, i.e. a bijection between action sequences and states, by disallowing flow between edges not in that bijection. By Proposition 4, we can recover some valid  $Q^{\mu}$ .

Since there is at least one flow for which this equivalence exists, we conjecture that more general mappings between flows and action-value functions exist.

**Conjecture** There exists f a function of n(s) the number of paths to s, A(s), and  $n_p(s) = |\{(p,a)|T(p,a)=s\}|$  the number of parents of s, such that  $f(s,n(s),n_p(s),A(s))Q^{\mu}(s,a;\hat{R}) = F(s,a;R)$  and  $\hat{R}(x) = R(x)f(x)$  for the uniform policy  $\mu$  and for some valid flow F(s,a;R).

# <span id="page-15-0"></span>A.3 Molecule domain details

We allow the agent to choose from a library of 72 predefined blocks. We duplicate blocks from the point of view of the agent to allow attaching to different symmetry groups of a given block. This yields a total of 105 actions per stem; stems are atoms where new blocks can be attached to. We choose the blocks via the process suggested by Jin et al. (2020) over the ZINC dataset (Sterling and Irwin, 2015). We allow the agent to generate up to 8 blocks.

The 72 block SMILES are Br, C, C#N, C1=CCCCC1, C1=CNC=CC1, C1CC1, C1CCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCCC1, C1CCC1, C1CCC1, C1CCC1, C1CCC1, C1CCC1, C1CCC1, C1CCC1, C1CCC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1CC1, C1C1, C1CC1, C1CC1, C1CC1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C1C1, C

O=[PH](O)O, O=[PH]([O-])O, O=[SH](=O)O, O=[SH](=O)[O-], O=c1[nH]cnc2[nH]cnc12, O=c1[nH]cnc2c1NCCN2, O=c1cc[nH]c(=O)[nH]1, O=c1nc2[nH]c3ccccc3nc-2c(=O)[nH]1, O=c1nccc[nH]1, S, c1cc[nH+]cc1, c1cc[nH]c1, c1ccc2[nH]ccc2c1, c1ccc2ccccc2c1, c1ccccc1, c1ccncc1, c1ccsc1, c1cn[nH]c1, c1cncnc1, c1cscn1, c1ncc2nc[nH]c2n1.

We illustrate these building blocks and their attachment points in Figure [8.](#page-16-1)

<span id="page-16-1"></span>Figure 8: The list of building blocks used in molecule design. The stem, the atom which connects the block to the rest of the molecule, is highlighted.

We compute the reward based on a proxy's prediction. This proxy is trained on a dataset of 300k randomly generated molecules, whose binding affinity with a target protein has been computed with AutoDock [\(Trott and Olson,](#page-12-13) [2010\)](#page-12-13). Since the binding affinity is an energy where lower is better, we takes its opposite and then renormalize it (subtract the mean, divide by the standard deviation) to obtain the reward.

We use the sEH protein and its 4JNC inhibitor. The soluble epoxide hydrolase, or sEH, is a well studied protein which plays a role in respiratory and heart disease, which makes it an interesting pharmacological target and benchmark for ML methods.

Note that we also experimented with other biologically relevant quantities, in particular logP (the n-octanol-water partition coefficient) and QED [\(Bickerton et al.,](#page-10-8) [2012\)](#page-10-8). Both were very easy to maximize with GFlowNet. For logP we quickly find molecules with a >20 logP, which at this point is biologically uninteresting (for reference, ibuprofen's logP is between 3.5 and 4). For QED, we also quickly find molecules with the maximum possible QED in our action space, which is 0.948 (in fact our top-1000 is >0.94 after 100k molecules seen). Since docking is a much harder oracle, we focused on it. Also note that we experimented with combining different scores multiplicatively (e.g. multiplying docking score by a renormalized QED and synthesizability), with some success. A more specific contribution in that regards is left to future work.

# <span id="page-16-0"></span>A.4 Molecule domain implementation details

For the proxy of the oracle, from which the reward is defined, we use an MPNN [\(Gilmer et al.,](#page-11-13) [2017\)](#page-11-13) that receives the atom graph as input. We compute the atom graph using RDKit. Each node in the graph has features including the one-hot vector of its atomic number, its hybridization type, its number of implicit hydrogens, and a binary indicator of it being an acceptor or a donor atom. The MPNN uses a GRU at each iteration as the graph convolution layer is applied iteratively for 12 steps, followed by a Set2Set operation to reduce the graph, followed by a 3-layer MLP. We use 64 hidden units in all of its parts, and LeakyReLU activations everywhere (except inside the GRU).

In the non-active-learning experiments, we train the proxy with a dataset of 300k molecules. To make the task interesting, we use 80% of molecules obtained from random trajectories, and 20% obtained from previous runs of RL agents. This extra 20% contains slightly higher scoring molecules that are varied enough to allow for an interesting challenge. See Fig. 3 for the reward distribution. The exact dataset is provided on our github repository for reproducibility.

For the flow predictor F we also use an MPNN, but it receives the block graph as input. This graph is a tree by construction. Each node in the graph is a learned embedding (each of the 105 blocks has its own embedding and each type of bond has an edge embedding). We again use a GRU over the convolution layer applied 10 times. For each stem of the graph (which represents an atom or block where the agent can attach a new block) we pass its corresponding embedding (the output of the 10 steps of graph convolution + GRU) into a 3-layer MLP to produce 105 logits representing the probability of attaching each block to this stem for MARS and PPO, or representing the flow F(s,a) for GFlowNet; since each block can have multiple stems, this MLP also receives the underlying atom within the block to which the stem corresponds. For the stop action, we perform a global mean pooling followed by a 3-layer MLP that outputs 1 logit for each flow prediction. We use 256 hidden units everywhere as well as LeakyReLU activations.

For further stability we found that multiplying the loss for terminal transitions by a factor  $\lambda_T > 1$  helped. Intuitively, doing so prioritizes correct predictions at the endpoints of the flow, which can then propagate through the rest of the network/state space via our bootstrapping objective. This is similar to using reward prediction as an auxiliary task in RL (Jaderberg et al., 2017).

Here is a summary of the flow model hyperparameters:

| Learning rate             | $5 \times 10^{-4}$      |                                       |  |
|---------------------------|-------------------------|---------------------------------------|--|
| Minibatch size            | 4                       | # of trajectories per SGD step        |  |
| Adam $\beta, \epsilon$    | $(0.9, 0.999), 10^{-8}$ |                                       |  |
| # hidden & # embed        | 256                     |                                       |  |
| # convolution steps       | 10                      |                                       |  |
| Loss $\epsilon$           | $2.5 \times 10^{-5}$    | $\epsilon$ in (12)                    |  |
| Reward $T$                | 8                       |                                       |  |
| Reward $\beta$            | 10                      | $\hat{R}(x) = (R(x)/T)^{\beta}$       |  |
| Random action probability | 0.05                    | exploratory factor                    |  |
| $\lambda_T$               | 10                      | leaf loss coefficient                 |  |
| $R_{min}$                 | 0.01                    | $R$ is clipped below $R_{min}$ , i.e. |  |
|                           |                         | $\hat{R}_{min} = (R_{min}/T)^{\beta}$ |  |

For MARS we use a learning rate of  $2.5\times10^{-4}$  and for PPO,  $1\times10^{-4}$ . For PPO we use an entropy regularization coefficient of  $10^{-6}$  and we set the reward  $\beta$  to 4 (higher did not help). For MARS we use the same algorithmic hyperparameters as those found in Xie et al. (2021). For JT-VAE, we use the code provided by Jin et al. (2020) as-is, only replacing the reward signal with ours.

# <span id="page-17-0"></span>**A.5** Multi-Round Experiments

Algorithm 1 defines the procedure to train the policy  $\pi_{\theta}$  and used in inner loop of the multi-round experiments in the hyper-grid and molecule domains. The effect of diverse generation becomes apparent in the multi-round setting. Since the proxy itself is trained based on the input samples proposed by the generative models (and scored by the oracle, e.g., using docking), if the generative model is not exploratory enough, the reward (defined by the proxy) would only give useful learning signals around the discovered modes. The oracle outcomes O(x) are scaled to be positive, and a

hyper-parameter  $\beta$  (a kind of inverse temperature) can be used to make the modes of the reward function more or less peaked.

```
Algorithm 1: Multi-Round Active Learning
```

```
Input: Initial dataset D_0 = \{x_i, y_i\}, i = 1, \dots, k; K for TopK evaluation; number of rounds
        (outer loop iterations) N; inverse temperature \beta
Result: A set TopK(D_N) of high valued x
Initialization:
Proxy M;
Generative policy \pi_{\theta};
Oracle O;
i = 1;
while i <= N do
    Fit M on dataset D_{i-1};
    Train \pi_{\theta} with unnormalized probability function r(x) = M(x)^{\beta} as target reward;
    Sample query batch B = \{x_1, \ldots, x_b\} with x_i \sim \pi_\theta;
    Evaluate batch B with O, \hat{D}_i = \{(x_1, O(x_1)), \dots, (x_b, O(x_b))\};
    Update dataset D_i = \hat{D_i} \cup D_{i-1};
    i = i + 1;
end
```

#### A.5.1 Hyper-grid

We use the Gaussian Process implementation from botorch<sup>2</sup> for the proxy. The query batch size of samples generated after each round is 16. The hyper-parameters for training the generative models are set to the best performing values from the single-round experiments.

The initial dataset only contains 4 of the modes. GFlowNet discovered 10 of the modes within 5 rounds, while MCMC discovered 10 within 10 rounds, whereas PPO managed to discover only 8 modes by the end (with  $R_0 = 10^{-1}$ ).

#### A.5.2 Molecules

The initial set  $D_0$  of 2000 molecules is sampled randomly from the 300k dataset. At each round, for the MPNN proxy retraining, we use a fixed validation set for determining the stopping criterion. This validation set of 3000 examples is also sampled randomly from the 300k dataset. We use fewer iterations when fitting the generative model, and the rest of the hyper-parameters are the same as in the single round setting.

|          | Reward after 1800 docking evaluations |                 |  |
|----------|---------------------------------------|-----------------|--|
| method   | top-10                                | top-100         |  |
| GFlowNet | $8.83 \pm 0.15$                       | $7.76 \pm 0.11$ |  |
| MARS     | $8.27 \pm 0.20$                       | $7.08 \pm 0.13$ |  |

Figure 9: (a) Highest reward molecule in  $D_0$  in the multi-round molecule experiments. (b) Highest Reward molecule generated by GFlowNet. (c)-(e) Samples from the top-10 molecules generated by GFlowNet.

<span id="page-18-1"></span><sup>&</sup>lt;sup>2</sup>http://botorch.org/

### <span id="page-19-0"></span>A.6 Hypergrid Experiments

Let's first look at what is learned by GFlowNet. What is the distribution of flows learned? First, in Figure 10 (Left), we can observe that the distribution learned,  $\pi_{\theta}(x)$ , matches almost perfectly  $p(x) \propto R(x)$  on a grid where n=2, H=8. In Figure 10 (Middle) we plot the visit distribution on all paths that lead to mode s=(6,6), starting at  $s_0=(0,0)$ . We see that it is fairly spread out, but not uniform: there seems to be some preference towards other corners, presumably due to early bias during learning as well as the position of the other modes. In Figure 10 (Right) we plot what the uniform distribution on paths from (0,0) to (6,6) would look like for reference. Note that our loss does not enforce any kind of distribution on flows, and a uniform flow is not necessarily desirable (investigating this could be interesting future work, perhaps some distributions of flows have better generalization properties).

<span id="page-19-1"></span>Figure 10: Grid with n=2, H=8. Left, the distribution  $\pi_{\theta}(x)$  learned on the grid matches p(x) almost perfectly; measured by sampling 30k points. Middle, the visit distribution on sampled paths leading to (6,6). Right, the uniform distribution on all paths leading to (6,6).

Note that we also ran Soft Actor Critic (Haarnoja et al., 2018) on this domain, but we were unable to find hyperparameters that pushed SAC to find all the modes for n=4, H=8; SAC would find at best 10 of the 16 modes even when strongly regularized (but not so much so that the policy trivially becomes the uniform policy). While we believe our implementation to be correct, we did not think it would be relevant to include these results in figures, as they are poor but not really surprising: as would be consistent with reward-maximization, SAC quickly finds a mode to latch onto, and concentrates all of its probability mass on that mode, which is the no-diversity failure mode of RL we are trying to avoid with GFlowNet.

Next let's look at the losses as a function of  $R_0$ , again in the n=4, H=8 setting. We separate the loss in two components, the leaf loss (loss for terminal transitions) and the inner flow loss (loss for non-terminals). In Figure 11 we see that as  $R_0$  decreases, both inner flow and leaf losses get larger. This is reasonable for two reasons: first, for e.g. with  $R_0=10^{-3}$ ,  $\log 10^{-3}$  is a larger magnitude number which is harder for DNNs to accurately output, and second, the terminal states for which  $\log 10^{-3}$  is the flow output are  $100\times$  rarer than in the  $R_0=10^{-1}$  case (because we are sampling states on-policy), thus a DNN is less inclined to correctly predict their value correctly. This incurs rare but large magnitude losses. Note that theses losses are nonetheless, small, in the order of  $10^{-3}$  or less, and at this point the distribution is largely fit and the model is simply converging.

**GFlowNet as an offline off-policy method** To demonstrate this feature of GFlowNet, we train it on a fixed dataset of trajectories and observe what the learned distribution is. For this experiment we use  $R(x) = 0.01 + \prod_i (\cos(50x_i) + 1) f_{\mathcal{N}}(5x_i)$ ,  $f_{\mathcal{N}}$  is the normal p.d.f., n=2 and H=30. We show results for two random datasets. First, in Figure 12 we show what is learned when the dataset is sampled from a uniform random policy, and second in Figure 13 when the dataset is created by sampling points uniformly on the grid and walking backwards to the root to generate trajectories. The first setting should be much harder than the second, and indeed the learned distribution matches p(x) much better when the dataset points are more uniform. Note that in both cases many points are left out intentionally as a generalization test.

<span id="page-20-1"></span>Figure 11: Losses during training for the "corners" reward function in the hypergrid, with n=4, H=8. Shaded regions are the min-max bounds.

<span id="page-20-2"></span>Figure 12: Grid with n=2, H=30. Left, the learned distribution  $\pi_{\theta}(x)$ . Middle, the true distribution. Right, the dataset distribution, here generated by executing a uniform random policy from  $s_0$ .

These results suggest that GFlowNet can easily be applied offline and off-policy. Note that we did not do hyperparameter search on these two plots, these are purely illustrative and we believe it is likely that better generalization can be achieved by tweaking hyperparameters.

# <span id="page-20-0"></span>A.7 GFlowNet results on the molecule domain

Here we present additional results to give insights on what is learned by our method, GFlowNet. Let's first examine the numerical results of Figure 4:

|             | Reward at $10^5$ samples                       |                 |                 |  |
|-------------|------------------------------------------------|-----------------|-----------------|--|
| method      | top-10                                         | top-100         | top-1000        |  |
| GFlowNet    | $8.36 \pm 0.01$                                | $8.21 \pm 0.03$ | $7.98 \pm 0.04$ |  |
| MARS        | $8.05 \pm 0.12$                                | $7.71 \pm 0.09$ | $7.13 \pm 0.19$ |  |
| PPO         | $8.06 \pm 0.26$                                | $7.87 \pm 0.29$ | $7.52 \pm 0.26$ |  |
|             | Reward at 10 <sup>6</sup> samples              |                 |                 |  |
| GFlowNet    | $8.45 \pm 0.03$                                | $8.34 \pm 0.02$ | $8.17 \pm 0.02$ |  |
| MARS        | $8.31 \pm 0.03$                                | $8.03 \pm 0.08$ | $7.64 \pm 0.16$ |  |
| PPO         | $8.25 \pm 0.12$                                | $8.08 \pm 0.12$ | $7.82 \pm 0.16$ |  |
|             | Reward for 10 <sup>6</sup> -equivalent compute |                 |                 |  |
| JT-VAE + BO | 6.03                                           | 5.86            | 5.31            |  |

These are means and standard deviations computed over 3 runs. We see that GFlowNet produces significantly better molecules. It also produces much more diverse ones: GFlowNet has a mean

<span id="page-21-0"></span>Figure 13: Grid with n=2, H=30. Left, the learned distribution  $\pi_{\theta}(x)$ . Middle, the true distribution. Right, the dataset distribution, here generated by sampling a point uniformly on the grid and sampling random parents until  $s_0$  is reached, thus generating a training trajectory in reverse.

pairwise Tanimoto similarity for its top-1000 molecules of  $0.44\pm0.01$ , PPO,  $0.62\pm0.03$ , and MARS,  $0.59\pm0.02$  (mean and std over 3 runs). A random agent for this environment would yield an average pairwise similarity of 0.231 (and very poor rewards).

<span id="page-21-1"></span>Figure 14: Number of Tanimoto-separated modes found above reward threshold T as a function of the number of molecules seen. See main text. Left, T = 7. Right, T = 8.

Figure 15: Number of diverse Bemis-Murcko scaffolds (Bemis and Murcko, 1996) found above reward threshold T as a function of the number of molecules seen. Left, T=7.5. Right, T=8.

We also see that GFlowNet produces much more diverse molecules by approximately counting the number of modes found within the high-reward molecules. Here, we define "modes" as molecules with an energy above some threshold T, at most similar to each other in Tanimoto space at threshold S. In other words, we consider having found a new mode representative when a new molecule has a

Tanimoto similarity smaller than S to every previously found mode's representative molecule. We choose a Tanimoto similarity S of 0.7 as a threshold, as it is commonly used in medicinal chemistry to find similar molecules, and a reward threshold of 7 or 8. We plot the results in Figure 14. We see that for R > 7, GFlowNet discovers many more modes than MARS or PPO, over 500, whereas MARS only discovers less than 100.

Another way to approximate the number of modes is to count the number of diverse Bemis-Murcko scaffolds present within molecules above a certain reward threshold. We plot these counts in Figure 5, where we again see that GFlowNet finds a greater number of modes.

Next, let's try to understand what is learned by GFlowNet. In a large scale domain without access to p(x), it is non-trivial to demonstrate that  $\pi_{\theta}(x)$  matches the desired distribution  $p(x) \propto R(x)$ . This is due to the many-paths problem: to compute the true  $p_{\theta}(x)$  we would need to sum the  $p_{\theta}(\tau)$  of all the trajectories that lead to x, of which there can be an extremely large number. Instead, we show various measures that suggest that the learned distribution is consistent with the hypothesis the  $\pi_{\theta}(x)$  matches  $p(x) \propto R(x)^{\beta}$  well enough.

<span id="page-22-0"></span>In Figure 16 we show how  $F_{\theta}$  partially learns to match R(x). In particular we plot the inflow of leaves (i.e. for leaves s' the  $\sum_{s,a:T(s,a)=s'}F(s,a)$ ) as versus the target score  $(R(x)^{\beta})$ .

Figure 16: Scatter of the score  $(R(x)^{\beta})$  vs the inflow of leaves (the predicted unnormalized probability). The two should match. We see that a log-log linear regression has a slope of 0.58 and a r of 0.69. The slope being less than 1 suggests that GFlowNet tends to underestimate high rewards (this is plausible since high rewards are visited less often due to their rarity), but nonetheless reasonably fits its data. Here  $\beta=10$ . We plot here the last 5k molecules generated by a run.

Another way to view that the learned probabilities are self-consistent is that the histograms of R(x)/Z and  $\hat{p}_{\theta}(x)/Z$  match, where we use the predicted  $Z = \sum_{a \in \mathcal{A}(s_0)} F(s_0, a)$ , and  $\hat{p}_{\theta}(x)$  is the inflow of the leaf x as above. We show this in Figure 17.

In terms of loss, it is interesting that our models behaves similarly to value prediction in deep RL, in the sense that the value loss never goes to 0. This is somewhat expected due to bootstrapping, and the size of the state space. Indeed, in our hypergrid experiments the loss does go to 0 as the model converges. We plot the loss separately for leaf transitions (where the inflow is trained to match the reward) and inner flow transitions (at visited states, where the inflow is trained to match the outflow) in Figure 18.

<span id="page-23-0"></span>Figure 17: Histogram of the predicted density vs histogram of reward. The two should match. We compute these with the last 10k molecules generated by a run. This plot again suggests that the model is underfitted. It thinks the low-reward molecules are less likely than they actually are, or vice-versa that the low-reward molecules are better than they actually are. This is consistent with the previous plot showing a lower-than-1 slope.

<span id="page-23-1"></span>Figure 18: Loss as a function of training for a typical run of GFlowNet on the molecule domain. The shaded regions represent the min-max over the interval. We note several phases: In the initial phase the scale of the predictions are off and the leaf loss is very high. As prediction scales adjust we observe the second phase where the flow becomes consistent and we observe a dip in the loss. Then, as the model starts discovering more interesting samples, the loss goes up, and then down as it starts to correctly fit the flow over a large variety of samples. The lack of convergence is expected due to the massive state space; this is akin to value-based methods in deep RL on domains such as Atari.
