---
key: sutton2018rl
title: 'Reinforcement Learning: An Introduction (2nd ed.)'
year: 2018
primary:
- rl
role:
- rl-theory
- root
status: context
reward_term: []
domain:
- ml
tags:
- sutton-barto
- textbook
- foundations
- policy-gradient
summary: Canonical RL reference; root anchor for the methods lineage.
---

# <span id="page-68-1"></span>Chapter 3

# Finite Markov Decision Processes

In this chapter we introduce the formal problem of finite Markov decision processes, or finite MDPs, which we try to solve in the rest of the book. This problem involves evaluative feedback, as in bandits, but also an associative aspect—choosing different actions in different situations. MDPs are a classical formalization of sequential decision making, where actions influence not just immediate rewards, but also subsequent situations, or states, and through those future rewards. Thus MDPs involve delayed reward and the need to trade off immediate and delayed reward. Whereas in bandit problems we estimated the value  $q_*(a)$  of each action a, in MDPs we estimate the value  $q_*(s,a)$  of each action a in each state s, or we estimate the value  $v_*(s)$  of each state given optimal action selections. These state-dependent quantities are essential to accurately assigning credit for long-term consequences to individual action selections.

MDPs are a mathematically idealized form of the reinforcement learning problem for which precise theoretical statements can be made. We introduce key elements of the problem's mathematical structure, such as returns, value functions, and Bellman equations. We try to convey the wide range of applications that can be formulated as finite MDPs. As in all of artificial intelligence, there is a tension between breadth of applicability and mathematical tractability. In this chapter we introduce this tension and discuss some of the trade-offs and challenges that it implies. Some ways in which reinforcement learning can be taken beyond MDPs are treated in Chapter 17.

### <span id="page-68-0"></span>3.1 The Agent–Environment Interface

MDPs are meant to be a straightforward framing of the problem of learning from interaction to achieve a goal. The learner and decision maker is called the *agent*. The thing it interacts with, comprising everything outside the agent, is called the *environment*. These interact continually, the agent selecting actions and the environment responding to

these actions and presenting new situations to the agent.[1](#page-69-0) The environment also gives rise to rewards, special numerical values that the agent seeks to maximize over time through its choice of actions.

Figure 3.1: The agent–environment interaction in a Markov decision process.

More specifically, the agent and environment interact at each of a sequence of discrete time steps, *t* = 0*,* 1*,* 2*,* 3*,...*. [2](#page-69-1) At each time step *t*, the agent receives some representation of the environment's *state*, *<sup>S</sup><sup>t</sup>* <sup>2</sup> <sup>S</sup>, and on that basis selects an *action*, *<sup>A</sup><sup>t</sup>* <sup>2</sup> <sup>A</sup>(*s*).[3](#page-69-2) One time step later, in part as a consequence of its action, the agent receives a numerical *reward*, *Rt*+1 2 R ⇢ R, and finds itself in a new state, *St*+1. [4](#page-69-3) The MDP and agent together thereby give rise to a sequence or *trajectory* that begins like this:

$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \dots$$
 (3.1)

In a *finite* MDP, the sets of states, actions, and rewards (S, A, and R) all have a finite number of elements. In this case, the random variables *R<sup>t</sup>* and *S<sup>t</sup>* have well defined discrete probability distributions dependent only on the preceding state and action. That is, for particular values of these random variables, *s*<sup>0</sup> 2 S and *r* 2 R, there is a probability of those values occurring at time *t*, given particular values of the preceding state and action:

<span id="page-69-4"></span>
$$p(s', r|s, a) \doteq \Pr\{S_t = s', R_t = r \mid S_{t-1} = s, A_{t-1} = a\},$$
 (3.2)

for all *s*<sup>0</sup> *, s* 2 S, *r* 2 R, and *a* 2 A(*s*). The function *p* defines the *dynamics* of the MDP. The dot over the equals sign in the equation reminds us that it is a definition (in this case of the function *p*) rather than a fact that follows from previous definitions. The dynamics function *p* : S ⇥ R ⇥ S ⇥ A ! [0*,* 1] is an ordinary deterministic function of four arguments. The '*|*' in the middle of it comes from the notation for conditional probability,

<span id="page-69-0"></span><sup>1</sup>We use the terms *agent*, *environment*, and *action* instead of the engineers' terms *controller*, *controlled system* (or *plant*), and *control signal* because they are meaningful to a wider audience.

<span id="page-69-1"></span><sup>2</sup>We restrict attention to discrete time to keep things as simple as possible, even though many of the ideas can be extended to the continuous-time case (e.g., see Bertsekas and Tsitsiklis, 1996; Doya, 1996).

<span id="page-69-2"></span><sup>3</sup>To simplify notation, we sometimes assume the special case in which the action set is the same in all states and write it simply as A.

<span id="page-69-3"></span><sup>4</sup>We use *Rt*+1 instead of *R<sup>t</sup>* to denote the reward due to *A<sup>t</sup>* because it emphasizes that the next reward and next state, *Rt*+1 and *St*+1, are jointly determined. Unfortunately, both conventions are widely used in the literature.

but here it just reminds us that p specifies a probability distribution for each choice of s and a, that is, that

<span id="page-70-0"></span>
$$\sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r | s, a) = 1, \text{ for all } s \in \mathcal{S}, a \in \mathcal{A}(s).$$

$$(3.3)$$

In a Markov decision process, the probabilities given by p completely characterize the environment's dynamics. That is, the probability of each possible value for  $S_t$  and  $R_t$  depends on the immediately preceding state and action,  $S_{t-1}$  and  $A_{t-1}$ , and, given them, not at all on earlier states and actions. This is best viewed as a restriction not on the decision process, but on the state. The state must include information about all aspects of the past agent—environment interaction that make a difference for the future. If it does, then the state is said to have the Markov property. We will assume the Markov property throughout this book, though starting in Part II we will consider approximation methods that do not rely on it, and in Chapter 17 we consider how a Markov state can be efficiently learned and constructed from non-Markov observations.

<span id="page-70-1"></span>From the four-argument dynamics function, p, one can compute anything else one might want to know about the environment, such as the *state-transition probabilities* (which we denote, with a slight abuse of notation, as a three-argument function  $p: S \times S \times A \rightarrow [0,1]$ ),

$$p(s'|s,a) \doteq \Pr\{S_t = s' \mid S_{t-1} = s, A_{t-1} = a\} = \sum_{r \in \mathcal{R}} p(s',r|s,a).$$
 (3.4)

We can also compute the expected rewards for state–action pairs as a two-argument function  $r: S \times A \to \mathbb{R}$ :

<span id="page-70-2"></span>
$$r(s,a) \doteq \mathbb{E}[R_t \mid S_{t-1} = s, A_{t-1} = a] = \sum_{r \in \mathcal{R}} r \sum_{s' \in \mathcal{S}} p(s', r \mid s, a),$$
 (3.5)

and the expected rewards for state–action–next-state triples as a three-argument function  $r: S \times A \times S \to \mathbb{R}$ ,

$$r(s, a, s') \doteq \mathbb{E}[R_t \mid S_{t-1} = s, A_{t-1} = a, S_t = s'] = \sum_{r \in \mathcal{R}} r \frac{p(s', r \mid s, a)}{p(s' \mid s, a)}.$$
 (3.6)

In this book, we usually use the four-argument p function (3.2), but each of these other notations are also occasionally convenient.

The MDP framework is abstract and flexible and can be applied to many different problems in many different ways. For example, the time steps need not refer to fixed intervals of real time; they can refer to arbitrary successive stages of decision making and acting. The actions can be low-level controls, such as the voltages applied to the motors of a robot arm, or high-level decisions, such as whether or not to have lunch or to go to graduate school. Similarly, the states can take a wide variety of forms. They can be completely determined by low-level sensations, such as direct sensor readings, or they can be more high-level and abstract, such as symbolic descriptions of objects in a room. Some of what makes up a state could be based on memory of past sensations or

even be entirely mental or subjective. For example, an agent could be in the state of not being sure where an object is, or of having just been surprised in some clearly defined sense. Similarly, some actions might be totally mental or computational. For example, some actions might control what an agent chooses to think about, or where it focuses its attention. In general, actions can be any decisions we want to learn how to make, and states can be anything we can know that might be useful in making them.

In particular, the boundary between agent and environment is typically not the same as the physical boundary of a robot's or an animal's body. Usually, the boundary is drawn closer to the agent than that. For example, the motors and mechanical linkages of a robot and its sensing hardware should usually be considered parts of the environment rather than parts of the agent. Similarly, if we apply the MDP framework to a person or animal, the muscles, skeleton, and sensory organs should be considered part of the environment. Rewards, too, presumably are computed inside the physical bodies of natural and artificial learning systems, but are considered external to the agent.

The general rule we follow is that anything that cannot be changed arbitrarily by the agent is considered to be outside of it and thus part of its environment. We do not assume that everything in the environment is unknown to the agent. For example, the agent often knows quite a bit about how its rewards are computed as a function of its actions and the states in which they are taken. But we always consider the reward computation to be external to the agent because it defines the task facing the agent and thus must be beyond its ability to change arbitrarily. In fact, in some cases the agent may know *everything* about how its environment works and still face a dicult reinforcement learning task, just as we may know exactly how a puzzle like Rubik's cube works, but still be unable to solve it. The agent–environment boundary represents the limit of the agent's *absolute control*, not of its knowledge.

The agent–environment boundary can be located at di↵erent places for di↵erent purposes. In a complicated robot, many di↵erent agents may be operating at once, each with its own boundary. For example, one agent may make high-level decisions which form part of the states faced by a lower-level agent that implements the high-level decisions. In practice, the agent–environment boundary is determined once one has selected particular states, actions, and rewards, and thus has identified a specific decision-making task of interest.

The MDP framework is a considerable abstraction of the problem of goal-directed learning from interaction. It proposes that whatever the details of the sensory, memory, and control apparatus, and whatever objective one is trying to achieve, any problem of learning goal-directed behavior can be reduced to three signals passing back and forth between an agent and its environment: one signal to represent the choices made by the agent (the actions), one signal to represent the basis on which the choices are made (the states), and one signal to define the agent's goal (the rewards). This framework may not be sucient to represent all decision-learning problems usefully, but it has proved to be widely useful and applicable.

Of course, the particular states and actions vary greatly from task to task, and how they are represented can strongly a↵ect performance. In reinforcement learning, as in other kinds of learning, such representational choices are at present more art than science. In this book we o↵er some advice and examples regarding good ways of representing states and actions, but our primary focus is on general principles for learning how to behave once the representations have been selected.

Example 3.1: Bioreactor Suppose reinforcement learning is being applied to determine moment-by-moment temperatures and stirring rates for a bioreactor (a large vat of nutrients and bacteria used to produce useful chemicals). The actions in such an application might be target temperatures and target stirring rates that are passed to lower-level control systems that, in turn, directly activate heating elements and motors to attain the targets. The states are likely to be thermocouple and other sensory readings, perhaps filtered and delayed, plus symbolic inputs representing the ingredients in the vat and the target chemical. The rewards might be moment-by-moment measures of the rate at which the useful chemical is produced by the bioreactor. Notice that here each state is a list, or vector, of sensor readings and symbolic inputs, and each action is a vector consisting of a target temperature and a stirring rate. It is typical of reinforcement learning tasks to have states and actions with such structured representations. Rewards, on the other hand, are always single numbers.

Example 3.2: Pick-and-Place Robot Consider using reinforcement learning to control the motion of a robot arm in a repetitive pick-and-place task. If we want to learn movements that are fast and smooth, the learning agent will have to control the motors directly and have low-latency information about the current positions and velocities of the mechanical linkages. The actions in this case might be the voltages applied to each motor at each joint, and the states might be the latest readings of joint angles and velocities. The reward might be +1 for each object successfully picked up and placed. To encourage smooth movements, on each time step a small, negative reward could be given as a function of the moment-to-moment jerkiness of the motion.

*Exercise 3.1* Devise three example tasks of your own that fit into the MDP framework, identifying for each its states, actions, and rewards. Make the three examples as *di*↵*erent* from each other as possible. The framework is abstract and flexible and can be applied in many di↵erent ways. Stretch its limits in some way in at least one of your examples. ⇤

*Exercise 3.2* Is the MDP framework adequate to usefully represent *all* goal-directed learning tasks? Can you think of any clear exceptions? ⇤

*Exercise 3.3* Consider the problem of driving. You could define the actions in terms of the accelerator, steering wheel, and brake, that is, where your body meets the machine. Or you could define them farther out—say, where the rubber meets the road, considering your actions to be tire torques. Or you could define them farther in—say, where your brain meets your body, the actions being muscle twitches to control your limbs. Or you could go to a really high level and say that your actions are your choices of *where* to drive. What is the right level, the right place to draw the line between agent and environment? On what basis is one location of the line to be preferred over another? Is there any fundamental reason for preferring one location over another, or is it a free choice? ⇤

### Example 3.3 Recycling Robot

<span id="page-73-0"></span>A mobile robot has the job of collecting empty soda cans in an office environment. It has sensors for detecting cans, and an arm and gripper that can pick them up and place them in an onboard bin; it runs on a rechargeable battery. The robot's control system has components for interpreting sensory information, for navigating, and for controlling the arm and gripper. High-level decisions about how to search for cans are made by a reinforcement learning agent based on the current charge level of the battery. To make a simple example, we assume that only two charge levels can be distinguished, comprising a small state set  $S = \{\text{high}, \text{low}\}$ . In each state, the agent can decide whether to (1) actively search for a can for a certain period of time, (2) remain stationary and wait for someone to bring it a can, or (3) head back to its home base to recharge its battery. When the energy level is high, recharging would always be foolish, so we do not include it in the action set for this state. The action sets are then  $A(\text{high}) = \{\text{search}, \text{wait}\}$  and  $A(\text{low}) = \{\text{search}, \text{wait}, \text{recharge}\}$ .

The rewards are zero most of the time, but become positive when the robot secures an empty can, or large and negative if the battery runs all the way down. The best way to find cans is to actively search for them, but this runs down the robot's battery, whereas waiting does not. Whenever the robot is searching, the possibility exists that its battery will become depleted. In this case the robot must shut down and wait to be rescued (producing a low reward). If the energy level is high, then a period of active search can always be completed without risk of depleting the battery. A period of searching that begins with a high energy level leaves the energy level high with probability  $\alpha$  and reduces it to low with probability  $1-\alpha$ . On the other hand, a period of searching undertaken when the energy level is low leaves it low with probability  $\beta$  and depletes the battery with probability  $1-\beta$ . In the latter case, the robot must be rescued, and the battery is then recharged back to high. Each can collected by the robot counts as a unit reward, whereas a reward of -3 results whenever the robot has to be rescued. Let  $r_{\text{search}}$  and  $r_{\text{wait}}$ , with  $r_{\text{search}} > r_{\text{wait}}$ , denote the expected number of cans the robot will collect (and hence the expected reward) while searching and while waiting respectively. Finally, suppose that no cans can be collected during a run home for recharging, and that no cans can be collected on a step in which the battery is depleted. This system is then a finite MDP, and we can write down the transition probabilities and the expected rewards, with dynamics as indicated in the table on the left:

Note that there is a row in the table for each possible combination of current state, s, action,  $a \in \mathcal{A}(s)$ , and next state, s'. Some transitions have zero probability of occurring, so no expected reward is specified for them. Shown on the right is another useful way of

summarizing the dynamics of a finite MDP, as a transition graph. There are two kinds of nodes: state nodes and action nodes. There is a state node for each possible state (a large open circle labeled by the name of the state), and an action node for each state—action pair (a small solid circle labeled by the name of the action and connected by a line to the state node). Starting in state s and taking action a moves you along the line from state node s to action node (s,a). Then the environment responds with a transition to the next state's node via one of the arrows leaving action node (s,a). Each arrow corresponds to a triple (s,s',a), where s' is the next state, and we label the arrow with the transition probability, p(s'|s,a), and the expected reward for that transition, r(s,a,s'). Note that the transition probabilities labeling the arrows leaving an action node always sum to 1.

Exercise 3.4 Give a table analogous to that in Example 3.3, but for p(s', r|s, a). It should have columns for s, a, s', r, and p(s', r|s, a), and a row for every 4-tuple for which p(s', r|s, a) > 0.

### 3.2 Goals and Rewards

In reinforcement learning, the purpose or goal of the agent is formalized in terms of a special signal, called the *reward*, passing from the environment to the agent. At each time step, the reward is a simple number,  $R_t \in \mathbb{R}$ . Informally, the agent's goal is to maximize the total amount of reward it receives. This means maximizing not immediate reward, but cumulative reward in the long run. We can clearly state this informal idea as the *reward hypothesis*:

That all of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward).

The use of a reward signal to formalize the idea of a goal is one of the most distinctive features of reinforcement learning.

Although formulating goals in terms of reward signals might at first appear limiting, in practice it has proved to be flexible and widely applicable. The best way to see this is to consider examples of how it has been, or could be, used. For example, to make a robot learn to walk, researchers have provided reward on each time step proportional to the robot's forward motion. In making a robot learn how to escape from a maze, the reward is often -1 for every time step that passes prior to escape; this encourages the agent to escape as quickly as possible. To make a robot learn to find and collect empty soda cans for recycling, one might give it a reward of zero most of the time, and then a reward of +1 for each can collected. One might also want to give the robot negative rewards when it bumps into things or when somebody yells at it. For an agent to learn to play checkers or chess, the natural rewards are +1 for winning, -1 for losing, and 0 for drawing and for all nonterminal positions.

You can see what is happening in all of these examples. The agent always learns to maximize its reward. If we want it to do something for us, we must provide rewards to it in such a way that in maximizing them the agent will also achieve our goals. It

is thus critical that the rewards we set up truly indicate what we want accomplished. In particular, the reward signal is not the place to impart to the agent prior knowledge about *how* to achieve what we want it to do.[5](#page-75-0) For example, a chess-playing agent should be rewarded only for actually winning, not for achieving subgoals such as taking its opponent's pieces or aining control of the center of the board. If achieving these sorts of subgoals were rewarded, then the agent might find a way to achieve them without achieving the real goal. For example, it might find a way to take the opponent's pieces even at the cost of losing the game. The reward signal is your way of communicating to the agent *what* you want achieved, not *how* you want it achieved.[6](#page-75-1)

### 3.3 Returns and Episodes

So far we have discussed informally the objective of learning. We have said that the agent's goal is to maximize the cumulative reward it receives in the long run. How might this be defined formally? If the sequence of rewards received after time step *t* is denoted *Rt*+1*, Rt*+2*, Rt*+3*,...*, then what precise aspect of this sequence do we wish to maximize? In general, we seek to maximize the *expected return*, where the return, denoted *Gt*, is defined as some specific function of the reward sequence. In the simplest case the return is the sum of the rewards:

<span id="page-75-3"></span>
$$G_t \doteq R_{t+1} + R_{t+2} + R_{t+3} + \dots + R_T,$$
 (3.7)

where *T* is a final time step. This approach makes sense in applications in which there is a natural notion of final time step, that is, when the agent–environment interaction breaks naturally into subsequences, which we call *episodes*, [7](#page-75-2) such as plays of a game, trips through a maze, or any sort of repeated interaction. Each episode ends in a special state called the *terminal state*, followed by a reset to a standard starting state or to a sample from a standard distribution of starting states. Even if you think of episodes as ending in di↵erent ways, such as winning and losing a game, the next episode begins independently of how the previous one ended. Thus the episodes can all be considered to end in the same terminal state, with di↵erent rewards for the di↵erent outcomes. Tasks with episodes of this kind are called *episodic tasks*. In episodic tasks we sometimes need to distinguish the set of all nonterminal states, denoted S, from the set of all states plus the terminal state, denoted S<sup>+</sup>. The time of termination, *T*, is a random variable that normally varies from episode to episode.

On the other hand, in many cases the agent–environment interaction does not break naturally into identifiable episodes, but goes on continually without limit. For example, this would be the natural way to formulate an on-going process-control task, or an application to a robot with a long life span. We call these *continuing tasks*. The return formulation [\(3.7\)](#page-75-3) is problematic for continuing tasks because the final time step would be *T* = 1, and the return, which is what we are trying to maximize, could easily be infinite.

<span id="page-75-0"></span><sup>5</sup>Better places for imparting this kind of prior knowledge are the initial policy or initial value function.

<span id="page-75-1"></span><sup>6</sup>[Section 17.4](#page-490-0) delves further into the issue of designing e↵ective reward signals.

<span id="page-75-2"></span><sup>7</sup>Episodes are sometimes called "trials" in the literature.

(For example, suppose the agent receives a reward of +1 at each time step.) Thus, in this book we usually use a definition of return that is slightly more complex conceptually but much simpler mathematically.

The additional concept that we need is that of *discounting*. According to this approach, the agent tries to select actions so that the sum of the discounted rewards it receives over the future is maximized. In particular, it chooses *A<sup>t</sup>* to maximize the expected *discounted return*:

<span id="page-76-0"></span>
$$G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1},$$
 (3.8)

where is a parameter, 0 1, called the *discount rate*.

The discount rate determines the present value of future rewards: a reward received *k* time steps in the future is worth only *<sup>k</sup>*<sup>1</sup> times what it would be worth if it were received immediately. If *<* 1, the infinite sum in [\(3.8\)](#page-76-0) has a finite value as long as the reward sequence *{Rk}* is bounded. If = 0, the agent is "myopic" in being concerned only with maximizing immediate rewards: its objective in this case is to learn how to choose *A<sup>t</sup>* so as to maximize only *Rt*+1. If each of the agent's actions happened to influence only the immediate reward, not future rewards as well, then a myopic agent could maximize [\(3.8\)](#page-76-0) by separately maximizing each immediate reward. But in general, acting to maximize immediate reward can reduce access to future rewards so that the return is reduced. As approaches 1, the return objective takes future rewards into account more strongly; the agent becomes more farsighted.

Returns at successive time steps are related to each other in a way that is important for the theory and algorithms of reinforcement learning:

<span id="page-76-2"></span>
$$G_{t} \doteq R_{t+1} + \gamma R_{t+2} + \gamma^{2} R_{t+3} + \gamma^{3} R_{t+4} + \cdots$$

$$= R_{t+1} + \gamma \left( R_{t+2} + \gamma R_{t+3} + \gamma^{2} R_{t+4} + \cdots \right)$$

$$= R_{t+1} + \gamma G_{t+1}$$
(3.9)

Note that this works for all time steps *t<T*, even if termination occurs at *t* + 1, provided we define *G<sup>T</sup>* = 0. This often makes it easy to compute returns from reward sequences.

Note that although the return [\(3.8\)](#page-76-0) is a sum of an infinite number of terms, it is still finite if the reward is nonzero and constant—if *<* 1. For example, if the reward is a constant +1, then the return is

<span id="page-76-1"></span>
$$G_t = \sum_{k=0}^{\infty} \gamma^k = \frac{1}{1 - \gamma}.$$
 (3.10)

*Exercise 3.5* The equations in [Section 3.1](#page-68-0) are for the continuing case and need to be modified (very slightly) to apply to episodic tasks. Show that you know the modifications needed by giving the modified version of [\(3.3\)](#page-70-0). ⇤

### Example 3.4: Pole-Balancing

The objective in this task is to apply forces to a cart moving along a track so as to keep a pole hinged to the cart from falling over: A failure is said to occur if the pole falls past a given angle from vertical or if the cart runs off the track. The pole is reset to vertical after each failure. This task could be treated as episodic, where the natural

episodes are the repeated attempts to balance the pole. The reward in this case could be +1 for every time step on which failure did not occur, so that the return at each time would be the number of steps until failure. In this case, successful balancing forever would mean a return of infinity. Alternatively, we could treat pole-balancing as a continuing task, using discounting. In this case the reward would be -1 on each failure and zero at all other times. The return at each time would then be related to  $-\gamma^{K-1}$ , where K is the number of time steps before failure (as well as to the times of later failures). In either case, the return is maximized by keeping the pole balanced for as long as possible.

Exercise 3.6 Suppose you treated pole-balancing as an episodic task but also used discounting, with all rewards zero except for -1 upon failure. What then would the return be at each time? How does this return differ from that in the discounted, continuing formulation of this task?

Exercise 3.7 Imagine that you are designing a robot to run a maze. You decide to give it a reward of +1 for escaping from the maze and a reward of zero at all other times. The task seems to break down naturally into episodes—the successive runs through the maze—so you decide to treat it as an episodic task, where the goal is to maximize expected total reward (3.7). After running the learning agent for a while, you find that it is showing no improvement in escaping from the maze. What is going wrong? Have you effectively communicated to the agent what you want it to achieve?

Exercise 3.8 Suppose  $\gamma = 0.5$  and the following sequence of rewards is received  $R_1 = -1$ ,  $R_2 = 2$ ,  $R_3 = 6$ ,  $R_4 = 3$ , and  $R_5 = 2$ , with T = 5. What are  $G_0, G_1, \ldots, G_5$ ? Hint: Work backwards.

Exercise 3.9 Suppose  $\gamma = 0.9$  and the reward sequence is  $R_1 = 2$  followed by an infinite sequence of 7s. What are  $G_1$  and  $G_0$ ?

Exercise 3.10 Prove the second equality in (3.10).

### 3.4 Unified Notation for Episodic and Continuing Tasks

In the preceding section we described two kinds of reinforcement learning tasks, one in which the agent—environment interaction naturally breaks down into a sequence of separate episodes (episodic tasks), and one in which it does not (continuing tasks). The former case is mathematically easier because each action affects only the finite number of rewards subsequently received during the episode. In this book we consider sometimes one kind of problem and sometimes the other, but often both. It is therefore useful to establish one notation that enables us to talk precisely about both cases simultaneously.

To be precise about episodic tasks requires some additional notation. Rather than one long sequence of time steps, we need to consider a series of episodes, each of which consists of a finite sequence of time steps. We number the time steps of each episode starting anew from zero. Therefore, we have to refer not just to  $S_t$ , the state representation at time t, but to  $S_{t,i}$ , the state representation at time t of episode i (and similarly for  $A_{t,i}$ ,  $R_{t,i}$ ,  $\pi_{t,i}$ ,  $T_i$ , etc.). However, it turns out that when we discuss episodic tasks we almost never have to distinguish between different episodes. We are almost always considering a particular episode, or stating something that is true for all episodes. Accordingly, in practice we almost always abuse notation slightly by dropping the explicit reference to episode number. That is, we write  $S_t$  to refer to  $S_{t,i}$ , and so on.

We need one other convention to obtain a single notation that covers both episodic and continuing tasks. We have defined the return as a sum over a finite number of terms in one case (3.7) and as a sum over an infinite number of terms in the other (3.8). These two can be unified by considering episode termination to be the entering of a special absorbing state that transitions only to itself and that generates only rewards of zero. For example, consider the state transition diagram:

Here the solid square represents the special absorbing state corresponding to the end of an episode. Starting from  $S_0$ , we get the reward sequence  $+1, +1, +1, 0, 0, 0, \ldots$  Summing these, we get the same return whether we sum over the first T rewards (here T=3) or over the full infinite sequence. This remains true even if we introduce discounting. Thus, we can define the return, in general, according to (3.8), using the convention of omitting episode numbers when they are not needed, and including the possibility that  $\gamma = 1$  if the sum remains defined (e.g., because all episodes terminate). Alternatively, we can write

$$G_t \doteq \sum_{k=t+1}^T \gamma^{k-t-1} R_k, \tag{3.11}$$

including the possibility that  $T = \infty$  or  $\gamma = 1$  (but not both). We use these conventions throughout the rest of the book to simplify notation and to express the close parallels

between episodic and continuing tasks. (Later, in Chapter 10, we will introduce a formulation that is both continuing and undiscounted.)

### 3.5 Policies and Value Functions

Almost all reinforcement learning algorithms involve estimating value functions—functions of states (or of state—action pairs) that estimate how good it is for the agent to be in a given state (or how good it is to perform a given action in a given state). The notion of "how good" here is defined in terms of future rewards that can be expected, or, to be precise, in terms of expected return. Of course the rewards the agent can expect to receive in the future depend on what actions it will take. Accordingly, value functions are defined with respect to particular ways of acting, called policies.

Formally, a *policy* is a mapping from states to probabilities of selecting each possible action. If the agent is following policy  $\pi$  at time t, then  $\pi(a|s)$  is the probability that  $A_t = a$  if  $S_t = s$ . Like p,  $\pi$  is an ordinary function; the "|" in the middle of  $\pi(a|s)$  merely reminds us that it defines a probability distribution over  $a \in \mathcal{A}(s)$  for each  $s \in \mathcal{S}$ . Reinforcement learning methods specify how the agent's policy is changed as a result of its experience.

Exercise 3.11 If the current state is  $S_t$ , and actions are selected according to a stochastic policy  $\pi$ , then what is the expectation of  $R_{t+1}$  in terms of  $\pi$  and the four-argument function p (3.2)?

The value function of a state s under a policy  $\pi$ , denoted  $v_{\pi}(s)$ , is the expected return when starting in s and following  $\pi$  thereafter. For MDPs, we can define  $v_{\pi}$  formally by

$$v_{\pi}(s) \doteq \mathbb{E}_{\pi}[G_t \mid S_t = s] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s \right], \text{ for all } s \in \mathcal{S},$$
 (3.12)

where  $\mathbb{E}_{\pi}[\cdot]$  denotes the expected value of a random variable given that the agent follows policy  $\pi$ , and t is any time step. Note that the value of the terminal state, if any, is always zero. We call the function  $v_{\pi}$  the state-value function for policy  $\pi$ .

Similarly, we define the value of taking action a in state s under a policy  $\pi$ , denoted  $q_{\pi}(s, a)$ , as the expected return starting from s, taking the action a, and thereafter following policy  $\pi$ :

$$q_{\pi}(s,a) \doteq \mathbb{E}_{\pi}[G_t \mid S_t = s, A_t = a] = \mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a\right].$$
 (3.13)

We call  $q_{\pi}$  the action-value function for policy  $\pi$ .

Exercise 3.12 Give an equation for  $v_{\pi}$  in terms of  $q_{\pi}$  and  $\pi$ .

Exercise 3.13 Give an equation for  $q_{\pi}$  in terms of  $v_{\pi}$  and the four-argument p.

The value functions  $v_{\pi}$  and  $q_{\pi}$  can be estimated from experience. For example, if an agent follows policy  $\pi$  and maintains an average, for each state encountered, of the actual returns that have followed that state, then the average will converge to the state's value,  $v_{\pi}(s)$ , as the number of times that state is encountered approaches infinity. If separate

averages are kept for each action taken in each state, then these averages will similarly converge to the action values,  $q_{\pi}(s,a)$ . We call estimation methods of this kind *Monte Carlo methods* because they involve averaging over many random samples of actual returns. These kinds of methods are presented in Chapter 5. Of course, if there are very many states, then it may not be practical to keep separate averages for each state individually. Instead, the agent would have to maintain  $v_{\pi}$  and  $q_{\pi}$  as parameterized functions (with fewer parameters than states) and adjust the parameters to better match the observed returns. This can also produce accurate estimates, although much depends on the nature of the parameterized function approximator. These possibilities are discussed in Part II of the book.

A fundamental property of value functions used throughout reinforcement learning and dynamic programming is that they satisfy recursive relationships similar to that which we have already established for the return (3.9). For any policy  $\pi$  and any state s, the following consistency condition holds between the value of s and the value of its possible successor states:

$$v_{\pi}(s) \doteq \mathbb{E}_{\pi}[G_{t} \mid S_{t} = s]$$

$$= \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} \mid S_{t} = s]$$

$$= \sum_{a} \pi(a|s) \sum_{s'} \sum_{r} p(s', r|s, a) \Big[ r + \gamma \mathbb{E}_{\pi}[G_{t+1}|S_{t+1} = s'] \Big]$$

$$= \sum_{a} \pi(a|s) \sum_{s'} p(s', r|s, a) \Big[ r + \gamma v_{\pi}(s') \Big], \text{ for all } s \in \mathcal{S},$$
(3.14)

where it is implicit that the actions, a, are taken from the set  $\mathcal{A}(s)$ , that the next states, s', are taken from the set  $\mathcal{S}$  (or from  $\mathcal{S}^+$  in the case of an episodic problem), and that the rewards, r, are taken from the set  $\mathcal{R}$ . Note also how in the last equation we have merged the two sums, one over all the values of s' and the other over all the values of r, into one sum over all the possible values of both. We use this kind of merged sum often to simplify formulas. Note how the final expression can be read easily as an expected value. It is really a sum over all values of the three variables, a, s', and r. For each triple, we compute its probability,  $\pi(a|s)p(s',r|s,a)$ , weight the quantity in brackets by that probability, then sum over all possibilities to get an expected value.

Equation (3.14) is the Bellman equation for  $v_{\pi}$ . It expresses a relationship between the value of a state and the values of its successor states. Think of looking ahead from a state to its possible successor states, as suggested by the diagram to the right. Each open circle represents a state and each solid circle represents a state—action pair. Starting from state s, the root node at the top, the agent could take any of some set of actions—three are shown in the diagram—based on its policy  $\pi$ . From

<span id="page-80-0"></span>each of these, the environment could respond with one of several next states, s' (two are shown in the figure), along with a reward, r, depending on its dynamics given by the function p. The Bellman equation (3.14) averages over all the possibilities, weighting each by its probability of occurring. It states that the value of the start state must equal the (discounted) value of the expected next state, plus the reward expected along the way.

The value function  $v_{\pi}$  is the unique solution to its Bellman equation. We show in subsequent chapters how this Bellman equation forms the basis of a number of ways to compute, approximate, and learn  $v_{\pi}$ . We call diagrams like that above backup diagrams because they diagram relationships that form the basis of the update or backup operations that are at the heart of reinforcement learning methods. These operations transfer value information back to a state (or a state-action pair) from its successor states (or state-action pairs). We use backup diagrams throughout the book to provide graphical summaries of the algorithms we discuss. (Note that, unlike transition graphs, the state nodes of backup diagrams do not necessarily represent distinct states; for example, a state might be its own successor.)

<span id="page-81-1"></span>**Example 3.5:** Gridworld Figure 3.2 (left) shows a rectangular gridworld representation of a simple finite MDP. The cells of the grid correspond to the states of the environment. At each cell, four actions are possible: north, south, east, and west, which deterministically cause the agent to move one cell in the respective direction on the grid. Actions that would take the agent off the grid leave its location unchanged, but also result in a reward of -1. Other actions result in a reward of 0, except those that move the agent out of the special states A and B. From state A, all four actions yield a reward of +10 and take the agent to A'. From state B, all actions yield a reward of +5 and take the agent to B'.

<span id="page-81-0"></span>Figure 3.2: Gridworld example: exceptional reward dynamics (left) and state-value function for the equiprobable random policy (right).

Suppose the agent selects all four actions with equal probability in all states. Figure 3.2 (right) shows the value function,  $v_{\pi}$ , for this policy, for the discounted reward case with  $\gamma = 0.9$ . This value function was computed by solving the system of linear equations (3.14). Notice the negative values near the lower edge; these are the result of the high probability of hitting the edge of the grid there under the random policy. State A is the best state to be in under this policy. Note that A's expected return is less than its immediate reward of 10, because from A the agent is taken to state A' from which it is likely to run into the edge of the grid. State B, on the other hand, is valued more than its immediate reward of 5, because from B the agent is taken to B' which has a positive value. From B' the expected penalty (negative reward) for possibly running into an edge is more than compensated for by the expected gain for possibly stumbling onto A or B.

Exercise 3.14 The Bellman equation (3.14) must hold for each state for the value function  $v_{\pi}$  shown in Figure 3.2 (right) of Example 3.5. Show numerically that this equation holds for the center state, valued at +0.7, with respect to its four neighboring states, valued at +2.3, +0.4, -0.4, and +0.7. (These numbers are accurate only to one decimal place.)

Exercise 3.15 In the gridworld example, rewards are positive for goals, negative for running into the edge of the world, and zero the rest of the time. Are the signs of these rewards important, or only the intervals between them? Prove, using (3.8), that adding a constant c to all the rewards adds a constant,  $v_c$ , to the values of all states, and thus does not affect the relative values of any states under any policies. What is  $v_c$  in terms of c and  $\gamma$ ?

Exercise 3.16 Now consider adding a constant c to all the rewards in an episodic task, such as maze running. Would this have any effect, or would it leave the task unchanged as in the continuing task above? Why or why not? Give an example.

Example 3.6: Golf To formulate playing a hole of golf as a reinforcement learning task, we count a penalty (negative reward) of -1 for each stroke until we hit the ball into the hole. The state is the location of the ball. The value of a state is the negative of the number of strokes to the hole from that location. Our actions are how we aim and swing at the ball, of course, and which club we select. Let us take the former as given and consider just the choice of club, which we assume is either a putter or a driver. The upper part of Figure 3.3 shows a possible state-value function,  $v_{\rm putt}(s)$ , for the policy that

always uses the putter. The terminal state *in-the-hole* has a value of 0. From anywhere on the green we assume we can make a putt; these states have value -1. Off the green we cannot reach the hole by putting, and the value is lower. If we can reach the green from a state by putting, then that state must have value one less than the green's value, that is, -2. For simplicity, let us assume we can put very precisely and deterministically, but with a limited range. This gives us the sharp contour line labeled -2 in the figure; all locations between that line and the green require exactly two strokes to complete the hole. Similarly, any location within putting range of the -2 contour line must have a value of -3, and so on to get all the contour lines shown in the figure. Putting doesn't get us out of sand traps, so they have a value of  $-\infty$ . Overall, it takes us six strokes to get from the tee to the hole by putting.

<span id="page-82-0"></span>**Figure 3.3:** A golf example: the state-value function for putting (upper) and the optimal action-value function for using the driver (lower).

Exercise 3.17 What is the Bellman equation for action values, that is, for  $q_{\pi}$ ? It must give the action value  $q_{\pi}(s,a)$  in terms of the action values,  $q_{\pi}(s',a')$ , of possible successors to the state–action pair (s,a). Hint: The backup diagram to the right corresponds to this equation. Show the sequence of equations analogous to (3.14), but for action values.

<span id="page-83-0"></span>Exercise 3.18 The value of a state depends on the values of the actions possible in that state and on how likely each action is to be taken under the current policy. We can think of this in terms of a small backup diagram rooted at the state and considering each possible action:

Give the equation corresponding to this intuition and diagram for the value at the root node,  $v_{\pi}(s)$ , in terms of the value at the expected leaf node,  $q_{\pi}(s,a)$ , given  $S_t = s$ . This equation should include an expectation conditioned on following the policy,  $\pi$ . Then give a second equation in which the expected value is written out explicitly in terms of  $\pi(a|s)$  such that no expected value notation appears in the equation.

<span id="page-83-1"></span>Exercise 3.19 The value of an action,  $q_{\pi}(s, a)$ , depends on the expected next reward and the expected sum of the remaining rewards. Again we can think of this in terms of a small backup diagram, this one rooted at an action (state–action pair) and branching to the possible next states:

Give the equation corresponding to this intuition and diagram for the action value,  $q_{\pi}(s, a)$ , in terms of the expected next reward,  $R_{t+1}$ , and the expected next state value,  $v_{\pi}(S_{t+1})$ , given that  $S_t = s$  and  $A_t = a$ . This equation should include an expectation but not one conditioned on following the policy. Then give a second equation, writing out the expected value explicitly in terms of p(s', r|s, a) defined by (3.2), such that no expected value notation appears in the equation.

### 3.6 Optimal Policies and Optimal Value Functions

Solving a reinforcement learning task means, roughly, finding a policy that achieves a lot of reward over the long run. For finite MDPs, we can precisely define an optimal policy in the following way. Value functions define a partial ordering over policies. A policy  $\pi$  is defined to be better than or equal to a policy  $\pi'$  if its expected return is greater than or equal to that of  $\pi'$  for all states. In other words,  $\pi \geq \pi'$  if and only if  $v_{\pi}(s) \geq v_{\pi'}(s)$  for all  $s \in \mathcal{S}$ . There is always at least one policy that is better than or equal to all other policies. This is an *optimal policy*. Although there may be more than one, we denote all the optimal policies by  $\pi_*$ . They share the same state-value function, called the *optimal state-value function*, denoted  $v_*$ , and defined as

$$v_*(s) \doteq \max_{\pi} v_{\pi}(s), \tag{3.15}$$

for all  $s \in S$ .

Optimal policies also share the same optimal action-value function, denoted  $q_*$ , and defined as

$$q_*(s,a) \doteq \max_{\pi} q_{\pi}(s,a), \tag{3.16}$$

for all  $s \in S$  and  $a \in A(s)$ . For the state-action pair (s, a), this function gives the expected return for taking action a in state s and thereafter following an optimal policy. Thus, we can write  $q_*$  in terms of  $v_*$  as follows:

$$q_*(s,a) = \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a]. \tag{3.17}$$

Example 3.7: Optimal Value Functions for Golf The lower part of Figure 3.3 shows the contours of a possible optimal action-value function  $q_*(s, \mathtt{driver})$ . These are the values of each state if we first play a stroke with the driver and afterward select either the driver or the putter, whichever is better. The driver enables us to hit the ball farther, but with less accuracy. We can reach the hole in one shot using the driver only if we are already very close; thus the -1 contour for  $q_*(s,\mathtt{driver})$  covers only a small portion of the green. If we have two strokes, however, then we can reach the hole from much farther away, as shown by the -2 contour. In this case we don't have to drive all the way to within the small -1 contour, but only to anywhere on the green; from there we can use the putter. The optimal action-value function gives the values after committing to a particular first action, in this case, to the driver, but afterward using whichever actions are best. The -3 contour is still farther out and includes the starting tee. From the tee, the best sequence of actions is two drives and one putt, sinking the ball in three strokes.

Because  $v_*$  is the value function for a policy, it must satisfy the self-consistency condition given by the Bellman equation for state values (3.14). Because it is the optimal value function, however,  $v_*$ 's consistency condition can be written in a special form without reference to any specific policy. This is the Bellman equation for  $v_*$ , or the Bellman optimality equation. Intuitively, the Bellman optimality equation expresses the fact that the value of a state under an optimal policy must equal the expected return for the best action from that state:

$$v_{*}(s) = \max_{a \in \mathcal{A}(s)} q_{\pi_{*}}(s, a)$$

$$= \max_{a} \mathbb{E}_{\pi_{*}}[G_{t} \mid S_{t} = s, A_{t} = a]$$

$$= \max_{a} \mathbb{E}_{\pi_{*}}[R_{t+1} + \gamma G_{t+1} \mid S_{t} = s, A_{t} = a]$$

$$= \max_{a} \mathbb{E}[R_{t+1} + \gamma v_{*}(S_{t+1}) \mid S_{t} = s, A_{t} = a]$$
(by (3.9))
$$= \max_{a} \mathbb{E}[R_{t+1} + \gamma v_{*}(S_{t+1}) \mid S_{t} = s, A_{t} = a]$$
(3.18)

<span id="page-84-0"></span>
$$= \max_{a} \sum_{s',r} p(s',r|s,a) [r + \gamma v_*(s')]. \tag{3.19}$$

The last two equations are two forms of the Bellman optimality equation for  $v_*$ . The Bellman optimality equation for  $q_*$  is

<span id="page-85-0"></span>
$$q_{*}(s,a) = \mathbb{E}\left[R_{t+1} + \gamma \max_{a'} q_{*}(S_{t+1}, a') \mid S_{t} = s, A_{t} = a\right]$$

$$= \sum_{s', r} p(s', r \mid s, a) \left[r + \gamma \max_{a'} q_{*}(s', a')\right]. \tag{3.20}$$

The backup diagrams in the figure below show graphically the spans of future states and actions considered in the Bellman optimality equations for  $v_*$  and  $q_*$ . These are the same as the backup diagrams for  $v_{\pi}$  and  $q_{\pi}$  presented earlier except that arcs have been added at the agent's choice points to represent that the maximum over that choice is taken rather than the expected value given some policy. The backup diagram on the left graphically represents the Bellman optimality equation (3.19) and the backup diagram on the right graphically represents (3.20).

<span id="page-85-1"></span>**Figure 3.4:** Backup diagrams for  $v_*$  and  $q_*$

For finite MDPs, the Bellman optimality equation for  $v_*$  (3.19) has a unique solution. The Bellman optimality equation is actually a system of equations, one for each state, so if there are n states, then there are n equations in n unknowns. If the dynamics p of the environment are known, then in principle one can solve this system of equations for  $v_*$  using any one of a variety of methods for solving systems of nonlinear equations. One can solve a related set of equations for  $q_*$ .

Once one has  $v_*$ , it is relatively easy to determine an optimal policy. For each state s, there will be one or more actions at which the maximum is obtained in the Bellman optimality equation. Any policy that assigns nonzero probability only to these actions is an optimal policy. You can think of this as a one-step search. If you have the optimal value function,  $v_*$ , then the actions that appear best after a one-step search will be optimal actions. Another way of saying this is that any policy that is greedy with respect to the optimal evaluation function  $v_*$  is an optimal policy. The term greedy is used in computer science to describe any search or decision procedure that selects alternatives based only on local or immediate considerations, without considering the possibility that such a selection may prevent future access to even better alternatives. Consequently, it describes policies that select actions based only on their short-term consequences. The beauty of  $v_*$  is that if one uses it to evaluate the short-term consequences of actions—specifically, the one-step consequences—then a greedy policy is actually optimal in the long-term sense in which we are interested because  $v_*$  already takes into account the reward consequences of all possible future behavior. By means of  $v_*$ , the optimal expected long-term return is

turned into a quantity that is locally and immediately available for each state. Hence, a one-step-ahead search yields the long-term optimal actions.

Having *q*⇤ makes choosing optimal actions even easier. With *q*⇤, the agent does not even have to do a one-step-ahead search: for any state *s*, it can simply find any action that maximizes *q*⇤(*s, a*). The action-value function e↵ectively caches the results of all one-step-ahead searches. It provides the optimal expected long-term return as a value that is locally and immediately available for each state–action pair. Hence, at the cost of representing a function of state–action pairs, instead of just of states, the optimal actionvalue function allows optimal actions to be selected without having to know anything about possible successor states and their values, that is, without having to know anything about the environment's dynamics.

Example 3.8: Solving the Gridworld Suppose we solve the Bellman equation for *v*⇤ for the simple grid task introduced in [Example 3.5](#page-81-1) and shown again in [Figure 3.5](#page-86-0) (left). Recall that state A is followed by a reward of +10 and transition to state A0 , while state B is followed by a reward of +5 and transition to state B0 . [Figure 3.5](#page-86-0) (middle) shows the optimal value function, and [Figure 3.5](#page-86-0) (right) shows the corresponding optimal policies. Where there are multiple arrows in a cell, all of the corresponding actions are optimal.

<span id="page-86-0"></span>Figure 3.5: Optimal solutions to the gridworld example.

Example 3.9: Bellman Optimality Equations for the Recycling Robot Using [\(3.19\)](#page-84-0), we can explicitly give the Bellman optimality equation for the recycling robot example. To make things more compact, we abbreviate the states high and low, and the actions search, wait, and recharge respectively by h, l, s, w, and re. Because there are only two states, the Bellman optimality equation consists of two equations. The equation for *v*⇤(h) can be written as follows:

$$\begin{aligned} v_*(\mathbf{h}) &= & \max \left\{ \begin{array}{l} p(\mathbf{h} | \mathbf{h}, \mathbf{s})[r(\mathbf{h}, \mathbf{s}, \mathbf{h}) + \gamma v_*(\mathbf{h})] + p(\mathbf{1} | \mathbf{h}, \mathbf{s})[r(\mathbf{h}, \mathbf{s}, \mathbf{1}) + \gamma v_*(\mathbf{1})], \\ p(\mathbf{h} | \mathbf{h}, \mathbf{w})[r(\mathbf{h}, \mathbf{w}, \mathbf{h}) + \gamma v_*(\mathbf{h})] + p(\mathbf{1} | \mathbf{h}, \mathbf{w})[r(\mathbf{h}, \mathbf{w}, \mathbf{1}) + \gamma v_*(\mathbf{1})], \\ \end{array} \right\} \\ &= & \max \left\{ \begin{array}{l} \alpha[r_{\mathbf{s}} + \gamma v_*(\mathbf{h})] + (1 - \alpha)[r_{\mathbf{s}} + \gamma v_*(\mathbf{1})], \\ 1[r_{\mathbf{w}} + \gamma v_*(\mathbf{h})] + 0[r_{\mathbf{w}} + \gamma v_*(\mathbf{1})], \\ \end{array} \right\} \\ &= & \max \left\{ \begin{array}{l} r_{\mathbf{s}} + \gamma[\alpha v_*(\mathbf{h}) + (1 - \alpha)v_*(\mathbf{1})], \\ r_{\mathbf{w}} + \gamma v_*(\mathbf{h}) \end{array} \right\}. \end{aligned}$$

Following the same procedure for  $v_*(1)$  yields the equation

$$v_*(\mathbf{1}) = \max \left\{ \begin{array}{l} \beta r_{\mathbf{s}} - 3(1-\beta) + \gamma[(1-\beta)v_*(\mathbf{h}) + \beta v_*(\mathbf{1})], \\ r_{\mathbf{w}} + \gamma v_*(\mathbf{1}), \\ \gamma v_*(\mathbf{h}) \end{array} \right\}.$$

For any choice of  $r_s$ ,  $r_w$ ,  $\alpha$ ,  $\beta$ , and  $\gamma$ , with  $0 \le \gamma < 1$ ,  $0 \le \alpha, \beta \le 1$ , there is exactly one pair of numbers,  $v_*(h)$  and  $v_*(1)$ , that simultaneously satisfy these two nonlinear equations.

Explicitly solving the Bellman optimality equation provides one route to finding an optimal policy, and thus to solving the reinforcement learning problem. However, this solution is rarely directly useful. It is akin to an exhaustive search, looking ahead at all possibilities, computing their probabilities of occurrence and their desirabilities in terms of expected rewards. This solution relies on at least three assumptions that are rarely true in practice: (1) the dynamics of the environment are accurately known; (2) computational resources are sufficient to complete the calculation; and (3) the states have the Markov property. For the kinds of tasks in which we are interested, one is generally not able to implement this solution exactly because various combinations of these assumptions are violated. For example, although the first and third assumptions present no problems for the game of backgammon, the second is a major impediment. Because the game has about  $10^{20}$  states, it would take thousands of years on today's fastest computers to solve the Bellman equation for  $v_*$ , and the same is true for finding  $q_*$ . In reinforcement learning one typically has to settle for approximate solutions.

Many different decision-making methods can be viewed as ways of approximately solving the Bellman optimality equation. For example, heuristic search methods can be viewed as expanding the right-hand side of (3.19) several times, up to some depth, forming a "tree" of possibilities, and then using a heuristic evaluation function to approximate  $v_*$  at the "leaf" nodes. (Heuristic search methods such as  $A^*$  are almost always based on the episodic case.) The methods of dynamic programming can be related even more closely to the Bellman optimality equation. Many reinforcement learning methods can be clearly understood as approximately solving the Bellman optimality equation, using actual experienced transitions in place of knowledge of the expected transitions. We consider a variety of such methods in the following chapters.

Exercise 3.20 Draw or describe the optimal state-value function for the golf example.  $\Box$

Exercise 3.21 Draw or describe the contours of the optimal action-value function for putting,  $q_*(s, putter)$ , for the golf example.

Exercise 3.22 Consider the continuing MDP shown to the right. The only decision to be made is that in the top state, where two actions are available, left and right. The numbers show the rewards that are received deterministically after each action. There are exactly two deterministic policies,  $\pi_{\text{left}}$  and  $\pi_{\text{right}}$ . What policy is optimal if  $\gamma = 0$ ? If  $\gamma = 0.9$ ? If  $\gamma = 0.5$ ?

### 3.7 Optimality and Approximation

We have defined optimal value functions and optimal policies. Clearly, an agent that learns an optimal policy has done very well, but in practice this rarely happens. For the kinds of tasks in which we are interested, optimal policies can be generated only with extreme computational cost. A well-defined notion of optimality organizes the approach to learning we describe in this book and provides a way to understand the theoretical properties of various learning algorithms, but it is an ideal that agents can only approximate. As we discussed above, even if we have a complete and accurate model of the environment's dynamics, it is usually not possible to simply compute an optimal policy by solving the Bellman optimality equation. For example, board games such as chess are a tiny fraction of human experience, yet large, custom-designed computers still cannot compute the optimal moves. A critical aspect of the problem facing the agent is always the computational power available to it, in particular, the amount of computation it can perform in a single time step.

The memory available is also an important constraint. A large amount of memory is often required to build up approximations of value functions, policies, and models. In tasks with small, finite state sets, it is possible to form these approximations using arrays or tables with one entry for each state (or state-action pair). This we call the *tabular* case, and the corresponding methods we call tabular methods. In many cases of practical interest, however, there are far more states than could possibly be entries in a table. In these cases the functions must be approximated, using some sort of more compact parameterized function representation.

Our framing of the reinforcement learning problem forces us to settle for approximations. However, it also presents us with some unique opportunities for achieving useful approximations. For example, in approximating optimal behavior, there may be many states that the agent faces with such a low probability that selecting suboptimal actions for them has little impact on the amount of reward the agent receives. Tesauro's backgammon player, for example, plays with exceptional skill even though it might make

very bad decisions on board configurations that never occur in games against experts. In fact, it is possible that TD-Gammon makes bad decisions for a large fraction of the game's state set. The online nature of reinforcement learning makes it possible to approximate optimal policies in ways that put more e↵ort into learning to make good decisions for frequently encountered states, at the expense of less e↵ort for infrequently encountered states. This is one key property that distinguishes reinforcement learning from other approaches to approximately solving MDPs.

### 3.8 Summary

Let us summarize the elements of the reinforcement learning problem that we have presented in this chapter. Reinforcement learning is about learning from interaction how to behave in order to achieve a goal. The reinforcement learning *agent* and its *environment* interact over a sequence of discrete time steps. The specification of their interface defines a particular task: the *actions* are the choices made by the agent; the *states* are the basis for making the choices; and the *rewards* are the basis for evaluating the choices. Everything inside the agent is known and controllable. Its environment, on the other hand, is incompletely controllable and may or may not be completely known. A *policy* is a stochastic rule by which the agent selects actions as a function of states. The agent's objective is to maximize the amount of reward it receives over time.

When the reinforcement learning setup described above is formulated with well defined transition probabilities it constitutes a *Markov decision process* (MDP). A *finite MDP* is an MDP with finite state, action, and (as we formulate it here) reward sets. Much of the current theory of reinforcement learning is restricted to finite MDPs, but the methods and ideas apply more generally.

The *return* is the function of future rewards that the agent seeks to maximize (in expected value). It has several di↵erent definitions depending upon the nature of the task and whether one wishes to *discount* delayed reward. The undiscounted formulation is appropriate for *episodic tasks*, in which the agent–environment interaction breaks naturally into *episodes*; the discounted formulation is appropriate for tabular *continuing tasks*, in which the interaction does not naturally break into episodes but continues without limit (but see Sections [10.3–](#page-270-0)4). We try to define the returns for the two kinds of tasks such that one set of equations can apply to both the episodic and continuing cases.

A policy's *value functions* (*v*⇡ and *q*⇡) assign to each state, or state–action pair, the expected return from that state, or state–action pair, given that the agent uses the policy. The *optimal value functions* (*v*⇤ and *q*⇤) assign to each state, or state–action pair, the largest expected return achievable by any policy. A policy whose value functions are optimal is an *optimal policy*. Whereas the optimal value functions for states and state–action pairs are unique for a given MDP, there can be many optimal policies. Any policy that is *greedy* with respect to the optimal value functions must be an optimal policy. The *Bellman optimality equations* are special consistency conditions that the optimal value functions must satisfy and that can, in principle, be solved for the optimal value functions, from which an optimal policy can be determined with relative ease.

*3.8. Summary 69*

A reinforcement learning problem can be posed in a variety of di↵erent ways depending on assumptions about the level of knowledge initially available to the agent. In problems of *complete knowledge*, the agent has a complete and accurate model of the environment's dynamics. If the environment is an MDP, then such a model consists of the complete fourargument dynamics function *p* [\(3.2\)](#page-69-4). In problems of *incomplete knowledge*, a complete and perfect model of the environment is not available.

Even if the agent had a complete and accurate environment model, the agent would typically be unable to fully use it because of limitations on its memory and computation per time step. In particular, extensive memory may be required to build up accurate approximations of value functions, policies, and models. In most cases of practical interest there are far more states than could possibly be entries in a table, and approximations must be made.

A well-defined notion of optimality organizes the approach to learning we describe in this book and provides a way to understand the theoretical properties of various learning algorithms, but it is an ideal that reinforcement learning agents can only approximate to varying degrees. In reinforcement learning we are very much concerned with cases in which optimal solutions cannot be found but must be approximated in some way.

### Bibliographical and Historical Remarks

The reinforcement learning problem is deeply indebted to the idea of Markov decision processes (MDPs) from the field of optimal control. These historical influences and other major influences from psychology are described in the brief history given in [Chapter 1.](#page-22-0) Reinforcement learning adds to MDPs a focus on approximation and incomplete information for realistically large problems. MDPs and the reinforcement learning problem are only weakly linked to traditional learning and decision-making problems in artificial intelligence. However, artificial intelligence is now vigorously exploring MDP formulations for planning and decision making from a variety of perspectives. MDPs are more general than previous formulations used in artificial intelligence in that they permit more general kinds of goals and uncertainty.

The theory of MDPs is treated by, for example, Bertsekas (2005), White (1969), Whittle (1982, 1983), and Puterman (1994). A particularly compact treatment of the finite case is given by Ross (1983). MDPs are also studied under the heading of stochastic optimal control, where *adaptive* optimal control methods are most closely related to reinforcement learning (e.g., Kumar, 1985; Kumar and Varaiya, 1986).

The theory of MDPs evolved from e↵orts to understand the problem of making sequences of decisions under uncertainty, where each decision can depend on the previous decisions and their outcomes. It is sometimes called the theory of multistage decision processes, or sequential decision processes, and has roots in the statistical literature on sequential sampling beginning with the papers by Thompson (1933, 1934) and Robbins (1952) that we cited in [Chapter 2](#page-46-0) in connection with bandit problems (which are prototypical MDPs if formulated as multiple-situation problems).

The earliest instance (that we are aware of) in which reinforcement learning was discussed using the MDP formalism is Andreae's (1969) description of a unified view of learning machines. Witten and Corbin (1973) experimented with a reinforcement learning system later analyzed by Witten (1977, 1976a) using the MDP formalism. Although he did not explicitly mention MDPs, Werbos (1977) suggested approximate solution methods for stochastic optimal control problems that are related to modern reinforcement learning methods (see also Werbos, 1982, 1987, 1988, 1989, 1992). Although Werbos's ideas were not widely recognized at the time, they were prescient in emphasizing the importance of approximately solving optimal control problems in a variety of domains, including artificial intelligence. The most influential integration of reinforcement learning and MDPs is due to Watkins (1989).

3.1 Our characterization of the dynamics of an MDP in terms of *p*(*s*<sup>0</sup> *, r|s, a*) is slightly unusual. It is more common in the MDP literature to describe the dynamics in terms of the state transition probabilities *p*(*s*<sup>0</sup> *|s, a*) and expected next rewards *r*(*s, a*). In reinforcement learning, however, we more often have to refer to individual actual or sample rewards (rather than just their expected values). Our notation also makes it plainer that *S<sup>t</sup>* and *R<sup>t</sup>* are in general jointly determined, and thus must have the same time index. In teaching reinforcement learning, we have found our notation to be more straightforward conceptually and easier to understand.

For a good intuitive discussion of the system-theoretic concept of state, see Minsky (1967).

The bioreactor example is based on the work of Ungar (1990) and Miller and Williams (1992). The recycling robot example was inspired by the can-collecting robot built by Jonathan Connell (1989). Kober and Peters (2012) present a collection of robotics applications of reinforcement learning.

- 3.2 An explicit statement of the reward hypothesis was suggested by Michael Littman (personal communication).
- 3.3–4 The terminology of *episodic* and *continuing* tasks is di↵erent from that usually used in the MDP literature. In that literature it is common to distinguish three types of tasks: (1) finite-horizon tasks, in which interaction terminates after a particular *fixed* number of time steps; (2) indefinite-horizon tasks, in which interaction can last arbitrarily long but must eventually terminate; and (3) infinite-horizon tasks, in which interaction does not terminate. Our episodic and continuing tasks are similar to indefinite-horizon and infinite-horizon tasks, respectively, but we prefer to emphasize the di↵erence in the nature of the interaction. This di↵erence seems more fundamental than the di↵erence in the objective functions emphasized by the usual terms. Often episodic tasks use an indefinite-horizon objective function and continuing tasks an infinite-horizon objective function, but we see this as a common coincidence rather than a fundamental di↵erence.

The pole-balancing example is from Michie and Chambers (1968) and Barto, Sutton, and Anderson (1983).

*3.8. Summary 71*

3.5–6 Assigning value on the basis of what is good or bad in the long run has ancient roots. In control theory, mapping states to numerical values representing the long-term consequences of control decisions is a key part of optimal control theory, which was developed in the 1950s by extending nineteenth century state-function theories of classical mechanics (see, for example, Schultz and Melsa, 1967). In describing how a computer could be programmed to play chess, Shannon (1950) suggested using an evaluation function that took into account the long-term advantages and disadvantages of chess positions.

Watkins's (1989) Q-learning algorithm for estimating *q*⇤ [\(Chapter 6\)](#page-140-0) made actionvalue functions an important part of reinforcement learning, and consequently these functions are often called "Q-functions." But the idea of an action-value function is much older than this. Shannon (1950) suggested that a function *h*(*P,M*) could be used by a chess-playing program to decide whether a move *M* in position *P* is worth exploring. Michie's (1961, 1963) MENACE system and Michie and Chambers's (1968) BOXES system can be understood as estimating action-value functions. In classical physics, Hamilton's principal function is an action-value function; Newtonian dynamics are greedy with respect to this function (e.g., Goldstein, 1957). Action-value functions also played a central role in Denardo's (1967) theoretical treatment of dynamic programming in terms of contraction mappings.

The Bellman optimality equation (for *v*⇤) was popularized by Richard Bellman (1957a), who called it the "basic functional equation." The counterpart of the Bellman optimality equation for continuous time and state problems is known as the Hamilton–Jacobi–Bellman equation (or often just the Hamilton–Jacobi equation), indicating its roots in classical physics (e.g., Schultz and Melsa, 1967).

The golf example was suggested by Chris Watkins.

# <span id="page-140-0"></span>Chapter 6

# Temporal-Di↵erence Learning

If one had to identify one idea as central and novel to reinforcement learning, it would undoubtedly be *temporal-di*↵*erence* (TD) learning. TD learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas. Like Monte Carlo methods, TD methods can learn directly from raw experience without a model of the environment's dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap). The relationship between TD, DP, and Monte Carlo methods is a recurring theme in the theory of reinforcement learning; this chapter is the beginning of our exploration of it. Before we are done, we will see that these ideas and methods blend into each other and can be combined in many ways. In particular, in [Chapter 7](#page-162-0) we introduce *n*-step algorithms, which provide a bridge from TD to Monte Carlo methods, and in [Chapter 12](#page-308-0) we introduce the TD() algorithm, which seamlessly unifies them.

As usual, we start by focusing on the policy evaluation or *prediction* problem, the problem of estimating the value function *v*⇡ for a given policy ⇡. For the *control* problem (finding an optimal policy), DP, TD, and Monte Carlo methods all use some variation of generalized policy iteration (GPI). The di↵erences in the methods are primarily di↵erences in their approaches to the prediction problem.

### 6.1 TD Prediction

Both TD and Monte Carlo methods use experience to solve the prediction problem. Given some experience following a policy ⇡, both methods update their estimate *V* of *v*⇡ for the nonterminal states *S<sup>t</sup>* occurring in that experience. Roughly speaking, Monte Carlo methods wait until the return following the visit is known, then use that return as a target for *V* (*St*). A simple every-visit Monte Carlo method suitable for nonstationary environments is

<span id="page-140-1"></span>
$$V(S_t) \leftarrow V(S_t) + \alpha \Big[ G_t - V(S_t) \Big], \tag{6.1}$$

where  $G_t$  is the actual return following time t, and  $\alpha$  is a constant step-size parameter (c.f., Equation 2.4). Let us call this method constant- $\alpha$  MC. Whereas Monte Carlo methods must wait until the end of the episode to determine the increment to  $V(S_t)$  (only then is  $G_t$  known), TD methods need to wait only until the next time step. At time t+1 they immediately form a target and make a useful update using the observed reward  $R_{t+1}$  and the estimate  $V(S_{t+1})$ . The simplest TD method makes the update

<span id="page-141-2"></span>
$$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$

$$(6.2)$$

immediately on transition to  $S_{t+1}$  and receiving  $R_{t+1}$ . In effect, the target for the Monte Carlo update is  $G_t$ , whereas the target for the TD update is  $R_{t+1} + \gamma V(S_{t+1})$ . This TD method is called  $TD(\theta)$ , or one-step TD, because it is a special case of the  $TD(\lambda)$  and n-step TD methods developed in Chapter 12 and Chapter 7. The box below specifies TD(0) completely in procedural form.

```
Tabular TD(0) for estimating v_{\pi}
Input: the policy \pi to be evaluated
```

Algorithm parameter: step size  $\alpha \in (0, 1]$

Initialize V(s), for all  $s \in S^+$ , arbitrarily except that V(terminal) = 0

Loop for each episode:

Initialize S

Loop for each step of episode:

 $A \leftarrow$  action given by  $\pi$  for S

Take action A, observe R, S'

$$V(S) \leftarrow V(S) + \alpha \big[ R + \gamma V(S') - V(S) \big]$$

<span id="page-141-0"></span>0, 0

until S is terminal

Because TD(0) bases its update in part on an existing estimate, we say that it is a bootstrapping method, like DP. We know from Chapter 3 that

<span id="page-141-1"></span>
$$v_{\pi}(s) \doteq \mathbb{E}_{\pi}[G_t \mid S_t = s]$$

$$= \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} \mid S_t = s]$$

$$= \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s].$$
(6.3)
$$(6.4)$$

Roughly speaking, Monte Carlo methods use an estimate of (6.3) as a target, whereas DP methods use an estimate of (6.4) as a target. The Monte Carlo target is an estimate because the expected value in (6.3) is not known; a sample return is used in place of the real expected return. The DP target is an estimate not because of the expected values, which are assumed to be completely provided by a model of the environment, but because  $v_{\pi}(S_{t+1})$  is not known and the current estimate,  $V(S_{t+1})$ , is used instead. The TD target is an estimate for both reasons: it samples the expected values in (6.4) and it uses the current estimate V instead of the true  $v_{\pi}$ . Thus, TD methods combine the sampling of

*6.1. TD Prediction 121*

Monte Carlo with the bootstrapping of DP. As we shall see, with care and imagination this can take us a long way toward obtaining the advantages of both Monte Carlo and DP methods.

Shown to the right is the backup diagram for tabular TD(0). The value estimate for the state node at the top of the backup diagram is updated on the basis of the one sample transition from it to the immediately following state. We refer to TD and Monte Carlo updates as *sample updates* because they involve looking ahead to a sample successor state (or state–action pair), using the value of the successor and the reward along the way to compute a backed-up value, and then updating the value of the original state (or state– action pair) accordingly. *Sample* updates di↵er from the *expected* updates

of DP methods in that they are based on a single sample successor rather than on a complete distribution of all possible successors.

Finally, note that the quantity in brackets in the TD(0) update is a sort of error, measuring the di↵erence between the estimated value of *S<sup>t</sup>* and the better estimate *Rt*+1 + *V* (*St*+1). This quantity, called the *TD error*, arises in various forms throughout reinforcement learning:

<span id="page-142-1"></span>
$$\delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t). \tag{6.5}$$

Notice that the TD error at each time is the error in the estimate *made at that time*. Because the TD error depends on the next state and next reward, it is not actually available until one time step later. That is, *<sup>t</sup>* is the error in *V* (*St*), available at time *t* + 1. Also note that if the array *V* does not change during the episode (as it does not in Monte Carlo methods), then the Monte Carlo error can be written as a sum of TD errors:

<span id="page-142-0"></span>
$$G_{t} - V(S_{t}) = R_{t+1} + \gamma G_{t+1} - V(S_{t}) + \gamma V(S_{t+1}) - \gamma V(S_{t+1})$$
 (from (3.9))

$$= \delta_{t} + \gamma \left(G_{t+1} - V(S_{t+1})\right)$$


$$= \delta_{t} + \gamma \delta_{t+1} + \gamma^{2} \left(G_{t+2} - V(S_{t+2})\right)$$


$$= \delta_{t} + \gamma \delta_{t+1} + \gamma^{2} \delta_{t+2} + \dots + \gamma^{T-t-1} \delta_{T-1} + \gamma^{T-t} \left(G_{T} - V(S_{T})\right)$$


$$= \delta_{t} + \gamma \delta_{t+1} + \gamma^{2} \delta_{t+2} + \dots + \gamma^{T-t-1} \delta_{T-1} + \gamma^{T-t} \left(0 - 0\right)$$


$$= \sum_{k=t}^{T-1} \gamma^{k-t} \delta_{k}.$$
 (6.6)

This identity is not exact if *V* is updated during the episode (as it is in TD(0)), but if the step size is small then it may still hold approximately. Generalizations of this identity play an important role in the theory and algorithms of temporal-di↵erence learning.

*Exercise 6.1* If *V* changes during the episode, then [\(6.6\)](#page-142-0) only holds approximately; what would the di↵erence be between the two sides? Let *V<sup>t</sup>* denote the array of state values used at time *t* in the TD error [\(6.5\)](#page-142-1) and in the TD update [\(6.2\)](#page-141-2). Redo the derivation above to determine the additional amount that must be added to the sum of TD errors in order to equal the Monte Carlo error. ⇤

Example 6.1: Driving Home Each day as you drive home from work, you try to predict how long it will take to get home. When you leave your oce, you note the time, the day of week, the weather, and anything else that might be relevant. Say on this Friday you are leaving at exactly 6 o'clock, and you estimate that it will take 30 minutes to get home. As you reach your car it is 6:05, and you notice it is starting to rain. Trac is often slower in the rain, so you reestimate that it will take 35 minutes from then, or a total of 40 minutes. Fifteen minutes later you have completed the highway portion of your journey in good time. As you exit onto a secondary road you cut your estimate of total travel time to 35 minutes. Unfortunately, at this point you get stuck behind a slow truck, and the road is too narrow to pass. You end up having to follow the truck until you turn onto the side street where you live at 6:40. Three minutes later you are home. The sequence of states, times, and predictions is thus as follows:

|                                      | Elapsed<br>Time | Predicted        | Predicted     |
|--------------------------------------|-----------------|------------------|---------------|
| State                                | (minutes)       | Time<br>to<br>Go | Total<br>Time |
| leaving<br>oce,<br>friday<br>at<br>6 | 0               | 30               | 30            |
| reach<br>car,<br>raining             | 5               | 35               | 40            |
| exiting<br>highway                   | 20              | 15               | 35            |
| 2ndary<br>road,<br>behind<br>truck   | 30              | 10               | 40            |
| entering<br>home<br>street           | 40              | 3                | 43            |
| arrive<br>home                       | 43              | 0                | 43            |

The rewards in this example are the elapsed times on each leg of the journey.[1](#page-143-0) We are not discounting ( = 1), and thus the return for each state is the actual time to go from that state. The value of each state is the *expected* time to go. The second column of numbers gives the current estimated value for each state encountered.

A simple way to view the operation of Monte Carlo methods is to plot the predicted total time (the last column) over the sequence, as in [Figure 6.1](#page-144-0) (left). The red arrows show the changes in predictions recommended by the constant-↵ MC method [\(6.1\)](#page-140-1), for ↵ = 1. These are exactly the errors between the estimated value (predicted time to go) in each state and the actual return (actual time to go). For example, when you exited the highway you thought it would take only 15 minutes more to get home, but in fact it took 23 minutes. Equation [6.1](#page-140-1) applies at this point and determines an increment in the estimate of time to go after exiting the highway. The error, *G<sup>t</sup> V* (*St*), at this time is eight minutes. Suppose the step-size parameter, ↵, is 1*/*2. Then the predicted time to go after exiting the highway would be revised upward by four minutes as a result of this experience. This is probably too large a change in this case; the truck was probably just an unlucky break. In any event, the change can only be made o↵-line, that is, after you have reached home. Only at this point do you know any of the actual returns.

Is it necessary to wait until the final outcome is known before learning can begin? Suppose on another day you again estimate when leaving your oce that it will take 30 minutes to drive home, but then you become stuck in a massive trac jam. Twenty-five minutes after leaving the oce you are still bumper-to-bumper on the highway. You now

<span id="page-143-0"></span><sup>1</sup>If this were a control problem with the objective of minimizing travel time, then we would of course make the rewards the *negative* of the elapsed time. But because we are concerned here only with prediction (policy evaluation), we can keep things simple by using positive numbers.

<span id="page-144-0"></span>6.1. TD Prediction 123

Figure 6.1: Changes recommended in the driving home example by Monte Carlo methods (left) and TD methods (right).

estimate that it will take another 25 minutes to get home, for a total of 50 minutes. As you wait in traffic, you already know that your initial estimate of 30 minutes was too optimistic. Must you wait until you get home before increasing your estimate for the initial state? According to the Monte Carlo approach you must, because you don't yet know the true return.

According to a TD approach, on the other hand, you would learn immediately, shifting your initial estimate from 30 minutes toward 50. In fact, each estimate would be shifted toward the estimate that immediately follows it. Returning to our first day of driving, Figure 6.1 (right) shows the changes in the predictions recommended by the TD rule (6.2) (these are the changes made by the rule if  $\alpha = 1$ ). Each error is proportional to the change over time of the prediction, that is, to the temporal differences in predictions.

Besides giving you something to do while waiting in traffic, there are several computational reasons why it is advantageous to learn based on your current predictions rather than waiting until termination when you know the actual return. We briefly discuss some of these in the next section.

Exercise 6.2 This is an exercise to help develop your intuition about why TD methods are often more efficient than Monte Carlo methods. Consider the driving home example and how it is addressed by TD and Monte Carlo methods. Can you imagine a scenario in which a TD update would be better on average than a Monte Carlo update? Give an example scenario—a description of past experience and a current state—in which you would expect the TD update to be better. Here's a hint: Suppose you have lots of experience driving home from work. Then you move to a new building and a new parking lot (but you still enter the highway at the same place). Now you are starting to learn predictions for the new building. Can you see why TD updates are likely to be much better, at least initially, in this case? Might the same sort of thing happen in the original scenario?

### 6.2 Advantages of TD Prediction Methods

TD methods update their estimates based in part on other estimates. They learn a guess from a guess—they *bootstrap*. Is this a good thing to do? What advantages do TD methods have over Monte Carlo and DP methods? Developing and answering such questions will take the rest of this book and more. In this section we briefly anticipate some of the answers.

Obviously, TD methods have an advantage over DP methods in that they do not require a model of the environment, of its reward and next-state probability distributions.

The next most obvious advantage of TD methods over Monte Carlo methods is that they are naturally implemented in an online, fully incremental fashion. With Monte Carlo methods one must wait until the end of an episode, because only then is the return known, whereas with TD methods one need wait only one time step. Surprisingly often this turns out to be a critical consideration. Some applications have very long episodes, so that delaying all learning until the end of the episode is too slow. Other applications are continuing tasks and have no episodes at all. Finally, as we noted in the previous chapter, some Monte Carlo methods must ignore or discount episodes on which experimental actions are taken, which can greatly slow learning. TD methods are much less susceptible to these problems because they learn from each transition regardless of what subsequent actions are taken.

But are TD methods sound? Certainly it is convenient to learn one guess from the next, without waiting for an actual outcome, but can we still guarantee convergence to the correct answer? Happily, the answer is yes. For any fixed policy ⇡, TD(0) has been proved to converge to *v*⇡, in the mean for a constant step-size parameter if it is suciently small, and with probability 1 if the step-size parameter decreases according to the usual stochastic approximation conditions [\(2.7\)](#page-54-0). Most convergence proofs apply only to the table-based case of the algorithm presented above [\(6.2\)](#page-141-2), but some also apply to the case of general linear function approximation. These results are discussed in a more general setting in [Section 9.4.](#page-225-0)

If both TD and Monte Carlo methods converge asymptotically to the correct predictions, then a natural next question is "Which gets there first?" In other words, which method learns faster? Which makes the more ecient use of limited data? At the current time this is an open question in the sense that no one has been able to prove mathematically that one method converges faster than the other. In fact, it is not even clear what is the most appropriate formal way to phrase this question! In practice, however, TD methods have usually been found to converge faster than constant-↵ MC methods on stochastic tasks, as illustrated in [Example 6.2.](#page-146-0)

### Example 6.2 Random Walk

<span id="page-146-0"></span>In this example we empirically compare the prediction abilities of TD(0) and constant- $\alpha$  MC when applied to the following Markov reward process:

A Markov reward process, or MRP, is a Markov decision process without actions. We will often use MRPs when focusing on the prediction problem, in which there is no need to distinguish the dynamics due to the environment from those due to the agent. In this MRP, all episodes start in the center state, C, then proceed either left or right by one state on each step, with equal probability. Episodes terminate either on the extreme left or the extreme right. When an episode terminates on the right, a reward of +1 occurs; all other rewards are zero. For example, a typical episode might consist of the following state-and-reward sequence: C, 0, B, 0, C, 0, D, 0, E, 1. Because this task is undiscounted, the true value of each state is the probability of terminating on the right if starting from that state. Thus, the true value of the center state is  $v_{\pi}(\mathsf{C}) = 0.5$ . The true values of all the states, A through E, are  $\frac{1}{6}, \frac{2}{6}, \frac{3}{6}, \frac{4}{6}$ , and  $\frac{5}{6}$ .

The left graph above shows the values learned after various numbers of episodes on a single run of TD(0). The estimates after 100 episodes are about as close as they ever come to the true values—with a constant step-size parameter ( $\alpha = 0.1$  in this example), the values fluctuate indefinitely in response to the outcomes of the most recent episodes. The right graph shows learning curves for the two methods for various values of  $\alpha$ . The performance measure shown is the root mean square (RMS) error between the value function learned and the true value function, averaged over the five states, then averaged over 100 runs. In all cases the approximate value function was initialized to the intermediate value V(s) = 0.5, for all s. The TD method was consistently better than the MC method on this task.

### 6.3 Optimality of TD(0)

Suppose there is available only a finite amount of experience, say 10 episodes or 100 time steps. In this case, a common approach with incremental learning methods is to present the experience repeatedly until the method converges upon an answer. Given an approximate value function, V, the increments specified by (6.1) or (6.2) are computed for every time step t at which a nonterminal state is visited, but the value function is changed only once, by the sum of all the increments. Then all the available experience is processed again with the new value function to produce a new overall increment, and so on, until the value function converges. We call this  $batch\ updating$  because updates are made only after processing each complete batch of training data.

Under batch updating, TD(0) converges deterministically to a single answer independent of the step-size parameter,  $\alpha$ , as long as  $\alpha$  is chosen to be sufficiently small. The constant- $\alpha$  MC method also converges deterministically under the same conditions, but to a different answer. Understanding these two answers will help us understand the difference between the two methods. Under normal updating the methods do not move all the way to their respective batch answers, but in some sense they take steps in these directions. Before trying to understand the two answers in general, for all possible tasks, we first look at a few examples.

Example 6.3: Random walk under batch updating Batch-updating versions of TD(0) and constant- $\alpha$  MC were applied as follows to the random walk prediction example (Example 6.2). After each new episode, all episodes seen so far were treated as a batch. They were repeatedly presented to the algorithm, either TD(0) or constant- $\alpha$  MC, with  $\alpha$  sufficiently small that the value function converged. The resulting value function was then compared with  $v_{\pi}$ , and the average root mean square error across the five states (and across 100 independent repetitions of the whole experiment) was plotted to obtain

the learning curves shown in Figure 6.2. Note that the batch TD method was consistently better than the batch Monte Carlo method.

Under batch training, constant- $\alpha$ MC converges to values, V(s), that are sample averages of the actual returns experienced after visiting each state s. These are optimal estimates in the sense that they minimize the mean square error from the actual returns in the training set. In this sense it is surprising that the batch TD method was able to perform better according to the root mean square error measure shown in the figure to the right. How is it that batch TD was able to perform better than this optimal method? The answer is that the Monte Carlo method is optimal only in a limited way, and

<span id="page-148-0"></span>Figure 6.2: Performance of TD(0) and constant- $\alpha$  MC under batch training on the random walk task.

that TD is optimal in a way that is more relevant to predicting returns.

<span id="page-148-1"></span>**Example 6.4: You are the Predictor** Place yourself now in the role of the predictor of returns for an unknown Markov reward process. Suppose you observe the following eight episodes:

| A, 0, B, 0 | B, 1 |
|------------|------|
| B, 1       | B, 1 |
| B, 1       | B, 1 |
| B, 1       | B. 0 |

This means that the first episode started in state A, transitioned to B with a reward of 0, and then terminated from B with a reward of 0. The other seven episodes were even shorter, starting from B and terminating immediately. Given this batch of data, what would you say are the optimal predictions, the best values for the estimates V(A) and V(B)? Everyone would probably agree that the optimal value for V(B) is  $\frac{3}{4}$ , because six out of the eight times in state B the process terminated immediately with a return of 1, and the other two times in B the process terminated immediately with a return of 0.

But what is the optimal value for the estimate V(A) given this data? Here there are two reasonable answers. One is to observe that 100% of the times the process was in state A it traversed immediately to

B (with a reward of 0); and because we have already decided that B has value  $\frac{3}{4}$ , therefore A must have value  $\frac{3}{4}$  as well. One way of viewing this answer is that it is based on first modeling the Markov process, in this case as shown to the right, and then computing the correct estimates given the model, which indeed in this case gives  $V(A) = \frac{3}{4}$ . This is also the answer that batch TD(0) gives.

The other reasonable answer is simply to observe that we have seen A once and the return that followed it was 0; we therefore estimate *V* (A) as 0. This is the answer that batch Monte Carlo methods give. Notice that it is also the answer that gives minimum squared error on the training data. In fact, it gives zero error on the data. But still we expect the first answer to be better. If the process is Markov, we expect that the first answer will produce lower error on *future* data, even though the Monte Carlo answer is better on the existing data.

[Example 6.4](#page-148-1) illustrates a general di↵erence between the estimates found by batch TD(0) and batch Monte Carlo methods. Batch Monte Carlo methods always find the estimates that minimize mean square error on the training set, whereas batch TD(0) always finds the estimates that would be exactly correct for the maximum-likelihood model of the Markov process. In general, the *maximum-likelihood estimate* of a parameter is the parameter value whose probability of generating the data is greatest. In this case, the maximum-likelihood estimate is the model of the Markov process formed in the obvious way from the observed episodes: the estimated transition probability from *i* to *j* is the fraction of observed transitions from *i* that went to *j*, and the associated expected reward is the average of the rewards observed on those transitions. Given this model, we can compute the estimate of the value function that would be exactly correct if the model were exactly correct. This is called the *certainty-equivalence estimate* because it is equivalent to assuming that the estimate of the underlying process was known with certainty rather than being approximated. In general, batch TD(0) converges to the certainty-equivalence estimate.

This helps explain why TD methods converge more quickly than Monte Carlo methods. In batch form, TD(0) is faster than Monte Carlo methods because it computes the true certainty-equivalence estimate. This explains the advantage of TD(0) shown in the batch results on the random walk task [\(Figure 6.2\)](#page-148-0). The relationship to the certaintyequivalence estimate may also explain in part the speed advantage of nonbatch TD(0) (e.g., [Example 6.2,](#page-146-0) page [125,](#page-146-0) right graph). Although the nonbatch methods do not achieve either the certainty-equivalence or the minimum squared error estimates, they can be understood as moving roughly in these directions. Nonbatch TD(0) may be faster than constant-↵ MC because it is moving toward a better estimate, even though it is not getting all the way there. At the current time nothing more definite can be said about the relative eciency of online TD and Monte Carlo methods.

Finally, it is worth noting that although the certainty-equivalence estimate is in some sense an optimal solution, it is almost never feasible to compute it directly. If *n* = *|*S*|* is the number of states, then just forming the maximum-likelihood estimate of the process may require on the order of *n*<sup>2</sup> memory, and computing the corresponding value function requires on the order of *n*<sup>3</sup> computational steps if done conventionally. In these terms it is indeed striking that TD methods can approximate the same solution using memory no more than order *n* and repeated computations over the training set. On tasks with large state spaces, TD methods may be the only feasible way of approximating the certainty-equivalence solution.

⇤ *Exercise 6.7* Design an o↵-policy version of the TD(0) update that can be used with arbitrary target policy ⇡ and covering behavior policy *b*, using at each step *t* the importance sampling ratio ⇢*t*:*<sup>t</sup>* [\(5.3\)](#page-125-0). ⇤

### 6.4 Sarsa: On-policy TD Control

We turn now to the use of TD prediction methods for the control problem. As usual, we follow the pattern of generalized policy iteration (GPI), only this time using TD methods for the evaluation or prediction part. As with Monte Carlo methods, we face the need to trade off exploration and exploitation, and again approaches fall into two main classes: on-policy and off-policy. In this section we present an on-policy TD control method.

The first step is to learn an action-value function rather than a state-value function. In particular, for an on-policy method we must estimate  $q_{\pi}(s,a)$  for the current behavior policy  $\pi$  and for all states s and actions a. This can be done using essentially the same TD method described above for learning  $v_{\pi}$ . Recall that an episode consists of an alternating sequence of states and state-action pairs:

$$\cdots \underbrace{S_{t}}_{A_{t}} \underbrace{R_{t+1}}_{A_{t}} \underbrace{S_{t+1}}_{A_{t+1}} \underbrace{S_{t+2}}_{A_{t+2}} \underbrace{S_{t+3}}_{A_{t+2}} \underbrace{S_{t+3}}_{A_{t+3}} \cdots$$

In the previous section we considered transitions from state to state and learned the values of states. Now we consider transitions from state—action pair to state—action pair, and learn the values of state—action pairs. Formally these cases are identical: they are both Markov chains with a reward process. The theorems assuring the convergence of state values under TD(0) also apply to the corresponding algorithm for action values:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \Big].$$
 (6.7)

This update is done after every transition from a nonterminal state  $S_t$ . If  $S_{t+1}$  is terminal, then  $Q(S_{t+1}, A_{t+1})$  is defined as zero. This rule uses every element of the quintuple of events,  $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ , that make up a transition from one state-action pair to the next. This quintuple gives rise to the name Sarsa for the algorithm. The backup diagram for Sarsa is as shown to the right.

It is straightforward to design an on-policy control algorithm based on the Sarsa prediction method. As in all on-policy methods, we continually estimate  $q_{\pi}$  for the behavior policy  $\pi$ , and at the same time change  $\pi$  toward greediness with respect to  $q_{\pi}$ . The general form of the Sarsa control algorithm is given in the box on the next page.

The convergence properties of the Sarsa algorithm depend on the nature of the policy's dependence on Q. For example, one could use  $\varepsilon$ -greedy or  $\varepsilon$ -soft policies. Sarsa converges with probability 1 to an optimal policy and action-value function, under the usual conditions on the step sizes (2.7), as long as all state—action pairs are visited an infinite number of times and the policy converges in the limit to the greedy policy (which can be arranged, for example, with  $\varepsilon$ -greedy policies by setting  $\varepsilon = 1/t$ ).

Exercise 6.8 Show that an action-value version of (6.6) holds for the action-value form of the TD error  $\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$ , again assuming that the values don't change from step to step.

# Sarsa (on-policy TD control) for estimating $Q \approx q_*$ Algorithm parameters: step size $\alpha \in (0,1]$ , small $\varepsilon > 0$ Initialize Q(s,a), for all $s \in S^+, a \in \mathcal{A}(s)$ , arbitrarily except that $Q(terminal,\cdot) = 0$ Loop for each episode: Initialize S Choose A from S using policy derived from Q (e.g., $\varepsilon$ -greedy) Loop for each step of episode: Take action A, observe R, S' Choose A' from S' using policy derived from Q (e.g., $\varepsilon$ -greedy) $Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma Q(S',A') - Q(S,A)]$ $S \leftarrow S'$ ; $A \leftarrow A'$ ; until S is terminal

Example 6.5: Windy Gridworld Shown inset below is a standard gridworld, with start and goal states, but with one difference: there is a crosswind running upward through the middle of the grid. The actions are the standard four—up, down, right, and left—but in the middle region the resultant next states are shifted upward by a "wind," the strength of which varies from column to column. The strength of the wind

is given below each column, in number of cells shifted upward. For example, if you are one cell to the right of the goal, then the action left takes you to the cell just above the goal. This is an undiscounted episodic task, with constant rewards of -1 until the goal state is reached.

The graph to the right shows the results of applying  $\varepsilon$ -greedy Sarsa to this task, with  $\varepsilon=0.1,\ \alpha=0.5,$  and the initial values Q(s,a)=0 for all s,a. The increasing slope of the graph shows that the goal was reached more quickly over time. By

8000 time steps, the greedy policy was long since optimal (a trajectory from it is shown inset); continued  $\varepsilon$ -greedy exploration kept the average episode length at about 17 steps, two more than the minimum of 15. Note that Monte Carlo methods cannot easily be used here because termination is not guaranteed for all policies. If a policy was ever found that caused the agent to stay in the same state, then the next episode would never end. Online learning methods such as Sarsa do not have this problem because they quickly learn during the episode that such policies are poor, and switch to something else.

Exercise 6.9: Windy Gridworld with King's Moves (programming) Re-solve the windy gridworld assuming eight possible actions, including the diagonal moves, rather than four. How much better can you do with the extra actions? Can you do even better by including a ninth action that causes no movement at all other than that caused by the wind?  $\Box$

Exercise 6.10: Stochastic Wind (programming) Re-solve the windy gridworld task with King's moves, assuming that the effect of the wind, if there is any, is stochastic, sometimes varying by 1 from the mean values given for each column. That is, a third of the time you move exactly according to these values, as in the previous exercise, but also a third of the time you move one cell above that, and another third of the time you move one cell below that. For example, if you are one cell to the right of the goal and you move left, then one-third of the time you move one cell above the goal, one-third of the time you move two cells above the goal, and one-third of the time you move to the goal.  $\Box$

### 6.5 Q-learning: Off-policy TD Control

One of the early breakthroughs in reinforcement learning was the development of an off-policy TD control algorithm known as *Q-learning* (Watkins, 1989), defined by

<span id="page-152-0"></span>
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right].$$
 (6.8)

In this case, the learned action-value function, Q, directly approximates  $q_*$ , the optimal action-value function, independent of the policy being followed. This dramatically simplifies the analysis of the algorithm and enabled early convergence proofs. The policy still has an effect in that it determines which state—action pairs are visited and updated. However, all that is required for correct convergence is that all pairs continue to be updated. As we observed in Chapter 5, this is a minimal requirement in the sense that any method guaranteed to find optimal behavior in the general case must require it. Under this assumption and a variant of the usual stochastic approximation conditions on the sequence of step-size parameters, Q has been shown to converge with probability 1 to  $q_*$ . The Q-learning algorithm is shown below in procedural form.

```
Q-learning (off-policy TD control) for estimating \pi \approx \pi_*

Algorithm parameters: step size \alpha \in (0,1], small \varepsilon > 0

Initialize Q(s,a), for all s \in \mathbb{S}^+, a \in \mathcal{A}(s), arbitrarily except that Q(terminal,\cdot) = 0

Loop for each episode:

Initialize S

Loop for each step of episode:

Choose A from S using policy derived from Q (e.g., \varepsilon-greedy)

Take action A, observe R, S'

Q(S,A) \leftarrow Q(S,A) + \alpha \left[R + \gamma \max_a Q(S',a) - Q(S,A)\right]
S \leftarrow S'
\nuntil S is terminal
```

What is the backup diagram for Q-learning? The rule (6.8) updates a state—action pair, so the top node, the root of the update, must be a small, filled action node. The update is also *from* action nodes, maximizing over all those actions possible in the next state. Thus the bottom nodes of the backup diagram should be all these action nodes. Finally, remember that we indicate taking the maximum of these "next action" nodes with an arc across them (Figure 3.4-right). Can you guess now what the diagram is? If so, please do make a guess before turning to the answer in Figure 6.4 on page 134.

**Example 6.6: Cliff Walking** This gridworld example compares Sarsa and Q-learning, highlighting the difference between on-policy (Sarsa) and off-policy (Q-learning) methods.

Consider the gridworld shown to the right. This is a standard undiscounted, episodic task, with start and goal states, and the usual actions causing movement up, down, right, and left. Reward is -1 on all transitions except those into the region marked "The Cliff." Stepping into this region incurs a reward of -100 and sends the agent instantly back to the start.

The graph to the right shows the performance of the Sarsa and Qlearning methods with  $\varepsilon$ -greedy action selection,  $\varepsilon = 0.1$ . After an initial transient, Q-learning learns values for the optimal policy, that which travels right along the edge of the cliff. Unfortunately, this results in its occasionally falling off the cliff because of the  $\varepsilon$ -greedy action selection. Sarsa, on the other hand, takes the action selection into account and learns the longer but safer path through the upper part of the grid. Although Q-learning actually learns the values of the optimal policy, its online performance is worse than that of Sarsa, which

learns the round about policy. Of course, if  $\varepsilon$  were gradually reduced, then both methods would asymptotically converge to the optimal policy.

Exercise 6.11 Why is Q-learning considered an off-policy control method?  $\Box$

Exercise 6.12 Suppose action selection is greedy. Is Q-learning then exactly the same algorithm as Sarsa? Will they make exactly the same action selections and weight updates?  $\Box$

### 6.6 Expected Sarsa

Consider the learning algorithm that is just like Q-learning except that instead of the maximum over next state—action pairs it uses the expected value, taking into account how likely each action is under the current policy. That is, consider the algorithm with the update rule

$$Q(S_{t}, A_{t}) \leftarrow Q(S_{t}, A_{t}) + \alpha \Big[ R_{t+1} + \gamma \mathbb{E}_{\pi} [Q(S_{t+1}, A_{t+1}) \mid S_{t+1}] - Q(S_{t}, A_{t}) \Big]$$

$$= Q(S_{t}, A_{t}) + \alpha \Big[ R_{t+1} + \gamma \sum_{a} \pi(a \mid S_{t+1}) Q(S_{t+1}, a) - Q(S_{t}, A_{t}) \Big], \qquad (6.9)$$

but that otherwise follows the schema of Q-learning. Given the next state,  $S_{t+1}$ , this algorithm moves deterministically in the same direction as Sarsa moves in expectation, and accordingly it is called Expected Sarsa. Its backup diagram is shown on the right in Figure 6.4.

Expected Sarsa is more complex computationally than Sarsa but, in return, it eliminates the variance due to the random selection of  $A_{t+1}$ . Given the same amount of experience we might expect it to perform slightly better than Sarsa, and indeed it generally does. Figure 6.3 shows summary results on the cliff-walking task with Expected Sarsa compared to Sarsa and Q-learning. Expected Sarsa retains the significant advantage of Sarsa over Q-learning on this problem. In addition, Expected Sarsa shows a significant improvement

<span id="page-154-0"></span>Figure 6.3: Interim and asymptotic performance of TD control methods on the cliff-walking task as a function of  $\alpha$ . All algorithms used an  $\varepsilon$ -greedy policy with  $\varepsilon = 0.1$ . Asymptotic performance is an average over 100,000 episodes whereas interim performance is an average over the first 100 episodes. These data are averages of over 50,000 and 10 runs for the interim and asymptotic cases respectively. The solid circles mark the best interim performance of each method. Adapted from van Seijen et al. (2009).

<span id="page-155-0"></span>Figure 6.4: The backup diagrams for Q-learning and Expected Sarsa.

over Sarsa over a wide range of values for the step-size parameter ↵. In cli↵ walking the state transitions are all deterministic and all randomness comes from the policy. In such cases, Expected Sarsa can safely set ↵ = 1 without su↵ering any degradation of asymptotic performance, whereas Sarsa can only perform well in the long run at a small value of ↵, at which short-term performance is poor. In this and other examples there is a consistent empirical advantage of Expected Sarsa over Sarsa.

In these cli↵ walking results Expected Sarsa was used on-policy, but in general it might use a policy di↵erent from the target policy ⇡ to generate behavior, in which case it becomes an o↵-policy algorithm. For example, suppose ⇡ is the greedy policy while behavior is more exploratory; then Expected Sarsa is exactly Q-learning. In this sense Expected Sarsa subsumes and generalizes Q-learning while reliably improving over Sarsa. Except for the small additional computational cost, Expected Sarsa may completely dominate both of the other more-well-known TD control algorithms.

### 6.7 Maximization Bias and Double Learning

All the control algorithms that we have discussed so far involve maximization in the construction of their target policies. For example, in Q-learning the target policy is the greedy policy given the current action values, which is defined with a max, and in Sarsa the policy is often "-greedy, which also involves a maximization operation. In these algorithms, a maximum over estimated values is used implicitly as an estimate of the maximum value, which can lead to a significant positive bias. To see why, consider a single state *s* where there are many actions *a* whose true values, *q*(*s, a*), are all zero but whose estimated values, *Q*(*s, a*), are uncertain and thus distributed some above and some below zero. The maximum of the true values is zero, but the maximum of the estimates is positive, a positive bias. We call this *maximization bias*.

Example 6.7: Maximization Bias Example The small MDP shown inset in [Figure 6.5](#page-156-0) provides a simple example of how maximization bias can harm the performance of TD control algorithms. The MDP has two non-terminal states A and B. Episodes always start in A with a choice between two actions, left and right. The right action transitions immediately to the terminal state with a reward and return of zero. The left action transitions to B, also with a reward of zero, from which there are many possible actions all of which cause immediate termination with a reward drawn from a normal distribution with mean 0*.*1 and variance 1*.*0. Thus, the expected return for any trajectory starting with left is 0*.*1, and thus taking left in state A is always a

<span id="page-156-0"></span>Figure 6.5: Comparison of Q-learning and Double Q-learning on a simple episodic MDP (shown inset). Q-learning initially learns to take the left action much more often than the right action, and always takes it significantly more often than the 5% minimum probability enforced by  $\varepsilon$ -greedy action selection with  $\varepsilon = 0.1$ . In contrast, Double Q-learning is essentially unaffected by maximization bias. These data are averaged over 10,000 runs. The initial action-value estimates were zero. Any ties in  $\varepsilon$ -greedy action selection were broken randomly.

mistake. Nevertheless, our control methods may favor left because of maximization bias making B appear to have a positive value. Figure 6.5 shows that Q-learning with  $\varepsilon$ -greedy action selection initially learns to strongly favor the left action on this example. Even at asymptote, Q-learning takes the left action about 5% more often than is optimal at our parameter settings ( $\varepsilon = 0.1$ ,  $\alpha = 0.1$ , and  $\gamma = 1$ ).

Are there algorithms that avoid maximization bias? To start, consider a bandit case in which we have noisy estimates of the value of each of many actions, obtained as sample averages of the rewards received on all the plays with each action. As we discussed above, there will be a positive maximization bias if we use the maximum of the estimates as an estimate of the maximum of the true values. One way to view the problem is that it is due to using the same samples (plays) both to determine the maximizing action and to estimate its value. Suppose we divided the plays in two sets and used them to learn two independent estimates, call them  $Q_1(a)$  and  $Q_2(a)$ , each an estimate of the true value q(a), for all  $a \in A$ . We could then use one estimate, say  $Q_1$ , to determine the maximizing action  $A^* = \arg\max_a Q_1(a)$ , and the other,  $Q_2$ , to provide the estimate of its value,  $Q_2(A^*) = Q_2(\arg\max_a Q_1(a))$ . This estimate will then be unbiased in the sense that  $\mathbb{E}[Q_2(A^*)] = q(A^*)$ . We can also repeat the process with the role of the two estimates reversed to yield a second unbiased estimate  $Q_1(\operatorname{arg\,max}_a Q_2(a))$ . This is the idea of double learning. Note that although we learn two estimates, only one estimate is updated on each play; double learning doubles the memory requirements, but does not increase the amount of computation per step.

The idea of double learning extends naturally to algorithms for full MDPs. For example, the double learning algorithm analogous to Q-learning, called Double Q-learning, divides the time steps in two, perhaps by flipping a coin on each step. If the coin comes up heads, the update is

$$Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q_2(S_{t+1}, \underset{a}{\operatorname{arg\,max}} Q_1(S_{t+1}, a)) - Q_1(S_t, A_t) \right]. \tag{6.10}$$

If the coin comes up tails, then the same update is done with *Q*<sup>1</sup> and *Q*<sup>2</sup> switched, so that *Q*<sup>2</sup> is updated. The two approximate value functions are treated completely symmetrically. The behavior policy can use both action-value estimates. For example, an "-greedy policy for Double Q-learning could be based on the average (or sum) of the two action-value estimates. A complete algorithm for Double Q-learning is given in the box below. This is the algorithm used to produce the results in [Figure 6.5.](#page-156-0) In that example, double learning seems to eliminate the harm caused by maximization bias. Of course there are also double versions of Sarsa and Expected Sarsa.

```
Double Q-learning, for estimating Q1 ⇡ Q2 ⇡ q⇤
Algorithm parameters: step size ↵ 2 (0, 1], small " > 0
Initialize Q1(s, a) and Q2(s, a), for all s 2 S+, a 2 A(s), such that Q(terminal, ·)=0
Loop for each episode:
  Initialize S
  Loop for each step of episode:
     Choose A from S using the policy "-greedy in Q1 + Q2
     Take action A, observe R, S0
     With 0.5 probabilility:
        Q1(S, A) Q1(S, A) + ↵
                                  ⇣
                                   R + Q2

                                             S0
                                               , argmaxa Q1(S0
                                                               , a)

                                                                     Q1(S, A)
                                                                               ⌘
     else:
        Q2(S, A) Q2(S, A) + ↵
                                  ⇣
                                   R + Q1

                                             S0
                                               , argmaxa Q2(S0
                                                               , a)

                                                                     Q2(S, A)
                                                                               ⌘
     S S0
  until S is terminal
```

⇤ *Exercise 6.13* What are the update equations for Double Expected Sarsa with an "-greedy target policy? ⇤

### 6.8 Games, Afterstates, and Other Special Cases

In this book we try to present a uniform approach to a wide class of tasks, but of course there are always exceptional tasks that are better treated in a specialized way. For example, our general approach involves learning an *action*-value function, but in [Chapter 1](#page-22-0) we presented a TD method for learning to play tic-tac-toe that learned something much more like a *state*-value function. If we look closely at that example, it becomes apparent that the function learned there is neither an action-value function nor a state-value function in the usual sense. A conventional state-value function evaluates states in which the agent has the option of selecting an action, but the state-value function used in

tic-tac-toe evaluates board positions *after* the agent has made its move. Let us call these *afterstates*, and value functions over these, *afterstate value functions*. Afterstates are useful when we have knowledge of an initial part of the environment's dynamics but not necessarily of the full dynamics. For example, in games we typically know the immediate e↵ects of our moves. We know for each possible chess move what the resulting position will be, but not how our opponent will reply. Afterstate value functions are a natural way to take advantage of this kind of knowledge and thereby produce a more ecient learning method.

The reason it is more ecient to design algorithms in terms of afterstates is apparent from the tic-tac-toe example. A conventional action-value function would map from positions *and* moves to an estimate of the value. But many position—move pairs produce the same resulting position, as in the example below: In such cases the

position–move pairs are di↵erent but produce the same "afterposition," and thus must have the same value. A conventional action-value function would have to separately assess both pairs, whereas an afterstate value function would immediately assess both equally. Any learning about the position–move pair on the left would immediately transfer to the pair on the right.

Afterstates arise in many tasks, not just games. For example, in queuing tasks there are actions

such as assigning customers to servers, rejecting customers, or discarding information. In such cases the actions are in fact defined in terms of their immediate e↵ects, which are completely known.

It is impossible to describe all the possible kinds of specialized problems and corresponding specialized learning algorithms. However, the principles developed in this book should apply widely. For example, afterstate methods are still aptly described in terms of generalized policy iteration, with a policy and (afterstate) value function interacting in essentially the same way. In many cases one will still face the choice between on-policy and o↵-policy methods for managing the need for persistent exploration.

*Exercise 6.14* Describe how the task of Jack's Car Rental [\(Example 4.2\)](#page-102-0) could be reformulated in terms of afterstates. Why, in terms of this specific task, would such a reformulation be likely to speed convergence? ⇤

### 6.9 Summary

In this chapter we introduced a new kind of learning method, temporal-di↵erence (TD) learning, and showed how it can be applied to the reinforcement learning problem. As usual, we divided the overall problem into a prediction problem and a control problem. TD methods are alternatives to Monte Carlo methods for solving the prediction problem. In both cases, the extension to the control problem is via the idea of generalized policy iteration (GPI) that we abstracted from dynamic programming. This is the idea that approximate policy and value functions should interact in such a way that they both move toward their optimal values.

One of the two processes making up GPI drives the value function to accurately predict returns for the current policy; this is the prediction problem. The other process drives the policy to improve locally (e.g., to be "-greedy) with respect to the current value function. When the first process is based on experience, a complication arises concerning maintaining sucient exploration. We can classify TD control methods according to whether they deal with this complication by using an on-policy or o↵-policy approach. Sarsa is an on-policy method, and Q-learning is an o↵-policy method. Expected Sarsa is also an o↵-policy method as we present it here. There is a third way in which TD methods can be extended to control which we did not include in this chapter, called actor–critic methods. These methods are covered in full in [Chapter 13.](#page-342-0)

The methods presented in this chapter are today the most widely used reinforcement learning methods. This is probably due to their great simplicity: they can be applied online, with a minimal amount of computation, to experience generated from interaction with an environment; they can be expressed nearly completely by single equations that can be implemented with small computer programs. In the next few chapters we extend these algorithms, making them slightly more complicated and significantly more powerful. All the new algorithms will retain the essence of those introduced here: they will be able to process experience online, with relatively little computation, and they will be driven by TD errors. The special cases of TD methods introduced in the present chapter should rightly be called *one-step, tabular, model-free* TD methods. In the next two chapters we extend them to *n*-step forms (a link to Monte Carlo methods) and forms that include a model of the environment (a link to planning and dynamic programming). Then, in the second part of the book we extend them to various forms of function approximation rather than tables (a link to deep learning and artificial neural networks).

Finally, in this chapter we have discussed TD methods entirely within the context of reinforcement learning problems, but TD methods are actually more general than this. They are general methods for learning to make long-term predictions about dynamical systems. For example, TD methods may be relevant to predicting financial data, life spans, election outcomes, weather patterns, animal behavior, demands on power stations, or customer purchases. It was only when TD methods were analyzed as pure prediction methods, independent of their use in reinforcement learning, that their theoretical properties first came to be well understood. Even so, these other potential applications of TD learning methods have not yet been extensively explored.

*6.9. Summary 139*

### Bibliographical and Historical Remarks

As we outlined in [Chapter 1,](#page-22-0) the idea of TD learning has its early roots in animal learning psychology and artificial intelligence, most notably the work of Samuel (1959) and Klopf (1972). Samuel's work is described as a case study in [Section 16.2.](#page-447-0) Also related to TD learning are Holland's (1975, 1976) early ideas about consistency among value predictions. These influenced one of the authors (Barto), who was a graduate student from 1970 to 1975 at the University of Michigan, where Holland was teaching. Holland's ideas led to a number of TD-related systems, including the work of Booker (1982) and the bucket brigade of Holland (1986), which is related to Sarsa as discussed below.

6.1–2 Most of the specific material from these sections is from Sutton (1988), including the TD(0) algorithm, the random walk example, and the term "temporaldi↵erence learning." The characterization of the relationship to dynamic programming and Monte Carlo methods was influenced by Watkins (1989), Werbos (1987), and others. The use of backup diagrams was new to the first edition of this book.

Tabular TD(0) was proved to converge in the mean by Sutton (1988) and with probability 1 by Dayan (1992), based on the work of Watkins and Dayan (1992). These results were extended and strengthened by Jaakkola, Jordan, and Singh (1994) and Tsitsiklis (1994) by using extensions of the powerful existing theory of stochastic approximation. Other extensions and generalizations are covered in later chapters.

- 6.3 The optimality of the TD algorithm under batch training was established by Sutton (1988). Illuminating this result is Barnard's (1993) derivation of the TD algorithm as a combination of one step of an incremental method for learning a model of the Markov chain and one step of a method for computing predictions from the model. The term *certainty equivalence* is from the adaptive control literature (e.g., Goodwin and Sin, 1984).
- 6.4 The Sarsa algorithm was introduced by Rummery and Niranjan (1994). They explored it in conjunction with artificial neural networks and called it "Modified Connectionist Q-learning". The name "Sarsa" was introduced by Sutton (1996). The convergence of one-step tabular Sarsa (the form treated in this chapter) has been proved by Singh, Jaakkola, Littman, and Szepesv´ari (2000). The "windy gridworld" example was suggested by Tom Kalt.

Holland's (1986) bucket brigade idea evolved into an algorithm closely related to Sarsa. The original idea of the bucket brigade involved chains of rules triggering each other; it focused on passing credit back from the current rule to the rules that triggered it. Over time, the bucket brigade came to be more like TD learning in passing credit back to any temporally preceding rule, not just to the ones that triggered the current rule. The modern form of the bucket brigade, when simplified in various natural ways, is nearly identical to one-step Sarsa, as detailed by Wilson (1994).

- 6.5 Q-learning was introduced by Watkins (1989), whose outline of a convergence proof was made rigorous by Watkins and Dayan (1992). More general convergence results were proved by Jaakkola, Jordan, and Singh (1994) and Tsitsiklis (1994).
- 6.6 The Expected Sarsa algorithm was introduced by George John (1994), who called it "Q-learning" and stressed its advantages over Q-learning as an o↵-policy algorithm. John's work was not known to us when we presented Expected Sarsa in the first edition of this book as an exercise, or to van Seijen, van Hasselt, Whiteson, and Weiring (2009) when they established Expected Sarsa's convergence properties and conditions under which it will outperform regular Sarsa and Q-learning. Our [Figure 6.3](#page-154-0) is adapted from their results. Van Seijen et al. defined "Expected Sarsa" to be an on-policy method exclusively (as we did in the first edition), whereas now we use this name for the general algorithm in which the target and behavior policies may di↵er. The general o↵-policy view of Expected Sarsa was noted by van Hasselt (2011), who called it "General Q-learning."
- 6.7 Maximization bias and double learning were introduced and extensively investigated by van Hasselt (2010, 2011). The example MDP in [Figure 6.5](#page-156-0) was adapted from that in his Figure 4.1 (van Hasselt, 2011).
- 6.8 The notion of an afterstate is the same as that of a "post-decision state" (Van Roy, Bertsekas, Lee, and Tsitsiklis, 1997; Powell, 2011).

## <span id="page-342-0"></span>Chapter 13

# Policy Gradient Methods

In this chapter we consider something new. So far in this book almost all the methods have been action-value methods; they learned the values of actions and then selected actions based on their estimated action values<sup>1</sup>; their policies would not even exist without the action-value estimates. In this chapter we consider methods that instead learn a parameterized policy that can select actions without consulting a value function. A value function may still be used to learn the policy parameter, but is not required for action selection. We use the notation  $\theta \in \mathbb{R}^{d'}$  for the policy's parameter vector. Thus we write  $\pi(a|s,\theta) = \Pr\{A_t = a \mid S_t = s, \theta_t = \theta\}$  for the probability that action a is taken at time t given that the environment is in state s at time t with parameter  $\theta$ . If a method uses a learned value function as well, then the value function's weight vector is denoted  $\mathbf{w} \in \mathbb{R}^d$  as usual, as in  $\hat{v}(s,\mathbf{w})$ .

In this chapter we consider methods for learning the policy parameter based on the gradient of some scalar performance measure  $J(\theta)$  with respect to the policy parameter. These methods seek to maximize performance, so their updates approximate gradient ascent in J:

<span id="page-342-2"></span>
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha \widehat{\nabla J(\boldsymbol{\theta}_t)}, \tag{13.1}$$

where  $\widehat{\nabla J(\boldsymbol{\theta}_t)} \in \mathbb{R}^{d'}$  is a stochastic estimate whose expectation approximates the gradient of the performance measure with respect to its argument  $\boldsymbol{\theta}_t$ . All methods that follow this general schema we call *policy gradient methods*, whether or not they also learn an approximate value function. Methods that learn approximations to both policy and value functions are often called *actor-critic methods*, where 'actor' is a reference to the learned policy, and 'critic' refers to the learned value function, usually a state-value function. First we treat the episodic case, in which performance is defined as the value of the start state under the parameterized policy, before going on to consider the continuing case, in

<span id="page-342-1"></span><sup>&</sup>lt;sup>1</sup>The lone exception is the gradient bandit algorithms of Section 2.8. In fact, that section goes through many of the same steps, in the single-state bandit case, as we go through here for full MDPs. Reviewing that section would be good preparation for fully understanding this chapter.

which performance is defined as the average reward rate, as in Section 10.3. In the end, we are able to express the algorithms for both cases in very similar terms.

### <span id="page-343-1"></span>13.1 Policy Approximation and its Advantages

In policy gradient methods, the policy can be parameterized in any way, as long as  $\pi(a|s, \theta)$  is differentiable with respect to its parameters, that is, as long as  $\nabla \pi(a|s, \theta)$  (the column vector of partial derivatives of  $\pi(a|s, \theta)$  with respect to the components of  $\theta$ ) exists and is finite for all  $s \in \mathcal{S}$ ,  $a \in \mathcal{A}(s)$ , and  $\theta \in \mathbb{R}^{d'}$ . In practice, to ensure exploration we generally require that the policy never becomes deterministic (i.e., that  $\pi(a|s, \theta) \in (0, 1)$ , for all  $s, a, \theta$ ). In this section we introduce the most common parameterization for discrete action spaces and point out the advantages it offers over action-value methods. Policy-based methods also offer useful ways of dealing with continuous action spaces, as we describe later in Section 13.7.

If the action space is discrete and not too large, then a natural and common kind of parameterization is to form parameterized numerical preferences  $h(s, a, \theta) \in \mathbb{R}$  for each state—action pair. The actions with the highest preferences in each state are given the highest probabilities of being selected, for example, according to an exponential soft-max distribution:

<span id="page-343-2"></span>
$$\pi(a|s,\boldsymbol{\theta}) \doteq \frac{e^{h(s,a,\boldsymbol{\theta})}}{\sum_{b} e^{h(s,b,\boldsymbol{\theta})}},\tag{13.2}$$

where  $e \approx 2.71828$  is the base of the natural logarithm. Note that the denominator here is just what is required so that the action probabilities in each state sum to one. We call this kind of policy parameterization soft-max in action preferences.

The action preferences themselves can be parameterized arbitrarily. For example, they might be computed by a deep artificial neural network (ANN), where  $\theta$  is the vector of all the connection weights of the network (as in the AlphaGo system described in Section 16.6). Or the preferences could simply be linear in features,

<span id="page-343-0"></span>
$$h(s, a, \boldsymbol{\theta}) = \boldsymbol{\theta}^{\top} \mathbf{x}(s, a), \tag{13.3}$$

using feature vectors  $\mathbf{x}(s, a) \in \mathbb{R}^{d'}$  constructed by any of the methods described in Section 9.5.

One advantage of parameterizing policies according to the soft-max in action preferences is that the approximate policy can approach a deterministic policy, whereas with  $\varepsilon$ -greedy action selection over action values there is always an  $\varepsilon$  probability of selecting a random action. Of course, one could select according to a soft-max distribution based on action values, but this alone would not allow the policy to approach a deterministic policy. Instead, the action-value estimates would converge to their corresponding true values, which would differ by a finite amount, translating to specific probabilities other than 0 and 1. If the soft-max distribution included a temperature parameter, then the temperature could be reduced over time to approach determinism, but in practice it would be difficult to choose the reduction schedule, or even the initial temperature, without more prior knowledge of the true action values than we would like to assume. Action preferences

are di↵erent because they do not approach specific values; instead they are driven to produce the optimal stochastic policy. If the optimal policy is deterministic, then the preferences of the optimal actions will be driven infinitely higher than all suboptimal actions (if permitted by the parameterization).

A second advantage of parameterizing policies according to the soft-max in action preferences is that it enables the selection of actions with arbitrary probabilities. In problems with significant function approximation, the best approximate policy may be stochastic. For example, in card games with imperfect information the optimal play is often to do two di↵erent things with specific probabilities, such as when blung in Poker. Action-value methods have no natural way of finding stochastic optimal policies, whereas policy approximating methods can, as shown in [Example 13.1.](#page-343-0)

### Example 13.1 Short corridor with switched actions

Consider the small corridor gridworld shown inset in the graph below. The reward is 1 per step, as usual. In each of the three nonterminal states there are only two actions, right and left. These actions have their usual consequences in the first and third states (left causes no movement in the first state), but in the second state they are reversed, so that right moves to the left and left moves to the right. The problem is dicult because all the states appear identical under the function approximation. In particular, we define x(*s,*right) = [1*,* 0]<sup>&</sup>gt; and x(*s,* left) = [0*,* 1]>, for all *s*. An action-value method with "-greedy action selection is forced to choose between just two policies: choosing right with high probability 1 "*/*2 on all steps or choosing left with the same high probability on all time steps. If " = 0*.*1, then these two policies achieve a value (at the start state) of less than 44 and 82, respectively, as shown in the graph. A method can do significantly better if it can learn a specific probability with which to select right. The best probability is about 0.59, which achieves a value of about 11*.*6.

Perhaps the simplest advantage that policy parameterization may have over actionvalue parameterization is that the policy may be a simpler function to approximate. Problems vary in the complexity of their policies and action-value functions. For some, the action-value function is simpler and thus easier to approximate. For others, the policy is simpler. In the latter case a policy-based method will typically learn faster and yield a superior asymptotic policy (as in Tetris; see S¸im¸sek, Alg´orta, and Kothiyal, 2016).

Finally, we note that the choice of policy parameterization is sometimes a good way of injecting prior knowledge about the desired form of the policy into the reinforcement learning system. This is often the most important reason for using a policy-based learning method.

*Exercise 13.1* Use your knowledge of the gridworld and its dynamics to determine an *exact* symbolic expression for the optimal probability of selecting the right action in [Example 13.1.](#page-343-0) ⇤

### <span id="page-345-0"></span>13.2 The Policy Gradient Theorem

In addition to the practical advantages of policy parameterization over "-greedy action selection, there is also an important theoretical advantage. With continuous policy parameterization the action probabilities change smoothly as a function of the learned parameter, whereas in "-greedy selection the action probabilities may change dramatically for an arbitrarily small change in the estimated action values, if that change results in a di↵erent action having the maximal value. Largely because of this, stronger convergence guarantees are available for policy-gradient methods than for action-value methods. In particular, it is the continuity of the policy dependence on the parameters that enables policy-gradient methods to approximate gradient ascent [\(13.1\)](#page-342-2).

The episodic and continuing cases define the performance measure, *J*(✓), di↵erently and thus have to be treated separately to some extent. Nevertheless, we will try to present both cases uniformly, and we develop a notation so that the major theoretical results can be described with a single set of equations.

In this section we treat the episodic case, for which we define the performance measure as the value of the start state of the episode. We can simplify the notation without losing any meaningful generality by assuming that every episode starts in some particular (non-random) state *s*0. Then, in the episodic case we define performance as

$$J(\boldsymbol{\theta}) \doteq v_{\pi_{\boldsymbol{\theta}}}(s_0), \tag{13.4}$$

where *v*⇡✓ is the true value function for ⇡✓, the policy determined by ✓. From here on in our discussion we will assume no discounting ( = 1) for the episodic case, although for completeness we do include the possibility of discounting in the boxed algorithms.

With function approximation it may seem challenging to change the policy parameter in a way that ensures improvement. The problem is that performance depends on both the action selections and the distribution of states in which those selections are made, and that both of these are a↵ected by the policy parameter. Given a state, the e↵ect of the policy parameter on the actions, and thus on reward, can be computed in a relatively straightforward way from knowledge of the parameterization. But the e↵ect of the policy

### Proof of the Policy Gradient Theorem (episodic case)

<span id="page-346-0"></span>With just elementary calculus and re-arranging of terms, we can prove the policy gradient theorem from first principles. To keep the notation simple, we leave it implicit in all cases that ⇡ is a function of ✓, and all gradients are also implicitly with respect to ✓. First note that the gradient of the state-value function can be written in terms of the action-value function as

$$\nabla v_{\pi}(s) = \nabla \left[ \sum_{a} \pi(a|s) q_{\pi}(s, a) \right], \quad \text{for all } s \in \mathbb{S}$$

$$= \sum_{a} \left[ \nabla \pi(a|s) q_{\pi}(s, a) + \pi(a|s) \nabla q_{\pi}(s, a) \right] \quad \text{(product rule of calculus)}$$

$$= \sum_{a} \left[ \nabla \pi(a|s) q_{\pi}(s, a) + \pi(a|s) \nabla \sum_{s', r} p(s', r|s, a) (r + v_{\pi}(s')) \right]$$

$$= \sum_{a} \left[ \nabla \pi(a|s) q_{\pi}(s, a) + \pi(a|s) \sum_{s', r} p(s'|s, a) \nabla v_{\pi}(s') \right]$$

$$= \sum_{a} \left[ \nabla \pi(a|s) q_{\pi}(s, a) + \pi(a|s) \sum_{s'} p(s'|s, a) \nabla v_{\pi}(s') \right]$$

$$= \sum_{a} \left[ \nabla \pi(a|s) q_{\pi}(s, a) + \pi(a|s) \sum_{s'} p(s'|s, a)$$

$$= \sum_{a'} \left[ \nabla \pi(a'|s') q_{\pi}(s', a') + \pi(a'|s') \sum_{s''} p(s''|s', a') \nabla v_{\pi}(s'') \right] \right]$$

$$= \sum_{x \in \mathbb{S}} \sum_{k=0}^{\infty} \Pr(s \to x, k, \pi) \sum_{a} \nabla \pi(a|x) q_{\pi}(x, a),$$

after repeated unrolling, where Pr(*s*!*x, k,* ⇡) is the probability of transitioning from state *s* to state *x* in *k* steps under policy ⇡. It is then immediate that

$$\nabla J(\boldsymbol{\theta}) = \nabla v_{\pi}(s_{0})$$

$$= \sum_{s} \left(\sum_{k=0}^{\infty} \Pr(s_{0} \to s, k, \pi)\right) \sum_{a} \nabla \pi(a|s) q_{\pi}(s, a)$$

$$= \sum_{s} \eta(s) \sum_{a} \nabla \pi(a|s) q_{\pi}(s, a) \qquad \text{(box page 199)}$$

$$= \sum_{s'} \eta(s') \sum_{s} \frac{\eta(s)}{\sum_{s'} \eta(s')} \sum_{a} \nabla \pi(a|s) q_{\pi}(s, a)$$

$$= \sum_{s'} \eta(s') \sum_{s} \mu(s) \sum_{a} \nabla \pi(a|s) q_{\pi}(s, a) \qquad \text{(Eq. 9.3)}$$

$$\propto \sum_{s} \mu(s) \sum_{a} \nabla \pi(a|s) q_{\pi}(s, a) \qquad \text{(Q.E.D.)}$$

on the state distribution is a function of the environment and is typically unknown. How can we estimate the performance gradient with respect to the policy parameter when the gradient depends on the unknown e↵ect of policy changes on the state distribution?

Fortunately, there is an excellent theoretical answer to this challenge in the form of the *policy gradient theorem*, which provides an analytic expression for the gradient of performance with respect to the policy parameter (which is what we need to approximate for gradient ascent [\(13.1\)](#page-342-2)) that does *not* involve the derivative of the state distribution. The policy gradient theorem for the episodic case establishes that

<span id="page-347-1"></span>
$$\nabla J(\boldsymbol{\theta}) \propto \sum_{s} \mu(s) \sum_{a} q_{\pi}(s, a) \nabla \pi(a|s, \boldsymbol{\theta}),$$
 (13.5)

where the gradients are column vectors of partial derivatives with respect to the components of ✓, and ⇡ denotes the policy corresponding to parameter vector ✓. The symbol / here means "proportional to". In the episodic case, the constant of proportionality is the average length of an episode, and in the continuing case it is 1, so that the relationship is actually an equality. The distribution *µ* here (as in Chapters [9](#page-218-0) and [10\)](#page-264-0) is the on-policy distribution under ⇡ (see page [199\)](#page-220-0). The policy gradient theorem is proved for the episodic case in the box on the previous page.

### <span id="page-347-2"></span>13.3 REINFORCE: Monte Carlo Policy Gradient

We are now ready to derive our first policy-gradient learning algorithm. Recall our overall strategy of stochastic gradient ascent [\(13.1\)](#page-342-2), which requires a way to obtain samples such that the expectation of the sample gradient is proportional to the actual gradient of the performance measure as a function of the parameter. The sample gradients need only be proportional to the gradient because any constant of proportionality can be absorbed into the step size ↵, which is otherwise arbitrary. The policy gradient theorem gives an exact expression proportional to the gradient; all that is needed is some way of sampling whose expectation equals or approximates this expression. Notice that the right-hand side of the policy gradient theorem is a sum over states weighted by how often the states occur under the target policy ⇡; if ⇡ is followed, then states will be encountered in these proportions. Thus

<span id="page-347-0"></span>
$$\nabla J(\boldsymbol{\theta}) \propto \sum_{s} \mu(s) \sum_{a} q_{\pi}(s, a) \nabla \pi(a|s, \boldsymbol{\theta})$$

$$= \mathbb{E}_{\pi} \left[ \sum_{a} q_{\pi}(S_{t}, a) \nabla \pi(a|S_{t}, \boldsymbol{\theta}) \right]. \tag{13.6}$$

We could stop here and instantiate our stochastic gradient-ascent algorithm [\(13.1\)](#page-342-2) as

$$\boldsymbol{\theta}_{t+1} \doteq \boldsymbol{\theta}_t + \alpha \sum_{a} \hat{q}(S_t, a, \mathbf{w}) \nabla \pi(a|S_t, \boldsymbol{\theta}),$$
 (13.7)

where *q*ˆ is some learned approximation to *q*⇡. This algorithm, which has been called an *all-actions* method because its update involves all of the actions, is promising and deserving of further study, but our current interest is the classical REINFORCE algorithm (Willams, 1992) whose update at time t involves just  $A_t$ , the one action actually taken at time t.

We continue our derivation of REINFORCE by introducing  $A_t$  in the same way as we introduced  $S_t$  in (13.6)—by replacing a sum over the random variable's possible values by an expectation under  $\pi$ , and then sampling the expectation. Equation (13.6) involves an appropriate sum over actions, but each term is not weighted by  $\pi(a|S_t, \theta)$  as is needed for an expectation under  $\pi$ . So we introduce such a weighting, without changing the equality, by multiplying and then dividing the summed terms by  $\pi(a|S_t, \theta)$ . Continuing from (13.6), we have

$$\nabla J(\boldsymbol{\theta}) \propto \mathbb{E}_{\pi} \left[ \sum_{a} \pi(a|S_{t}, \boldsymbol{\theta}) q_{\pi}(S_{t}, a) \frac{\nabla \pi(a|S_{t}, \boldsymbol{\theta})}{\pi(a|S_{t}, \boldsymbol{\theta})} \right]$$

$$= \mathbb{E}_{\pi} \left[ q_{\pi}(S_{t}, A_{t}) \frac{\nabla \pi(A_{t}|S_{t}, \boldsymbol{\theta})}{\pi(A_{t}|S_{t}, \boldsymbol{\theta})} \right] \qquad \text{(replacing } a \text{ by the sample } A_{t} \sim \pi)$$

$$= \mathbb{E}_{\pi} \left[ G_{t} \frac{\nabla \pi(A_{t}|S_{t}, \boldsymbol{\theta})}{\pi(A_{t}|S_{t}, \boldsymbol{\theta})} \right], \qquad \text{(because } \mathbb{E}_{\pi}[G_{t}|S_{t}, A_{t}] = q_{\pi}(S_{t}, A_{t}))$$

where  $G_t$  is the return as usual. The final expression in brackets is exactly what is needed, a quantity that can be sampled on each time step whose expectation is proportional to the gradient. Using this sample to instantiate our generic stochastic gradient ascent algorithm (13.1) yields the REINFORCE update:

<span id="page-348-0"></span>
$$\boldsymbol{\theta}_{t+1} \doteq \boldsymbol{\theta}_t + \alpha G_t \frac{\nabla \pi(A_t | S_t, \boldsymbol{\theta}_t)}{\pi(A_t | S_t, \boldsymbol{\theta}_t)}.$$
(13.8)

This update has an intuitive appeal. Each increment is proportional to the product of a return  $G_t$  and a vector, the gradient of the probability of taking the action actually taken divided by the probability of taking that action. The vector is the direction in parameter space that most increases the probability of repeating the action  $A_t$  on future visits to state  $S_t$ . The update increases the parameter vector in this direction proportional to the return, and inversely proportional to the action probability. The former makes sense because it causes the parameter to move most in the directions that favor actions that yield the highest return. The latter makes sense because otherwise actions that are selected frequently are at an advantage (the updates will be more often in their direction) and might win out even if they do not yield the highest return.

Note that REINFORCE uses the complete return from time t, which includes all future rewards up until the end of the episode. In this sense REINFORCE is a Monte Carlo algorithm and is well defined only for the episodic case with all updates made in retrospect after the episode is completed (like the Monte Carlo algorithms in Chapter 5). This is shown explicitly in the boxed algorithm on the next page.

Notice that the update in the last line of pseudocode appears rather different from the REINFORCE update rule (13.8). One difference is that the pseudocode uses the compact expression  $\nabla \ln \pi(A_t|S_t, \boldsymbol{\theta}_t)$  for the fractional vector  $\frac{\nabla \pi(A_t|S_t, \boldsymbol{\theta}_t)}{\pi(A_t|S_t, \boldsymbol{\theta}_t)}$  in (13.8). That these two expressions for the vector are equivalent follows from the identity  $\nabla \ln x = \frac{\nabla x}{x}$ .

This vector has been given several names and notations in the literature; we will refer to it simply as the *eligibility vector*. Note that it is the only place that the policy parameterization appears in the algorithm.

```
REINFORCE: Monte-Carlo Policy-Gradient Control (episodic) for \pi_*

Input: a differentiable policy parameterization \pi(a|s, \boldsymbol{\theta})
Algorithm parameter: step size \alpha > 0
Initialize policy parameter \boldsymbol{\theta} \in \mathbb{R}^{d'} (e.g., to \boldsymbol{0})

Loop forever (for each episode):

Generate an episode S_0, A_0, R_1, \ldots, S_{T-1}, A_{T-1}, R_T, following \pi(\cdot|\cdot, \boldsymbol{\theta})
Loop for each step of the episode t = 0, 1, \ldots, T-1:

G \leftarrow \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \gamma^t G \nabla \ln \pi(A_t|S_t, \boldsymbol{\theta})
```

The second difference between the pseudocode update and the REINFORCE update equation (13.8) is that the former includes a factor of  $\gamma^t$ . This is because, as mentioned earlier, in the text we are treating the non-discounted case ( $\gamma=1$ ) while in the boxed algorithms we are giving the algorithms for the general discounted case. All of the ideas go through in the discounted case with appropriate adjustments (including to the box on page 199) but involve additional complexity that distracts from the main ideas.

\*Exercise 13.2 Generalize the box on page 199, the policy gradient theorem (13.5), the proof of the policy gradient theorem (page 325), and the steps leading to the REINFORCE update equation (13.8), so that (13.8) ends up with a factor of  $\gamma^t$  and thus aligns with the general algorithm given in the pseudocode.

Figure 13.1 shows the performance of REINFORCE on the short-corridor gridworld from Example 13.1.

<span id="page-349-0"></span>Figure 13.1: REINFORCE on the short-corridor gridworld (Example 13.1). With a good step size, the total reward per episode approaches the optimal value of the start state.

As a stochastic gradient method, REINFORCE has good theoretical convergence properties. By construction, the expected update over an episode is in the same direction as the performance gradient. This assures an improvement in expected performance for suciently small ↵, and convergence to a local optimum under standard stochastic approximation conditions for decreasing ↵. However, as a Monte Carlo method REINFORCE may be of high variance and thus produce slow learning.

*Exercise 13.3* In [Section 13.1](#page-343-1) we considered policy parameterizations using the soft-max in action preferences [\(13.2\)](#page-343-2) with linear action preferences [\(13.3\)](#page-343-0). For this parameterization, prove that the eligibility vector is

$$\nabla \ln \pi(a|s, \boldsymbol{\theta}) = \mathbf{x}(s, a) - \sum_{b} \pi(b|s, \boldsymbol{\theta}) \mathbf{x}(s, b), \tag{13.9}$$

using the definitions and elementary calculus. ⇤

### <span id="page-350-2"></span>13.4 REINFORCE with Baseline

The policy gradient theorem [\(13.5\)](#page-347-1) can be generalized to include a comparison of the action value to an arbitrary *baseline b*(*s*):

<span id="page-350-0"></span>
$$\nabla J(\boldsymbol{\theta}) \propto \sum_{s} \mu(s) \sum_{a} \left( q_{\pi}(s, a) - b(s) \right) \nabla \pi(a|s, \boldsymbol{\theta}).$$
 (13.10)

The baseline can be any function, even a random variable, as long as it does not vary with *a*; the equation remains valid because the subtracted quantity is zero:

$$\sum_a b(s) \nabla \pi(a|s, \boldsymbol{\theta}) \ = \ b(s) \nabla \sum_a \pi(a|s, \boldsymbol{\theta}) \ = \ b(s) \nabla 1 \ = \ 0.$$

The policy gradient theorem with baseline [\(13.10\)](#page-350-0) can be used to derive an update rule using similar steps as in the previous section. The update rule that we end up with is a new version of REINFORCE that includes a general baseline:

<span id="page-350-1"></span>
$$\boldsymbol{\theta}_{t+1} \doteq \boldsymbol{\theta}_t + \alpha \Big( G_t - b(S_t) \Big) \frac{\nabla \pi(A_t | S_t, \boldsymbol{\theta}_t)}{\pi(A_t | S_t, \boldsymbol{\theta}_t)}.$$
(13.11)

Because the baseline could be uniformly zero, this update is a strict generalization of REINFORCE. In general, the baseline leaves the expected value of the update unchanged, but it can have a large e↵ect on its variance. For example, we saw in [Section 2.8](#page-58-0) that an analogous baseline can significantly reduce the variance (and thus speed the learning) of gradient bandit algorithms. In the bandit algorithms the baseline was just a number (the average of the rewards seen so far), but for MDPs the baseline should vary with state. In some states all actions have high values and we need a high baseline to di↵erentiate the higher valued actions from the less highly valued ones; in other states all actions will have low values and a low baseline is appropriate.

One natural choice for the baseline is an estimate of the state value, *v*ˆ(*St,*w), where <sup>w</sup> <sup>2</sup> <sup>R</sup>*<sup>d</sup>* is a weight vector learned by one of the methods presented in previous chapters. Because REINFORCE is a Monte Carlo method for learning the policy parameter,  $\theta$ , it seems natural to also use a Monte Carlo method to learn the state-value weights,  $\mathbf{w}$ . A complete pseudocode algorithm for REINFORCE with baseline using such a learned state-value function as the baseline is given in the box below.

```
REINFORCE with Baseline (episodic), for estimating \pi_{\theta} \approx \pi_*

Input: a differentiable policy parameterization \pi(a|s,\theta)
Input: a differentiable state-value function parameterization \hat{v}(s,\mathbf{w})
Algorithm parameters: step sizes \alpha^{\theta} > 0, \alpha^{\mathbf{w}} > 0
Initialize policy parameter \theta \in \mathbb{R}^{d'} and state-value weights \mathbf{w} \in \mathbb{R}^{d} (e.g., to \mathbf{0})

Loop forever (for each episode):

Generate an episode S_0, A_0, R_1, \dots, S_{T-1}, A_{T-1}, R_T, following \pi(\cdot|\cdot,\theta)
Loop for each step of the episode t = 0, 1, \dots, T-1:

G \leftarrow \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k
\delta \leftarrow G - \hat{v}(S_t, \mathbf{w})
\mathbf{w} \leftarrow \mathbf{w} + \alpha^{\mathbf{w}} \delta \nabla \hat{v}(S_t, \mathbf{w})
\mathbf{w} \leftarrow \mathbf{w} + \alpha^{\mathbf{w}} \delta \nabla \hat{v}(S_t, \mathbf{w})
\mathbf{w} \leftarrow \mathbf{w} + \alpha^{\mathbf{w}} \delta \nabla \hat{v}(S_t, \mathbf{w})
```

This algorithm has two step sizes, denoted  $\alpha^{\theta}$  and  $\alpha^{\mathbf{w}}$  (where  $\alpha^{\theta}$  is the  $\alpha$  in (13.11)). Choosing the step size for values (here  $\alpha^{\mathbf{w}}$ ) is relatively easy; in the linear case we have rules of thumb for setting it, such as  $\alpha^{\mathbf{w}} = 0.1/\mathbb{E}[\|\nabla \hat{v}(S_t, \mathbf{w})\|_{\mu}^2]$  (see Section 9.6). It is much less clear how to set the step size for the policy parameters,  $\alpha^{\theta}$ , whose best value depends on the range of variation of the rewards and on the policy parameterization.

<span id="page-351-0"></span>Figure 13.2: Adding a baseline to REINFORCE can make it learn much faster, as illustrated here on the short-corridor gridworld (Example 13.1). The step size used here for plain REINFORCE is that at which it performs best (to the nearest power of two; see Figure 13.1).

[Figure 13.2](#page-351-0) compares the behavior of REINFORCE with and without a baseline on the short-corridor gridword [\(Example 13.1\)](#page-343-0). Here the approximate state-value function used in the baseline is ˆ*v*(*s,*w) = *w*. That is, w is a single component, *w*.

### <span id="page-352-1"></span>13.5 Actor–Critic Methods

In REINFORCE with baseline, the learned state-value function estimates the value of the *first* state of each state transition. This estimate sets a baseline for the subsequent return, but is made prior to the transition's action and thus cannot be used to assess that action. In actor–critic methods, on the other hand, the state-value function is applied also to the *second* state of the transition. The estimated value of the second state, when discounted and added to the reward, constitutes the one-step return, *Gt*:*t*+1, which is a useful estimate of the actual return and thus *is* a way of assessing the action. As we have seen in the TD learning of value functions throughout this book, the one-step return is often superior to the actual return in terms of its variance and computational congeniality, even though it introduces bias. We also know how we can flexibly modulate the extent of the bias with *n*-step returns and eligibility traces (Chapters [7](#page-162-0) and [12\)](#page-308-0). When the state-value function is used to assess actions in this way it is called a *critic*, and the overall policy-gradient method is termed an *actor–critic* method. Note that the bias in the gradient estimate is not due to bootstrapping as such; the actor would be biased even if the critic was learned by a Monte Carlo method.

First consider one-step actor–critic methods, the analog of the TD methods introduced in [Chapter 6](#page-140-0) such as TD(0), Sarsa(0), and Q-learning. The main appeal of one-step methods is that they are fully online and incremental, yet avoid the complexities of eligibility traces. They are a special case of the eligibility trace methods, but easier to understand. One-step actor–critic methods replace the full return of REINFORCE [\(13.11\)](#page-350-1) with the one-step return (and use a learned state-value function as the baseline) as follows:

$$\boldsymbol{\theta}_{t+1} \doteq \boldsymbol{\theta}_t + \alpha \Big( G_{t:t+1} - \hat{v}(S_t, \mathbf{w}) \Big) \frac{\nabla \pi(A_t | S_t, \boldsymbol{\theta}_t)}{\pi(A_t | S_t, \boldsymbol{\theta}_t)}$$
(13.12)

<span id="page-352-0"></span>
$$= \boldsymbol{\theta}_t + \alpha \left( R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w}) \right) \frac{\nabla \pi(A_t | S_t, \boldsymbol{\theta}_t)}{\pi(A_t | S_t, \boldsymbol{\theta}_t)}$$
(13.13)

$$= \boldsymbol{\theta}_t + \alpha \delta_t \frac{\nabla \pi(A_t | S_t, \boldsymbol{\theta}_t)}{\pi(A_t | S_t, \boldsymbol{\theta}_t)}.$$
 (13.14)

The natural state-value-function learning method to pair with this is semi-gradient TD(0). Pseudocode for the complete algorithm is given in the box at the top of the next page. Note that it is now a fully online, incremental algorithm, with states, actions, and rewards processed as they occur and then never revisited.

```
One-step Actor-Critic (episodic), for estimating \pi_{\theta} \approx \pi_*
Input: a differentiable policy parameterization \pi(a|s,\theta)
Input: a differentiable state-value function parameterization \hat{v}(s, \mathbf{w})
Parameters: step sizes \alpha^{\theta} > 0, \alpha^{\mathbf{w}} > 0
Initialize policy parameter \theta \in \mathbb{R}^{d'} and state-value weights \mathbf{w} \in \mathbb{R}^{d} (e.g., to 0)
Loop forever (for each episode):
    Initialize S (first state of episode)
    I \leftarrow 1
    Loop while S is not terminal (for each time step):
         A \sim \pi(\cdot|S, \boldsymbol{\theta})
         Take action A, observe S', R
         \delta \leftarrow R + \gamma \hat{v}(S', \mathbf{w}) - \hat{v}(S, \mathbf{w})
                                                                     (if S' is terminal, then \hat{v}(S',\mathbf{w}) \doteq 0)
         \mathbf{w} \leftarrow \mathbf{w} + \alpha^{\mathbf{w}} \delta \nabla \hat{v}(S, \mathbf{w})
         \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}} I \delta \nabla \ln \pi(A|S, \boldsymbol{\theta})
         I \leftarrow \gamma I
          S \leftarrow S'
```

The generalizations to the forward view of n-step methods and then to a  $\lambda$ -return algorithm are straightforward. The one-step return in (13.12) is merely replaced by  $G_{t:t+n}$  or  $G_t^{\lambda}$  respectively. The backward view of the  $\lambda$ -return algorithm is also straightforward, using separate eligibility traces for the actor and critic, each after the patterns in Chapter 12. Pseudocode for the complete algorithm is given in the box below.

```
Actor-Critic with Eligibility Traces (episodic), for estimating \pi_{\theta} \approx \pi_*
Input: a differentiable policy parameterization \pi(a|s, \theta)
Input: a differentiable state-value function parameterization \hat{v}(s,\mathbf{w})
Parameters: trace-decay rates \lambda^{\theta} \in [0,1], \lambda^{\mathbf{w}} \in [0,1]; step sizes \alpha^{\theta} > 0, \alpha^{\mathbf{w}} > 0
Initialize policy parameter \boldsymbol{\theta} \in \mathbb{R}^{d'} and state-value weights \mathbf{w} \in \mathbb{R}^{d} (e.g., to 0)
Loop forever (for each episode):
     Initialize S (first state of episode)
     \mathbf{z}^{\boldsymbol{\theta}} \leftarrow \mathbf{0} \ (d'-component eligibility trace vector)
     \mathbf{z}^{\mathbf{w}} \leftarrow \mathbf{0} (d-component eligibility trace vector)
     I \leftarrow 1
     Loop while S is not terminal (for each time step):
           A \sim \pi(\cdot|S,\boldsymbol{\theta})
           Take action A, observe S', R
                                                                                 (if S' is terminal, then \hat{v}(S', \mathbf{w}) \doteq 0)
           \delta \leftarrow R + \gamma \hat{v}(S', \mathbf{w}) - \hat{v}(S, \mathbf{w})
           \mathbf{z}^{\mathbf{w}} \leftarrow \gamma \lambda^{\mathbf{w}} \mathbf{z}^{\mathbf{w}} + \nabla \hat{v}(S, \mathbf{w})
           \mathbf{z}^{\boldsymbol{\theta}} \leftarrow \gamma \lambda^{\boldsymbol{\theta}} \mathbf{z}^{\boldsymbol{\theta}} + I \nabla \ln \pi(A|S, \boldsymbol{\theta})
           \mathbf{w} \leftarrow \mathbf{w} + \alpha^{\mathbf{w}} \delta \mathbf{z}^{\mathbf{w}}
           \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}} \delta \mathbf{z}^{\boldsymbol{\theta}}
           I \leftarrow \gamma I
           S \leftarrow S'
```

### 13.6 Policy Gradient for Continuing Problems

As discussed in Section 10.3, for continuing problems without episode boundaries we need to define performance in terms of the average rate of reward per time step:

<span id="page-354-0"></span>
$$J(\boldsymbol{\theta}) \doteq r(\pi) \doteq \lim_{h \to \infty} \frac{1}{h} \sum_{t=1}^{h} \mathbb{E}[R_t \mid S_0, A_{0:t-1} \sim \pi]$$

$$= \lim_{t \to \infty} \mathbb{E}[R_t \mid S_0, A_{0:t-1} \sim \pi]$$

$$= \sum_{s} \mu(s) \sum_{a} \pi(a|s) \sum_{s',r} p(s', r|s, a)r,$$
(13.15)

where  $\mu$  is the steady-state distribution under  $\pi$ ,  $\mu(s) \doteq \lim_{t\to\infty} \Pr\{S_t = s | A_{0:t} \sim \pi\}$ , which is assumed to exist and to be independent of  $S_0$  (an ergodicity assumption). Remember that this is the special distribution under which, if you select actions according to  $\pi$ , you remain in the same distribution:

<span id="page-354-1"></span>
$$\sum_{s} \mu(s) \sum_{a} \pi(a|s, \boldsymbol{\theta}) p(s'|s, a) = \mu(s'), \text{ for all } s' \in \mathcal{S}.$$
 (13.16)

Complete pseudocode for the actor–critic algorithm in the continuing case (backward view) is given in the box below.

# Actor-Critic with Eligibility Traces (continuing), for estimating $\pi_{\theta} \approx \pi_*$ Input: a differentiable policy parameterization $\pi(a|s,\theta)$ Input: a differentiable state-value function parameterization $\hat{v}(s,\mathbf{w})$ Algorithm parameters: $\lambda^{\mathbf{w}} \in [0,1], \ \lambda^{\theta} \in [0,1], \ \alpha^{\mathbf{w}} > 0, \ \alpha^{\theta} > 0, \ \alpha^{\bar{R}} > 0$ Initialize $\bar{R} \in \mathbb{R}$ (e.g., to 0) Initialize state-value weights $\mathbf{w} \in \mathbb{R}^d$ and policy parameter $\theta \in \mathbb{R}^{d'}$ (e.g., to 0) Initialize $S \in \mathcal{S}$ (e.g., to $s_0$ ) $\mathbf{z}^{\mathbf{w}} \leftarrow \mathbf{0}$ (d-component eligibility trace vector) $\mathbf{z}^{\theta} \leftarrow \mathbf{0}$ (d-component eligibility trace vector) Loop forever (for each time step): $A \sim \pi(\cdot|S,\theta)$ Take action A, observe S',R $\delta \leftarrow R - \bar{R} + \hat{v}(S',\mathbf{w}) - \hat{v}(S,\mathbf{w})$ $\bar{R} \leftarrow \bar{R} + \alpha^{\bar{R}}\delta$ $\mathbf{z}^{\mathbf{w}} \leftarrow \lambda^{\mathbf{w}}\mathbf{z}^{\mathbf{w}} + \nabla \hat{v}(S,\mathbf{w})$ $\mathbf{z}^{\theta} \leftarrow \lambda^{\theta}\mathbf{z}^{\theta} + \nabla \ln \pi(A|S,\theta)$ $\mathbf{w} \leftarrow \mathbf{w} + \alpha^{\mathbf{w}}\delta\mathbf{z}^{\mathbf{w}}$ $\theta \leftarrow \theta + \alpha^{\theta}\delta\mathbf{z}^{\theta}$ $S \leftarrow S'$

Naturally, in the continuing case, we define values,  $v_{\pi}(s) \doteq \mathbb{E}_{\pi}[G_t|S_t=s]$  and  $q_{\pi}(s,a) \doteq \mathbb{E}_{\pi}[G_t|S_t=s, A_t=a]$ , with respect to the differential return:

<span id="page-355-0"></span>
$$G_t \doteq R_{t+1} - r(\pi) + R_{t+2} - r(\pi) + R_{t+3} - r(\pi) + \cdots$$
 (13.17)

With these alternate definitions, the policy gradient theorem as given for the episodic case (13.5) remains true for the continuing case. A proof is given in the box on the next page. The forward and backward view equations also remain the same.

Proof of the Policy Gradient Theorem (continuing case)

<span id="page-355-1"></span>The proof of the policy gradient theorem for the continuing case begins similarly to the episodic case. Again we leave it implicit in all cases that  $\pi$  is a function of  $\boldsymbol{\theta}$  and that the gradients are with respect to  $\boldsymbol{\theta}$ . Recall that in the continuing case  $J(\boldsymbol{\theta}) = r(\pi)$  (13.15) and that  $v_{\pi}$  and  $q_{\pi}$  denote values with respect to the differential return (13.17). The gradient of the state-value function can be written, for any  $s \in \mathcal{S}$ , as

$$\nabla v_{\pi}(s) = \nabla \left[ \sum_{a} \pi(a|s) q_{\pi}(s, a) \right], \quad \text{for all } s \in \mathbb{S}$$

$$= \sum_{a} \left[ \nabla \pi(a|s) q_{\pi}(s, a) + \pi(a|s) \nabla q_{\pi}(s, a) \right] \quad \text{(product rule of calculus)}$$

$$= \sum_{a} \left[ \nabla \pi(a|s) q_{\pi}(s, a) + \pi(a|s) \nabla \sum_{s', r} p(s', r|s, a) (r - r(\boldsymbol{\theta}) + v_{\pi}(s')) \right]$$

$$= \sum_{a} \left[ \nabla \pi(a|s) q_{\pi}(s, a) + \pi(a|s) \left[ -\nabla r(\boldsymbol{\theta}) + \sum_{s'} p(s'|s, a) \nabla v_{\pi}(s') \right] \right].$$

After re-arranging terms, we obtain

$$\nabla r(\boldsymbol{\theta}) = \sum_{a} \left[ \nabla \pi(a|s) q_{\pi}(s,a) + \pi(a|s) \sum_{s'} p(s'|s,a) \nabla v_{\pi}(s') \right] - \nabla v_{\pi}(s).$$

Notice that the left-hand side can be written  $\nabla J(\theta)$ , and that it does not depend on s. Thus the right-hand side does not depend on s either, and we can safely sum it over all  $s \in \mathcal{S}$ , weighted by  $\mu(s)$ , without changing it (because  $\sum_{s} \mu(s) = 1$ ):

$$\nabla J(\boldsymbol{\theta}) = \sum_{s} \mu(s) \left( \sum_{a} \left[ \nabla \pi(a|s) q_{\pi}(s, a) + \pi(a|s) \sum_{s'} p(s'|s, a) \nabla v_{\pi}(s') \right] - \nabla v_{\pi}(s) \right)$$

$$= \sum_{s} \mu(s) \sum_{a} \nabla \pi(a|s) q_{\pi}(s, a)$$

$$+ \sum_{s} \mu(s) \sum_{a} \pi(a|s) \sum_{s'} p(s'|s, a) \nabla v_{\pi}(s') - \sum_{s} \mu(s) \nabla v_{\pi}(s)$$

$$= \sum_{s} \mu(s) \sum_{a} \nabla \pi(a|s) q_{\pi}(s, a)$$

$$+ \sum_{s'} \underbrace{\sum_{s} \mu(s) \sum_{a} \pi(a|s) p(s'|s, a)}_{\mu(s') \text{ (13.16)}} \nabla v_{\pi}(s') - \sum_{s} \mu(s) \nabla v_{\pi}(s)$$

$$= \sum_{s} \mu(s) \sum_{a} \nabla \pi(a|s) q_{\pi}(s, a) + \sum_{s'} \mu(s') \nabla v_{\pi}(s') - \sum_{s} \mu(s) \nabla v_{\pi}(s)$$

$$= \sum_{s} \mu(s) \sum_{a} \nabla \pi(a|s) q_{\pi}(s, a). \qquad Q.E.D.$$

### <span id="page-356-0"></span>13.7 Policy Parameterization for Continuous Actions

Policy-based methods offer practical ways of dealing with large action spaces, even continuous spaces with an infinite number of actions. Instead of computing learned probabilities for each of the many actions, we instead learn statistics of the probability distribution. For example, the action set might be the real numbers, with actions chosen from a normal (Gaussian) distribution.

The probability density function for the normal distribution is conventionally written

$$p(x) \doteq \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right),\tag{13.18}$$

where  $\mu$  and  $\sigma$  here are the mean and standard deviation of the normal distribution, and of course  $\pi$  here is just the number  $\pi \approx 3.14159$ . The probability density functions for several different means and standard deviations are shown to the right. The value p(x) is the density of the probability at x, not the probability of that x are under x that x is the total area under x that must sum to 1. In general, one can take the integral under x for any range of x values to get the probability of x falling within that range.

<span id="page-356-2"></span><span id="page-356-1"></span>To produce a policy parameterization, the policy can be defined as the normal probability density over a real-valued scalar action, with mean and standard deviation given by parametric function approximators that depend on the state. That is,

$$\pi(a|s,\boldsymbol{\theta}) \doteq \frac{1}{\sigma(s,\boldsymbol{\theta})\sqrt{2\pi}} \exp\left(-\frac{(a-\mu(s,\boldsymbol{\theta}))^2}{2\sigma(s,\boldsymbol{\theta})^2}\right),\tag{13.19}$$

where  $\mu: \mathbb{S} \times \mathbb{R}^{d'} \to \mathbb{R}$  and  $\sigma: \mathbb{S} \times \mathbb{R}^{d'} \to \mathbb{R}^+$  are two parameterized function approximators.

To complete the example we need only give a form for these approximators. For this we divide the policy's parameter vector into two parts,  $\boldsymbol{\theta} = [\boldsymbol{\theta}_{\mu}, \boldsymbol{\theta}_{\sigma}]^{\top}$ , one part to be used for the approximation of the mean and one part for the approximation of the standard deviation. The mean can be approximated as a linear function. The standard deviation must always be positive and is better approximated as the exponential of a linear function. Thus

<span id="page-357-0"></span>
$$\mu(s, \boldsymbol{\theta}) \doteq \boldsymbol{\theta}_{\mu}^{\top} \mathbf{x}_{\mu}(s) \quad \text{and} \quad \sigma(s, \boldsymbol{\theta}) \doteq \exp(\boldsymbol{\theta}_{\sigma}^{\top} \mathbf{x}_{\sigma}(s)),$$
 (13.20)

where  $\mathbf{x}_{\mu}(s)$  and  $\mathbf{x}_{\sigma}(s)$  are state feature vectors perhaps constructed by one of the methods described in Section 9.5. With these definitions, all the algorithms described in the rest of this chapter can be applied to learn to select real-valued actions.

Exercise 13.4 Show that for the Gaussian policy parameterization (Equations 13.19 and 13.20) the eligibility vector has the following two parts:

$$\nabla \ln \pi(a|s, \boldsymbol{\theta}_{\mu}) = \frac{\nabla \pi(a|s, \boldsymbol{\theta}_{\mu})}{\pi(a|s, \boldsymbol{\theta})} = \frac{1}{\sigma(s, \boldsymbol{\theta})^2} (a - \mu(s, \boldsymbol{\theta})) \mathbf{x}_{\mu}(s), \text{ and}$$

$$\nabla \ln \pi(a|s, \boldsymbol{\theta}_{\sigma}) = \frac{\nabla \pi(a|s, \boldsymbol{\theta}_{\sigma})}{\pi(a|s, \boldsymbol{\theta})} = \left(\frac{\left(a - \mu(s, \boldsymbol{\theta})\right)^{2}}{\sigma(s, \boldsymbol{\theta})^{2}} - 1\right) \mathbf{x}_{\sigma}(s).$$

Exercise 13.5 A Bernoulli-logistic unit is a stochastic neuron-like unit used in some ANNs (Section 9.7). Its input at time t is a feature vector  $\mathbf{x}(S_t)$ ; its output,  $A_t$ , is a random variable having two values, 0 and 1, with  $\Pr\{A_t = 1\} = P_t$  and  $\Pr\{A_t = 0\} = 1 - P_t$  (the Bernoulli distribution). Let  $h(s,0,\boldsymbol{\theta})$  and  $h(s,1,\boldsymbol{\theta})$  be the preferences in state s for the unit's two actions given policy parameter  $\boldsymbol{\theta}$ . Assume that the difference between the action preferences is given by a weighted sum of the unit's input vector, that is, assume that  $h(s,1,\boldsymbol{\theta}) - h(s,0,\boldsymbol{\theta}) = \boldsymbol{\theta}^{\top}\mathbf{x}(s)$ , where  $\boldsymbol{\theta}$  is the unit's weight vector.

- (a) Show that if the exponential soft-max distribution (13.2) is used to convert action preferences to policies, then  $P_t = \pi(1|S_t, \boldsymbol{\theta}_t) = 1/(1 + \exp(-\boldsymbol{\theta}_t^{\top} \mathbf{x}(S_t)))$  (the logistic function).
- (b) What is the Monte-Carlo REINFORCE update of  $\theta_t$  to  $\theta_{t+1}$  upon receipt of return  $G_t$ ?
- (c) Express the eligibility  $\nabla \ln \pi(a|s, \boldsymbol{\theta})$  for a Bernoulli-logistic unit, in terms of a,  $\mathbf{x}(s)$ , and  $\pi(a|s, \boldsymbol{\theta})$  by calculating the gradient.

Hint for part (c): Define  $P = \pi(1|s, \boldsymbol{\theta})$  and compute the derivative of the logarithm, for each action, using the chain rule on P. Combine the two results into one expression that depends on a and P, and then use the chain rule again, this time on  $\boldsymbol{\theta}^{\top}\mathbf{x}(s)$ , noting that the derivative of the logistic function  $f(x) = 1/(1 + e^{-x})$  is f(x)(1 - f(x)).

*13.8. Summary 337*

### 13.8 Summary

Prior to this chapter, this book focused on *action-value methods*—meaning methods that learn action values and then use them to determine action selections. In this chapter, on the other hand, we considered methods that learn a parameterized policy that enables actions to be taken without consulting action-value estimates. In particular, we have considered *policy-gradient methods*—meaning methods that update the policy parameter on each step in the direction of an estimate of the gradient of performance with respect to the policy parameter.

Methods that learn and store a policy parameter have many advantages. They can learn specific probabilities for taking the actions. They can learn appropriate levels of exploration and approach deterministic policies asymptotically. They can naturally handle continuous action spaces. All these things are easy for policy-based methods but awkward or impossible for "-greedy methods and for action-value methods in general. In addition, on some problems the policy is just simpler to represent parametrically than the value function; these problems are more suited to parameterized policy methods.

Parameterized policy methods also have an important theoretical advantage over action-value methods in the form of the *policy gradient theorem*, which gives an exact formula for how performance is a↵ected by the policy parameter that does not involve derivatives of the state distribution. This theorem provides a theoretical foundation for all policy gradient methods.

The REINFORCE method follows directly from the policy gradient theorem. Adding a state-value function as a *baseline* reduces REINFORCE's variance without introducing bias. If the state-value function is also used to assess—or criticize—the policy's action selections, then the value function is called a *critic* and the policy is called an *actor* ; the overall method is called an *actor–critic* method. The critic introduces bias into the actor's gradient estimates, but is often desirable for the same reason that bootstrapping TD methods are often superior to Monte Carlo methods (substantially reduced variance).

Overall, policy-gradient methods provide a significantly di↵erent set of strengths and weaknesses than action-value methods. Today they are less well understood in some respects, but a subject of excitement and ongoing research.

### Bibliographical and Historical Remarks

Methods that we now see as related to policy gradients were actually some of the earliest to be studied in reinforcement learning (Witten, 1977; Barto, Sutton, and Anderson, 1983; Sutton, 1984; Williams, 1987, 1992) and in predecessor fields (see Phansalkar and Thathachar, 1995). They were largely supplanted in the 1990s by the action-value methods that are the focus of the other chapters of this book. In recent years, however, attention has returned to actor–critic methods and to policy-gradient methods in general. Among the further developments beyond what we cover here are natural-gradient methods (Amari, 1998; Kakade, 2002, Peters, Vijayakumar and Schaal, 2005; Peters and Schaal, 2008; Park, Kim and Kang, 2005; Bhatnagar, Sutton, Ghavamzadeh and Lee, 2009; see Grondman, Busoniu, Lopes and Babuska, 2012), deterministic policy-gradient methods

(Silver et al., 2014), o↵-policy policy-gradient methods (Degris, White, and Sutton, 2012; Maei, 2018), and entropy regularization (see Schulman, Chen, and Abbeel, 2017). Major applications include acrobatic helicopter autopilots and AlphaGo [\(Section 16.6\)](#page-462-0).

Our presentation in this chapter is based primarily on that by Sutton, McAllester, Singh, and Mansour (2000), who introduced the term "policy gradient methods." A useful overview is provided by Bhatnagar et al. (2009). One of the earliest related works is by Aleksandrov, Sysoyev, and Shemeneva (1968). Thomas (2014) first realized that the factor of *<sup>t</sup>* , as specified in the boxed algorithms of this chapter, was needed in the case of discounted episodic problems.

- [13.1](#page-343-1) [Example 13.1](#page-343-0) and the results with it in this chapter were developed with Eric Graves.
- [13.2](#page-345-0) The policy gradient theorem here and on [page 334](#page-355-1) was first obtained by Marbach and Tsitsiklis (1998, 2001) and then independently by Sutton et al. (2000). A similar expression was obtained by Cao and Chen (1997). Other early results are due to Konda and Tsitsiklis (2000, 2003), Baxter and Bartlett (2001), and Baxter, Bartlett, and Weaver (2001). Some additional results are developed by Sutton, Singh, and McAllester (2000).
- [13.3](#page-347-2) REINFORCE is due to Williams (1987, 1992). Phansalkar and Thathachar (1995) proved both local and global convergence theorems for modified versions of REINFORCE algorithms.
  - The all-actions algorithm was first presented in an unpublished but widely circulated incomplete paper (Sutton, Singh, and McAllester, 2000) and then developed further by Ciosek and Whiteson (2017, 2018), who termed it "expected policy gradients," and by Asadi, Allen, Roderick, Mohamed, Konidaris, and Littman (2017), who called it "mean actor critic."
- [13.4](#page-350-2) The baseline was introduced in Williams's (1987, 1992) original work. Greensmith, Bartlett, and Baxter (2004) analyzed an arguably better baseline (see Dick, 2015). Thomas and Brunskill (2017) argue that an action-dependent baseline can be used without incurring bias.
- [13.5–](#page-352-1)6 Actor–critic methods were among the earliest to be investigated in reinforcement learning (Witten, 1977; Barto, Sutton, and Anderson, 1983; Sutton, 1984). The algorithms presented here are based on the work of Degris, White, and Sutton (2012). Actor–critic methods are sometimes referred to as advantage actor–critic ("A2C") methods in the literature.
- [13.7](#page-356-0) The first to show how continuous actions could be handled this way appears to have been Williams (1987, 1992). The figure on [page 335](#page-356-2) is adapted from Wikipedia.

### <span id="page-490-0"></span>17.4 Designing Reward Signals

A major advantage of reinforcement learning over supervised learning is that reinforcement learning does not rely on detailed instructional information: generating a reward signal does not depend on knowledge of what the agent's correct actions should be. But the success of a reinforcement learning application strongly depends on how well the reward signal frames the goal of the application's designer and how well the signal assesses progress in reaching that goal. For these reasons, designing a reward signal is a critical part of any application of reinforcement learning.

By designing a reward signal we mean designing the part of an agent's environment that is responsible for computing each scalar reward *R<sup>t</sup>* and sending it to the agent at each time *t*. In our discussion of terminology at the end of [Chapter 14,](#page-362-0) we said that *R<sup>t</sup>* is more like a signal generated inside an animal's brain than it is like an object or event in the animal's external environment. The parts of our brains that generate these signals for us evolved over millions of years to be well suited to the challenges our ancestors had to face in their struggles to propagate their genes to future generations. We should therefore not think that designing a good reward signal is always an easy thing to do!

One challenge is to design a reward signal so that as an agent learns, its behavior approaches, and ideally eventually achieves, what the application's designer actually desires. This can be easy if the designer's goal is simple and easy to identify, such as finding the solution to a well-defined problem or earning a high score in a well-defined game. In cases like these, it is usual to reward the agent according to its success in solving the problem or its success in improving its score. But some problems involve goals that are dicult to translate into reward signals. This is especially true when the problem requires the agent to skillfully perform a complex task or set of tasks, such as would be required of a useful household robotic assistant. Further, reinforcement learning agents can discover unexpected ways to make their environments deliver reward, some of which might be undesirable, or even dangerous. This is a longstanding and critical challenge for any method, like reinforcement learning, that is based on optimization. We discuss this issue more in [Section 17.6,](#page-496-0) the final section of this book.

Even when there is a simple and easily identifiable goal, the problem of *sparse reward* often arises. Delivering non-zero reward frequently enough to allow the agent to achieve the goal once, let alone to learn to achieve it eciently from multiple initial conditions, can be a daunting challenge. State–action pairs that clearly deserve to trigger reward may be few and far between, and rewards that mark progress toward a goal can be infrequent because progress is dicult or even impossible to detect. The agent may wander aimlessly for long periods of time (what Minsky, 1961, called the "plateau problem").

In practice, designing a reward signal is often left to an informal trial-and-error search for a signal that produces acceptable results. If the agent fails to learn, learns too slowly, or learns the wrong thing, then the designer tweaks the reward signal and tries again. To do this, the designer judges the agent's performance by criteria that he or she is attempting to translate into a reward signal so that the agent's goal matches his or her own. And if learning is too slow, the designer may try to design a non-sparse reward signal that e↵ectively guides learning throughout the agent's interaction with its environment. It is tempting to address the sparse reward problem by rewarding the agent for achieving subgoals that the designer thinks are important way stations to the overall goal. But augmenting the reward signal with well-intentioned supplemental rewards may lead the agent to behave differently from what is intended; the agent may end up not achieving the overall goal. A better way to provide such guidance is to leave the reward signal alone and instead augment the value-function approximation with an initial guess of what it should ultimately be, or augment it with initial guesses as to what certain parts of it should be. For example, suppose we wants to offer  $v_0: \mathcal{S} \to \mathbb{R}$  as an initial guess at the true optimal value function  $v_*$ , and that we are using linear function approximation with features  $\mathbf{x}: \mathcal{S} \to \mathbb{R}^d$ . Then we would define the initial value function approximation as

<span id="page-491-0"></span>
$$\hat{v}(s, \mathbf{w}) \doteq \mathbf{w}^{\top} \mathbf{x}(s) + v_0(s), \tag{17.11}$$

and update the weights  $\mathbf{w}$  as usual. If the initial weight vector is  $\mathbf{0}$ , then the initial value function will be  $v_0$ , but the asymptotic solution quality will be determined by the feature vectors as usual. This initialization can also be done for arbitrary nonlinear approximators and arbitrary forms of  $v_0$ , though it is not guaranteed to always accelerate learning.

A particularly effective approach to the sparse reward problem is the *shaping* technique introduced by the psychologist B. F. Skinner and described in Section 14.3. The effectiveness of this technique relies on the fact that sparse reward problems are not just problems with the reward signal; they are also problems with an agent's policy in that it prevents the agent from frequently encountering rewarding states. Shaping involves changing the reward signal as learning proceeds, starting from a reward signal that is not sparse given the agent's initial behavior, and gradually modifying it toward a reward signal suited to the problem of original interest. Shaping might also involve modifying the dynamics of the task as learning proceeds. Each modification is made so that the agent is frequently rewarded given its current behavior. The agent faces a sequence of increasingly-difficult reinforcement learning problems, where what is learned at each stage makes the next-harder problem relatively easy because the agent now encounters reward more frequently than it would if it did not have prior experience with easier problems. This kind of shaping is an essential technique in training animals, and it is effective in computational reinforcement learning as well.

What if you have no idea what the rewards should be, but there is another agent, perhaps a person, who is already expert at the task and whose behavior can be observed? In this case you could use methods known variously as "imitation learning," "learning from demonstration," and "apprenticeship learning." The idea here is to benefit from the expert agent but leave open the possibility of eventually performing better. Learning from an expert's behavior can be done either by learning directly by supervised learning or by extracting a reward signal using what is known as "inverse reinforcement learning" and then using a reinforcement learning algorithm with that reward signal to learn a policy. The task of inverse reinforcement learning as explored by Ng and Russell (2000) is to try to recover the expert's reward signal from the expert's behavior alone. This cannot be done exactly because a policy can be optimal with respect to many different reward signals (for example, all policies are optimal with respect to a constant reward signal), but it is possible to find plausible reward signal candidates. Unfortunately, strong

assumptions are required, including knowledge of the environment's dynamics and of the feature vectors in which the reward signal is linear. The method also requires completely solving the problem (e.g., by dynamic programming methods) multiple times. These diculties notwithstanding, Abbeel and Ng (2004) argue that the inverse reinforcement learning approach can sometimes be more e↵ective than supervised learning for benefiting from the behavior of an expert.

Another approach to finding a good reward signal is to automate the trial-and-error search for a good signal that we mentioned above. From an application perspective, the reward signal is a parameter of the learning algorithm. As is true for other algorithm parameters, the search for a good reward signal can be automated by defining a space of feasible candidates and applying an optimization algorithm. The optimization algorithm evaluates each candidate reward signal by running the reinforcement learning system with that signal for some number of steps, and then scoring the overall result by a "high-level" objective function intended to encode the designer's true goal, ignoring the limitations of the agent. Reward signals can even be improved via online gradient ascent, where the gradient is that of the high-level objective function (Sorg, Lewis, and Singh, 2010). Relating this approach to the natural world, the algorithm for optimizing the high-level objective function is analogous to evolution, where the high-level objective function is an animal's evolutionary fitness determined by the number of its o↵spring that survive to reproductive age.

Computational experiments with this bilevel optimization approach—one level analogous to evolution, and the other due to reinforcement learning by individual agents—have confirmed that intuition alone is not always adequate to devise a good reward signal (Singh, Lewis, and Barto, 2009). The performance of a reinforcement learning agent as evaluated by the high-level objective function can be very sensitive to details of the agent's reward signal in subtle ways determined by the agent's limitations and the environment in which it acts and learns. These experiments have also demonstrated that an agent's goal should not always be the same as the goal of the agent's designer.

At first this seems counterintuitive, but it may be impossible for the agent to achieve the designer's goal no matter what its reward signal is. The agent has to learn under various kinds of constraints, such as limited computational power, limited access to information about its environment, or limited time to learn. When there are constraints like these, learning to achieve a goal that is di↵erent from the designer's goal can sometimes end up getting closer to the designer's goal than if that goal were pursued directly (Sorg, Singh, and Lewis, 2010; Sorg, 2011). Examples of this in the natural world are easy to find. Because we cannot directly assess the nutritional value of most foods, evolution—the designer of our reward signal—gave us a reward signal that makes us seek certain tastes. Though certainly not infallible (indeed, possibly detrimental in environments that di↵er in certain ways from ancestral environments), this compensates for many of our limitations: our limited sensory abilities, the limited time over which we can learn, and the risks involved in finding a healthy diet through personal experimentation. Similarly, because an animal cannot always observe its own evolutionary fitness, that objective function does not work as a reward signal for learning. Evolution instead provides reward signals that are sensitive to observable predictors of evolutionary fitness.

Finally, remember that a reinforcement learning agent is not necessarily like a complete

### <span id="page-496-0"></span>17.6 Reinforcement Learning and the Future of Artificial Intelligence

When we were writing the first edition of this book in the mid-1990s, artificial intelligence was making significant progress and was having an impact on society, though it was mostly still the *promise* of artificial intelligence that was inspiring developments. Machine learning was part of that outlook, but it had not yet become indispensable to artificial intelligence. By today that promise has transitioned to applications that are changing the lives of millions of people, and machine learning has come into its own as a key technology. As we write this second edition, some of the most remarkable developments in artificial intelligence have involved reinforcement learning, most notably "deep reinforcement learning"—reinforcement learning with function approximation by deep artificial neural networks. We are at the beginning of a wave of real-world applications of artificial intelligence, many of which will include reinforcement learning, deep and otherwise, that will impact our lives in ways that are hard to predict.

But an abundance of successful real-world applications does not mean that true artificial intelligence has arrived. Despite great progress in many areas, the gulf between artificial intelligence and the intelligence of humans, and other animals, remains great. Superhuman performance can be achieved in some domains, even formidable domains like Go, but it remains a significant challenge to develop systems that are like us in being complete, interactive agents having general adaptability and problem-solving skills, emotional sophistication, creativity, and the ability to learn quickly from experience. With its focus on learning by interacting with dynamic environments, reinforcement learning, as it develops over the future, will be a critical component of agents with these abilities.

Reinforcement learning's connections to psychology and neuroscience (Chapters [14](#page-362-0) and [15\)](#page-398-0) underscore its relevance to another longstanding goal of artificial intelligence: shedding light on fundamental questions about the mind and how it emerges from the brain. Reinforcement learning theory is already contributing to our understanding of the brain's reward, motivation, and decision-making processes, and there is good reason to believe that through its links to computational psychiatry, reinforcement learning theory will contribute to methods for treating mental disorders, including drug abuse and addiction.

Another contribution that reinforcement learning can make over the future is as an aid to human decision making. Policies derived by reinforcement learning in simulated environments can advise human decision makers in such areas as education, healthcare, transportation, energy, and public-sector resource allocation. Particularly relevant is the key feature of reinforcement learning that it takes long-term consequences of decisions into account. This is very clear in games like backgammon and Go, where some of the most impressive results of reinforcement learning have been demonstrated, but it is also a property of many high-stakes decisions that a↵ect our lives and our planet. Reinforcement learning follows related methods for advising human decision making that have been developed in the past by decision analysts in many disciplines. With advanced function approximation methods and massive computational power, reinforcement learning

methods have the potential to overcome some of the diculties of scaling up traditional decision-support methods to larger and more complex problems.

The rapid pace of advances in artificial intelligence has led to warnings that artificial intelligence poses serious threats to our societies, even to humanity itself. The renowned scientist and artificial intelligence pioneer Herbert Simon anticipated the warnings we are hearing today in a presentation at the Earthware Symposium at CMU in 2000 (Simon, 2000). He spoke of the eternal conflict between the promise and perils of any new knowledge, reminding us of the Greek myths of Prometheus, the idealized hero of modern science, who stole fire from the gods for the benefit of mankind, and of Pandora, whose mythical box could be opened by a small and innocent action to release untold perils on the world. While accepting that this conflict is inevitable, Simon urged us to recognize that as designers of our future and not mere spectators, the decisions *we* make can tilt the scale in Prometheus' favor. This is certainly true for reinforcement learning, which can benefit society but can also produce undesirable outcomes if it is carelessly deployed. Thus, the *safety* of artificial intelligence applications involving reinforcement learning is a topic that deserves careful attention.

A reinforcement learning agent can learn by interacting with either the real world or with a simulation of some piece of the real world, or by a mixture of these two sources of experience. Simulators provide safe environments in which an agent can explore and learn without risking real damage to itself or to its environment. In most current applications, policies are learned from simulated experience instead of direct interaction with the real world. In addition to avoiding undesirable real-world consequences, learning from simulated experience can make virtually unlimited data available for learning, generally at less cost than needed to obtain real experience, and because simulations typically run much faster than real time, learning can often occur more quickly than if it relied on real experience.

Nevertheless, the full potential of reinforcement learning requires reinforcement learning agents to be embedded into the flow of real-world experience, where they act, explore, and learn in *our* world, and not just in *their* worlds. After all, reinforcement learning algorithms—at least those upon which we focus in this book—are designed to learn online, and they emulate many aspects of how animals are able to survive in nonstationary and hostile environments. Embedding reinforcement learning agents in the real world can be transformative in realizing the promises of artificial intelligence to amplify and extend human abilities.

A major reason for wanting a reinforcement learning agent to act and learn in the real world is that it is often dicult, sometimes impossible, to simulate real-world experience with enough fidelity to make the resulting policies, whether derived by reinforcement learning or by other methods, work well—and safely—when directing real actions. This is especially true for environments whose dynamics depend on the behavior of humans, such as in education, healthcare, transportation, and public policy—domains that can surely benefit from improved decision making. However, it is for real-world embedded agents that warnings about potential dangers of artificial intelligence need to be heeded.

Some of these warnings are particularly relevant to reinforcement learning. Because reinforcement learning is based on optimization, it inherits the plusses and minuses of all optimization methods. On the minus side is the problem of devising objective functions,

or reward signals in the case of reinforcement learning, so that optimization produces the desired results while avoiding undesirable results. We said in [Section 17.4](#page-490-0) that reinforcement learning agents can discover unexpected ways to make their environments deliver reward, some of which might be undesirable, or even dangerous. When we specify what we want a system to learn only indirectly, as we do in designing a reinforcement learning system's reward signal, we will not know how closely the agent will fulfill our desire until its learning is complete. This is hardly a new problem with reinforcement learning; recognition of it has a long history in both literature and engineering. For example, in Goethe's poem "The Sorcerer's Apprentice" (Goethe, 1878), the apprentice uses magic to enchant a broom to do his job of fetching water, but the result is an unintended flood due to the apprentice's inadequate knowledge of magic. In the engineering context, Norbert Wiener, the founder of cybernetics, warned of this problem more than half a century ago by relating the supernatural story of "The Monkey's Paw" (Wiener, 1964): "... it grants what you ask for, not what you should have asked for or what you intend" (p. 59). The problem has also been discussed at length in a modern context by Nick Bostrom (2014). Anyone having experience with reinforcement learning has likely seen their systems discover unexpected ways to obtain a lot of reward. Sometimes the unexpected behavior is good: it solves a problem in a nice new way. In other instances, what the agent learns violates considerations that the system designer may never have thought about. Careful design of reward signals is essential if an agent is to act in the real world with no opportunity for human vetting of its actions or means to easily interrupt its behavior.

Despite the possibility of unintended negative consequences, optimization has been used for hundreds of years by engineers, architects, and others whose designs have positively impacted the world. We owe much that is good in our environment to the application of optimization methods. Many approaches have been developed to mitigate the risk of optimization, such as adding hard and soft constraints, restricting optimization to robust and risk-sensitive policies, and optimizing with multiple objective functions. Some of these approaches have been adapted to reinforcement learning, and more research is needed to address these concerns. The problem of ensuring that a reinforcement learning agent's goal is attuned to our own remains a challenge.

Another challenge if reinforcement learning agents are to act and learn in the real world is not just about what they might learn *eventually*, but about how they will behave while they are learning. How do you make sure that an agent gets enough experience to learn a high-performing policy, all the while not harming its environment, other agents, or itself (or more realistically, while keeping the probability of harm acceptably low)? This problem is also not novel or unique to reinforcement learning. Risk management and mitigation for embedded reinforcement learning is similar to what control engineers have had to confront from the beginning of using automatic control in situations where a controller's behavior can have unacceptable, possibly catastrophic, consequences, as in the control of an aircraft or a delicate chemical process. Control applications rely on careful system modeling, model validation, and extensive testing, and there is a highly-developed body of theory aimed at ensuring convergence and stability of adaptive controllers designed for use when the dynamics of the system to be controlled are not fully known. Theoretical guarantees are never iron-clad because they depend on the validity of the assumptions underlying the mathematics, but without this theory, combined with risk-management

and mitigation practices, automatic control—adaptive and otherwise—would not be as beneficial as it is today in improving the quality, eciency, and cost-e↵ectiveness of processes on which we have come to rely. One of the most pressing areas for future reinforcement learning research is to adapt and extend methods developed in control engineering with the goal of making it acceptably safe to fully embed reinforcement learning agents into physical environments.

In closing, we return to Simon's call for us to recognize that we are designers of our future and not simply spectators. By decisions we make as individuals, and by the influence we can exert on how our societies are governed, we can work toward ensuring that the benefits made possible by a new technology outweigh the harm it can cause. There is ample opportunity to do this in the case of reinforcement learning, which can help improve the quality, fairness, and sustainability of life on our planet, but which can also release new perils. A threat already here is the displacement of jobs caused by applications of artificial intelligence. Still there are good reasons to believe that the benefits of artificial intelligence can outweigh the disruption it causes. As to safety, hazards possible with reinforcement learning are not completely di↵erent from those that have been managed successfully for related applications of optimization and control methods. As reinforcement learning moves out into the real world in future applications, developers have an obligation to follow best practices that have evolved for similar technologies, while at the same time extending them to make sure that Prometheus keeps the upper hand.

### Bibliographical and Historical Remarks

[17.1](#page-480-1) General value functions were first explicitly identified by Sutton and colleagues (Sutton, 1995a; Sutton et al., 2011; Modayil, White, and Sutton, 2013). Ring (in preparation) developed an extensive thought experiment with GVFs ("forecasts") that has been influential despite not yet having been published.

The first demonstrations of multi-headed learning in reinforcement learning were by Jaderberg et al. (2017). Bellemare, Dabney, and Munos (2017) showed that predicting more things about the distribution of reward could significantly accelerate learning to optimize its expectation, an instance of auxiliary tasks. Many others have since taken up this line of research.

The general theory of classical conditioning as learned predictions together with built-in, reflexive reactions to the predictions has not to our knowledge been clearly articulated in the psychological literature. Modayil and Sutton (2014) describe it as an approach to the engineering of robots and other agents, calling it "Pavlovian control" to allude to its roots in classical conditioning.

[17.2](#page-482-0) The formalization of temporally abstract courses of action as options was introduced by Sutton, Precup, and Singh (1999), building on prior work by Parr (1998) and Sutton (1995a), and on classical work on Semi-MDPs (e.g., see Puterman, 1994). Precup's (2000) PhD thesis developed option ideas fully. An important limitation of these early works is that they did not treat the o↵-policy case

with function approximation. Intra-option learning in general requires o↵-policy learning, which could not be done reliably with function approximation at that time. Although now we have a variety of stable o↵-policy learning methods using function approximation, their combination with option ideas had not been significantly explored at the time of publication of this book. Barto and Mahadevan (2003) and Hengst (2012) review the options formalism and other approaches to temporal abstraction.

Using GVFs to implement option models has not previously been described. Our presentation uses the trick introduced by Modayil, White, and Sutton (2014) for predicting signals at the termination of policies.

Among the few works that have learned option models with function approximation are those by Sorg and Singh (2010), and by Bacon, Harb, and Precup (2017).

The extension of options and option models to the average-reward setting has not yet been developed in the literature.

[17.3](#page-485-0) A good presentation of the POMDP approach is given by Monahan (1982). PSRs and tests were introduced by Littman, Sutton, and Singh (2002). OOMs were introduced by Jaeger (1997, 1998, 2000). Sequential Systems, which unify PSRs, OOMs, and many other works, were introduced in the PhD thesis of Michael Thon (2017; Thon and Jaeger, 2015). Extensions to networks of temporal relationships were developed by Tanner (2006; Sutton and Tanner, 2005) and then extended to options (Sutton, Rafols, and Koop, 2006).

The theory of reinforcement learning with a non-Markov state representation was developed explicitly by Singh, Jaakkola, and Jordan (1994; Jaakkola, Singh, and Jordan, 1995). Early reinforcement learning approaches to partial observability were developed by Chrisman (1992), McCallum (1993, 1995), Parr and Russell (1995), Littman, Cassandra, and Kaelbling (1995), and by Lin and Mitchell (1992).

[17.4](#page-490-0) Early e↵orts to include advice and teaching in reinforcement learning include those by Lin (1992), Maclin and Shavlik (1994), Clouse (1996), and Clouse and Utgo↵ (1992).

Skinner's shaping should not be confused with the "potential-based shaping" technique introduced by Ng, Harada, and Russell (1999). Their technique has been shown by Wiewiora (2003) to be equivalent to the simpler idea of providing an initial approximation to the value function, as in [\(17.11\)](#page-491-0).

[17.5](#page-493-0) We recommend the book by Goodfellow, Bengio, and Courville (2016) for discussion of today's deep learning techniques. The problem of catastrophic interference in ANNs was developed by McCloskey and Cohen (1989), Ratcli↵ (1990), and French (1999). The idea of a replay bu↵er was introduced by Lin (1992) and used prominently in deep learning in the Atari game playing system [\(Section 16.5,](#page-457-0) Mnih et al., 2013, 2015).

Minsky (1961) was one of the first to identify the problem of representation learning.

Among the few works to consider planning with learned, approximate models are those by Kuvayev and Sutton (1996), Sutton, Szepesvari, Geramifard, and Bowling (2008), Nouri and Littman (2009), and Hester and Stone (2012).

The need to be selective in model construction to avoid slowing planning is well known in artificial intelligence. Some of the classic work is by Minton (1990) and Tambe, Newell, and Rosenbloom (1990). Hauskrecht, Meuleau, Kaelbling, Dean, and Boutilier (1998) showed this e↵ect in MDPs with deterministic options.

Schmidhuber (1991a, b) proposed how something like curiosity would result if reward signals were a function of how quickly an agent's environment model is improving. The empowerment function proposed by Klyubin, Polani, and Nehaniv (2005) is an information-theoretic measure of an agent's ability to control its environment that can function as an intrinsic reward signal. Baldassarre and Mirolli (2013) is a collection of contributions by researchers studying intrinsic reward and motivation from both biological and computational perspectives, including a perspective on "intrinsically-motivated reinforcement learning," to use the term introduced by Singh, Barto, and Chentenez (2004). See also Oudeyer and Kaplan (2007), Oudeyer, Kaplan, and Hafner (2007), and Barto (2013).
