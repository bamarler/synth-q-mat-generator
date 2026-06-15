"""synth-q-mat-generator: RL-guided discovery of topologically constrained
quantum materials.

Package layout:
    data/    dataset loaders and light-element filtering
    models/  thin wrappers around DiffCSP, ALIGNN, and M3GNet
    rl/      MatInvent PPO pipeline and the multi-objective reward
    eval/    baseline evaluation and Pareto-front analysis
    viz/     publication (matplotlib/seaborn) and interactive (plotly) figures
"""

__version__ = "0.1.0"
