from __future__ import annotations

import numpy as np

from synth_q_mat.rl.reward.weights import OBJECTIVES, RewardWeights


def compute_reward(
    objectives: dict[str, float],
    weights: RewardWeights,
) -> float:
    """Weighted sum of the four objective scores (each must be in [0, 1]; missing -> 0.0)."""
    values = np.array([objectives.get(name, 0.0) for name in OBJECTIVES], dtype=float)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"Objective scores must be in [0, 1], got {objectives}")
    return float(values @ weights.as_array())
