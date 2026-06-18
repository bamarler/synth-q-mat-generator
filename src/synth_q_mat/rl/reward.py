from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OBJECTIVES = ("topology", "dos_fermi", "formation", "composition")


@dataclass(frozen=True)
class RewardWeights:
    """Weights for each objective. Normalized to sum to 1 on construction."""

    w_topology: float = 0.40
    w_dos_fermi: float = 0.30
    w_formation: float = 0.20
    w_composition: float = 0.10

    @classmethod
    def from_config(cls, reward_cfg: dict[str, float]) -> RewardWeights:
        return cls(
            w_topology=float(reward_cfg.get("w_topology", 0.40)),
            w_dos_fermi=float(reward_cfg.get("w_dos_fermi", 0.30)),
            w_formation=float(reward_cfg.get("w_formation", 0.20)),
            w_composition=float(reward_cfg.get("w_composition", 0.10)),
        )

    def as_array(self) -> np.ndarray:
        raw = np.array(
            [self.w_topology, self.w_dos_fermi, self.w_formation, self.w_composition],
            dtype=float,
        )
        total = raw.sum()
        if total <= 0:
            raise ValueError("Reward weights must sum to a positive value")
        return raw / total


def formation_to_score(e_above_hull: float, cutoff: float = 0.10) -> float:
    """Map formation energy above hull (eV/atom) to a [0, 1] stability score.

    On-hull (<=0) scores 1.0; at the synthesizability cutoff it scores 0.0.
    """
    if e_above_hull <= 0:
        return 1.0
    return float(np.clip(1.0 - e_above_hull / cutoff, 0.0, 1.0))


def compute_reward(
    objectives: dict[str, float],
    weights: RewardWeights,
) -> float:
    """Weighted sum of the four objective scores (each must be in [0, 1]; missing -> 0.0)."""
    values = np.array([objectives.get(name, 0.0) for name in OBJECTIVES], dtype=float)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"Objective scores must be in [0, 1], got {objectives}")
    return float(values @ weights.as_array())
