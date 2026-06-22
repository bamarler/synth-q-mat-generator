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
