from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from pymatgen.core import Structure


@runtime_checkable
class Evaluator(Protocol):
    """Scores structures on one physical quantity, feeding one reward objective."""

    objective: str  # which reward objective this raw quantity maps to

    def score(self, structures: list[Structure]) -> np.ndarray: ...


def build_evaluator(cfg: dict[str, Any]) -> Evaluator:
    name = cfg["models"].get("evaluator", "energy_above_hull")
    if name == "energy_above_hull":
        from synth_q_mat.models.stability import EnergyAboveHullEvaluator

        return EnergyAboveHullEvaluator.from_config(cfg)
    raise ValueError(f"unknown evaluator: {name!r}")
