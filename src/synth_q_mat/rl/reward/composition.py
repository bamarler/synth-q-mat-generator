from __future__ import annotations

import numpy as np
from pymatgen.core import Element, Structure


def within_z_range(structure: Structure, min_z: int, max_z: int) -> bool:
    """True if every element's atomic number falls in [min_z, max_z] (light-element gate)."""
    return all(min_z <= el.Z <= max_z for el in structure.composition.elements)


def sample_light_chemsys(
    rng: np.random.Generator, min_z: int, max_z: int, k: int
) -> str:
    """A '-'-joined chemical system of k distinct elements from [min_z, max_z].

    Noble gases are excluded — they form no compounds, so conditioning on them
    yields degenerate (single-element) systems.
    """
    els = [Element.from_Z(z) for z in range(min_z, max_z + 1)]
    symbols = [el.symbol for el in els if not el.is_noble_gas]
    chosen = rng.choice(symbols, size=k, replace=False)
    return "-".join(sorted(chosen))
