from __future__ import annotations

from pymatgen.core import Structure


def within_z_range(structure: Structure, min_z: int, max_z: int) -> bool:
    """True if every element's atomic number falls in [min_z, max_z] (light-element gate)."""
    return all(min_z <= el.Z <= max_z for el in structure.composition.elements)
