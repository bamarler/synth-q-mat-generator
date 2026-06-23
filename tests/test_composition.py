from __future__ import annotations

from pymatgen.core import Lattice, Structure

from synth_q_mat.rl.reward.composition import within_z_range


def _struct(symbol: str) -> Structure:
    return Structure(Lattice.cubic(3.0), [symbol], [[0, 0, 0]])


def test_light_element_passes():
    assert within_z_range(_struct("Si"), 3, 14)  # Z=14


def test_heavy_element_gated():
    assert not within_z_range(_struct("Fe"), 3, 14)  # Z=26
    # mixed: one heavy element fails the whole structure
    mixed = Structure(Lattice.cubic(4.0), ["Li", "W"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert not within_z_range(mixed, 3, 14)
