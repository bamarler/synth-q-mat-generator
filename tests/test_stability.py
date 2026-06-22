from __future__ import annotations

from pymatgen.core import Composition

from synth_q_mat.models.stability import EnergyAboveHullEvaluator


def _evaluator():
    # One compound on the Li-Si line at -0.2 eV/atom; no terminals supplied.
    entries = [(frozenset({"Li", "Si"}), Composition("Li2Si"), -0.2)]
    return EnergyAboveHullEvaluator(entries, relax=False)


def test_terminals_injected_so_hull_builds():
    # mp_entries has no pure-element rows; hull must still build (terminals added).
    pd = _evaluator()._hull(frozenset({"Li", "Si"}))
    assert {el.symbol for el in pd.elements} == {"Li", "Si"}


def test_on_hull_scores_zero():
    ev = _evaluator()
    assert abs(ev._e_above_hull_for(Composition("Li2Si"), -0.2)) < 1e-6


def test_above_hull_positive():
    ev = _evaluator()
    # Same composition, worse (higher) formation energy -> above the hull.
    e = ev._e_above_hull_for(Composition("Li2Si"), -0.1)
    assert e > 0
    assert abs(e - 0.1) < 1e-6


def test_off_hull_composition_positive():
    ev = _evaluator()
    # A composition with no stabilizing compound, at 0 eV/atom, sits above the
    # tie-line pulled down by Li2Si.
    assert ev._e_above_hull_for(Composition("LiSi"), 0.0) > 0


def test_oxidation_states_stripped():
    # Species-decorated comp (as MP structures carry) must score like the plain one.
    ev = _evaluator()
    oxi = Composition({"Li+": 2, "Si4-": 1})
    assert abs(ev._e_above_hull_for(oxi, -0.2)) < 1e-6
