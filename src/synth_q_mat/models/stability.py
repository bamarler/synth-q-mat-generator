from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Structure

# (element symbols, parsed composition, formation energy eV/atom) per MP entry.
MPEntry = tuple[frozenset[str], Composition, float]


class EnergyAboveHullEvaluator:
    """Energy above hull (eV/atom) from M3GNet-Eform + a Materials Project convex hull."""

    objective = "formation"

    def __init__(self, mp_entries: list[MPEntry], relax: bool = True) -> None:
        self._mp = mp_entries
        self.relax = relax
        self._hull_cache: dict[frozenset[str], PhaseDiagram] = {}
        self._model = None
        self._relaxer = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> EnergyAboveHullEvaluator:
        cache = Path(cfg["paths"]["data_processed"]) / "hull_eform.parquet"
        relax = cfg.get("generate", {}).get("relax", True)
        return cls.from_cache(cache, relax=relax)

    @classmethod
    def from_cache(cls, cache: Path, relax: bool = True) -> EnergyAboveHullEvaluator:
        # Hull built from M3GNet-Eform of MP structures (see data/hull_cache.py),
        # NOT MP's DFT formation energies — the candidate is scored by the same
        # model, so the reference cancels; mixing in MP/MP2020 energies does not.
        import pandas as pd

        df = pd.read_parquet(cache)
        entries: list[MPEntry] = []
        for formula, eform in zip(
            df["formula_pretty"], df["eform_per_atom"], strict=True
        ):
            if eform is None or np.isnan(eform):
                continue
            comp = Composition(formula)
            entries.append(
                (frozenset(el.symbol for el in comp.elements), comp, float(eform))
            )
        return cls(entries, relax=relax)

    def score(self, structures: list[Structure]) -> np.ndarray:
        return np.array([self._e_above_hull(s) for s in structures], dtype=float)

    def _hull(self, elements: frozenset[str]) -> PhaseDiagram:
        pd = self._hull_cache.get(elements)
        if pd is None:
            # Terminal references at formation energy 0 — PhaseDiagram requires
            # one per element and does not add them itself.
            pd_entries = [PDEntry(Composition(el), 0.0) for el in elements]
            for symbols, comp, eform in self._mp:
                if symbols <= elements:
                    pd_entries.append(PDEntry(comp, eform * comp.num_atoms))
            pd = PhaseDiagram(pd_entries)
            self._hull_cache[elements] = pd
        return pd

    def _e_above_hull_for(self, comp: Composition, eform_per_atom: float) -> float:
        comp = comp.element_composition  # drop oxidation states -> Element keys
        pd = self._hull(frozenset(el.symbol for el in comp.elements))
        # PDEntry energy is total, not per-atom.
        entry = PDEntry(comp, eform_per_atom * comp.num_atoms)
        # allow_negative: a below-hull candidate returns <0 (-> stability score 1).
        _, e = pd.get_decomp_and_e_above_hull(entry, allow_negative=True)
        return float(e)

    def _e_above_hull(self, structure: Structure) -> float:
        if self.relax:
            structure = self._relax_structure(structure)
        # ponytail: M3GNet-Eform <-> MP formation-energy reference is
        # close-but-not-identical; add MP2020 corrections only if accuracy bites.
        eform = float(self._eform().predict_structure(structure))
        return self._e_above_hull_for(structure.composition, eform)

    def _eform(self) -> Any:
        if self._model is None:
            import matgl

            self._model = matgl.load_model("M3GNet-Eform-MP-2018.6.1")
        return self._model

    def _relax_structure(self, structure: Structure) -> Structure:
        if self._relaxer is None:
            import matgl
            from matgl.ext.ase import Relaxer

            pot = matgl.load_model("M3GNet-MP-2021.2.8-DIRECT-PES")
            self._relaxer = Relaxer(potential=pot, relax_cell=True)
        return self._relaxer.relax(structure)["final_structure"]
