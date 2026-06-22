from __future__ import annotations

import numpy as np


def formation_to_score(e_above_hull: float, cutoff: float = 0.10) -> float:
    """Map energy above hull (eV/atom) to a [0, 1] stability score.

    s_stab = clip((E_cut - E_hull) / E_cut, 0, 1). On-hull (<=0) -> 1.0; at the
    synthesizability cutoff -> 0.0. E_hull def. from Bartel 2022; E_cut ~ 0.1
    eV/atom (or the system amorphous limit, Aykol 2018).
    """
    if e_above_hull <= 0:
        return 1.0
    return float(np.clip(1.0 - e_above_hull / cutoff, 0.0, 1.0))
