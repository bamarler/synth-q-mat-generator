from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from pymatgen.core import Composition, Element, Structure

from synth_q_mat.config import load_config


def _normalize(obj: object) -> object:
    # Parquet round-trips nested lattice/coords as object ndarrays; pymatgen's
    # from_dict needs plain lists.
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    return obj


def load_light_mp(
    parquet: Path, min_z: int, max_z: int
) -> Iterator[tuple[str, Structure]]:
    """Yield (formula, Structure) for MP entries whose elements all fall in [min_z, max_z]."""
    import pandas as pd

    allowed = {Element.from_Z(z).symbol for z in range(min_z, max_z + 1)}
    df = pd.read_parquet(parquet, columns=["formula_pretty", "structure"])
    for formula, raw in zip(df["formula_pretty"], df["structure"], strict=True):
        try:
            comp = Composition(formula)
        except Exception:
            continue
        if {el.symbol for el in comp.elements} <= allowed:
            yield formula, Structure.from_dict(_normalize(raw))


def build(parquet: Path, out: Path, min_z: int, max_z: int) -> int:
    """Predict M3GNet-Eform for every light-element MP structure; cache to parquet."""
    import matgl
    import pandas as pd

    model = matgl.load_model("M3GNet-Eform-MP-2018.6.1")
    formulas, eforms = [], []
    for i, (formula, struct) in enumerate(load_light_mp(parquet, min_z, max_z)):
        try:
            eforms.append(float(model.predict_structure(struct)))
            formulas.append(formula)
        except Exception as e:  # noqa: BLE001 — skip the rare unparseable structure
            print(f"[hull-cache] skip {formula}: {e}")
        if (i + 1) % 500 == 0:
            print(f"[hull-cache] {i + 1} structures...")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"formula_pretty": formulas, "eform_per_atom": eforms}).to_parquet(out)
    print(f"[hull-cache] wrote {len(formulas)} entries -> {out}")
    return len(formulas)


def main(argv: list[str] | None = None) -> int:
    cfg = load_config(overrides=argv if argv is not None else sys.argv[1:])
    comp = cfg["composition"]
    parquet = Path(cfg["paths"]["data_raw"]) / "mp_summary.parquet"
    out = Path(cfg["paths"]["data_processed"]) / "hull_eform.parquet"
    build(parquet, out, comp["min_z"], comp["max_z"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
