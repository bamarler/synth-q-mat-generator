"""Download source databases into data/raw/ (raw dumps only — no filtering).

Sources:
    - JARVIS-DFT (dft_3d): topological labels via spin-orbit `spillage`.
    - Materials Project: structures + formation energy / stability via mp-api.
    - Topological Quantum Chemistry (TQC): no programmatic API -> manual note.

Parsing, light-element filtering, and conversion to a clean dataset happen in a
later (research) phase. Here we just fetch and stage the raw data, then it gets
versioned with `dvc add data/raw && make push`.

Usage:
    make download-data ARGS="--source jarvis"
    make download-data ARGS="--source mp --elements Li,Si,O --stable-only"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Importing the project config loads .env (so MP_API_KEY is available).
from synth_q_mat.config import load_config

RAW_DIR = Path("data/raw")

# Default Materials Project field set — small, structure-bearing summary docs.
MP_FIELDS = [
    "material_id",
    "formula_pretty",
    "structure",
    "formation_energy_per_atom",
    "energy_above_hull",
    "band_gap",
    "is_stable",
    "symmetry",
    "nelements",
]


def _spillage_value(record: dict) -> float:
    """JARVIS stores spillage as a float or the string 'na'."""
    raw = record.get("spillage")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def download_jarvis(dataset: str = "dft_3d") -> None:
    """Fetch a JARVIS-DFT dataset; jarvis-tools caches the raw JSON in store_dir."""
    from jarvis.db.figshare import data

    out_dir = RAW_DIR / "jarvis"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[jarvis] downloading '{dataset}' into {out_dir} (this can take a while)...")
    records = data(dataset, store_dir=str(out_dir))
    n_topo = sum(1 for r in records if _spillage_value(r) > 0.25)
    print(
        f"[jarvis] {len(records)} records; ~{n_topo} with spillage > 0.25 "
        "(topological indicator; not filtered here)"
    )


def download_mp(
    elements: list[str] | None,
    stable_only: bool,
    max_results: int | None,
) -> None:
    """Query Materials Project summary docs; save raw to data/raw/mp_summary.json."""
    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        print("[mp] ERROR: MP_API_KEY not set. Add it to .env (see .env.example).")
        return

    from monty.json import jsanitize
    from mp_api.client import MPRester

    query: dict = {"fields": MP_FIELDS}
    if elements:
        query["elements"] = elements
    if stable_only:
        query["is_stable"] = True

    scope = {k: v for k, v in query.items() if k != "fields"}
    print(f"[mp] querying summary.search({scope})...")
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(**query)
    if max_results:
        docs = docs[:max_results]
    print(f"[mp] retrieved {len(docs)} documents")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "mp_summary.json"
    payload = jsanitize([d.model_dump() for d in docs], strict=True)
    with open(out, "w") as fh:
        json.dump(payload, fh)
    print(f"[mp] saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


def write_tqc_note() -> None:
    """TQC has no API; leave instructions for the manual web download."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    note = RAW_DIR / "TQC_README.md"
    note.write_text(
        "# Topological Quantum Chemistry (TQC)\n\n"
        "TQC has **no programmatic API** — access is via the web portal:\n"
        "  https://www.topologicalquantumchemistry.com/\n\n"
        "Export the classified materials (CIF/POSCAR) you need and place them in\n"
        "this directory under `tqc/`. They can then be read with\n"
        "`pymatgen.core.Structure.from_file(...)`.\n\n"
        "For programmatic topological labels, prefer the JARVIS-DFT `spillage`\n"
        "field (downloaded automatically by this script).\n"
    )
    print(f"[tqc] wrote manual-download instructions to {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["jarvis", "mp", "tqc", "all"],
        default="all",
        help="which source to download",
    )
    parser.add_argument(
        "--elements",
        type=lambda s: [e.strip() for e in s.split(",") if e.strip()],
        default=None,
        help="Materials Project: comma-separated elements that must be present",
    )
    parser.add_argument(
        "--stable-only",
        action="store_true",
        help="Materials Project: restrict to is_stable=True",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Materials Project: cap number of saved documents (for testing)",
    )
    args = parser.parse_args()

    # Touch config so .env is loaded and to surface the light-element scope.
    cfg = load_config()
    z = cfg["composition"]
    print(
        f"[download_data] light-element scope from config: Z {z['min_z']}-{z['max_z']}"
    )

    if args.source in ("jarvis", "all"):
        download_jarvis("dft_3d")
    if args.source in ("mp", "all"):
        download_mp(args.elements, args.stable_only, args.max_results)
    if args.source in ("tqc", "all"):
        write_tqc_note()

    print("\n[download_data] done. Version the raw data with:")
    print("    uv run dvc add data/raw && make push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
