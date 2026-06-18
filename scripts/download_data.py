from __future__ import annotations

import argparse
import os
from pathlib import Path

# Importing the project config loads .env (so MP_API_KEY is available).
from synth_q_mat.config import load_config

RAW_DIR = Path("data/raw")

# Columns kept in the parquet snapshot (the mp-api cache holds every field).
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

# Where mp-api stores its DeltaTable-backed dataset cache.
MP_DATASET_DIR = (
    Path(os.getenv("MP_DATASET_CACHE", Path.home() / "mp_datasets"))
    / "build"
    / "collections"
    / "summary"
)


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
    stable_only: bool,
    max_results: int | None,
    refresh: bool,
) -> None:
    """Snapshot the Materials Project summary to data/raw/mp_summary.parquet."""
    if refresh or not MP_DATASET_DIR.exists():
        api_key = os.getenv("MP_API_KEY")
        if not api_key:
            print("[mp] ERROR: MP_API_KEY not set. Add it to .env (see .env.example).")
            return
        from mp_api.client import MPRester

        print(
            f"[mp] fetching summary collection (DeltaTable cache: {MP_DATASET_DIR})..."
        )
        with MPRester(api_key) as mpr:
            mpr.materials.summary.search()  # populates the local cache
    else:
        print(
            f"[mp] reusing cached DeltaTable at {MP_DATASET_DIR} (--refresh to re-fetch)"
        )

    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from deltalake import DeltaTable

    # Convert the DeltaTable cache directly; iterating mp-api's 163k pydantic
    # docs one-by-one is orders of magnitude slower.
    table = DeltaTable(str(MP_DATASET_DIR)).to_pyarrow_table()
    n_total = table.num_rows
    cols = [c for c in MP_FIELDS if c in table.column_names]
    missing = set(MP_FIELDS) - set(cols)
    if missing:
        print(f"[mp] WARNING: columns missing from cache, skipped: {sorted(missing)}")
    table = table.select(cols)
    if stable_only:
        table = table.filter(pc.field("is_stable"))
    if max_results:
        table = table.slice(0, max_results)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "mp_summary.parquet"
    pq.write_table(table, out)
    print(
        f"[mp] saved {out}: {table.num_rows}/{n_total} rows, "
        f"{len(cols)} columns, {out.stat().st_size / 1e6:.1f} MB"
    )


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
    parser = argparse.ArgumentParser(
        description="Download source databases (JARVIS, Materials Project, TQC) into data/raw/."
    )
    parser.add_argument(
        "--source",
        choices=["jarvis", "mp", "tqc", "all"],
        default="all",
        help="which source to download",
    )
    parser.add_argument(
        "--stable-only",
        action="store_true",
        help="Materials Project: keep only is_stable=True rows in the parquet",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Materials Project: cap number of saved rows (for testing)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Materials Project: re-fetch from the API even if a local cache exists",
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
        download_mp(args.stable_only, args.max_results, args.refresh)
    if args.source in ("tqc", "all"):
        write_tqc_note()

    print("\n[download_data] done. Version the raw data with:")
    print("    uv run dvc add data/raw && make push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
