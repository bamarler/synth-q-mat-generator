from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

PRETRAINED_DIR = Path("models/pretrained")

# Pretrained model handle (verified against the installed matgl package).
M3GNET_EFORM = "M3GNet-Eform-MP-2018.6.1"  # formation energy (stability)


def fetch_m3gnet() -> str:
    """Download + load the M3GNet formation-energy model via matgl."""
    import matgl

    print(f"[m3gnet] loading {M3GNET_EFORM}...")
    model = matgl.load_model(M3GNET_EFORM)
    print(
        f"[m3gnet] OK: {type(model).__name__} (matgl {matgl.__version__}, fetched on demand)"
    )
    return f"M3GNet | {M3GNET_EFORM} | matgl {matgl.__version__} | fetched on demand"


def write_manifest(lines: list[str]) -> None:
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = PRETRAINED_DIR / "MODELS.md"
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    body = (
        "# Pretrained models\n\n"
        f"_Last verified: {stamp}_\n\n"
        "Weights re-fetch reproducibly from (package version + handle); they are\n"
        "not DVC-tracked. This file records what loaded successfully.\n\n"
        "| model | handle | package | source |\n"
        "| --- | --- | --- | --- |\n"
        + "\n".join(f"| {ln} |" for ln in lines)
        + "\n\n## Not pretrained (built later)\n\n"
        "- **Topology (spillage) predictor** — trained in the research phase on\n"
        "  JARVIS-DFT spillage labels with a DGL-free framework (matgl or MACE-MP).\n"
        "- **Generator (MatterGen)** — isolated env; see `scripts/setup_mattergen.sh`\n"
        "  and `models/pretrained/mattergen/`.\n"
    )
    manifest.write_text(body)
    print(f"[manifest] wrote {manifest}")


def main() -> int:
    argparse.ArgumentParser(
        description="Fetch + smoke-load the pretrained M3GNet stability model via matgl."
    ).parse_args()

    lines = [fetch_m3gnet()]
    write_manifest(lines)

    print("\n[download_models] done.")
    print("    (M3GNet re-fetches on demand; nothing to DVC-track here.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
