from __future__ import annotations

import sys
from pathlib import Path

from synth_q_mat.config import load_config


def main(argv: list[str] | None = None) -> int:
    cfg = load_config(overrides=argv if argv is not None else sys.argv[1:])
    figs = Path(cfg["paths"]["results"]) / "figures"
    print("[viz] config loaded")
    print(f"[viz] reading results from: {cfg['paths']['results']}")
    print("[viz] TODO: render Pareto fronts, band structures, training curves (week 9)")

    figs.mkdir(parents=True, exist_ok=True)
    (figs / "PLACEHOLDER").write_text("scaffold figures dir\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
