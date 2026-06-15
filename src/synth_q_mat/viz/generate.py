"""Generate figures from evaluated results (week 9).

Publication figures (matplotlib/seaborn): band structures with topological
features, phonon dispersions, multi-objective Pareto fronts, RL-vs-baseline
training curves. Interactive exploration (plotly): high-dimensional candidate
browsing.
"""

from __future__ import annotations

import sys

from synth_q_mat.config import load_config


def main(argv: list[str] | None = None) -> int:
    cfg = load_config(overrides=argv if argv is not None else sys.argv[1:])
    print("[viz] config loaded")
    print(f"[viz] reading results from: {cfg['paths']['results']}")
    print("[viz] TODO: render Pareto fronts, band structures, training curves (week 9)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
