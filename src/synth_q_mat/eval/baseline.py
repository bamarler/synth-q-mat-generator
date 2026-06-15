"""Baseline: evaluate unconditional DiffCSP generation (week 2).

Generates structures with no RL guidance and records hit rates for topology,
stability, and the combined reward. These numbers are the bar the trained
policy must beat.
"""

from __future__ import annotations

import sys

from synth_q_mat.config import load_config


def main(argv: list[str] | None = None) -> int:
    cfg = load_config(overrides=argv if argv is not None else sys.argv[1:])
    print("[baseline] config loaded")
    print(f"[baseline] generator : {cfg['models']['generator']}")
    print(
        f"[baseline] Z range    : {cfg['composition']['min_z']}-{cfg['composition']['max_z']}"
    )
    print("[baseline] TODO: sample N structures, score with ALIGNN + M3GNet (week 2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
