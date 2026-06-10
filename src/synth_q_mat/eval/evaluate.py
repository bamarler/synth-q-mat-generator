"""Evaluate and rank generated candidates; extract the Pareto front (weeks 5, 9)."""

from __future__ import annotations

import sys

from synth_q_mat.config import load_config


def main(argv: list[str] | None = None) -> int:
    cfg = load_config(overrides=argv if argv is not None else sys.argv[1:])
    print("[evaluate] config loaded")
    print(f"[evaluate] checkpoints: {cfg['paths']['checkpoints']}")
    print(f"[evaluate] results    : {cfg['paths']['results']}")
    print("[evaluate] TODO: rank candidates, compute Pareto front (weeks 5, 9)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
