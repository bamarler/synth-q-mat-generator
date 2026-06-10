"""Entrypoint for MatInvent PPO training.

Scaffold only: wires config -> reward -> (eventually) the PPO loop over the
DiffCSP generator. Fill in the policy/env in weeks 3-4. Run with:

    make train ARGS="train.total_steps=200000 reward.w_topology=0.5"
"""

from __future__ import annotations

import sys

from synth_q_mat.config import load_config
from synth_q_mat.rl.reward import RewardWeights


def main(argv: list[str] | None = None) -> int:
    cfg = load_config(overrides=argv if argv is not None else sys.argv[1:])
    weights = RewardWeights.from_config(cfg["reward"])

    print("[train] config loaded")
    print(f"[train] algorithm     : {cfg['train']['algorithm']}")
    print(f"[train] total_steps   : {cfg['train']['total_steps']}")
    print(f"[train] reward weights: {weights}")
    print(f"[train] checkpoints   : {cfg['paths']['checkpoints']}")
    print("[train] TODO: build env over DiffCSP + run PPO (weeks 3-4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
