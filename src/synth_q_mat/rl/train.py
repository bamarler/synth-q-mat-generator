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
    print("[train] TODO: build env over MatterGen + run PPO (weeks 3-4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
