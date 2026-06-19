from __future__ import annotations

import json
import sys
from pathlib import Path

from synth_q_mat.config import load_config
from synth_q_mat.rl.reward import RewardWeights


def main(argv: list[str] | None = None) -> int:
    cfg = load_config(overrides=argv if argv is not None else sys.argv[1:])
    weights = RewardWeights.from_config(cfg["reward"])

    print("[train] config loaded")
    print(f"[train] algorithm     : {cfg['train']['algorithm']}")
    print(f"[train] total_steps   : {cfg['train']['total_steps']}")
    print(f"[train] reward weights: {weights}")
    print("[train] TODO: build env over MatterGen + run PPO (weeks 3-4)")

    # Scaffold outputs so the DVC stage produces its declared outs/metrics.
    ckpt = Path(cfg["paths"]["checkpoints"])
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "PLACEHOLDER").write_text("scaffold checkpoint dir\n")
    metric = Path(cfg["paths"]["results"]) / "metrics" / "train.json"
    metric.parent.mkdir(parents=True, exist_ok=True)
    metric.write_text(json.dumps({"stage": "train", "placeholder": True}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
