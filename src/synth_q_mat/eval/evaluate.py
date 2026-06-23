from __future__ import annotations

import json
import sys
from pathlib import Path

from synth_q_mat.config import load_config


def main(argv: list[str] | None = None) -> int:
    cfg = load_config(overrides=argv if argv is not None else sys.argv[1:])
    results = Path(cfg["paths"]["results"])
    candidates = json.loads((results / "candidates.json").read_text())

    ranked = sorted(candidates, key=lambda c: c["reward"], reverse=True)
    top = ranked[:10]
    metric = results / "metrics" / "eval.json"
    metric.parent.mkdir(parents=True, exist_ok=True)
    metric.write_text(
        json.dumps(
            {
                "stage": "eval",
                "n": len(candidates),
                "best_reward": top[0]["reward"] if top else 0.0,
                "top": [{"formula": c["formula"], "reward": c["reward"]} for c in top],
            }
        )
        + "\n"
    )
    print(
        f"[evaluate] ranked {len(candidates)}; best reward "
        f"{top[0]['reward'] if top else 0.0:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
