from __future__ import annotations

import json
import sys
from pathlib import Path

from synth_q_mat.config import load_config


def main(argv: list[str] | None = None) -> int:
    cfg = load_config(overrides=argv if argv is not None else sys.argv[1:])
    print("[baseline] config loaded")
    print(f"[baseline] generator : {cfg['models']['generator']}")
    print(
        f"[baseline] Z range    : {cfg['composition']['min_z']}-{cfg['composition']['max_z']}"
    )
    print("[baseline] TODO: sample N structures, score with spillage + M3GNet (week 2)")

    metric = Path(cfg["paths"]["results"]) / "metrics" / "baseline.json"
    metric.parent.mkdir(parents=True, exist_ok=True)
    metric.write_text(json.dumps({"stage": "baseline", "placeholder": True}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
