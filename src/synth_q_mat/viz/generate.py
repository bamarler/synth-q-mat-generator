from __future__ import annotations

import json
import sys
from pathlib import Path

from synth_q_mat.config import load_config


def main(argv: list[str] | None = None) -> int:
    cfg = load_config(overrides=argv if argv is not None else sys.argv[1:])
    results = Path(cfg["paths"]["results"])
    candidates = json.loads((results / "candidates.json").read_text())
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    rewards = [c["reward"] for c in candidates]
    ehull = [c["e_above_hull"] for c in candidates if c["e_above_hull"] is not None]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(rewards, bins=20, ax=axes[0])
    axes[0].set(title="Reward", xlabel="reward", ylabel="count")
    if ehull:
        sns.histplot(ehull, bins=20, ax=axes[1])
        axes[1].axvline(0, color="k", ls="--")
    axes[1].set(title="Energy above hull (light)", xlabel="eV/atom", ylabel="count")
    fig.tight_layout()
    out = figures / "baseline.png"
    fig.savefig(out, dpi=150)
    print(f"[viz] wrote {out} ({len(candidates)} candidates, {len(ehull)} scored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
