from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from synth_q_mat.config import load_config
from synth_q_mat.models.evaluator import build_evaluator
from synth_q_mat.models.generator import MatterGenGenerator
from synth_q_mat.rl.reward import RewardWeights, compute_reward, formation_to_score
from synth_q_mat.rl.reward.composition import within_z_range


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def main(argv: list[str] | None = None) -> int:
    cfg = load_config(overrides=argv if argv is not None else sys.argv[1:])
    comp, gen_cfg, rcfg = cfg["composition"], cfg["generate"], cfg["reward"]
    weights = RewardWeights.from_config(rcfg)
    cutoff = rcfg["formation_energy_cutoff"]

    generator = MatterGenGenerator.from_config(cfg)
    evaluator = build_evaluator(cfg)
    structures = generator.generate(gen_cfg["batch_size"], gen_cfg["num_batches"])

    # Light-element hard gate: heavy-element structures get the floor reward (0)
    # and are not scored. MatterGen base has no element constraint; a true hard
    # constraint would need a fine-tuned adapter (M4).
    light = [within_z_range(s, comp["min_z"], comp["max_z"]) for s in structures]
    light_idx = [i for i, ok in enumerate(light) if ok]
    ehull = np.full(len(structures), np.nan)
    if light_idx:
        scored = evaluator.score([structures[i] for i in light_idx])
        for i, e in zip(light_idx, scored, strict=True):
            ehull[i] = e

    candidates = []
    for s, ok, e in zip(structures, light, ehull, strict=True):
        if ok:
            s_stab = formation_to_score(float(e), cutoff)
            reward = compute_reward({evaluator.objective: s_stab}, weights)
        else:
            s_stab = reward = 0.0
        candidates.append(
            {
                "formula": s.composition.reduced_formula,
                "n_atoms": len(s),
                "light": ok,
                "e_above_hull": None if np.isnan(e) else float(e),
                "s_stab": s_stab,
                "reward": reward,
                "cif": s.to(fmt="cif"),
            }
        )

    rewards = np.array([c["reward"] for c in candidates])
    metrics = {
        "stage": "baseline",
        "n": len(candidates),
        "reward_mean": float(rewards.mean()) if len(rewards) else 0.0,
        "reward_max": float(rewards.max()) if len(rewards) else 0.0,
        "light_fraction": float(np.mean(light)) if light else 0.0,
        "ehull_best": float(np.nanmin(ehull)) if light_idx else None,
        "on_hull_fraction": (
            float(np.nansum(ehull <= 0) / len(light_idx)) if light_idx else 0.0
        ),
    }

    # Durable outputs first — a down tracking server must never lose a run.
    results = Path(cfg["paths"]["results"])
    (results / "metrics").mkdir(parents=True, exist_ok=True)
    (results / "candidates.json").write_text(json.dumps(candidates, indent=2))
    (results / "metrics" / "baseline.json").write_text(json.dumps(metrics) + "\n")
    print(f"[baseline] {metrics}")

    _log_mlflow(cfg, metrics)
    return 0


def _log_mlflow(cfg: dict[str, Any], metrics: dict[str, Any]) -> None:
    import mlflow

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    keep = {k: cfg[k] for k in ("reward", "generate", "composition", "models", "seed")}
    try:
        mlflow.set_experiment(cfg["logging"]["mlflow_experiment"])
        with mlflow.start_run(run_name="baseline"):
            mlflow.log_params(_flatten(keep))
            if sha:
                mlflow.log_param("git_commit", sha)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
    except Exception as e:  # noqa: BLE001 — tracking is best-effort
        print(f"[baseline] MLflow logging skipped ({e!r}); start `make mlflow-ui`.")


if __name__ == "__main__":
    raise SystemExit(main())
