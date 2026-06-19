from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = _ROOT / "conf"
PARAMS = _ROOT / "params.yaml"

# Load .env so entrypoints (download scripts, training) pick up
# MP_API_KEY / MLFLOW_TRACKING_URI / AWS creds automatically.
load_dotenv(_ROOT / ".env")


def load_config(overrides: list[str] | None = None) -> dict[str, Any]:
    """Config as a plain dict: DVC-composed params.yaml if present, else Hydra-composed conf/."""
    if not overrides and PARAMS.exists():
        cfg = OmegaConf.load(PARAMS)
    else:
        if os.getenv("SQMG_DVC") and not PARAMS.exists():
            raise RuntimeError(
                "SQMG_DVC set but params.yaml missing — run via `dvc repro` / `dvc exp run`."
            )
        with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
            cfg = compose(config_name="config", overrides=overrides or [])
    return OmegaConf.to_container(cfg, resolve=True)
