from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
DEFAULTS = CONFIG_DIR / "defaults.yaml"

# Load .env from the project root so entrypoints (download scripts, training)
# pick up MP_API_KEY / MLFLOW_TRACKING_URI / AWS creds automatically.
load_dotenv(CONFIG_DIR.parent / ".env")


def _coerce(value: str) -> Any:
    """Turn a CLI string into a bool/int/float/list where possible."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _apply_override(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot override into non-mapping at '{key}'")
    node[keys[-1]] = value


def load_config(
    path: str | Path = DEFAULTS,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Load YAML config and apply ``key.subkey=value`` overrides."""
    with open(path) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Bad override '{item}'; expected key.subkey=value")
        dotted_key, raw = item.split("=", 1)
        _apply_override(cfg, dotted_key.strip(), _coerce(raw.strip()))

    return cfg
