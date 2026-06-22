from __future__ import annotations

import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from pymatgen.core import Structure


class MatterGenGenerator:
    """Generate structures via the MatterGen CLI in its isolated venv (subprocess)."""

    def __init__(
        self, venv_dir: str = ".venv-mattergen", pretrained_name: str = "mattergen_base"
    ) -> None:
        self.bin = Path(venv_dir) / "bin" / "mattergen-generate"
        self.pretrained_name = pretrained_name

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> MatterGenGenerator:
        g = cfg.get("generate", {})
        return cls(pretrained_name=g.get("pretrained_name", "mattergen_base"))

    def generate(self, batch_size: int, num_batches: int = 1) -> list[Structure]:
        if not self.bin.exists():
            raise FileNotFoundError(
                f"{self.bin} not found — run `make setup-generator`."
            )
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    str(self.bin),
                    tmp,
                    f"--pretrained-name={self.pretrained_name}",
                    f"--batch_size={batch_size}",
                    f"--num_batches={num_batches}",
                    "--record-trajectories=False",
                ],
                check=True,
            )
            return _load_cifs(Path(tmp) / "generated_crystals_cif.zip")


def _load_cifs(zip_path: Path) -> list[Structure]:
    with zipfile.ZipFile(zip_path) as zf:
        return [
            Structure.from_str(zf.read(n).decode(), fmt="cif")
            for n in sorted(zf.namelist())
            if n.endswith(".cif")
        ]
