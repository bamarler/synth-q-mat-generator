from __future__ import annotations

import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pymatgen.core import Structure


@runtime_checkable
class Generator(Protocol):
    """Produces a batch of structures, optionally conditioned on a chemical system."""

    def generate(
        self, batch_size: int, num_batches: int = 1, chemical_system: str | None = None
    ) -> list[Structure]: ...


def build_generator(cfg: dict[str, Any]) -> Generator:
    name = cfg["models"].get("generator", "mattergen")
    if name == "mattergen":
        return MatterGenGenerator.from_config(cfg)
    raise ValueError(f"unknown generator: {name!r}")


class MatterGenGenerator:
    """Generate structures via the MatterGen CLI in its isolated venv (subprocess)."""

    def __init__(
        self,
        venv_dir: str = ".venv-mattergen",
        pretrained_name: str = "chemical_system",
        guidance_factor: float = 2.0,
    ) -> None:
        self.bin = Path(venv_dir) / "bin" / "mattergen-generate"
        self.pretrained_name = pretrained_name
        self.guidance_factor = guidance_factor

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> MatterGenGenerator:
        g = cfg.get("generate", {})
        return cls(
            pretrained_name=g.get("pretrained_name", "chemical_system"),
            guidance_factor=g.get("guidance_factor", 2.0),
        )

    def generate(
        self, batch_size: int, num_batches: int = 1, chemical_system: str | None = None
    ) -> list[Structure]:
        """Generate structures; if chemical_system is given (e.g. 'Li-Si-O'), condition on it."""
        if not self.bin.exists():
            raise FileNotFoundError(
                f"{self.bin} not found — run `make setup-generator`."
            )
        cmd = [
            str(self.bin),
            "",  # output dir, set once tmp exists
            f"--pretrained-name={self.pretrained_name}",
            f"--batch_size={batch_size}",
            f"--num_batches={num_batches}",
            "--record-trajectories=False",
        ]
        if chemical_system:
            cmd += [
                f"--properties_to_condition_on={{'chemical_system':'{chemical_system}'}}",
                f"--diffusion_guidance_factor={self.guidance_factor}",
            ]
        with tempfile.TemporaryDirectory() as tmp:
            cmd[1] = tmp
            subprocess.run(cmd, check=True)
            return _load_cifs(Path(tmp) / "generated_crystals_cif.zip")


def _load_cifs(zip_path: Path) -> list[Structure]:
    with zipfile.ZipFile(zip_path) as zf:
        return [
            Structure.from_str(zf.read(n).decode(), fmt="cif")
            for n in sorted(zf.namelist())
            if n.endswith(".cif")
        ]
