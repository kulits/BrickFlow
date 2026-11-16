"""Released weights and shared data paths."""

from pathlib import Path

import procbrick.paths
from huggingface_hub import snapshot_download

REPO = "kulits/BrickFlow"
REVISION = "880983ef38804897c75167d0321d557a39d4c30d"
WEIGHTS = tuple(f"weights/{stage}-{size}" for stage in ("pt", "ic") for size in "sbl")


def data_dir() -> Path:
    """`brickflow/` in BrickNet's data directory, beside ProcBrick's `procbrick/`."""
    return procbrick.paths.data_dir().parent / "brickflow"


def find(name: str | Path) -> Path:
    """`name` if it exists, else that file or directory in the data directory or in ProcBrick's."""
    if Path(name).exists():
        return Path(name)
    installed = data_dir() / name
    if installed.exists():
        return installed
    released = Path(name).with_suffix("").as_posix()
    if Path(name).suffix == ".pt" and released in WEIGHTS:
        raise FileNotFoundError(f"{installed} is not installed; run `python -m brickflow fetch {released}`")
    return procbrick.paths.find(name)


def fetch(names: list[str]) -> None:
    """Download weights (`WEIGHTS`) and ProcBrick's released files (`procbrick.paths.NAMES`)."""
    unknown = [name for name in names if name not in WEIGHTS + procbrick.paths.NAMES]
    if unknown:
        known = ", ".join(WEIGHTS + procbrick.paths.NAMES)
        raise ValueError(f"unknown names {', '.join(unknown)}; the release has {known}")
    weights = [f"{name}.pt" for name in names if name in WEIGHTS]
    if weights:
        snapshot_download(REPO, revision=REVISION, allow_patterns=weights, local_dir=data_dir())
    released = [name for name in names if name in procbrick.paths.NAMES]
    if released:
        procbrick.paths.fetch(released)
