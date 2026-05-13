from __future__ import annotations

import os
from pathlib import Path


def resolve_root(root: str | Path | None = None) -> Path:
    if root is None:
        root_path = Path(__file__).resolve().parents[1]
    else:
        root_path = Path(root).resolve()
    if root_path.name != "sampling_optimization_study":
        raise RuntimeError(f"Study root must be sampling_optimization_study, got {root_path}")
    return root_path


def assert_inside_root(path: str | Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Path escapes study root: {resolved}") from exc
    return resolved


def prepare_runtime(root: Path) -> None:
    root = root.resolve()
    assert_inside_root(root, root)
    for name in ("src", "outputs", "logs", "configs", "reports", "models"):
        target = root / name
        assert_inside_root(target, root)
        target.mkdir(parents=True, exist_ok=True)
    mpl_dir = root / "logs" / "matplotlib_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ["PYTHONPYCACHEPREFIX"] = str(root / "logs" / "pycache")


def checked_open(path: str | Path, root: Path, mode: str = "r", encoding: str = "utf-8"):
    resolved = assert_inside_root(path, root)
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return open(resolved, mode, encoding=encoding)

