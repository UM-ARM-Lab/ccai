"""Path discovery helpers for CCAI recovery scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _ancestors(path: Path) -> Iterable[Path]:
    current = path.resolve()
    if current.is_file():
        current = current.parent
    yield current
    yield from current.parents


def _is_model_mismatch_root(path: Path) -> bool:
    return (
        (path / "model_mismatch" / "__init__.py").is_file()
        and (path / "examples" / "evaluation").is_dir()
    )


def find_model_mismatch_root(start: str | Path) -> Path:
    """Find the model_mismatch repository root from nested or sibling layouts."""
    checked: list[Path] = []
    for base in _ancestors(Path(start)):
        candidates = (base, base / "model_mismatch")
        for candidate in candidates:
            candidate = candidate.resolve()
            checked.append(candidate)
            if _is_model_mismatch_root(candidate):
                return candidate
    raise RuntimeError(
        "Could not find model_mismatch repository root. Checked: "
        + ", ".join(str(path) for path in checked)
    )


def _first_existing_package_root(
    candidates: Iterable[Path],
    package_dir: str,
    fallback: Path,
) -> Path:
    for candidate in candidates:
        if (candidate / package_dir).is_dir():
            return candidate
    return fallback


def resolve_isaacsim_hand_envs_path(model_mismatch_root: str | Path) -> Path:
    root = Path(model_mismatch_root).resolve()
    candidates = (
        root / "src" / "isaacsim-arm-envs",
        root / "src" / "isaacsim-hand-envs",
        root.parent / "isaacsim-arm-envs",
        root.parent / "isaacsim-hand-envs",
        root.parent / "github" / "isaacsim-hand-envs",
        root.parent / "github" / "isaacsim-arm-envs",
    )
    return _first_existing_package_root(
        candidates,
        "isaacsim_hand_envs",
        root.parent / "github" / "isaacsim-hand-envs",
    )


def resolve_isaac_victor_envs_path(model_mismatch_root: str | Path) -> Path:
    root = Path(model_mismatch_root).resolve()
    candidates = (
        root / "src" / "isaac-victor-envs",
        root / "src" / "isaacgym-arm-envs",
        root.parent / "isaac-victor-envs",
        root.parent / "isaacgym-arm-envs",
        root.parent / "github" / "isaac-victor-envs",
        root.parent / "github" / "isaacgym-arm-envs",
    )
    return _first_existing_package_root(
        candidates,
        "isaac_victor_envs",
        root.parent / "github" / "isaacgym-arm-envs",
    )


def resolve_torch_cg_path(model_mismatch_root: str | Path) -> Path:
    root = Path(model_mismatch_root).resolve()
    candidates = (
        root / "src" / "torch-cg",
        root / "src" / "torch_cg",
        root.parent / "torch-cg",
        root.parent / "torch_cg",
        root.parent / "github" / "torch-cg",
        root.parent / "github" / "torch_cg",
    )
    return _first_existing_package_root(
        candidates,
        "torch_cg",
        root.parent / "torch_cg",
    )
