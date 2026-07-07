"""Render saved Proto5 hardware recovery trajectories with the planner mesh scene."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import yaml


PROTO5_WRIST_JOINT_NAMES = ("RHand_WRZ_joint", "RHand_WRY_joint")
PROTO5_DEFAULT_WRIST_JOINT_POS = (0.271, -0.005)


@dataclass
class _VisualizationContext:
    scene: Any
    fingers: list[str]
    obj_dof: int
    full_dof_reference: Any
    joint_index: dict[str, list[int]]
    controlled_joint_index: list[int]
    camera_parameters_path: Path | None
    device: str


def _repo_roots() -> tuple[Path, Path]:
    ccai_root = Path(__file__).resolve().parents[1]
    workspace_root = ccai_root.parent
    return ccai_root, workspace_root


def _add_workspace_import_paths() -> None:
    ccai_root, workspace_root = _repo_roots()
    candidates = [
        ccai_root,
        workspace_root / "torch_cg",
        workspace_root / "pytorch_kinematics" / "src",
        workspace_root / "pytorch_volumetric" / "src",
        workspace_root / "model_mismatch",
    ]
    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def _ensure_isaac_victor_envs_assets_shim() -> None:
    try:
        from isaac_victor_envs.utils import get_assets_dir  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    _, workspace_root = _repo_roots()
    assets_dir = workspace_root / "github" / "isaacsim-hand-envs" / "isaacsim_hand_envs" / "assets" / "urdf"
    if not assets_dir.exists():
        raise FileNotFoundError(f"Cannot locate Isaac hand URDF assets for visualization: {assets_dir}")

    package = types.ModuleType("isaac_victor_envs")
    utils = types.ModuleType("isaac_victor_envs.utils")
    utils.get_assets_dir = lambda: str(assets_dir)
    package.utils = utils
    sys.modules.setdefault("isaac_victor_envs", package)
    sys.modules.setdefault("isaac_victor_envs.utils", utils)


def _load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping config in {config_path}")
    return loaded


def _as_trajectory_list(value: Any) -> list[np.ndarray]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        trajectories: list[np.ndarray] = []
        for item in value:
            trajectories.extend(_as_trajectory_list(item))
        return trajectories
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=object if isinstance(value, np.ndarray) and value.dtype == object else np.float32)
    if array.dtype == object:
        trajectories = []
        for item in array.reshape(-1):
            trajectories.extend(_as_trajectory_list(item))
        return trajectories
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        return [arr.reshape(1, -1)]
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        return [arr[idx] for idx in range(arr.shape[0])]
    flat = arr.reshape((-1,) + arr.shape[-2:])
    return [flat[idx] for idx in range(flat.shape[0])]


def _state_trajectory_for_visualization(traj: Any, *, robot_dof: int = 12, obj_dof: int = 3) -> np.ndarray:
    rows = []
    expected = robot_dof + obj_dof
    for item in _as_trajectory_list(traj):
        arr = np.asarray(item, dtype=np.float32).reshape(item.shape[0], -1)
        if arr.shape[1] < expected:
            continue
        state = arr[:, :expected]
        finite_rows = np.isfinite(state).all(axis=1)
        if finite_rows.any():
            rows.append(state[finite_rows])
    if not rows:
        return np.empty((0, expected + 1), dtype=np.float32)
    states = np.concatenate(rows, axis=0).astype(np.float32)
    cap_joint = np.zeros((states.shape[0], 1), dtype=np.float32)
    return np.concatenate([states, cap_joint], axis=1)


def _first_full_dof_reference(traj_data: dict[Any, Any], config: dict[str, Any]) -> tuple[np.ndarray, bool]:
    for record in traj_data.get("hri_diffpf_records") or []:
        if not isinstance(record, dict):
            continue
        full_joint_pos = record.get("full_joint_pos")
        if full_joint_pos is None:
            continue
        arr = np.asarray(full_joint_pos, dtype=np.float32)
        if arr.shape[-1] == 18:
            return arr.reshape(-1, 18)[0].astype(np.float32), True

    configured = config.get("default_full_joint_pos")
    if configured is not None:
        arr = np.asarray(configured, dtype=np.float32).reshape(-1)
        if arr.shape[0] == 18:
            return arr, False

    from ccai.utils.isaacsim_screwdriver_recovery import PROTO5_DEFAULT_FULL_JOINT_POS

    return np.asarray(PROTO5_DEFAULT_FULL_JOINT_POS, dtype=np.float32), False


def _build_visualization_context(
    *,
    config: dict[str, Any],
    full_dof_reference: np.ndarray,
    render_device: str | None = None,
) -> _VisualizationContext:
    _add_workspace_import_paths()
    _ensure_isaac_victor_envs_assets_shim()

    import torch
    import pytorch_kinematics as pk

    from ccai.allegro_screwdriver_problem import PROTO5_FULL_JOINT_INDEX, Proto5Screwdriver
    from ccai.utils.isaacsim_screwdriver_recovery import (
        DEFAULT_SCREWDRIVER_TABLE_POSE,
        create_world_transform,
        get_hand_spec,
    )

    device = render_device or str(config.get("render_device", "cpu"))
    hand_spec = get_hand_spec("proto5")
    with open(hand_spec.urdf_path, "r", encoding="utf-8") as handle:
        chain = pk.build_chain_from_urdf(handle.read()).to(device=device)

    fingers = list(config.get("fingers") or ["index", "middle", "thumb"])
    controlled_joint_index = sum([PROTO5_FULL_JOINT_INDEX[finger] for finger in fingers], [])
    table_pose = torch.tensor(DEFAULT_SCREWDRIVER_TABLE_POSE, dtype=torch.float32, device=device)
    object_location = torch.tensor(config.get("object_location", [0.0, 0.0, 0.0]), dtype=torch.float32, device=device)
    start = torch.zeros(len(controlled_joint_index) + 3, dtype=torch.float32, device=device)
    goal = start.clone()
    goal[-1] = float(config.get("goal", -1.5))

    asset_urdf_dir = Path(sys.modules["isaac_victor_envs.utils"].get_assets_dir())
    object_asset_path = Path(config.get("object_asset_path", asset_urdf_dir / "screwdriver" / "screwdriver.urdf"))
    robot_sdf_path_prefix = config.get("robot_sdf_path_prefix") or str(hand_spec.planner_robot_sdf_path_prefix)

    problem = Proto5Screwdriver(
        start=start,
        goal=goal,
        T=max(1, int(config.get("T", 1))),
        chain=chain,
        object_location=object_location,
        object_type=str(config.get("object_type", "screwdriver")),
        world_trans=create_world_transform("proto5", device),
        object_asset_pos=table_pose,
        contact_fingers=fingers,
        regrasp_fingers=[],
        friction_coefficient=float(config.get("friction_coefficient", 0.9)),
        yaw_joint_friction=float(config.get("yaw_joint_friction", 0.0)),
        obj_dof=3,
        obj_joint_dim=1,
        optimize_force=bool(config.get("optimize_force", True)),
        obj_gravity=bool(config.get("obj_gravity", True)),
        device=device,
        skip_csvto=True,
        full_dof_reference=full_dof_reference,
        robot_sdf_path_prefix=robot_sdf_path_prefix,
        control_wrist=bool(config.get("proto5_control_wrist", False)),
        object_asset_path=str(object_asset_path),
        object_mass=float(config.get("object_mass", 0.0851)),
        contact_patch_link_frame_z_max=float(config.get("contact_patch_link_frame_z_max", -0.003)),
        filter_self_collision_query_points=False,
    )

    return _VisualizationContext(
        scene=problem.contact_scenes_for_viz,
        fingers=fingers,
        obj_dof=4,
        full_dof_reference=torch.as_tensor(full_dof_reference, dtype=torch.float32, device=device),
        joint_index=PROTO5_FULL_JOINT_INDEX,
        controlled_joint_index=controlled_joint_index,
        camera_parameters_path=None,
        device=device,
    )


def _trial_dir(experiment: Path, trial: int) -> Path:
    direct = experiment / f"trial_{trial}"
    csvgd = experiment / "csvgd" / f"trial_{trial}"
    if direct.exists():
        return direct
    return csvgd


def _iter_stage_items(traj_data: dict[Any, Any], key: str) -> Iterable[tuple[int, int, np.ndarray]]:
    for horizon in sorted(k for k in traj_data if isinstance(k, int)):
        stage = traj_data.get(horizon) or {}
        for item_idx, traj in enumerate(_as_trajectory_list(stage.get(key))):
            yield int(horizon), int(item_idx), traj


def _render_trajectory(
    output_dir: Path,
    traj: np.ndarray,
    *,
    context: _VisualizationContext,
    render_backend: str,
    visualizer: Callable[..., Any] | None = None,
) -> bool:
    if traj.size == 0:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "img").mkdir(parents=True, exist_ok=True)
    (output_dir / "gif").mkdir(parents=True, exist_ok=True)

    if visualizer is None:
        import torch
        from ccai.utils.allegro_utils import visualize_trajectory

        visualizer = visualize_trajectory
        trajectory = torch.as_tensor(traj, dtype=torch.float32, device=context.device)
    else:
        trajectory = traj

    visualizer(
        trajectory,
        context.scene,
        str(output_dir),
        context.fingers,
        context.obj_dof,
        headless=True,
        task="screwdriver",
        render_backend=render_backend,
        full_dof_reference=context.full_dof_reference,
        joint_index=context.joint_index,
        controlled_joint_index=context.controlled_joint_index,
        camera_mode="auto",
        camera_parameters_path=context.camera_parameters_path,
    )
    return True


def render_posthoc_visualizations(
    *,
    experiment: Path,
    trial: int,
    config: Path | None,
    include: set[str],
    render_backend: str,
    visualizer: Callable[..., Any] | None = None,
    context_builder: Callable[..., _VisualizationContext] | None = None,
) -> dict[str, Any]:
    trial_dir = _trial_dir(experiment, trial)
    traj_data_path = trial_dir / "traj_data.p"
    trajectory_path = trial_dir / "trajectory.pkl"
    if not traj_data_path.exists():
        raise FileNotFoundError(f"Missing saved traj_data.p: {traj_data_path}")

    with open(traj_data_path, "rb") as handle:
        traj_data = pickle.load(handle)
    loaded_config = _load_config(config)
    full_dof_reference, has_live_wrist = _first_full_dof_reference(traj_data, loaded_config)
    if context_builder is None:
        context = _build_visualization_context(config=loaded_config, full_dof_reference=full_dof_reference)
    else:
        context = context_builder(config=loaded_config, full_dof_reference=full_dof_reference)

    output_root = trial_dir / "posthoc_viz"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, int] = {"plans": 0, "samples": 0, "executed": 0}

    if "plans" in include:
        for horizon, item_idx, traj in _iter_stage_items(traj_data, "plans"):
            state_traj = _state_trajectory_for_visualization(traj)
            rendered = _render_trajectory(
                output_root / "plans" / f"horizon_{horizon:03d}" / f"plan_{item_idx:04d}",
                state_traj,
                context=context,
                render_backend=render_backend,
                visualizer=visualizer,
            )
            outputs["plans"] += int(rendered)

    if "samples" in include:
        for horizon, item_idx, traj in _iter_stage_items(traj_data, "inits"):
            state_traj = _state_trajectory_for_visualization(traj)
            rendered = _render_trajectory(
                output_root / "samples" / f"horizon_{horizon:03d}" / f"sample_{item_idx:04d}",
                state_traj,
                context=context,
                render_backend=render_backend,
                visualizer=visualizer,
            )
            outputs["samples"] += int(rendered)

    if trajectory_path.exists():
        with open(trajectory_path, "rb") as handle:
            executed = pickle.load(handle)
        executed_traj = _state_trajectory_for_visualization(executed)
        rendered = _render_trajectory(
            output_root / "executed",
            executed_traj,
            context=context,
            render_backend=render_backend,
            visualizer=visualizer,
        )
        outputs["executed"] = int(rendered)

    manifest = {
        "schema": "proto5_saved_recovery_posthoc_viz_v2",
        "experiment": str(experiment),
        "trial": int(trial),
        "trial_dir": str(trial_dir),
        "config": None if config is None else str(config),
        "render_backend": str(render_backend),
        "outputs": outputs,
        "tracks_live_wrist_state": bool(has_live_wrist),
        "wrist_source": "saved_full_joint_pos" if has_live_wrist else "default_proto5_wrist_values",
        "default_wrist_joint_names": list(PROTO5_WRIST_JOINT_NAMES),
        "default_wrist_joint_pos": list(PROTO5_DEFAULT_WRIST_JOINT_POS),
        "limitation": None if has_live_wrist else (
            "Saved run does not contain live wrist state; default Proto5 wrist values were used."
        ),
    }
    with open(output_root / "manifest.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--trial", required=True, type=int)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--render_backend", choices=("offscreen", "window"), default="offscreen")
    parser.add_argument("--include", default="plans,samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include = {item.strip() for item in str(args.include).split(",") if item.strip()}
    manifest = render_posthoc_visualizations(
        experiment=args.experiment.expanduser(),
        trial=args.trial,
        config=None if args.config is None else args.config.expanduser(),
        include=include,
        render_backend=args.render_backend,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
