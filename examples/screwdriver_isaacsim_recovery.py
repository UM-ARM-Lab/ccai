"""Generate CCAI screwdriver recovery datasets in Isaac Sim.

This entrypoint keeps the legacy recovery loop and pickle artifacts, but uses
IsaacLab screwdriver environments through ``IsaacSimScrewdriverRecoveryEnv``.
"""

from __future__ import annotations

import argparse
import datetime
import faulthandler
import importlib
import importlib.util
import os
import pathlib
import pickle
import shutil
import sys
import types

import yaml


CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
DOCUMENTS_PATH = CCAI_PATH.parent
MODEL_MISMATCH_PATH = DOCUMENTS_PATH / "model_mismatch"
ISAACSIM_HAND_ENVS_PATH = DOCUMENTS_PATH / "github" / "isaacsim-hand-envs"
ISAACGYM_ARM_ENVS_PATH = DOCUMENTS_PATH / "github" / "isaacgym-arm-envs"
TORCH_CG_PATH = DOCUMENTS_PATH / "torch_cg"
ISAAC_VICTOR_COMPAT_ASSETS_DIR = ISAACSIM_HAND_ENVS_PATH / "isaacsim_hand_envs" / "assets" / "urdf"
DEFAULT_CONFIG_PATH = CCAI_PATH / "examples" / "config" / "proto5" / "proto_screwdriver_csvto_TODR_recovery_data_gen_no_belief_reset.yaml"
DEFAULT_PLANNER_YAW_FRICTION_MODEL_PATH = (
    MODEL_MISMATCH_PATH / "results" / "csvto_yaw_joint_fit" / "yaw_friction_model.json"
)
DEFAULT_PLANNER_YAW_INERTIA_MODEL_PATH = (
    MODEL_MISMATCH_PATH / "results" / "csvto_yaw_inertia_fit_full" / "yaw_inertia_model.json"
)
PROTO5_POINT_CACHE_NAMES = (
    "RHand_I6AF_LINK_points_cache.pkl",
    "RHand_M6AF_LINK_points_cache.pkl",
    "RHand_T6AF_LINK_points_cache.pkl",
)
PROTO5_POINT_CACHE_SOURCE_DIRS = (
    DOCUMENTS_PATH / "model_mismatch",
    DOCUMENTS_PATH / "model_mismatch" / "scripts",
    DOCUMENTS_PATH / "model_mismatch" / "examples",
)
DEFAULT_OBJ_ORIENTATION_NOISE_STD = 0.03
DEFAULT_OBJ_POSITION_NOISE_RANGE = (-0.0075, 0.0075)
DEFAULT_DIFFPF_SAMPLE_HORIZON = 8
DEFAULT_DIFFPF_EXECUTION_HORIZON = 1
DEFAULT_DIFFPF_NUM_TRAJECTORIES = 128
DEFAULT_DIFFPF_TRAJECTORY_SELECTION_MODE = "max_reward_times_exp_likelihood"
DEFAULT_DIFFPF_LIKELIHOOD_MASK = "inverse_dynamics"
DEFAULT_DIFFPF_LIKELIHOOD_TEMPERATURE = 10.0
DEFAULT_DIFFPF_LIKELIHOOD_REWARD_SCOPE = "per_step"
RECOVERY_DIFFPF_OVERRIDE_SUFFIXES = (
    "ema_decay",
    "compile_model",
    "sample_horizon",
    "execution_horizon",
    "num_trajectories",
    "temporal_ensemble_actions",
    "temporal_ensemble_coeff",
    "autoregressive_planning",
    "freeze_autoregressive_mamba_token",
    "context_augmentation",
    "eqm_nesterov_override",
    "beam_search",
    "beam_width",
    "beam_children_per_branch",
    "trajectory_selection_mode",
    "likelihood_mask",
    "likelihood_proposal_correction",
    "likelihood_temperature",
    "likelihood_step_min",
    "likelihood_reward_scope",
    "likelihood_min",
    "likelihood_min_mode",
    "likelihood_stop_min",
    "likelihood_stop_stat",
    "likelihood_stop_mode",
    "likelihood_stop_warn_without_proposal_correction",
    "trajectory_selection_discount",
    "reward_trust_decay",
    "reward_source",
    "drop_reward_indicator",
    "r_cond",
    "r_cond_value",
    "r_cond_dataset_path",
    "allow_planning_beyond_episode_end",
    "denoising_steps",
    "info_gain_scale",
    "all_particles_info_gain_mult",
    "attn_mask_method",
)
DEFAULT_HARDWARE_ROS_CONFIG = (
    MODEL_MISMATCH_PATH / "examples" / "evaluation" / "config" / "screwdriver_hardware_ros_profiles.yaml"
)
DEFAULT_MIN_FORCE_BY_HAND = {
    "allegro": {
        "thumb": 1.0,
        "middle": 1.0,
        "index": 1.0,
    },
    "proto5": {
        "thumb": 1.0,
        "middle": 1.0,
        "index": 1.0,
    },
}


def _bool_from_cli(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--hand", choices=("allegro", "proto5"), default='proto5')
    parser.add_argument("--headless", type=_bool_from_cli, default=None)
    parser.add_argument("--no_video", type=_bool_from_cli, default=False)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--sim_device", type=str, default='cuda:0')
    parser.add_argument("--proto5_control_wrist", action="store_true", default=None)
    parser.add_argument("--steps_per_action", type=int, default=None)
    parser.add_argument("--action_repeat", type=int, default=None)
    parser.add_argument("--save_recovery_frames", type=_bool_from_cli, default=None)
    parser.add_argument("--start_ind", type=int, default=None)
    parser.add_argument("--end_ind", type=int, default=None)
    parser.add_argument("--skip_pregrasp", type=_bool_from_cli, default=None)
    pregrasp_only_group = parser.add_mutually_exclusive_group()
    pregrasp_only_group.add_argument(
        "--pregrasp_only",
        dest="pregrasp_only",
        action="store_true",
        default=None,
    )
    pregrasp_only_group.add_argument(
        "--no_pregrasp_only",
        dest="pregrasp_only",
        action="store_false",
    )
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--debug_progress", action="store_true")
    parser.add_argument("--planner_yaw_joint_friction_override", type=float, default=None)
    yaw_friction_group = parser.add_mutually_exclusive_group()
    yaw_friction_group.add_argument(
        "--planner_use_env_yaw_joint_friction",
        dest="planner_use_env_yaw_joint_friction",
        action="store_true",
        default=None,
    )
    yaw_friction_group.add_argument(
        "--disable_planner_use_env_yaw_joint_friction",
        dest="planner_use_env_yaw_joint_friction",
        action="store_false",
    )
    parser.add_argument("--planner_yaw_friction_model_path", type=str, default=None)
    parser.add_argument("--disable_planner_yaw_friction_model", action="store_true", default=None)
    parser.add_argument("--planner_yaw_inertia_model_path", type=str, default=None)
    yaw_inertia_group = parser.add_mutually_exclusive_group()
    yaw_inertia_group.add_argument(
        "--enable_planner_yaw_inertia_model",
        dest="planner_use_yaw_inertia_model",
        action="store_true",
        default=None,
    )
    yaw_inertia_group.add_argument(
        "--disable_planner_yaw_inertia_model",
        dest="planner_use_yaw_inertia_model",
        action="store_false",
    )
    pregrasp_target_group = parser.add_mutually_exclusive_group()
    pregrasp_target_group.add_argument(
        "--enable_pregrasp_reference_targets",
        dest="use_pregrasp_reference_targets",
        action="store_true",
        default=None,
    )
    pregrasp_target_group.add_argument(
        "--disable_pregrasp_reference_targets",
        dest="use_pregrasp_reference_targets",
        action="store_false",
    )
    parser.add_argument("--hardware_ros_config", type=str, default=None)
    parser.add_argument("--hardware_profile", type=str, default=None)
    parser.add_argument("--hardware_execute", type=_bool_from_cli, default=None)
    parser.add_argument("--hardware_command_topic", type=str, default=None)
    parser.add_argument("--hardware_joint_state_topic", type=str, default=None)
    parser.add_argument("--hardware_mocap_topic", type=str, default=None)
    parser.add_argument("--hardware_proto5_wrench_topic", type=str, default=None)
    parser.add_argument("--hardware_proto5_wrench_fixed_joint_names", type=str, default=None)
    parser.add_argument("--hardware_num_repeat", type=int, default=None)
    parser.add_argument("--hardware_command_mode", type=str, default=None)
    parser.add_argument("--hardware_command_duration_s", type=float, default=None)
    parser.add_argument("--hardware_allow_placeholder_wrenches", type=_bool_from_cli, default=None)
    parser.add_argument("--hardware_use_live_screwdriver_orientation", type=_bool_from_cli, default=None)
    parser.add_argument("--hardware_debug_mocap_orientation", type=_bool_from_cli, default=None)
    return parser.parse_args()


def _parse_min_force_config(config: dict) -> dict[str, float]:
    hand = str(config.get("hand", "allegro")).lower()
    fallback = DEFAULT_MIN_FORCE_BY_HAND.get(hand, DEFAULT_MIN_FORCE_BY_HAND["allegro"])
    configured = config.get("min_force", config.get("min_force_dict", fallback))
    if configured is None:
        return dict(fallback)
    if isinstance(configured, (int, float)):
        return {finger: float(configured) for finger in fallback}
    if not isinstance(configured, dict):
        raise ValueError("min_force must be a scalar or a mapping from finger name to force.")

    unknown_fingers = set(configured) - set(fallback)
    if unknown_fingers:
        raise ValueError(
            "min_force contains unknown finger(s): "
            + ", ".join(sorted(unknown_fingers))
            + f". Expected any of: {', '.join(sorted(fallback))}."
        )
    min_force_dict = dict(fallback)
    min_force_dict.update({finger: float(value) for finger, value in configured.items()})
    return min_force_dict


def load_config(args) -> dict:
    config_path = args.config
    if not config_path.is_absolute():
        config_path = CCAI_PATH / config_path
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    headless_from_cli = getattr(args, "headless", None) is not None
    config["config_path"] = str(config_path)
    config["simulator"] = "isaacsim"
    config.setdefault("mode", "simulation")

    for key in (
        "hand",
        "headless",
        "no_video",
        "num_envs",
        "sim_device",
        "steps_per_action",
        "action_repeat",
        "save_recovery_frames",
        "start_ind",
        "end_ind",
        "skip_pregrasp",
        "pregrasp_only",
        "experiment_name",
        "planner_yaw_joint_friction_override",
        "planner_use_env_yaw_joint_friction",
        "planner_yaw_friction_model_path",
        "disable_planner_yaw_friction_model",
        "planner_yaw_inertia_model_path",
        "planner_use_yaw_inertia_model",
        "use_pregrasp_reference_targets",
        "hardware_ros_config",
        "hardware_profile",
        "hardware_execute",
        "hardware_command_topic",
        "hardware_joint_state_topic",
        "hardware_mocap_topic",
        "hardware_proto5_wrench_topic",
        "hardware_proto5_wrench_fixed_joint_names",
        "hardware_num_repeat",
        "hardware_command_mode",
        "hardware_command_duration_s",
        "hardware_allow_placeholder_wrenches",
        "hardware_use_live_screwdriver_orientation",
        "hardware_debug_mocap_orientation",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if args.proto5_control_wrist is not None:
        config["proto5_control_wrist"] = bool(args.proto5_control_wrist)

    config.setdefault("hand", "allegro")
    if not headless_from_cli and not bool(config.get("visualize", True)):
        config["headless"] = True
    elif "headless" not in config:
        config["headless"] = not bool(config.get("visualize", False))
    config.setdefault("no_video", True)
    config.setdefault("visualize_executed_rollout", False)
    config.setdefault("num_envs", 1)
    config.setdefault("sim_device", "cuda:0")
    config.setdefault("proto5_control_wrist", False)
    config.setdefault("steps_per_action", 40)
    config.setdefault("action_repeat", 3)
    config.setdefault("save_recovery_frames", True)
    config.setdefault("pregrasp_only", False)
    config.setdefault("planner_use_env_yaw_joint_friction", True)
    config.setdefault("planner_yaw_joint_friction_override", 0.0)
    config.setdefault("planner_yaw_friction_model_path", str(DEFAULT_PLANNER_YAW_FRICTION_MODEL_PATH))
    config.setdefault("disable_planner_yaw_friction_model", False)
    config.setdefault("planner_yaw_inertia_model_path", str(DEFAULT_PLANNER_YAW_INERTIA_MODEL_PATH))
    config.setdefault("planner_use_yaw_inertia_model", False)
    config.setdefault("use_pregrasp_reference_targets", False)
    config.setdefault("obj_orientation_noise_std", DEFAULT_OBJ_ORIENTATION_NOISE_STD)
    config.setdefault("obj_position_noise_range_x", DEFAULT_OBJ_POSITION_NOISE_RANGE)
    config.setdefault("obj_position_noise_range_y", DEFAULT_OBJ_POSITION_NOISE_RANGE)
    config.setdefault("obj_position_noise_range_z", DEFAULT_OBJ_POSITION_NOISE_RANGE)
    config.setdefault("diffpf_checkpoint", None)
    config.setdefault("diffpf_ema_decay", None)
    config.setdefault("diffpf_compile_model", False)
    config.setdefault("compile_models", True)
    config.setdefault("diffpf_sample_horizon", DEFAULT_DIFFPF_SAMPLE_HORIZON)
    config.setdefault("diffpf_execution_horizon", DEFAULT_DIFFPF_EXECUTION_HORIZON)
    config.setdefault("diffpf_num_trajectories", DEFAULT_DIFFPF_NUM_TRAJECTORIES)
    config.setdefault("diffpf_trajectory_selection_mode", DEFAULT_DIFFPF_TRAJECTORY_SELECTION_MODE)
    config.setdefault("diffpf_likelihood_mask", DEFAULT_DIFFPF_LIKELIHOOD_MASK)
    config.setdefault("diffpf_likelihood_temperature", DEFAULT_DIFFPF_LIKELIHOOD_TEMPERATURE)
    config.setdefault("diffpf_likelihood_reward_scope", DEFAULT_DIFFPF_LIKELIHOOD_REWARD_SCOPE)
    config.setdefault("diffpf_reset_belief_after_recovery", True)
    config.setdefault("recovery_controller", "csvgd")
    config.setdefault("recovery_diffpf_checkpoint", None)
    for suffix in RECOVERY_DIFFPF_OVERRIDE_SUFFIXES:
        config.setdefault(f"recovery_diffpf_{suffix}", None)
    config.setdefault("hardware_ros_config", str(DEFAULT_HARDWARE_ROS_CONFIG))
    config.setdefault("hardware_profile", str(config.get("hand", "proto5")))
    config.setdefault("hardware_execute", False)
    config.setdefault("hardware_num_repeat", 10)
    config.setdefault("hardware_command_mode", "repeat")
    config.setdefault("hardware_command_duration_s", 1.0 / 12.0)
    config.setdefault("hardware_allow_placeholder_wrenches", False)
    config.setdefault("hardware_use_live_screwdriver_position", True)
    config.setdefault("hardware_use_live_screwdriver_orientation", True)
    config.setdefault("hardware_debug_mocap_orientation", False)
    config["mode"] = str(config.get("mode", "simulation")).lower()
    if config["mode"] not in {"simulation", "hardware", "hardware_copy"}:
        raise ValueError(f"Unsupported mode {config['mode']!r}; expected simulation, hardware, or hardware_copy.")
    if str(config.get("recovery_controller", "")).lower() == "diffpf":
        if config.get("recovery_diffpf_checkpoint") in (None, ""):
            raise ValueError("recovery_controller: diffpf requires recovery_diffpf_checkpoint.")
        if str(config.get("OOD_metric", "")).lower() != "likelihood":
            raise ValueError("recovery_controller: diffpf requires OOD_metric: likelihood.")
    if config["mode"] == "hardware":
        config["external_wrench_perturb"] = False
        config["randomize_obj_start"] = False
        config["save_recovery_frames"] = False
        config["simulator"] = "hardware"
    config["min_force_dict"] = _parse_min_force_config(config)
    if bool(config["save_recovery_frames"]) and bool(config["no_video"]):
        raise ValueError("save_recovery_frames=True requires cameras; run without --no_video true.")
    if config["no_video"]:
        config["visualize"] = False
        config["visualize_plan"] = False
        config["visualize_recovery_plan"] = False
        config["visualize_contact_plan"] = False
        config["visualize_recovery_planning_samples"] = False
    if args.debug_progress:
        config["debug_progress"] = True
    return config


def launch_isaaclab(config):
    if str(ISAACSIM_HAND_ENVS_PATH) not in sys.path:
        sys.path.insert(0, str(ISAACSIM_HAND_ENVS_PATH))
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        headless=bool(config.get("headless", True)),
        device=str(config.get("sim_device", "cuda:0")),
        enable_cameras=not bool(config.get("no_video", True)),
    )
    return app_launcher.app


def ensure_proto5_point_cache(config):
    if str(config.get("hand", "allegro")).lower() != "proto5":
        return
    cache_dir = CCAI_PATH / "data" / "cache" / "proto5_points"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTORCH_VOLUMETRIC_POINTS_CACHE_DIR"] = str(cache_dir)
    missing = []
    for cache_name in PROTO5_POINT_CACHE_NAMES:
        dest = cache_dir / cache_name
        if dest.exists():
            continue
        source = next((src_dir / cache_name for src_dir in PROTO5_POINT_CACHE_SOURCE_DIRS if (src_dir / cache_name).exists()), None)
        if source is None:
            missing.append(cache_name)
            continue
        shutil.copy2(source, dest)
    if missing:
        raise FileNotFoundError(
            "Missing Proto5 point caches required for headless recovery: "
            + ", ".join(missing)
            + f". Expected them in one of: {', '.join(str(path) for path in PROTO5_POINT_CACHE_SOURCE_DIRS)}"
        )


def resolve_planner_yaw_model_paths(config) -> tuple[str | None, str | None]:
    yaw_friction_model_path = None if config.get("disable_planner_yaw_friction_model", False) else config.get(
        "planner_yaw_friction_model_path",
        str(DEFAULT_PLANNER_YAW_FRICTION_MODEL_PATH),
    )
    yaw_inertia_model_path = (
        config.get("planner_yaw_inertia_model_path", str(DEFAULT_PLANNER_YAW_INERTIA_MODEL_PATH))
        if config.get("planner_use_yaw_inertia_model", False)
        else None
    )
    for label, path in (
        ("Planner yaw friction model", yaw_friction_model_path),
        ("Planner yaw inertia model", yaw_inertia_model_path),
    ):
        if path is not None and not pathlib.Path(path).expanduser().exists():
            raise FileNotFoundError(f"{label} JSON not found: {path}")
    return yaw_friction_model_path, yaw_inertia_model_path


def get_recovery_planner_physical_kwargs(env, config) -> dict:
    if str(MODEL_MISMATCH_PATH) not in sys.path:
        sys.path.insert(0, str(MODEL_MISMATCH_PATH))
    from model_mismatch.utils.screwdriver_csvto_planning import get_screwdriver_turn_problem_physical_kwargs

    yaw_friction_model_path, yaw_inertia_model_path = resolve_planner_yaw_model_paths(config)
    yaw_joint_friction_override = (
        None
        if config.get("planner_use_env_yaw_joint_friction", True)
        else config.get("planner_yaw_joint_friction_override", 0.0)
    )
    env_params = env.get_environment_parameters(env_id=0)
    physical_kwargs = get_screwdriver_turn_problem_physical_kwargs(
        env_params,
        yaw_joint_friction_override=yaw_joint_friction_override,
        yaw_friction_model_path=yaw_friction_model_path,
        yaw_inertia_model_path=yaw_inertia_model_path,
    )
    if "friction_coefficient" in config:
        physical_kwargs["friction_coefficient"] = float(config["friction_coefficient"])
    if "yaw_joint_friction" in config:
        physical_kwargs["yaw_joint_friction"] = float(config["yaw_joint_friction"])
    return physical_kwargs


def _is_proto5_hardware_dataset_initialization_mode(config: dict) -> bool:
    return (
        str(config.get("mode", "simulation")).lower() == "hardware"
        and str(config.get("hand", "")).lower() == "proto5"
    )


def _resolve_proto5_validation_dataset_row(seed: int) -> int:
    if str(MODEL_MISMATCH_PATH) not in sys.path:
        sys.path.insert(0, str(MODEL_MISMATCH_PATH))
    from scripts.proto5_screwdriver_val_indices import proto5_screwdriver_val_indices

    ordinal = int(seed)
    if ordinal < 0 or ordinal >= len(proto5_screwdriver_val_indices):
        raise ValueError(
            "Proto5 hardware seed selects an ordinal in "
            "scripts/proto5_screwdriver_val_indices.py; "
            f"got {ordinal}, expected 0 <= seed < {len(proto5_screwdriver_val_indices)}."
        )
    return int(proto5_screwdriver_val_indices[ordinal])


def _load_proto5_hardware_initialization(config: dict) -> dict | None:
    if not _is_proto5_hardware_dataset_initialization_mode(config):
        return None
    dataset_path = config.get("dataset_path")
    if dataset_path in (None, ""):
        raise ValueError(
            "Proto5 hardware initialization requires dataset_path in the recovery YAML "
            "so seed can select an initial pose from scripts/proto5_screwdriver_val_indices.py."
        )

    import numpy as np

    if str(MODEL_MISMATCH_PATH) not in sys.path:
        sys.path.insert(0, str(MODEL_MISMATCH_PATH))
    from model_mismatch.utils.trajectory_dataset_io import open_trajectory_dataset, trajectory_dataset_keys

    validation_ordinal = int(config.get("seed", 0))
    trajectory_row = _resolve_proto5_validation_dataset_row(validation_ordinal)
    initial_orientation = None
    with open_trajectory_dataset(str(dataset_path), allow_pickle=True) as data:
        dataset_keys = set(trajectory_dataset_keys(data))
        if "initial_joint_targets" in dataset_keys:
            initial_target = np.asarray(
                data["initial_joint_targets"][trajectory_row],
                dtype=np.float32,
            ).reshape(-1)
            target_source = "initial_joint_targets"
        elif "initial_state" in dataset_keys:
            initial_state = np.asarray(data["initial_state"][trajectory_row], dtype=np.float32).reshape(-1)
            initial_target = initial_state[:12].astype(np.float32, copy=True)
            if initial_state.shape[0] >= 15:
                initial_orientation = initial_state[12:15].astype(np.float32, copy=True)
            target_source = "initial_state[:12]"
        elif "initial_states" in dataset_keys:
            initial_state = np.asarray(data["initial_states"][trajectory_row], dtype=np.float32).reshape(-1)
            initial_target = initial_state[:12].astype(np.float32, copy=True)
            if initial_state.shape[0] >= 15:
                initial_orientation = initial_state[12:15].astype(np.float32, copy=True)
            target_source = "initial_states[:12]"
        elif "q" in dataset_keys:
            q0 = np.asarray(data["q"][trajectory_row], dtype=np.float32)
            q0_state = q0.reshape(-1) if q0.ndim == 1 else q0.reshape(q0.shape[0], -1)[0]
            initial_target = q0_state[:12].astype(np.float32, copy=True)
            if q0_state.shape[0] >= 15:
                initial_orientation = q0_state[12:15].astype(np.float32, copy=True)
            target_source = "q[0, :12]"
        else:
            raise ValueError(
                "Proto5 hardware initialization dataset must contain one of "
                "initial_joint_targets, initial_state, initial_states, or q."
            )

    expected_dim = 14 if bool(config.get("proto5_control_wrist", False)) else 12
    if initial_target.shape != (expected_dim,):
        raise ValueError(
            "Proto5 hardware initial pose target has incompatible dimension: "
            f"selected {target_source} from dataset row {trajectory_row} with shape "
            f"{tuple(initial_target.shape)}, expected ({expected_dim},) for "
            f"proto5_control_wrist={bool(config.get('proto5_control_wrist', False))}."
        )

    return {
        "validation_ordinal": validation_ordinal,
        "trajectory_row": trajectory_row,
        "target_source": target_source,
        "initial_target": initial_target,
        "initial_orientation": initial_orientation,
    }


def send_proto5_hardware_initial_pose_and_wait(env, initialization: dict, *, device) -> None:
    import torch

    initial_target = torch.as_tensor(
        initialization["initial_target"],
        device=device,
        dtype=torch.float32,
    ).reshape(1, -1)
    print(
        "Proto5 hardware initial pose: "
        f"validation ordinal {int(initialization['validation_ordinal'])} -> "
        f"dataset row {int(initialization['trajectory_row'])}; "
        f"target_dim={initial_target.shape[-1]} "
        f"source={initialization['target_source']}.",
        flush=True,
    )
    env.step(initial_target)
    initial_orientation = initialization.get("initial_orientation")
    if initial_orientation is not None:
        print(
            "Proto5 hardware dataset initial object orientation available: "
            f"{initial_orientation.tolist()}. "
            "The execution start orientation will be captured after confirmation.",
            flush=True,
        )
    input(
        "Proto5 initial hand pose command sent. "
        "Confirm the hand is ready, then press Enter to start policy execution."
    )
    if hasattr(env, "capture_observed_object_orientation"):
        confirmed_orientation = env.capture_observed_object_orientation()
        confirmed_orientation = torch.as_tensor(confirmed_orientation, dtype=torch.float32)
        print(
            "Proto5 hardware initial object orientation captured after confirmation: "
            f"{confirmed_orientation.detach().cpu().reshape(-1).tolist()}",
            flush=True,
        )
    elif initial_orientation is not None and hasattr(env, "set_observed_object_orientation"):
        env.set_observed_object_orientation(initial_orientation)
        print(
            "Proto5 hardware initial object orientation loaded from dataset after confirmation: "
            f"{initial_orientation.tolist()}",
            flush=True,
        )
    if hasattr(env, "print_mocap_orientation_diagnostic"):
        env.print_mocap_orientation_diagnostic(context="after_initial_pose_confirm")


def _friction_range(config, prefix: str, fallback):
    low_key = f"{prefix}_min"
    high_key = f"{prefix}_max"
    if low_key in config or high_key in config:
        low = float(config.get(low_key, fallback[0]))
        high = float(config.get(high_key, fallback[1]))
        return (low, high)
    value = config.get(prefix)
    if value is not None:
        value = float(value)
        return (value, value)
    return fallback


def _float_pair(value, *, key: str) -> tuple[float, float]:
    if value is None:
        raise ValueError(f"{key} must be a two-value range, got None.")
    if isinstance(value, str):
        value = value.strip().strip("[]()")
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError(f"{key} must be a two-value range, got {value!r}.")
    return (float(parts[0]), float(parts[1]))


def get_object_randomization_params(config: dict) -> dict:
    return {
        "obj_orientation_noise_std": float(
            config.get("obj_orientation_noise_std", DEFAULT_OBJ_ORIENTATION_NOISE_STD)
        ),
        "obj_position_noise_range_x": _float_pair(
            config.get("obj_position_noise_range_x", DEFAULT_OBJ_POSITION_NOISE_RANGE),
            key="obj_position_noise_range_x",
        ),
        "obj_position_noise_range_y": _float_pair(
            config.get("obj_position_noise_range_y", DEFAULT_OBJ_POSITION_NOISE_RANGE),
            key="obj_position_noise_range_y",
        ),
        "obj_position_noise_range_z": _float_pair(
            config.get("obj_position_noise_range_z", DEFAULT_OBJ_POSITION_NOISE_RANGE),
            key="obj_position_noise_range_z",
        ),
    }


def randomize_isaacsim_object_start(env, config: dict, rng):
    """Match the collector's object-start randomization fields and sampling."""
    if not bool(config.get("randomize_obj_start", False)):
        return None

    import numpy as np
    import torch

    params = get_object_randomization_params(config)
    obj = env.scene["obj"]
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    if hasattr(obj.data, "root_pos_w"):
        root_pos = obj.data.root_pos_w[env_ids].clone()
    else:
        root_pos = obj.data.root_link_pos_w[env_ids].clone()
    if hasattr(obj.data, "root_quat_w"):
        root_quat = obj.data.root_quat_w[env_ids].clone()
    else:
        root_quat = obj.data.root_link_quat_w[env_ids].clone()

    position_offset_world = torch.as_tensor(
        np.stack(
            (
                rng.uniform(*params["obj_position_noise_range_x"], size=env.num_envs),
                rng.uniform(*params["obj_position_noise_range_y"], size=env.num_envs),
                rng.uniform(*params["obj_position_noise_range_z"], size=env.num_envs),
            ),
            axis=-1,
        ),
        device=root_pos.device,
        dtype=root_pos.dtype,
    )
    randomized_root_pos = root_pos + position_offset_world
    root_pose = torch.cat((randomized_root_pos, root_quat), dim=-1)
    root_velocity = torch.zeros((len(env_ids), 6), device=root_pose.device, dtype=root_pose.dtype)
    if hasattr(obj, "write_root_pose_to_sim"):
        obj.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
    else:
        obj.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
        obj.write_root_com_velocity_to_sim(root_velocity, env_ids=env_ids)
    if hasattr(obj.data, "root_pos_w"):
        obj.data.root_pos_w[env_ids] = randomized_root_pos
    if hasattr(obj.data, "root_link_pos_w"):
        obj.data.root_link_pos_w[env_ids] = randomized_root_pos
    if hasattr(obj.data, "root_lin_vel_w"):
        obj.data.root_lin_vel_w[env_ids] = 0.0
    if hasattr(obj.data, "root_ang_vel_w"):
        obj.data.root_ang_vel_w[env_ids] = 0.0

    joint_ids = env._obj_orientation_joint_ids() if hasattr(env, "_obj_orientation_joint_ids") else [0, 1, 2]
    if len(joint_ids) < 3:
        raise ValueError(f"Expected at least three object orientation joints, got {joint_ids}.")
    obj_joint_pos = obj.data.joint_pos[env_ids].clone()
    joint_vel_source = getattr(obj.data, "joint_vel", torch.zeros_like(obj.data.joint_pos))
    obj_joint_vel = joint_vel_source[env_ids].clone()
    roll_pitch_noise = torch.as_tensor(
        rng.normal(0.0, params["obj_orientation_noise_std"] * 0.2, (len(env_ids), 2)),
        device=obj_joint_pos.device,
        dtype=obj_joint_pos.dtype,
    )
    yaw_noise = torch.as_tensor(
        rng.normal(0.0, params["obj_orientation_noise_std"], (len(env_ids), 1)),
        device=obj_joint_pos.device,
        dtype=obj_joint_pos.dtype,
    )
    obj_joint_pos[:, joint_ids[0:2]] += roll_pitch_noise
    obj_joint_pos[:, joint_ids[2:3]] += yaw_noise
    obj_joint_vel[:, joint_ids[0:3]] = 0.0
    obj.write_joint_state_to_sim(obj_joint_pos, obj_joint_vel, env_ids=env_ids)
    obj.data.joint_pos[env_ids] = obj_joint_pos
    if hasattr(obj.data, "joint_vel"):
        obj.data.joint_vel[env_ids] = obj_joint_vel

    env.table_pose = randomized_root_pos[0].detach().clone().to(dtype=torch.float32)
    env.obj_pose = env.table_pose
    if hasattr(env, "_sync_scene"):
        env._sync_scene()
    return {
        "screwdriver_pos_world": randomized_root_pos.detach().clone(),
        "screwdriver_pos_offset_world": position_offset_world.detach().clone(),
        "obj_joint_pos": obj_joint_pos.detach().clone(),
    }


def make_isaacsim_env(config):
    import gymnasium as gym

    from ccai.utils.isaacsim_screwdriver_recovery import (
        IsaacSimScrewdriverRecoveryEnv,
        get_hand_spec,
    )

    hand = str(config.get("hand", "allegro"))
    spec = get_hand_spec(hand)
    steps_per_action = int(config.get("steps_per_action", 60))
    episode_length_s = float(config.get("episode_length_s", 100000.0))
    seed = config.get("seed", None)
    enable_camera = not bool(config.get("no_video", True))
    sim_device = str(config.get("sim_device", "cuda:0"))
    save_recovery_frames = bool(config.get("save_recovery_frames", True))

    screwdriver_friction_range = _friction_range(
        config,
        "screwdriver_friction",
        (float(config.get("friction_coefficient", 1.0)), float(config.get("friction_coefficient", 1.0))),
    )
    yaw_friction_range = _friction_range(config, "yaw_joint_friction", (0.0, 0.3))

    if hand == "proto5":
        from isaacsim_hand_envs.proto5_screwdriver_turning import get_proto5_screwdriver_turning_rl_env_cfg

        env_cfg = get_proto5_screwdriver_turning_rl_env_cfg(
            num_envs=int(config.get("num_envs", 1)),
            steps_per_action=steps_per_action,
            episode_length_s=episode_length_s,
            device=sim_device,
            enable_camera=enable_camera,
            seed=seed,
            contact_friction_range=screwdriver_friction_range,
            screwdriver_joint_friction_range=yaw_friction_range,
            control_wrist=bool(config.get("proto5_control_wrist", False)),
        )
    else:
        from isaacsim_hand_envs.allegro_screwdriver_turning import get_allegro_screwdriver_turning_rl_env_cfg

        env_cfg = get_allegro_screwdriver_turning_rl_env_cfg(
            num_envs=int(config.get("num_envs", 1)),
            steps_per_action=steps_per_action,
            episode_length_s=episode_length_s,
            device=sim_device,
            enable_camera=enable_camera,
            seed=seed,
            screwdriver_friction_range=screwdriver_friction_range,
            yaw_joint_friction_range=yaw_friction_range,
        )

    env_cfg.sim.create_stage_in_memory = True
    if hasattr(env_cfg, "terminations"):
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None
        if hasattr(env_cfg.terminations, "screwdriver_dropped"):
            env_cfg.terminations.screwdriver_dropped = None

    env = gym.make(spec.gym_id, cfg=env_cfg)
    return IsaacSimScrewdriverRecoveryEnv(
        env,
        hand=hand,
        proto5_control_wrist=bool(config.get("proto5_control_wrist", False)),
        external_wrench_perturb=bool(config["external_wrench_perturb"]),
        rand_pct=float(config["rand_pct"]),
        random_force_magnitude=float(config["random_force_magnitude"]),
        action_repeat=int(config.get("action_repeat", 3)),
        save_recovery_frames=save_recovery_frames,
    )


def make_hardware_env(config):
    from ccai.utils.isaacsim_screwdriver_recovery import HardwareScrewdriverRecoveryEnv

    return HardwareScrewdriverRecoveryEnv(config, device=config.get("sim_device", "cpu"))


def ensure_isaac_victor_envs_compat():
    try:
        from isaac_victor_envs.utils import get_assets_dir  # noqa: F401

        return
    except ImportError:
        pass

    assets_dir = ISAAC_VICTOR_COMPAT_ASSETS_DIR
    if not assets_dir.exists():
        raise ImportError(
            "Could not import isaac_victor_envs and compatibility assets were not found at "
            f"{assets_dir}. Install isaac_victor_envs or provide the IsaacSim hand assets."
        )

    package = sys.modules.get("isaac_victor_envs") or types.ModuleType("isaac_victor_envs")
    package.__path__ = []
    utils_module = types.ModuleType("isaac_victor_envs.utils")
    utils_module.get_assets_dir = lambda: str(assets_dir)
    tasks_module = sys.modules.get("isaac_victor_envs.tasks") or types.ModuleType("isaac_victor_envs.tasks")
    tasks_module.__path__ = []

    sys.modules["isaac_victor_envs"] = package
    sys.modules["isaac_victor_envs.utils"] = utils_module
    sys.modules["isaac_victor_envs.tasks"] = tasks_module


def prepare_legacy_module(config):
    if str(CCAI_PATH) not in sys.path:
        sys.path.insert(0, str(CCAI_PATH))
    if str(ISAACGYM_ARM_ENVS_PATH) not in sys.path:
        sys.path.insert(0, str(ISAACGYM_ARM_ENVS_PATH))
    if str(TORCH_CG_PATH) not in sys.path:
        sys.path.insert(0, str(TORCH_CG_PATH))
    ensure_isaac_victor_envs_compat()

    legacy_path = CCAI_PATH / "examples" / "allegro_screwdriver.py"
    spec = importlib.util.spec_from_file_location("_ccai_legacy_allegro_screwdriver", legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load legacy recovery helpers from {legacy_path}")
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)

    from ccai.allegro_screwdriver_problem import AllegroScrewdriver, Proto5Screwdriver

    if config.get("hand") == "proto5":
        legacy.AllegroScrewdriver = Proto5Screwdriver
    else:
        legacy.AllegroScrewdriver = AllegroScrewdriver
    return legacy


def build_diffpf_action_policy(config, env, device):
    checkpoint = config.get("diffpf_checkpoint")
    if checkpoint in (None, ""):
        return None
    if str(config.get("hand", "allegro")).lower() != "proto5":
        raise ValueError("diffpf_checkpoint normal execution is currently supported only with hand: proto5.")
    checkpoint_path = pathlib.Path(str(checkpoint)).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DiffPF checkpoint not found: {checkpoint_path}")
    eval_path = MODEL_MISMATCH_PATH / "examples" / "evaluation"
    if str(eval_path) not in sys.path:
        sys.path.insert(0, str(eval_path))
    try:
        from screwdriver_diffpf_policy import DiffPFScrewdriverActionPolicy
    except ImportError as exc:
        raise ImportError(
            "Could not import DiffPFScrewdriverActionPolicy. Expected "
            f"{eval_path / 'screwdriver_diffpf_policy.py'} to exist and be import-safe."
        ) from exc
    return DiffPFScrewdriverActionPolicy.from_config(config, env, device)


def build_normal_action_policy(config, env, device):
    return build_diffpf_action_policy(config, env, device)


def build_recovery_diffpf_config(config: dict) -> dict:
    recovery_config = dict(config)
    recovery_config["diffpf_checkpoint"] = config.get("recovery_diffpf_checkpoint")
    for suffix in RECOVERY_DIFFPF_OVERRIDE_SUFFIXES:
        recovery_key = f"recovery_diffpf_{suffix}"
        diffpf_key = f"diffpf_{suffix}"
        value = config.get(recovery_key, None)
        if value is not None:
            recovery_config[diffpf_key] = value
    return recovery_config


def build_recovery_action_policy(config, env, device):
    if str(config.get("recovery_controller", "")).lower() != "diffpf":
        return None
    return build_diffpf_action_policy(build_recovery_diffpf_config(config), env, device)


def main():
    faulthandler.enable(all_threads=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    config = load_config(args)
    ensure_proto5_point_cache(config)
    hardware_mode = config.get("mode") == "hardware"
    simulation_app = None if hardware_mode else launch_isaaclab(config)

    import numpy as np
    import pytorch_kinematics as pk
    import torch
    from tqdm import tqdm

    from ccai.models.management.model_manager import ModelManager
    from ccai.utils.isaacsim_screwdriver_recovery import HardwareVisualizationShim, get_hand_spec, tee_stdout_to_file

    legacy = prepare_legacy_module(config)
    env = make_hardware_env(config) if hardware_mode else make_isaacsim_env(config)
    sim_viz_env = HardwareVisualizationShim(env) if hardware_mode else None
    hand_spec = get_hand_spec(config["hand"])
    normal_action_policy = build_normal_action_policy(config, env, config.get("sim_device", "cuda:0"))
    recovery_action_policy = build_recovery_action_policy(config, env, config.get("sim_device", "cuda:0"))

    if "recovery_controller" not in config:
        config["recovery_controller"] = "csvgd"
    config["obj_dof"] = 3
    config["robot_sdf_path_prefix"] = str(hand_spec.planner_robot_sdf_path_prefix)

    now = ""
    if config.get("timestamp_experiment", False):
        now = "." + datetime.datetime.now().strftime("%m.%d.%y:%I:%M:%S")
    experiment_dir = CCAI_PATH / "data" / "experiments" / f"{config['experiment_name']}{now}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with open(experiment_dir / "config_isaacsim.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    params = config.copy()
    params.pop("controllers")
    params.update(config["controllers"]["csvgd"])
    params["simulator"] = config.get("simulator", "isaacsim")
    params["mode"] = config.get("mode", "simulation")
    params["hand"] = config["hand"]
    params["proto5_control_wrist"] = bool(config.get("proto5_control_wrist", False))
    params["visualize_executed_rollout"] = bool(config.get("visualize_executed_rollout", False))
    params["robot_sdf_path_prefix"] = str(hand_spec.planner_robot_sdf_path_prefix)
    params["controller"] = "csvgd"
    params.setdefault("skip_pregrasp", False)
    params.setdefault("pregrasp_only", False)

    model_manager = ModelManager(config, params, CCAI_PATH)
    trajectory_sampler, trajectory_sampler_orig, classifier = model_manager.load_trajectory_samplers()

    if not hand_spec.urdf_path.exists():
        raise FileNotFoundError(f"Could not find {config['hand']} URDF at {hand_spec.urdf_path}")
    chain = pk.build_chain_from_urdf(open(hand_spec.urdf_path, "r", encoding="utf-8").read())

    proto5_hardware_initialization = _load_proto5_hardware_initialization(config)
    using_dataset_initial_grasp = proto5_hardware_initialization is not None
    if using_dataset_initial_grasp:
        params["skip_pregrasp_stage"] = True
        params["skip_pregrasp"] = True

    pregrasp_states = None
    if params["skip_pregrasp"] and not using_dataset_initial_grasp:
        pregrasp_states, _ = legacy.load_pregrasp_states(config, experiment_dir)

    start_ind = int(config["start_ind"])
    num_episodes = int(config["num_episodes"])
    if "end_ind" in config:
        num_episodes = int(config["end_ind"])

    seed = 0
    base_seed = 0 if config.get("seed", None) is None else int(config["seed"])
    for i in tqdm(range(start_ind, num_episodes)):
        fpath = experiment_dir / "csvgd" / f"trial_{i + 1}"
        fpath.mkdir(parents=True, exist_ok=True)
        with tee_stdout_to_file(fpath / "stdout.log"):
            print(f"\nTrial {i + 1}")
            if using_dataset_initial_grasp:
                if config.get("debug_progress", False):
                    print("debug_progress: resetting hardware env for dataset initial grasp", flush=True)
                env.reset()
                send_proto5_hardware_initial_pose_and_wait(
                    env,
                    proto5_hardware_initialization,
                    device=params["device"],
                )
            elif not params["skip_pregrasp"]:
                if config.get("debug_progress", False):
                    backend_name = "hardware env" if hardware_mode else "IsaacSim env"
                    print(f"debug_progress: resetting {backend_name}", flush=True)
                env.reset()
                object_randomization = None
                if not hardware_mode:
                    object_randomization = randomize_isaacsim_object_start(
                        env,
                        config,
                        np.random.default_rng(base_seed + i),
                    )
                if object_randomization is not None and config.get("debug_progress", False):
                    print(
                        "debug_progress: randomized object start "
                        f"offset_world={object_randomization['screwdriver_pos_offset_world'][0].tolist()} "
                        f"pos_world={object_randomization['screwdriver_pos_world'][0].tolist()}",
                        flush=True,
                    )
            else:
                if config.get("debug_progress", False):
                    print("debug_progress: applying saved pregrasp state", flush=True)
                legacy.apply_saved_pregrasp_state(env, None, pregrasp_states, i, start_ind, params)
            if normal_action_policy is not None and hasattr(normal_action_policy, "reset_from_env"):
                normal_action_policy.reset_from_env(env)
            physical_kwargs = get_recovery_planner_physical_kwargs(env, config)
            params.update(physical_kwargs)
            if config.get("debug_progress", False):
                print(
                    "debug_progress: planner physical kwargs "
                    f"friction_coefficient={physical_kwargs['friction_coefficient']} "
                    f"yaw_joint_friction={physical_kwargs['yaw_joint_friction']} "
                    f"yaw_friction_model_path={physical_kwargs['yaw_friction_model_path']} "
                    f"yaw_inertia_model_path={physical_kwargs['yaw_inertia_model_path']} "
                    f"cache_dir={os.environ.get('PYTORCH_VOLUMETRIC_POINTS_CACHE_DIR')}",
                    flush=True,
                )

            goal = torch.tensor([0, 0, float(config["goal"])])

            params["valve_goal"] = goal.to(device=params["device"])
            params["chain"] = chain.to(device=params["device"])
            params["object_location"] = torch.as_tensor(env.table_pose, device=params["device"], dtype=torch.float32)
            params["controller"] = "csvgd"
            params["perturb_action"] = bool(params.get("perturb_action", False))
            params["trial_index"] = i

            if config.get("debug_progress", False):
                print("debug_progress: entering legacy do_trial", flush=True)
            final_distance_to_goal, dropped = legacy.do_trial(
                env,
                params,
                fpath,
                sim_viz_env=sim_viz_env,
                ros_copy_node=None,
                seed=seed,
                proj_path=None,
                perturb_this_trial=params["perturb_action"],
                trajectory_sampler=trajectory_sampler,
                trajectory_sampler_orig=trajectory_sampler_orig,
                config=config,
                classifier=classifier,
                normal_action_policy=normal_action_policy,
                recovery_action_policy=recovery_action_policy,
            )
            print(f"Trial {i + 1} yaw delta: {final_distance_to_goal}; dropped={dropped}")
        seed += 1

        if not params["skip_pregrasp"]:
            with open(experiment_dir / "pregrasp_states.pkl", "wb") as handle:
                pickle.dump(legacy.all_pregrasp_states, handle)

    if legacy.all_yaw_deltas:
        print("All yaw deltas:", legacy.all_yaw_deltas)
        print("Mean yaw delta:", np.mean(legacy.all_yaw_deltas))
        print("Std yaw delta:", np.std(legacy.all_yaw_deltas))
    env.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
