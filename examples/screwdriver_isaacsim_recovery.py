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
import pathlib
import pickle
import sys

import yaml


CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
DOCUMENTS_PATH = CCAI_PATH.parent
ISAACSIM_HAND_ENVS_PATH = DOCUMENTS_PATH / "github" / "isaacsim-hand-envs"
ISAACGYM_ARM_ENVS_PATH = DOCUMENTS_PATH / "github" / "isaacgym-arm-envs"
DEFAULT_CONFIG_PATH = CCAI_PATH / "examples" / "config" / "proto5" / "proto_screwdriver_csvto_TODR_recovery_data_gen.yaml"


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
    parser.add_argument("--hand", choices=("allegro", "proto5"), default=None)
    parser.add_argument("--headless", type=_bool_from_cli, default=None)
    parser.add_argument("--no_video", type=_bool_from_cli, default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--sim_device", type=str, default=None)
    parser.add_argument("--proto5_control_wrist", action="store_true", default=None)
    parser.add_argument("--steps_per_action", type=int, default=None)
    parser.add_argument("--start_ind", type=int, default=None)
    parser.add_argument("--end_ind", type=int, default=None)
    parser.add_argument("--skip_pregrasp", type=_bool_from_cli, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--debug_progress", action="store_true")
    return parser.parse_args()


def load_config(args) -> dict:
    config_path = args.config
    if not config_path.is_absolute():
        config_path = CCAI_PATH / config_path
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["config_path"] = str(config_path)
    config["simulator"] = "isaacsim"
    config["mode"] = "simulation"

    for key in (
        "hand",
        "headless",
        "no_video",
        "num_envs",
        "sim_device",
        "steps_per_action",
        "start_ind",
        "end_ind",
        "skip_pregrasp",
        "experiment_name",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.proto5_control_wrist is not None:
        config["proto5_control_wrist"] = bool(args.proto5_control_wrist)

    config.setdefault("hand", "allegro")
    config.setdefault("headless", True)
    config.setdefault("no_video", True)
    config.setdefault("num_envs", 1)
    config.setdefault("sim_device", "cuda:0")
    config.setdefault("proto5_control_wrist", False)
    config.setdefault("steps_per_action", 60)
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
    )


def prepare_legacy_module(config):
    if str(CCAI_PATH) not in sys.path:
        sys.path.insert(0, str(CCAI_PATH))
    if str(ISAACGYM_ARM_ENVS_PATH) not in sys.path:
        sys.path.insert(0, str(ISAACGYM_ARM_ENVS_PATH))

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


def main():
    faulthandler.enable(all_threads=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    config = load_config(args)
    simulation_app = launch_isaaclab(config)

    import numpy as np
    import pytorch_kinematics as pk
    import torch
    from tqdm import tqdm

    from ccai.models.management.model_manager import ModelManager
    from ccai.utils.isaacsim_screwdriver_recovery import get_hand_spec

    legacy = prepare_legacy_module(config)
    env = make_isaacsim_env(config)
    hand_spec = get_hand_spec(config["hand"])

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
    params["simulator"] = "isaacsim"
    params["mode"] = "simulation"
    params["hand"] = config["hand"]
    params["proto5_control_wrist"] = bool(config.get("proto5_control_wrist", False))
    params["robot_sdf_path_prefix"] = str(hand_spec.planner_robot_sdf_path_prefix)
    params["controller"] = "csvgd"
    params.setdefault("skip_pregrasp", False)

    model_manager = ModelManager(config, params, CCAI_PATH)
    trajectory_sampler, trajectory_sampler_orig, classifier = model_manager.load_trajectory_samplers()

    if not hand_spec.urdf_path.exists():
        raise FileNotFoundError(f"Could not find {config['hand']} URDF at {hand_spec.urdf_path}")
    chain = pk.build_chain_from_urdf(open(hand_spec.urdf_path, "r", encoding="utf-8").read())

    pregrasp_states = None
    if params["skip_pregrasp"]:
        pregrasp_states, _ = legacy.load_pregrasp_states(config, experiment_dir)

    start_ind = int(config["start_ind"])
    num_episodes = int(config["num_episodes"])
    if "end_ind" in config:
        num_episodes = int(config["end_ind"])

    seed = 0
    try:
        for i in tqdm(range(start_ind, num_episodes)):
            print(f"\nTrial {i + 1}")
            if not params["skip_pregrasp"]:
                if config.get("debug_progress", False):
                    print("debug_progress: resetting IsaacSim env", flush=True)
                env.reset()
            else:
                if config.get("debug_progress", False):
                    print("debug_progress: applying saved pregrasp state", flush=True)
                legacy.apply_saved_pregrasp_state(env, None, pregrasp_states, i, start_ind, params)

            goal = torch.tensor([0, 0, float(config["goal"])])
            fpath = experiment_dir / "csvgd" / f"trial_{i + 1}"
            fpath.mkdir(parents=True, exist_ok=True)

            params["valve_goal"] = goal.to(device=params["device"])
            params["chain"] = chain.to(device=params["device"])
            params["object_location"] = torch.as_tensor(env.table_pose, device=params["device"], dtype=torch.float32)
            params["controller"] = "csvgd"
            params["perturb_action"] = bool(params.get("perturb_action", False))

            if config.get("debug_progress", False):
                print("debug_progress: entering legacy do_trial", flush=True)
            final_distance_to_goal, dropped = legacy.do_trial(
                env,
                params,
                fpath,
                sim_viz_env=None,
                ros_copy_node=None,
                seed=seed,
                proj_path=None,
                perturb_this_trial=params["perturb_action"],
                trajectory_sampler=trajectory_sampler,
                trajectory_sampler_orig=trajectory_sampler_orig,
                config=config,
                classifier=classifier,
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
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
