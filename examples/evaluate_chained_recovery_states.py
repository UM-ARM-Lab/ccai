"""
Evaluate chained recovery planning from saved Allegro screwdriver recovery states.

Dry smoke command:

python examples/evaluate_chained_recovery_states.py --config examples/config/screwdriver/allegro_screwdriver_TODR_chained_recovery.yaml --states data/recovery_states_screwdriver.pkl --start-index 52 --disable-model-compilation
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import pickle
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml


CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
if str(CCAI_PATH) not in sys.path:
    sys.path.insert(0, str(CCAI_PATH))
DEFAULT_CONFIG = CCAI_PATH / "examples/config/screwdriver/allegro_screwdriver_TODR_chained_recovery.yaml"
DEFAULT_STATES = CCAI_PATH / "data/recovery_states_screwdriver.pkl"
OBJ_DOF = 3


def ensure_torch():
    global torch
    if "torch" not in globals():
        import torch as torch_module

        torch = torch_module
    return torch


@dataclass
class EvaluationContext:
    config: Dict[str, Any]
    params: Dict[str, Any]
    env: Any
    trajectory_sampler: Any
    trajectory_sampler_orig: Any
    contact_planner: Any
    trajectory_executor: Any
    turn_problem: Any
    mode_planner_dict: Dict[str, Any]
    min_force_dict: Dict[str, float]
    allegro_screwdriver_cls: Any
    sim: Any = None
    gym: Any = None
    viewer: Any = None
    sim_viz_env: Any = None


SUMMARY_FIELDS = [
    "state_index",
    "contact_sequence",
    "plan_time",
    "predicted_likelihood",
    "actual_likelihood",
    "likelihood_error",
    "initial_likelihood",
    "final_roll",
    "final_pitch",
    "final_yaw",
    "dropped",
    "trial_dir",
]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate chained recovery contact plans from saved 15-D recovery states."
    )
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--states", type=pathlib.Path, default=DEFAULT_STATES)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--disable-model-compilation",
        action="store_true",
        help="Disable torch.compile for trajectory sampler models.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def resolve_path(path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    if path.is_absolute():
        return path
    return CCAI_PATH / path


def load_config(path: pathlib.Path, no_viewer: bool = False) -> Dict[str, Any]:
    with open(resolve_path(path), "r") as f:
        config = yaml.safe_load(f)
    if no_viewer:
        config["visualize"] = False
    return config


def validate_config(config: Dict[str, Any]) -> None:
    if config.get("mode") == "hardware":
        raise ValueError("evaluate_chained_recovery_states.py currently supports simulation mode only.")
    required_truthy = {
        "task_model_path": config.get("task_model_path"),
        "model_path": config.get("model_path"),
        "generate_context": config.get("generate_context"),
        "chained_recovery_contact_search": config.get("chained_recovery_contact_search"),
    }
    missing = [key for key, value in required_truthy.items() if not value]
    if missing:
        raise ValueError(
            "Chained recovery state evaluation requires: " + ", ".join(missing)
        )


def build_params(config: Dict[str, Any]) -> Dict[str, Any]:
    params = config.copy()
    params.pop("controllers", None)
    params.update(config["controllers"]["csvgd"])
    params["controller"] = "csvgd"
    return params


def load_recovery_states(path: pathlib.Path) -> List[torch.Tensor]:
    torch = ensure_torch()
    with open(resolve_path(path), "rb") as f:
        raw_states = pickle.load(f)

    if isinstance(raw_states, dict):
        iterable = [raw_states[key] for key in sorted(raw_states)]
    else:
        iterable = list(raw_states)

    states = []
    for idx, state in enumerate(iterable):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).reshape(-1)
        if state_tensor.numel() != 15:
            raise ValueError(f"Expected saved recovery state {idx} to be 15-D, got {state_tensor.numel()}.")
        states.append(state_tensor)
    return states


def state_range(states: List[torch.Tensor], start_index: int, end_index: Optional[int]) -> range:
    if start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    stop = len(states) if end_index is None else min(end_index, len(states))
    if stop < start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index.")
    return range(start_index, stop)


def set_seed(seed: int) -> None:
    torch = ensure_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def simulator_q(env: Any) -> torch.Tensor:
    torch = ensure_torch()
    q = env.get_state()["q"]
    if not torch.is_tensor(q):
        q = torch.as_tensor(q, dtype=torch.float32)
    return q.reshape(-1)


def apply_recovery_state(env: Any, recovery_state: torch.Tensor) -> torch.Tensor:
    torch = ensure_torch()
    if recovery_state.numel() != 15:
        raise ValueError(f"Recovery state must be 15-D, got {recovery_state.numel()}.")

    env.reset()
    full_q = simulator_q(env).clone()
    if full_q.numel() < 16:
        raise ValueError(f"Simulator q must have at least 16 entries, got {full_q.numel()}.")

    full_q[:15] = recovery_state.to(device=full_q.device, dtype=full_q.dtype)
    target = full_q.to(device=getattr(env, "device", full_q.device))
    try:
        env.set_pose(target, zero_velocity=True)
    except TypeError:
        env.set_pose(target)
    env.zero_obj_velocity()
    return full_q


def current_planner_state(env: Any) -> torch.Tensor:
    return simulator_q(env)[:15].clone()


def likelihood_scalar(value: Any) -> float:
    torch = ensure_torch()
    if isinstance(value, tuple):
        value = value[0]
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1).max().cpu().item())
    return float(np.asarray(value).reshape(-1).max())


def tensor_to_cpu(value: Any) -> Any:
    torch = ensure_torch()
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: tensor_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensor_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tensor_to_cpu(item) for item in value)
    return value


def init_executor_data(params: Dict[str, Any]) -> Dict[str, Any]:
    t_range = params.get("T_orig", params["T"])
    data = {
        t: {
            "plans": [],
            "starts": [],
            "inits": [],
            "init_sim_rollouts": [],
            "optimizer_paths": [],
            "contact_points": [],
            "contact_distance": [],
            "contact_state": [],
        }
        for t in range(1, 1 + t_range)
    }
    data["pre_action_likelihoods"] = []
    data["final_likelihoods"] = []
    data["csvto_times"] = []
    data["project_times"] = []
    data["all_samples_"] = []
    data["all_likelihoods_"] = []
    data["contact_plan_times"] = []
    data["executed_contacts"] = []
    return data


def is_regrasp_mode(mode: str) -> bool:
    return mode in {"index", "middle", "thumb", "thumb_middle", "all"}


def execute_contact_sequence(
    ctx: EvaluationContext,
    contact_sequence: List[str],
    goal_config: Any,
    initial_samples: Any,
    trial_dir: pathlib.Path,
) -> List[Dict[str, Any]]:
    artifacts = []
    data = init_executor_data(ctx.params)
    episode_num_steps = 0
    max_episode_num_steps = 100

    for contact_idx, contact in enumerate(contact_sequence):
        contact_initial_samples = initial_samples if contact_idx == 0 else None
        goal = goal_config if is_regrasp_mode(contact) else None
        planner = ctx.mode_planner_dict.get(contact)
        fname = f"{contact}_{contact_idx}"
        result = ctx.trajectory_executor.execute_traj(
            planner=planner,
            mode=contact,
            env=ctx.env,
            goal=goal,
            fname=fname,
            initial_samples=contact_initial_samples,
            recover=True,
            start_timestep=0,
            max_timesteps=None,
            ctrl=None,
            mppi_warmup=False,
            fpath=trial_dir,
            baseline_controller=None,
            baseline_ood_detector=None,
            data=data,
            trajectory_sampler=ctx.trajectory_sampler,
            trajectory_sampler_orig=ctx.trajectory_sampler_orig,
            turn_problem=ctx.turn_problem,
            num_fingers=len(ctx.params["fingers"]),
            obj_dof=OBJ_DOF,
            episode_num_steps=episode_num_steps,
            max_episode_num_steps=max_episode_num_steps,
            min_force_dict=ctx.min_force_dict,
            proj_path=None,
            AllegroScrewdriver=ctx.allegro_screwdriver_cls,
            tactile_controller=ctx.params.get("tactile_controller", False),
            skip_csvto=ctx.params.get("skip_csvto", False),
        )
        (
            actual_trajectory,
            planned_trajectories,
            used_initial_samples,
            sim_rollouts,
            optimizer_paths,
            contact_points,
            contact_distance,
            recover,
            episode_num_steps,
        ) = result
        artifacts.append(
            {
                "contact": contact,
                "actual_trajectory": tensor_to_cpu(actual_trajectory),
                "planned_trajectories": tensor_to_cpu(planned_trajectories),
                "initial_samples": tensor_to_cpu(used_initial_samples),
                "sim_rollouts": tensor_to_cpu(sim_rollouts),
                "optimizer_paths": tensor_to_cpu(optimizer_paths),
                "contact_points": tensor_to_cpu(contact_points),
                "contact_distance": tensor_to_cpu(contact_distance),
                "recover": recover,
            }
        )

    return artifacts


def evaluate_one_state(
    ctx: EvaluationContext,
    recovery_state: torch.Tensor,
    state_index: int,
    output_dir: pathlib.Path,
) -> Dict[str, Any]:
    trial_dir = output_dir / f"trial_{state_index}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    full_initial_q = apply_recovery_state(ctx.env, recovery_state)
    initial_state = current_planner_state(ctx.env).to(device=ctx.params["device"])
    initial_likelihood = ctx.trajectory_sampler_orig.check_id(
        initial_state,
        ctx.params["likelihood_num_samples"],
        threshold=ctx.params.get("likelihood_threshold", -15),
        likelihood_only=True,
    )

    contact_sequence, goal_config, initial_samples, likelihood, plan_time = ctx.contact_planner.plan_recovery_contacts(
        initial_state,
        stage=0,
        fpath=trial_dir,
        all_stage=0,
        index_regrasp_planner=None,
    )
    contact_sequence = list(contact_sequence)
    predicted_likelihood = likelihood_scalar(likelihood)

    artifacts = execute_contact_sequence(
        ctx,
        contact_sequence=contact_sequence,
        goal_config=goal_config,
        initial_samples=initial_samples,
        trial_dir=trial_dir,
    )

    final_state = current_planner_state(ctx.env).to(device=ctx.params["device"])
    actual_likelihood_raw = ctx.trajectory_sampler_orig.check_id(
        final_state,
        ctx.params["likelihood_num_samples"],
        threshold=ctx.params.get("likelihood_threshold", -15),
        likelihood_only=True,
    )
    initial_likelihood_scalar = likelihood_scalar(initial_likelihood)
    actual_likelihood = likelihood_scalar(actual_likelihood_raw)
    final_roll, final_pitch, final_yaw = [float(x) for x in final_state[-3:].detach().cpu().tolist()]
    dropped = abs(final_roll) > 0.25 or abs(final_pitch) > 0.25

    return {
        "summary": {
            "state_index": state_index,
            "contact_sequence": json.dumps(contact_sequence),
            "plan_time": float(plan_time),
            "predicted_likelihood": predicted_likelihood,
            "actual_likelihood": actual_likelihood,
            "likelihood_error": actual_likelihood - predicted_likelihood,
            "initial_likelihood": initial_likelihood_scalar,
            "final_roll": final_roll,
            "final_pitch": final_pitch,
            "final_yaw": final_yaw,
            "dropped": dropped,
            "trial_dir": str(trial_dir),
        },
        "record": {
            "state_index": state_index,
            "initial_state": tensor_to_cpu(recovery_state),
            "full_initial_q": tensor_to_cpu(full_initial_q),
            "goal_config": tensor_to_cpu(goal_config),
            "initial_samples": tensor_to_cpu(initial_samples),
            "predicted_likelihood": tensor_to_cpu(likelihood),
            "initial_likelihood": tensor_to_cpu(initial_likelihood),
            "actual_likelihood": tensor_to_cpu(actual_likelihood_raw),
            "final_state": tensor_to_cpu(final_state),
            "executed_contacts": contact_sequence,
            "executor_artifacts": artifacts,
            "trial_dir": str(trial_dir),
        },
    }


def write_summary(rows: List[Dict[str, Any]], output_dir: pathlib.Path) -> None:
    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_results(records: List[Dict[str, Any]], output_dir: pathlib.Path) -> None:
    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(records, f)


def setup_context(config: Dict[str, Any]) -> EvaluationContext:
    from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
    from isaac_victor_envs.utils import get_assets_dir

    torch = ensure_torch()
    import pytorch_kinematics as pk

    from ccai.execution.trial_executor import TrajectoryExecutor
    from ccai.models.management.model_manager import ModelManager
    from ccai.planning.contact_planning import ContactPlanner
    from ccai.utils.recovery_utils import (
        create_allegro_screwdriver_problem,
        create_mode_planner_dict,
        get_contact_state_mappings,
    )
    from examples.allegro_screwdriver import AllegroScrewdriver

    get_contact_state_mappings()

    params = build_params(config)
    requested_device = str(params.get("device", "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        params["device"] = "cpu"
    elif requested_device == "cuda:1" and torch.cuda.device_count() <= 1:
        params["device"] = "cuda:0"

    default_dof_pos = torch.cat(
        (
            torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float(),
            torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float(),
            torch.tensor([[0.0, 0.0, 0.0, 0.0]]).float(),
            torch.tensor([[1.2, 0.3, 0.3, 1.2]]).float(),
        ),
        dim=1,
    )
    del default_dof_pos

    img_save_dir = None if not config.get("visualize", False) else CCAI_PATH / "data/experiments/videos"
    env = AllegroScrewdriverTurningEnv(
        1,
        control_mode="joint_impedance",
        use_cartesian_controller=False,
        viewer=config.get("visualize", False),
        steps_per_action=60,
        friction_coefficient=2.5,
        device=config["sim_device"],
        video_save_path=img_save_dir,
        joint_stiffness=config["kp"],
        fingers=config["fingers"],
        gradual_control=False,
        gravity=True,
        randomize_obj_start=False,
        randomize_rob_start=False,
        external_wrench_perturb=False,
        force_sensors=config.get("tactile_controller", False),
    )
    sim, gym, viewer = env.get_sim()

    config["obj_dof"] = OBJ_DOF
    asset = f"{get_assets_dir()}/xela_models/allegro_hand_right.urdf"
    chain = pk.build_chain_from_urdf(open(asset).read())

    goal = torch.tensor([0, 0, float(config["goal"])])
    params["valve_goal"] = goal.to(device=params["device"])
    params["chain"] = chain.to(device=params["device"])
    params["object_location"] = torch.tensor([0, 0, 1.205]).to(params["device"])

    model_manager = ModelManager(config, params, CCAI_PATH)
    trajectory_sampler, trajectory_sampler_orig, _ = model_manager.load_trajectory_samplers()

    min_force_dict = {"thumb": 1.0, "middle": 1.0, "index": 1.0}
    start = current_planner_state(env).to(device=params["device"])
    turn_problem = create_allegro_screwdriver_problem(
        "turn",
        start,
        params["valve_goal"],
        params,
        env,
        params["device"],
        min_force_dict=min_force_dict,
        proj_path=None,
        AllegroScrewdriver=AllegroScrewdriver,
    )
    mode_planner_dict = create_mode_planner_dict(
        env,
        params,
        params["device"],
        min_force_dict,
        start,
        AllegroScrewdriver,
    )
    contact_planner = ContactPlanner(
        params,
        env,
        trajectory_sampler,
        trajectory_sampler_orig,
        turn_problem,
        mode_planner_dict,
    )
    trajectory_executor = TrajectoryExecutor(params, env, None)

    return EvaluationContext(
        config=config,
        params=params,
        env=env,
        trajectory_sampler=trajectory_sampler,
        trajectory_sampler_orig=trajectory_sampler_orig,
        contact_planner=contact_planner,
        trajectory_executor=trajectory_executor,
        turn_problem=turn_problem,
        mode_planner_dict=mode_planner_dict,
        min_force_dict=min_force_dict,
        allegro_screwdriver_cls=AllegroScrewdriver,
        sim=sim,
        gym=gym,
        viewer=viewer,
    )


def cleanup_context(ctx: EvaluationContext) -> None:
    if ctx.gym is None:
        return
    if ctx.viewer is not None:
        ctx.gym.destroy_viewer(ctx.viewer)
    if ctx.sim is not None:
        ctx.gym.destroy_sim(ctx.sim)


def evaluate_states(
    config_path: pathlib.Path,
    states_path: pathlib.Path,
    output_dir: Optional[pathlib.Path] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
    no_viewer: bool = False,
    disable_model_compilation: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    config = load_config(config_path, no_viewer=no_viewer)
    if disable_model_compilation:
        config["compile_models"] = False
    validate_config(config)

    if output_dir is None:
        output_dir = CCAI_PATH / "data/experiments" / f"{config['experiment_name']}_recovery_state_eval"
    else:
        output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = setup_context(config)
    set_seed(seed)
    states = load_recovery_states(states_path)
    rows = []
    records = []
    try:
        for idx in state_range(states, start_index, end_index):
            result = evaluate_one_state(ctx, states[idx], idx, output_dir)
            rows.append(result["summary"])
            records.append(result["record"])
            write_summary(rows, output_dir)
            write_results(records, output_dir)
    finally:
        cleanup_context(ctx)

    return {"summary_rows": rows, "records": records, "output_dir": output_dir}


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    result = evaluate_states(
        config_path=args.config,
        states_path=args.states,
        output_dir=args.output_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        no_viewer=args.no_viewer,
        disable_model_compilation=args.disable_model_compilation,
        seed=args.seed,
    )
    print(f"Wrote {len(result['summary_rows'])} rows to {result['output_dir'] / 'summary.csv'}")
    print(f"Wrote result records to {result['output_dir'] / 'results.pkl'}")


if __name__ == "__main__":
    main(sys.argv[1:])
