#!/usr/bin/env python3
"""
IDTO Parameter Tuning Script for Allegro Screwdriver Task

This script performs a grid search over key IDTO parameters to optimize
the screwdriver turning performance. It measures success as the amount
of clockwise rotation achieved (original_yaw - final_yaw).
"""

import pathlib
import time
import numpy as np
import itertools
import pickle
import json
import argparse
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv, AllegroValveTurningEnv
from isaac_victor_envs.utils import get_assets_dir

from pydrake.all import Parser, PdControllerGains, JointActuatorIndex, RigidTransform, RollPitchYaw
from pyidto import FindIdtoResource, ProblemDefinition, SolverParameters
import torch

# Import existing functionality
from python_examples.isaac_bridge import (
    IdtoMpcController,
    action_sequence_from_solution_positions,
    map_isaac_to_drake_positions,
    map_isaac_to_drake_velocities,
    build_screwdriver_nominal_updater,
)
from allegro_screwdriver_idto import (
    build_drake_model_file_for_screwdriver,
    isaac_to_drake_state,
    add_actuators_for_selected_fingers,
    compute_actuated_position_indices,
    interpolate_targets_from_solution,
)

import sys
sys.stdout = open('./tune_idto_parameters.log', 'w')


# Task-specific helper functions
def build_drake_model_files(task: str) -> List[str]:
    """Build Drake model files for the specified task."""
    allegro_urdf = str(pathlib.Path(get_assets_dir()) / "xela_models/allegro_hand_right_obj.urdf")
    if task == 'screwdriver':
        obj_urdf = str(pathlib.Path(get_assets_dir()) / "screwdriver/screwdriver.urdf")
    elif task == 'valve':
        obj_urdf = str(pathlib.Path(get_assets_dir()) / "valve/valve_cross.urdf")
    else:
        raise ValueError(f"Unknown task '{task}'")
    return [allegro_urdf, obj_urdf]


def extract_screwdriver_yaw(state: Dict[str, Any]) -> float:
    """Extract screwdriver yaw from state."""
    if 'screwdriver_angle' in state:
        ori = state['screwdriver_angle']
        return float(ori[0][-1].item()) if hasattr(ori, 'shape') else float(ori[-1])
    return 0.0


def extract_valve_yaw(state: Dict[str, Any]) -> float:
    """Extract valve yaw from state."""
    if 'valve' in state:
        try:
            return float(state['valve'][0][0].item())
        except Exception:
            try:
                return float(state['valve'][0].item())
            except Exception:
                pass
    if 'valve_angle' in state:
        try:
            return float(state['valve_angle'][0][-1].item())
        except Exception:
            pass
    if 'valve_ori' in state:
        try:
            return float(state['valve_ori'][0, -1].item())
        except Exception:
            pass
    return 0.0


def has_dropped(state: Dict[str, Any], task: str = 'screwdriver', threshold: float = 0.35) -> bool:
    """Return True if dropped (screwdriver) or goal reached (valve)."""
    if task == 'valve':
        return False  # Valve cannot drop
    
    if 'screwdriver_ori' not in state:
        return False
    ori = state['screwdriver_ori']
    try:
        roll = float(ori[0, -3].item())
        pitch = float(ori[0, -2].item())
    except Exception:
        try:
            roll = float(ori[-3].item() if hasattr(ori[-3], 'item') else ori[-3])
            pitch = float(ori[-2].item() if hasattr(ori[-2], 'item') else ori[-2])
        except Exception:
            return False
    return (abs(roll) > threshold) or (abs(pitch) > threshold)


def has_reached_goal(state: Dict[str, Any], task: str, initial_yaw: float, goal_tolerance: float = 0.087) -> bool:
    """Check if valve goal has been reached (within tolerance of target angle)."""
    if task == 'screwdriver':
        return False  # Screwdriver uses has_dropped instead
    
    if task == 'valve':
        current_yaw = extract_valve_yaw(state)
        # Goal: 60 degrees clockwise from initialization
        target_yaw = initial_yaw - np.pi/3  # 60 degrees clockwise
        return abs(current_yaw - target_yaw) < goal_tolerance
    
    return False


def post_parse_setup_factory_screwdriver(hand_rpy_deg=(40.0, 0.0, 90.0)):
    """Post-parse setup for screwdriver task."""
    def _setup(plant):
        hand_base = plant.GetFrameByName("allegro_hand_base_link")
        table = plant.GetFrameByName("table")

        hand_rpy = RollPitchYaw(np.deg2rad(hand_rpy_deg[0]),
                                 np.deg2rad(hand_rpy_deg[1]),
                                 np.deg2rad(hand_rpy_deg[2]))
        hand_transform = RigidTransform(hand_rpy, np.array([0, -0.095, 1.33]))
        plant.WeldFrames(plant.world_frame(), hand_base, hand_transform)

        screwdriver_transform = RigidTransform()
        screwdriver_transform.set_translation(np.array([0.0, 0, 1.205]))
        plant.WeldFrames(plant.world_frame(), table, screwdriver_transform)

        add_actuators_for_selected_fingers(plant)
    return _setup


def post_parse_setup_factory_valve():
    """Post-parse setup for valve task."""
    def _setup(plant):
        hand_base = plant.GetFrameByName("allegro_hand_base_link")
        wall = plant.GetFrameByName("wall")
        
        hand_rpy = RollPitchYaw(0, 0, np.pi/2)
        hand_transform = RigidTransform(hand_rpy, np.array([0.02, -0.35, .376]))
        plant.WeldFrames(plant.world_frame(), hand_base, hand_transform)

        valve_transform = RigidTransform()
        valve_transform.set_translation(np.array([0.0, 0, .4]))
        plant.WeldFrames(plant.world_frame(), wall, valve_transform)
        
        add_actuators_for_selected_fingers(plant)
    return _setup


def create_parameterized_problem_ctor_screwdriver(
    num_steps: int,
    q_init: np.ndarray = None,
    Qq_hand: float = 1e-2,
    Qq: float = 1e1,
    Qv_hand: float = 3e-3,
    Qv: float = 1.0,
    hand_R: float = 1e-2,
    screw_r: float = 1e1,
    Qf_q_hand: float = 1e-2,
    Qf_q: float = 1e3,
    Qf_v: float = 1e1,
):
    """
    Create a parameterized ProblemDefinition constructor.
    
    Args:
        num_steps: Horizon steps
        q_init: Initial generalized positions
        Qq_hand: Goal cost for hand joint positions
        Qq: Goal cost for screwdriver angles
        Qv_hand: Goal cost for hand joint velocities
        Qv: Goal cost for screwdriver angular velocities
        hand_R: Control cost for hand joints
        screw_r: Control cost for screwdriver joints
        Qf_q_hand: Final goal cost for hand positions
        Qf_q: Final goal cost for screwdriver angles
        Qf_v: Final goal cost for screwdriver angular velocities
    """
    def q_init_provider(plant):
        return q_init if q_init is not None else np.zeros(plant.num_positions())

    def v_init_provider(plant):
        return np.zeros(plant.num_velocities())
    
    def _ctor(plant) -> ProblemDefinition:
        nq = plant.num_positions()
        nv = plant.num_velocities()
        q0 = q_init_provider(plant)
        v0 = v_init_provider(plant)
        assert q0.shape[0] == nq and v0.shape[0] == nv

        prob = ProblemDefinition()
        prob.num_steps = int(num_steps)
        prob.q_init = np.asarray(q0)
        prob.v_init = np.asarray(v0)

        # Use the parameterized values
        prob.Qq = np.diag(np.array([Qq_hand, Qq_hand, Qq_hand, Qq_hand,
                                    Qq_hand, Qq_hand, Qq_hand, Qq_hand,
                                    Qq_hand, Qq_hand, Qq_hand, Qq_hand,
                                    Qq, Qq, Qq, 0,]))
        prob.Qv = np.diag(np.array([Qv_hand, Qv_hand, Qv_hand, Qv_hand,
                                    Qv_hand, Qv_hand, Qv_hand, Qv_hand,
                                    Qv_hand, Qv_hand, Qv_hand,
                                    Qv, Qv, Qv, 0,]))
        prob.R = np.diag(np.array([hand_R, hand_R, hand_R, hand_R,
                                   hand_R, hand_R, hand_R, hand_R,
                                   hand_R, hand_R, hand_R, hand_R,
                                   screw_r, screw_r, screw_r, 0,]))
        prob.Qf_q = np.diag(np.array([Qf_q_hand, Qf_q_hand, Qf_q_hand, Qf_q_hand,
                                     Qf_q_hand, Qf_q_hand, Qf_q_hand, Qf_q_hand,
                                     Qf_q_hand, Qf_q_hand, Qf_q_hand, Qf_q_hand,
                                     Qf_q, Qf_q, Qf_q, 0,]))
        prob.Qf_v = Qf_v * np.eye(nv)

        q_nom = []
        v_nom = []
        for _ in range(prob.num_steps + 1):
            q_nom.append(q0.copy())
            v_nom.append(np.zeros(nv))
        prob.q_nom = q_nom
        prob.v_nom = v_nom
        return prob

    return _ctor


def create_parameterized_problem_ctor_valve(
    num_steps: int,
    q_init: np.ndarray = None,
    Qq_hand: float = 1e-3,
    Qq: float = 1e0,
    Qv_hand: float = 1e-4,
    Qv: float = 1e-1,
    hand_R: float = 1e-1,
    valve_r: float = 1e2,
    Qf_q_hand: float = 1e-3,
    Qf_q: float = 1e4,
    Qf_v: float = 1e1,
):
    """
    Create a parameterized ProblemDefinition constructor for valve task.
    
    Args:
        num_steps: Horizon steps
        q_init: Initial generalized positions
        Qq_hand: Goal cost for hand joint positions
        Qq: Goal cost for valve angles
        Qv_hand: Goal cost for hand joint velocities
        Qv: Goal cost for valve angular velocities
        hand_R: Control cost for hand joints
        valve_r: Control cost for valve joints
        Qf_q_hand: Final goal cost for hand positions
        Qf_q: Final goal cost for valve angles
        Qf_v: Final goal cost for valve angular velocities
    """
    def q_init_provider(plant):
        return q_init if q_init is not None else np.zeros(plant.num_positions())

    def v_init_provider(plant):
        return np.zeros(plant.num_velocities())
    
    def _ctor(plant) -> ProblemDefinition:
        nq = plant.num_positions()
        nv = plant.num_velocities()
        q0 = q_init_provider(plant)
        v0 = v_init_provider(plant)
        assert q0.shape[0] == nq and v0.shape[0] == nv

        prob = ProblemDefinition()
        prob.num_steps = int(num_steps)
        prob.q_init = np.asarray(q0)
        prob.v_init = np.asarray(v0)

        # Valve cost structure: 12 hand + 1 valve coordinate
        prob.Qq = np.diag(np.array([Qq_hand, Qq_hand, Qq_hand, Qq_hand,
                                    Qq_hand, Qq_hand, Qq_hand, Qq_hand,
                                    Qq_hand, Qq_hand, Qq_hand, Qq_hand,
                                    Qq]))
        prob.Qv = np.diag(np.array([Qv_hand, Qv_hand, Qv_hand, Qv_hand,
                                    Qv_hand, Qv_hand, Qv_hand, Qv_hand,
                                    Qv_hand, Qv_hand, Qv_hand,
                                    Qv]))
        prob.R = np.diag(np.array([hand_R, hand_R, hand_R, hand_R,
                                   hand_R, hand_R, hand_R, hand_R,
                                   hand_R, hand_R, hand_R, hand_R,
                                   valve_r]))
        prob.Qf_q = np.diag(np.array([Qf_q_hand, Qf_q_hand, Qf_q_hand, Qf_q_hand,
                                     Qf_q_hand, Qf_q_hand, Qf_q_hand, Qf_q_hand,
                                     Qf_q_hand, Qf_q_hand, Qf_q_hand, Qf_q_hand,
                                     Qf_q]))
        prob.Qf_v = Qf_v * np.eye(nv)

        q_nom = []
        v_nom = []
        for _ in range(prob.num_steps + 1):
            q_nom.append(q0.copy())
            v_nom.append(np.zeros(nv))
        prob.q_nom = q_nom
        prob.v_nom = v_nom
        return prob

    return _ctor


def get_solver_params(num_threads: int = 8) -> SolverParameters:
    """Get solver parameters for IDTO optimization.
    
    Args:
        num_threads: Number of threads to allocate to this solver instance
    """
    params = SolverParameters()
    params.max_iterations = 1
    params.scaling = True
    params.equality_constraints = True
    params.Delta0 = 1e1
    params.Delta_max = 1e5
    params.num_threads = num_threads
    # Contact params tuned conservatively
    params.contact_stiffness = 100.0
    params.dissipation_velocity = 0.1
    params.smoothing_factor = 0.001
    params.friction_coefficient = 1.0
    params.stiction_velocity = 0.1
    params.verbose = False  # Reduce output during tuning
    return params


def run_single_trial(
    param_config: Dict[str, float], 
    trial_id: int,
    task: str = 'screwdriver',
    max_cycles: int = 200,
    config_base: Dict[str, Any] = None,
    device: str = 'cuda:0',
    num_threads: int = 8
) -> Dict[str, Any]:
    """
    Run a single trial with the given parameter configuration.
    
    Args:
        param_config: Dictionary of parameter values
        trial_id: Trial identifier for logging
        max_cycles: Maximum number of MPC cycles to run
        config_base: Base configuration for the simulation
        device: CUDA device to use for simulation
        num_threads: Number of threads for IDTO solver
        
    Returns:
        Dictionary with trial results including initial/final yaw and performance metrics
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running trial {trial_id} with params: {param_config}")
    
    if config_base is None:
        config_base = {
            'visualize': False,
            'sim_device': 'cpu',
            'kp': 3.0,
            'fingers': ['index', 'middle', 'thumb'],
            'randomize_obj_start': True,
            'randomize_rob_start': True,
            'external_wrench_perturb': True,
            'tactile_controller': False,
        }
    else:
        # Override device in config_base if provided
        config_base = config_base.copy()
        config_base['sim_device'] = device
    
    # Setup environment
    num_envs = 1
    if task == 'screwdriver':
        env = AllegroScrewdriverTurningEnv(
            num_envs, control_mode='joint_impedance',
            use_cartesian_controller=False,
            viewer=config_base['visualize'],
            steps_per_action=6,
            friction_coefficient=2.5,
            device=config_base['sim_device'],
            joint_stiffness=config_base['kp'],
            fingers=config_base['fingers'],
            gradual_control=False,
            gravity=True,
            randomize_obj_start=config_base['randomize_obj_start'],
            randomize_rob_start=False,
            external_wrench_perturb=config_base['external_wrench_perturb'],
            force_sensors=config_base['tactile_controller'],
            reinforcement_learning=False
        )
        env.external_wrench_perturb_rand_pct = 1/60
    elif task == 'valve':
        env = AllegroValveTurningEnv(
            num_envs, control_mode='joint_impedance',
            use_cartesian_controller=False,
            viewer=config_base['visualize'],
            steps_per_action=6,
            friction_coefficient=0.1,
            device=config_base['sim_device'],
            joint_stiffness=config_base['kp'],
            fingers=config_base['fingers'],
            gravity=True,
            randomize_obj_start=config_base['randomize_obj_start'],
            randomize_rob_start=config_base['randomize_rob_start'],
        )
    else:
        raise ValueError(f"Unknown task '{task}'")
    
    sim, gym, viewer = env.get_sim()
    
    # Get initial state
    if task == 'screwdriver':
        initial_state = env.get_state(cap=True, include_velocity=False)
        initial_yaw = initial_state['screwdriver_ori'][0, -1].item()  # Yaw is the last component
    else:  # valve
        initial_state = env.get_state(include_velocity=False)
        initial_yaw = extract_valve_yaw(initial_state)
    
    q_init = torch.cat([
        env.default_dof_pos[0, :8].cpu(),
        env.default_dof_pos[0, 12:].cpu()
    ], dim=0).numpy()

    # Drake model for optimization
    model_files = build_drake_model_files(task)
    opt_dt = 0.05
    horizon_steps = 60

    if task == 'screwdriver':
        post_parse_setup = post_parse_setup_factory_screwdriver()
    else:  # valve
        post_parse_setup = post_parse_setup_factory_valve()

    # Create parameterized problem constructor
    if task == 'screwdriver':
        problem_ctor = create_parameterized_problem_ctor_screwdriver(
            horizon_steps, q_init=q_init, **param_config
        )
    else:  # valve
        problem_ctor = create_parameterized_problem_ctor_valve(
            horizon_steps, q_init=q_init, **param_config
        )
    
    # q_guess ctor
    from python_examples.screwdriver_task import screwdriver_q_guess_ctor_factory
    q_guess_ctor = screwdriver_q_guess_ctor_factory(
        model_files, opt_dt, horizon_steps, post_parse_setup, q_init_=q_init
    )

    # Create MPC controller with specified thread count
    def params_ctor():
        return get_solver_params(num_threads=num_threads)
    
    mpc = IdtoMpcController(
        model_file=model_files,
        opt_dt=opt_dt,
        problem_ctor=problem_ctor,
        params_ctor=params_ctor,
        q_guess_ctor=q_guess_ctor,
        post_parse_setup=post_parse_setup,
    )

    # Task-specific yaw index and goal (computed per trial)
    if task == 'screwdriver':
        yaw_index = -2
        goal_delta_yaw = -np.pi / 2.0 / 6  # Clockwise rotation per horizon
    else:  # valve
        yaw_index = -1
        # Compute absolute goal for this trial, then convert to per-horizon delta
        valve_goal_abs = initial_yaw - (np.pi / 3.0)  # 60 deg clockwise from init
        goal_delta_yaw = valve_goal_abs - initial_yaw  # equals -pi/3
    nominal_updater = build_screwdriver_nominal_updater(goal_delta_yaw, yaw_index)

    # Run MPC cycles
    successful_cycles = 0
    failed_cycles = 0
    
    for i in range(max_cycles):
        # Read Isaac state and map to Drake
        q0, v0 = isaac_to_drake_state(env, mpc.plant, task)
        
        # Solve MPC in Drake
        solution, stats = mpc.step(
            q0, v0, 
            maybe_update_nominal=nominal_updater, 
            set_warm_start=i==0
        )
        
        # Convert solution to position targets
        pos_indices = compute_actuated_position_indices(mpc.plant)
        seq = interpolate_targets_from_solution(solution, pos_indices, 1)

        # Apply to Isaac
        for q_target in seq:
            action = torch.tensor(
                q_target, dtype=torch.float32, device=env.device
            ).reshape(1, -1)
            state = env.step(action)
        
        # Check for failure conditions
        if has_dropped(state, task):
            logger.warning(f"Trial {trial_id} stopped due to drop at cycle {i}")
            break
        if has_reached_goal(state, task, initial_yaw):
            logger.info(f"Trial {trial_id} reached goal at cycle {i}")
            break
            
        # Update warm start
        new_warm_start = solution.q[1:] + [solution.q[-1]]
        if task == 'screwdriver':
            new_warm_start[0] = env.get_state(cap=True, include_velocity=False)['q'][0, :].cpu().numpy()
        else:  # valve
            new_warm_start[0] = env.get_state(include_velocity=False)['q'][0, :].cpu().numpy()
        mpc.warm_start.set_q(new_warm_start)
        
        successful_cycles += 1
    
    # Get final state
    if task == 'screwdriver':
        final_state = env.get_state(cap=True, include_velocity=False)
        final_yaw = final_state['screwdriver_ori'][0, -1].item()
    else:  # valve
        final_state = env.get_state(include_velocity=False)
        final_yaw = extract_valve_yaw(final_state)
    
    # Calculate performance metric
    if task == 'screwdriver':
        # Maximize clockwise rotation
        performance_metric = initial_yaw - final_yaw
    else:  # valve
        # Minimize distance to 60-degree clockwise goal
        performance_metric = abs(final_yaw - valve_goal_abs)
    
    # Cleanup
    if viewer:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    
    result = {
        'trial_id': trial_id,
        'task': task,
        'param_config': param_config,
        'initial_yaw': initial_yaw,
        'final_yaw': final_yaw,
        'performance_metric': performance_metric,
        'successful_cycles': successful_cycles,
        'failed_cycles': failed_cycles,
        'total_cycles': successful_cycles + failed_cycles,
        'success': successful_cycles > 0,
    }
    
    if task == 'screwdriver':
        logger.info(f"Trial {trial_id} completed: clockwise_rotation={performance_metric:.4f}, cycles={successful_cycles}")
    else:  # valve
        logger.info(f"Trial {trial_id} completed: goal_distance={performance_metric:.4f} rad, cycles={successful_cycles}")
    return result


def run_configuration_trials(
    config_idx: int,
    param_config: Dict[str, float],
    task: str,
    num_trials: int,
    max_cycles_per_trial: int,
    config_base: Dict[str, Any],
    device: str = 'cuda:0',
    num_threads: int = 8
) -> List[Dict[str, Any]]:
    """
    Run all trials for a single parameter configuration.
    This function is designed to be run in parallel.
    
    Args:
        config_idx: Configuration index for identification
        param_config: Parameter configuration dictionary
        num_trials: Number of trials to run for this configuration
        max_cycles_per_trial: Maximum MPC cycles per trial
        config_base: Base simulation configuration
        device: CUDA device to use
        num_threads: Number of threads for IDTO solver
        
    Returns:
        List of trial results for this configuration
    """
    # Set up logging for this process
    process_id = os.getpid()
    logger = logging.getLogger(f"config_{config_idx:04d}_pid_{process_id}")
    
    logger.info(f"Starting configuration {config_idx} with {num_trials} trials on device {device}")
    logger.info(f"Configuration parameters: {param_config}")
    
    config_results = []
    
    for trial_idx in range(num_trials):
        trial_id = f"config_{config_idx:04d}_trial_{trial_idx:02d}"
        
        result = run_single_trial(
            param_config=param_config,
            trial_id=trial_id,
            task=task,
            max_cycles=max_cycles_per_trial,
            config_base=config_base,
            device=device,
            num_threads=num_threads,
        )
        
        config_results.append(result)
        
        # Log progress
        successful_trials = [r for r in config_results if r.get('success', False)]
        if successful_trials:
            if task == 'screwdriver':
                rotations = [r['performance_metric'] for r in successful_trials]
                mean_rotation = np.mean(rotations)
                logger.info(f"Config {config_idx}, Trial {trial_idx}: mean_rotation={mean_rotation:.4f} ({len(successful_trials)}/{trial_idx+1} successful)")
            else:  # valve
                distances = [r['performance_metric'] for r in successful_trials]
                mean_dist = np.mean(distances)
                logger.info(f"Config {config_idx}, Trial {trial_idx}: mean_goal_distance={mean_dist:.4f} rad ({len(successful_trials)}/{trial_idx+1} successful)")
    
    # Calculate summary for this configuration
    successful_trials = [r for r in config_results if r.get('success', False)]
    if successful_trials:
        if task == 'screwdriver':
            rotations = [r['performance_metric'] for r in successful_trials]
            mean_rotation = np.mean(rotations)
            std_rotation = np.std(rotations)
            logger.info(f"Config {config_idx} completed: mean_rotation={mean_rotation:.4f}±{std_rotation:.4f} ({len(successful_trials)}/{num_trials} successful)")
        else:  # valve
            distances = [r['performance_metric'] for r in successful_trials]
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            logger.info(f"Config {config_idx} completed: mean_goal_distance={mean_dist:.4f}±{std_dist:.4f} rad ({len(successful_trials)}/{num_trials} successful)")
    else:
        logger.warning(f"Config {config_idx} completed with no successful trials")
    
    return config_results


def generate_parameter_grid(param_ranges: Dict[str, Tuple[float, float, int]]) -> List[Dict[str, float]]:
    """
    Generate a parameter grid using logspace for each parameter.
    
    Args:
        param_ranges: Dictionary mapping parameter names to (min_exp, max_exp, num_points)
                     where values are generated as 10^x for x in [min_exp, max_exp]
    
    Returns:
        List of parameter configuration dictionaries
    """
    param_names = list(param_ranges.keys())
    param_values = []
    
    for param_name, (min_exp, max_exp, num_points) in param_ranges.items():
        values = np.logspace(min_exp, max_exp, num_points, base=10.0)
        param_values.append(values)
    
    # Generate all combinations
    param_combinations = list(itertools.product(*param_values))
    
    # Convert to dictionaries
    param_configs = []
    for combination in param_combinations:
        config = dict(zip(param_names, combination))
        param_configs.append(config)
    
    return param_configs


def run_parameter_tuning(
    param_ranges: Dict[str, Tuple[float, float, int]],
    task: str = 'screwdriver',
    num_trials_per_config: int = 5,
    max_cycles_per_trial: int = 200,
    output_dir: str = "tuning_results",
    config_base: Dict[str, Any] = None,
    parallel: bool = True,
    num_parallel_configs: int = 4,
    threads_per_config: int = 8,
    devices: List[str] = None,
) -> Dict[str, Any]:
    """
    Run parameter tuning over the specified parameter space.
    
    Args:
        param_ranges: Parameter ranges for grid search
        num_trials_per_config: Number of trials per parameter configuration
        max_cycles_per_trial: Maximum MPC cycles per trial
        output_dir: Directory to save results
        config_base: Base configuration for simulation
        parallel: Whether to run configurations in parallel
        num_parallel_configs: Number of configurations to run in parallel
        threads_per_config: Number of threads per configuration (IDTO solver threads)
        devices: List of CUDA devices to use (e.g., ['cuda:0', 'cuda:1']). If None, uses cuda:0 for all
        
    Returns:
        Dictionary with complete tuning results
    """
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    log_file = output_dir / f"tuning_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Setup device allocation
    if devices is None:
        devices = ['cuda:0']
    if len(devices) == 1 and num_parallel_configs > 1:
        # Replicate single device for all parallel processes
        devices = devices * num_parallel_configs
    elif len(devices) < num_parallel_configs:
        # Cycle through available devices
        devices = (devices * ((num_parallel_configs // len(devices)) + 1))[:num_parallel_configs]
    
    logger.info(f"Using devices: {devices[:num_parallel_configs]}")
    
    # Validate thread allocation
    total_threads = num_parallel_configs * threads_per_config
    available_threads = mp.cpu_count()
    if total_threads > available_threads:
        logger.warning(f"Requested {total_threads} total threads but only {available_threads} CPU threads available")
    else:
        logger.info(f"Thread allocation: {num_parallel_configs} configs × {threads_per_config} threads = {total_threads} total threads")
    
    logger.info(f"Starting parameter tuning with {len(param_ranges)} parameters")
    logger.info(f"Parameter ranges: {param_ranges}")
    logger.info(f"Parallel execution: {'enabled' if parallel else 'disabled'}")
    
    # Generate parameter grid
    param_configs = generate_parameter_grid(param_ranges)
    logger.info(f"Generated {len(param_configs)} parameter configurations")
    
    # Run trials
    all_results = []
    total_trials = len(param_configs) * num_trials_per_config
    completed_configs = 0
    
    if parallel and num_parallel_configs > 1:
        # Parallel execution using ProcessPoolExecutor
        logger.info(f"Running {num_parallel_configs} configurations in parallel")
        
        with ProcessPoolExecutor(max_workers=num_parallel_configs) as executor:
            # Submit all configuration jobs
            future_to_config = {}
            
            for config_idx, param_config in enumerate(param_configs):
                device = devices[config_idx % len(devices)]
                
                future = executor.submit(
                    run_configuration_trials,
                    config_idx=config_idx,
                    param_config=param_config,
                    task=task,
                    num_trials=num_trials_per_config,
                    max_cycles_per_trial=max_cycles_per_trial,
                    config_base=config_base,
                    device=device,
                    num_threads=threads_per_config,
                )
                
                future_to_config[future] = (config_idx, param_config)
            
            # Collect results as they complete
            for future in as_completed(future_to_config):
                config_idx, param_config = future_to_config[future]
                
                config_results = future.result()
                all_results.extend(config_results)
                completed_configs += 1
                
                # Log progress
                completed_trials = len(all_results)
                logger.info(f"Configuration {config_idx} completed ({completed_configs}/{len(param_configs)})")
                logger.info(f"Total progress: {completed_trials}/{total_trials} trials completed")
                
                # Save intermediate results periodically
                if completed_configs % 5 == 0:
                    intermediate_file = output_dir / f"intermediate_results_{timestamp}.pkl"
                    with open(intermediate_file, 'wb') as f:
                        pickle.dump(all_results, f)
                    logger.info(f"Saved intermediate results after {completed_configs} configurations")
                    
    else:
        # Sequential execution (original behavior)
        logger.info("Running configurations sequentially")
        
        for config_idx, param_config in enumerate(param_configs):
            logger.info(f"Running configuration {config_idx + 1}/{len(param_configs)}: {param_config}")
            
            device = devices[0] if devices else 'cuda:0'
            
            config_results = run_configuration_trials(
                config_idx=config_idx,
                param_config=param_config,
                task=task,
                num_trials=num_trials_per_config,
                max_cycles_per_trial=max_cycles_per_trial,
                config_base=config_base,
                device=device,
                num_threads=threads_per_config,
            )
            
            all_results.extend(config_results)
            completed_configs += 1
            
            # Save intermediate results periodically
            completed_trials = len(all_results)
            if completed_configs % 5 == 0:
                intermediate_file = output_dir / f"intermediate_results_{timestamp}.pkl"
                with open(intermediate_file, 'wb') as f:
                    pickle.dump(all_results, f)
                logger.info(f"Saved intermediate results: {completed_trials}/{total_trials} trials completed")
    
    # Final trial count
    completed_trials = len(all_results)
    
    # Save final results
    results_summary = {
        'task': task,
        'param_ranges': param_ranges,
        'num_trials_per_config': num_trials_per_config,
        'max_cycles_per_trial': max_cycles_per_trial,
        'parallel': parallel,
        'num_parallel_configs': num_parallel_configs if parallel else 1,
        'threads_per_config': threads_per_config,
        'devices_used': devices[:num_parallel_configs] if parallel else [devices[0] if devices else 'cuda:0'],
        'timestamp': timestamp,
        'total_configs': len(param_configs),
        'completed_configs': completed_configs,
        'total_trials': total_trials,
        'completed_trials': completed_trials,
        'all_results': all_results,
    }
    
    # Save as pickle
    results_file = output_dir / f"tuning_results_{timestamp}.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results_summary, f)
    
    # Save as JSON (without the full results for readability)
    json_summary = {k: v for k, v in results_summary.items() if k != 'all_results'}
    json_file = output_dir / f"tuning_summary_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(json_summary, f, indent=2, default=str)
    
    logger.info(f"Tuning completed. Results saved to {results_file}")
    return results_summary


def main():
    """Main function to run parameter tuning."""
    parser = argparse.ArgumentParser(description='IDTO Parameter Tuning')
    parser.add_argument('--task', type=str, default='screwdriver', choices=['screwdriver', 'valve'],
                       help='Task to tune parameters for')
    parser.add_argument('--samples', type=int, default=3,
                       help='Number of samples per parameter (total configs = samples^num_params)')
    args = parser.parse_args()
    
    task = args.task
    samples = args.samples
    
    # Define parameter ranges for grid search (log base 10)
    if task == 'screwdriver':
        param_ranges = {
            'Qq_hand': (-3, -1, samples),    # 10^-3 to 10^-1
            'Qq': (0, 2, samples),          # 10^0 to 10^2
            'Qv_hand': (-4, -2, samples),   # 10^-4 to 10^-2
            'Qv': (-1, 1, samples),         # 10^-1 to 10^1
            'hand_R': (-3, -1, samples),    # 10^-3 to 10^-1
            'screw_r': (0, 2, samples),     # 10^0 to 10^2
            'Qf_q_hand': (-3, -1, samples), # 10^-3 to 10^-1
            'Qf_q': (2, 4, samples),        # 10^2 to 10^4
            'Qf_v': (0, 2, samples),        # 10^0 to 10^2
        }
    else:  # valve
        param_ranges = {
            'Qq_hand': (-3, -1, samples),    # 10^-3 to 10^-1
            'Qq': (0, 2, samples),          # 10^0 to 10^2
            'Qv_hand': (-4, -2, samples),   # 10^-4 to 10^-2
            'Qv': (-1, 1, samples),         # 10^-1 to 10^1
            'hand_R': (-3, -1, samples),    # 10^-3 to 10^-1
            'valve_r': (0, 2, samples),     # 10^0 to 10^2
            'Qf_q_hand': (-3, -1, samples), # 10^-3 to 10^-1
            'Qf_q': (2, 4, samples),        # 10^2 to 10^4
            'Qf_v': (0, 2, samples),        # 10^0 to 10^2
        }
    
    # Base configuration
    config_base = {
        'visualize': False,
        'sim_device': 'cpu',
        'kp': 3.0,
        'fingers': ['index', 'middle', 'thumb'],
        'randomize_obj_start': True,
        'randomize_rob_start': True,
        'external_wrench_perturb': True,
        'tactile_controller': False,
    }
    
    # Run tuning with parallel execution
    results = run_parameter_tuning(
        param_ranges=param_ranges,
        task=task,
        num_trials_per_config=5,
        max_cycles_per_trial=200,
        output_dir=f"tuning_results_{task}",
        config_base=config_base,
        parallel=True,
        num_parallel_configs=4,
        threads_per_config=8,
        devices=['cpu'],  # Add more devices if available: ['cuda:0', 'cuda:1']
    )
    
    print(f"Tuning completed with {results['completed_trials']} total trials")


if __name__ == "__main__":
    main()
