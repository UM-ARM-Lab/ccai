from isaac_victor_envs.utils import get_assets_dir
try:
    from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
except ImportError:
    AllegroScrewdriverTurningEnv = None
try:
    from isaac_victor_envs.tasks.allegro_ros import RosAllegroScrewdriverTurningEnv
except ImportError:
    RosAllegroScrewdriverTurningEnv = None
    print('No ROS install found, continuing')

import numpy as np
import pickle
import json
from copy import deepcopy

import torch
from torch import nn
import time
import datetime
import copy
import yaml
import pathlib
from functools import partial
import sys

sys.path.append('..')

import pytorch_kinematics as pk



import matplotlib.pyplot as plt
from ccai.utils.allegro_utils import (
    convert_yaw_to_sine_cosine, convert_sine_cosine_to_yaw,
    visualize_trajectory, partial_to_full_state, 
    extract_state_vector
)
from ccai.utils.recovery_utils import (
    create_allegro_screwdriver_problem, create_planner, add_to_dataset, partial_to_full_trajectory,
    full_to_partial_trajectory, create_mode_planner_dict, build_pregrasp_reference_target_kwargs,
    stack_execution_timeseries_for_save
)
from ccai.utils.screwdriver_yaw_wrap import (
    reset_screwdriver_yaw_wrap,
    update_screwdriver_yaw_wrap_after_recovery,
    wrap_screwdriver_task_state_yaw,
)

from ccai.allegro_contact import AllegroManipulationProblem, PositionControlConstrainedSVGDMPC

# Baseline imports
from ccai.baselines.allegro_recovery_baselines import (
    BaselineRecoveryController, BaselineOODDetector, 
    get_baseline_contact_sequence, get_num_envs_for_baseline,
    handle_baseline_trajectory_processing
)

# Module imports for streamlined architecture
from ccai.planning.contact_planning import ContactPlanner
from ccai.execution.trial_executor import TrajectoryExecutor
from ccai.models.management.model_manager import ModelManager

from collections import defaultdict

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

print("CCAI_PATH", CCAI_PATH)

# Degrees of freedom of the object
obj_dof = 3
RECOVERY_STATES_PATH = CCAI_PATH / 'data' / 'recovery_states_screwdriver.pkl'

# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')


class AllegroScrewdriver(AllegroManipulationProblem):
    def __init__(self,
                 start,
                 goal,
                 T,
                 chain,
                 object_location,
                 object_type,
                 world_trans,
                 object_asset_pos,
                 regrasp_fingers=[],
                 contact_fingers=['index', 'middle', 'ring', 'thumb'],
                 friction_coefficient=0.95,
                 obj_dof=1,
                 obj_ori_rep='euler',
                 obj_joint_dim=0,
                 optimize_force=False,
                 turn=False,
                 obj_gravity=False,
                 min_force_dict=None,
                 device='cuda:0',
                 proj_path=None,
                 full_dof_goal=False, 
                 project=False,
                 default_dof_pos=None,
                 contact_constraint_only=False,
                 tactile_controller=False,
                 skip_csvto=False,
                 **kwargs):
        self.tactile_controller = tactile_controller
        self.skip_csvto = skip_csvto
        # Mass of the object. Hardcoded for now.
        self.obj_mass = 0.0851
        self.obj_dof_type = None
        self.object_type = 'screwdriver'
        object_link_name = 'screwdriver_body'
        self.obj_translational_dim = 0
        self.obj_rotational_dim = 3
        self.obj_link_name = object_link_name

        self.contact_points = None
        contact_points_object = None
        if proj_path is not None:
            self.proj_path = proj_path.to(device=device)
        else:
            self.proj_path = None

        # Set default DOF positions if not provided
        if default_dof_pos is None:
            self.default_dof_pos = torch.cat((torch.tensor([[0.0819, 0.3447, 0.7860, 0.7333]]).float().to(device=device),
                                        torch.tensor([[-.0578, 0.7718, 0.5937, 0.7523]]).float().to(device=device),
                                        torch.tensor([[0., 0.5, 0.65, 0.65]]).float().to(device=device),
                                        torch.tensor([[.7946, 0.8216, 0.7075, .8364]]).float().to(device=device)),
                                        dim=1).to(device)
        else:
            self.default_dof_pos = default_dof_pos

        super(AllegroScrewdriver, self).__init__(start=start, goal=goal, T=T, chain=chain,
                                                 object_location=object_location,
                                                 object_type=object_type, world_trans=world_trans,
                                                 object_asset_pos=object_asset_pos,
                                                 regrasp_fingers=regrasp_fingers,
                                                 contact_fingers=contact_fingers,
                                                 friction_coefficient=friction_coefficient,
                                                 obj_dof=obj_dof,
                                                 obj_ori_rep=obj_ori_rep, obj_joint_dim=1,
                                                 optimize_force=optimize_force, device=device,
                                                 turn=turn, obj_gravity=obj_gravity,
                                                 min_force_dict=min_force_dict, 
                                                 full_dof_goal=full_dof_goal,
                                                  contact_points_object=contact_points_object,
                                                  contact_points_dict = self.contact_points,
                                                  project=project,
                                                  contact_constraint_only=contact_constraint_only,
                                                   **kwargs)
        self.friction_coefficient = friction_coefficient

    def _cost(self, xu, rob_link_pts, nearest_robot_pts, start, goal, projected_diffusion=False):
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it

        # Smoothness cost for object degrees of freedom
        smoothness_cost = torch.sum((state[1:, -self.obj_dof:] - state[:-1, -self.obj_dof:]) ** 2)
        
        upright_cost = 0
        if not self.project:
            upright_cost = 500 * torch.sum(
                (state[:, -self.obj_dof:-1] + goal[-self.obj_dof:-1]) ** 2)  # the screwdriver should only rotate in z direction
        return smoothness_cost + upright_cost + super()._cost(xu, rob_link_pts, nearest_robot_pts, start, goal, projected_diffusion=projected_diffusion)

all_yaw_deltas = []
all_pregrasp_states = []
all_hri_diffpf_records = []


def _state_to_numpy(state):
    if torch.is_tensor(state):
        return state.detach().cpu().numpy().copy()
    return np.asarray(state).copy()


def append_unique_recovery_state(state, path=RECOVERY_STATES_PATH):
    return True
    state_np = _state_to_numpy(state).reshape(-1)
    if path.exists():
        with open(path, 'rb') as f:
            recovery_states = pickle.load(f)
    else:
        recovery_states = []

    for saved_state in recovery_states:
        saved_np = _state_to_numpy(saved_state).reshape(-1)
        if saved_np.shape == state_np.shape and np.allclose(saved_np, state_np):
            print(f'Recovery state already logged in {path}')
            return False

    recovery_states.append(state_np)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with open(tmp_path, 'wb') as f:
        pickle.dump(recovery_states, f)
    tmp_path.replace(path)
    print(f'Logged recovery state {len(recovery_states)} to {path}')
    return True


def write_hri_diffpf_records_for_experiment(data, trial_fpath):
    records = data.get('hri_diffpf_records', [])
    if not records:
        return None
    try:
        from model_mismatch.utils.ccai_screwdriver_diffusion_export import write_hri_diffpf_training_hdf5
    except ImportError as exc:
        print(f'Skipping HRI DiffPF training HDF5 export; import failed: {exc}')
        return None

    exported_count = int(data.get('hri_diffpf_records_exported', 0))
    new_records = records[exported_count:]
    if new_records:
        all_hri_diffpf_records.extend(new_records)
        data['hri_diffpf_records_exported'] = len(records)
    if not all_hri_diffpf_records:
        return None

    trial_fpath = pathlib.Path(trial_fpath)
    experiment_dir = trial_fpath.parent.parent if trial_fpath.parent.name == 'csvgd' else trial_fpath.parent
    output_path = experiment_dir / 'proto5_diffpf_training_data.h5'
    written_path = write_hri_diffpf_training_hdf5(all_hri_diffpf_records, output_path)
    print(f'Wrote HRI DiffPF training HDF5 with {len(all_hri_diffpf_records)} rows to {written_path}')
    return written_path


def resolve_pregrasp_states_path(config, experiment_dir):
    configured_path = config.get('pregrasp_states_path', None)
    if configured_path is not None:
        configured_path = pathlib.Path(configured_path)
        if not configured_path.is_absolute():
            configured_path = CCAI_PATH / configured_path
        return configured_path

    default_path = experiment_dir / 'pregrasp_states.pkl'
    if default_path.exists():
        return default_path

    legacy_screwdriver_path = CCAI_PATH / 'data/experiments/allegro_screwdriver_pregrasp_gen/pregrasp_states_screwdriver.pkl'
    if config.get('object_type') == 'screwdriver' and legacy_screwdriver_path.exists():
        return legacy_screwdriver_path

    return default_path


def load_pregrasp_states(config, experiment_dir):
    pregrasp_states_path = resolve_pregrasp_states_path(config, experiment_dir)
    if not pregrasp_states_path.exists():
        raise FileNotFoundError(
            f'skip_pregrasp=True requires saved pregrasp states at {pregrasp_states_path}'
        )
    with open(pregrasp_states_path, 'rb') as f:
        pregrasp_states = pickle.load(f)
    print(f'Loaded {len(pregrasp_states)} pregrasp states from {pregrasp_states_path}')
    return pregrasp_states, pregrasp_states_path


def select_pregrasp_state(pregrasp_states, trial_index, start_ind):
    if isinstance(pregrasp_states, dict):
        for key in (trial_index, trial_index + 1, str(trial_index), str(trial_index + 1)):
            if key in pregrasp_states:
                return pregrasp_states[key], key
        raise IndexError(f'No pregrasp state for trial index {trial_index}')

    if trial_index < len(pregrasp_states):
        return pregrasp_states[trial_index], trial_index

    local_index = trial_index - start_ind
    if 0 <= local_index < len(pregrasp_states):
        return pregrasp_states[local_index], local_index

    raise IndexError(
        f'No pregrasp state for trial index {trial_index}; loaded {len(pregrasp_states)} states'
    )


def apply_saved_pregrasp_state(env, sim_viz_env, pregrasp_states, trial_index, start_ind, params):
    pregrasp_state, state_index = select_pregrasp_state(pregrasp_states, trial_index, start_ind)
    pregrasp_state = torch.as_tensor(pregrasp_state).float().reshape(-1)
    num_fingers = len(params['fingers'])
    expected_dim = 4 * num_fingers + 4
    if pregrasp_state.numel() != expected_dim:
        raise ValueError(
            f'Expected pregrasp state dim {expected_dim}, got {pregrasp_state.numel()} '
            f'for state index {state_index}'
        )

    print(f'Applying saved pregrasp state index {state_index}: {pregrasp_state}')
    if hasattr(env, 'set_pose'):
        if params['mode'] != 'hardware':
            env.reset()
        env.set_pose(pregrasp_state.to(device=env.device))
    else:
        action = pregrasp_state[:4 * num_fingers].reshape(1, -1).to(device=env.device)
        env.step(action)

    if sim_viz_env is not None and hasattr(sim_viz_env, 'set_pose'):
        sim_viz_env.set_pose(pregrasp_state.cpu())


def _flatten_actual_rollout_for_visualization(actual_trajectory, final_state, state_dim):
    frames = []
    for item in actual_trajectory:
        if isinstance(item, list):
            continue
        if item is None:
            continue
        item = item.detach().cpu().float() if torch.is_tensor(item) else torch.as_tensor(item).float()
        if item.numel() == 0:
            continue
        if item.ndim == 1:
            item = item.reshape(1, -1)
        else:
            item = item.reshape(-1, item.shape[-1])
        frames.append(item[:, :state_dim])

    if final_state is not None:
        final_state = final_state.detach().cpu().float() if torch.is_tensor(final_state) else torch.as_tensor(final_state).float()
        frames.append(final_state.reshape(1, -1)[:, :state_dim])

    if not frames:
        return None
    return torch.cat(frames, dim=0)


def _sanitize_temperature_for_path(temperature):
    temperature = 1.0 if temperature is None else float(temperature)
    text = f"{temperature:g}"
    if "e" not in text and "." not in text:
        text = f"{text}.0"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def _recovery_log_run_name(temperature=None):
    timestamp = time.strftime("%Y%m%d_%H%M")
    return f"{timestamp}_temp{_sanitize_temperature_for_path(temperature)}"


def _collision_safe_child_dir(parent, preferred_name):
    parent = pathlib.Path(parent)
    candidate = parent / preferred_name
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        candidate = parent / f"{preferred_name}_run{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _experiment_log_run_dir(controller_dir, temperature=None):
    return _collision_safe_child_dir(
        controller_dir,
        _recovery_log_run_name(temperature),
    )


def _trial_log_run(trial_dir):
    trial_dir = pathlib.Path(trial_dir)
    return trial_dir.parent.name, trial_dir.parent


def save_executed_rollout_visualization(
    fpath,
    actual_trajectory,
    final_state,
    executed_contacts,
    turn_problem,
    num_fingers,
    obj_dof,
    selected_recovery=None,
    temperature=None,
    all_stage=None,
    log_run_name=None,
):
    state_dim = 4 * num_fingers + obj_dof
    rollout = _flatten_actual_rollout_for_visualization(actual_trajectory, final_state, state_dim)
    if rollout is None:
        return None

    viz_fpath = pathlib.Path(fpath) / "executed_rollout"
    img_fpath, gif_fpath = viz_fpath / "img", viz_fpath / "gif"
    img_fpath.mkdir(parents=True, exist_ok=True)
    gif_fpath.mkdir(parents=True, exist_ok=True)

    tmp = torch.zeros((rollout.shape[0], 1), dtype=rollout.dtype)
    traj_for_viz = torch.cat((rollout, tmp), dim=1)
    visualize_trajectory(
        traj_for_viz,
        turn_problem.contact_scenes_for_viz,
        viz_fpath,
        turn_problem.fingers,
        obj_dof + 1,
    )

    run_name, run_dir = _trial_log_run(fpath)
    effective_temperature = (
        selected_recovery.get("recovery_likelihood_temperature")
        if selected_recovery is not None and selected_recovery.get("recovery_likelihood_temperature") is not None
        else (1.0 if temperature is None else temperature)
    )
    metadata = {
        "executed_contacts": list(executed_contacts),
        "num_frames": int(traj_for_viz.shape[0]),
        "trial_dir": str(pathlib.Path(fpath)),
        "log_run_name": log_run_name or (selected_recovery or {}).get("log_run_name") or run_name,
        "log_run_dir": str(run_dir),
        "all_stage": all_stage,
        "recovery_likelihood_temperature": effective_temperature,
        "selected_recovery": selected_recovery,
    }
    with open(viz_fpath / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return viz_fpath


def do_trial(env, params, fpath, sim_viz_env=None, ros_copy_node=None, inits_noise=None, noise_noise=None, sim=None, seed=None,
             proj_path=None, perturb_this_trial=False, trajectory_sampler=None, trajectory_sampler_orig=None, config=None,
             classifier=None, normal_action_policy=None, recovery_action_policy=None):
    global all_yaw_deltas
    debug_progress = bool(params.get('debug_progress', False))
    episode_num_steps = 0
    max_episode_num_steps = 100
    num_fingers = len(params['fingers'])
    if debug_progress:
        print('debug_progress: do_trial get_state start', flush=True)
    state = env.get_state()
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
        if params['mode'] == 'hardware':
            sim_viz_env.frame_fpath = fpath
            sim_viz_env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None
    if debug_progress:
        print('debug_progress: extracting initial state', flush=True)
    start = extract_state_vector(state, num_fingers, params['device'])

    if params.get('external_wrench_perturb', False):
        rand_pct = params.get('rand_pct', 1/3)  # Get from params with default value
        print(f'Random perturbation %: {rand_pct:.2f}')

    # Initialize baseline controller if needed
    baseline_controller = None
    baseline_ood_detector = None
    if 'recovery_controller' in params:
        if debug_progress:
            print('debug_progress: initializing baseline recovery helpers', flush=True)
        baseline_controller = BaselineRecoveryController(env, config, params, trajectory_sampler_orig)
        baseline_ood_detector = BaselineOODDetector(params, trajectory_sampler_orig, 
                                                   getattr(baseline_controller, 'running_cost', None))
    
    # Initialize MPPI controller if using MPPI
    mppi_ctrl = None
    if baseline_controller and baseline_controller.is_mppi_controller():
        mppi_ctrl = baseline_controller.create_mppi_controller()
        mppi_needs_warmup = True
    
    fingers = params['fingers']
    # Minimum magnitude of contact forces for trajectory optimization
    min_force_dict = params.get('min_force_dict')
    if min_force_dict is not None:
        min_force_dict = {finger: float(force) for finger, force in min_force_dict.items()}
    elif params['mode'] == 'hardware':
        min_force_dict = {
            'thumb': 1.,
            'middle': 1.,
            'index': 1.,
        }
    else:
        min_force_dict = {
            'thumb': 1.0,
            'middle': 1.0,
            'index': 1.0,
        }

    goal_pregrasp = params['valve_goal']
    pregrasp_params = copy.deepcopy(params)
    pregrasp_params['warmup_iters'] = 100
    pregrasp_params['contact_only_warmup_iters'] = 0
    pregrasp_params['contact_only_online_iters'] = 0
    pregrasp_params['tactile_controller'] = False
    pregrasp_params['skip_csvto'] = False

    skip_pregrasp_stage = bool(params.get('skip_pregrasp_stage', False))
    if not skip_pregrasp_stage:
        start[-4:] = 0
    pregrasp_reference_target_kwargs = {}
    if params.get('use_pregrasp_reference_targets', False):
        if debug_progress:
            print('debug_progress: building pregrasp reference targets', flush=True)
        pregrasp_reference_target_kwargs = build_pregrasp_reference_target_kwargs(
            pregrasp_params,
            env,
            pregrasp_params['device'],
            AllegroScrewdriver,
        )
    if debug_progress:
        print('debug_progress: creating pregrasp problem', flush=True)
    pregrasp_problem = create_allegro_screwdriver_problem(
        'pregrasp', 
        start[:4 * num_fingers + obj_dof], 
        goal_pregrasp, 
        pregrasp_params, 
        env, 
        pregrasp_params['device'],
        regrasp_fingers=fingers,
        proj_path=proj_path,
        obj_dof=obj_dof,
        AllegroScrewdriver=AllegroScrewdriver,
        **pregrasp_reference_target_kwargs,
    )
    if debug_progress:
        print('debug_progress: creating pregrasp planner', flush=True)
    pregrasp_planner = create_planner(pregrasp_problem, 'pregrasp', pregrasp_params)

    if debug_progress:
        print('debug_progress: creating turn problem', flush=True)
    turn_problem = create_allegro_screwdriver_problem(
        'turn',
        start[:4 * num_fingers + obj_dof],
        params['valve_goal'],
        params,
        env,
        params['device'],
        min_force_dict=min_force_dict,
        proj_path=proj_path,
        AllegroScrewdriver=AllegroScrewdriver
    )
    if debug_progress:
        print('debug_progress: created turn problem', flush=True)

    # Initialize regrasp planners as None
    index_regrasp_planner = None
    thumb_and_middle_regrasp_planner = None
    all_regrasp_planner = None

    model_path = params.get('model_path', None)
    if model_path is not None:
        print('Loaded trajectory sampler')
        trajectory_sampler_orig.model.diffusion_model.classifier = None

    state = env.get_state()
    start = extract_state_vector(state, num_fingers, params['device'])

    actual_trajectory = [start]

    # Initialize executors and managers
    trajectory_executor = TrajectoryExecutor(params, env, sim_viz_env)
    
    def execute_traj(planner, mode, env, goal=None, fname=None, initial_samples=None, recover=False, 
                     start_timestep=0, max_timesteps=None, ctrl=None, mppi_warmup=False,
                     reset_recovery_policy=True):
        """
        Execute a trajectory with the given planner and mode.
        
        Args:
            planner: The planner to use
            mode: Contact mode ('index', 'thumb_middle', 'turn', etc.)
            goal: Goal state
            fname: Filename for saving
            initial_samples: Initial trajectory samples
            recover: Whether this is a recovery execution
            start_timestep: Timestep to start from (for resuming after recovery)
            max_timesteps: Maximum timesteps to execute (for limiting execution after recovery)
        
        Returns:
            actual_trajectory: Executed trajectory
            planned_trajectories: Planned trajectories
            initial_samples: Initial samples used
            sim_rollouts: Simulation rollouts
            optimizer_paths: Optimizer paths
            contact_points: Contact points
            contact_distance: Contact distances
            recover: Whether recovery is needed
            executed_steps: Number of steps executed before OOD detection
            pre_recovery_state: State before recovery
            pre_recovery_likelihood: Likelihood before recovery
        """
        nonlocal episode_num_steps
        was_recovering = recover
        

        # Execute trajectory using TrajectoryExecutor
        actual_trajectory, planned_trajectories, initial_samples, sim_rollouts, optimizer_paths, contact_points, contact_distance, recover, episode_num_steps = trajectory_executor.execute_traj(
            planner=planner,
            mode=mode,
            env=env,
            goal=goal,
            fname=fname,
            initial_samples=initial_samples,
            recover=recover,
            start_timestep=start_timestep,
            max_timesteps=max_timesteps,
            ctrl=ctrl,
            mppi_warmup=mppi_warmup,
            fpath=fpath,
            baseline_controller=baseline_controller,
            baseline_ood_detector=baseline_ood_detector,
            data=data,
            trajectory_sampler=trajectory_sampler,
            trajectory_sampler_orig=trajectory_sampler_orig,
            turn_problem=turn_problem,
            num_fingers=num_fingers,
            obj_dof=obj_dof,
            episode_num_steps=episode_num_steps,
            max_episode_num_steps=max_episode_num_steps,
            min_force_dict=min_force_dict,
            proj_path=proj_path,
            AllegroScrewdriver=AllegroScrewdriver,
            tactile_controller=params.get('tactile_controller', False),
            skip_csvto=params.get('skip_csvto', False),
            normal_action_policy=normal_action_policy,
            recovery_action_policy=recovery_action_policy,
            reset_recovery_policy=reset_recovery_policy,
        )

        if params.get('live_recovery', False) and not was_recovering and recover:
            recovery_state = extract_state_vector(
                env.get_state(), num_fingers, params['device'], slice_end=15
            )
            append_unique_recovery_state(recovery_state)
               
        return actual_trajectory, planned_trajectories, initial_samples, sim_rollouts, optimizer_paths, contact_points, contact_distance, recover

    data = {}
    t_range = params['T']
    if 'T_orig' in params and params['T_orig'] > t_range:
        t_range = params['T_orig']
    for t in range(1, 1 + t_range):
        data[t] = {'plans': [], 'starts': [], 'inits': [], 'init_sim_rollouts': [], 'optimizer_paths': [], 'contact_points': [], 'contact_distance': [], 'contact_state': [], 'contact_plan': []}
    data['pre_action_likelihoods'] = []
    data['final_likelihoods'] = []
    data['csvto_times'] = []
    data['project_times'] = []
    data['all_samples_'] = []
    data['all_likelihoods_'] = []
    data['contact_plan_times'] = []
    data['executed_contacts'] = []
    data['contact_state'] = []
    data['contact_plan'] = []
    data['contact_wrenches'] = []
    data['contact_forces'] = []
    data['contact_points'] = []
        # sample initial trajectory with diffusion model to get contact sequence
    state = env.get_state()
    state = extract_state_vector(state, num_fingers, params['device'])

    def _add_to_dataset(traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, contact_state):
        add_to_dataset(data, traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, contact_state)

    def plan_recovery_contacts_w_model(state):
        return contact_planner.plan_recovery_contacts_w_model(state, contact_state_dict_flip, classifier)

    def plan_recovery_contacts(state, stage):
        return contact_planner.plan_recovery_contacts(state, stage, fpath, all_stage, index_regrasp_planner)



    state = env.get_state()
    state_16 = extract_state_vector(state, num_fingers, params['device'])
    state = state_16[:15]

    contact = None
    state = env.get_state()
    state = extract_state_vector(state, num_fingers, params['device'], slice_end=15)

    executed_contacts = []
    recover = False
    pre_recover = False
    diffpf_recovery_active = False
    stage = 1 if skip_pregrasp_stage else 0
    all_stage = 1 if skip_pregrasp_stage else 0
    done = False
    max_episode_num_steps = 100 if params['mode'] != 'hardware' else 50
    
    # Store initial yaw for tracking rotation progress
    initial_yaw = state[-1].item()
    reset_screwdriver_yaw_wrap(params, state)
    max_stages = 2  # Maximum number of stages for non-live recovery mode

    def should_continue_loop():
        if params.get('live_recovery', False):
            return episode_num_steps < max_episode_num_steps
        return all_stage < max_stages

    contact_planner = None
    if not params.get('live_recovery', False):
        contact_sequence = ['turn'] * (max_stages - 1) # minus 1 because pregrasp will iterate the all_stage counter
    if skip_pregrasp_stage:
        post_pregrasp_state = state_16.clone()

    while should_continue_loop():
        params['current_stage'] = all_stage

        initial_samples = None
        state = env.get_state()
        state = extract_state_vector(state, num_fingers, params['device'], slice_end=15)
        planned = False
        recovery_controller_name = str(params.get('recovery_controller', '')).lower()
        if params.get('live_recovery', False) and recover:
            if recovery_controller_name == 'diffpf':
                contact_sequence = ['diffpf_recovery']
            else:
                contact_sequence = get_baseline_contact_sequence(params, recover=True)
            goal_config = None
            initial_samples = None
            likelihood = None

            if recovery_controller_name == 'diffpf':
                pass
            elif recovery_controller_name == 'mppi':
                # MPPI baseline - no additional planning needed
                pass
            elif params.get('task_model_path', None) and params.get('generate_context', False):
                # Use recovery model to get contact mode

                contact_sequence, goal_config, initial_samples, likelihood, plan_time = plan_recovery_contacts(state, stage)
                goal_config[-1] = state[-1]

                data['all_likelihoods_'].append(likelihood)
                planned = True
            elif params.get('task_model_path', None):
                # Use recovery model to get contact mode
                contact_sequence, goal_config, initial_samples, likelihood, plan_time = plan_recovery_contacts_w_model(state)
                goal_config[-1] = state[-1]
                # goal_config = None
                # initial_samples = None
                planned = True
            else:
                contact_sequence, initial_samples, plan_time = plan_recovery_contacts(state, stage)
                planned = True
        elif params.get('live_recovery', False) and not recover:
            contact_sequence = get_baseline_contact_sequence(params, recover=False)

        if planned:
            print('Plan time:', plan_time)
            data['contact_plan_times'].append(plan_time)
        state = env.get_state()
        state = extract_state_vector(state, num_fingers, params['device'], slice_end=15)
        ori = state[:15][-3:]
        print('Current orientation:', ori)

        if stage == 0:
            orig_torque_perturb = env.external_wrench_perturb if params['mode'] != 'hardware' else False
            if params['mode'] != 'hardware':
                env.set_external_wrench_perturb(False)
            else:
                input('Ready to pregrasp. Press <ENTER> to continue.')
            if not params['skip_pregrasp']:
                contact = 'pregrasp'
                start = env.get_state()['q'].reshape(-1, 4 * num_fingers + 4).to(device=params['device'])[0]
                if debug_progress:
                    print('debug_progress: running pregrasp planner step', flush=True)
                best_traj, _ = pregrasp_planner.step(start[:pregrasp_planner.problem.dx])
                for x in best_traj[:, :4 * num_fingers]:
                    action = x.reshape(-1, 4 * num_fingers).to(device=env.device) # move the rest fingers
                    if getattr(env, 'hand', None) == 'proto5' and hasattr(env, 'set_pose'):
                        if debug_progress:
                            print('debug_progress: setting Proto5 pregrasp pose', flush=True)
                        s = start.clone()
                        s[:4 * num_fingers] = action.reshape(-1)
                        env.set_pose(s.to(device=env.device))
                    else:
                        if debug_progress:
                            print('debug_progress: stepping pregrasp action', flush=True)
                        env.step(action)
                        # After stepping, reset the screwdriver to where it was initially
                        if params['mode'] != 'hardware':
                            s = env.get_state()['q'].reshape(-1, 4 * num_fingers + 4).to(device=params['device'])[0]
                            s[-4:] = start[-4:]
                            env.set_pose(s.to(device=env.device))

            # for _ in range(50):
            #     env._step_sim()
            post_pregrasp_state = env.get_state()['q'].reshape(-1, 4 * num_fingers + 4).to(device=params['device'])[0]
            post_pregrasp_state_for_viz = post_pregrasp_state.clone()
            print(post_pregrasp_state)
            if not params['skip_pregrasp']:
                all_pregrasp_states.append(post_pregrasp_state)
            if params.get('pregrasp_only', False):
                break
            if params['mode'] == 'hardware':
                # print(set_state.shape)
                sim_viz_env.set_pose(post_pregrasp_state_for_viz.cpu())  
                sim_viz_env.write_image()

                state = sim_viz_env.get_state()['q'].reshape(-1).to(device=params['device'])
                print(state[:15][-3:])
                input("Pregrasp complete. Ready to execute. Press <ENTER> to continue.")
            stage += 1
            all_stage += 1
            if params['mode'] != 'hardware' and params.get('external_wrench_perturb', False):
                env.set_external_wrench_perturb(orig_torque_perturb, rand_pct)
            continue
        else:
            contact = contact_sequence.pop(0)
            
        start = extract_state_vector(env.get_state(), num_fingers, params['device'], slice_end=15)
        if stage == 2:
            initial_yaw = start[-1].item()

        data['executed_contacts'].append(contact)
        print(stage, contact)
        torch.cuda.empty_cache()

        contact_state_dict = {
            'all': torch.tensor([0.0, 0.0, 0.0]),
            'index': torch.tensor([0.0, 1.0, 1.0]),
            'thumb_middle': torch.tensor([1.0, 0.0, 0.0]),
            'turn': torch.tensor([1.0, 1.0, 1.0]),
            'diffpf_recovery': torch.tensor([1.0, 1.0, 1.0]),
            'thumb': torch.tensor([1.0, 1.0, 0.0]),
            'middle': torch.tensor([1.0, 0.0, 1.0]),
            # 'mppi': None
        }

        # Make contact_state_dict into a defaultdict that returns None if a key is not found
        contact_state_dict = defaultdict(lambda: None, contact_state_dict)

        contact_state_dict_flip = dict([(tuple(v.numpy()),k) for k, v in contact_state_dict.items()])
        state = env.get_state()
        state = extract_state_vector(state, num_fingers, params['device'], slice_end=15)

        pre_recover = recover
        if contact == 'index':
            _goal = None
            if params.get('task_model_path', None):
                _goal = goal_config
            # If we're recovering and have a saved goal/timesteps, use them
            start_timestep = 0
            max_timesteps = None
            # Execute trajectory
            result = execute_traj(
                index_regrasp_planner, 'index', env, goal=_goal, 
                fname=f'index_regrasp_{all_stage}', initial_samples=initial_samples, 
                recover=recover, start_timestep=start_timestep, max_timesteps=max_timesteps)
            state = env.get_state()
            state = extract_state_vector(state, num_fingers, params['device'], slice_end=15)

            # Backward compatibility with old return format
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, recover = result

            plans = [torch.cat((plan[..., :-6],
                                torch.zeros(*plan.shape[:-1], 3).to(device=params['device']),
                                plan[..., -6:]),
                            dim=-1) for plan in plans]
            traj = torch.cat((traj[..., :-6], torch.zeros(*traj.shape[:-1], 3).to(device=params['device']),
                            traj[..., -6:]), dim=-1)
                                
        elif contact == 'thumb_middle':
            default_pose = post_pregrasp_state.clone()
            default_pose[-3:] = torch.tensor([0, 0, state[-1]])
            
            _goal = None
            if params.get('task_model_path', None):
                _goal = goal_config
            # If we're recovering and have a saved goal/timesteps, use them
            start_timestep = 0
            max_timesteps = None
            result = execute_traj(
                thumb_and_middle_regrasp_planner, 'thumb_middle',
                env,
                goal=_goal, fname=f'thumb_middle_regrasp_{all_stage}', initial_samples=initial_samples, 
                recover=recover, start_timestep=start_timestep, max_timesteps=max_timesteps)
            state = env.get_state()
            state = extract_state_vector(state, num_fingers, params['device'], slice_end=15)
                
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, recover = result
                
            plans = [torch.cat((plan,
                                torch.zeros(*plan.shape[:-1], 6).to(device=params['device'])),
                            dim=-1) for plan in plans]
            traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 6).to(device=params['device'])), dim=-1)

        elif contact == 'all':
            _goal = None
            if params.get('task_model_path', None):
                _goal = goal_config
            result = execute_traj(
                all_regrasp_planner, mode='all', goal=_goal, 
                fname=f'all_regrasp_{all_stage}', initial_samples=initial_samples, 
                recover=recover, start_timestep=start_timestep, max_timesteps=max_timesteps)
            state = env.get_state()
            state = extract_state_vector(state, num_fingers, params['device'], slice_end=15, hardcoded_dim=16)

            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, recover = result

            plans = [torch.cat((plan,
                                torch.zeros(*plan.shape[:-1], 9).to(device=params['device']),
                                ),
                            dim=-1) for plan in plans]
            traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 9).to(device=params['device']),
                            ), dim=-1)

        elif contact == 'turn':
            mppi_ctrl = None
            # Goal is to turn clockwise by 60 degrees
            _goal = torch.tensor([0, 0, state[-1] - np.pi / 2]).to(device=params['device'])
                
            # If we're recovering and have a saved goal/timesteps, use them
            start_timestep = 0
            max_timesteps = None
                
            result = execute_traj(
                None, 'turn', env, goal=_goal, fname=f'turn_{all_stage}', initial_samples=initial_samples, 
                recover=recover, start_timestep=start_timestep, max_timesteps=max_timesteps)
                
            state = env.get_state()
            state = extract_state_vector(state, num_fingers, params['device'], slice_end=15)
            
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, recover = result

        elif contact == 'diffpf_recovery':
            result = execute_traj(
                None,
                'diffpf_recovery',
                env,
                goal=None,
                fname=f'diffpf_recovery_{all_stage}',
                initial_samples=initial_samples,
                recover=recover,
                start_timestep=0,
                max_timesteps=None,
                reset_recovery_policy=not diffpf_recovery_active,
            )

            state = env.get_state()
            state = extract_state_vector(state, num_fingers, params['device'], slice_end=15)
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, recover = result

        elif contact == 'mppi':
            # Use baseline MPPI controller
            if baseline_controller and baseline_controller.is_mppi_controller():
                if mppi_ctrl is None:
                    mppi_ctrl = baseline_controller.create_mppi_controller()
                    mppi_needs_warmup = True
                result = execute_traj(
                    None, mode='mppi', goal=None, fname=f'mppi_{all_stage}', initial_samples=initial_samples,
                    recover=recover, ctrl=mppi_ctrl, mppi_warmup=mppi_needs_warmup)
                traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, recover = result
                mppi_needs_warmup = False
                
                # Handle trajectory processing for MPPI
                traj, plans = handle_baseline_trajectory_processing(traj, plans, contact, params['device'])

        # done = False
        add = not recover or params['live_recovery']
        state = env.get_state()
        state = extract_state_vector(state, num_fingers, params['device'], slice_end=15)
        
        start = state[:4 * num_fingers + obj_dof]
        if params.get('live_recovery', True):
            if (not pre_recover and recover):
                # If we stopped turning because of OOD, then last measured likelihood is post-action
                likelihood = data['pre_action_likelihoods'][-1][-1]
                if len(data['pre_action_likelihoods'][-1]) > 1:
                    data['final_likelihoods'][-1].append(likelihood)
                else:
                    data['final_likelihoods'][-1].append(None)
            elif recovery_controller_name == 'diffpf' and pre_recover:
                pass
            else:
                # If we just recovered, assume we are done. If we are not, next turn will catch it.
                recover = False

            if all_stage > 1 and len(data['final_likelihoods'][-2]) == 0:
                data['final_likelihoods'][-2].append(data['pre_action_likelihoods'][-1][0])

        just_finished_recovery = bool(pre_recover) and not bool(recover)
        if just_finished_recovery:
            update_screwdriver_yaw_wrap_after_recovery(params, start)
        diffpf_recovery_active = bool(recover) and recovery_controller_name == 'diffpf'

        stage += 1
        all_stage += 1

        roll_abs = np.abs(start[-3].item())
        pitch_abs = np.abs(start[-2].item())
        drop_cutoff = np.float32(0.15).item()
        dropped = (roll_abs > drop_cutoff) or (pitch_abs > drop_cutoff)

        if dropped:
            print('Probably dropped the object')
            print(start[-obj_dof:])
            done = True
        data['dropped'] = dropped
        data['dropped_recovery'] = dropped and pre_recover
        
        if recover and not done and params.get('task_diffuse_goal', False) and recovery_controller_name != 'diffpf':
            state = env.get_state()
            state = extract_state_vector(state, num_fingers, params['device'], slice_end=15)
            
            start = state[:4 * num_fingers + obj_dof]
            task_start = wrap_screwdriver_task_state_yaw(params, start)
            start_sine_cosine = convert_yaw_to_sine_cosine(task_start)
            
            # Project the state back into distribution if we are computing recovery trajectories
            pre_project_time = time.perf_counter()
            projected_samples, _, _, _, (all_losses, all_samples, all_likelihoods) = trajectory_sampler_orig.sample(
                16, H=trajectory_sampler_orig.T, start=start_sine_cosine.reshape(1, -1), project=True,
                constraints=torch.ones(16, 3).to(device=params['device'])
            )
            data['project_times'].append(time.perf_counter() - pre_project_time)
            print('Final likelihood:', all_likelihoods[-1])
            
            threshold = params.get('likelihood_threshold', -15)
            if all_likelihoods[-1].mean().item() < threshold:
                print('1 mode projection failed, trying anyway')
            else:
                print('1 mode projection succeeded')
                
            goal = convert_sine_cosine_to_yaw(projected_samples[0][0])[:15]
            goal[-1] = start[-1]

            params_for_recovery = deepcopy(params)
            # if index_regrasp_planner is None:
            #     index_regrasp_problem = AllegroScrewdriver(
            #         start=start[:4 * num_fingers + obj_dof],
            #         goal=goal,
            #         T=params['T'],
            #         chain=params['chain'],
            #         device=params['device'],
            #         object_asset_pos=env.table_pose,
            #         object_location=params['object_location'],
            #         object_type=params['object_type'],
            #         world_trans=env.world_trans,
            #         regrasp_fingers=['index'],
            #         contact_fingers=['middle', 'thumb'],
            #         obj_dof=3,
            #         obj_joint_dim=1,
            #         optimize_force=params['optimize_force'],
            #         default_dof_pos=env.default_dof_pos[:, :16],
            #         obj_gravity=params.get('obj_gravity', False),
            #         min_force_dict=min_force_dict,
            #         full_dof_goal=True,
            #         proj_path=None,
            #         project=True,
            #     )
            #     index_regrasp_planner = PositionControlConstrainedSVGDMPC(index_regrasp_problem, params_for_recovery)

            # if thumb_and_middle_regrasp_planner is None:
            #     thumb_and_middle_regrasp_problem = AllegroScrewdriver(
            #         start=start[:4 * num_fingers + obj_dof],
            #         goal=goal,
            #         T=params['T'],
            #         chain=params['chain'],
            #         device=params['device'],
            #         object_asset_pos=env.table_pose,
            #         object_location=params['object_location'],
            #         object_type=params['object_type'],
            #         world_trans=env.world_trans,
            #         contact_fingers=['index'],
            #         regrasp_fingers=['middle', 'thumb'],
            #         obj_dof=3,
            #         obj_joint_dim=1,        
            #         optimize_force=params['optimize_force'],
            #         default_dof_pos=env.default_dof_pos[:, :16],
            #         obj_gravity=params.get('obj_gravity', False),
            #         min_force_dict=min_force_dict,
            #         full_dof_goal=True,
            #         proj_path=None,
            #         project=True,
            #     )
            #     thumb_and_middle_regrasp_planner = PositionControlConstrainedSVGDMPC(thumb_and_middle_regrasp_problem, params_for_recovery)

            # if all_regrasp_planner is None:
            #     all_regrasp_problem = AllegroScrewdriver(
            #         start=state[:4 * num_fingers + obj_dof],
            #         goal=goal,
            #         T=params['T'],
            #         chain=params['chain'],
            #         device=params['device'],
            #         object_asset_pos=env.obj_pose,
            #         object_location=params['object_location'],
            #         object_type=params['object_type'],
            #         world_trans=env.world_trans,
            #         contact_fingers=[],
            #         regrasp_fingers=['index', 'middle', 'thumb'],
            #         obj_dof=obj_dof,
            #         obj_joint_dim=1,
            #         optimize_force=params['optimize_force'],
            #         default_dof_pos=env.default_dof_pos[:, :16],
            #         obj_gravity=params.get('obj_gravity', False),
            #         min_force_dict=min_force_dict,
            #         full_dof_goal=True,
            #         proj_path=None,
            #         project=True,
            #     )
            #     all_regrasp_planner = PositionControlConstrainedSVGDMPC(all_regrasp_problem, params_for_recovery)

            # mode_planner_dict = {
            #     'all': all_regrasp_planner,
            #     'index': index_regrasp_planner,
            #     'thumb_middle': thumb_and_middle_regrasp_planner,
            # }
            if index_regrasp_planner is None or thumb_and_middle_regrasp_planner is None or all_regrasp_planner is None:
                mode_planner_dict = create_mode_planner_dict(env, params, params['device'], min_force_dict, goal, AllegroScrewdriver)
                index_regrasp_planner = mode_planner_dict['index']
                thumb_and_middle_regrasp_planner = mode_planner_dict['thumb_middle']
                all_regrasp_planner = mode_planner_dict['all']
                
            mode_planner_dict['index'].reset(start, goal=goal)
            mode_planner_dict['thumb_middle'].reset(start, goal=goal)
            mode_planner_dict['all'].reset(start, goal=goal)

            if contact_planner is None:
                contact_planner = ContactPlanner(params, env, trajectory_sampler, trajectory_sampler_orig, 
                                turn_problem,
                                mode_planner_dict)
            
            print('New goal:', goal)

            if torch.allclose(start, goal):
                print('Goal is the same as current state')
                recover = False
        elif recover and not done and params.get('task_model_path', None) and contact_planner is None and recovery_controller_name != 'diffpf':
            contact_planner = ContactPlanner(params, env, trajectory_sampler, trajectory_sampler_orig, 
                                turn_problem,
                                )

        if add:
            _add_to_dataset(traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance,
                            contact_state=contact_state_dict[contact])
        if contact != 'pregrasp' and add:
            actual_trajectory.append(traj)
        # change to numpy and save data

        if params['mode'] != 'hardware':
            pickle.dump(env.wrench_perturb_inds, open(f"{fpath}/wrench_perturb_inds.p", "wb"))
        
        data_save = deepcopy(data)
        for t in range(1, 1 + params['T']):
            try:
                data_save[t]['plans'] = torch.stack(data_save[t]['plans']).cpu().numpy()
                data_save[t]['starts'] = torch.stack(data_save[t]['starts']).cpu().numpy()
                data_save[t]['contact_points'] = torch.stack(data_save[t]['contact_points']).cpu().numpy()
                data_save[t]['contact_distance'] = torch.stack(data_save[t]['contact_distance']).cpu().numpy()
                data_save[t]['contact_state'] = torch.stack(data_save[t]['contact_state']).cpu().numpy()
                data_save[t]['contact_plan'] = torch.stack(data_save[t]['contact_plan']).cpu().numpy()
            except:
                pass
        stack_execution_timeseries_for_save(data_save)
        
        pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
        pickle.dump(data_save, open(f"{fpath}/traj_data.p", "wb"))
        del data_save
        state = env.get_state()
        state = extract_state_vector(state, num_fingers, params['device'], obj_dof=obj_dof, slice_end=15)
        actual_trajectory_save = deepcopy(actual_trajectory)
        actual_trajectory_save.append(state.clone()[: 4 * num_fingers + obj_dof])


        if add:
            with open(f'{fpath.resolve()}/trajectory.pkl', 'wb') as f:
                # Filter empty lists from actual_trajectory_save
                actual_trajectory_save = [i for i in actual_trajectory_save if type(i) != list]
                pickle.dump([i.cpu().numpy() for i in actual_trajectory_save], f)
            selected_recovery = getattr(contact_planner, 'last_chained_recovery_selection', None) if contact_planner is not None else None
            if params.get("visualize_executed_rollout", True):
                save_executed_rollout_visualization(
                    fpath,
                    actual_trajectory,
                    state.clone()[:4 * num_fingers + obj_dof],
                    data.get('executed_contacts', []),
                    turn_problem,
                    num_fingers,
                    obj_dof,
                    selected_recovery=selected_recovery,
                    temperature=params.get("recovery_likelihood_temperature", 1.0),
                    all_stage=all_stage,
                )
        del actual_trajectory_save
        write_hri_diffpf_records_for_experiment(data, fpath)

        if done:
            break
    if (params.get('live_recovery', False) and data['final_likelihoods'] and
            len(data['final_likelihoods'][-1]) == 0 and params['OOD_metric'] != 'q_function'):
        task_state = wrap_screwdriver_task_state_yaw(params, state)
        id, likelihood = trajectory_sampler_orig.check_id(task_state, params['likelihood_num_samples'], threshold=params.get('likelihood_threshold', -15))
        data['final_likelihoods'][-1].append(likelihood)
        data_save = deepcopy(data)
        for t in range(1, 1 + params['T']):
            try:
                data_save[t]['plans'] = torch.stack(data_save[t]['plans']).cpu().numpy()
                data_save[t]['starts'] = torch.stack(data_save[t]['starts']).cpu().numpy()
                data_save[t]['contact_points'] = torch.stack(data_save[t]['contact_points']).cpu().numpy()
                data_save[t]['contact_distance'] = torch.stack(data_save[t]['contact_distance']).cpu().numpy()
                data_save[t]['contact_state'] = torch.stack(data_save[t]['contact_state']).cpu().numpy()
                data_save[t]['contact_plan'] = torch.stack(data_save[t]['contact_plan']).cpu().numpy()
            except:
                pass
        stack_execution_timeseries_for_save(data_save)
        
        pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
        pickle.dump(data_save, open(f"{fpath}/traj_data.p", "wb"))
        del data_save

    state = env.get_state()
    state = extract_state_vector(state, num_fingers, params['device'], slice_end=15)
    selected_recovery = getattr(contact_planner, 'last_chained_recovery_selection', None) if contact_planner is not None else None
    if params.get("visualize_executed_rollout", True):
        save_executed_rollout_visualization(
            fpath,
            actual_trajectory,
            state.clone()[:4 * num_fingers + obj_dof],
            data.get('executed_contacts', []),
            turn_problem,
            num_fingers,
            obj_dof,
            selected_recovery=selected_recovery,
            temperature=params.get("recovery_likelihood_temperature", 1.0),
            all_stage=all_stage,
        )
    final_yaw = state[-1].item()
    print('Final yaw:', final_yaw)
    try:
        print('Initial yaw:', initial_yaw)
    except:
        initial_yaw = final_yaw
    print('Difference:', final_yaw - initial_yaw)
    all_yaw_deltas.append(final_yaw - initial_yaw)
    print('All yaw deltas:', all_yaw_deltas)
    roll_abs = np.abs(state[-3].item())
    pitch_abs = np.abs(state[-2].item())
    drop_cutoff = np.float32(0.15).item()
    dropped = (roll_abs > drop_cutoff) or (pitch_abs > drop_cutoff)
    write_hri_diffpf_records_for_experiment(data, fpath)
    env.reset()
    return final_yaw - initial_yaw, dropped


if __name__ == "__main__":
    # get config. First option is to get the config from the command line.
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/screwdriver/{sys.argv[1]}.yaml').read_text())
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/screwdriver/allegro_screwdriver_TODR_chained_recovery.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/screwdriver/allegro_screwdriver_TODR_N_16.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/screwdriver/allegro_screwdriver_diff_tactile_control_eval.yaml').read_text())
    # Write to log file in the experiment's directory

    # Get datetime
    if config['mode'] == 'hardware':
        import datetime
        now = datetime.datetime.now().strftime("%m.%d.%y:%I:%M:%S")
        now_ = '.' + now
        now = now_
    else:
        now = ''
    experiment_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}{now}')
    pathlib.Path.mkdir(experiment_dir, parents=True, exist_ok=True)
    log_file = experiment_dir / 'log.log'
    log_file.touch()
    sys.stdout = open(log_file, 'w', buffering=1)

    from tqdm import tqdm

    sim_env = None
    ros_copy_node = None

    if 'recovery_controller' not in config:
        config['recovery_controller'] = 'csvgd'
    num_envs = get_num_envs_for_baseline(config)
    
    # default_dof_pos = torch.cat((torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float(),
    #                            torch.tensor([[-0.0535, 0.7626, 0.4006, 1.2064]]).float(),
    #                            torch.tensor([[0,0,0,0]]).float(),
    #                            torch.tensor([[.9830, 0.6005, 0.5771, .8364]]).float()),
    #                            dim=1)
    default_dof_pos = torch.cat((torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float(),
                                torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float(),
                                torch.tensor([[0., 0.0, 0.0, 0.0]]).float(),
                                torch.tensor([[1.2, 0.3, .3, 1.2]]).float()),
                                dim=1)
    if config['mode'] == 'hardware':
        # roslaunch allegro_hand allegro_hand_modified.launch
        from hardware.hardware_env import HardwareEnv
        if RosAllegroScrewdriverTurningEnv is None:
            raise ImportError("RosAllegroScrewdriverTurningEnv requires the legacy IsaacGym/ROS environment.")

        env = HardwareEnv(default_dof_pos[:, :16], 
                          finger_list=config['fingers'], 
                          kp=config['kp'], 
                          obj='blue_screwdriver_catching',
                          mode='relative',
                          gradual_control=True,
                          num_repeat=10)
        env.get_state()
        for _ in range(5):
            root_coor, root_ori = env.obj_reader.get_state_world_frame_pos()
        print('Root coor:', root_coor)
        print('Root ori:', root_ori)
        root_coor = root_coor # convert to meters
        # robot_p = np.array([-0.025, -0.1, 1.33])
        robot_p = np.array([0, -0.095, 1.33])
        root_coor = root_coor + robot_p
        sim_env = RosAllegroScrewdriverTurningEnv(num_envs, control_mode='joint_impedance',
                                 use_cartesian_controller=False,
                                 viewer=True,
                                 steps_per_action=60,
                                 friction_coefficient=2.5,
                                 device=config['sim_device'],
                                 valve=config['object_type'],
                                 video_save_path=img_save_dir,
                                 joint_stiffness=config['kp'],
                                 fingers=config['fingers'],
                                 table_pose=None, # Since I ran the IK before the sim, I shouldn't need to set the table pose. 
                                 gravity=True,
                                #  random_force_magnitude=config.get('random_force_magnitude', 1.5),
                                #  default_dof_pos=default_dof_pos
                                 )
        
        sim, gym, viewer = sim_env.get_sim()
        assert (np.array(sim_env.robot_p) == robot_p).all()
        assert (sim_env.default_dof_pos[:, :16] == default_dof_pos.to(config['sim_device'])).all()

        env.world_trans = sim_env.world_trans
        env.joint_stiffness = sim_env.joint_stiffness
        env.device = sim_env.device
        env.table_pose = sim_env.table_pose
    else:
        if not config['visualize']:
            img_save_dir = None

        if AllegroScrewdriverTurningEnv is None:
            raise ImportError("AllegroScrewdriverTurningEnv requires the legacy IsaacGym environment.")
        env = AllegroScrewdriverTurningEnv(num_envs, control_mode='joint_impedance',
                                           use_cartesian_controller=False,
                                           viewer=config['visualize'],
                                           steps_per_action=60,
                                           friction_coefficient=2.5,
                                           device=config['sim_device'],
                                           video_save_path=img_save_dir,
                                           joint_stiffness=config['kp'],
                                           fingers=config['fingers'],
                                           gradual_control=False,
                                           gravity=True, 
                                           randomize_obj_start=config.get('randomize_obj_start', False),
                                           randomize_rob_start=config.get('randomize_rob_start', False),
                                           external_wrench_perturb=config.get('external_wrench_perturb', False),
                                           force_sensors=config.get('tactile_controller', False)
                                        #    random_force_magnitude=config.get('random_force_magnitude', 1.5),
                                        #    default_dof_pos=default_dof_pos
                                           )



        sim, gym, viewer = env.get_sim()

    state = env.get_state()


    results = {}
   
    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'

    config['obj_dof'] = 3

    screwdriver_asset = f'{get_assets_dir()}/screwdriver/screwdriver.urdf'

    chain = pk.build_chain_from_urdf(open(asset).read())


    partial_to_full_state = partial(partial_to_full_state, fingers=config['fingers'])

    # Get datetime
    # if config['mode'] == 'hardware':
    #     import datetime
    #     now = datetime.datetime.now().strftime("%m.%d.%y:%I:%M:%S")
    #     now_ = '.' + now
    #     now = now_

    #     print('Hardware mode')
    #     print('Datetime:', now)
    #     print('Config:', config)

    # else:
    #     now = ''

    trajectory_sampler = None
    model_path = config.get('model_path', None)
    task_model_path = config.get('task_model_path', None)
    model_path_rl_adjusted = config.get('model_path_rl_adjusted', None)

    params = config.copy()
    params.pop('controllers')
    params.update(config['controllers']['csvgd'])
    
    # Load models using ModelManager
    model_manager = ModelManager(config, params, CCAI_PATH)
    trajectory_sampler, trajectory_sampler_orig, classifier = model_manager.load_trajectory_samplers()

    start_ind = config.get('start_ind', 0)
    pregrasp_states = None
    if params['skip_pregrasp']:
        pregrasp_states, pregrasp_states_path = load_pregrasp_states(config, experiment_dir)
    step_size = 1
    num_episodes = config['num_episodes']
    if 'end_ind' in config:
        num_episodes = config['end_ind']
    seed = 0
    controller_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}{now}/csvgd')
    trial_run_dir = _experiment_log_run_dir(
        controller_dir,
        temperature=params.get("recovery_likelihood_temperature", 1.0),
    )
    pathlib.Path.mkdir(trial_run_dir, parents=True, exist_ok=True)
    for i in tqdm(range(start_ind, num_episodes, step_size)):
        print(f'\nTrial {i+1}')

        if not params['skip_pregrasp']:
            env.reset()
        else:
            apply_saved_pregrasp_state(env, sim_env, pregrasp_states, i, start_ind, params)
        goal = torch.tensor([0, 0, float(config['goal'])]) # Ignore. Deprecated
        # goal = goal + 0.025 * torch.randn(1) + 0.2

        fpath = trial_run_dir / f'trial_{i + 1}'
        if config['mode'] != 'hardware':
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
        # set up params

        params['controller'] = 'csvgd'
        params['valve_goal'] = goal.to(device=params['device'])
        params['chain'] = chain.to(device=params['device'])
        params['object_location'] = torch.tensor([0, 0, 1.205]).to(
            params['device'])

        succ = False
        while not succ:
            perturb_this_trial = params['perturb_action']

            if config['mode']  == 'hardware':
                params['perturb_action'] = False
                perturb_this_trial = False

            if not perturb_this_trial:
                print('No action perturbation this trial')
            # try:
            final_distance_to_goal = do_trial(env, params, fpath, sim_env, ros_copy_node,
                                            seed=seed, proj_path=None, perturb_this_trial=perturb_this_trial,
                                            trajectory_sampler=trajectory_sampler, trajectory_sampler_orig=trajectory_sampler_orig,
                                            config=config, classifier=classifier)
            succ = True
            # except Exception as e:
            #     print(f'Error: {e}')
            seed += 1
        if not params['skip_pregrasp']:
            with open(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}{now}/pregrasp_states.pkl', 'wb') as f:
                pickle.dump(all_pregrasp_states, f)
        print('All yaw deltas:', all_yaw_deltas)
        print('Mean yaw delta:', np.mean(all_yaw_deltas))
        print('Std yaw delta:', np.std(all_yaw_deltas))
        print('Min yaw delta:', np.min(all_yaw_deltas))
        print('Max yaw delta:', np.max(all_yaw_deltas))
        

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
