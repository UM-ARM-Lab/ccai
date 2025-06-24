from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
try:
    from isaac_victor_envs.tasks.allegro_ros import RosAllegroScrewdriverTurningEnv
except:
    print('No ROS install found, continuing')

import numpy as np
import pickle
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
    full_to_partial_trajectory,
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
                 **kwargs):
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

def do_trial(env, params, fpath, sim_viz_env=None, ros_copy_node=None, inits_noise=None, noise_noise=None, sim=None, seed=None,
             proj_path=None, perturb_this_trial=False, trajectory_sampler=None, trajectory_sampler_orig=None, config=None, classifier=None):
    
    episode_num_steps = 0
    max_episode_num_steps = 100
    num_fingers = len(params['fingers'])
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
    start = extract_state_vector(state, num_fingers, params['device'])

    if params.get('external_wrench_perturb', False):
        rand_pct = params.get('rand_pct', 1/3)  # Get from params with default value
        print(f'Random perturbation %: {rand_pct:.2f}')

    # Initialize baseline controller if needed
    baseline_controller = None
    baseline_ood_detector = None
    if 'recovery_controller' in params:
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
    min_force_dict = None
    if params['mode'] == 'hardware':
        min_force_dict = {
            'thumb': 1.,
            'middle': 1.,
            'index': 1.,
        }
    else:
        min_force_dict = {
            'thumb': .5,
            'middle': .5,
            'index': .5,
        }

    goal_pregrasp = params['valve_goal']
    pregrasp_params = copy.deepcopy(params)
    pregrasp_params['warmup_iters'] = 80

    start[-4:] = 0
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
        AllegroScrewdriver=AllegroScrewdriver
    )
    pregrasp_planner = create_planner(pregrasp_problem, pregrasp_params)

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

    actual_trajectory = []

    # Initialize executors and managers
    trajectory_executor = TrajectoryExecutor(params, env, sim_viz_env)
    
    def execute_traj(planner, mode, goal=None, fname=None, initial_samples=None, recover=False, 
                     start_timestep=0, max_timesteps=None, ctrl=None, mppi_warmup=False):
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
        

        # Execute trajectory using TrajectoryExecutor
        actual_trajectory, planned_trajectories, initial_samples, sim_rollouts, optimizer_paths, contact_points, contact_distance, recover, episode_num_steps = trajectory_executor.execute_traj(
            planner=planner,
            mode=mode,
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
            AllegroScrewdriver=AllegroScrewdriver
        )
               
        return actual_trajectory, planned_trajectories, initial_samples, sim_rollouts, optimizer_paths, contact_points, contact_distance, recover

    data = {}
    t_range = params['T']
    if 'T_orig' in params and params['T_orig'] > t_range:
        t_range = params['T_orig']
    for t in range(1, 1 + t_range):
        data[t] = {'plans': [], 'starts': [], 'inits': [], 'init_sim_rollouts': [], 'optimizer_paths': [], 'contact_points': [], 'contact_distance': [], 'contact_state': []}
    data['pre_action_likelihoods'] = []
    data['final_likelihoods'] = []
    data['csvto_times'] = []
    data['project_times'] = []
    data['all_samples_'] = []
    data['all_likelihoods_'] = []
    data['contact_plan_times'] = []
    data['executed_contacts'] = []
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
    stage = 0
    all_stage = 0
    done = False
    max_episode_num_steps = 100 if params['mode'] != 'hardware' else 50
    max_stages = 2  # Maximum number of stages for non-live recovery mode

    def should_continue_loop():
        if params.get('live_recovery', False):
            return episode_num_steps < max_episode_num_steps
        return all_stage < max_stages

    contact_planner = None
    if not params.get('live_recovery', False):
        contact_sequence = ['turn'] * (max_stages - 1) # minus 1 because pregrasp will iterate the all_stage counter

    while should_continue_loop():

        initial_samples = None
        state = env.get_state()
        state = extract_state_vector(state, num_fingers, params['device'], slice_end=15)
        planned = False
        if params.get('live_recovery', False) and recover:
            contact_sequence = get_baseline_contact_sequence(params, recover=True)
            goal_config = None
            initial_samples = None
            likelihood = None

            if params['recovery_controller'] == 'mppi':
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
            contact = 'pregrasp'
            start = env.get_state()['q'].reshape(-1, 4 * num_fingers + 4).to(device=params['device'])[0]
            best_traj, _ = pregrasp_planner.step(start[:pregrasp_planner.problem.dx])
            for x in best_traj[:, :4 * num_fingers]:
                action = x.reshape(-1, 4 * num_fingers).to(device=env.device) # move the rest fingers
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
            post_pregrasp_state = post_pregrasp_state[:15]
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
        data['executed_contacts'].append(contact)
        print(stage, contact)
        torch.cuda.empty_cache()

        contact_state_dict = {
            'all': torch.tensor([0.0, 0.0, 0.0]),
            'index': torch.tensor([0.0, 1.0, 1.0]),
            'thumb_middle': torch.tensor([1.0, 0.0, 0.0]),
            'turn': torch.tensor([1.0, 1.0, 1.0]),
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
                index_regrasp_planner, mode='index', goal=_goal, 
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
                thumb_and_middle_regrasp_planner, mode='thumb_middle',
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
            # Goal is to turn clockwise by 90 degrees
            _goal = torch.tensor([0, 0, state[-1] - np.pi / 2]).to(device=params['device'])
                
            # If we're recovering and have a saved goal/timesteps, use them
            start_timestep = 0
            max_timesteps = None
                
            result = execute_traj(
                None, mode='turn', goal=_goal, fname=f'turn_{all_stage}', initial_samples=initial_samples, 
                recover=recover, start_timestep=start_timestep, max_timesteps=max_timesteps)
                
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
            else:
                # If we just recovered, assume we are done. If we are not, next turn will catch it.
                recover = False

            if all_stage > 1 and len(data['final_likelihoods'][-2]) == 0:
                data['final_likelihoods'][-2].append(data['pre_action_likelihoods'][-1][0])

        stage += 1
        all_stage += 1

        roll_abs = np.abs(start[-3].item())
        pitch_abs = np.abs(start[-2].item())
        drop_cutoff = .35
        dropped = (roll_abs > drop_cutoff) or (pitch_abs > drop_cutoff)

        if dropped:
            print('Probably dropped the object')
            print(start[-obj_dof:])
            done = True
        data['dropped'] = dropped
        data['dropped_recovery'] = dropped and pre_recover
        
        if recover and not done and params.get('task_diffuse_goal', False):
            state = env.get_state()
            state = extract_state_vector(state, num_fingers, params['device'], slice_end=15)
            
            start = state[:4 * num_fingers + obj_dof]
            start_sine_cosine = convert_yaw_to_sine_cosine(start)
            
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
            if index_regrasp_planner is None:
                index_regrasp_problem = AllegroScrewdriver(
                    start=start[:4 * num_fingers + obj_dof],
                    goal=goal,
                    T=params['T'],
                    chain=params['chain'],
                    device=params['device'],
                    object_asset_pos=env.table_pose,
                    object_location=params['object_location'],
                    object_type=params['object_type'],
                    world_trans=env.world_trans,
                    regrasp_fingers=['index'],
                    contact_fingers=['middle', 'thumb'],
                    obj_dof=3,
                    obj_joint_dim=1,
                    optimize_force=params['optimize_force'],
                    default_dof_pos=env.default_dof_pos[:, :16],
                    obj_gravity=params.get('obj_gravity', False),
                    min_force_dict=min_force_dict,
                    full_dof_goal=True,
                    proj_path=None,
                    project=True,
                )
                index_regrasp_planner = PositionControlConstrainedSVGDMPC(index_regrasp_problem, params_for_recovery)

            if thumb_and_middle_regrasp_planner is None:
                thumb_and_middle_regrasp_problem = AllegroScrewdriver(
                    start=start[:4 * num_fingers + obj_dof],
                    goal=goal,
                    T=params['T'],
                    chain=params['chain'],
                    device=params['device'],
                    object_asset_pos=env.table_pose,
                    object_location=params['object_location'],
                    object_type=params['object_type'],
                    world_trans=env.world_trans,
                    contact_fingers=['index'],
                    regrasp_fingers=['middle', 'thumb'],
                    obj_dof=3,
                    obj_joint_dim=1,        
                    optimize_force=params['optimize_force'],
                    default_dof_pos=env.default_dof_pos[:, :16],
                    obj_gravity=params.get('obj_gravity', False),
                    min_force_dict=min_force_dict,
                    full_dof_goal=True,
                    proj_path=None,
                    project=True,
                )
                thumb_and_middle_regrasp_planner = PositionControlConstrainedSVGDMPC(thumb_and_middle_regrasp_problem, params_for_recovery)

            if all_regrasp_planner is None:
                all_regrasp_problem = AllegroScrewdriver(
                    start=state[:4 * num_fingers + obj_dof],
                    goal=goal,
                    T=params['T'],
                    chain=params['chain'],
                    device=params['device'],
                    object_asset_pos=env.obj_pose,
                    object_location=params['object_location'],
                    object_type=params['object_type'],
                    world_trans=env.world_trans,
                    contact_fingers=[],
                    regrasp_fingers=['index', 'middle', 'thumb'],
                    obj_dof=obj_dof,
                    obj_joint_dim=1,
                    optimize_force=params['optimize_force'],
                    default_dof_pos=env.default_dof_pos[:, :16],
                    obj_gravity=params.get('obj_gravity', False),
                    min_force_dict=min_force_dict,
                    full_dof_goal=True,
                    proj_path=None,
                    project=True,
                )
                all_regrasp_planner = PositionControlConstrainedSVGDMPC(all_regrasp_problem, params_for_recovery)

            mode_planner_dict = {
                'all': all_regrasp_planner,
                'index': index_regrasp_planner,
                'thumb_middle': thumb_and_middle_regrasp_planner,
            }
            index_regrasp_planner.reset(start, goal=goal)
            thumb_and_middle_regrasp_planner.reset(start, goal=goal)
            all_regrasp_planner.reset(start, goal=goal)

            if contact_planner is None:
                contact_planner = ContactPlanner(params, env, trajectory_sampler, trajectory_sampler_orig, 
                                turn_problem,
                                mode_planner_dict)
            
            print('New goal:', goal)



            if torch.allclose(start, goal):
                print('Goal is the same as current state')
                recover = False
        elif recover and not done and params.get('task_model_path', None) and contact_planner is None:
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
            except:
                pass
        
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
        del actual_trajectory_save

        if done:
            break
    if params.get('live_recovery', False) and len(data['final_likelihoods'][-1]) == 0 and params['OOD_metric'] != 'q_function':
        id, likelihood = trajectory_sampler_orig.check_id(state, params['likelihood_num_samples'], threshold=params.get('likelihood_threshold', -15))
        data['final_likelihoods'][-1].append(likelihood)
        data_save = deepcopy(data)
        for t in range(1, 1 + params['T']):
            try:
                data_save[t]['plans'] = torch.stack(data_save[t]['plans']).cpu().numpy()
                data_save[t]['starts'] = torch.stack(data_save[t]['starts']).cpu().numpy()
                data_save[t]['contact_points'] = torch.stack(data_save[t]['contact_points']).cpu().numpy()
                data_save[t]['contact_distance'] = torch.stack(data_save[t]['contact_distance']).cpu().numpy()
                data_save[t]['contact_state'] = torch.stack(data_save[t]['contact_state']).cpu().numpy()
            except:
                pass
        
        pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
        pickle.dump(data_save, open(f"{fpath}/traj_data.p", "wb"))
        del data_save

    env.reset()
    return 0


if __name__ == "__main__":
    # get config. First option is to get the config from the command line.
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/screwdriver/{sys.argv[1]}.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/screwdriver/allegro_screwdriver_csvto_only.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/screwdriver/allegro_screwdriver_TODR_contact_constraint_only.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/screwdriver/touchlegro_screwdriver_csvto_recovery_hardware_hri.yaml').read_text())

    from tqdm import tqdm

    sim_env = None
    ros_copy_node = None

    if 'recovery_controller' not in config:
        config['recovery_controller'] = 'csvgd'
    num_envs = get_num_envs_for_baseline(config)
    
    default_dof_pos = torch.cat((torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float(),
                                torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float(),
                                torch.tensor([[0., 0.5, 0.65, 0.65]]).float(),
                                torch.tensor([[1.2, 0.3, 0.3, 1.2]]).float()),
                                dim=1)
    if config['mode'] == 'hardware':
        # roslaunch allegro_hand allegro_hand_modified.launch
        from hardware.hardware_env_hri import HardwareEnv

        env = HardwareEnv(default_dof_pos[:, :16], 
                          finger_list=config['fingers'], 
                          kp=config['kp'], 
                          obj='blue_screwdriver',
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
                                 random_force_magnitude=config.get('random_force_magnitude', 1.5),
                                 default_dof_pos=default_dof_pos
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
                                           random_force_magnitude=config.get('random_force_magnitude', 1.5),
                                           default_dof_pos=default_dof_pos
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
    if config['mode'] == 'hardware':
        import datetime
        now = datetime.datetime.now().strftime("%m.%d.%y:%I:%M:%S")
        now_ = '.' + now
        now = now_

        print('Hardware mode')
        print('Datetime:', now)
        print('Config:', config)

    else:
        now = ''

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
    step_size = 1
    num_episodes = config['num_episodes']
    if 'end_ind' in config:
        num_episodes = config['end_ind']
    for i in tqdm(range(start_ind, num_episodes, step_size)):
        print(f'\nTrial {i+1}')


        env.reset()
        goal = torch.tensor([0, 0, float(config['goal'])]) # Ignore. Deprecated
        # goal = goal + 0.025 * torch.randn(1) + 0.2

        fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}{now}/csvgd/trial_{i + 1}')
        if config['mode'] != 'hardware':
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
        # set up params

        if torch.cuda.device_count() == 1 and torch.cuda.current_device() == 1:
            params['device'] = 'cuda:0'
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
            final_distance_to_goal = do_trial(env, params, fpath, sim_env, ros_copy_node,
                                            seed=i, proj_path=None, perturb_this_trial=perturb_this_trial,
                                            trajectory_sampler=trajectory_sampler, trajectory_sampler_orig=trajectory_sampler_orig,
                                            config=config, classifier=classifier)
            succ = True

        print(results)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

