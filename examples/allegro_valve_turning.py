from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroValveTurningEnv
try:
    from isaac_victor_envs.tasks.allegro_ros import RosAllegroValveTurningEnv
except:
    print('No ROS install found, continuing')

import numpy as np
import pickle as pkl
import pickle
from statsmodels.distributions.empirical_distribution import ECDF
from copy import deepcopy

import torch
import time
import datetime
import copy
import yaml
import pathlib
from functools import partial
from pprint import pprint
import sys
import os
sys.path.append('..')

import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
from torch.func import vmap, jacrev, hessian, jacfwd
# import pytorch3d.transforms as tf

import matplotlib.pyplot as plt
from ccai.utils.allegro_utils import *
# from allegro_valve_roll import AllegroValveTurning, AllegroContactProblem, PositionControlConstrainedSVGDMPC, \
#    add_trajectories, add_trajectories_hardware

from ccai.allegro_contact import AllegroManipulationProblem, PositionControlConstrainedSVGDMPC, add_trajectories, \
    add_trajectories_hardware
# from ccai.mpc.diffusion_policy import Diffusion_Policy, DummyProblem
from scipy.spatial.transform import Rotation as R

from ccai.models.trajectory_samplers_sac import TrajectorySampler
from ccai.models.contact_samplers import GraphSearch, Node

from model import LatentDiffusionModel

from diffusion_mcts import DiffusionMCTS

from ccai.trajectory_shortcut import shortcut_trajectory

from baselines.allegro_valve_turning import RunningCostSafeRL, TerminalCostDiffusionLikelihood
from baselines.dynamics_model import DynamicsModel
from baselines.mppi_planner import MPPIPlanner

from collections import defaultdict
from pytorch_mppi import MPPI


CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

print("CCAI_PATH", CCAI_PATH)

obj_dof = 1
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')
# sys.stdout = open('./examples/logs/allegro_valve_recovery_data_gen_.1_len_thresh_40.log', 'w', buffering=1)

def vector_cos(a, b):
    return torch.dot(a.reshape(-1), b.reshape(-1)) / (torch.norm(a.reshape(-1)) * torch.norm(b.reshape(-1)))

def euler_to_quat(euler):
    matrix = tf.euler_angles_to_matrix(euler, convention='XYZ')
    quat = tf.matrix_to_quaternion(matrix)
    return quat

def euler_to_angular_velocity(current_euler, next_euler):
    current_quat = euler_to_quat(current_euler)
    next_quat = euler_to_quat(next_euler)
    dquat = next_quat - current_quat
    con_quat = - current_quat  # conjugate
    con_quat[..., 0] = current_quat[..., 0]
    omega = 2 * tf.quaternion_raw_multiply(dquat, con_quat)[..., 1:]
    # TODO: quaternion and its negative are the same, but it is not true for angular velocity. Might have some bug here 
    return omega

def convert_yaw_to_sine_cosine(xu):
    """
    xu is shape (N, T, 36)
    Replace the yaw in xu with sine and cosine and return the new xu
    """
    yaw = xu[FULL_DOF-1]
    sine = torch.sin(yaw)
    cosine = torch.cos(yaw)
    xu_new = torch.cat([xu[:(FULL_DOF-1)], cosine.unsqueeze(-1), sine.unsqueeze(-1), xu[FULL_DOF:]], dim=-1)
    return xu_new

def convert_sine_cosine_to_yaw(xu):
    """
    xu is shape (N, T, 37)
    Replace the sine and cosine in xu with yaw and return the new xu
    """
    sine = xu[..., FULL_DOF]
    cosine = xu[..., FULL_DOF-1]
    yaw = torch.atan2(sine, cosine)
    xu_new = torch.cat([xu[..., :(FULL_DOF-1)], yaw.unsqueeze(-1), xu[..., (FULL_DOF+1):]], dim=-1)
    return xu_new
    
class AllegroValve(AllegroManipulationProblem):
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
                 friction_coefficient=0.95*.1,
                #  friction_coefficient=0.5,
                #  friction_coefficient=1000,
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
                 test_recovery_trajectory=False, **kwargs):
        self.obj_mass = 0.0851
        self.obj_dof_type = None
        self.object_type = 'valve'
        if obj_dof == 3:
            object_link_name = 'screwdriver_body'
            self.obj_translational_dim = 0
            self.obj_rotational_dim = 3
        elif obj_dof == 1:
            object_link_name = 'valve'
            self.obj_translational_dim = 0
            self.obj_rotational_dim = 1
        elif obj_dof == 6:
            object_link_name = 'card'
            self.obj_translational_dim = 2
            self.obj_rotational_dim = 1
        self.obj_link_name = object_link_name


        self.contact_points = None
        contact_points_object = None
        if proj_path is not None:
            self.proj_path = proj_path.to(device=device)
        else:
            self.proj_path = None

        super(AllegroValve, self).__init__(start=start, goal=goal, T=T, chain=chain,
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
                                                   **kwargs)
        self.friction_coefficient = friction_coefficient

    def _cost(self, xu, rob_link_pts, nearest_robot_pts, start, goal, projected_diffusion=False):
        # TODO: check if the addtional term of the smoothness cost and running goal cost is necessary
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it

        smoothness_cost = torch.sum((state[1:, -self.obj_dof:] - state[:-1, -self.obj_dof:]) ** 2)
        upright_cost = 0
        return smoothness_cost + super()._cost(xu, rob_link_pts, nearest_robot_pts, start, goal, projected_diffusion=projected_diffusion)

def do_trial(env, params, fpath, sim_viz_env=None, ros_copy_node=None, inits_noise=None, noise_noise=None, sim=None, seed=None,
             proj_path=None, perturb_this_trial=False):
    episode_num_steps = 0
    num_fingers = len(params['fingers'])
    state = env.get_state()
    action_list = []
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
        if params['mode'] == 'hardware':
            sim_viz_env.frame_fpath = fpath
            sim_viz_env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None
    start = state['q'].reshape(-1, 4 * num_fingers + obj_dof).to(device=params['device'])[0]
    initial_angle = start[-1]
    goal_yaw = start[-1] - np.pi / 3
    if params.get('external_wrench_perturb', False):
        rand_pct = (np.random.rand()) / 3
        print(f'Random perturbation %: {rand_pct:.2f}')
        
    mppi_ctrl = None
    # index finger is used for stability
    if 'index' in params['fingers']:
        fingers = params['fingers']
    else:
        fingers = ['index'] + params['fingers']


    # else:
    min_force_dict = None
    if params['mode'] == 'hardware':
        min_force_dict = {
            'thumb': .5,
            'middle': .5,
            'index': .5,
        }
    else:
        min_force_dict = {
            'thumb': 1,
            'middle': 1,
            'index': 1,
        }

    # if params.get('compute_recovery_trajectory', False):
    #     goal_pregrasp = start[-4: -1]
    # else:
    goal_pregrasp = params['valve_goal']
    pregrasp_params = copy.deepcopy(params)
    pregrasp_params['warmup_iters'] = 80

    pregrasp_problem = AllegroValve(
        start=start[:4 * num_fingers + obj_dof],
        goal=start[-obj_dof:],
        T=2,
        chain=pregrasp_params['chain'],
        device=pregrasp_params['device'],
        object_asset_pos=env.obj_pose,
        object_location=pregrasp_params['object_location'],
        object_type=pregrasp_params['object_type'],
        world_trans=env.world_trans,
        regrasp_fingers=fingers,
        contact_fingers=[],
        obj_dof=obj_dof,
        obj_joint_dim=1,
        optimize_force=pregrasp_params['optimize_force'],
        default_dof_pos=env.default_dof_pos[:, :16],
        obj_gravity=pregrasp_params.get('obj_gravity', False),
        full_dof_goal=False,
        proj_path=proj_path,
    )
    pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, pregrasp_params)

    turn_problem = AllegroValve(
        start=start[:4 * num_fingers + obj_dof],
        goal=params['valve_goal'],
        T=params['T_orig'],
        chain=params['chain'],
        device=params['device'],
        object_asset_pos=env.obj_pose,
        object_location=params['object_location'],
        object_type=params['object_type'],
        world_trans=env.world_trans,
        contact_fingers=['index', 'middle', 'thumb'],
        obj_dof=obj_dof,
        obj_joint_dim=1,
        optimize_force=params['optimize_force'],
        default_dof_pos=env.default_dof_pos[:, :16],
        turn=True,
        obj_gravity=params.get('obj_gravity', False),
        min_force_dict=min_force_dict,
        full_dof_goal=False,
        proj_path=proj_path,
        project=False,
    )

    all_regrasp_planner = None
    index_regrasp_planner = None
    middle_regrasp_planner = None
    thumb_regrasp_planner = None
    turn_planner = None

    if model_path is not None:
        # if not params.get('live_recovery', False):
        #     index_regrasp_problem.model = trajectory_sampler_orig
        #     thumb_and_middle_regrasp_problem.model = trajectory_sampler_orig
        print('Loaded trajectory sampler')
        trajectory_sampler_orig.model.diffusion_model.classifier = None
        trajectory_sampler_orig.model.diffusion_model.cutoff = params['project_threshold']

    state = env.get_state()
    start = state['q'].reshape(-1, 4 * num_fingers + obj_dof).to(device=params['device'])[0]
    if params['exclude_index']:
        turn_problem_fingers = copy.copy(params['fingers'])
        turn_problem_fingers.remove('index')
        turn_problem_start = start[4:4 * num_fingers + obj_dof]
    else:
        turn_problem_fingers = params['fingers']
        turn_problem_start = start[:4 * num_fingers + obj_dof]

    actual_trajectory = []

    if params['exclude_index']:
        num_fingers_to_plan = num_fingers - 1
    else:
        num_fingers_to_plan = num_fingers
    info_list = []
    
    def execute_traj(planner, mode, goal=None, fname=None, initial_samples=None, recover=False, 
                     start_timestep=0, max_timesteps=None, ctrl=None, mppi_warmup=False):
        """
        Execute a trajectory with the given planner and mode.
        
        Args:
            planner: The planner to use
            mode: Contact mode ('index', 'thumb', 'middle', 'turn', etc.)
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
        data['pre_action_likelihoods'].append([])
        data['final_likelihoods'].append([])
        data['action_perturbations'].append([])
        data['csvto_times'].append([])
        orig_torque_perturb = env.external_wrench_perturb if params['mode'] != 'hardware' else False
        # if recover and params['mode'] != 'hardware':
        #     env.set_external_wrench_perturb(False)
        # Initialize variables that might be referenced before assignment
        pre_recovery_state = None
        pre_recovery_likelihood = None
        
        # reset planner
        state = env.get_state()
        state = state['q'].reshape(-1, 4 * num_fingers + obj_dof)[0].to(device=params['device'])
        initial_angle = state[-1]
        planned_trajectories = []
        actual_trajectory = []
        optimizer_paths = []
        contact_points = {
        }
        contact_distance = {
        }
        plans = None

        mode_fpath = f'{fpath}/{fname}'
        pathlib.Path.mkdir(pathlib.Path(mode_fpath), parents=True, exist_ok=True)

        if recover and params['recovery_controller'] != 'mppi':
            id_check, final_likelihood = True, None
        elif recover and params['recovery_controller'] == 'mppi':
            id_check, final_likelihood = False, None
        elif not params.get('live_recovery', False):
            id_check, final_likelihood = True, None
        else:
            if params['OOD_metric'] == 'likelihood':
                id_check, final_likelihood = trajectory_sampler_orig.check_id(state, 8, threshold=params.get('likelihood_threshold', -15), yaw_idx=12)
            elif params['OOD_metric'] == 'q_function':
                id_check = True
                final_likelihood = None
        if final_likelihood is not None:
            data['pre_action_likelihoods'][-1].append(final_likelihood)

        dropped = False

        
        dropped = dropped and params['recovery_controller'] == 'mppi'
        if dropped:
            print('dropped')

        cur_yaw = state[-1].item()
        
        cutoff_degrees = 5
        cutoff_radians = np.deg2rad(cutoff_degrees)
        done = False
        if (cur_yaw - goal_yaw) < cutoff_radians:
            print('Reached goal yaw')
            done = True
        # # If we are generating task data (there is no diff_init), we don't want to return
        done = done and params['diff_init']

        if done:
            if planner is not None:
                planner.problem.data = {}
            if len(actual_trajectory) > 0:
                actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=params['device'])

            return actual_trajectory, planned_trajectories, initial_samples, None, None, None, None, False
        if params['recovery_controller'] == 'mppi' and recover and id_check:
            print('MPPI returned state to ID. Exiting recovery loop')
            if len(actual_trajectory) > 0:
                actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=params['device'])

            return actual_trajectory, planned_trajectories, initial_samples, None, None, None, None, False
        elif not (params['recovery_controller'] == 'mppi' and recover) and not id_check or dropped:
            # State is OOD. Save state and likelihood but DON'T collect data for RL yet.
            # We will collect it only AFTER recovery completes
            pre_recovery_state = state[:4 * num_fingers + obj_dof].clone()
            pre_recovery_likelihood = final_likelihood
            
            # State is OOD. Return and move to recovery pipeline
            if planner is not None:
                planner.problem.data = {}
            if len(actual_trajectory) > 0:
                actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=params['device'])

            # Zero obj velocity
            env.zero_obj_velocity()
            # Return how many steps we've executed for resuming later
            return actual_trajectory, planned_trajectories, initial_samples, None, None, None, None, True
        elif dropped:
            print('dropped')
            if len(actual_trajectory) > 0:
                actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=params['device'])

            return actual_trajectory, planned_trajectories, initial_samples, None, None, None, None, False
  

        # generate context from mode
        contact = -torch.ones(params['N'], 3).to(device=params['device'])
        if mode == 'thumb_middle':
            contact[:, 0] = 1
        elif mode == 'index':
            contact[:, 1] = 1
            contact[:, 2] = 1
        elif mode == 'turn':
            contact[:, :] = 1
        elif mode == 'thumb':
            contact[:, 0] = 1
            contact[:, 1] = 1
        elif mode == 'middle':
            contact[:, 0] = 1
            contact[:, 2] = 1

        recovery_params = copy.deepcopy(params)
        # recovery_params['warmup_iters'] = 100 #if not params.get('model_path_orig', None) else 25
        # recovery_params['online_iters'] = 30 #if not params.get('model_path_orig', None) else 0

        skip_diff_init = False
        planner_returns_action = False
        if 'mppi' in params['recovery_controller'] and recover:
            planner = MPPIPlanner(ctrl, 12, params['T'], warmup=mppi_warmup)
            skip_diff_init = True
            planner_returns_action = True
        else:
            if mode == 'index' and planner is None:
                # finger gate index
                index_regrasp_problem = AllegroValve(
                    start=state[:4 * num_fingers + obj_dof],
                    goal=goal,
                    T=params['T'],
                    chain=params['chain'],
                    device=params['device'],
                    object_asset_pos=env.obj_pose,
                    object_location=params['object_location'],
                    object_type=params['object_type'],
                    world_trans=env.world_trans,
                    regrasp_fingers=['index'],
                    contact_fingers=['middle', 'thumb'],
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
                planner = PositionControlConstrainedSVGDMPC(index_regrasp_problem, recovery_params)

            elif mode == 'middle' and planner is None:
                middle_regrasp_problem = AllegroValve(
                    start=state[:4 * num_fingers + obj_dof],
                    goal=goal,
                    T=params['T'],
                    chain=params['chain'],
                    device=params['device'],
                    object_asset_pos=env.obj_pose,
                    object_location=params['object_location'],
                    object_type=params['object_type'],
                    world_trans=env.world_trans,
                    contact_fingers=['index', 'thumb'],
                    regrasp_fingers=['middle'],
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
                planner = PositionControlConstrainedSVGDMPC(middle_regrasp_problem, recovery_params)
            elif mode == 'thumb' and planner is None:
                thumb_regrasp_problem = AllegroValve(
                    start=state[:4 * num_fingers + obj_dof],
                    goal=goal,
                    T=params['T'],
                    chain=params['chain'],
                    device=params['device'],
                    object_asset_pos=env.obj_pose,
                    object_location=params['object_location'],
                    object_type=params['object_type'],
                    world_trans=env.world_trans,
                    contact_fingers=['index', 'middle'],
                    regrasp_fingers=['thumb'],
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
                planner = PositionControlConstrainedSVGDMPC(thumb_regrasp_problem, recovery_params)
                
            elif mode == 'all' and planner is None:
                all_regrasp_problem = AllegroValve(
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
                planner = PositionControlConstrainedSVGDMPC(all_regrasp_problem, recovery_params)
            elif mode == 'turn' and planner is None:
                tp = AllegroValve(
                    start=state[:4 * num_fingers + obj_dof],
                    goal=goal,
                    T=params['T_orig'] if max_timesteps is None else max_timesteps,
                    chain=params['chain'],
                    device=params['device'],
                    object_asset_pos=env.obj_pose,
                    object_location=params['object_location'],
                    object_type=params['object_type'],
                    world_trans=env.world_trans,
                    contact_fingers=['index', 'middle', 'thumb'],
                    obj_dof=obj_dof,
                    obj_joint_dim=1,
                    optimize_force=params['optimize_force'],
                    default_dof_pos=env.default_dof_pos[:, :16],
                    turn=True,
                    obj_gravity=params.get('obj_gravity', False),
                    min_force_dict=min_force_dict,
                    full_dof_goal=False,
                    proj_path=proj_path,
                    project=False,
                )
                planner = PositionControlConstrainedSVGDMPC(tp, params)
            
            # if recover:
            #     planner.problem.goal[-1] = state[-1]
            #     planner.problem.cost = vmap(partial(planner.problem._cost, start=planner.problem.start, goal=planner.problem.goal, ), randomness='same')
            #     planner.problem.grad_cost = vmap(jacrev(partial(planner.problem._cost, start=planner.problem.start, goal=planner.problem.goal), argnums=(0, 1, 2)), randomness='same')
        # contact = -torch.ones(params['N'], 3).to(device=params['device'])
        # contact[:, 0] = 1
        # contact[:, 1] = 1

        # generate initial samples with diffusion model
        sim_rollouts = None
        initial_samples_0 = None
        new_T = params['T'] if (mode in ['index', 'thumb', 'middle', 'all']) else params['T_orig']
        # if trajectory_sampler is not None and params.get('diff_init', True) and initial_samples is None:
        if not skip_diff_init and trajectory_sampler is not None and params.get('diff_init', True) and initial_samples is None and (not params.get('model_path_orig', None) or not recover) and (not (mode != 'turn' and not params.get('live_recovery'))):

            sampler = trajectory_sampler if recover else trajectory_sampler_orig
            # with torch.no_grad():
            start = state.clone()

            # start = torch.tensor([
            #     -0.00866661,  0.5414786 ,  0.5291143 ,  0.602711  , -0.02442463,
            #     0.39878428,  0.95840853,  0.9327192 ,  1.2067503 ,  0.34080115,
            #     0.33771813,  1.253074  ,  0.09828146,  0.01678719, -1.2030787
            # ], device=params['device'])
            # if state[-1] < -1.0:
            #     start[-1] += 0.75
            a = time.perf_counter()
            # start_for_diff = start#convert_yaw_to_sine_cosine(start)
            if params['sine_cosine']:
                start_for_diff = convert_yaw_to_sine_cosine(start)
            else:
                start_for_diff = start
            ret = sampler.sample(N=params['N'], start=start_for_diff.reshape(1, -1),
                                                                H=sampler.T,
                                                                constraints=contact,
                                                                project=params['project_state'],)
            
            print('Sampling time', time.perf_counter() - a)
            if params['project_state']:
                initial_samples, _, _, initial_samples_0, (all_losses, all_samples, all_likelihoods) = ret
            else:
                initial_samples, _, likelihood = ret
                # initial_samples[..., -(num_fingers * 3)] *= 10
            if params['sine_cosine']:
                initial_samples = convert_sine_cosine_to_yaw(initial_samples)
                if initial_samples_0 is not None:
                    initial_samples_0 = convert_sine_cosine_to_yaw(initial_samples_0)
        if initial_samples is not None:
            initial_samples = initial_samples.to(device=params['device'])
            mode_fpath = f'{fpath}/{fname}'
            pathlib.Path.mkdir(pathlib.Path(mode_fpath), parents=True, exist_ok=True)
            if params['project_state']:
                with open(mode_fpath+ '/projection_results.pkl', 'wb') as f:
                    pickle.dump((initial_samples, initial_samples_0, all_losses, all_samples, all_likelihoods), f)
            if params.get('shortcut_trajectory', False) and mode != 'turn':
                s = time.perf_counter()
                # for traj_idx in range(initial_samples.shape[0]):
                initial_samples = shortcut_trajectory(initial_samples, 4 * num_fingers, obj_dof, epsilon=.04)
                print(f'Shortcut time', time.perf_counter() - s)

                new_T = initial_samples.shape[1] - 1
            # if state[-1] < -1.0:
            #     initial_samples[:, :, -1] -= 0.75
            if False and ((params['visualize_plan'] and not recover) or (params['visualize_recovery_plan'] and recover)):
                for (name, traj_set) in [('initial_samples_project', initial_samples), ('initial_samples_0', initial_samples_0)]:
                    if traj_set is None:
                        continue
                    for k in range(params['N']):
                        traj_for_viz = traj_set[k, :, :planner.problem.dx]
                        # if params['exclude_index']:
                        #     traj_for_viz = torch.cat((state[4:4 + planner.problem.dx].unsqueeze(0), traj_for_viz), dim=0)
                        # else:
                        #     traj_for_viz = torch.cat((state[:planner.problem.dx].unsqueeze(0), traj_for_viz), dim=0)
                        # traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof] = axis_angle_to_euler(traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof])

                        viz_fpath = pathlib.PurePath.joinpath(fpath, f"{fname}/{name}/{k}")
                        img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
                        gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
                        pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
                        pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
                        visualize_trajectory(traj_for_viz, turn_problem.contact_scenes_for_viz, viz_fpath,
                                            turn_problem.fingers, turn_problem.obj_dof + 1)
            sim_rollouts = torch.zeros_like(initial_samples)
            torch.cuda.empty_cache()

            # for i in range(params['N']):
            #     sim_rollout = rollout_trajectory_in_sim(env_sim_rollout, initial_samples[i])
            #     sim_rollouts[i] = sim_rollout
        # if initial_samples is not None and params.get('diff_init', True) and (not recover or params.get('model_path_orig', None)):
        if not skip_diff_init and (not (mode != 'turn' and not params.get('live_recovery'))) and initial_samples is not None and params.get('diff_init', True) and (not recover or params.get('model_path_orig', None)):
            # if params['mode'] == 'hardware' and mode == 'turn':
            #     initial_samples[..., 30:] = 1.5 * torch.randn(params['N'], initial_samples.shape[1], 6, device=initial_samples.device)
            # elif params['mode'] == 'hardware':
            #     initial_samples[..., 27:] = .15 * torch.randn(params['N'], initial_samples.shape[1], initial_samples.shape[-1]-27, device=initial_samples.device)

            initial_samples = _full_to_partial(initial_samples, mode)
            initial_x = initial_samples[:, 1:, :planner.problem.dx]
            initial_u = initial_samples[:, :-1, -planner.problem.du:]
            initial_samples = torch.cat((initial_x, initial_u), dim=-1)


        state = env.get_state()
        state = state['q'].reshape(-1, 4 * num_fingers + obj_dof)[0].to(device=params['device'])
        state = state[:planner.problem.dx]
        # print(params['T'], state.shape, initial_samples)

        # if params.get('compute_recovery_trajectory', False):
        #     planner.reset(state, T=new_T, initial_x=initial_samples, proj_path=proj_path)
        #     planner.warmup_iters = 0
        # else:
        planner.reset(state, T=new_T, goal=goal, initial_x=initial_samples, proj_path=proj_path)
        if initial_samples is None and skip_diff_init:
            initial_samples = torch.zeros(params['N'], 12)
        elif initial_samples is None:
            initial_samples = planner.x.detach().clone()  # this is the initial x from the planner, which is the state at t=0
        elif params['controller'] != 'diffusion_policy' and (trajectory_sampler is None or not params.get('diff_init', True)):
            initial_samples = planner.x.detach().clone()
            sim_rollouts = torch.zeros_like(initial_samples)
        elif params['controller'] == 'diffusion_policy':
            initial_samples = torch.tensor([])
            sim_rollouts = torch.zeros_like(initial_samples)
        resample = params.get('diffusion_resample', False)
        
        # Store how many steps we've executed
        executed_steps = 0
        
        # Get max steps to execute
        total_steps = planner.problem.T if max_timesteps is None else max_timesteps
        if params['recovery_controller'] == 'mppi' and recover:
            total_steps = 100 - episode_num_steps
        
        # Skip to start_timestep if resuming after recovery
        best_traj = None
        for k in range(start_timestep, total_steps):
            state = env.get_state()
            state = state['q'].reshape(-1, 4 * num_fingers + obj_dof)[0].to(device=params['device'])
            print(state)

            if k > 0:
                if recover and params['recovery_controller'] != 'mppi':
                    id_check, final_likelihood = True, None
                elif not params.get('live_recovery', False):
                    id_check, final_likelihood = True, None
                else:
                    id_check, final_likelihood = trajectory_sampler_orig.check_id(state, 8, threshold=params.get('likelihood_threshold', -15), yaw_idx=12, obj_dof=obj_dof)
                if final_likelihood is not None:
                    data['pre_action_likelihoods'][-1].append(final_likelihood)
                # Only return based on dropped if we are using MPPI
                dropped = False
                dropped = dropped and params['recovery_controller'] == 'mppi'
                if dropped:
                    print('dropped')

                cur_yaw = state[-1].item()
                
                cutoff_degrees = 5
                cutoff_radians = np.deg2rad(cutoff_degrees)
                done = False
                if (cur_yaw - goal_yaw) < cutoff_radians:
                    print('Reached goal yaw')
                    done = True
                # If we are generating task data (there is no diff_init), we don't want to return
                done = done and params['diff_init']

                if done:
                    if planner is not None:
                        planner.problem.data = {}
                    if len(actual_trajectory) > 0:
                        actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=params['device'])

                    return actual_trajectory, planned_trajectories, initial_samples, None, None, None, None, False

                if params['recovery_controller'] == 'mppi' and recover and id_check:
                    print('MPPI returned state to ID. Exiting recovery loop')
                    if len(actual_trajectory) > 0:
                        actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=params['device'])

                    return actual_trajectory, planned_trajectories, initial_samples, None, None, None, None, False
                elif not (params['recovery_controller'] == 'mppi' and recover) and not id_check:
                    # State is OOD. Save state and likelihood but DON'T collect data for RL yet.
                    # We will collect it only AFTER recovery completes
                    pre_recovery_state = state[:4 * num_fingers + obj_dof].clone()
                    pre_recovery_likelihood = final_likelihood
                    
                    # State is OOD. Return and move to recovery pipeline
                    if planner is not None:
                        planner.problem.data = {}
                    if len(actual_trajectory) > 0:
                        actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=params['device'])

                    # Zero obj velocity
                    env.zero_obj_velocity()
                    # Return how many steps we've executed for resuming later
                    return actual_trajectory, planned_trajectories, initial_samples, None, None, None, None, True
                elif dropped:
                    print('dropped')
                    if planner is not None:
                        planner.problem.data = {}
                    if len(actual_trajectory) > 0:
                        actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=params['device'])

                    return actual_trajectory, planned_trajectories, initial_samples, None, None, None, None, False

            # We're in distribution, collect data for staying (action=0)
            current_state = state[:4 * num_fingers + obj_dof].clone()
            state = state[:planner.problem.dx]

            # Do diffusion replanning
            if params['controller'] != 'diffusion_policy' and plans is not None and resample:
                # combine past with plans
                executed_trajectory = torch.stack(actual_trajectory, dim=0)
                executed_trajectory = executed_trajectory.reshape(1, -1, planner.problem.dx + planner.problem.du)
                executed_trajectory = executed_trajectory.repeat(params['N'], 1, 1)
                executed_trajectory = _partial_to_full(executed_trajectory, mode)
                plans = _partial_to_full(plans, mode)
                plans = torch.cat((executed_trajectory, plans), dim=1)
                # if state[-1] < -1.0:
                #     plans[:, :, 14] += 0.75
                #     executed_trajectory[:, :, 14] += 0.75

                if trajectory_sampler is not None:
                    with torch.no_grad():
                        initial_samples, _ = trajectory_sampler.resample(
                            start=state.reshape(1, -1).repeat(params['N'], 1),
                            goal=None,
                            constraints=contact,
                            initial_trajectory=plans,
                            past=executed_trajectory,
                            timestep=50)
                    initial_samples = _full_to_partial(initial_samples, mode)
                    initial_x = initial_samples[:, 1:, :planner.problem.dx]
                    initial_u = initial_samples[:, :-1, -planner.problem.du:]
                    initial_samples = torch.cat((initial_x, initial_u), dim=-1)

                    # if state[-1] < -1.0:
                    #     initial_samples[:, :, 14] -= 0.75
                    # update the initial samples
                    planner.x = initial_samples[:, k:]

            s = time.perf_counter()
            best_traj, plans = planner.step(state)
            # if best_traj is None or not recover:
            #     best_traj, plans = planner.step(state)
            # else:
            #     best_traj = best_traj[1:]
            #     plans = plans[:, 1:]
            csvto_time = time.perf_counter() - s
            data['csvto_times'][-1].append(csvto_time)
            print(f'Solve time for step {k+1} (global step {episode_num_steps})', csvto_time)

            planned_trajectories.append(plans)
            optimizer_paths.append(copy.deepcopy(planner.path))
            N, T, _ = plans.shape

            if planner.problem.data is not None and len(planner.problem.data) > 0:
                contact_distance[T] = torch.stack((planner.problem.data['index']['sdf'][:, -T-1:].reshape(N, T + 1),
                                                planner.problem.data['middle']['sdf'][:, -T-1:].reshape(N, T + 1),
                                                planner.problem.data['thumb']['sdf'][:, -T-1:].reshape(N, T + 1)),
                                                dim=1).detach().cpu()

                contact_points[T] = torch.stack((planner.problem.data['index']['closest_pt_world'].reshape(N, -1, 3)[:, -T-1:],
                                                planner.problem.data['middle']['closest_pt_world'].reshape(N, -1, 3)[:, -T-1:],
                                                planner.problem.data['thumb']['closest_pt_world'].reshape(N, -1, 3)[:, -T-1:]),
                                                dim=2).detach().cpu()

            # execute the action
            state = env.get_state()
            state = state['q'].reshape(-1, 4 * num_fingers + obj_dof)[0].to(device=params['device'])
            ori = state[:FULL_DOF][-1]
            print('Current ori:', ori)
            # record the actual trajectory
            if mode == 'turn':
                index_force = torch.norm(best_traj[..., 27:30], dim=-1)
                middle_force = torch.norm(best_traj[..., 30:33], dim=-1)
                thumb_force = torch.norm(best_traj[..., 33:36], dim=-1)
                print('Middle force:', middle_force)
                print('Thumb force:', thumb_force)
                print('Index force:', index_force)
            elif mode == 'index':
                middle_force = torch.norm(best_traj[..., 27:30], dim=-1)
                thumb_force = torch.norm(best_traj[..., 30:33], dim=-1)
                print('Middle force:', middle_force)
                print('Thumb force:', thumb_force)
            elif mode == 'thumb':
                middle_force = torch.norm(best_traj[..., 27:30], dim=-1)
                index_force = torch.norm(best_traj[..., 27:30], dim=-1)
                print('Index force:', index_force)
                print('Middle force:', middle_force)
            elif mode == 'middle':
                index_force = torch.norm(best_traj[..., 27:30], dim=-1)
                thumb_force = torch.norm(best_traj[..., 30:33], dim=-1)
                index_force = torch.norm(best_traj[..., 27:30], dim=-1)
                print('Thumb force:', thumb_force)

            if planner_returns_action or params['controller'] == 'diffusion_policy':
                action = best_traj
            elif params['controller'] != 'diffusion_policy':
                action = best_traj[0, planner.problem.dx:planner.problem.dx + planner.problem.du]
                x = best_traj[0, :planner.problem.dx + planner.problem.du]
                x = x.reshape(1, planner.problem.dx + planner.problem.du)
                action = x[:, planner.problem.dx:planner.problem.dx + planner.problem.du].to(device=env.device)

            xu = torch.cat((state.cpu(), action[0].cpu()))
            actual_trajectory.append(xu)
            if params.get('perturb_action', False):# and mode == 'turn':
                # rand_pct = .75 if not params.get('shortcut_trajectory', False) else .25
                # rand_pct = .75
                rand_pct = 1
                if np.random.rand() < rand_pct:
                    r = np.random.rand()
                    std = .03 if perturb_this_trial else .0
                    # if mode != 'turn':
                    #     std /= 4
                    # if r > .66:
                    #     action[:, :4 * num_fingers_to_plan] += std * torch.randn_like(action[:, :4 * num_fingers_to_plan])
                    # elif r > .4 and r < .7:
                    #     action[:, 4 * 1: 4*3] += std * torch.randn_like(action[:, 4 * 1: 4*3])
                    # else:
                    # if mode == 'turn':
                        # if r < .4:
                        # action[:, :4 * 1] += std * torch.randn_like(action[:, :4 * 1])
                        # if r < .4 and r > .8:
                        #     action[:, 4 * 1: 4*3] += std * torch.randn_like(action[:, 4 * 1: 4*3])
                        # else:
                    noise = std * torch.randn_like(action[:, :4 * num_fingers_to_plan])
                    data['action_perturbations'][-1].append(noise.cpu())
                    action[:, :4 * num_fingers_to_plan] += noise


            action = action[:, :4 * num_fingers_to_plan]

            if params['exclude_index']:
                action = state.unsqueeze(0)[:, 4:4 * num_fingers] + action
                action = torch.cat((state.unsqueeze(0)[:, :4], action), dim=1)  # add the index finger back
            else:
                action = action.to(device=env.device) + state.unsqueeze(0)[:, :4 * num_fingers].to(device=env.device)

            if params['mode'] == 'hardware':
                set_state = env.get_state()['q']
                set_state = set_state.reshape(-1, 4 * num_fingers + obj_dof)[0].to(device=params['device'])
                # print(set_state.shape)
                sim_viz_env.set_pose(set_state)
                sim_viz_env.write_image()

                state = sim_viz_env.get_state()['q'].reshape(-1).to(device=params['device'])
                print(state[:FULL_DOF][-1:])
            elif params['mode'] == 'hardware_copy':
                ros_copy_node.apply_action(partial_to_full_state(action[0], params['fingers']))

            if params['visualize'] and best_traj.shape[0] > 1 and False:
                add_trajectories(plans.to(device=env.device), best_traj.to(device=env.device),
                                 axes, env, sim=sim, gym=gym, viewer=viewer,
                                 config=params, state2ee_pos_func=state2ee_pos,
                                 show_force=(planner == turn_planner and params['optimize_force']))
            if (params['visualize_plan'] and not recover) or (params['visualize_recovery_plan'] and recover):
                traj_for_viz = best_traj[:, :planner.problem.dx]
                if params['exclude_index']:
                    traj_for_viz = torch.cat((state[4:4 + planner.problem.dx].unsqueeze(0), traj_for_viz), dim=0)
                else:
                    traj_for_viz = torch.cat((state[:planner.problem.dx].unsqueeze(0), traj_for_viz), dim=0)
                # traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof] = axis_angle_to_euler(traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof])

                viz_fpath = pathlib.PurePath.joinpath(fpath, f"{fname}/timestep_{k}")
                img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
                gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
                pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
                pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
                visualize_trajectory(traj_for_viz, turn_problem.contact_scenes_for_viz, viz_fpath,
                                     turn_problem.fingers, turn_problem.obj_dof + 1)
            # if np.random.rand() < .5:
            #     # Reduce friction of valve
            #     env.reduce_friction()
            #     print('reduced friction of valve')
            env.step(action.to(device=env.device))
            episode_num_steps += 1
            executed_steps = k + 1

        actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=params['device'])
        # can't stack plans as each is of a different length

        # for memory reasons we clear the data
        if params['controller'] != 'diffusion_policy':
            planner.problem.data = {}
        env.set_external_wrench_perturb(orig_torque_perturb)
        return actual_trajectory, planned_trajectories, initial_samples, sim_rollouts, optimizer_paths, contact_points, contact_distance, recover

    data = {}
    t_range = params['T']
    if 'T_orig' in params and params['T_orig'] > t_range:
        t_range = params['T_orig']
    for t in range(1, 1 + t_range):
        data[t] = {'plans': [], 'starts': [], 'inits': [], 'init_sim_rollouts': [], 'optimizer_paths': [], 'contact_points': [], 'contact_distance': [], 'contact_state': []}
    data['pre_action_likelihoods'] = []
    data['final_likelihoods'] = []
    data['project_times'] = []
    data['csvto_times'] = []
    data['action_perturbations'] = []
    data['all_samples_'] = []
    data['all_likelihoods_'] = []
    data['contact_plan_times'] = []
    data['executed_contacts'] = []
        # sample initial trajectory with diffusion model to get contact sequence
    state = env.get_state()
    state = state['q'].reshape(-1, 4 * num_fingers + obj_dof).to(device=params['device'])[0]

    def dec2bin(x, bits):
        # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    def bin2dec(b, bits):
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
        return torch.sum(mask * b, -1)

    def _add_to_dataset(traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, contact_state):
        for i, plan in enumerate(plans):
            t = plan.shape[1]
            data[t]['plans'].append(plan)
            data[t]['inits'].append(inits.cpu().numpy())
            data[t]['init_sim_rollouts'].append(init_sim_rollouts)
            try:
                data[t]['optimizer_paths'].append([i.cpu().numpy() for i in optimizer_paths])
            except:
                pass            
            data[t]['starts'].append(traj[i].reshape(1, -1).repeat(plan.shape[0], 1))
            try:
                data[t]['contact_points'].append(contact_points[t])
                data[t]['contact_distance'].append(contact_distance[t])
                data[t]['contact_state'].append(contact_state)
            except:
                pass

    def _partial_to_full(traj, mode):
        if mode == 'index':
            traj = torch.cat((traj[..., :-6], torch.zeros(*traj.shape[:-1], 3).to(device=params['device']),
                              traj[..., -6:]), dim=-1)
        if mode == 'thumb_middle':
            traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 6).to(device=params['device'])), dim=-1)
        if mode == 'pregrasp' or mode == 'all':
            traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 9).to(device=params['device'])), dim=-1)
        if mode == 'thumb':
            traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 3).to(device=params['device'])), dim=-1)
        if mode == 'middle':
            traj = torch.cat((traj[..., :-3], torch.zeros(*traj.shape[:-1], 3).to(device=params['device']),
                              traj[..., -3:]), dim=-1)
        return traj

    def _full_to_partial(traj, mode):
        if mode == 'index':
            traj = torch.cat((traj[..., :-9], traj[..., -6:]), dim=-1)
        if mode == 'thumb_middle':
            traj = traj[..., :-6]
        if mode == 'pregrasp' or mode == 'all':
            traj = traj[..., :-9]
        if mode == 'thumb':
            traj = traj[..., :-3]
        if mode == 'middle':
            traj = torch.cat((traj[..., :-6], traj[..., -3:]), dim=-1)
        return traj

    def visualize_trajectory_wrapper(traj, contact_scenes, fname, plan_or_init, index, fingers, obj_dof, k):
        viz_fpath = pathlib.PurePath.joinpath(fpath, f"{fname}/{plan_or_init}/{index}/timestep_{k}")
        img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
        gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
        pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
        pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
        visualize_trajectory(traj, contact_scenes, viz_fpath, fingers, obj_dof + 1)

    def plan_recovery_contacts(state, stage):
        start_plan_time = time.perf_counter()
        if params.get('model_path_orig', None):
            # Use recovery model to get contact mode
            state = state[:FULL_DOF]
            start = convert_yaw_to_sine_cosine(state)
            start = start.unsqueeze(0)
            contact_mode_str_max = 'unknown'  # Default in case we can't find a match
            while contact_mode_str_max == 'unknown':
                initial_samples, raw_contact_mode, likelihood = trajectory_sampler.sample(
                    N=params['N'], start=start, H=params['T']+1, constraints=None, project=False)
                scaled_raw_contact_mode = (raw_contact_mode + 1) / 2  # Scale to [0, 1] for binning
                # print('Rounded raw contact mode:', rounded_raw_contact_mode)
                # rounded_raw_contact_mode = torch.round(raw_contact_mode)
                # contact_vec = (rounded_raw_contact_mode + 1) / 2
                contact_vec = torch.round(scaled_raw_contact_mode)
                # print('Contact vec:', contact_vec)
                all_c_mode_str = []
                for i in range(contact_vec.shape[0]):
                    try:
                        c_mode_str = contact_state_dict_flip[tuple(contact_vec[i].cpu().numpy())]
                        all_c_mode_str.append(c_mode_str)
                    except:
                        print('Warning: Contact mode not found for:', contact_vec[i].cpu().numpy())
                        # Fallback to a default if not found
                        # This should not happen if the model is trained correctly
                        c_mode_str = 'unknown'
                        all_c_mode_str.append(c_mode_str)
                all_c_mode_str = np.array(all_c_mode_str)
                likelihood_sort, indices = torch.sort(likelihood, descending=True)
                indices = indices.cpu().numpy()

                # Convert contact mode to string
                contact_mode_str_sort = all_c_mode_str[indices]
                contact_mode_str_max = contact_mode_str_sort[0]

                initial_samples = initial_samples[indices]
                initial_samples = convert_sine_cosine_to_yaw(initial_samples)
                plan_time = time.perf_counter() - start_plan_time
                print('Likelihoods:', likelihood_sort)
                print('Contact modes', contact_mode_str_sort)
                # goal_config = initial_samples[0, -1, :15] # Use highest likelihood trajectory

                # Compute the average likelihood, grouped by mode
                likelihood_grouped = {}
                inds_grouped = {}
                for i, mode in enumerate(contact_mode_str_sort):
                    if mode not in likelihood_grouped:
                        likelihood_grouped[mode] = []
                        inds_grouped[mode] = []
                    likelihood_grouped[mode].append(likelihood_sort[i].item())
                    inds_grouped[mode].append(indices[i])
                likelihood_mean = {}
                likelihood_sum = {}
                for mode in likelihood_grouped:
                    likelihood_mean[mode] = np.mean(np.exp(likelihood_grouped[mode]))
                    likelihood_sum[mode] = np.sum(np.exp(likelihood_grouped[mode]))
                print('Likelihood grouped mean:', likelihood_mean)
                print('Likelihood grouped sum:', likelihood_sum)

                # Pick mode with highest likelihood sum
                if 'unknown' in likelihood_sum:
                    # If 'unknown' is present, fallback to the highest likelihood sum excluding 'unknown'
                    likelihood_sum.pop('unknown', None)
                    if not likelihood_sum:
                        contact_mode_str_max = 'unknown'
                        print('All modes are unknown')
                        continue
                contact_mode_str_max = max(likelihood_sum, key=likelihood_sum.get)
                print('Contact mode with highest likelihood sum:', contact_mode_str_max)

                # # Pick mode of the highest likelihood trajectory
                # contact_mode_str_max = contact_mode_str_sort[0]
                # print('Contact mode with highest likelihood trajectory:', contact_mode_str_max)

                # goal config is last state of highest likelihood trajectory for contact_mode_str_max. Has to be for the right mode
                best_traj_idx = inds_grouped[contact_mode_str_max][0]
                goal_config = initial_samples[best_traj_idx, -1, :FULL_DOF]
            
                if params['task_diffuse_goal']:
                    goal = index_regrasp_planner.problem.goal.clone()
                    # goal[-1] = state[-1]
                    initial_samples = []
                    pre_goal_viz = time.perf_counter()
                    plan_time += time.perf_counter() - pre_goal_viz
                    if params['visualize_contact_plan']:
                        # Visualize the goal
                        viz_fpath = pathlib.PurePath.joinpath(fpath, f"{fpath}/recovery_stage_{all_stage}/goal")
                        pathlib.Path.mkdir(viz_fpath, parents=True, exist_ok=True)
                        img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
                        gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
                        pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
                        pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
                        

                        traj_for_viz = torch.cat((state.unsqueeze(0), goal.unsqueeze(0)), dim=0)
                        visualize_trajectory(traj_for_viz
                        , turn_problem.contact_scenes_for_viz, viz_fpath,
                                            turn_problem.fingers, turn_problem.obj_dof + 1)

            return [contact_mode_str_max], goal_config, initial_samples, likelihood, plan_time
        else:
            return plan_recovery_contacts_offline(state, stage)

    def plan_recovery_contacts_offline(state, stage):
        plan_time = 0
        start_plan_time = time.perf_counter()
        distances = []
        # modes = ['all', 'index', 'middle', 'thumb']
        modes = ['index', 'middle', 'thumb']
        # modes = ['thumb_middle', 'index', 'turn']
        # modes = ['thumb_middle', 'index']
        # for planner in [index_regrasp_planner, thumb_and_middle_regrasp_planner]:
        goal = index_regrasp_planner.problem.goal.clone()
        # goal[-1] = state[-1]
        initial_samples = []
        pre_goal_viz = time.perf_counter()
        plan_time += time.perf_counter() - pre_goal_viz
        if params['visualize_contact_plan']:
            # Visualize the goal
            viz_fpath = pathlib.PurePath.joinpath(fpath, f"{fpath}/recovery_stage_{all_stage}/goal")
            pathlib.Path.mkdir(viz_fpath, parents=True, exist_ok=True)
            img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
            gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
            pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
            pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
            

            traj_for_viz = torch.cat((state.unsqueeze(0), goal.unsqueeze(0)), dim=0)
            visualize_trajectory(traj_for_viz
            , turn_problem.contact_scenes_for_viz, viz_fpath,
                                turn_problem.fingers, turn_problem.obj_dof + 1)
        post_goal_viz = time.perf_counter()
        dist_min = 3e-3
        mode_skip = []

        
        pre_mode_loop = time.perf_counter()
        plan_time += pre_mode_loop - post_goal_viz
        for mode in modes:
            if mode in mode_skip:
                distances.append(float('inf'))
                initial_samples.append(None)
                continue
            begin_mode_loop = time.perf_counter()
            planner = mode_planner_dict[mode]
            planner.reset(state, T=params['T'], goal=goal)
            
            # planner.problem.goal[-1] = state[-1]
            # planner.problem.cost = vmap(partial(planner.problem._cost, start=planner.problem.start, goal=planner.problem.goal, ), randomness='same')
            # planner.problem.grad_cost = vmap(jacrev(partial(planner.problem._cost, start=planner.problem.start, goal=planner.problem.goal), argnums=(0, 1, 2)), randomness='same')

            # old_warmup_iters = planner.warmup_iters
            planner.warmup_iters = 100

            xu, plans = planner.step(state)
            planner.problem.data = {}

            planner.warmup_iters = 0
            initial_samples.append(plans)
            # x_last = xu[-1, :planner.problem.num_fingers * 4 + planner.problem.obj_dof-1]
            # goal_cost = (x_last - planner.problem.goal[:-1]).pow(2).sum(dim=-1)#.sum(dim=-1)
            x = xu[:, :planner.problem.num_fingers * 4 + planner.problem.obj_dof]

            start = x[-1]
            likelihood, samples = trajectory_sampler_orig.check_id(start, 8, likelihood_only=True, return_samples=True, threshold=params.get('likelihood_threshold', -15), yaw_idx=12, obj_dof=obj_dof)
            distances.append(-likelihood)
            end_mode_loop = time.perf_counter()
            plan_time += end_mode_loop - begin_mode_loop
            viz_fpath = pathlib.PurePath.joinpath(fpath, f"{fpath}/recovery_stage_{all_stage}/{mode}")
            pathlib.Path.mkdir(viz_fpath, parents=True, exist_ok=True)
            # Dump plans, samples, likelihood to file
            with open(f"{viz_fpath}/recovery_info.pkl", "wb") as f:
                pkl.dump((plans, samples, likelihood), f)
            if params['visualize_contact_plan']:
                traj_for_viz = x[:, :planner.problem.dx]
                if params['exclude_index']:
                    traj_for_viz = torch.cat((state[4:4 + planner.problem.dx].unsqueeze(0), traj_for_viz), dim=0)
                else:
                    traj_for_viz = torch.cat((state[:planner.problem.dx].unsqueeze(0), traj_for_viz), dim=0)


                img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
                gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
                pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
                pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
                visualize_trajectory(traj_for_viz, turn_problem.contact_scenes_for_viz, viz_fpath,
                                     turn_problem.fingers, turn_problem.obj_dof + 1)

        dists_dict = dict(zip(modes, distances))
        pprint(dists_dict)

        if all(d == float('inf') for d in distances):
            return ['turn'], None, plan_time
        else:
            return [modes[np.argmin(distances)]], initial_samples[np.argmin(distances)], plan_time

    state = env.get_state()
    state = state['q'].reshape(-1, 4 * num_fingers + obj_dof).to(device=params['device'])[0]

    sample_contact = params.get('sample_contact', False)
    num_stages = 20

    contact = None
    state = env.get_state()
    state = state['q'].reshape(-1, 4 * num_fingers + obj_dof).to(device=params['device'])[0]

    executed_contacts = []
    recover = False
    pre_recover = False
    stage = 0
    all_stage = 0
    done = False
    max_episode_num_steps = 100
    # for stage in range(num_stages):
    # <= so because pregrasp will iterate the all_stage counter

    if not params.get('live_recovery', False):
        contact_sequence = ['turn']*100

        # while len(contact_sequence) < 50:
        #     contact_options = ['index', 'middle', 'thumb']
        #     perm = np.random.permutation(3)
        #     # perm = [1, 0]
        #     for idx in perm:
        #         contact = contact_options[idx]
        #         contact_sequence.append(contact)
        #     contact_sequence.append('turn')
    while episode_num_steps < max_episode_num_steps:
        sample_contact = params['sample_contact'] and not recover
        initial_samples = None
        state = env.get_state()
        state = state['q'].reshape(-1, 4 * num_fingers + obj_dof).to(device=params['device'])[0]
        planned = False
        if params.get('live_recovery', False) and recover:
            if params['recovery_controller'] == 'mppi':
                contact_sequence = ['mppi']
                goal_config = None
                initial_samples = None
                likelihood = None
            elif params.get('model_path_orig', None) and params.get('generate_context', False):
                # Use recovery model to get contact mode'):
                contact_sequence, goal_config, initial_samples, likelihood, plan_time = plan_recovery_contacts(state, stage)
                # goal_config[-1] = state[-1]
                if params.get('task_diffuse_goal', False):
                    goal_config = None
                if not params.get('recovery_diff_traj', False):
                    initial_samples = None
                data['all_likelihoods_'].append(likelihood)
                planned = True
            elif params.get('model_path_orig', None):
                # Use recovery model to get contact mode
                contact_sequence, goal_config, initial_samples, likelihood, plan_time = plan_recovery_contacts_w_model(state)
                # goal_config[-1] = state[-1]
                # goal_config = None
                # initial_samples = None
                planned = True
            else:
                contact_sequence, initial_samples, plan_time = plan_recovery_contacts(state, stage)
                planned = True
        elif params.get('live_recovery', False) and not recover:
            contact_sequence = ['turn']

        if planned:
            print('Plan time:', plan_time)
            data['contact_plan_times'].append(plan_time)
        state = env.get_state()
        state = state['q'].reshape(-1, 4 * num_fingers + obj_dof).to(device=params['device'])[0]
        ori = state[:FULL_DOF][-1:]
        yaw = ori[-1]
        print('Current yaw:', ori)

        if stage == 0:

            orig_torque_perturb = env.external_wrench_perturb if params['mode'] != 'hardware' else False
            if params['mode'] != 'hardware':
                env.set_external_wrench_perturb(False)
            contact = 'pregrasp'
            start = env.get_state()['q'].reshape(-1, 4 * num_fingers + obj_dof).to(device=params['device'])[0]
            best_traj, _ = pregrasp_planner.step(start[:pregrasp_planner.problem.dx])
            if (params['visualize_plan'] and not recover) or (params['visualize_recovery_plan'] and recover):
                traj_for_viz = best_traj[:, :13]
                if params['exclude_index']:
                    traj_for_viz = torch.cat((state[4:4 + 13].unsqueeze(0), traj_for_viz), dim=0)
                else:
                    traj_for_viz = torch.cat((state[:13].unsqueeze(0), traj_for_viz), dim=0)
                # traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof] = axis_angle_to_euler(traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof])

                viz_fpath = pathlib.PurePath.joinpath(fpath, f"pregrasp/timestep_{0}")
                img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
                gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
                pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
                pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
                visualize_trajectory(traj_for_viz, turn_problem.contact_scenes_for_viz, viz_fpath,
                                     turn_problem.fingers, turn_problem.obj_dof + 1)

            for x in best_traj[:, :4 * num_fingers]:
                action = x.reshape(-1, 4 * num_fingers).to(device=env.device) # move the rest fingers
                env.step(action)
                # After stepping, reset the valve to where it was initially
                # s = env.get_state()['q'].reshape(-1, 4 * num_fingers + obj_dof).to(device=params['device'])[0]
                # s[-obj_dof:] = start[-obj_dof:]
                # env.set_pose(s.to(device=env.device))

                if params['mode'] == 'hardware':
                    set_state = env.get_state()['q'].to(device=env.device)
                    # print(set_state.shape)
                    sim_viz_env.set_pose(set_state)  

                    state = sim_viz_env.get_state()['q'].reshape(-1).to(device=params['device'])
                    print(state[:FULL_DOF][-1:])
            # for _ in range(50):
            #     env._step_sim()
            post_pregrasp_state = env.get_state()['q'].reshape(-1, 4 * num_fingers + obj_dof).to(device=params['device'])[0]
            post_pregrasp_state = post_pregrasp_state[:FULL_DOF]
            if params['mode'] == 'hardware':
                input("Pregrasp complete. Ready to execute. Press <ENTER> to continue.")
            stage += 1
            all_stage += 1
            if params['mode'] != 'hardware':
                env.set_external_wrench_perturb(orig_torque_perturb)
            continue
        else:
            if len(contact_sequence) == 0:
                break
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
        state = state['q'].reshape(-1, FULL_DOF)[0].to(device=params['device'])

        pre_recover = recover
        pre_mode_yaw = state[-1]
        if contact == 'index':
            _goal = None
            if params.get('model_path_orig', None):
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
            state = state['q'].reshape(-1, FULL_DOF)[0].to(device=params['device'])

            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, recover = result

            plans = [torch.cat((plan[..., :-6],
                                torch.zeros(*plan.shape[:-1], 3).to(device=params['device']),
                                plan[..., -6:]),
                            dim=-1) for plan in plans]
            traj = torch.cat((traj[..., :-6], torch.zeros(*traj.shape[:-1], 3).to(device=params['device']),
                            traj[..., -6:]), dim=-1)

        if contact == 'thumb':
            _goal = None
            if params.get('model_path_orig', None):
                _goal = goal_config
            # If we're recovering and have a saved goal/timesteps, use them
            start_timestep = 0
            max_timesteps = None
            # Execute trajectory
            result = execute_traj(
                thumb_regrasp_planner, mode='thumb', goal=_goal, 
                fname=f'thumb_regrasp_{all_stage}', initial_samples=initial_samples, 
                recover=recover, start_timestep=start_timestep, max_timesteps=max_timesteps)
            state = env.get_state()
            state = state['q'].reshape(-1, FULL_DOF)[0].to(device=params['device'])

            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, recover = result

            plans = [torch.cat((plan,
                                torch.zeros(*plan.shape[:-1], 3).to(device=params['device'])),
                            dim=-1) for plan in plans]
            traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 3).to(device=params['device'])), dim=-1)

        if contact == 'middle':
            _goal = None
            if params.get('model_path_orig', None):
                _goal = goal_config
            # If we're recovering and have a saved goal/timesteps, use them
            start_timestep = 0
            max_timesteps = None
            # Execute trajectory
            result = execute_traj(
                middle_regrasp_planner, mode='middle', goal=_goal, 
                fname=f'middle_regrasp_{all_stage}', initial_samples=initial_samples, 
                recover=recover, start_timestep=start_timestep, max_timesteps=max_timesteps)
            state = env.get_state()
            state = state['q'].reshape(-1, FULL_DOF)[0].to(device=params['device'])

            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, recover = result

            plans = [torch.cat((plan[..., :-3],
                                torch.zeros(*plan.shape[:-1], 3).to(device=params['device']),
                                plan[..., -3:]),
                            dim=-1) for plan in plans]
            traj = torch.cat((traj[..., :-3], torch.zeros(*traj.shape[:-1], 3).to(device=params['device']),
                            traj[..., -3:]), dim=-1)

        elif contact == 'all':
            _goal = None
            if params.get('model_path_orig', None):
                _goal = goal_config
            result = execute_traj(
                all_regrasp_planner, mode='all', goal=_goal, 
                fname=f'all_regrasp_{all_stage}', initial_samples=initial_samples, 
                recover=recover, start_timestep=start_timestep, max_timesteps=max_timesteps)
            state = env.get_state()
            state = state['q'].reshape(-1, FULL_DOF)[0].to(device=params['device'])

            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, recover = result

            plans = [torch.cat((plan,
                                torch.zeros(*plan.shape[:-1], 9).to(device=params['device']),
                                ),
                            dim=-1) for plan in plans]
            traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 9).to(device=params['device']),
                            ), dim=-1)

        elif contact == 'turn':
            mppi_ctrl = None
            _goal = torch.tensor([state[-1] - np.pi/4]).to(device=params['device'])
            # _goal = torch.tensor([0.]).to(device=params['device'])
                
            # If we're recovering and have a saved goal/timesteps, use them
            start_timestep = 0
            max_timesteps = None
                
            result = execute_traj(
                None, mode='turn', goal=_goal, fname=f'turn_{all_stage}', initial_samples=initial_samples, 
                recover=recover, start_timestep=start_timestep, max_timesteps=max_timesteps)
                
            state = env.get_state()
            state = state['q'].reshape(-1, FULL_DOF)[0].to(device=params['device'])

            # Backward compatibility with old return format
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, recover = result
        
        elif contact == 'mppi':
            if mppi_ctrl is None:
                mppi_ctrl = MPPI(dynamics=dynamics, running_cost=running_cost, terminal_state_cost=terminal_cost, nx=nx, noise_sigma=noise_sigma, 
                            num_samples=50, horizon=params['T'], lambda_=params['lambda_'], u_min=u_min, u_max=u_max,
                            device=params['device'])
                mppi_needs_warmup = True
            result = execute_traj(
                None, mode='mppi', goal=None, fname=f'mppi_{all_stage}', initial_samples=initial_samples,
                recover=recover, ctrl=mppi_ctrl, mppi_warmup=mppi_needs_warmup)
            mppi_needs_warmup = False
           
        # done = False
        add = not recover or params['live_recovery']
        state = env.get_state()
        state = state['q'].reshape(-1, FULL_DOF)[0].to(device=params['device'])
        
        start = state[:4 * num_fingers + obj_dof]
        if params.get('live_recovery', True):
            if (not pre_recover and recover):
                # If we stopped turning because of OOD, then last measured likelihood is post-action
                likelihood = data['pre_action_likelihoods'][-1][-1]
                # if all_stage == 1:
                if len(data['pre_action_likelihoods'][-1]) > 1:
                    data['final_likelihoods'][-1].append(likelihood)
                else:
                    data['final_likelihoods'][-1].append(None)
                # skip_first_likelihood_eval = False
            else:
                # If we just recovered, assume we are done. If we are not, next turn will catch it.
                recover = False

                # start_sine_cosine = convert_yaw_to_sine_cosine(start)
                # else branch means we either fully executed the turn or we just completed a recovery trajectory. In either case, there is no measure of likelihood after last executed action.
                # id, likelihood = trajectory_sampler_orig.check_id(start_sine_cosine, 8, threshold=params.get('likelihood_threshold', -15))
                # skip_first_likelihood_eval = True
            # data['final_likelihoods'][-1].append(likelihood)
            # recover = not id
            if all_stage > 1 and len(data['final_likelihoods'][-2]) == 0:
                data['final_likelihoods'][-2].append(data['pre_action_likelihoods'][-1][0])

        stage += 1
        all_stage += 1

        drop_cutoff = .25
        dropped = False

        # dropped = likelihood < -1250
        if dropped:
            print('Probably dropped the object')
            done = True
            # add = False
        data['dropped'] = dropped
        data['dropped_recovery'] = dropped and pre_recover
        
        cur_yaw = state[-1].item()
        
        cutoff_degrees = 5
        cutoff_radians = np.deg2rad(cutoff_degrees)
        if params['diff_init'] and (cur_yaw - goal_yaw) < cutoff_radians:
            print('Reached goal yaw')
            done = True
       
        # If first mode didn't turn, end episode to save compute (TASK DATA GEN ONLY)

        if not params['diff_init'] and (pre_mode_yaw - cur_yaw) < .15:
            print('Failed to turn past .15 rad')
            done = True

        
        if recover and not done and params.get('task_diffuse_goal', False):
        # if recover and not done:
            state = env.get_state()
            state = state['q'].reshape(-1, FULL_DOF)[0].to(device=params['device'])
            
            start = state[:4 * num_fingers + obj_dof]
            start_sine_cosine = convert_yaw_to_sine_cosine(start)
            
            # Project the state back into distribution if we are computing recovery trajectories
            pre_project_time = time.perf_counter()
            projected_samples, _, _, _, (all_losses, all_samples, all_likelihoods) = trajectory_sampler_orig.sample(
                4, H=trajectory_sampler_orig.T, start=start_sine_cosine.reshape(1, -1), project=True,
                constraints=torch.ones(4, 3).to(device=params['device'])
            )
            data['project_times'].append(time.perf_counter() - pre_project_time)
            print('Final likelihood:', all_likelihoods[-1])
            
            threshold = params.get('likelihood_threshold', -15)
            if all_likelihoods[-1].mean().item() < threshold:
                print('1 mode projection failed, trying anyway')
            else:
                print('1 mode projection succeeded')
                
            goal = convert_sine_cosine_to_yaw(projected_samples[0][0])[:FULL_DOF]
            # goal[-1] = start[-1]

            params_for_recovery = deepcopy(params)
            if index_regrasp_planner is None:
                index_regrasp_problem = AllegroValve(
                    start=start[:4 * num_fingers + obj_dof],
                    goal=goal,
                    T=params['T'],
                    chain=params['chain'],
                    device=params['device'],
                    object_asset_pos=env.obj_pose,
                    object_location=params['object_location'],
                    object_type=params['object_type'],
                    world_trans=env.world_trans,
                    regrasp_fingers=['index'],
                    contact_fingers=['middle', 'thumb'],
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
                index_regrasp_planner = PositionControlConstrainedSVGDMPC(index_regrasp_problem, params_for_recovery)

            if middle_regrasp_planner is None:
                middle_regrasp_problem = AllegroValve(
                    start=state[:4 * num_fingers + obj_dof],
                    goal=goal,
                    T=params['T'],
                    chain=params['chain'],
                    device=params['device'],
                    object_asset_pos=env.obj_pose,
                    object_location=params['object_location'],
                    object_type=params['object_type'],
                    world_trans=env.world_trans,
                    contact_fingers=['index', 'thumb'],
                    regrasp_fingers=['middle'],
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
                middle_regrasp_planner = PositionControlConstrainedSVGDMPC(middle_regrasp_problem, params_for_recovery)
            
            if thumb_regrasp_planner is None:
                thumb_regrasp_problem = AllegroValve(
                    start=state[:4 * num_fingers + obj_dof],
                    goal=goal,
                    T=params['T'],
                    chain=params['chain'],
                    device=params['device'],
                    object_asset_pos=env.obj_pose,
                    object_location=params['object_location'],
                    object_type=params['object_type'],
                    world_trans=env.world_trans,
                    contact_fingers=['index', 'middle'],
                    regrasp_fingers=['thumb'],
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
                thumb_regrasp_planner = PositionControlConstrainedSVGDMPC(thumb_regrasp_problem, params_for_recovery)
                
            if all_regrasp_planner is None:
                all_regrasp_problem = AllegroValve(
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
                'middle': middle_regrasp_planner,
                'thumb': thumb_regrasp_planner,
            }
            index_regrasp_planner.reset(start, goal=goal)
            middle_regrasp_planner.reset(start, goal=goal)
            thumb_regrasp_planner.reset(start, goal=goal)
            all_regrasp_planner.reset(start, goal=goal)

            print('New goal:', goal)

            if torch.allclose(start, goal):
                print('Goal is the same as current state')
                recover = False
        if add:
            _add_to_dataset(traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance,
                            contact_state=contact_state_dict[contact])
        if contact != 'pregrasp' and add:
            actual_trajectory.append(traj)
        # change to numpy and save data

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
        state = state['q'].reshape(-1, 4 * num_fingers + obj_dof).to(device=params['device'])[0]
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
    if params.get('live_recovery', False) and len(data['final_likelihoods'][-1]) == 0 and params['recovery_controller'] != 'mppi':
        id, likelihood = trajectory_sampler_orig.check_id(state, 8, threshold=params.get('likelihood_threshold', -15), yaw_idx=12, obj_dof=obj_dof)
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
    return (state[-1] - initial_angle).item()


if __name__ == "__main__":
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/valve/{sys.argv[1]}.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/valve/allegro_valve_csvto_only.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/valve/allegro_valve_csvto_recovery_data_gen.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/valve/allegro_valve_csvto_safe_rl_data_gen.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/valve/allegro_valve_mppi_likelihood_recovery.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_csvto_recovery_model_alt_2_noised_s0_9000_bto_recovery_diff_traj.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_csvto_safe_rl_recovery.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_csvto_OOD_ID_orig_likelihood_rl_data_gen_wrench_perturb.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_csvto_OOD_ID_orig_likelihood_rl_wrench_perturb_new_project.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_csvto_OOD_ID_perturbed_data_gen.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_csvto_OOD_ID_orig_likelihood.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_csvto_OOD_ID_live_recovery_shortcut_0.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_csvto_diff_demo.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_csvto_OOD_ID_live_recovery_shortcut_hardware.yaml').read_text())

    from tqdm import tqdm

    sim_env = None
    ros_copy_node = None

    if 'recovery_controller' not in config:
        config['recovery_controller'] = 'csvgd'

    if config['mode'] == 'hardware':
        # roslaunch allegro_hand allegro_hand_modified.launch
        from hardware.hardware_env import HardwareEnv
        default_dof_pos = torch.cat((torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float(),
                                    torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float(),
                                    torch.tensor([[0., 0.5, 0.65, 0.65]]).float(),
                                    torch.tensor([[1.2, 0.3, 3, 1.2]]).float()),
                                    dim=1)
        env = HardwareEnv(default_dof_pos[:, :16], 
                          finger_list=config['fingers'], 
                          kp=config['kp'], 
                          obj='screwdriver',
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
        sim_env = RosAllegroValveTurningEnv(1, control_mode='joint_impedance',
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
                                 gravity=False
                                 )
        
        sim, gym, viewer = sim_env.get_sim()
        assert (np.array(sim_env.robot_p) == robot_p).all()
        assert (sim_env.default_dof_pos[:, :16] == default_dof_pos.to(config['sim_device'])).all()

        env.world_trans = sim_env.world_trans
        env.joint_stiffness = sim_env.joint_stiffness
        env.device = sim_env.device
        env.obj_pose = sim_env.table_pose
    else:
        if not config['visualize']:
            img_save_dir = None

        num_envs = 50 if 'mppi' in config['recovery_controller'] else 1
        env = AllegroValveTurningEnv(num_envs, control_mode='joint_impedance',
                                           use_cartesian_controller=False,
                                           viewer=config['visualize'],
                                           steps_per_action=60,
                                           friction_coefficient=config['friction_coefficient'] * 1.0,
                                           device=config['sim_device'],
                                           video_save_path=img_save_dir,
                                           joint_stiffness=config['kp'],
                                           fingers=config['fingers'],
                                           gravity=True, 
                                           randomize_obj_start=config['randomize_obj_start'],
                                           randomize_rob_start=config['randomize_rob_start']
                                           )
        if config['mode'] == 'hardware':
            nx = sim_viz_env.dof_states.shape[1] * 2
        else:
            nx = env.dof_states.shape[1] * 2


        sim, gym, viewer = env.get_sim()

    state = env.get_state()


    results = {}

    # set up the kinematic chain
    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    ee_names = {
        'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
        'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
        'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
        'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
    }
    config['ee_names'] = ee_names
    config['obj_dof'] = obj_dof

    valve_asset = f'{get_assets_dir()}/valve/valve_cross.urdf'

    chain = pk.build_chain_from_urdf(open(asset).read())
    valve_chain = pk.build_chain_from_urdf(open(valve_asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]  # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices,
                           world_trans=env.world_trans)

    forward_kinematics = partial(chain.forward_kinematics,
                                 frame_indices=frame_indices)  # full_to= _partial_state = partial(full_to_partial_state, fingers=config['fingers'])
    partial_to_full_state = partial(partial_to_full_state, fingers=config['fingers'])
    

    for key in ['compute_recovery_trajectory', 'test_recovery_trajectory', 'viz_states']:
        if key not in config:
            config[key] = False


    # Get datetime
    if config['mode'] == 'hardware':
        import datetime
        now = datetime.datetime.now().strftime("%m.%d.%y:%I:%M:%S")
        now_ = '.' + now
        now = now_
    else:
        now = ''

    trajectory_sampler = None
    model_path = config.get('model_path', None)
    model_path_orig = config.get('model_path_orig', None)
    model_path_rl_adjusted = config.get('model_path_rl_adjusted', None)

    params = config.copy()
    params.pop('controllers')
    params.update(config['controllers']['csvgd'])
    FULL_DOF = 4 * len(config['fingers']) + obj_dof

    if model_path is not None:
        problem_for_sampler = None
        if 'type' not in config:
            config['type'] = 'diffusion'

        loading_recovery_model = params.get('model_path_orig', None) is not None
        vae = None
        model_t = config['type'] == 'latent_diffusion'
        if model_t:
            vae_path = config.get('vae_path', None)
            vae = LatentDiffusionModel(config, None).to(params['device'])
            vae.load_state_dict(torch.load(f'{CCAI_PATH}/{vae_path}'))
            for param in vae.parameters():
                param.requires_grad = False
        def load_sampler(path, dim_mults=(1,2), T=config['T'], recovery=False):
            trajectory_sampler = TrajectorySampler(T=T + 1, dx=(FULL_DOF + (1 if config['sine_cosine'] else 0)) if not model_t else config['nzt'], du=21 if not model_t else 0, type=config['type'],
                                                timesteps=256, hidden_dim=128 if not model_t else 64,
                                                context_dim=3, 
                                                constrain=config['projected'],
                                                problem=problem_for_sampler,
                                                guided=config['use_guidance'],
                                                state_control_only=config.get('state_control_only', False),
                                                vae=vae,
                                                initial_threshold=config.get('likelihood_threshold', -15),
                                                new_projection=config.get('new_projection', False),
                                                generate_context=config.get('generate_context', False) if recovery else False,
                                                trajectory_condition=config.get('trajectory_condition', False) if recovery else False,
                                                true_s0=config.get('true_s0', False),
                                                )
            d = torch.load(f'{CCAI_PATH}/{path}', map_location=torch.device(params['device']))

            # if 'recovery' in path:
            #     trajectory_sampler.model.diffusion_model.add_classifier(dim_mults)
            # else:
            
            trajectory_sampler.model.diffusion_model.classifier = None
            d = {k:v for k, v in d.items() if 'classifier' not in k}
            trajectory_sampler.load_state_dict(d, strict=recovery)
            trajectory_sampler.to(device=params['device'])
            trajectory_sampler.send_norm_constants_to_submodels()
            # if config['project_state'] or config['compute_recovery_trajectory'] or config['test_recovery_trajectory']:
            trajectory_sampler.model.diffusion_model.cutoff = config['project_threshold']
            trajectory_sampler.model.diffusion_model.subsampled_t = '5_10_15' in config['experiment_name']
            trajectory_sampler.model.diffusion_model.classifier = None
            
            trajectory_sampler.model.diffusion_model.cutoff_timesteps = 96

            return trajectory_sampler
        
        if params['recovery_controller'] != 'mppi':
            T_for_diff = config['T'] if loading_recovery_model else config['T_orig']
            trajectory_sampler = load_sampler(model_path, dim_mults=(1,2,4), T=T_for_diff, recovery=loading_recovery_model)

        if model_path_orig is not None:
            trajectory_sampler_orig = load_sampler(model_path_orig, dim_mults=(1,2,4), T=config['T_orig'], recovery=False)
        else:
            trajectory_sampler_orig = trajectory_sampler

        if config['recovery_controller'] == 'mppi':
            dynamics = DynamicsModel(env, num_fingers=len(config['fingers']), include_velocity=True, obj_joint_dim=0, hardware=False)
            safety_critic_path = config['model_path']
            if params['OOD_metric'] == 'q_function':
                running_cost = RunningCostSafeRL(safety_critic_path, params['q_cutoff'], env, config['device'], include_velocity=True)
                terminal_cost = None
            elif params['OOD_metric'] == 'likelihood':
                running_cost = lambda x, y: 0
                terminal_cost = TerminalCostDiffusionLikelihood(trajectory_sampler_orig, env, config['device'])
            else:
                raise ValueError('Invalid OOD metric')
            u_max = torch.ones(4 * len(config['fingers'])) * np.pi / 5 
            u_min = - torch.ones(4 * len(config['fingers'])) * np.pi / 5
            noise_sigma = torch.eye(4 * len(config['fingers'])).to(config['device']) * .01


    start_ind = config.get('start_ind', 0)
    step_size = 1
    num_episodes = config['num_episodes']
    if 'end_ind' in config:
        num_episodes = config['end_ind']
    all_turns = []
    for i in tqdm(range(start_ind, num_episodes, step_size)):
        print(f'\nTrial {i+1}')
    # for i in tqdm([1, 2, 4, 7]):
        if config['mode'] != 'hardware':
            torch.manual_seed(i)
            np.random.seed(i)
        # 8709 11200

        env.reset()
        goal = torch.tensor([0, 0, float(config['goal'])])
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
        object_location = torch.from_numpy(env.obj_pose).to(device=params['device']).float()  # TODO: confirm if this is the correct location
        params['object_location'] = object_location

        # If params['device'] is cuda:1 but the computer only has 1 gpu, change to cuda:0
        succ = False
        while not succ:
            perturb_this_trial = (np.random.rand() < .1 or params['perturb_action']) and params['perturb_action']

            if config['mode']  == 'hardware':
                params['perturb_action'] = False
                perturb_this_trial = False

            if not perturb_this_trial:
                print('No action perturbation this trial')
            turn_amount = do_trial(env, params, fpath, sim_env, ros_copy_node,
                                            seed=i, proj_path=None, perturb_this_trial=perturb_this_trial)
            all_turns.append(turn_amount)
            print(f'Average turn amount: {np.mean(all_turns)} over {len(all_turns)} trials')
            succ = True

        print(results)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

