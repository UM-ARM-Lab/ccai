from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv

import numpy as np
import pickle as pkl
import pickle
from copy import deepcopy

import torch
import time
import copy
import yaml
import pathlib
from functools import partial
import sys
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
from ccai.allegro_screwdriver_problem_diffusion import AllegroScrewdriverDiff
from train_allegro_screwdriver import rollout_trajectory_in_sim
from scipy.spatial.transform import Rotation as R

# from ccai.mpc.ipopt import IpoptMPC
# from ccai.problem import IpoptProblem
from ccai.models.trajectory_samplers import TrajectorySampler
from ccai.models.contact_samplers import GraphSearch, Node

from model import LatentDiffusionModel

from diffusion_mcts import DiffusionMCTS

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

print("CCAI_PATH", CCAI_PATH)

device = 'cuda:0'
obj_dof = 3
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')


def vector_cos(a, b):
    return torch.dot(a.reshape(-1), b.reshape(-1)) / (torch.norm(a.reshape(-1)) * torch.norm(b.reshape(-1)))


def euler_to_quat(euler):
    matrix = tf.euler_angles_to_matrix(euler, convention='XYZ')
    quat = tf.matrix_to_quaternion(matrix)
    return quat


def euler_to_angular_velocity(current_euler, next_euler):
    # using matrix

    # current_mat = tf.euler_angles_to_matrix(current_euler, convention='XYZ')
    # next_mat = tf.euler_angles_to_matrix(next_euler, convention='XYZ')
    # dmat = next_mat - current_mat
    # omega_mat = dmat @ current_mat.transpose(-1, -2)
    # omega_x = (omega_mat[..., 2, 1] - omega_mat[..., 1, 2]) / 2
    # omega_y = (omega_mat[..., 0, 2] - omega_mat[..., 2, 0]) / 2
    # omega_z = (omega_mat[..., 1, 0] - omega_mat[..., 0, 1]) / 2
    # omega = torch.stack((omega_x, omega_y, omega_z), dim=-1)

    # R.from_euler('XYZ', current_euler.cpu().detach().numpy().reshape(-1, 3)).as_quat().reshape(3, 12, 4)

    # quaternion 
    current_quat = euler_to_quat(current_euler)
    next_quat = euler_to_quat(next_euler)
    dquat = next_quat - current_quat
    con_quat = - current_quat  # conjugate
    con_quat[..., 0] = current_quat[..., 0]
    omega = 2 * tf.quaternion_raw_multiply(dquat, con_quat)[..., 1:]
    # TODO: quaternion and its negative are the same, but it is not true for angular velocity. Might have some bug here 
    return omega


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
                 device='cuda:0', **kwargs):
        self.obj_mass = 0.1
        self.obj_dof_type = None
        if obj_dof == 3:
            object_link_name = 'screwdriver_body'
        elif obj_dof == 1:
            object_link_name = 'valve'
        elif obj_dof == 6:
            object_link_name = 'card'
        self.obj_link_name = object_link_name
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
                                                 min_force_dict=min_force_dict, **kwargs)
        self.friction_coefficient = friction_coefficient

    def _cost(self, xu, start, goal):
        # TODO: check if the addtional term of the smoothness cost and running goal cost is necessary
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it

        smoothness_cost = torch.sum((state[1:, -self.obj_dof:] - state[:-1, -self.obj_dof:]) ** 2)
        upright_cost = 500 * torch.sum(
            (state[:, -self.obj_dof:-1]) ** 2)  # the screwdriver should only rotate in z direction
        return smoothness_cost + upright_cost + super()._cost(xu, start, goal)

def do_trial(env, params, fpath, sim_viz_env=None, ros_copy_node=None, inits_noise=None, noise_noise=None, sim=None, seed=None):
    "only turn the valve once"
    num_fingers = len(params['fingers'])
    state = env.get_state()
    action_list = []
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None

    start = state['q'].reshape(4 * num_fingers + 4).to(device=params['device'])
    # start = torch.cat((state['q'].reshape(10), torch.zeros(1).to(state['q'].device))).to(device=params['device'])
    if 'csvgd' in params['controller']:
        # index finger is used for stability
        if 'index' in params['fingers']:
            fingers = params['fingers']
        else:
            fingers = ['index'] + params['fingers']

        # if params['mode'] == 'hardware':
        #     min_force_dict = {
        #         'thumb': 1,
        #         'middle': 1,
        #         'index': .0,
        #     }
        # else:
        min_force_dict = None

        # min_force_dict = {
        #     'thumb': 0,
        #     'middle': 0,
        #     'index': 0,
        # }
        # initial grasp
        pregrasp_params = copy.deepcopy(params)
        pregrasp_params['warmup_iters'] = 80
        pregrasp_params['N'] = 16
        pregrasp_params['ode_solve'] = False
        pregrasp_problem = AllegroScrewdriver(
            start=start[:4 * num_fingers + obj_dof],
            goal=pregrasp_params['valve_goal'] * 0,
            T=2,
            chain=pregrasp_params['chain'],
            device=pregrasp_params['device'],
            object_asset_pos=env.table_pose,
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
        )
        # finger gate index
        index_regrasp_problem = AllegroScrewdriver(
            start=start[:4 * num_fingers + obj_dof],
            goal=params['valve_goal'] * 0,
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.table_pose,
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
            min_force_dict=min_force_dict
        )
        thumb_and_middle_regrasp_problem = AllegroScrewdriver(
            start=start[:4 * num_fingers + obj_dof],
            goal=params['valve_goal'] * 0,
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.table_pose,
            object_location=params['object_location'],
            object_type=params['object_type'],
            world_trans=env.world_trans,
            contact_fingers=['index'],
            regrasp_fingers=['middle', 'thumb'],
            obj_dof=obj_dof,
            obj_joint_dim=1,
            optimize_force=params['optimize_force'],
            default_dof_pos=env.default_dof_pos[:, :16],
            obj_gravity=params.get('obj_gravity', False),
            min_force_dict=min_force_dict
        )
        turn_problem = AllegroScrewdriver(
            start=start[:4 * num_fingers + obj_dof],
            goal=params['valve_goal'] * 0,
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.table_pose,
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
            min_force_dict=min_force_dict
        )

        pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, pregrasp_params)
        index_regrasp_planner = PositionControlConstrainedSVGDMPC(index_regrasp_problem, params)
        thumb_and_middle_regrasp_planner = PositionControlConstrainedSVGDMPC(thumb_and_middle_regrasp_problem, params)
        turn_planner = PositionControlConstrainedSVGDMPC(turn_problem, params)


    # elif params['controller'] == 'ipopt':
    #     # index finger is used for stability
    #     if 'index' in params['fingers']:
    #         fingers = params['fingers']
    #     else:
    #         fingers = ['index'] + params['fingers']

    #     # initial grasp
    #     pregrasp_problem = IpoptScrewdriver(
    #         start=start[:4 * num_fingers + obj_dof],
    #         goal=params['valve_goal'] * 0,
    #         T=params['T'],
    #         chain=params['chain'],
    #         device=params['device'],
    #         object_asset_pos=env.table_pose,
    #         object_location=params['object_location'],
    #         object_type=params['object_type'],
    #         world_trans=env.world_trans,
    #         regrasp_fingers=fingers,
    #         contact_fingers=[],
    #         obj_dof=obj_dof,
    #         obj_joint_dim=1,
    #         optimize_force=params['optimize_force'],
    #     )
    #     # finger gate index
    #     index_regrasp_problem = IpoptScrewdriver(
    #         start=start[:4 * num_fingers + obj_dof],
    #         goal=params['valve_goal'] * 0,
    #         T=params['T'],
    #         chain=params['chain'],
    #         device=params['device'],
    #         object_asset_pos=env.table_pose,
    #         object_location=params['object_location'],
    #         object_type=params['object_type'],
    #         world_trans=env.world_trans,
    #         regrasp_fingers=['index'],
    #         contact_fingers=['middle', 'thumb'],
    #         obj_dof=obj_dof,
    #         obj_joint_dim=1,
    #         optimize_force=params['optimize_force'],
    #         default_dof_pos=env.default_dof_pos[:, :16]
    #     )
    #     thumb_and_middle_regrasp_problem = IpoptScrewdriver(
    #         start=start[:4 * num_fingers + obj_dof],
    #         goal=params['valve_goal'] * 0,
    #         T=params['T'],
    #         chain=params['chain'],
    #         device=params['device'],
    #         object_asset_pos=env.table_pose,
    #         object_location=params['object_location'],
    #         object_type=params['object_type'],
    #         world_trans=env.world_trans,
    #         contact_fingers=['index'],
    #         regrasp_fingers=['middle', 'thumb'],
    #         obj_dof=obj_dof,
    #         obj_joint_dim=1,
    #         optimize_force=params['optimize_force'],
    #         default_dof_pos=env.default_dof_pos[:, :16]
    #     )
    #     turn_problem = IpoptScrewdriver(
    #         start=start[:4 * num_fingers + obj_dof],
    #         goal=params['valve_goal'] * 0,
    #         T=params['T'],
    #         chain=params['chain'],
    #         device=params['device'],
    #         object_asset_pos=env.table_pose,
    #         object_location=params['object_location'],
    #         object_type=params['object_type'],
    #         world_trans=env.world_trans,
    #         contact_fingers=['index', 'middle', 'thumb'],
    #         obj_dof=obj_dof,
    #         obj_joint_dim=1,
    #         optimize_force=params['optimize_force'],
    #         default_dof_pos=env.default_dof_pos[:, :16]
    #     )
    #     pregrasp_planner = IpoptMPC(pregrasp_problem, params)
    #     index_regrasp_planner = IpoptMPC(index_regrasp_problem, params)
    #     thumb_and_middle_regrasp_planner = IpoptMPC(thumb_and_middle_regrasp_problem, params)
    #     turn_planner = IpoptMPC(turn_problem, params)
    # else:
    #     raise ValueError('Invalid controller')

    elif params['controller'] == 'diffusion_policy':
        from ccai.mpc.diffusion_policy import Diffusion_Policy, DummyProblem
        if 'index' in params['fingers']:
            fingers = params['fingers']
        else:
            fingers = ['index'] + params['fingers']
        problem = DummyProblem(params['dx'], params['T'])
        planner = Diffusion_Policy(problem, params)
        pregrasp_params = copy.deepcopy(params)
        pregrasp_params['warmup_iters'] = 80
        pregrasp_problem = AllegroScrewdriver(
            start=start[:4 * num_fingers + obj_dof],
            goal=pregrasp_params['valve_goal'] * 0,
            T=2,
            chain=pregrasp_params['chain'],
            device=pregrasp_params['device'],
            object_asset_pos=env.table_pose,
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
        )
        pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, pregrasp_params)

        turn_problem = AllegroScrewdriver(
            start=start[:4 * num_fingers + obj_dof],
            goal=params['valve_goal'] * 0,
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.table_pose,
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
        )
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    # warm-starting using learned sampler
    trajectory_sampler = None
    model_path = params.get('model_path', None)

    if model_path is not None:
        problem_for_sampler = None
        if params['projected'] or params['sample_contact'] or params['type'] == 'cnf':
            pregrasp_problem_diff = AllegroScrewdriver(
                start=start[:4 * num_fingers + obj_dof],
                goal=params['valve_goal'],
                T=1,
                chain=params['chain'],
                device=params['device'],
                object_asset_pos=env.table_pose,
                object_location=params['object_location'],
                object_type=params['object_type'],
                world_trans=env.world_trans,
                regrasp_fingers=fingers,
                contact_fingers=[],
                obj_dof=obj_dof,
                obj_joint_dim=1,
                optimize_force=params['optimize_force'],
            )
            # finger gate index
            index_regrasp_problem_diff = AllegroScrewdriver(
                start=start[:4 * num_fingers + obj_dof],
                goal=params['valve_goal'],
                T=1,
                chain=params['chain'],
                device=params['device'],
                object_asset_pos=env.table_pose,
                object_location=params['object_location'],
                object_type=params['object_type'],
                world_trans=env.world_trans,
                regrasp_fingers=['index'],
                contact_fingers=['middle', 'thumb'],
                obj_dof=obj_dof,
                obj_joint_dim=1,
                optimize_force=params['optimize_force'],
                default_dof_pos=env.default_dof_pos[:, :16]
            )
            thumb_and_middle_regrasp_problem_diff = AllegroScrewdriver(
                start=start[:4 * num_fingers + obj_dof],
                goal=params['valve_goal'],
                T=1,
                chain=params['chain'],
                device=params['device'],
                object_asset_pos=env.table_pose,
                object_location=params['object_location'],
                object_type=params['object_type'],
                world_trans=env.world_trans,
                contact_fingers=['index'],
                regrasp_fingers=['middle', 'thumb'],
                obj_dof=obj_dof,
                obj_joint_dim=1,
                optimize_force=params['optimize_force'],
                default_dof_pos=env.default_dof_pos[:, :16]
            )
            turn_problem_diff = AllegroScrewdriver(
                start=start[:4 * num_fingers + obj_dof],
                goal=params['valve_goal'],
                T=1,
                chain=params['chain'],
                device=params['device'],
                object_asset_pos=env.table_pose,
                object_location=params['object_location'],
                object_type=params['object_type'],
                world_trans=env.world_trans,
                contact_fingers=['index', 'middle', 'thumb'],
                obj_dof=obj_dof,
                obj_joint_dim=1,
                optimize_force=params['optimize_force'],
                default_dof_pos=env.default_dof_pos[:, :16]
            )

            if params['use_partial_constraint']:
                problem_for_sampler = {
                    (-1, -1, -1): pregrasp_problem_diff,
                    (-1, 1, 1): index_regrasp_problem_diff,
                    (1, -1, -1): thumb_and_middle_regrasp_problem_diff,
                    (1, 1, 1): turn_problem_diff
                }
            else:
                problem_for_sampler = {
                    (-1, -1, -1): pregrasp_problem,
                    (-1, 1, 1): index_regrasp_problem,
                    (1, -1, -1): thumb_and_middle_regrasp_problem,
                    (1, 1, 1): turn_problem
                }
        if 'type' not in params:
            params['type'] = 'diffusion'

        vae = None
        model_t = params['type'] == 'latent_diffusion'
        if model_t:
            vae_path = params.get('vae_path', None)
            vae = LatentDiffusionModel(params, None).to(params['device'])
            vae.load_state_dict(torch.load(f'{CCAI_PATH}/{vae_path}'))
            for param in vae.parameters():
                param.requires_grad = False
        trajectory_sampler = TrajectorySampler(T=params['T'] + 1, dx=(15 + (1 if params['sine_cosine'] else 0)) if not model_t else params['nzt'], du=21 if not model_t else 0, type=params['type'],
                                               timesteps=256, hidden_dim=256 if not model_t else 64,
                                               context_dim=3, generate_context=False,
                                               constrain=params['projected'],
                                               problem=problem_for_sampler,
                                               inits_noise=inits_noise, noise_noise=noise_noise,
                                               guided=params['use_guidance'],
                                               state_control_only=params.get('state_control_only', False),
                                               vae=vae)
        # try:
        trajectory_sampler.load_state_dict(torch.load(f'{CCAI_PATH}/{model_path}', map_location=torch.device(params['device'])), strict=True)
        # except:
        #     print('failed to load model')
        trajectory_sampler.to(device=params['device'])
        trajectory_sampler.send_norm_constants_to_submodels()
        if params['project_state']:
            trajectory_sampler.model.diffusion_model.classifier=None

        # trajectory_sampler.eval()
        print('Loaded trajectory sampler')

    # start = env.get_state()['q'].reshape(4 * num_fingers + 4).to(device=params['device'])
    # best_traj, _ = pregrasp_planner.step(start[:4 * num_fingers + obj_dof])
    #
    # for x in best_traj[:, :4 * num_fingers]:
    #     action = x.reshape(-1, 4 * num_fingers).to(device=env.device)  # move the rest fingers
    #     if params['mode'] == 'hardware':
    #         sim_viz_env.set_pose(env.get_state()['all_state'].to(device=env.device))
    #         sim_viz_env.step(action)
    #     env.step(action)
    #     action_list.append(action)
    #     if params['mode'] == 'hardware_copy':
    #         ros_copy_node.apply_action(partial_to_full_state(x.reshape(-1, 4 * num_fingers)[0], params['fingers']))


    state = env.get_state()
    start = state['q'].reshape(4 * num_fingers + 4).to(device=params['device'])
    if params['exclude_index']:
        turn_problem_fingers = copy.copy(params['fingers'])
        turn_problem_fingers.remove('index')
        turn_problem_start = start[4:4 * num_fingers + obj_dof]
    else:
        turn_problem_fingers = params['fingers']
        turn_problem_start = start[:4 * num_fingers + obj_dof]

    actual_trajectory = []
    duration = 0

    fig = plt.figure()
    axes = {params['fingers'][i]: fig.add_subplot(int(f'1{num_fingers}{i + 1}'), projection='3d') for i in
            range(num_fingers)}
    for finger in params['fingers']:
        axes[finger].set_title(finger)
        axes[finger].set_aspect('equal')
        axes[finger].set_xlabel('x', labelpad=20)
        axes[finger].set_ylabel('y', labelpad=20)
        axes[finger].set_zlabel('z', labelpad=20)
        axes[finger].set_xlim3d(-0.05, 0.1)
        axes[finger].set_ylim3d(-0.06, 0.04)
        axes[finger].set_zlim3d(1.32, 1.43)
    finger_traj_history = {}
    for finger in params['fingers']:
        finger_traj_history[finger] = []

    for finger in params['fingers']:
        ee = state2ee_pos(start[:4 * num_fingers], turn_problem.ee_names[finger])
        finger_traj_history[finger].append(ee.detach().cpu().numpy())

    if params['exclude_index']:
        num_fingers_to_plan = num_fingers - 1
    else:
        num_fingers_to_plan = num_fingers
    info_list = []

    def _partial_to_full(traj, mode):
        if mode == 'index':
            traj = torch.cat((traj[..., :-6], torch.zeros(*traj.shape[:-1], 3).to(device=params['device']),
                              traj[..., -6:]), dim=-1)
        if mode == 'thumb_middle':
            traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 6).to(device=params['device'])), dim=-1)
        if mode == 'pregrasp':
            traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 9).to(device=params['device'])), dim=-1)
        return traj

    def _full_to_partial(traj, mode):
        if mode == 'index':
            traj = torch.cat((traj[..., :-9], traj[..., -6:]), dim=-1)
        if mode == 'thumb_middle':
            traj = traj[..., :-6]
        if mode == 'pregrasp':
            traj = traj[..., :-9]
        return traj
    
    def convert_sine_cosine_to_yaw(xu):
        """
        xu is shape (N, T, 37)
        Replace the sine and cosine in xu with yaw and return the new xu
        """
        sine = xu[..., 15]
        cosine = xu[..., 14]
        yaw = torch.atan2(sine, cosine)
        xu_new = torch.cat([xu[..., :14], yaw.unsqueeze(-1), xu[..., 16:]], dim=-1)
        return xu_new
    
    def convert_yaw_to_sine_cosine(xu):
        """
        xu is shape (N, T, 36)
        Replace the yaw in xu with sine and cosine and return the new xu
        """
        yaw = xu[14]
        sine = torch.sin(yaw)
        cosine = torch.cos(yaw)
        xu_new = torch.cat([xu[:14], cosine.unsqueeze(-1), sine.unsqueeze(-1), xu[15:]], dim=-1)
        return xu_new

    def execute_traj(planner, mode, goal=None, fname=None, initial_samples=None):
        # reset planner
        state = env.get_state()
        state = state['q'].reshape(-1)[:15].to(device=params['device'])

        # generate context from mode
        contact = -torch.ones(params['N'], 3).to(device=params['device'])
        if mode == 'thumb_middle':
            contact[:, 0] = 1
        elif mode == 'index':
            contact[:, 1] = 1
            contact[:, 2] = 1
        elif mode == 'turn':
            contact[:, :] = 1

        # generate initial samples with diffusion model
        sim_rollouts = None
        if trajectory_sampler is not None and params.get('diff_init', True):
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
            
            if params['type'] == 'cnf':
                orig_yaw = start_for_diff[14]
                start_for_diff[14] = 0
            if params['project_state']:
                initial_samples, _, _, initial_samples_0, (all_losses, all_samples, all_likelihoods) = trajectory_sampler.sample(N=params['N'], start=start_for_diff.reshape(1, -1),
                                                                    H=params['T'] + 1,
                                                                    constraints=contact,
                                                                    project=params['project_state'],)
            else:
                initial_samples, _, _ = trajectory_sampler.sample(N=params['N'], start=start_for_diff.reshape(1, -1),
                                                                    H=params['T'] + 1,
                                                                    constraints=contact,
                                                                    project=params['project_state'],)
            mode_fpath = f'{fpath}/{fname}'
            pathlib.Path.mkdir(pathlib.Path(mode_fpath), parents=True, exist_ok=True)
            if params['project_state']:    
                with open(mode_fpath+ '/projection_results.pkl', 'wb') as f:
                    pickle.dump((initial_samples, initial_samples_0, all_losses, all_samples, all_likelihoods), f)
            if params['sine_cosine']:
                initial_samples = convert_sine_cosine_to_yaw(initial_samples)
                if params['project_state']:
                    initial_samples_0 = convert_sine_cosine_to_yaw(initial_samples_0)
            print('Sampling time', time.perf_counter() - a)

            if params['type'] == 'cnf':
                initial_samples[..., 14] += orig_yaw
            # if state[-1] < -1.0:
            #     initial_samples[:, :, -1] -= 0.75
            if params['visualize_plan']:
                if params['project_state']:
                    iter_set = [('initial_samples_project', initial_samples), ('initial_samples_0', initial_samples_0)]
                else:
                    iter_set = [('initial_samples', initial_samples)]
                for (name, traj_set) in iter_set:
                    for k in range(params['N']):
                        traj_for_viz = traj_set[k, :, :planner.problem.dx]
                        # if params['exclude_index']:
                        #     traj_for_viz = torch.cat((state[4:4 + planner.problem.dx].unsqueeze(0), traj_for_viz), dim=0)
                        # else:
                        #     traj_for_viz = torch.cat((state[:planner.problem.dx].unsqueeze(0), traj_for_viz), dim=0)
                        tmp = torch.zeros((traj_for_viz.shape[0], 1),
                                        device=traj_for_viz.device)  # add the joint for the screwdriver cap
                        traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)
                        # traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof] = axis_angle_to_euler(traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof])

                        viz_fpath = pathlib.PurePath.joinpath(fpath, f"{fname}/{name}/{k}")
                        img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
                        gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
                        pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
                        pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
                        visualize_trajectory(traj_for_viz, turn_problem.contact_scenes, viz_fpath,
                                            turn_problem.fingers, turn_problem.obj_dof + 1)
            sim_rollouts = torch.zeros_like(initial_samples)
            torch.cuda.empty_cache()

            # for i in range(params['N']):
            #     sim_rollout = rollout_trajectory_in_sim(env_sim_rollout, initial_samples[i])
            #     sim_rollouts[i] = sim_rollout
        if initial_samples is not None:
            if params['mode'] == 'hardware' and mode == 'turn':
                initial_samples[..., 30:] = 1.5 * torch.randn(params['N'], params['T']+1, 6, device=initial_samples.device)

            initial_samples = _full_to_partial(initial_samples, mode)
            initial_x = initial_samples[:, 1:, :planner.problem.dx]
            initial_u = initial_samples[:, :-1, -planner.problem.du:]
            initial_samples = torch.cat((initial_x, initial_u), dim=-1)


        state = env.get_state()
        state = state['q'].reshape(-1).to(device=params['device'])
        state = state[:planner.problem.dx]
        # print(params['T'], state.shape, initial_samples)

        planner.reset(state, T=params['T'], goal=goal, initial_x=initial_samples)
        if params['controller'] != 'diffusion_policy' and (trajectory_sampler is None or not params.get('diff_init', True)):
            initial_samples = planner.x.detach().clone()
            sim_rollouts = torch.zeros_like(initial_samples)
        elif params['controller'] == 'diffusion_policy':
            initial_samples = torch.tensor([])
            sim_rollouts = torch.zeros_like(initial_samples)
        planned_trajectories = []
        actual_trajectory = []
        optimizer_paths = []
        contact_points = {
        }
        contact_distance = {
        }
        plans = None
        resample = params.get('diffusion_resample', False)
        for k in range(planner.problem.T):  # range(params['num_steps']):
            state = env.get_state()
            state = state['q'].reshape(4 * num_fingers + 4).to(device=params['device'])
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

            s = time.time()
            best_traj, plans = planner.step(state)
            print(f'Solve time for step {k+1}', time.time() - s)
            planned_trajectories.append(plans)
            optimizer_paths.append(copy.deepcopy(planner.path))
            N, T, _ = plans.shape

            if planner.problem.data is not None:
                contact_distance[T] = torch.stack((planner.problem.data['index']['sdf'].reshape(N, T + 1),
                                                planner.problem.data['middle']['sdf'].reshape(N, T + 1),
                                                planner.problem.data['thumb']['sdf'].reshape(N, T + 1)),
                                                dim=1).detach().cpu()

                contact_points[T] = torch.stack((planner.problem.data['index']['closest_pt_world'].reshape(N, T + 1, 3),
                                                planner.problem.data['middle']['closest_pt_world'].reshape(N, T + 1, 3),
                                                planner.problem.data['thumb']['closest_pt_world'].reshape(N, T + 1, 3)),
                                                dim=2).detach().cpu()

            # execute the action
            state = env.get_state()
            state = state['q'].reshape(-1).to(device=params['device'])
            ori = state[:15][-3:]
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
            elif mode == 'thumb_middle':
                index_force = torch.norm(best_traj[..., 27:30], dim=-1)
                print('Index force:', index_force)
            if params['controller'] != 'diffusion_policy':
                action = best_traj[0, planner.problem.dx:planner.problem.dx + planner.problem.du]
                x = best_traj[0, :planner.problem.dx + planner.problem.du]
                x = x.reshape(1, planner.problem.dx + planner.problem.du)
                action = x[:, planner.problem.dx:planner.problem.dx + planner.problem.du].to(device=env.device)
            else:
                action = best_traj
            xu = torch.cat((state[:-1].cpu(), action[0].cpu()))
            actual_trajectory.append(xu)
            # print(action)
            action = action[:, :4 * num_fingers_to_plan]
            if params['exclude_index']:
                action = state.unsqueeze(0)[:, 4:4 * num_fingers] + action
                action = torch.cat((state.unsqueeze(0)[:, :4], action), dim=1)  # add the index finger back
            else:
                action = action.to(device=env.device) + state.unsqueeze(0)[:, :4 * num_fingers].to(device=env.device)

            if params['mode'] == 'hardware':
                set_state = env.get_state()['q'].to(device=env.device)
                # print(set_state.shape)
                sim_viz_env.set_pose(set_state)
                # sim_viz_env.step(action)
                # for _ in range(3):
                #     sim_viz_env.step(action)
                state = sim_viz_env.get_state()['q'].reshape(-1).to(device=params['device'])
                print(state[:15][-3:])
            elif params['mode'] == 'hardware_copy':
                ros_copy_node.apply_action(partial_to_full_state(action[0], params['fingers']))

            if params['visualize'] and best_traj.shape[0] > 1 and False:
                add_trajectories(plans.to(device=env.device), best_traj.to(device=env.device),
                                 axes, env, sim=sim, gym=gym, viewer=viewer,
                                 config=params, state2ee_pos_func=state2ee_pos,
                                 show_force=(planner == turn_planner and params['optimize_force']))
            if params['visualize_plan']:
                traj_for_viz = best_traj[:, :planner.problem.dx]
                if params['exclude_index']:
                    traj_for_viz = torch.cat((state[4:4 + planner.problem.dx].unsqueeze(0), traj_for_viz), dim=0)
                else:
                    traj_for_viz = torch.cat((state[:planner.problem.dx].unsqueeze(0), traj_for_viz), dim=0)
                tmp = torch.zeros((traj_for_viz.shape[0], 1),
                                  device=best_traj.device)  # add the joint for the screwdriver cap
                traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)
                # traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof] = axis_angle_to_euler(traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof])

                viz_fpath = pathlib.PurePath.joinpath(fpath, f"{fname}/timestep_{k}")
                img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
                gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
                pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
                pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
                visualize_trajectory(traj_for_viz, turn_problem.contact_scenes, viz_fpath,
                                     turn_problem.fingers, turn_problem.obj_dof + 1)

            env.step(action.to(device=env.device))

            # turn_problem._preprocess(best_traj.unsqueeze(0))
            # equality_constr_dict = turn_problem._con_eq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False,
            #                                             verbose=True)
            # inequality_constr_dict = turn_problem._con_ineq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False,
            #                                             verbose=True)
            # print("--------------------------------------")
            # distance2goal = (
            #             params['valve_goal'].cpu() - env.get_state()['q'][:, -obj_dof - 1: -1].cpu()).detach().cpu()
            # print(distance2goal)
            # info = {**equality_constr_dict, **inequality_constr_dict, **{'distance2goal': distance2goal}}
            # info_list.append(info)
            if params['visualize'] and False:
                gym.clear_lines(viewer)
                state = env.get_state()
                start = state['q'][:, :4 * num_fingers + obj_dof].squeeze(0).to(device=params['device'])
                for finger in params['fingers']:
                    ee = state2ee_pos(start[:4 * num_fingers], turn_problem.ee_names[finger])
                    finger_traj_history[finger].append(ee.detach().cpu().numpy())
                for finger in params['fingers']:
                    traj_history = finger_traj_history[finger]
                    temp_for_plot = np.stack(traj_history, axis=0)
                    if k >= 2:
                        axes[finger].plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'gray',
                                            label='actual')

        # actual_trajectory.append(env.get_state()['q'].reshape(9).to(device=params['device']))
        actual_trajectory = torch.stack(actual_trajectory, dim=0).to(device=params['device'])
        # can't stack plans as each is of a different length

        # for memory reasons we clear the data
        if params['controller'] != 'diffusion_policy':
            planner.problem.data = {}
        return actual_trajectory, planned_trajectories, initial_samples, sim_rollouts, optimizer_paths, contact_points, contact_distance

    data = {}
    for t in range(1, 1 + params['T']):
        data[t] = {'plans': [], 'starts': [], 'inits': [], 'init_sim_rollouts': [], 'optimizer_paths': [], 'contact_points': [], 'contact_distance': [], 'contact_state': []}

        # sample initial trajectory with diffusion model to get contact sequence
    state = env.get_state()
    state = state['q'].reshape(-1).to(device=params['device'])

    # generate initial samples with diffusion model
    initial_samples = None

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
            data[t]['optimizer_paths'].append([i.cpu().numpy() for i in optimizer_paths])
            data[t]['starts'].append(traj[i].reshape(1, -1).repeat(plan.shape[0], 1))
            try:
                data[t]['contact_points'].append(contact_points[t])
                data[t]['contact_distance'].append(contact_distance[t])
                data[t]['contact_state'].append(contact_state)
            except:
                pass

    def visualize_trajectory_wrapper(traj, contact_scenes, fname, plan_or_init, index, fingers, obj_dof, k):
        viz_fpath = pathlib.PurePath.joinpath(fpath, f"{fname}/{plan_or_init}/{index}/timestep_{k}")
        img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
        gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
        pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
        pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
        visualize_trajectory(traj, contact_scenes, viz_fpath, fingers, obj_dof + 1)

    def plan_contacts(state, stage, depth, next_node, multi_particle=False):
        torch.cuda.empty_cache()
        state_for_search = state#convert_yaw_to_sine_cosine(state)
        next_node_init = next_node is None

        num_samples_per_node = 1
        if multi_particle:
            num_samples_per_node = 16
        if next_node is None:
            next_node = Node(
                # torch.empty(num_samples_per_node, 0, 36),
                torch.empty(num_samples_per_node, 0, 37 if params['sine_cosine'] else 36),#.to(device=params['device']),
                0,
                tuple(),
                # torch.empty(num_samples_per_node * 4, 0, 36),
                torch.zeros(num_samples_per_node),
                # torch.arange(16)
            )
            initial_run = True
        else:
            next_node = Node(
                # torch.empty(num_samples_per_node, 0, 36),
                torch.empty(num_samples_per_node, 0, 37 if params['sine_cosine'] else 36),#.to(device=params['device']),
                0,
                (next_node, ),
                # torch.empty(num_samples_per_node * 4, 0, 36),
                torch.zeros(num_samples_per_node),
                # torch.arange(16)
            )
            initial_run = False
        
        contact_sequence_sampler = GraphSearch(state_for_search, trajectory_sampler, params['T'], problem_for_sampler, 
                                               depth, params['heuristic'], params['goal'], 
                                               torch.device(params['device']), initial_run=initial_run,
                                               multi_particle=multi_particle,
                                               prior=params['prior'],
                                               sine_cosine=params['sine_cosine'])
        a = time.perf_counter()
        contact_node_sequence = contact_sequence_sampler.astar(next_node, None)
        planning_time = time.perf_counter() - a
        print('Contact sequence search time:', planning_time)
        closed_set = contact_sequence_sampler.closed_set
        if contact_node_sequence is not None:
            contact_node_sequence = list(contact_node_sequence)
        pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)

        with open(f"{fpath}/contact_planning_{stage}.pkl", "wb") as f:
            pkl.dump((contact_node_sequence, closed_set, planning_time, contact_sequence_sampler.iter), f)
        if contact_node_sequence is None:
            print('No contact sequence found')
            # Find the node in the closed set with the lowest cost
            min_yaw = float('inf')
            min_node = None
            for node in closed_set:
                yaw = contact_sequence_sampler.get_expected_yaw(node.data)
                if yaw < min_yaw:
                    min_yaw = yaw
                    min_node = node
            contact_node_sequence = [min_node.data]
            # return None, None, None
        last_node = contact_node_sequence[-1]

        if next_node_init:
            offset = 0
        else:
            offset = 1
        if offset == 1 and len(contact_node_sequence) == 1:
            return [], None, None
        next_node = last_node.contact_sequence[offset]
        contact_sequence = np.array(last_node.contact_sequence)
        contact_sequence = (contact_sequence + 1)/2
        contact_sequence = [contact_vec_to_label[contact_sequence[i].sum()] for i in range(contact_sequence.shape[0])]
        contact_sequence = contact_sequence[offset:] 

        initial_samples = None

        torch.cuda.empty_cache()
        return contact_sequence, next_node, initial_samples

    state = env.get_state()
    state = state['q'].reshape(-1)[:15].to(device=params['device'])


    contact_label_to_vec = {'pregrasp': 0,
                            'index': 2,
                            'thumb_middle': 1,
                            'turn': 3
                            }
    contact_vec_to_label = dict((v, k) for k, v in contact_label_to_vec.items())


    sample_contact = params.get('sample_contact', False)
    num_stages = 2 + 3 * (params['num_turns'] - 1)
    if not sample_contact:
        contact_sequence = ['turn']
        for k in range(params['num_turns'] - 1):
            contact_options = ['index', 'thumb_middle']
            perm = np.random.permutation(2)
            # perm = [0, 1]
            contact_sequence += [contact_options[perm[0]], contact_options[perm[1]], 'turn']
        # contact_sequence = ['thumb_middle']
    else:
        contact_sequence = None

    # state = state['q'].reshape(-1)[:15].to(device=params['device'])
    # initial_samples = gen_initial_samples_multi_mode(contact_sequence)
    # pkl.dump(initial_samples, open(f"{fpath}/long_horizon_inits.p", "wb"))
    # return -1
    contact = None
    next_node = None
    state = env.get_state()
    state = state['q'].reshape(-1)[:15].to(device=params['device'])

    executed_contacts = []
    stages_since_plan = 0

    for stage in range(num_stages):
        state = env.get_state()
        state = state['q'].reshape(-1)[:15].to(device=params['device'])
        ori = state[:15][-3:]
        yaw = ori[-1]
        print('Current yaw:', ori)
        # valve_goal = torch.tensor([0, 0, state[-1]]).to(device=params['device'])

        # if sample_contact:
        #     with torch.no_grad():
        #         start = state.clone()
        #         if state[-1] < -1.0:
        #             start[-1] += 0.75
        #         trajectories, contact, likelihoods = trajectory_sampler.sample(N=512, start=start.reshape(1, -1),
        #                                                                        H=(num_stages - stage) * 16)
        #     # choose highest likelihood trajectory
        #     contact_sequence = contact[torch.argmax(likelihoods)]
        #     contact_sequence = torch.round(((contact_sequence + 1) / 2)).int()
        #     print(contact_sequence)
        #     # convert to number
        #     contact_sequence = bin2dec(contact_sequence, 3)
        #     # choose first in sequence
        #     contact = contact_vec_to_label[contact_sequence.reshape(-1)[0].item()]

        # else:
        # contact = contact_sequence[stage]
        initial_samples = None
        if params['controller'] == 'diffusion_policy' and stage > 0:
            _goal = torch.tensor([0, 0, state[-1]]).to(device=params['device'])
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance = execute_traj(
                planner, mode='diffusion_policy', goal=_goal, fname=f'diffusion_policy_{stage}')

            _add_to_dataset(traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance,
                            contact_state=torch.tensor([0.0, 0.0, 0.0]))
            actual_trajectory.append(traj)
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
            pickle.dump(data_save, open(f"{fpath}/traj_data.p", "wb"))
            del data_save
            state = env.get_state()
            state = state['q'].reshape(4 * num_fingers + obj_dof + 1).to(device=params['device'])
            actual_trajectory_save = deepcopy(actual_trajectory)
            actual_trajectory_save.append(state.clone()[: 4 * num_fingers + obj_dof])
            # actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 4 * num_fingers + obj_dof)
            # turn_problem.T = actual_trajectory.shape[0]
            # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
            # final_distance_to_goal = (state.clone()[:, -obj_dof:] - params['valve_goal']).abs()

            # print(f'Controller: {params["controller"]} Final distance to goal: {torch.min(final_distance_to_goal)}')
            # print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')

            with open(f'{fpath.resolve()}/trajectory.pkl', 'wb') as f:
                pickle.dump([i.cpu().numpy() for i in actual_trajectory_save], f)
            del actual_trajectory_save
            continue
        if stage == 0:
            contact = 'pregrasp'
            start = env.get_state()['q'].reshape(4 * num_fingers + 4).to(device=params['device'])
            a = time.perf_counter()
            best_traj, _ = pregrasp_planner.step(start[:pregrasp_planner.problem.dx])
            print(f'Solve time for pregrasp', time.perf_counter() - a)
            for x in best_traj[:, :4 * num_fingers]:
                action = x.reshape(-1, 4 * num_fingers).to(device=env.device) # move the rest fingers
                env.step(action)
                if params['mode'] == 'hardware':
                    set_state = env.get_state()['q'].to(device=env.device)
                    # print(set_state.shape)
                    sim_viz_env.set_pose(set_state)  
                    # for i in range(3):
                    #     sim_viz_env.step(action)
                    state = sim_viz_env.get_state()['q'].reshape(-1).to(device=params['device'])
            if params['mode'] == 'hardware':
                input("Pregrasp complete. Ready to execute. Press <ENTER> to continue.")
            print('pregrasp complete:', state[:15][-3:])
            continue
        elif sample_contact and (stage == 1 or (params['replan'] and (stages_since_plan == 0 or len(contact_sequence) == 1))):
            # if yaw <= params['goal']:
            #     # params['goal'] -= .5
            #     params['goal'] = yaw + float(params['goal_update'])
            #     print('Adjusting goal to', params['goal'])
            # new_contact_sequence, new_next_node, initial_samples = plan_contacts(state, num_stages - stage, next_node, params['multi_particle_search'])
            
            if params['replan']:
                params['goal'] = yaw + float(params['goal_update'])
            print('Adjusting goal to', params['goal'])
            for key in problem_for_sampler:
                problem_for_sampler[key].goal = torch.tensor([0, 0, params['goal']]).to(device=params['device'])
            new_contact_sequence, new_next_node, initial_samples = plan_contacts(state, stage, 7, next_node, params['multi_particle_search'])
            stages_since_plan = 0
            if new_contact_sequence is not None and len(new_contact_sequence) == 0:
                print('Planner thinks task is complete')
                print(executed_contacts)
                with open(f"{fpath}/executed_contacts.pkl", "wb") as f:
                    pkl.dump(executed_contacts, f)
                break
            if new_contact_sequence is not None:
                contact_sequence = new_contact_sequence
                next_node = new_next_node
            else:
                if len(contact_sequence) < 2:
                    print('Planner thinks task is complete')
                    print(executed_contacts)
                    with open(f"{fpath}/executed_contacts.pkl", "wb") as f:
                        pkl.dump(executed_contacts, f)
                    break
                contact_sequence = contact_sequence[1:]
                if contact_sequence[0] == 'turn':
                    next_node = (1, 1, 1)
                elif contact_sequence[0] == 'index':
                    next_node = (-1, 1, 1)
                elif contact_sequence[0] == 'thumb_middle':
                    next_node = (1, -1, -1)
                else:
                    next_node = (-1, -1, -1)
            print(contact_sequence)
            # return -1
            contact = contact_sequence[0]  
        elif stage > len(contact_sequence):
            print('Planner thinks task is complete')
            print(executed_contacts)
            break
        else:
            contact = contact_sequence[stage-1]
        executed_contacts.append(contact)
        print(stage, contact)
        if contact == 'index':
            _goal = torch.tensor([0, 0, state[-1]]).to(device=params['device'])
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance = execute_traj(
                index_regrasp_planner, mode='index', goal=_goal, fname=f'index_regrasp_{stage}', initial_samples=initial_samples)

            plans = [torch.cat((plan[..., :-6],
                                torch.zeros(*plan.shape[:-1], 3).to(device=params['device']),
                                plan[..., -6:]),
                               dim=-1) for plan in plans]
            traj = torch.cat((traj[..., :-6], torch.zeros(*traj.shape[:-1], 3).to(device=params['device']),
                              traj[..., -6:]), dim=-1)
            _add_to_dataset(traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance,
                            contact_state=torch.tensor([0.0, 1.0, 1.0]))
        elif contact == 'thumb_middle':
            _goal = torch.tensor([0, 0, state[-1]]).to(device=params['device'])
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance = execute_traj(
                thumb_and_middle_regrasp_planner, mode='thumb_middle',
                goal=_goal, fname=f'thumb_middle_regrasp_{stage}', initial_samples=initial_samples)
            plans = [torch.cat((plan,
                                torch.zeros(*plan.shape[:-1], 6).to(device=params['device'])),
                               dim=-1) for plan in plans]
            traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 6).to(device=params['device'])), dim=-1)

            _add_to_dataset(traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance,
                            contact_state=torch.tensor([1.0, 0.0, 0.0]))
        elif contact == 'turn':
            _goal = torch.tensor([0, 0, state[-1] - np.pi / 6]).to(device=params['device'])
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance = execute_traj(
                turn_planner, mode='turn', goal=_goal, fname=f'turn_{stage}', initial_samples=initial_samples)

            _add_to_dataset(traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, contact_state=torch.ones(3))
        if contact != 'pregrasp':
            actual_trajectory.append(traj)
        # change to numpy and save data
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
        state = state['q'].reshape(4 * num_fingers + obj_dof + 1).to(device=params['device'])
        actual_trajectory_save = deepcopy(actual_trajectory)
        actual_trajectory_save.append(state.clone()[: 4 * num_fingers + obj_dof])
        # actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 4 * num_fingers + obj_dof)
        # turn_problem.T = actual_trajectory.shape[0]
        # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
        # final_distance_to_goal = (state.clone()[:, -obj_dof:] - params['valve_goal']).abs()

        # print(f'Controller: {params["controller"]} Final distance to goal: {torch.min(final_distance_to_goal)}')
        # print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')

        with open(f'{fpath.resolve()}/trajectory.pkl', 'wb') as f:
            pickle.dump([i.cpu().numpy() for i in actual_trajectory_save], f)
        del actual_trajectory_save
    # np.savez(f'{fpath.resolve()}/trajectory.npz', x=[i.cpu().numpy() for i in actual_trajectory],)
             #  constr=constraint_val.cpu().numpy(),
            #  d2goal=final_distance_to_goal.cpu().numpy())
    if params['mode'] == 'hardware':
        input("Ready to regrasp. Press <ENTER> to continue.")

    env.reset()
    return -1#torch.min(final_distance_to_goal).cpu().numpy()


if __name__ == "__main__":
    # get config
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/{sys.argv[1]}.y/aml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_csvto_only.yaml').read_text())
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_cnf_only.yaml').read_text())

    from tqdm import tqdm

    sim_env = None
    ros_copy_node = None

    if config['mode'] == 'hardware':
        from isaac_victor_envs.tasks.allegro_ros import RosAllegroScrewdriverTurningEnv

        from hardware.hardware_env import HardwareEnv
        default_dof_pos = torch.cat((torch.tensor([[0.1, 0.6, 0.6, 0.6]]).float(),
                                    torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float(),
                                    torch.tensor([[0., 0.5, 0.65, 0.65]]).float(),
                                    torch.tensor([[1.2, 0.3, 0.3, 1.2]]).float()),
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
            root_coor, root_ori = env.obj_reader.get_state()
        print('Root coor:', root_coor)
        print('Root ori:', root_ori)
        root_coor = root_coor / 1000 # convert to meters
        # robot_p = np.array([-0.025, -0.1, 1.33])
        robot_p = np.array([0, -0.095, 1.33])
        root_coor = root_coor + robot_p
        sim_env = RosAllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                 use_cartesian_controller=False,
                                 viewer=True,
                                 steps_per_action=60,
                                 friction_coefficient=2.5,
                                 device=config['sim_device'],
                                 valve=config['object_type'],
                                 video_save_path=img_save_dir,
                                 joint_stiffness=config['kp'],
                                 fingers=config['fingers'],
                                 table_pose=root_coor,
                                 gravity=False
                                 )
        
        sim, gym, viewer = sim_env.get_sim()
        assert (np.array(sim_env.robot_p) == robot_p).all()
        assert (sim_env.default_dof_pos[:, :16] == default_dof_pos.to(config['sim_device'])).all()
        # for _ in range(1):
        #     sim_env.step(default_dof_pos[:, :16])
            # state = sim_env.get_state()
            # state = state['q'].reshape(-1)[:15]


        env.world_trans = sim_env.world_trans
        env.joint_stiffness = sim_env.joint_stiffness
        env.device = sim_env.device
        env.table_pose = sim_env.table_pose
    else:
        if not config['visualize']:
            img_save_dir = None

        env = AllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                           use_cartesian_controller=False,
                                           viewer=config['visualize'],
                                           steps_per_action=60,
                                           friction_coefficient=config['friction_coefficient'] * 2.5,
                                           # friction_coefficient=1.0,  # DEBUG ONLY, set the friction very high
                                           device=config['sim_device'],
                                           video_save_path=img_save_dir,
                                           joint_stiffness=config['kp'],
                                           fingers=config['fingers'],
                                           gradual_control=False,
                                           gravity=True, # For data generation only
                                           randomize_obj_start=config.get('randomize_obj_start', False),
                                           )

        sim, gym, viewer = env.get_sim()

    state = env.get_state()
    # try:
    #     while True:
    #         start = env.get_state()['q'][:, :-1]
    #         env.step(start)
    #         print('waiting for you to finish camera adjustment, ctrl-c when done')
    #         time.sleep(0.1)
    # except KeyboardInterrupt:
    #     pass


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
    config['obj_dof'] = 3

    screwdriver_asset = f'{get_assets_dir()}/screwdriver/screwdriver.urdf'

    chain = pk.build_chain_from_urdf(open(asset).read())
    screwdriver_chain = pk.build_chain_from_urdf(open(screwdriver_asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]  # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices,
                           world_trans=env.world_trans)

    forward_kinematics = partial(chain.forward_kinematics,
                                 frame_indices=frame_indices)  # full_to= _partial_state = partial(full_to_partial_state, fingers=config['fingers'])
    partial_to_full_state = partial(partial_to_full_state, fingers=config['fingers'])
    
    inits_noise, noise_noise = [None]*config['num_trials'], [None]*config['num_trials']
    if config['use_saved_noise']:
        if config['T'] > 16:
            inits_noise, noise_noise = torch.load(f'{CCAI_PATH}/examples/saved_noise_long_horizon.pt')
        else:
            inits_noise, noise_noise = torch.load(f'{CCAI_PATH}/examples/saved_noise.pt')
            if len(inits_noise.shape) == 5:
                inits_noise = inits_noise[:, :, 0, :, :]
            if len(noise_noise.shape) == 6:
                noise_noise = noise_noise[:, :, :, 0, :, :]

    # Get datetime
    if config['mode'] == 'hardware':
        import datetime
        now = datetime.datetime.now().strftime("%m.%d.%y:%I:%M:%S")
    else:
        now = ''
    start_ind = 0# if config['experiment_name'] == 'allegro_screwdriver_csvto_diff_sine_cosine_eps_.015_2.5_damping_pi_6' else 0
    for i in tqdm(range(start_ind, config['num_trials'])):
    # for i in tqdm([1, 2, 4, 7]):
        if config['mode'] != 'hardware':
            torch.manual_seed(i)
            np.random.seed(i)

        goal = torch.tensor([0, 0, float(config['goal'])])
        # goal = goal + 0.025 * torch.randn(1) + 0.2
        for controller in config['controllers'].keys():
            env.reset()
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}.{now}/{controller}/trial_{i + 1}')
            if config['mode'] != 'hardware':
                pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
            # set up params
            params = config.copy()
            params.pop('controllers')
            params.update(config['controllers'][controller])
            if torch.cuda.device_count() == 1 and torch.cuda.current_device() == 1:
                params['device'] = 'cuda:0'
            params['controller'] = controller
            params['valve_goal'] = goal.to(device=params['device'])
            params['chain'] = chain.to(device=params['device'])
            object_location = torch.tensor([0, 0, 1.205]).to(
                params['device'])  # TODO: confirm if this is the correct location
            params['object_location'] = object_location
            # If params['device'] is cuda:1 but the computer only has 1 gpu, change to cuda:0
            final_distance_to_goal = do_trial(env, params, fpath, sim_env, ros_copy_node, inits_noise[i], noise_noise[i], seed=i)
            #
            # try:
            #     final_distance_to_goal = do_trial(env, params, fpath, sim_env, ros_copy_node)
            #     # final_distance_to_goal = turn(env, params, fpath)
            #
            #     if controller not in results.keys():
            #         results[controller] = [final_distance_to_goal]
            #     else:
            #         results[controller].append(final_distance_to_goal)
            # except Exception as e:
            #     print(e)
            #     continue
        print(results)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
