from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
from isaac_victor_envs.tasks.allegro_ros import RosAllegroScrewdriverTurningEnv

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
from pprint import pprint
import sys
import os
sys.path.append('..')

# sys.stdout = open('./logs/live_recovery_shorcut_honda_meeting_full_dof_noise_train_indexing_fix_6500_.08_std_.75_pct_diff_likelihood_no_resample_no_cpc_on_pregrasp.log', 'w', buffering=1)
# sys.stdout = open('./examples/logs/recovery_as_contact_search.log', 'w', buffering=1)
# sys.stdout = open('./examples/logs/live_recovery_hardware.log', 'w', buffering=1)

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
# from ccai.mpc.diffusion_policy import Diffusion_Policy, DummyProblem
from examples.train_allegro_screwdriver import rollout_trajectory_in_sim
from scipy.spatial.transform import Rotation as R

# from ccai.mpc.ipopt import IpoptMPC
# from ccai.problem import IpoptProblem
from ccai.models.trajectory_samplers import TrajectorySampler
from ccai.models.contact_samplers import GraphSearch, Node

from examples.model import LatentDiffusionModel

from examples.diffusion_mcts import DiffusionMCTS

from ccai.trajectory_shortcut import shortcut_trajectory

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

print("CCAI_PATH", CCAI_PATH)

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
        self.obj_mass = 0.1
        self.obj_dof_type = None
        if obj_dof == 3:
            object_link_name = 'screwdriver_body'
        elif obj_dof == 1:
            object_link_name = 'valve'
        elif obj_dof == 6:
            object_link_name = 'card'
        self.obj_link_name = object_link_name


        self.contact_points = None

        # if len(regrasp_fingers) == 3:
        #     self.contact_points = {
        #         'index': (torch.tensor([0.00563815, -0.01147131,  0.20851198], device=device), 0.01),
        #         'middle': (torch.tensor([ 0.02000177, -0.00662838,  0.16340923], device=device), 0.02),
        #         'thumb': (torch.tensor([-0.02152098,  0.00736444,  0.13183959], device=device), 0.02),
        #     }
        if len(regrasp_fingers) > 0 and self.contact_points is not None:
            contact_points_object = torch.stack([self.contact_points[finger][0] for finger in regrasp_fingers], dim=0)
        else:
            contact_points_object = None
        if proj_path is not None:
            self.proj_path = proj_path.to(device=device)
        else:
            self.proj_path = None

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
                                                   **kwargs)
        self.friction_coefficient = friction_coefficient

    def _cost(self, xu, start, goal):
        # TODO: check if the addtional term of the smoothness cost and running goal cost is necessary
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it

        smoothness_cost = torch.sum((state[1:, -self.obj_dof:] - state[:-1, -self.obj_dof:]) ** 2)
        upright_cost = 500 * torch.sum(
            (state[:, -self.obj_dof:-1]) ** 2)  # the screwdriver should only rotate in z direction
        return smoothness_cost + upright_cost + super()._cost(xu, start, goal)


def do_trial(env, params, fpath, sim_viz_env=None, ros_copy_node=None, inits_noise=None, noise_noise=None, sim=None, seed=None,
             proj_path=None, perturb_this_trial=False):
    has_recovered = False
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

    if params['controller'] == 'diffusion_policy':
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
    model_path_orig = params.get('model_path_orig', None)

    if model_path is not None:
        problem_for_sampler = None
        if params['projected'] or params.get('sample_contact', False) or params['type'] == 'cnf':
            pregrasp_problem_diff = AllegroScrewdriverDiff(
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
            index_regrasp_problem_diff = AllegroScrewdriverDiff(
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
            thumb_and_middle_regrasp_problem_diff = AllegroScrewdriverDiff(
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
            turn_problem_diff = AllegroScrewdriverDiff(
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
        def load_sampler(path, dim_mults=(1,2), T=params['T']):
            trajectory_sampler = TrajectorySampler(T=T + 1, dx=(15 + (1 if params['sine_cosine'] else 0)) if not model_t else params['nzt'], du=21 if not model_t else 0, type=params['type'],
                                                timesteps=256, hidden_dim=128 if not model_t else 64,
                                                context_dim=3, generate_context=False,
                                                constrain=params['projected'],
                                                problem=problem_for_sampler,
                                                inits_noise=inits_noise, noise_noise=noise_noise,
                                                guided=params['use_guidance'],
                                                state_control_only=params.get('state_control_only', False),
                                                vae=vae)
            d = torch.load(f'{CCAI_PATH}/{path}', map_location=torch.device(params['device']))
            # if 'recovery' in path:
            #     trajectory_sampler.model.diffusion_model.add_classifier(dim_mults)
            # else:
            trajectory_sampler.model.diffusion_model.classifier = None
            d = {k:v for k, v in d.items() if 'classifier' not in k}
            trajectory_sampler.load_state_dict(d, strict=False)
            trajectory_sampler.to(device=params['device'])
            trajectory_sampler.send_norm_constants_to_submodels()
            if params['project_state'] or params['compute_recovery_trajectory'] or params['test_recovery_trajectory']:
                trajectory_sampler.model.diffusion_model.cutoff = -75#params['likelihood_threshold']

            return trajectory_sampler
        trajectory_sampler = load_sampler(model_path, dim_mults=(1,2,4), T=params['T'] if not params['compute_recovery_trajectory'] else params['T_orig'])
        trajectory_sampler.subsampled_t = True
        trajectory_sampler.model.diffusion_model.classifier = None
        if model_path_orig is not None:
            trajectory_sampler_orig = load_sampler(model_path_orig, dim_mults=(1,2,4), T=params['T_orig'])
        else:
            trajectory_sampler_orig = trajectory_sampler

    if params['compute_recovery_trajectory']:
        start_sine_cosine = convert_yaw_to_sine_cosine(start[:4 * num_fingers + obj_dof])
        projected_samples, _, _, _, (all_losses, all_samples, all_likelihoods) = trajectory_sampler_orig.sample(16, H=trajectory_sampler_orig.T, start=start_sine_cosine.reshape(1, -1), project=True,
                constraints=torch.ones(16, 3).to(device=params['device']))
        print('Final likelihood:', all_likelihoods[-1])
        if all_likelihoods[-1].mean().item() < params.get('likelihood_threshold', -15):
            print('1 mode projection failed, trying anyway')
        else:
            print('1 mode projection succeeded')
        goal = convert_sine_cosine_to_yaw(projected_samples[-1][0])[:15]
        goal[-1] = start[-2]
            # index_regrasp_planner.reset(start, goal=goal)
            # thumb_and_middle_regrasp_planner.reset(start, goal=goal)
            # turn_planner.reset(start, goal=goal)
            # thumb_regrasp_planner.reset(start, goal=goal)
            # middle_regrasp_planner.reset(start, goal=goal)

        params['valve_goal'] = goal

        print('New goal:', goal)
    if 'csvgd' in params['controller']:
        # index finger is used for stability
        if 'index' in params['fingers']:
            fingers = params['fingers']
        else:
            fingers = ['index'] + params['fingers']


        # else:
        # min_force_dict = None
        # if params['mode'] == 'hardware':
        min_force_dict = {
            'thumb': 1,
            'middle': 1,
            'index': .0,
        }

        # min_force_dict = {
        #     'thumb': 0,
        #     'middle': 0,
        #     'index': 0,
        # }
        # initial grasp
        if params.get('compute_recovery_trajectory', False):
            goal_pregrasp = start[-4: -1]
        else:
            goal_pregrasp = params['valve_goal']
        pregrasp_params = copy.deepcopy(params)
        pregrasp_params['warmup_iters'] = 80

        start[-4:] = 0
        pregrasp_problem = AllegroScrewdriver(
            start=start[:4 * num_fingers + obj_dof],
            goal=goal_pregrasp,
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
            full_dof_goal=False,
            proj_path=proj_path,
        )
        pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, pregrasp_params)

        # thumb_regrasp_problem = AllegroScrewdriver(
        #     start=start[:4 * num_fingers + obj_dof],
        #     goal=params['valve_goal'],
        #     T=params['T'],
        #     chain=params['chain'],
        #     device=params['device'],
        #     object_asset_pos=env.table_pose,
        #     object_location=params['object_location'],
        #     object_type=params['object_type'],
        #     world_trans=env.world_trans,
        #     contact_fingers=['index', 'middle'],
        #     regrasp_fingers=['thumb'],
        #     obj_dof=obj_dof,
        #     obj_joint_dim=1,
        #     optimize_force=params['optimize_force'],
        #     default_dof_pos=env.default_dof_pos[:, :16],
        #     turn=False,
        #     obj_gravity=params.get('obj_gravity', False),
        #     min_force_dict=min_force_dict,
        #     full_dof_goal=params.get('compute_recovery_trajectory', False),
        #     proj_path=proj_path,
        # )

        # middle_regrasp_problem = AllegroScrewdriver(
        #     start=start[:4 * num_fingers + obj_dof],
        #     goal=params['valve_goal'],
        #     T=params['T'],
        #     chain=params['chain'],
        #     device=params['device'],
        #     object_asset_pos=env.table_pose,
        #     object_location=params['object_location'],
        #     object_type=params['object_type'],
        #     world_trans=env.world_trans,
        #     contact_fingers=['middle'],
        #     regrasp_fingers=['index', 'thumb'],
        #     obj_dof=obj_dof,
        #     obj_joint_dim=1,
        #     optimize_force=params['optimize_force'],
        #     default_dof_pos=env.default_dof_pos[:, :16],
        #     turn=False,
        #     obj_gravity=params.get('obj_gravity', False),
        #     min_force_dict=min_force_dict,
        #     full_dof_goal=params.get('compute_recovery_trajectory', False),
        #     proj_path=proj_path,
        # )

        # thumb_regrasp_planner = PositionControlConstrainedSVGDMPC(thumb_regrasp_problem, params)
        # middle_regrasp_planner = PositionControlConstrainedSVGDMPC(middle_regrasp_problem, params)
        turn_problem = AllegroScrewdriver(
            start=start[:4 * num_fingers + obj_dof],
            goal=params['valve_goal'],
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.table_pose,
            object_location=params['object_location'],
            object_type=params['object_type'],
            world_trans=env.world_trans,
            contact_fingers=['index', 'middle', 'thumb'],
            obj_dof=3,
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

        index_regrasp_planner = None
        thumb_and_middle_regrasp_planner = None
        turn_planner = None
        if not params.get('live_recovery', False):
            # finger gate index
            index_regrasp_problem = AllegroScrewdriver(
                start=start[:4 * num_fingers + obj_dof],
                goal=params['valve_goal'],
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
                full_dof_goal=params.get('compute_recovery_trajectory', False) or params.get('test_recovery_trajectory', False) or params.get('live_recovery', False) and (len(goal) > 3),
                proj_path=None,
                project=params.get('compute_recovery_trajectory', False) or params.get('test_recovery_trajectory', False) or params.get('live_recovery', False),
            )
            index_regrasp_planner = PositionControlConstrainedSVGDMPC(index_regrasp_problem, params)

            thumb_and_middle_regrasp_problem = AllegroScrewdriver(
                start=start[:4 * num_fingers + obj_dof],
                goal=params['valve_goal'],
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
                full_dof_goal=params.get('compute_recovery_trajectory', False) or params.get('test_recovery_trajectory', False) or params.get('live_recovery', False) and (len(goal) > 3),
                proj_path=None,
                project=params.get('compute_recovery_trajectory', False) or params.get('test_recovery_trajectory', False) or params.get('live_recovery', False),
            )
            thumb_and_middle_regrasp_planner = PositionControlConstrainedSVGDMPC(thumb_and_middle_regrasp_problem, params)
            tp = AllegroScrewdriver(
                start=start[:4 * num_fingers + obj_dof],
                goal=params['valve_goal'],
                T=params['T'],
                chain=params['chain'],
                device=params['device'],
                object_asset_pos=env.table_pose,
                object_location=params['object_location'],
                object_type=params['object_type'],
                world_trans=env.world_trans,
                contact_fingers=['index', 'middle', 'thumb'],
                obj_dof=3,
                obj_joint_dim=1,
                optimize_force=params['optimize_force'],
                default_dof_pos=env.default_dof_pos[:, :16],
                turn=True,
                obj_gravity=params.get('obj_gravity', False),
                min_force_dict=min_force_dict,
                full_dof_goal=params.get('compute_recovery_trajectory', False),
                proj_path=proj_path,
                project=params.get('compute_recovery_trajectory', False) or params.get('test_recovery_trajectory', False) or params.get('live_recovery', False),
            )
            turn_planner_recovery = PositionControlConstrainedSVGDMPC(tp, params)
    if model_path is not None:
        if not params.get('live_recovery', False):
            index_regrasp_problem.model = trajectory_sampler_orig
            thumb_and_middle_regrasp_problem.model = trajectory_sampler_orig
        print('Loaded trajectory sampler')
        trajectory_sampler_orig.model.diffusion_model.classifier = None
        # trajectory_sampler_orig.model.diffusion_model.cutoff = params['likelihood_threshold']

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
        if mode == 'pregrasp':
            traj = traj[..., :-9]
        if mode == 'thumb':
            traj = traj[..., :-3]
        if mode == 'middle':
            traj = torch.cat((traj[..., :-6], traj[..., -3:]), dim=-1)
        return traj
    
    def check_id(state, threshold=params.get('likelihood_threshold', -15)):
        start = state[:4 * num_fingers + obj_dof]
        start_sine_cosine = convert_yaw_to_sine_cosine(start)
        samples, _, likelihood = trajectory_sampler_orig.sample(N=params['N'], H=trajectory_sampler_orig.T, start=start_sine_cosine.reshape(1, -1),
                constraints=torch.ones(params['N'], 3).to(device=params['device']))
        likelihood = likelihood.reshape(params['N']).mean().item()
        # samples = samples.cpu().numpy()
        print('Likelihood:', likelihood)
        data['final_likelihoods'].append(likelihood)
        if likelihood < threshold:
            print('State is out of distribution')
            return False
        return True

    def execute_traj(planner, mode, goal=None, fname=None, initial_samples=None, recover=False):
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
        elif mode == 'thumb':
            contact[:, 0] = 1
            contact[:, 1] = 1
        elif mode == 'middle':
            contact[:, 0] = 1
            contact[:, 2] = 1

        if mode == 'index' and planner is None:
            # finger gate index
            index_regrasp_problem = AllegroScrewdriver(
                start=state[:4 * num_fingers + obj_dof],
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
                full_dof_goal=recover,
                proj_path=None,
                project=recover,
            )
            planner = PositionControlConstrainedSVGDMPC(index_regrasp_problem, params)

        elif mode == 'thumb_middle' and planner is None:
            thumb_and_middle_regrasp_problem = AllegroScrewdriver(
                start=state[:4 * num_fingers + obj_dof],
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
                full_dof_goal=recover,
                proj_path=None,
                project=recover,
            )
            planner = PositionControlConstrainedSVGDMPC(thumb_and_middle_regrasp_problem, params)
        elif mode == 'turn' and planner is None:
            tp = AllegroScrewdriver(
                start=state[:4 * num_fingers + obj_dof],
                goal=goal,
                T=params['T'],
                chain=params['chain'],
                device=params['device'],
                object_asset_pos=env.table_pose,
                object_location=params['object_location'],
                object_type=params['object_type'],
                world_trans=env.world_trans,
                contact_fingers=['index', 'middle', 'thumb'],
                obj_dof=3,
                obj_joint_dim=1,
                optimize_force=params['optimize_force'],
                default_dof_pos=env.default_dof_pos[:, :16],
                turn=True,
                obj_gravity=params.get('obj_gravity', False),
                min_force_dict=min_force_dict,
                full_dof_goal=recover,
                proj_path=proj_path,
                project=recover,
            )
            planner = PositionControlConstrainedSVGDMPC(tp, params)
        # contact = -torch.ones(params['N'], 3).to(device=params['device'])
        # contact[:, 0] = 1
        # contact[:, 1] = 1

        # generate initial samples with diffusion model
        sim_rollouts = None
        initial_samples_0 = None
        new_T = params['T']
        if trajectory_sampler is not None and params.get('diff_init', True) and initial_samples is None:

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
            if params.get('shortcut_trajectory', False):
                s = time.perf_counter()
                # for traj_idx in range(initial_samples.shape[0]):
                initial_samples = shortcut_trajectory(initial_samples, 4 * num_fingers, obj_dof, epsilon=.04)
                print(f'Shortcut time', time.perf_counter() - s)

                new_T = initial_samples.shape[1] - 1
            # if state[-1] < -1.0:
            #     initial_samples[:, :, -1] -= 0.75
            if (params['visualize_plan'] and not recover) or (params['visualize_recovery_plan'] and recover):
                for (name, traj_set) in [('initial_samples_project', initial_samples), ('initial_samples_0', initial_samples_0)]:
                    if traj_set is None:
                        continue
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
                        visualize_trajectory(traj_for_viz, turn_problem.contact_scenes_for_viz, viz_fpath,
                                            turn_problem.fingers, turn_problem.obj_dof + 1)
            sim_rollouts = torch.zeros_like(initial_samples)
            torch.cuda.empty_cache()

            # for i in range(params['N']):
            #     sim_rollout = rollout_trajectory_in_sim(env_sim_rollout, initial_samples[i])
            #     sim_rollouts[i] = sim_rollout
        if initial_samples is not None and params.get('diff_init', True):
            # if params['mode'] == 'hardware' and mode == 'turn':
            #     initial_samples[..., 30:] = 1.5 * torch.randn(params['N'], initial_samples.shape[1], 6, device=initial_samples.device)
            # elif params['mode'] == 'hardware':
            #     initial_samples[..., 27:] = .15 * torch.randn(params['N'], initial_samples.shape[1], initial_samples.shape[-1]-27, device=initial_samples.device)

            initial_samples = _full_to_partial(initial_samples, mode)
            initial_x = initial_samples[:, 1:, :planner.problem.dx]
            initial_u = initial_samples[:, :-1, -planner.problem.du:]
            initial_samples = torch.cat((initial_x, initial_u), dim=-1)


        state = env.get_state()
        state = state['q'].reshape(-1).to(device=params['device'])
        state = state[:planner.problem.dx]
        # print(params['T'], state.shape, initial_samples)

        if params.get('compute_recovery_trajectory', False):
            planner.reset(state, T=new_T, initial_x=initial_samples, proj_path=proj_path)
            planner.warmup_iters = 0
        else:
            planner.reset(state, T=new_T, goal=goal, initial_x=initial_samples, proj_path=proj_path)
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
            print(state)

            if params['live_recovery']:
                id_check = check_id(state) if not recover else True
            elif params.get('compute_recovery_trajectory', False) or params.get('test_recovery_trajectory', False):
                id_check = check_id(state, threshold=-300)
            else:
                id_check = True

            if not id_check:
                # State is OOD. Return and move to recovery pipeline
                planner.problem.data = {}
                return None, None, None, None, None, None, None, True
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
            print(f'Solve time for step {k+1}', time.perf_counter() - s)

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
            if params.get('perturb_action', False):# and mode == 'turn':
                # rand_pct = .75 if not params.get('shortcut_trajectory', False) else .25
                rand_pct = .75
                if np.random.rand() < rand_pct:
                    r = np.random.rand()
                    std = .08 if perturb_this_trial else .0
                    # if mode != 'turn':
                    #     std /= 4
                    # if r > .66:
                    #     action[:, :4 * num_fingers_to_plan] += std * torch.randn_like(action[:, :4 * num_fingers_to_plan])
                    # elif r > .4 and r < .7:
                    #     action[:, 4 * 1: 4*3] += std * torch.randn_like(action[:, 4 * 1: 4*3])
                    # else:
                    if mode == 'turn':
                        # action[:, :4 * 1] += std * torch.randn_like(action[:, :4 * 1])
                        # action[:, 4 * 1: 4*3] += std * torch.randn_like(action[:, 4 * 1: 4*3])
                        action[:, :4 * num_fingers_to_plan] += std * torch.randn_like(action[:, :4 * num_fingers_to_plan])

            xu = torch.cat((state[:-1].cpu(), action[0].cpu()))
            actual_trajectory.append(xu)

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
            if (params['visualize_plan'] and not recover) or (params['visualize_recovery_plan'] and recover):
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
                visualize_trajectory(traj_for_viz, turn_problem.contact_scenes_for_viz, viz_fpath,
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
        return actual_trajectory, planned_trajectories, initial_samples, sim_rollouts, optimizer_paths, contact_points, contact_distance, recover

    data = {}
    for t in range(1, 1 + params['T']):
        data[t] = {'plans': [], 'starts': [], 'inits': [], 'init_sim_rollouts': [], 'optimizer_paths': [], 'contact_points': [], 'contact_distance': [], 'contact_state': []}
    data['final_likelihoods'] = []
    data['all_samples_'] = []
    data['all_likelihoods_'] = []
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
        
        contact_sequence_sampler = GraphSearch(state_for_search, trajectory_sampler_orig, params['T'], problem_for_sampler, 
                                               depth, params['heuristic'], params['goal'], 
                                               torch.device(params['device']), initial_run=initial_run,
                                               multi_particle=multi_particle,
                                               prior=params['prior'],
                                               sine_cosine=params['sine_cosine'],)
        a = time.perf_counter()
        contact_node_sequence = contact_sequence_sampler.astar(next_node, None)
        planning_time = time.perf_counter() - a
        print('Contact sequence search time:', planning_time)
        closed_set = None#contact_sequence_sampler.closed_set
        if contact_node_sequence is not None:
            contact_node_sequence = list(contact_node_sequence)
        pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)

        with open(f"{fpath}/contact_planning_{all_stage}.pkl", "wb") as f:
            pkl.dump((contact_node_sequence, closed_set, planning_time, contact_sequence_sampler.iter), f)
        if contact_node_sequence is None and closed_set is not None:
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
        elif contact_node_sequence is None and closed_set is None:
            print('No contact sequence found')
            return None, None, None
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

        goal_config = contact_sequence_sampler.get_goal_config(last_node)
        goal_config = goal_config.to(device=params['device'])

        torch.cuda.empty_cache()
        return contact_sequence, next_node, initial_samples, goal_config

    mode_planner_dict = {
        'index': index_regrasp_planner,
        'thumb_middle': thumb_and_middle_regrasp_planner,
        'turn': turn_planner,
        # 'thumb': thumb_regrasp_planner,
        # 'middle': middle_regrasp_planner,
    }

    def plan_recovery_contacts(state, stage):
        distances = []
        # modes = ['index', 'thumb_middle', 'middle', 'thumb']
        modes = ['thumb_middle', 'index', 'turn']
        # for planner in [index_regrasp_planner, thumb_and_middle_regrasp_planner]:
        goal = index_regrasp_planner.problem.goal.clone()
        goal[-1] = state[-1]
        initial_samples = []
        if params['visualize_contact_plan']:
            # Visualize the goal
            viz_fpath = pathlib.PurePath.joinpath(fpath, f"{fpath}/recovery_stage_{all_stage}/goal")
            pathlib.Path.mkdir(viz_fpath, parents=True, exist_ok=True)
            img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
            gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
            pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
            pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
            
            tmp = torch.zeros((2, 1), device=goal.device)  # add the joint for the screwdriver cap

            traj_for_viz = torch.cat((state.unsqueeze(0), goal.unsqueeze(0)), dim=0)
            traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)
            visualize_trajectory(traj_for_viz
            , turn_problem.contact_scenes_for_viz, viz_fpath,
                                turn_problem.fingers, turn_problem.obj_dof + 1)
        
        dist_min = 3e-3
        mode_skip = []
        planner = mode_planner_dict['index']
        cur_q = state[:4 * num_fingers]
        cur_theta = state[4 * num_fingers: 4 * num_fingers + obj_dof]
        planner.problem._preprocess_fingers(cur_q[None, None], cur_theta[None, None], compute_closest_obj_point=True)
        print(planner.problem.data['thumb']['sdf'], planner.problem.data['middle']['sdf'], planner.problem.data['index']['sdf'])
        for mode in modes:
            if mode == 'index':
                if planner.problem.data['thumb']['sdf'].max() > dist_min or planner.problem.data['middle']['sdf'].max() > dist_min:
                    mode_skip.append(mode)
            elif mode == 'thumb_middle':
                if planner.problem.data['index']['sdf'].max() > dist_min:
                    mode_skip.append(mode)

        if 'index' in mode_skip and 'thumb_middle' in mode_skip:
            mode_skip = []

        for mode in modes:
            if mode in mode_skip:
                distances.append(float('inf'))
                initial_samples.append(None)
                continue

            planner = mode_planner_dict[mode]
            planner.reset(state, T=params['T'], goal=goal)
            # old_warmup_iters = planner.warmup_iters
            planner.warmup_iters = params['warmup_iters']

            xu, plans = planner.step(state)
            planner.problem.data = {}

            # planner.warmup_iters = old_warmup_iters
            initial_samples.append(plans)
            # x_last = xu[-1, :planner.problem.num_fingers * 4 + planner.problem.obj_dof-1]
            # goal_cost = (x_last - planner.problem.goal[:-1]).pow(2).sum(dim=-1)#.sum(dim=-1)
            x = xu[:, :planner.problem.num_fingers * 4 + planner.problem.obj_dof]

            start = x[-1]
            start_sine_cosine = convert_yaw_to_sine_cosine(start)
            samples, _, likelihood = trajectory_sampler_orig.sample(N=params['N'], H=trajectory_sampler_orig.T, start=start_sine_cosine.reshape(1, -1),
                    constraints=torch.ones(params['N'], 3).to(device=params['device']))
            likelihood = likelihood.mean().item()
            distances.append(-likelihood)
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
                tmp = torch.zeros((traj_for_viz.shape[0], 1),
                                  device=x.device)  # add the joint for the screwdriver cap
                traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)
                # traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof] = axis_angle_to_euler(traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof])


                img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
                gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
                pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
                pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
                visualize_trajectory(traj_for_viz, turn_problem.contact_scenes_for_viz, viz_fpath,
                                     turn_problem.fingers, turn_problem.obj_dof + 1)

        dists_dict = dict(zip(modes, distances))
        pprint(dists_dict)

        if all(d == float('inf') for d in distances):
            return ['turn'], None
        else:
            return [modes[np.argmin(distances)]], initial_samples[np.argmin(distances)]
    
    def calc_samples_likeilhoods(mode, start_for_diff, state):
        N_ = params['N']# * 3
        contact = -torch.ones(N_, 3).to(device=params['device'])
        if mode == 'thumb_middle':
            contact[:, 0] = 1
        elif mode == 'index':
            contact[:, 1] = 1
            contact[:, 2] = 1
        elif mode =='turn':
            contact[:, 0] = 1
            contact[:, 1] = 1
            contact[:, 2] = 1

        samples, _, _likelihoods = trajectory_sampler.sample(N=N_, start=start_for_diff.reshape(1, -1),
                                                            H=trajectory_sampler.T,
                                                            constraints=contact,
                                                            project=params['project_state'],)
        samples = samples.detach()
        _likelihoods = _likelihoods.detach()
        samples = convert_sine_cosine_to_yaw(samples)
        # Weighted average of samples based on likelihoods
        # weights = torch.softmax(_likelihoods.flatten(), dim=0)
        # top_likelihoods = torch.multinomial(weights, N_, replacement=True)

        top_likelihoods = torch.arange(N_)
        highest_likelihood_idx = _likelihoods.flatten().argmax(0)

        resampled_samples = samples[top_likelihoods]
        resampled_likelihoods = _likelihoods.flatten()[top_likelihoods]

        if params['visualize_contact_plan']:
            traj_for_viz = resampled_samples[highest_likelihood_idx, :, :15]
            if params['exclude_index']:
                traj_for_viz = torch.cat((state[4:15].unsqueeze(0), traj_for_viz), dim=0)
            else:
                traj_for_viz = torch.cat((state[:15].unsqueeze(0), traj_for_viz), dim=0)
            tmp = torch.zeros((traj_for_viz.shape[0], 1),
                                device=x.device)  # add the joint for the screwdriver cap
            traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)
            # traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof] = axis_angle_to_euler(traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof])

            viz_fpath = pathlib.PurePath.joinpath(fpath, f"{fpath}/recovery_stage_{all_stage}/{mode}")

            img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
            gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
            pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
            pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
            visualize_trajectory(traj_for_viz, turn_problem.contact_scenes_for_viz, viz_fpath,
                                    turn_problem.fingers, turn_problem.obj_dof + 1)

        # resampled_weights = torch.softmax(resampled_likelihoods, dim=0)

        # Choose weighted average terminal config
        # mean_sample = torch.sum(resampled_samples * resampled_weights.reshape(-1, 1, 1), dim=0)
        # mean_obj_config = mean_sample[-1, :4 * num_fingers + obj_dof].detach()

        # Choose terminal config from highest likelihood particle
        mean_obj_config = resampled_samples[resampled_likelihoods.argmax(0), -1, :4 * num_fingers + obj_dof].squeeze()
        
        mean = torch.mean(resampled_likelihoods)

        return mean.item(), mean_obj_config, resampled_samples, resampled_likelihoods

        # start = mean_obj_config
        # start_sine_cosine = convert_yaw_to_sine_cosine(start)
        # samples, _, likelihood = trajectory_sampler_orig.sample(N=N, H=params['T']+1, start=start_sine_cosine.reshape(1, -1))
        # likelihood = likelihood.mean().item()

        # return likelihood, mean_obj_config, resampled_samples, resampled_likelihoods

    def plan_recovery_contacts_w_model(state):
        N_ = params['N'] * 3
        modes = ['thumb_middle', 'index']#, 'turn'] TODO: Add turn back in when switching to new model
        # modes = ['thumb_middle']
        if params['sine_cosine']:
            start_for_diff = convert_yaw_to_sine_cosine(state)
        else:
            start_for_diff = start

        likelihoods= []
        mean_obj_configs = []
        all_samples_ = {}
        all_likelihoods_ = {}

        max_likelihood = 0
        iter = 0
        # while max_likelihood < .5 and iter < 3:
        likelihood_list = []
        for mode in modes:
            if params['visualize_contact_plan']:
                # Visualize the goal
                viz_fpath = pathlib.PurePath.joinpath(fpath, f"{fpath}/recovery_stage_{all_stage}/goal")
                pathlib.Path.mkdir(viz_fpath, parents=True, exist_ok=True)
                img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
                gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
                pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
                pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
            likelihood, mean_obj_config, all_samples, all_likelihoods = calc_samples_likeilhoods(mode, start_for_diff, state)
            likelihoods.append(likelihood)
            mean_obj_configs.append(mean_obj_config)
            all_samples_[mode] = all_samples.detach().cpu()
            all_likelihoods_[mode] = all_likelihoods.detach().cpu()

            print(f'Likelihood for {mode}:', likelihood)
            likelihood_list.append(likelihood)
        iter += 1
        # max_likelihood = max(likelihood_list)

        return [modes[np.argmax(likelihoods) % len(modes)]], mean_obj_configs[np.argmax(likelihoods)], all_samples_, all_likelihoods_

        # Repeat for thumb and middle
    def get_next_node(contact_sequence):
        if contact_sequence[0] == 'turn':
            next_node = (1, 1, 1)
        elif contact_sequence[0] == 'index':
            next_node = (-1, 1, 1)
        elif contact_sequence[0] == 'thumb_middle':
            next_node = (1, -1, -1)
        elif contact_sequence[0] == 'thumb':
            next_node = (1, 1, -1)
        elif contact_sequence[0] == 'middle':
            next_node = (1, -1, 1)
        else:
            next_node = (-1, -1, -1)
        return next_node
    
    state = env.get_state()
    state = state['q'].reshape(-1)[:15].to(device=params['device'])


    contact_label_to_vec = {'pregrasp': 0,
                            'thumb_middle': 1,
                            'index': 2,
                            'turn': 3,
                            'thumb': 4,
                            'middle': 5
                            }
    contact_vec_to_label = dict((v, k) for k, v in contact_label_to_vec.items())


    sample_contact = params.get('sample_contact', False)
    num_stages = 2 + 3 * (params['num_turns'] - 1)
    if params.get('compute_recovery_trajectory', False) or params.get('test_recovery_trajectory', False):
        num_stages = params['max_recovery_stages']
        # contact_sequence = plan_recovery_contacts(state)
        # contact_sequence = ['thumb_middle']
    elif params.get('live_recovery', False):
        contact_sequence = ['turn'] * 100
        num_stages = params['num_stages']
    elif not sample_contact and perturb_this_trial:
        # contact_sequence = ['turn', 'thumb_middle'] * 100
        gen = np.random.default_rng(seed)
        contact_sequence = ['turn']
        num_stages = params['num_stages']
        # contact_sequence = ['turn']
        for k in range(params['num_turns'] - 1):
            contact_options = ['index', 'thumb_middle', 'turn']
            for l in range(3):
                contact_sequence.append(gen.choice(contact_options, 1)[0])
                if contact_sequence[-1] == 'turn':
                    break
    elif not sample_contact:
        contact_sequence = ['turn']
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
    recover = False
    pre_recover = False
    stage = 0
    all_stage = 0
    done = False
    # for stage in range(num_stages):
    # <= so because pregrasp will iterate the all_stage counter
    while all_stage <= num_stages:
        sample_contact = params['sample_contact'] and not recover
        initial_samples = None
        state = env.get_state()
        state = state['q'].reshape(-1)[:15].to(device=params['device'])
        if params.get('compute_recovery_trajectory', False) and not sample_contact:
            contact_sequence, initial_samples = plan_recovery_contacts(state, stage)
        elif params.get('test_recovery_trajectory', False) or (params.get('live_recovery', False) and recover):
            contact_sequence, goal_config, all_samples_, all_likelihoods_ = plan_recovery_contacts_w_model(state)
            goal_config[-1] = state[-1]
            data['all_samples_'].append(all_samples_)
            data['all_likelihoods_'].append(all_likelihoods_)

            # Use the resampled particles to initialize CSVTO. Maybe turn this off. Could be collapsing diversity
            initial_samples = all_samples_[contact_sequence[0]][:params['N']]
        elif params.get('live_recovery', False) and not recover:
            # if contact == 'turn':
            #     contact_sequence = ['thumb_middle']
            # else:
            contact_sequence = ['turn']
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
        if params['controller'] == 'diffusion_policy' and stage > 0:
            _goal = torch.tensor([0, 0, state[-1]]).to(device=params['device'])
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance = execute_traj(
                planner, mode='diffusion_policy', goal=_goal, fname=f'diffusion_policy_{all_stage}')

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
            stage += 1
            all_stage += 1
            continue
        if stage == 0 and not params.get('compute_recovery_trajectory', False) and not params.get('test_recovery_trajectory', False):

            orig_torque_perturb = env.external_wrench_perturb if params['mode'] != 'hardware' else False
            if params['mode'] != 'hardware':
                env.set_external_wrench_perturb(False)
            contact = 'pregrasp'
            start = env.get_state()['q'].reshape(4 * num_fingers + 4).to(device=params['device'])
            best_traj, _ = pregrasp_planner.step(start[:pregrasp_planner.problem.dx])
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
                    print(state[:15][-3:])
            if params['mode'] == 'hardware':
                input("Pregrasp complete. Ready to execute. Press <ENTER> to continue.")
            stage += 1
            all_stage += 1
            if params['mode'] != 'hardware':
                env.set_external_wrench_perturb(orig_torque_perturb)
            continue
        elif sample_contact and (stage == 1 or (params['replan'] and (stages_since_plan == 0 or len(contact_sequence) == 1))):
            # if yaw <= params['goal']:
            #     # params['goal'] -= .5
            #     params['goal'] = yaw + float(params['goal_update'])
            #     print('Adjusting goal to', params['goal'])
            # new_contact_sequence, new_next_node, initial_samples = plan_contacts(state, num_stages - stage, next_node, params['multi_particle_search'])
            plan_func = plan_contacts
            if params['replan']:
                params['goal'] = yaw + float(params['goal_update'])
            print('Adjusting goal to', params['goal'])
            for key in problem_for_sampler:
                problem_for_sampler[key].goal = torch.tensor([0, 0, params['goal']]).to(device=params['device'])
            new_contact_sequence, new_next_node, initial_samples, goal_config = plan_func(state, stage, 7, next_node, params['multi_particle_search'])
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
                next_node = get_next_node(contact_sequence)
            print(contact_sequence)
            # return -1
            contact = contact_sequence[0]  
        elif params.get('compute_recovery_trajectory', False) or params.get('test_recovery_trajectory', False) or (recover and params.get('live_recovery')):
            contact = contact_sequence[0]
            next_node = get_next_node(contact_sequence)
        elif params.get('live_recovery') and not recover:
            contact = contact_sequence[0]
            next_node = get_next_node(contact_sequence)
        elif stage > len(contact_sequence):
            print('Planner thinks task is complete')
            print(executed_contacts)
            break
        else:
            contact = contact_sequence[stage-1]
        executed_contacts.append(contact)
        print(stage, contact)
        torch.cuda.empty_cache()

        contact_state_dict = {
            'pregrasp': torch.tensor([0.0, 0.0, 0.0]),
            'index': torch.tensor([0.0, 1.0, 1.0]),
            'thumb_middle': torch.tensor([1.0, 0.0, 0.0]),
            'turn': torch.tensor([1.0, 1.0, 1.0]),
            'thumb': torch.tensor([1.0, 1.0, 0.0]),
            'middle': torch.tensor([1.0, 0.0, 1.0]),
        }

        pre_recover = recover
        if contact == 'index':
            _goal = torch.tensor([0, 0, state[-1]]).to(device=params['device'])
            if params.get('test_recovery_trajectory', False) or params.get('live_recovery', False):
                _goal = goal_config
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, recover = execute_traj(
                index_regrasp_planner, mode='index', goal=_goal, fname=f'index_regrasp_{all_stage}', initial_samples=initial_samples, recover=recover)

            if not recover:
                plans = [torch.cat((plan[..., :-6],
                                    torch.zeros(*plan.shape[:-1], 3).to(device=params['device']),
                                    plan[..., -6:]),
                                dim=-1) for plan in plans]
                traj = torch.cat((traj[..., :-6], torch.zeros(*traj.shape[:-1], 3).to(device=params['device']),
                                traj[..., -6:]), dim=-1)
        elif contact == 'thumb_middle':
            _goal = torch.tensor([0, 0, state[-1]]).to(device=params['device'])
            if params.get('test_recovery_trajectory', False) or params.get('live_recovery', False):
                try:
                    _goal = goal_config
                except:
                    pass
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, recover = execute_traj(
                thumb_and_middle_regrasp_planner, mode='thumb_middle',
                goal=_goal, fname=f'thumb_middle_regrasp_{all_stage}', initial_samples=initial_samples, recover=recover)
            if not recover:
                plans = [torch.cat((plan,
                                    torch.zeros(*plan.shape[:-1], 6).to(device=params['device'])),
                                dim=-1) for plan in plans]
                traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 6).to(device=params['device'])), dim=-1)

        elif contact == 'turn':
            _goal = torch.tensor([0, 0, state[-1] - np.pi / 6]).to(device=params['device'])
            if params.get('live_recovery', False) and recover:
                _goal = goal_config
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, recover = execute_traj(
                turn_planner, mode='turn', goal=_goal, fname=f'turn_{all_stage}', initial_samples=initial_samples, recover=recover)
            
        elif contact == 'thumb':
            _goal = torch.tensor([0, 0, state[-1]]).to(device=params['device'])
            if params.get('test_recovery_trajectory', False):
                _goal = goal_config
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, recover = execute_traj(
                thumb_regrasp_planner, mode='thumb', goal=_goal, fname=f'thumb_regrasp_{all_stage}', initial_samples=initial_samples, recover=recover)
            if not recover:
                plans = [torch.cat((plan,
                                    torch.zeros(*plan.shape[:-1], 3).to(device=params['device'])),
                                dim=-1) for plan in plans]
                traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 3).to(device=params['device'])), dim=-1)
        if contact == 'middle':
            _goal = torch.tensor([0, 0, state[-1]]).to(device=params['device'])
            if params.get('test_recovery_trajectory', False):
                _goal = goal_config
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, recover = execute_traj(
                middle_regrasp_planner, mode='middle', goal=_goal, fname=f'middle_regrasp_{all_stage}', initial_samples=initial_samples, recover=recover)

            if not recover:
                plans = [torch.cat((plan[..., :-3],
                                    torch.zeros(*plan.shape[:-1], 3).to(device=params['device']),
                                    plan[..., -3:]),
                                dim=-1) for plan in plans]
                traj = torch.cat((traj[..., :-3], torch.zeros(*traj.shape[:-1], 3).to(device=params['device']),
                                traj[..., -3:]), dim=-1)
            

        all_stage += 1
        # done = False
        add = not recover
        if params['compute_recovery_trajectory'] or params['test_recovery_trajectory'] or (recover and pre_recover):
            # Check if the current state is in distribution
            state = env.get_state()
            state = state['q'].reshape(-1)[:15].to(device=params['device'])
            
            start = state[:4 * num_fingers + obj_dof]
            start_sine_cosine = convert_yaw_to_sine_cosine(start)
            samples, _, likelihood = trajectory_sampler_orig.sample(N=params['N'], H=trajectory_sampler_orig.T, start=start_sine_cosine.reshape(1, -1),
                    constraints=torch.ones(params['N'], 3).to(device=params['device']))
            likelihood = likelihood.reshape(params['N']).mean().item()
            # samples = samples.cpu().numpy()
            print('Likelihood:', likelihood)
            data['final_likelihoods'].append(likelihood)
            if likelihood > params.get('likelihood_threshold', -15):
                print('State is in distribution')
                done = True
                recover = False
                perturb_this_trial = False
            elif likelihood < -300:
                print('Probably dropped the object')
                done = True
                add = False
            else:
                print('State is out of distribution')
                # Project the state back into distribution if we are computing recovery trajectories
                if params['compute_recovery_trajectory']:
                    projected_samples, _, _, _, (all_losses, all_samples, all_likelihoods) = trajectory_sampler_orig.sample(16, H=trajectory_sampler_orig.T, start=start_sine_cosine.reshape(1, -1), project=True,
                            constraints=torch.ones(16, 3).to(device=params['device']))
                    print('Final likelihood:', all_likelihoods[-1])
                    if all_likelihoods[-1].mean().item() < params.get('likelihood_threshold', -15):
                        print('1 mode projection failed, trying anyway')
                    else:
                        print('1 mode projection succeeded')
                    goal = convert_sine_cosine_to_yaw(projected_samples[-1][0])[:15]
                    index_regrasp_planner.reset(start, goal=goal)
                    thumb_and_middle_regrasp_planner.reset(start, goal=goal)
                    turn_planner.reset(start, goal=goal)
                    # thumb_regrasp_planner.reset(start, goal=goal)
                    # middle_regrasp_planner.reset(start, goal=goal)
                    print('New goal:', goal)
        if add:
            _add_to_dataset(traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance,
                            contact_state=contact_state_dict[contact])
        if not (params.get('live_recovery', False) and (recover or pre_recover)):
            stage += 1
            if params.get('live_recovery', False):
                contact_sequence = ['turn']
        if contact != 'pregrasp' and add:
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
        if add:
            with open(f'{fpath.resolve()}/trajectory.pkl', 'wb') as f:
                pickle.dump([i.cpu().numpy() for i in actual_trajectory_save], f)
        del actual_trajectory_save

        if done and params.get('compute_recovery_trajectory', False):
            break

    # np.savez(f'{fpath.resolve()}/trajectory.npz', x=[i.cpu().numpy() for i in actual_trajectory],)
             #  constr=constraint_val.cpu().numpy(),
            #  d2goal=final_distance_to_goal.cpu().numpy())
    if params['mode'] == 'hardware':
        input("Ready to regrasp. Press <ENTER> to continue.")

    env.reset()
    return -1#torch.min(final_distance_to_goal).cpu().numpy()


if __name__ == "__main__":
    # get config
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/{sys.argv[1]}.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_csvto_only.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_csvto_OOD_ID_perturbed_data_gen.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_csvto_OOD_ID_recovery_as_contact_mode_planning.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_csvto_OOD_ID_live_recovery_shortcut.yaml').read_text())
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_csvto_OOD_ID_live_recovery_shortcut_hardware.yaml').read_text())

    from tqdm import tqdm

    sim_env = None
    ros_copy_node = None

    if config['mode'] == 'hardware':
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
            root_coor, root_ori = env.obj_reader.get_state_world_frame_pos()
        print('Root coor:', root_coor)
        print('Root ori:', root_ori)
        root_coor = root_coor # convert to meters
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
                                 table_pose=None, # Since I ran the IK before the sim, I shouldn't need to set the table pose. 
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
                                           randomize_rob_start=config.get('randomize_rob_start', False),
                                           external_wrench_perturb=config.get('external_wrench_perturb', False),
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
    

    for key in ['compute_recovery_trajectory', 'test_recovery_trajectory', 'viz_states']:
        if key not in config:
            config[key] = False

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
        now_ = '.' + now
        now = now_
    else:
        now = ''
    
    if config['compute_recovery_trajectory'] or config['test_recovery_trajectory']:
        step_size = 2
        start_ind = config['parity'] + 552
    else:
        start_ind = 0
        step_size = 1
    
    if config['compute_recovery_trajectory'] or config['test_recovery_trajectory']:
        N = 16
        model_dir = config.get('model_dir', None)
        proj_data = pickle.load(open(f'{CCAI_PATH}/{model_dir}/ood_projection/proj_data.pkl', 'rb'))
    elif config['viz_states']:
        model_dir = config.get('model_dir', None)
        proj_data = torch.tensor(np.load(f'{CCAI_PATH}/{model_dir}/ood_states.npy'))

        candidate_ood_likelihoods = np.load(f'{CCAI_PATH}/{model_dir}/candidate_ood_likelihoods.npy')
        train_likelihoods = np.load(f'{CCAI_PATH}/{model_dir}/gen_train_likelihoods_all_states.pkl', allow_pickle=True)
        quantile = np.quantile(train_likelihoods, .4)
        # ood_states = ood_states[(ood_likelihoods < quantile) & (ood_likelihoods > -500)]
        # ood_states = ood_states[(ood_likelihoods < quantile) & (ood_likelihoods > -75) & (ood_likelihoods < -100)]
        # ood_states = ood_states[:, :16]

    num_trials = (config['num_trials']+start_ind) if not (config['compute_recovery_trajectory'] or config['test_recovery_trajectory'] or config['viz_states']) else len(proj_data)
    for i in tqdm(range(start_ind, num_trials, step_size)):
        print(f'\nTrial {i+1}')
    # for i in tqdm([1, 2, 4, 7]):
        if config['mode'] != 'hardware':
            torch.manual_seed(i)
            np.random.seed(i)
        # 8709 11200

        if config['compute_recovery_trajectory'] or config['test_recovery_trajectory'] or config['viz_states']:
            this_pair = proj_data[i]
            if config['compute_recovery_trajectory'] or config['test_recovery_trajectory']:
                initial_likelihood = this_pair['initial_likelihood']
                final_likelihood = this_pair['final_likelihood']

                initial_state = this_pair['initial_state']
            else:
                initial_state = this_pair
                likelihood = candidate_ood_likelihoods[i]
                ood_str = f'{likelihood:.1f}_OOD' if likelihood < quantile else f'{likelihood:.1f}_ID'
                env.frame_fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}{now}/')
                pathlib.Path.mkdir(env.frame_fpath, parents=True, exist_ok=True)
            env.reset(
                dof_pos=torch.cat(
                                    [
                                        initial_state[:8],
                                        torch.tensor([0, 0, 0, 0,]),
                                        initial_state[8:15], 
                                        torch.tensor([0.]) ], dim=0).reshape(1, -1)
            )
            if config['viz_states']:
                env.gym.write_viewer_image_to_file(env.viewer, f'{env.frame_fpath}/{ood_str}_{i}.png')
                continue
            else:
                goal = torch.tensor([0, 0, float(config['goal'])])
        else:
            env.reset()
            goal = torch.tensor([0, 0, float(config['goal'])])
            # goal = goal + 0.025 * torch.randn(1) + 0.2
        for controller in config['controllers'].keys():

            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}{now}/{controller}/trial_{i + 1}')
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
            succ = False
            while not succ:
                perturb_this_trial = (np.random.rand() < .1 or params['perturb_action']) and params['perturb_action']

                if config['mode']  == 'hardware':
                    params['perturb_action'] = False
                    perturb_this_trial = False

                if not perturb_this_trial:
                    print('No action perturbation this trial')
                final_distance_to_goal = do_trial(env, params, fpath, sim_env, ros_copy_node,
                                                seed=i, proj_path=None, perturb_this_trial=perturb_this_trial)
                succ = True
                # try:
                #     final_distance_to_goal = do_trial(env, params, fpath, sim_env, ros_copy_node,
                #                                     seed=i, proj_path=this_pair['proj_path'][:, :-1])
                #     succ = True
                # except Exception as e:
                #     print(e)
                #     torch.cuda.empty_cache()
                #     continue
            
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