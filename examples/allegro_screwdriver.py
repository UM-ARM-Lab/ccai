from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
# from isaac_victor_envs.tasks.allegro_ros import RosAllegroValveTurningEnv

import numpy as np
import pickle as pkl

import torch
import time
import copy
import yaml
import pathlib
from functools import partial
import sys

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

from model import LatentDiffusionModel

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


class AllegroIndexPlanner:
    "The index finger is desgie"

    def __init__(self, chain_hand, chain_screwdriver, world_trans, screwdriver_asset_pose, fingers, ee_names,
                 frame_indices) -> None:
        self.chain_hand = chain_hand
        self.chain_screwdriver = chain_screwdriver
        self.world_trans = world_trans
        self.screwdriver_asset_pose = screwdriver_asset_pose
        self.fingers = fingers
        self.finger_target_location
        self.ee_names = ee_names
        self.frame_indices = frame_indices

    def step(self):
        forward_kinematics(partial_to_full_state(state['q'][:, :12], fingers=params['fingers']))[ee_names['index']]

    # def inverse_kinematics(self, q):
    #     eps = 1e-3
    #     for _ in range(10):
    #         J = chain.jacobian(partial_to_full_state(q), link_indices=torch.tensor([self.frame_indices['index']],
    #                                                                             device=params['device']))[:, :3, -4:]

    #         # get update in robot frame
    #         dx = self.world_trans.inverse().transform_normals(torch.tensor([[-1.0, 0.0, 0.0]],
    #                                                                 device=params['device']).reshape(1, 3)).reshape(1, 3,
    #                                                                                                                 1)
    #         # joint update
    #         dq = J.permute(0, 2, 1) @ torch.linalg.inv(J @ J.permute(0, 2, 1) + 1e-5 * eye) @ dx
    #         q[:, 4:] += eps * dq.reshape(1, 4)

    #     return q


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
                 device='cuda:0', **kwargs):
        super(AllegroScrewdriver, self).__init__(start=start, goal=goal, T=T, chain=chain,
                                                 object_location=object_location,
                                                 object_type=object_type, world_trans=world_trans,
                                                 object_asset_pos=object_asset_pos,
                                                 regrasp_fingers=regrasp_fingers,
                                                 contact_fingers=contact_fingers,
                                                 friction_coefficient=friction_coefficient,
                                                 obj_dof=obj_dof,
                                                 obj_ori_rep=obj_ori_rep, obj_joint_dim=1,
                                                 optimize_force=optimize_force, device=device)
        self.friction_coefficient = friction_coefficient

    def _cost(self, xu, start, goal):
        # TODO: check if the addtional term of the smoothness cost and running goal cost is necessary
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it

        smoothness_cost = torch.sum((state[1:, -self.obj_dof:] - state[:-1, -self.obj_dof:]) ** 2)
        upright_cost = 500 * torch.sum(
            (state[:, -self.obj_dof:-1]) ** 2)  # the screwdriver should only rotate in z direction
        return smoothness_cost + upright_cost + super()._cost(xu, start, goal)



# class IpoptScrewdriver(AllegroScrewdriver, IpoptProblem):

#     def __init__(self, *args, **kwargs):
#         device = kwargs.get('device', None)
#         if device is not None:
#             kwargs.pop('device')
#         super().__init__(*args, **kwargs, N=1, device='cpu')

def do_trial(env, params, fpath, sim_viz_env=None, ros_copy_node=None, inits_noise=None, noise_noise=None, sim=None,):
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

        # initial grasp
        pregrasp_problem = AllegroScrewdriver(
            start=start[:4 * num_fingers + obj_dof],
            goal=params['valve_goal'] * 0,
            T=params['T'],
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
            default_dof_pos=env.default_dof_pos[:, :16]
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
            default_dof_pos=env.default_dof_pos[:, :16]
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
            default_dof_pos=env.default_dof_pos[:, :16]
        )
        pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, params)
        index_regrasp_planner = PositionControlConstrainedSVGDMPC(index_regrasp_problem, params)
        thumb_and_middle_regrasp_planner = PositionControlConstrainedSVGDMPC(thumb_and_middle_regrasp_problem, params)
        turn_planner = PositionControlConstrainedSVGDMPC(turn_problem, params)


    elif params['controller'] == 'ipopt':
        # index finger is used for stability
        if 'index' in params['fingers']:
            fingers = params['fingers']
        else:
            fingers = ['index'] + params['fingers']

        # initial grasp
        pregrasp_problem = IpoptScrewdriver(
            start=start[:4 * num_fingers + obj_dof],
            goal=params['valve_goal'] * 0,
            T=params['T'],
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
        index_regrasp_problem = IpoptScrewdriver(
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
            default_dof_pos=env.default_dof_pos[:, :16]
        )
        thumb_and_middle_regrasp_problem = IpoptScrewdriver(
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
            default_dof_pos=env.default_dof_pos[:, :16]
        )
        turn_problem = IpoptScrewdriver(
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
            default_dof_pos=env.default_dof_pos[:, :16]
        )
        pregrasp_planner = IpoptMPC(pregrasp_problem, params)
        index_regrasp_planner = IpoptMPC(index_regrasp_problem, params)
        thumb_and_middle_regrasp_planner = IpoptMPC(thumb_and_middle_regrasp_problem, params)
        turn_planner = IpoptMPC(turn_problem, params)
    else:
        raise ValueError('Invalid controller')

    # warm-starting using learned sampler
    trajectory_sampler = None
    model_path = params.get('model_path', None)

    if model_path is not None:
        problem_for_sampler = None
        if params['projected']:
            pregrasp_problem_diff = AllegroScrewdriverDiff(
                start=start[:4 * num_fingers + obj_dof],
                goal=params['valve_goal'] * 0,
                T=params['T'],
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
                default_dof_pos=env.default_dof_pos[:, :16]
            )
            thumb_and_middle_regrasp_problem_diff = AllegroScrewdriverDiff(
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
                default_dof_pos=env.default_dof_pos[:, :16]
            )
            turn_problem_diff = AllegroScrewdriverDiff(
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
                default_dof_pos=env.default_dof_pos[:, :16]
            )

            problem_for_sampler = {
                (-1, -1, -1): pregrasp_problem_diff,
                (-1, 1, 1): index_regrasp_problem_diff,
                (1, -1, -1): thumb_and_middle_regrasp_problem_diff,
                (1, 1, 1): turn_problem_diff
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
        trajectory_sampler = TrajectorySampler(T=params['T'] + 1, dx=15 if not model_t else params['nzt'], du=21 if not model_t else 0, type=params['type'],
                                               timesteps=256, hidden_dim=128 if not model_t else 64,
                                               context_dim=3, generate_context=True,
                                               constrain=params['projected'],
                                               problem=problem_for_sampler,
                                               inits_noise=inits_noise, noise_noise=noise_noise,
                                               guided=params['use_guidance'],
                                               vae=vae)
        trajectory_sampler.load_state_dict(torch.load(f'{CCAI_PATH}/{model_path}'))
        trajectory_sampler.to(device=params['device'])
        trajectory_sampler.send_norm_constants_to_submodels()
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
            traj = torch.cat((traj[..., :-3], traj[..., -3:]), dim=-1)
        if mode == 'thumb_middle':
            traj = traj[..., :-6]
        if mode == 'pregrasp':
            traj = traj[..., :-9]
        return traj

    def execute_traj(planner, mode, goal=None, fname=None):
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
        initial_samples = None
        sim_rollouts = None
        if trajectory_sampler is not None:
            with torch.no_grad():
                start = state.clone()
                if state[-1] < -1.0:
                    start[-1] += 0.75
                a = time.perf_counter()
                initial_samples, _, _ = trajectory_sampler.sample(N=params['N'], start=start.reshape(1, -1),
                                                                  H=params['T'] + 1,
                                                                  constraints=contact)
                print('Sampling time', time.perf_counter() - a)
                if state[-1] < -1.0:
                    initial_samples[:, :, -1] -= 0.75
            
            sim_rollouts = torch.zeros_like(initial_samples)
            # for i in range(params['N']):
            #     sim_rollout = rollout_trajectory_in_sim(env_sim_rollout, initial_samples[i])
            #     sim_rollouts[i] = sim_rollout

            initial_samples = _full_to_partial(initial_samples, mode)
            initial_x = initial_samples[:, 1:, :planner.problem.dx]
            initial_u = initial_samples[:, :-1, -planner.problem.du:]
            initial_samples = torch.cat((initial_x, initial_u), dim=-1)

        state = env.get_state()
        state = state['q'].reshape(-1).to(device=params['device'])
        state = state[:planner.problem.dx]
        # print(params['T'], state.shape, initial_samples)
        planner.reset(state, T=params['T'], goal=goal, initial_x=initial_samples)
        if trajectory_sampler is None:
            initial_samples = planner.x.detach().clone()
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
            if plans is not None and resample:
                # combine past with plans
                executed_trajectory = torch.stack(actual_trajectory, dim=0)
                executed_trajectory = executed_trajectory.reshape(1, -1, planner.problem.dx + planner.problem.du)
                executed_trajectory = executed_trajectory.repeat(params['N'], 1, 1)
                executed_trajectory = _partial_to_full(executed_trajectory, mode)
                plans = _partial_to_full(plans, mode)
                plans = torch.cat((executed_trajectory, plans), dim=1)
                if state[-1] < -1.0:
                    plans[:, :, 14] += 0.75
                    executed_trajectory[:, :, 14] += 0.75

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

                    if state[-1] < -1.0:
                        initial_samples[:, :, 14] -= 0.75
                    # update the initial samples
                    planner.x = initial_samples[:, k:]

            s = time.time()
            best_traj, plans = planner.step(state)
            print('Solve time for step', time.time() - s)
            planned_trajectories.append(plans)
            optimizer_paths.append(copy.deepcopy(planner.path))
            N, T, _ = plans.shape

            contact_distance[T] = torch.stack((planner.problem.data['index']['sdf'].reshape(N, T + 1),
                                               planner.problem.data['middle']['sdf'].reshape(N, T + 1),
                                               planner.problem.data['thumb']['sdf'].reshape(N, T + 1)),
                                              dim=1).detach().cpu()
            contact_points[T] = torch.stack((planner.problem.data['index']['closest_pt_world'].reshape(N, T + 1, 3),
                                             planner.problem.data['middle']['closest_pt_world'].reshape(N, T + 1, 3),
                                             planner.problem.data['thumb']['closest_pt_world'].reshape(N, T + 1, 3)),
                                            dim=2).detach().cpu()

            # execute the action
            action = best_traj[0, planner.problem.dx:planner.problem.dx + planner.problem.du]
            state = env.get_state()
            state = state['q'].reshape(-1).to(device=params['device'])
            xu = torch.cat((state[:-1], action))

            # record the actual trajectory
            actual_trajectory.append(xu)
            x = best_traj[0, :planner.problem.dx + planner.problem.du]
            x = x.reshape(1, planner.problem.dx + planner.problem.du)
            action = x[:, planner.problem.dx:planner.problem.dx + planner.problem.du].to(device=env.device)
            # print(action)
            action = action[:, :4 * num_fingers_to_plan]
            if params['exclude_index']:
                action = state.unsqueeze(0)[:, 4:4 * num_fingers] + action
                action = torch.cat((state.unsqueeze(0)[:, :4], action), dim=1)  # add the index finger back
            else:
                action = action + state.unsqueeze(0)[:, :4 * num_fingers].to(device=env.device)

            if params['mode'] == 'hardware':
                sim_viz_env.set_pose(env.get_state()['all_state'].to(device=env.device))
                sim_viz_env.step(action)
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
        actual_trajectory = torch.stack(actual_trajectory, dim=0)
        # can't stack plans as each is of a different length

        # for memory reasons we clear the data
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
            data[t]['contact_points'].append(contact_points[t])
            data[t]['contact_distance'].append(contact_distance[t])
            data[t]['contact_state'].append(contact_state)

    state = env.get_state()

    contact_label_to_vec = {'pregrasp': 0,
                            'index': 1,
                            'thumb_middle': 2,
                            'turn': 3
                            }
    contact_vec_to_label = dict((v, k) for k, v in contact_label_to_vec.items())


    sample_contact = params.get('sample_contact', False)
    num_stages = 2 + 3 * (params['num_turns'] - 1)
    if not sample_contact:
        contact_sequence = ['pregrasp', 'turn']
        for k in range(params['num_turns'] - 1):
            contact_options = ['index', 'thumb_middle']
            perm = np.random.permutation(2)
            contact_sequence += [contact_options[perm[0]], contact_options[perm[1]], 'turn']
    else:
        contact_sequence = None
    print(contact_sequence)
    for stage in range(num_stages):
        state = env.get_state()
        state = state['q'].reshape(-1)[:15].to(device=params['device'])
        valve_goal = torch.tensor([0, 0, state[-1]]).to(device=params['device'])

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
        contact = contact_sequence[stage]
        print(stage, contact)
        if contact == 'pregrasp':
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance = execute_traj(
                pregrasp_planner, mode='pregrasp', fname=f'pregrasp_{stage}')

            # include zero for the contact forces
            plans = [torch.cat((plan,
                                torch.zeros(*plan.shape[:-1], 9).to(device=params['device'])),
                               dim=-1) for plan in plans]
            traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 9).to(device=params['device'])), dim=-1)
            _add_to_dataset(traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, contact_state=torch.zeros(3))
        elif contact == 'index':
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance = execute_traj(
                index_regrasp_planner, mode='index', goal=valve_goal, fname=f'index_regrasp_{stage}')

            plans = [torch.cat((plan[..., :-6],
                                torch.zeros(*plan.shape[:-1], 3).to(device=params['device']),
                                plan[..., -6:]),
                               dim=-1) for plan in plans]
            traj = torch.cat((traj[..., :-6], torch.zeros(*traj.shape[:-1], 3).to(device=params['device']),
                              traj[..., -6:]), dim=-1)
            _add_to_dataset(traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance,
                            contact_state=torch.tensor([0.0, 1.0, 1.0]))
        elif contact == 'thumb_middle':
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance = execute_traj(
                thumb_and_middle_regrasp_planner, mode='thumb_middle',
                goal=valve_goal, fname=f'thumb_middle_regrasp_{stage}')
            plans = [torch.cat((plan,
                                torch.zeros(*plan.shape[:-1], 6).to(device=params['device'])),
                               dim=-1) for plan in plans]
            traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 6).to(device=params['device'])), dim=-1)

            _add_to_dataset(traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance,
                            contact_state=torch.tensor([1.0, 0.0, 0.0]))
        elif contact == 'turn':
            valve_goal = torch.tensor([0, 0, state[-2] - np.pi / 3.0]).to(device=params['device'])
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance = execute_traj(
                turn_planner, mode='turn', goal=valve_goal, fname=f'turn_{stage}')

            _add_to_dataset(traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance, contact_state=torch.ones(3))
        actual_trajectory.append(traj)
    # change to numpy and save data
    for t in range(1, 1 + params['T']):
        data[t]['plans'] = torch.stack(data[t]['plans']).cpu().numpy()
        data[t]['starts'] = torch.stack(data[t]['starts']).cpu().numpy()
        data[t]['contact_points'] = torch.stack(data[t]['contact_points']).cpu().numpy()
        data[t]['contact_distance'] = torch.stack(data[t]['contact_distance']).cpu().numpy()
        data[t]['contact_state'] = torch.stack(data[t]['contact_state']).cpu().numpy()

    import pickle
    pickle.dump(data, open(f"{fpath}/traj_data.p", "wb"))
    state = env.get_state()
    state = state['q'].reshape(4 * num_fingers + obj_dof + 1).to(device=params['device'])
    actual_trajectory.append(state.clone()[: 4 * num_fingers + obj_dof])
    # actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 4 * num_fingers + obj_dof)
    # turn_problem.T = actual_trajectory.shape[0]
    # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    # final_distance_to_goal = (state.clone()[:, -obj_dof:] - params['valve_goal']).abs()

    # print(f'Controller: {params["controller"]} Final distance to goal: {torch.min(final_distance_to_goal)}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')

    with open(f'{fpath.resolve()}/trajectory.pkl', 'wb') as f:
        pickle.dump([i.cpu().numpy() for i in actual_trajectory], f)
    # np.savez(f'{fpath.resolve()}/trajectory.npz', x=[i.cpu().numpy() for i in actual_trajectory],)
             #  constr=constraint_val.cpu().numpy(),
            #  d2goal=final_distance_to_goal.cpu().numpy())
    env.reset()
    return -1#torch.min(final_distance_to_goal).cpu().numpy()


if __name__ == "__main__":
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/{sys.argv[1]}.yaml').read_text())
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver_latent_diff_init_csvto.yaml').read_text())
    from tqdm import tqdm

    if config['mode'] == 'hardware':
        env = RosAllegroValveTurningEnv(1, control_mode='joint_impedance',
                                        use_cartesian_controller=False,
                                        viewer=True,
                                        steps_per_action=60,
                                        friction_coefficient=1.0,
                                        device=config['sim_device'],
                                        valve=config['object_type'],
                                        video_save_path=img_save_dir,
                                        joint_stiffness=config['kp'],
                                        fingers=config['fingers'],
                                        )
    else:
        if not config['visualize']:
            img_save_dir = None

        env = AllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                           use_cartesian_controller=False,
                                           viewer=config['visualize'],
                                           steps_per_action=60,
                                           friction_coefficient=config['friction_coefficient'] * 1.05,
                                           # friction_coefficient=1.0,  # DEBUG ONLY, set the friction very high
                                           device=config['sim_device'],
                                           video_save_path=img_save_dir,
                                           joint_stiffness=config['kp'],
                                           fingers=config['fingers'],
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

    sim_env = None
    ros_copy_node = None
    if config['mode'] == 'hardware':
        sim_env = env
        from hardware.hardware_env import HardwareEnv

        env = HardwareEnv(sim_env.default_dof_pos[:, :16], finger_list=['index', 'thumb'], kp=config['kp'])
        env.world_trans = sim_env.world_trans
        env.joint_stiffness = sim_env.joint_stiffness
        env.device = sim_env.device
        env.valve_pose = sim_env.valve_pose
    elif config['mode'] == 'hardware_copy':
        from hardware.hardware_env import RosNode

        ros_copy_node = RosNode()

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
    for i in tqdm(range(0, config['num_trials'])):
        
        torch.manual_seed(i)
        np.random.seed(i)

        goal = - 0.5 * torch.tensor([0, 0, np.pi])
        # goal = goal + 0.025 * torch.randn(1) + 0.2
        for controller in config['controllers'].keys():
            env.reset()
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
            # set up params
            params = config.copy()
            params.pop('controllers')
            params.update(config['controllers'][controller])
            params['controller'] = controller
            params['valve_goal'] = goal.to(device=params['device'])
            params['chain'] = chain.to(device=params['device'])
            object_location = torch.tensor([0, 0, 1.205]).to(
                params['device'])  # TODO: confirm if this is the correct location
            params['object_location'] = object_location
            final_distance_to_goal = do_trial(env, params, fpath, sim_env, ros_copy_node, inits_noise[i], noise_noise[i])
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
