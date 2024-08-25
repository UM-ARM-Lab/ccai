from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroCardSlidingEnv
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
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ccai.utils.allegro_utils import *
from ccai.allegro_contact import AllegroManipulationExternalContactProblem, PositionControlConstrainedSVGDMPC
from ccai.allegro_screwdriver_problem_diffusion import AllegroScrewdriverDiff
from ccai.models.trajectory_samplers import TrajectorySampler
from ccai.models.contact_samplers import GraphSearch, Node

from model import LatentDiffusionModel

from diffusion_mcts import DiffusionMCTS

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

print("CCAI_PATH", CCAI_PATH)

# device = 'cuda:0'
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


class AllegroCard(AllegroManipulationExternalContactProblem):
    def __init__(self,
                 start,
                 goal,
                 T,
                 chain,
                 object_location,
                 object_type,
                 world_trans,
                 object_asset_pos,
                 table_asset_pos,
                 regrasp_fingers=[],
                 contact_fingers=['index', 'middle', 'ring', 'thumb'],
                 friction_coefficient=0.95,
                 obj_dof=1,
                 obj_ori_rep='euler',
                 optimize_force=False,
                 turn=False,
                 obj_gravity=False,
                 device='cuda:0', **kwargs):
        self.obj_mass = 0.1
        super(AllegroCard, self).__init__(start=start, goal=goal, T=T, chain=chain,
                                                 object_location=object_location,
                                                 object_type=object_type, world_trans=world_trans,
                                                 object_asset_pos=object_asset_pos,
                                                 table_asset_pos=table_asset_pos,
                                                 regrasp_fingers=regrasp_fingers,
                                                 contact_fingers=contact_fingers,
                                                 friction_coefficient=friction_coefficient,
                                                 obj_dof=obj_dof,
                                                 obj_ori_rep=obj_ori_rep, obj_joint_dim=0,
                                                 optimize_force=optimize_force, device=device,
                                                 desired_ee_in_world_frame=True,
                                                 turn=turn, obj_gravity=obj_gravity,
                                                 env_contact=True)
        self.friction_coefficient = friction_coefficient
        
    def get_initial_xu(self, N):
        """
        use delta joint movement to get the initial trajectory
        the action (force at the finger tip) is not used. it is randomly intiailized
        the actual dynamics model is not used
        """

        u = 0.025 * torch.randn(N, self.T, self.du, device=self.device)
        # u = 0.5 * torch.randn(N, self.T, self.du, device=self.device)
        # u[:, :, 0] -= 0.05
        if self.optimize_force:
            u[..., 4*self.num_fingers:] = .75 * torch.randn(N, self.T, 3 * (self.num_contacts + 1), device=self.device)
            # for i, finger in enumerate(self.contact_fingers):
            #     idx = self.contact_force_indices_dict[finger]
            #     # if finger != 'index':
            #     #     u[..., idx] = 1.5 * torch.randn(N, self.T, 3, device=self.device)
            #     # else:
            #     u[..., idx] = 1.5 *.01 * torch.randn(N, self.T, 3, device=self.device)

        x = [self.start.reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            next_q = x[-1][:, :4 * self.num_fingers] + u[:, t, :4 * self.num_fingers]
            x.append(next_q)

        x = torch.stack(x[1:], dim=1)

        # if valve angle in state
        if self.dx == (4 * self.num_fingers + self.obj_dof):
            theta = np.linspace(self.start[-self.obj_dof:].cpu().numpy(), self.goal.cpu().numpy(), self.T + 1)[1:]
            theta = torch.tensor(theta, device=self.device, dtype=torch.float32)
            theta = theta.unsqueeze(0).repeat((N, 1, 1))
            # theta = self.start[-self.obj_dof:].unsqueeze(0).repeat((N, self.T, 1))
            # theta = torch.ones((N, self.T, self.obj_dof)).to(self.device) * self.start[-self.obj_dof:]
            x = torch.cat((x, theta), dim=-1)

        xu = torch.cat((x, u), dim=2)
        return xu
    def _cost(self, xu, start, goal):
        # TODO: check if the addtional term of the smoothness cost and running goal cost is necessary
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it

        smoothness_cost = 1 * torch.sum((state[1:, -self.obj_dof:] - state[:-1, -self.obj_dof:]) ** 2)

        theta = xu[:, self.dx - self.obj_dof:self.dx]
        # goal cost
        goal_cost = 15000 * torch.sum((theta[-1] - goal) ** 2)
        goal_cost += torch.sum((15000 * (theta[:-1] - goal) ** 2))

        return smoothness_cost + super()._cost(xu, start, goal) + goal_cost
    
    


# class IpoptScrewdriver(AllegroCard, IpoptProblem):

#     def __init__(self, *args, **kwargs):
#         device = kwargs.get('device', None)
#         if device is not None:
#             kwargs.pop('device')
#         super().__init__(*args, **kwargs, N=1, device='cpu')

def do_trial(env, params, fpath, inits_noise=None, noise_noise=None, sim=None,):
    "only turn the valve once"
    num_fingers = len(params['fingers'])
    state = env.get_state()
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None

    start = state['q'].reshape(4 * num_fingers + obj_dof).to(device=params['device'])
    # start = torch.cat((state['q'].reshape(10), torch.zeros(1).to(state['q'].device))).to(device=params['device'])
    if 'csvgd' in params['controller']:
        # use index finger to slide the card
        index_problem = AllegroCard(
            start=start[:4 * num_fingers + obj_dof],
            goal=params['valve_goal'],
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.card_pose,
            table_asset_pos=env.table_pose,
            object_location=params['object_location'],
            object_type='card',
            world_trans=env.world_trans,
            regrasp_fingers=['middle'],
            contact_fingers=['index'],
            obj_dof=obj_dof,
            obj_joint_dim=1,
            optimize_force=params['optimize_force'],
            default_dof_pos=env.default_dof_pos[:, :16],
            obj_gravity=params.get('obj_gravity', False),
        )
        # use middle finger to slide the card
        middle_problem = AllegroCard(
            start=start[:4 * num_fingers + obj_dof],
            goal=params['valve_goal'],
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.card_pose,
            table_asset_pos=env.table_pose,
            object_location=params['object_location'],
            object_type='card',
            world_trans=env.world_trans,
            contact_fingers=['middle'],
            regrasp_fingers=['index'],
            obj_dof=obj_dof,
            obj_joint_dim=1,
            optimize_force=params['optimize_force'],
            default_dof_pos=env.default_dof_pos[:, :16],
            obj_gravity=params.get('obj_gravity', False),
        )
        # use index and middle finger to slide the card
        index_middle_problem = AllegroCard(
            start=start[:4 * num_fingers + obj_dof],
            goal=params['valve_goal'],
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.card_pose,
            table_asset_pos=env.table_pose,
            object_location=params['object_location'],
            object_type='card',
            world_trans=env.world_trans,
            contact_fingers=['index', 'middle'],
            regrasp_fingers=[],
            obj_dof=obj_dof,
            obj_joint_dim=1,
            optimize_force=params['optimize_force'],
            default_dof_pos=env.default_dof_pos[:, :16],
            turn=True,
            obj_gravity=params.get('obj_gravity', False),
        )
        # reposition all fingers
        reposition_problem = AllegroCard(
            start=start[:4 * num_fingers + obj_dof],
            goal=params['valve_goal'],
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.card_pose,
            table_asset_pos=env.table_pose,
            object_location=params['object_location'],
            object_type='card',
            world_trans=env.world_trans,
            contact_fingers=[],
            regrasp_fingers=['index', 'middle'],
            obj_dof=obj_dof,
            obj_joint_dim=1,
            optimize_force=params['optimize_force'],
            default_dof_pos=env.default_dof_pos[:, :16],
            turn=True,
            obj_gravity=params.get('obj_gravity', False),
        )
        index_planner = PositionControlConstrainedSVGDMPC(index_problem, params)
        middle_planner = PositionControlConstrainedSVGDMPC(middle_problem, params)
        index_middle_planner = PositionControlConstrainedSVGDMPC(index_middle_problem, params)
        reposition_planner = PositionControlConstrainedSVGDMPC(reposition_problem, params)


    # warm-starting using learned sampler
    trajectory_sampler = None
    model_path = params.get('model_path', None)

    if model_path is not None:
        problem_for_sampler = None
        if params['projected'] or params['sample_contact']:
            pregrasp_problem_diff = AllegroScrewdriverDiff(
                start=start[:4 * num_fingers + obj_dof],
                goal=params['valve_goal'],
                T=params['T'],
                chain=params['chain'],
                device=params['device'],
                object_asset_pos=env.card_pose,
                object_location=params['object_location'],
                object_type='card',
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
                T=params['T'],
                chain=params['chain'],
                device=params['device'],
                object_asset_pos=env.card_pose,
                object_location=params['object_location'],
                object_type='card',
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
                T=params['T'],
                chain=params['chain'],
                device=params['device'],
                object_asset_pos=env.card_pose,
                object_location=params['object_location'],
                object_type='card',
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
                T=params['T'],
                chain=params['chain'],
                device=params['device'],
                object_asset_pos=env.card_pose,
                object_location=params['object_location'],
                object_type='card',
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
        trajectory_sampler = TrajectorySampler(T=params['T'] + 1, dx=(11 + (1 if params['sine_cosine'] else 0)) if not model_t else params['nzt'], du=21 if not model_t else 0, type=params['type'],
                                               timesteps=256, hidden_dim=128 if not model_t else 64,
                                               context_dim=3, generate_context=False,
                                               constrain=params['projected'],
                                               problem=problem_for_sampler,
                                               inits_noise=inits_noise, noise_noise=noise_noise,
                                               guided=params['use_guidance'],
                                               vae=vae)
        trajectory_sampler.load_state_dict(torch.load(f'{CCAI_PATH}/{model_path}', map_location=torch.device(params['device'])), strict=True)
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
    start = state['q'].reshape(4 * num_fingers + obj_dof).to(device=params['device'])
    actual_trajectory = []
    duration = 0

    def _partial_to_full(traj, mode):
        if mode == 'index':
            traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 3).to(device=params['device'])), dim=-1)
        if mode == 'middle':
            traj = torch.cat((traj[..., :-3], torch.zeros(*traj.shape[:-1], 3).to(device=params['device']),
                              traj[..., -3:]), dim=-1)
        if mode == "reposition":
            traj = torch.cat((traj, torch.zeros(*traj.shape[:-1], 6).to(device=params['device'])), dim=-1)
        return traj

    def _full_to_partial(traj, mode):
        if mode == 'index':
            traj = traj[..., :-3]
        if mode == 'middle':
            traj = torch.cat((traj[..., :-6], traj[..., -3:]), dim=-1)
        if mode == "reposition":
            traj = traj[..., :-6]
        return traj
    
    def convert_sine_cosine_to_yaw(xu):
        """
        xu is shape (N, T, 37)
        Replace the sine and cosine in xu with yaw and return the new xu
        """
        sine = xu[..., 11]
        cosine = xu[..., 10]
        yaw = torch.atan2(sine, cosine)
        xu_new = torch.cat([xu[..., :10], yaw.unsqueeze(-1), xu[..., 12:]], dim=-1)
        return xu_new
    
    def convert_yaw_to_sine_cosine(xu):
        """
        xu is shape (N, T, 36)
        Replace the yaw in xu with sine and cosine and return the new xu
        """
        yaw = xu[10]
        sine = torch.sin(yaw)
        cosine = torch.cos(yaw)
        xu_new = torch.cat([xu[:10], cosine.unsqueeze(-1), sine.unsqueeze(-1), xu[11:]], dim=-1)
        return xu_new
    
    def execute_traj(planner, mode, goal=None, fname=None):
        # reset planner
        state = env.get_state()
        state = state['q'].reshape(-1).to(device=params['device'])

        # generate context from mode
        contact = -torch.ones(params['N'], 2).to(device=params['device'])
        if mode == 'index': 
            contact[:, 0] = 1
        elif mode == 'middle':
            contact[:, 1] = 1
        elif mode == 'index_middle':
            contact[:, 0] = 1
            contact[:, 1] = 1

        # generate initial samples with diffusion model
        initial_samples = None
        sim_rollouts = None
        if trajectory_sampler is not None and params.get('diff_init', True):
            with torch.no_grad():
                start = state.clone()
                # if state[-1] < -1.0:
                #     start[-1] += 0.75
                a = time.perf_counter()
                # start_for_diff = start#convert_yaw_to_sine_cosine(start)
                if params['sine_cosine']:
                    start_for_diff = convert_yaw_to_sine_cosine(start)
                else:
                    start_for_diff = start
                initial_samples, _, _ = trajectory_sampler.sample(N=params['N'], start=start_for_diff.reshape(1, -1),
                                                                  H=params['T'] + 1,
                                                                  constraints=contact)
                if params['sine_cosine']:
                    initial_samples = convert_sine_cosine_to_yaw(initial_samples)
                print('Sampling time', time.perf_counter() - a)
                # if state[-1] < -1.0:
                #     initial_samples[:, :, -1] -= 0.75
            
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
        if trajectory_sampler is None or not params.get('diff_init', True):
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
            state = state['q'].reshape(4 * num_fingers + params['obj_dof']).to(device=params['device'])
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
            print('Solve time for step', time.time() - s)
            planned_trajectories.append(plans)
            optimizer_paths.append(copy.deepcopy(planner.path))
            N, T, _ = plans.shape

            contact_distance[T] = torch.stack((planner.problem.data['index']['sdf'].reshape(N, T + 1),
                                               planner.problem.data['middle']['sdf'].reshape(N, T + 1)),
                                            #    planner.problem.data['thumb']['sdf'].reshape(N, T + 1)),
                                              dim=1).detach().cpu()

            contact_points[T] = torch.stack((planner.problem.data['index']['closest_pt_world'].reshape(N, T + 1, 3),
                                             planner.problem.data['middle']['closest_pt_world'].reshape(N, T + 1, 3)),
                                            #  planner.problem.data['thumb']['closest_pt_world'].reshape(N, T + 1, 3)),
                                            dim=2).detach().cpu()

            # execute the action
            action = best_traj[0, planner.problem.dx:planner.problem.dx + planner.problem.du]
            state = env.get_state()
            state = state['q'].reshape(-1).to(device=params['device'])
            xu = torch.cat((state, action))

            # record the actual trajectory
            actual_trajectory.append(xu)
            x = best_traj[0, :planner.problem.dx + planner.problem.du]
            x = x.reshape(1, planner.problem.dx + planner.problem.du)
            action = x[:, planner.problem.dx:planner.problem.dx + planner.problem.du].to(device=env.device)
            # print(action)
            action = action[:, :4 * num_fingers]


            if params['visualize_plan']:
                traj_for_viz = best_traj[:, :planner.problem.dx]
                traj_for_viz = torch.cat((state[:planner.problem.dx].unsqueeze(0), traj_for_viz), dim=0)
                tmp = torch.zeros((traj_for_viz.shape[0], 1),
                                  device=best_traj.device)  # add the joint for the screwdriver cap
                traj_for_viz = torch.cat((traj_for_viz[:, :-1], tmp, tmp, tmp, traj_for_viz[:, -1:]), dim=1)
                # traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof] = axis_angle_to_euler(traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof])

                viz_fpath = pathlib.PurePath.joinpath(fpath, f"{fname}/timestep_{k}")
                img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
                gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
                pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
                pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
                visualize_trajectory(traj_for_viz, index_middle_problem.contact_scenes, viz_fpath,
                                     index_middle_problem.fingers, 6, task='card')

            action = action + state[:4 * num_fingers].unsqueeze(0).to(env.device)

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

    def visualize_trajectory_wrapper(traj, contact_scenes, fname, plan_or_init, index, fingers, obj_dof, k):
        viz_fpath = pathlib.PurePath.joinpath(fpath, f"{fname}/{plan_or_init}/{index}/timestep_{k}")
        img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
        gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
        pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
        pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
        visualize_trajectory(traj, contact_scenes, viz_fpath, fingers, obj_dof + 1)

    def plan_contacts(state, depth, next_node, multi_particle=False):
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
        with open(f"{fpath}/contact_planning_{depth}.pkl", "wb") as f:
            pkl.dump((contact_node_sequence, closed_set, planning_time, contact_sequence_sampler.iter), f)
        if contact_node_sequence is None:
            print('No contact sequence found')
            # Find the node in the closed set with the lowest cost
            # min_cost = float('inf')
            # min_node = None
            # for node in closed_set:
            #     if node.fscore < min_cost and node.fscore > 0:
            #         min_cost = node.fscore
            #         min_node = node
            # contact_node_sequence = [min_node]
            return None, None
        last_node = contact_node_sequence[-1]

        if next_node_init:
            offset = 0
        else:
            offset = 1
        if offset == 1 and len(contact_node_sequence) == 1:
            return [], None
        next_node = last_node.contact_sequence[offset]
        contact_sequence = np.array(last_node.contact_sequence)
        contact_sequence = (contact_sequence + 1)/2
        contact_sequence = [contact_vec_to_label[contact_sequence[i].sum()] for i in range(contact_sequence.shape[0])]
        contact_sequence = contact_sequence[offset:] 
        del contact_sequence_sampler
        torch.cuda.empty_cache()
        return contact_sequence, next_node

    state = env.get_state()
    state = state['q'].reshape(-1)[:11].to(device=params['device'])


    contact_label_to_vec = {
                            'index': 0,
                            'middle': 1,
                            'index_middle': 2,
                            'reposition': 3
                            }
    contact_vec_to_label = dict((v, k) for k, v in contact_label_to_vec.items())


    sample_contact = params.get('sample_contact', False)
    # num_stages = 2 + 3 * (params['num_turns'] - 1)
    if not sample_contact:
        contact_sequence = []
        contact_options = list(contact_label_to_vec.keys())
        for k in range(params['num_turns']):
            # single random choice from contact_label_to_vec keys
            contact_sequence.append(np.random.choice(contact_options))
    else:
        contact_sequence = None
    num_stages = len(contact_sequence)
    # state = state['q'].reshape(-1)[:11].to(device=params['device'])
    # initial_samples = gen_initial_samples_multi_mode(contact_sequence)
    # pkl.dump(initial_samples, open(f"{fpath}/long_horizon_inits.p", "wb"))
    # return -1
    contact = None
    next_node = None
    state = env.get_state()
    state = state['q'].reshape(-1)[:11].to(device=params['device'])

    executed_contacts = []
    for stage in range(num_stages):
        state = env.get_state()
        state = state['q'].reshape(-1)[:11].to(device=params['device'])
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
        # if stage == 0:
        #     contact = 'pregrasp'
        #     start = env.get_state()['q'].reshape(4 * num_fingers + 4).to(device=params['device'])
        #     best_traj, _ = pregrasp_planner.step(start[:pregrasp_planner.problem.dx])
        #     for x in best_traj[:, :4 * num_fingers]:
        #         action = x.reshape(-1, 4 * num_fingers).to(device=env.device) # move the rest fingers
        #         env.step(action)
        #         continue
        if sample_contact and (stage == 1 or params['replan']):
            new_contact_sequence, new_next_node = plan_contacts(state, num_stages - stage, next_node, params['multi_particle_search'])
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
                if contact_sequence[0] == 'index':
                    next_node = (1, -1)
                elif contact_sequence[0] == 'middle':
                    next_node = (-1, 1)
                elif contact_sequence[0] == 'index_middle':
                    next_node = (1, 1)
                else:
                    next_node = (-1, -1)
            print(contact_sequence)
            # return -1
            contact = contact_sequence[0]  
        elif stage >= len(contact_sequence):
            print('Planner thinks task is complete')
            break
        elif sample_contact:
            contact = contact_sequence[stage - 1]
        else:
            contact = contact_sequence[stage]
        executed_contacts.append(contact)
        print(stage, contact)
        if contact == 'index':
            # _goal = torch.tensor([0, 0, state[-1]]).to(device=params['device'])
            _goal = torch.tensor([0, -0.02 + state[-2], 0]).to(device=params['device'])
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance = execute_traj(
                index_planner, mode='index', goal=_goal, fname=f'index_{stage}')
            print('traj pre index', traj.shape)
            plans = [_partial_to_full(plan, 'index') for plan in plans]
            traj = torch.cat((traj[..., :-3], torch.zeros(*traj.shape[:-1], 3).to(device=params['device']), traj[..., -3:]), dim=-1)
            print('traj post index', traj.shape)
            # plans = [torch.cat((plan[..., :-6],
            #                     torch.zeros(*plan.shape[:-1], 3).to(device=params['device']),
            #                     plan[..., -6:]),
            #                    dim=-1) for plan in plans]
            # traj = torch.cat((traj[..., :-6], torch.zeros(*traj.shape[:-1], 3).to(device=params['device']),
            #                   traj[..., -6:]), dim=-1)
            _add_to_dataset(traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance,
                            contact_state=torch.tensor([1.0, 0.0]))
        elif contact == 'middle':
            _goal = torch.tensor([0, -0.02 + state[-2], 0]).to(device=params['device'])
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance = execute_traj(
                middle_planner, mode='middle', goal=_goal, fname=f'middle_{stage}')
            
            print('traj pre middle', traj.shape)
            plans = [_partial_to_full(plan, 'middle') for plan in plans]
            traj = torch.cat((traj[..., :-6], torch.zeros(*traj.shape[:-1], 3).to(device=params['device']), traj[..., -6:]), dim=-1)
            print('traj post middle', traj.shape)

            _add_to_dataset(traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance,
                            contact_state=torch.tensor([0.0, 1.0]))
        elif contact == 'index_middle':
            _goal = torch.tensor([0, -0.02 + state[-2], 0]).to(device=params['device'])
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance = execute_traj(
                index_middle_planner, mode='index_middle', goal=_goal, fname=f'index_middle_{stage}')
            
            print('traj index middle', traj.shape)
            _add_to_dataset(traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance,
                            contact_state=torch.tensor([1.0, 1.0]))
        elif contact == 'reposition':
            _goal = torch.tensor([0, state[-2], 0]).to(device=params['device'])
            traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance = execute_traj(
                reposition_planner, mode='reposition', goal=_goal, fname=f'reposition_{stage}')
            
            print('traj pre reposition', traj.shape)
            plans = [_partial_to_full(plan, 'reposition') for plan in plans]
            traj = torch.cat((traj[..., :-3], torch.zeros(*traj.shape[:-1], 6).to(device=params['device']), traj[..., -3:]), dim=-1)
            print('traj post reposition', traj.shape)
            _add_to_dataset(traj, plans, inits, init_sim_rollouts, optimizer_paths, contact_points, contact_distance,
                            contact_state=torch.tensor([0.0, 0.0]))          
        if contact != 'pregrasp':
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
    state = state['q'].reshape(4 * num_fingers + obj_dof).to(device=params['device'])
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
    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_card.yaml').read_text())
    from tqdm import tqdm

    if not config['visualize']:
        img_save_dir = None

    env = AllegroCardSlidingEnv(1, control_mode='joint_impedance',
                                        use_cartesian_controller=False,
                                        viewer=config['visualize'],
                                        steps_per_action=60,
                                        friction_coefficient=config['friction_coefficient'],
                                        # friction_coefficient=1.0,  # DEBUG ONLY, set the friction very high
                                        device=config['sim_device'],
                                        video_save_path=img_save_dir,
                                        joint_stiffness=config['kp'],
                                        fingers=config['fingers'],
                                        gradual_control=False,
                                        gravity=True, # For data generation only
                                        randomize_obj_start=config.get('randomize_obj_start', False)
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
    start_ind = 0 if not config['sample_contact'] else 0
    for i in tqdm(range(0, config['num_trials'])):
    # for i in tqdm(range(0, 7)):
        if not config['data_gen']:
            torch.manual_seed(i)
            np.random.seed(i)

        goal = torch.tensor([0, 0.0, 0])
        # goal = goal + 0.025 * torch.randn(1) + 0.2
        for controller in config['controllers'].keys():
            env.reset()
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
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
            object_location = torch.tensor(env.card_pose).to(
                params['device'])  # TODO: confirm if this is the correct location
            params['object_location'] = object_location
            # If params['device'] is cuda:1 but the computer only has 1 gpu, change to cuda:0
            final_distance_to_goal = do_trial(env, params, fpath, inits_noise[i], noise_noise[i])
        print(results)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
