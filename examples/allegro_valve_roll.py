"""
This script turns the valve assuming the contact point does not change, which means it does not considering rolling
"""
import os

from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroValveTurningEnv

import numpy as np
import pickle as pkl

import torch
import time
import yaml
import pathlib
from functools import partial
from torch.func import vmap, jacrev, hessian, jacfwd

from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
from ccai.kernels import rbf_kernel, structured_rbf_kernel

from ccai.problem import ConstrainedSVGDProblem
from ccai.mpc.csvgd import Constrained_SVGD_MPC

import time
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
# import pytorch3d.transforms as tf

import matplotlib.pyplot as plt
from utils.allegro_utils import partial_to_full_state, full_to_partial_state, combine_finger_constraints, state2ee_pos

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

device = 'cuda:0'
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')


class PositionControlConstrainedSteinTrajOpt(ConstrainedSteinTrajOpt):
    def __init__(self, problem, params):
        super().__init__(problem, params)
        self.torque_limit = params.get('torque_limit', 10.0)
        self.kp = params.get('kp', 50.0)

    def _clamp_in_bounds(self, xuz):
        N = xuz.shape[0]
        min_x = self.problem.x_min.reshape(1, 1, -1).repeat(1, self.problem.T, 1)
        max_x = self.problem.x_max.reshape(1, 1, -1).repeat(1, self.problem.T, 1)
        if self.problem.dz > 0:
            min_x = torch.cat((min_x, -1e3 * torch.ones(1, self.problem.T, self.problem.dz)), dim=-1)
            max_x = torch.cat((max_x, 1e3 * torch.ones(1, self.problem.T, self.problem.dz)), dim=-1)

        torch.clamp_(xuz, min=min_x.to(device=xuz.device).reshape(1, -1),
                     max=max_x.to(device=xuz.device).reshape(1, -1))

        if self.problem.du > 0:
            xuz_copy = xuz.reshape((N, self.problem.T, -1))
            robot_joint_angles = xuz_copy[:, :-1, :8]
            robot_joint_angles = torch.cat(
                (self.problem.start[:8].reshape((1, 1, 8)).repeat((N, 1, 1)), robot_joint_angles), dim=1)

            # make the commanded delta position respect the joint limits
            min_u_jlim = self.problem.robot_joint_x_min.repeat((N, self.problem.T, 1)).to(
                xuz.device) - robot_joint_angles
            max_u_jlim = self.problem.robot_joint_x_max.repeat((N, self.problem.T, 1)).to(
                xuz.device) - robot_joint_angles

            # make the commanded delta position respect the torque limits
            min_u_tlim = -self.torque_limit / self.kp * torch.ones_like(min_u_jlim)
            max_u_tlim = self.torque_limit / self.kp * torch.ones_like(max_u_jlim)

            # overall commanded delta position limits
            min_u = torch.where(min_u_jlim > min_u_tlim, min_u_jlim, min_u_tlim)
            max_u = torch.where(max_u_tlim > max_u_jlim, max_u_jlim, max_u_tlim)
            min_x = min_x.repeat((N, 1, 1)).to(device=xuz.device)
            max_x = max_x.repeat((N, 1, 1)).to(device=xuz.device)
            min_x[:, :, self.problem.dx:self.problem.dx + self.problem.du] = min_u
            max_x[:, :, self.problem.dx:self.problem.dx + self.problem.du] = max_u

            # print(min_u)
            # print(max_u)
            # print(xuz.shape)
            _x = xuz.reshape(N, self.problem.T, -1)

            # print(torch.where(_x[:, :, :self.problem.dx:self.problem.dx+self.problem.du] > max_u, 1, 0))
            torch.clamp_(xuz, min=min_x.reshape((N, -1)), max=max_x.reshape((N, -1)))
    def resample(self, xuz):
        xuz = xuz.to(dtype=torch.float32)
        self.problem._preprocess(xuz)
        return super().resample(xuz)
class PositionControlConstrainedSVGDMPC(Constrained_SVGD_MPC):

    def __init__(self, problem, params):
        super().__init__(problem, params)
        self.solver = PositionControlConstrainedSteinTrajOpt(problem, params)
    
class AllegroValveProblem(ConstrainedSVGDProblem):

    def __init__(self, 
                 dx,
                 du,
                 start, 
                 goal, 
                 T, 
                 chain, 
                 valve_location, 
                 valve_type,
                 world_trans,
                 valve_asset_pos,
                 initial_valve_angle=0, 
                 device='cuda:0'):
        super().__init__(start, goal, T, device)
        self.dx, self.du = dx, du
        self.device = device
        self.dt = 0.1
        self.T = T
        self.start = start
        self.goal = goal
        self.K = rbf_kernel
        self.squared_slack = True
        self.compute_hess = True

        self.valve_location = valve_location
        self.initial_valve_angle = initial_valve_angle

        self.chain = chain
        self.index_ee_name = 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link'
        self.thumb_ee_name = 'allegro_hand_oya_finger_3_aftc_base_link'
        self.index_ee_link = chain.frame_to_idx[self.index_ee_name]
        self.thumb_ee_link = chain.frame_to_idx[self.thumb_ee_name]
        self.frame_indices = torch.tensor([self.index_ee_link, self.thumb_ee_link])

        self.grad_kernel = jacrev(rbf_kernel, argnums=0)

        self.world_trans = world_trans.to(device=device)
        self.alpha = 10

        # for honda hand
        index_x_max = torch.tensor([0.47, 1.6099999999, 1.7089999, 1.61799999]) - 0.05
        index_x_min = torch.tensor([-0.47, -0.195999999999, -0.174000000, -0.227]) + 0.05
        thumb_x_max = torch.tensor([1.396, 1.1629999999999, 1.644, 1.71899999]) - 0.05
        thumb_x_min = torch.tensor([0.26, -0.1049999999, -0.1889999999, -0.162]) + 0.05
        self.x_max = torch.cat((index_x_max, thumb_x_max))
        self.x_min = torch.cat((index_x_min, thumb_x_min))

        self.robot_joint_x_max = torch.cat([index_x_max, thumb_x_max])
        self.robot_joint_x_min = torch.cat([index_x_min, thumb_x_min])
        if self.du > 0:
            self.u_max = torch.tensor([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]) / 5
            self.u_min = torch.tensor([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]) / 5

            self.x_max = torch.cat((self.x_max, self.u_max))
            self.x_min = torch.cat((self.x_min, self.u_min))
        self.data = {}

        self.cost = vmap(partial(self._cost, start=self.start, goal=self.goal))
        self.grad_cost = vmap(jacrev(partial(self._cost, start=self.start, goal=self.goal)))
        self.hess_cost = vmap(hessian(partial(self._cost, start=self.start, goal=self.goal)))

    def _cost(self, x, start, goal):
        raise NotImplementedError
    
    def _objective(self, x):
        x = x[:, :, :self.dx + self.du]
        N = x.shape[0]
        J, grad_J, hess_J = self.cost(x), self.grad_cost(x), self.hess_cost(x)

        N = x.shape[0]
        return (self.alpha * J.reshape(N),
                self.alpha * grad_J.reshape(N, -1),
                self.alpha * hess_J.reshape(N, self.T * (self.dx + self.du), self.T * (self.dx + self.du)))
    
    @staticmethod
    def get_rotation_from_normal(normal_vector):
        """
        :param normal_vector: (batch_size, 3)
        :return: (batch_size, 3, 3) rotation matrix with normal vector as the z-axis
        """
        z_axis = normal_vector / torch.norm(normal_vector, dim=1, keepdim=True)
        # y_axis = torch.randn_like(z_axis)
        y_axis = torch.tensor([0.0, 1.0, 0.0],
                              device=normal_vector.device).unsqueeze(0).repeat(normal_vector.shape[0], 1)
        y_axis = y_axis - torch.sum(y_axis * z_axis, dim=1).unsqueeze(-1) * z_axis
        y_axis = y_axis / torch.norm(y_axis, dim=1, keepdim=True)
        x_axis = torch.cross(y_axis, z_axis)
        x_axis = x_axis / torch.norm(x_axis, dim=1, keepdim=True)
        R = torch.stack((x_axis, y_axis, z_axis), dim=2)
        return R
    
    def eval(self, augmented_trajectory):
        N = augmented_trajectory.shape[0]
        augmented_trajectory = augmented_trajectory.clone().reshape(N, self.T, -1)
        x = augmented_trajectory[:, :, :self.dx + self.du]

        # preprocess fingers
        self._preprocess(x)

        # compute objective
        J, grad_J, hess_J = self._objective(x)
        hess_J = None
        grad_J = torch.cat((grad_J.reshape(N, self.T, -1),
                            torch.zeros(N, self.T, self.dz, device=x.device)), dim=2).reshape(N, -1)

        Xk = x.reshape(N, self.T, -1)
        K = self.K(Xk, Xk, None)  # hess_J.mean(dim=0))
        grad_K = -self.grad_kernel(Xk, Xk, None)  # @hess_J.mean(dim=0))
        grad_K = grad_K.reshape(N, N, N, self.T * (self.dx + self.du))
        grad_K = torch.einsum('nmmi->nmi', grad_K)
        grad_K = torch.cat((grad_K.reshape(N, N, self.T, self.dx + self.du),
                            torch.zeros(N, N, self.T, self.dz, device=x.device)), dim=-1)
        grad_K = grad_K.reshape(N, N, -1)
        G, dG, hessG = self.combined_constraints(augmented_trajectory, compute_hess=self.compute_hess)

        # print(J.mean(), G.abs().mean(), G.abs().max())
        
        if hessG is not None:
            hessG.detach_()

        return grad_J.detach(), hess_J, K.detach(), grad_K.detach(), G.detach(), dG.detach(), hessG

    def update(self, start, goal=None, T=None):
        self.start = start
        if goal is not None:
            self.goal = goal

        # update functions that require start
        self.cost = vmap(partial(self._cost, start=self.start, goal=self.goal))
        self.grad_cost = vmap(jacrev(partial(self._cost, start=self.start, goal=self.goal)))
        self.hess_cost = vmap(hessian(partial(self._cost, start=self.start, goal=self.goal)))

        if T is not None:
            self.T = T
            self.dh = self.dz * T
            self.dg = self.dg_per_t * T + self.dg_constant  # terminal contact points, terminal sdf=0, and dynamics

    def get_initial_xu(self, N):
        """
        use delta joint movement to get the initial trajectory
        the action (force at the finger tip) is not used. it is randomly intiailized
        the actual dynamics model is not used
        """

        u = 0.025 * torch.randn(N, self.T, 8, device=self.device)

        x = [self.start.reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            next_q = x[-1][:, :8] + u[:, t]
            x.append(next_q)

        x = torch.stack(x[1:], dim=1)

        # if valve angle in state
        if self.dx == 9:
            theta = torch.linspace(self.start[-1], self.goal.item(), self.T + 1)[:self.T]
            theta = theta.repeat((N, 1)).unsqueeze(-1).to(self.start.device)
            # theta = torch.ones_like(theta) * self.start[-1]
            x = torch.cat((x, theta), dim=-1)

        xu = torch.cat((x, u), dim=2)
        return xu


class AllegroContactProblem(AllegroValveProblem):
    
    def get_constraint_dim(self, T):
        self.dg = 2  # # of fingers, only at the final step
        self.dz = 0  # one friction constraints per finger
        self.dh = self.dz * T  # inequality

    def __init__(self, 
                 dx,
                 du,
                 start, 
                 goal, 
                 T, 
                 chain, 
                 valve_location, 
                 valve_type,
                 world_trans,
                 valve_asset_pos,
                 initial_valve_angle=0, 
                 device='cuda:0'):
        super().__init__(dx=dx, du=du, start=start, goal=goal, T=T, chain=chain, valve_location=valve_location,
                         valve_type=valve_type, world_trans=world_trans, valve_asset_pos=valve_asset_pos,
                         initial_valve_angle=initial_valve_angle, device=device)
        self.get_constraint_dim(T)

        # add collision checking
        # collision check all of the non-finger tip links
        collision_check_oya = ['allegro_hand_oya_finger_link_13',
                               'allegro_hand_oya_finger_link_14',
                               ]
        collision_check_hitosashi = [
            'allegro_hand_hitosashi_finger_finger_link_2',
            'allegro_hand_hitosashi_finger_finger_link_1'
        ]

        # add contact checking
        contact_check_oya = ['allegro_hand_oya_finger_3_aftc_base_link']
        contact_check_hitosashi = ['allegro_hand_hitosashi_finger_finger_0_aftc_base_link']

        if valve_type == 'cuboid':
            asset_valve = get_assets_dir() + '/valve/valve_cuboid.urdf'
        elif valve_type == 'cylinder':
            asset_valve = get_assets_dir() + '/valve/valve_cylinder.urdf'

        chain_valve = pk.build_chain_from_urdf(open(asset_valve).read())
        chain_valve = chain_valve.to(device=device)
        valve_sdf = pv.RobotSDF(chain_valve, path_prefix=get_assets_dir() + '/valve')
        robot_sdf = pv.RobotSDF(chain, path_prefix=get_assets_dir() + '/xela_models')

        scene_trans = world_trans.inverse().compose(
            pk.Transform3d(device=device).translate(valve_asset_pos[0], valve_asset_pos[1], valve_asset_pos[2]))

        self.index_collision_scene = pv.RobotScene(robot_sdf, valve_sdf, scene_trans,
                                                   collision_check_links=collision_check_hitosashi,
                                                   softmin_temp=100.0)
        self.thumb_collision_scene = pv.RobotScene(robot_sdf, valve_sdf, scene_trans,
                                                   collision_check_links=collision_check_oya,
                                                   softmin_temp=100.0)
        # contact checking
        self.index_contact_scene = pv.RobotScene(robot_sdf, valve_sdf, scene_trans,
                                                 collision_check_links=contact_check_hitosashi,
                                                 softmin_temp=1.0e3,
                                                 points_per_link=1000,
                                                 )
        self.thumb_contact_scene = pv.RobotScene(robot_sdf, valve_sdf, scene_trans,
                                                 collision_check_links=contact_check_oya,
                                                 softmin_temp=1.0e3,
                                                 points_per_link=1000)
        # self.index_contact_scene.visualize_robot(partial_to_full_state(self.start[:8]), self.start[-1])
    def _preprocess(self, xu):
        N = xu.shape[0]
        xu = xu.reshape(N, self.T, -1)
        x = xu[:, :, :self.dx]
        # expand to include start
        q = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)
        theta = self.initial_valve_angle * torch.ones(N, self.T + 1, 1, device=self.device)

        self._preprocess_finger(q, theta, self.index_contact_scene, self.index_ee_link, 'index')
        self._preprocess_finger(q, theta, self.thumb_contact_scene, self.thumb_ee_link, 'thumb')

    def _preprocess_finger(self, q, theta, finger_scene, finger_ee_link, finger_name):
        N, _, _ = q.shape

        # reshape to batch across time
        q_b = q.reshape(-1, 8)
        theta_b = theta.reshape(-1, 1)
        full_q = partial_to_full_state(q_b)
        ret_scene = finger_scene.scene_collision_check(full_q, theta_b,
                                                       compute_gradient=True,
                                                       compute_hessian=False)

        link_indices = torch.ones(full_q.shape[0], dtype=torch.int64, device=full_q.device) * finger_ee_link

        # reshape and throw away data for unused fingers
        grad_g_q = ret_scene.get('grad_sdf', None)
        ret_scene['grad_sdf'] = grad_g_q[:, (0, 1, 2, 3, 12, 13, 14, 15)].reshape(N, self.T + 1, 8)

        # contact jacobian
        contact_jacobian = ret_scene.get('contact_jacobian', None)
        contact_jacobian = contact_jacobian.reshape(N, self.T + 1, 3, 16)[:, :, :, (0, 1, 2, 3, 12, 13, 14, 15)]
        ret_scene['contact_jacobian'] = contact_jacobian

        # contact hessian
        contact_hessian = ret_scene.get('contact_hessian', None)
        contact_hessian = contact_hessian.reshape(N, self.T + 1, 3, 16, 16)[:, :, :, (0, 1, 2, 3, 12, 13, 14, 15)]
        contact_hessian = contact_hessian[:, :, :, :, (0, 1, 2, 3, 12, 13, 14, 15)]  # shape (N, T+1, 3, 8, 8)

        # gradient of contact point
        d_contact_loc_dq = ret_scene.get('closest_pt_q_grad', None)
        d_contact_loc_dq = d_contact_loc_dq.reshape(N, self.T + 1, 3, 16)[:, :, :, (0, 1, 2, 3, 12, 13, 14, 15)]
        ret_scene['closest_pt_q_grad'] = d_contact_loc_dq
        ret_scene['contact_hessian'] = contact_hessian

        # gradient of contact normal
        ret_scene['dnormal_dq'] = ret_scene['dnormal_dq'].reshape(N, self.T + 1, 3, 16)[:, :, :,
                                  (0, 1, 2, 3, 12, 13, 14, 15)]

        # Gradient of contact jacobian -- from tests just using contact hessian works best
        # djacobian_dcontact = finger_scene.robot_sdf.chain.calc_djacobian_dtool(full_q, link_indices=link_indices)
        # djacobian_dcontact = djacobian_dcontact.reshape(N, -1, 3, 6, 16)[:, :, :, :3, (0, 1, 2, 3, 12, 13, 14, 15)]
        # dJ_dq = djacobian_dcontact.reshape(N, self.T + 1, 3, 3, 8, 1)
        # dJ_dq = dJ_dq * d_contact_loc_dq.reshape(N, self.T + 1, 3, 1, 1, 8)
        # dJ_dq = dJ_dq.sum(dim=2)  # should be N x T x 3 x 8 x 8
        # dJ_dq = dJ_dq + contact_hessian
        dJ_dq = contact_hessian
        ret_scene['dJ_dq'] = dJ_dq

        self.data[finger_name] = ret_scene
    
    @combine_finger_constraints
    def _contact_constraints(self, xu, finger_name, compute_grads=True, compute_hess=False, terminal=False):
        """
            Computes contact constraints
            constraint that sdf value is zero
        """
        N, T, _ = xu.shape
        # Retrieve pre-processed data
        ret_scene = self.data[finger_name]
        g = ret_scene.get('sdf').reshape(N, T + 1, 1)  # - 0.0025
        grad_g_q = ret_scene.get('grad_sdf', None)
        hess_g_q = ret_scene.get('hess_sdf', None)
        grad_g_theta = ret_scene.get('grad_env_sdf', None)
        hess_g_theta = ret_scene.get('hess_env_sdf', None)

        # Ignore first value, as it is the start state
        g = g[:, 1:].reshape(N, -1)

        # If terminal, only consider last state
        if terminal:
            g = g[:, -1].reshape(N, 1)

        if compute_grads:
            T_range = torch.arange(T, device=xu.device)
            # compute gradient of sdf
            grad_g = torch.zeros(N, T, T, self.dx + self.du, device=xu.device)
            grad_g[:, T_range, T_range, :8] = grad_g_q[:, 1:]
            # is valve in state
            if self.dx == 9:
                grad_g[:, T_range, T_range, 8] = grad_g_theta.reshape(N, T + 1)[:, 1:]
            grad_g = grad_g.reshape(N, -1, T, self.dx + self.du)
            grad_g = grad_g.reshape(N, -1, T * (self.dx + self.du))
            if terminal:
                grad_g = grad_g[:, -1].reshape(N, 1, T * (self.dx + self.du))
        else:
            return g, None, None

        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * (self.dx + self.du), T * (self.dx + self.du), device=self.device)
            return g, grad_g, hess

        return g, grad_g, None
        
    def _cost(self, xu, start, goal):
        state = xu[:, :self.dx]
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it
        action = xu[:, self.dx:]
        action_cost = torch.sum(action ** 2)
        smoothness_cost = 10 * torch.sum((state[1:] - state[:-1]) ** 2)
        return smoothness_cost + 10 * action_cost
    
    def _con_eq(self, xu, compute_grads=True, compute_hess=True):
        N = xu.shape[0]
        T = xu.shape[1]
        g_contact, grad_g_contact, hess_g_contact = self._contact_constraints(xu=xu.reshape(N, T, self.dx + self.du),
                                                                              compute_grads=compute_grads,
                                                                              compute_hess=compute_hess,
                                                                              terminal=True)
        return g_contact, grad_g_contact, hess_g_contact   
    def _con_ineq(self, x, compute_grads=True, compute_hess=True):
        return None, None, None
class AllegroValveTurning(AllegroContactProblem):
    def get_constraint_dim(self, T):
        self.dg = self.dg_per_t * T + self.dg_constant  # terminal contact points, terminal sdf=0, and dynamics
        self.dz = self.friction_polytope_k * 2  # one friction constraints per finger
        self.dh = self.dz * T  # inequality
    
    def __init__(self,
                 start,
                 goal,
                 T,
                 chain,
                 valve_location,
                 valve_type,
                 world_trans,
                 valve_asset_pos,
                 initial_valve_angle=0,
                 friction_coefficient=0.95,
                 device='cuda:0', **kwargs):
        self.friction_polytope_k = 4
        self.dg_per_t = 2 * (1 + 3 + 2)
        self.dg_constant = 0
        super().__init__(dx=9, du=8, start=start, goal=goal, T=T, chain=chain, valve_location=valve_location,
                         valve_type=valve_type, world_trans=world_trans, valve_asset_pos=valve_asset_pos,
                         initial_valve_angle=initial_valve_angle, device=device)
        
        self.friction_coefficient = friction_coefficient
        self.dynamics_constr = vmap(self._dynamics_constr)
        self.grad_dynamics_constr = vmap(jacrev(self._dynamics_constr, argnums=(0, 1, 2, 3, 4)))
        self.friction_constr = vmap(self._friction_constr, randomness='same')
        self.grad_friction_constr = vmap(jacrev(self._friction_constr, argnums=(0, 1, 2)))

        # update x_max with valve angle
        valve_x_max = torch.tensor([10.0 * np.pi])
        valve_x_min = torch.tensor([-10.0 * np.pi])
        self.x_max = torch.cat((self.x_max[:8], valve_x_max, self.x_max[8:]))
        self.x_min = torch.cat((self.x_min[:8], valve_x_min, self.x_min[8:]))

    def _cost(self, xu, start, goal):
        # TODO: check if the addtional term of the smoothness cost and running goal cost is necessary
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it
        
        action = xu[:, self.dx:]  # action dim = 8
        next_q = state[:-1, :-1] + action
        action_cost = torch.sum((state[1:, :-1] - next_q) ** 2)
        action_cost = action_cost + 10 * torch.sum(action ** 2)

        smoothness_cost = 10 * torch.sum((state[1:] - state[:-1]) ** 2)
        smoothness_cost += 50 * torch.sum((state[1:, -1] - state[:-1, -1]) ** 2)

        goal_cost = torch.sum((10 * (state[:, -1] - goal) ** 2),
                              dim=0) + (10 * (state[-1, -1] - goal) ** 2).reshape(-1)

        return smoothness_cost + 10 * action_cost + goal_cost
    
    def _preprocess(self, xu):
        # TODO: put this in the child class instead of the parent class
        N = xu.shape[0]
        xu = xu.reshape(N, self.T, -1)
        x = xu[:, :, :self.dx]
        # expand to include start
        x_expanded = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)

        q = x_expanded[:, :, :8]
        theta = x_expanded[:, :, 8].unsqueeze(-1)
        self._preprocess_finger(q, theta, self.index_contact_scene, self.index_ee_link, 'index')
        self._preprocess_finger(q, theta, self.thumb_contact_scene, self.thumb_ee_link, 'thumb')

    def get_friction_polytope(self):
        """
        :param k: the number of faces of the friction cone
        :return: a list of normal vectors of the faces of the friction cone
        """
        normal_vectors = []
        for i in range(self.friction_polytope_k):
            theta = 2 * np.pi * i / self.friction_polytope_k
            normal_vector = torch.tensor([np.cos(theta), np.sin(theta), self.friction_coefficient]).to(
                device=self.device,
                dtype=torch.float32)
            normal_vectors.append(normal_vector)
        normal_vectors = torch.stack(normal_vectors, dim=0)
        return normal_vectors
    
    @combine_finger_constraints
    def _valve_kinematics_constraint(self, xu, finger_name, compute_grads=True, compute_hess=False):
        """
            Computes on the kinematics of the valve and the finger being consistant
        """
        x = xu[:, :, :self.dx]
        N, T, _ = x.shape

        # we want to add the start state to x, this x is now T + 1
        x = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)

        # Retrieve pre-processed data
        ret_scene = self.data[finger_name]
        contact_jacobian = ret_scene.get('contact_jacobian', None)
        contact_loc = ret_scene.get('closest_pt_world', None)
        d_contact_loc_dq = ret_scene.get('closest_pt_q_grad', None)
        dJ_dq = ret_scene.get('dJ_dq', None)

        # approximate q dot and theta dot
        dq = (x[:, 1:, :8] - x[:, :-1, :8])
        dtheta = (x[:, 1:, 8] - x[:, :-1, 8])

        # angular velocity of the valve
        valve_omega = torch.stack((torch.zeros_like(dtheta),
                                   dtheta,
                                   torch.zeros_like(dtheta)), -1)  # should be N x T-1 x 3

        # compute robot contact point velocity
        contact_point_v = (contact_jacobian[:, :-1] @ dq.reshape(N, T, 8, 1)).squeeze(-1)  # should be N x T x 3

        # compute valve contact point velocity
        valve_robot_frame = self.world_trans.inverse().transform_points(self.valve_location.reshape(1, 3))
        contact_point_r_valve = contact_loc.reshape(N, T + 1, 3) - valve_robot_frame.reshape(1, 1, 3)
        valve_omega_robot_frame = self.world_trans.inverse().transform_normals(valve_omega)
        object_contact_point_v = torch.cross(valve_omega_robot_frame, contact_point_r_valve[:, :-1])

        # kinematics constraint, should be 3-dimensional
        # we actually ended up computing T+1 contact constraints, but start state is fixed so we throw that away
        g = (contact_point_v - object_contact_point_v).reshape(N, -1)

        if compute_grads:
            T_range = torch.arange(T, device=x.device)
            T_range_minus = torch.arange(T - 1, device=x.device)
            T_range_plus = torch.arange(1, T, device=x.device)

            # Compute gradient w.r.t q
            dcontact_v_dq = (dJ_dq[:, 1:] @ dq.reshape(N, T, 1, 8, 1)).squeeze(-1) - contact_jacobian[:, 1:]
            tmp = torch.cross(d_contact_loc_dq[:, 1:], valve_omega.reshape(N, T, 3, 1), dim=2)  # N x T x 3 x 8
            dg_dq = dcontact_v_dq - tmp

            # Compute gradient w.r.t valve angle
            d_omega_dtheta = torch.stack((torch.zeros_like(dtheta),
                                          torch.ones_like(dtheta),
                                          torch.zeros_like(dtheta)), dim=-1)  # N x T x 3
            d_omega_dtheta = self.world_trans.inverse().transform_normals(d_omega_dtheta)
            dg_dtheta = torch.cross(d_omega_dtheta, contact_point_r_valve[:, :-1], dim=-1)  # N x T x 3

            # assemble gradients into a single (sparse) tensor
            grad_g = torch.zeros((N, T, T, 3, self.dx + self.du), device=x.device)
            grad_g[:, T_range_plus, T_range_minus, :, :8] = dg_dq[:, 1:]
            grad_g[:, T_range_plus, T_range_minus, :, 8] = dg_dtheta[:, 1:]
            grad_g[:, T_range, T_range, :, :8] = contact_jacobian[:, :-1]
            grad_g[:, T_range, T_range, :, 8] = -dg_dtheta
            grad_g = grad_g.permute(0, 1, 3, 2, 4).reshape(N, -1, T * (self.dx + self.du))
        else:
            return g, None, None

        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * (self.dx + self.du), T * (self.dx + self.du), device=self.device)
            return g, grad_g, hess

        return g, grad_g, None

    def _dynamics_constr(self, q, u, next_q, contact_jacobian, contact_normal):
        # this will be vmapped, so takes in a 3 vector and a 3 x 8 jacobian and a dq vector
        dq = next_q - q
        contact_v = (contact_jacobian @ dq.unsqueeze(-1)).squeeze(-1)  # should be 3 vector
        # from commanded
        contact_v_u = (contact_jacobian @ u.unsqueeze(-1)).squeeze(-1)  # should be 3 vector

        # convert to world frame
        contact_v_world = self.world_trans.transform_normals(contact_v.unsqueeze(0)).squeeze(0)
        contact_v_u_world = self.world_trans.transform_normals(contact_v_u.unsqueeze(0)).squeeze(0)
        contact_normal_world = self.world_trans.transform_normals(contact_normal.unsqueeze(0)).squeeze(0)

        # compute projection onto normal
        normal_projection = contact_normal_world.unsqueeze(-1) @ contact_normal_world.unsqueeze(-2)

        # must find a lower dimensional representation of the constraint to avoid numerical issues
        # TODO for now hand coded, but need to find a better solution
        R = self.get_rotation_from_normal(contact_normal_world.unsqueeze(0)).squeeze(0).detach().permute(0, 1)
        R = R[:2]
        # compute contact v tangential to surface
        contact_v_tan = contact_v_world - (normal_projection @ contact_v_world.unsqueeze(-1)).squeeze(-1)
        contact_v_u_tan = contact_v_u_world - (normal_projection @ contact_v_u_world.unsqueeze(-1)).squeeze(-1)

        # should have same tangential components
        # TODO: this constraint value is super small
        return (R @ (contact_v_tan - contact_v_u_tan).unsqueeze(-1)).squeeze(-1)
    
    @combine_finger_constraints
    def _dynamics_constraints(self, xu, finger_name, compute_grads=True, compute_hess=False):
        """ Computes dynamics constraints
            constraint that sdf value is zero
            also constraint on contact kinematics to get the valve dynamics
        """
        x = xu[:, :, :self.dx]
        N, T, _ = x.shape

        # we want to add the start state to x, this x is now T + 1
        x = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)

        q = x[:, :-1, :8]
        next_q = x[:, 1:, :8]
        u = xu[:, :, self.dx:]
        contact_jac = self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, 8)[:, :-1]
        contact_normal = self.data[finger_name]['contact_normal'].reshape(N, T + 1, 3)[:, :-1]
        dnormal_dq = self.data[finger_name]['dnormal_dq'].reshape(N, T + 1, 3, 8)[:, :-1]
        dnormal_dtheta = self.data[finger_name]['dnormal_denv_q'].reshape(N, T + 1, 3, 1)[:, :-1]

        g = self.dynamics_constr(q.reshape(-1, 8), u.reshape(-1, 8), next_q.reshape(-1, 8),
                                 contact_jac.reshape(-1, 3, 8),
                                 contact_normal.reshape(-1, 3)).reshape(N, T, -1)

        if compute_grads:
            T_range = torch.arange(T, device=x.device)
            T_plus = torch.arange(1, T, device=x.device)
            T_minus = torch.arange(T - 1, device=x.device)
            grad_g = torch.zeros(N, g.shape[2], T, T, self.dx + self.du, device=self.device)
            # dnormal_dq = torch.zeros(N, T, 3, 8, device=self.device)  # assume zero SDF hessian
            djac_dq = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, 8, 8)[:, :-1]
            dg_dq, dg_du, dg_dnext_q, dg_djac, dg_dnormal = self.grad_dynamics_constr(q.reshape(-1, 8),
                                                                                      u.reshape(-1, 8),
                                                                                      next_q.reshape(-1, 8),
                                                                                      contact_jac.reshape(-1, 3, 8),
                                                                                      contact_normal.reshape(-1, 3))

            dg_dq = dg_dq.reshape(N, T, g.shape[2], -1) + dg_dnormal.reshape(N, T, g.shape[2], -1) @ dnormal_dq  #
            dg_dq = dg_dq + dg_djac.reshape(N, T, g.shape[2], -1) @ djac_dq.reshape(N, T, -1, 8)
            dg_dtheta = dg_dnormal.reshape(N, T, g.shape[2], -1) @ dnormal_dtheta

            grad_g[:, :, T_plus, T_minus, :8] = dg_dq[:, 1:].transpose(1, 2)  # first q is the start
            grad_g[:, :, T_range, T_range, self.dx:] = dg_du.reshape(N, T, -1, self.du).transpose(1, 2)
            grad_g[:, :, T_plus, T_minus, 8] = dg_dtheta[:, 1:].squeeze(-1).transpose(1, 2)
            grad_g[:, :, T_range, T_range, :8] = dg_dnext_q.reshape(N, T, -1, 8).transpose(1, 2)
            grad_g = grad_g.transpose(1, 2)
        else:
            return g.reshape(N, -1), None, None

        if compute_hess:
            hess_g = torch.zeros(N, T * 2,
                                 T * (self.dx + self.du),
                                 T * (self.dx + self.du), device=self.device)

            return g.reshape(N, -1), grad_g.reshape(N, -1, T * (self.dx + self.du)), hess_g

        return g.reshape(N, -1), grad_g.reshape(N, -1, T * (self.dx + self.du)), None
    
    def _friction_constr(self, dq, contact_normal, contact_jacobian):
        # this will be vmapped, so takes in a 3 vector and a 3 x 8 jacobian and a dq vector

        # compute the force in robot frame
        # force = (torch.linalg.lstsq(contact_jacobian.transpose(-1, -2),
        #                            dq.unsqueeze(-1))).solution.squeeze(-1)
        # force_world_frame = self.world_trans.transform_normals(force.unsqueeze(0)).squeeze(0)
        # transform contact normal to world frame
        contact_normal_world = self.world_trans.transform_normals(contact_normal.unsqueeze(0)).squeeze(0)

        # transform force to contact frame
        R = self.get_rotation_from_normal(contact_normal_world.unsqueeze(0)).squeeze(0)
        # force_contact_frame = R.transpose(0, 1) @ force_world_frame.unsqueeze(-1)
        B = self.get_friction_polytope().detach()

        # compute contact point velocity in contact frame
        contact_v_contact_frame = R.transpose(0, 1) @ self.world_trans.transform_normals(
            (contact_jacobian @ dq).unsqueeze(0)).squeeze(0)

        # TODO: there are two different ways of doing a friction cone
        # Linearized friction cone - but based on the contact point velocity
        return B @ contact_v_contact_frame

    @combine_finger_constraints
    def _friction_constraint(self, xu, finger_name, compute_grads=True, compute_hess=False):

        # assume access to class member variables which have already done some of the computation
        N, T, d = xu.shape
        u = xu[:, :, self.dx:]

        # u is the delta q commanded
        # retrieved cached values
        contact_jac = self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, 8)[:, :-1]
        contact_normal = self.data[finger_name]['contact_normal'].reshape(N, T + 1, 3)[:, :-1]
        dnormal_dq = self.data[finger_name]['dnormal_dq'].reshape(N, T + 1, 3, 8)[:, :-1]
        dnormal_dtheta = self.data[finger_name]['dnormal_denv_q'].reshape(N, T + 1, 3, 1)[:, :-1]

        # compute constraint value
        h = self.friction_constr(u.reshape(-1, 8),
                                 contact_normal.reshape(-1, 3),
                                 contact_jac.reshape(-1, 3, 8)).reshape(N, -1)

        # compute the gradient
        if compute_grads:
            dh_du, dh_dnormal, dh_djac = self.grad_friction_constr(u.reshape(-1, 8),
                                                                   contact_normal.reshape(-1, 3),
                                                                   contact_jac.reshape(-1, 3, 8))

            djac_dq = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, 8, 8)[:, :-1]

            dh = dh_dnormal.shape[1]
            dh_dq = dh_dnormal.reshape(N, T, dh, -1) @ dnormal_dq
            dh_dq = dh_dq + dh_djac.reshape(N, T, dh, -1) @ djac_dq.reshape(N, T, -1, 8)
            dh_dtheta = dh_dnormal.reshape(N, T, dh, -1) @ dnormal_dtheta
            grad_h = torch.zeros(N, dh, T, T, d, device=self.device)
            T_range = torch.arange(T, device=self.device)
            T_range_minus = torch.arange(T - 1, device=self.device)
            T_range_plus = torch.arange(1, T, device=self.device)
            grad_h[:, :, T_range_plus, T_range_minus, :8] = dh_dq[:, 1:].transpose(1, 2)
            grad_h[:, :, T_range_plus, T_range_minus, 8] = dh_dtheta[:, 1:].squeeze(-1).transpose(1, 2)
            grad_h[:, :, T_range, T_range, self.dx:] = dh_du.reshape(N, T, dh, self.du).transpose(1, 2)
            grad_h = grad_h.transpose(1, 2).reshape(N, -1, T * d)
        else:
            return h, None, None

        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], T * d, T * d, device=self.device)
            return h, grad_h, hess_h

        return h, grad_h, None

    def _con_ineq(self, xu, compute_grads=True, compute_hess=False):
        N = xu.shape[0]
        T = xu.shape[1]

        h, grad_h, hess_h = self._friction_constraint(
            xu=xu.reshape(-1, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)

        h = h.reshape(N, -1)
        if compute_grads:
            grad_h = grad_h.reshape(N, -1, self.T * (self.dx + self.du))
        else:
            return h, None, None
        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], self.T * (self.dx + self.du), self.T * (self.dx + self.du),
                                 device=self.device)
            return h, grad_h, hess_h
        return h, grad_h, None

    def _con_eq(self, xu, compute_grads=True, compute_hess=False):
        N = xu.shape[0]
        T = xu.shape[1]
        g_contact, grad_g_contact, hess_g_contact = self._contact_constraints(xu=xu.reshape(N, T, self.dx + self.du),
                                                                              compute_grads=compute_grads,
                                                                              compute_hess=compute_hess)
        g_dynamics, grad_g_dynamics, hess_g_dynamics = self._dynamics_constraints(
            xu=xu.reshape(N, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)

        g_valve, grad_g_valve, hess_g_valve = self._valve_kinematics_constraint(
            xu=xu.reshape(N, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)

        g_contact = torch.cat((g_contact, g_dynamics, g_valve), dim=1)

        if grad_g_contact is not None:
            grad_g_contact = torch.cat((grad_g_contact, grad_g_dynamics, grad_g_valve), dim=1)
        if hess_g_contact is not None:
            hess_g_contact = torch.cat((hess_g_contact, hess_g_dynamics, hess_g_valve), dim=1)

        return g_contact, grad_g_contact, hess_g_contact
    
    # def _y_axis_constr(self, xu):
    #     N = xu.shape[0]
    #     T = xu.shape[1]
    #     q = xu[:, :, :9]
    #     finger_names = [index_ee_name, thumb_ee_name]
    #     constraint = {}
    #     fk_dict = chain.forward_kinematics(partial_to_full_state(q[:, :, :8].reshape(-1, 8)),
    #                                     frame_indices=frame_indices)  # pytorch_kinematics only supprts one additional dim

    #     for finger_name in finger_names:
    #         m = world_trans.compose(fk_dict[finger_name])
    #         points_finger_frame = torch.tensor([0.00, 0.03, 0.00], device=xu.device).unsqueeze(0)
    #         # it is just the thrid column of the rotation matrix
    #         ee_p = m.transform_points(points_finger_frame).reshape((N, T, 3))
    #         "2nd constraint y range of the finger tip should be within a range"
    #         constraint_y_1 = -(valve_location[1] - ee_p[:, :, 1]) + 0.02 # do not consider the current time step
    #         constraint_y_2 = (valve_location[1] - ee_p[:, :, 1]) - 0.2
    #         constraint[f'{finger_name}_y_range_1'] = constraint_y_1
    #         constraint[f'{finger_name}_y_range_2'] = constraint_y_2
    #     return torch.cat(list(constraint.values()), dim=-1)
    
    # def y_axis_constraint(self, xu, compute_grads=True, compute_hess=False):
    #     N = xu.shape[0]
    #     T = xu.shape[1]
    #     h = self._y_axis_constr(xu)
    #     grad_h, hess_h = None, None
    #     if compute_grads:
    #         grad_h = self.grad_y_axis_constraint(xu)
    #         grad_h = torch.stack([grad_h[i, :, i] for i in range(len(grad_h))], dim=0)  # data from different batches should not affect each other
    #     if compute_hess:
    #         hess_h = torch.zeros(grad_h.shape + grad_h.shape[2:], device=self.device)
    #     return h, grad_h, hess_h


class AllegroValveRegrasping(AllegroContactProblem):
    def get_constraint_dim(self, T):
        self.dz = 2  # collision constraint per finger
        self.dg = self.dg_per_t * T + self.dg_constant  # terminal contact points, terminal sdf=0, and dynamics
        self.dh = self.dz * T

    def __init__(self,
                 start,
                 goal,
                 T,
                 chain,
                 valve_location,
                 valve_type,
                 world_trans,
                 valve_asset_pos,
                 initial_valve_angle=0,
                 device='cuda:0'):
        dx = 8
        du = 8
        self.dg_per_t = 8
        self.dg_constant = 6
        super().__init__(dx=dx, du=du, start=start, goal=goal, T=T, chain=chain, valve_location=valve_location,
                         valve_type=valve_type, world_trans=world_trans, valve_asset_pos=valve_asset_pos,
                         initial_valve_angle=initial_valve_angle, device=device)

        

    def _cost(self, xu, start, goal):
        state = xu[:, :self.dx]
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it
        action = xu[:, self.dx:]
        action_cost = torch.sum(action ** 2)
        smoothness_cost = 10 * torch.sum((state[1:] - state[:-1]) ** 2)
        return smoothness_cost + 10 * action_cost

    def _dynamics_constraints(self, xu, compute_grads=True, compute_hess=False):
        N, T, _ = xu.shape
        x = xu[:, :, :self.dx]
        # add start
        x = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)
        next_x = x[:, 1:]
        x = x[:, :-1]
        u = xu[:, :, self.dx:]

        # compute constraint value - just a linear constraint
        g = (next_x - x - u).reshape(N, -1)
        if compute_grads:
            # all gradients are just multiples of identity
            eye = torch.eye(self.dx, device=x.device).reshape(1, 1, self.dx, self.dx).repeat(N, T, 1, 1)
            eye = eye.permute(0, 2, 1, 3)
            grad_g = torch.zeros(N, self.dx, T, T, self.dx + self.du, device=x.device)
            T_range = torch.arange(T, device=x.device)
            T_range_minus = torch.arange(T - 1, device=x.device)
            T_range_plus = torch.arange(1, T, device=x.device)
            grad_g[:, :, T_range, T_range, :self.dx] = eye
            grad_g[:, :, T_range, T_range, self.dx:] = -eye
            grad_g[:, :, T_range_plus, T_range_minus, :self.dx] = -eye[:, :, 1:]
            grad_g = grad_g.permute(0, 2, 1, 3, 4).reshape(N, -1, T * (self.dx + self.du))
        else:
            return g, None, None
        if compute_hess:
            hess_g = torch.zeros(N, g.shape[1], T * (self.dx + self.du), T * (self.dx + self.du), device=self.device)
            return g, grad_g, hess_g
        return g, grad_g, None

    @combine_finger_constraints
    def _contact_target_constraints(self, xu, finger_name, compute_grads=True, compute_hess=False):
        """
        Constraint that contact point is equal to target value
        """
        N, T, _ = xu.shape

        # Retrieve pre-processed data
        ret_scene = self.data[finger_name]
        contact_loc = ret_scene.get('closest_pt_world', None)
        grad_contact_loc = ret_scene.get('closest_pt_q_grad', None)[:, -1]
        contact_loc = contact_loc.reshape(N, T + 1, 3)[:, -1]

        g = (contact_loc - self.goal[finger_name]).reshape(N, 3)
        if compute_grads:
            grad_g = torch.zeros(N, 3, T, (self.dx + self.du), device=self.device)
            grad_g[:, :, -1, :8] = grad_contact_loc.reshape(N, 3, -1)
            grad_g = grad_g.reshape(N, 3, -1)
        else:
            return g, None, None
        if compute_hess:
            hess_g = torch.zeros(N, 3, T * (self.dx + self.du), T * (self.dx + self.du), device=self.device)
            return g, grad_g, hess_g
        return g, grad_g, None

    def _con_ineq(self, xu, compute_grads=True, compute_hess=False):
        N = xu.shape[0]
        T = xu.shape[1]
        # contact constraints are inequality
        h_contact, grad_h_contact, hess_h_contact = self._contact_constraints(xu=xu.reshape(N, T, self.dx + self.du),
                                                                              compute_grads=compute_grads,
                                                                              compute_hess=compute_hess)

        # reverse sign, add a small epsilon to make it strictly less than for all but final time_step
        eps = torch.zeros_like(h_contact)
        eps[:, :-1] = 3e-3
        h_contact = -h_contact + eps
        if grad_h_contact is not None:
            grad_h_contact = -grad_h_contact
        if hess_h_contact is not None:
            hess_h_contact = -hess_h_contact

        return h_contact, grad_h_contact, hess_h_contact

    def _con_eq(self, xu, compute_grads=True, compute_hess=False):
        N = xu.shape[0]
        T = xu.shape[1]

        g_dynamics, grad_g_dynamics, hess_g_dynamics = self._dynamics_constraints(
            xu=xu.reshape(N, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)

        g_contact_target, grad_g_contact_target, hess_g_contact_target = self._contact_target_constraints(
            xu=xu.reshape(N, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)

        g = torch.cat((g_dynamics, g_contact_target), dim=1)
        
        grad_g, hess_g = None, None
        if compute_grads:
            grad_g = torch.cat((grad_g_dynamics, grad_g_contact_target), dim=1)
        if compute_hess:
            hess_g = torch.cat((hess_g_dynamics, hess_g_contact_target), dim=1)

        return g, grad_g, hess_g

    def update(self, start, valve_angle=0.0, goal=None, T=None):
        self.initial_valve_angle = valve_angle
        super().update(start, goal, T)

    def get_initial_xu(self, N):
        xu = super().get_initial_xu(N)
        if self.du == 0:
            return xu[:, :, :self.dx]
        return xu
    
def do_trial(env, params, fpath):
    "only turn the valve once"
    state = env.get_state()
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None

    start = state['q'].reshape(9).to(device=params['device'])
    # start = torch.cat((state['q'].reshape(10), torch.zeros(1).to(state['q'].device))).to(device=params['device'])
    if params['controller'] == 'csvgd':
        pregrasp_problem = AllegroContactProblem(
            dx=8,
            du=8,
            start=start[:8],
            goal=params['valve_goal'] * 0,
            T=4,
            chain=params['chain'],
            device=params['device'],
            valve_asset_pos=env.valve_pose,
            valve_location=params['valve_location'],
            valve_type=params['valve_type'],
            world_trans=env.world_trans
        )

        pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, params)

        turn_problem = AllegroValveTurning(
            start=start,
            goal=params['valve_goal'],
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            valve_asset_pos=env.valve_pose,
            valve_location=params['valve_location'],
            valve_type=params['valve_type'],
            friction_coefficient=params['friction_coefficient'],
            world_trans=env.world_trans
        )

        turn_planner = PositionControlConstrainedSVGDMPC(turn_problem, params)
    else:
        raise ValueError('Invalid controller')
    
    # first we move the hand to grasp the valve
    start = state['q'].reshape(9).to(device=params['device'])
    best_traj, _ = pregrasp_planner.step(start[:8])

    # we will just execute this open loop
    for x in best_traj[:, :8]:
        env.step(x.reshape(-1, 8).to(device=env.device))

    actual_trajectory = []
    duration = 0

    # debug: plot the thumb traj
    fig = plt.figure()
    ax_index = fig.add_subplot(122, projection='3d')
    ax_thumb = fig.add_subplot(121, projection='3d')
    ax_index.set_title('index')
    ax_thumb.set_title('thumb')
    axes = [ax_index, ax_thumb]
    for i, ax in enumerate(axes):
        axes[i].set_aspect('equal')
        axes[i].set_xlabel('x', labelpad=20)
        axes[i].set_ylabel('y', labelpad=20)
        axes[i].set_zlabel('z', labelpad=20)
        axes[i].set_xlim3d(0.8, 0.87)
        axes[i].set_ylim3d(0.52, 0.58)
        axes[i].set_zlim3d(1.36, 1.46)
    # ax.set_xlim((0.8, 0.8))
    # ax.set_ylim((0.6, 0.7))
    # ax.set_zlim((1.35, 1.45))
    thumb_traj_history = []
    index_traj_history = []
    state = env.get_state()
    start = state['q'].reshape(9).to(device=params['device'])
    thumb_ee = state2ee_pos(start[:8], turn_problem.thumb_ee_name)
    thumb_traj_history.append(thumb_ee.detach().cpu().numpy())
    index_ee = state2ee_pos(start[:8], turn_problem.index_ee_name)
    index_traj_history.append(index_ee.detach().cpu().numpy())

    info_list = []

    for k in range(params['num_steps']):
        state = env.get_state()
        start = state['q'].reshape(9).to(device=params['device'])

        actual_trajectory.append(state['q'].reshape(9).clone())
        start_time = time.time()
        best_traj, trajectories = turn_planner.step(start)

        print(f"solve time: {time.time() - start_time}")
        # add trajectory lines to sim
        if params['hardware']:
            add_trajectories_hardware(trajectories, best_traj, chain, axes)
        else:
            add_trajectories(trajectories, best_traj, chain, axes)

        # process the action
        ## end effector force to torque
        x = best_traj[0, :turn_problem.dx+turn_problem.du]
        x = x.reshape(1, turn_problem.dx+turn_problem.du)
        action = x[:, turn_problem.dx:turn_problem.dx+turn_problem.du].to(device=env.device)
        if params['joint_friction'] > 0:
            action = action + params['joint_friction'] / env.joint_stiffness * torch.sign(action)
        action = action + start.unsqueeze(0)[:, :8] # NOTE: this is required since we define action as delta action
        # action = best_traj[0, :8]
        # action[:, 4:] = 0
        env.step(action)
        # if params['hardware']:
        #     # ros_node.apply_action(action[0].detach().cpu().numpy())
        #     ros_node.apply_action(partial_to_full_state(action[0]).detach().cpu().numpy())
        turn_problem._preprocess(best_traj.unsqueeze(0))
        
        print(turn_problem.thumb_contact_scene.scene_collision_check(partial_to_full_state(x[:, :8]), x[:, 8],
                                                                compute_gradient=False, compute_hessian=False))
        # distance2surface = torch.sqrt((best_traj_ee[:, 2] - valve_location[2].unsqueeze(0)) ** 2 + (best_traj_ee[:, 0] - valve_location[0].unsqueeze(0))**2)
        distance2goal = (params['valve_goal'].cpu() - env.get_state()['q'][:, -1].cpu()).detach().cpu().item()
        print(distance2goal)
        info_list.append({
                        #   'distance': distance, 
                          'distance2goal': distance2goal, 
                        })

        gym.clear_lines(viewer)
        # for debugging
        state = env.get_state()
        start = state['q'].reshape(9).to(device=params['device'])
        thumb_ee = state2ee_pos(start[:8], turn_problem.thumb_ee_name)
        thumb_traj_history.append(thumb_ee.detach().cpu().numpy())
        temp_for_plot = np.stack(thumb_traj_history, axis=0)
        if k >= 2:
            ax_thumb.plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'gray', label='actual')
        index_ee = state2ee_pos(start[:8], turn_problem.index_ee_name)
        index_traj_history.append(index_ee.detach().cpu().numpy())
        temp_for_plot = np.stack(index_traj_history, axis=0)
        if k >= 2:
            ax_index.plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'gray', label='actual')
    with open(f'{fpath.resolve()}/info.pkl', 'wb') as f:
        pkl.dump(info_list, f)
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
      if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)
    fig.tight_layout()
    fig.legend(newHandles, newLabels, loc='lower center', ncol=3)
    # plt.savefig(f'{fpath.resolve()}/traj.png')
    # plt.close()
    plt.show()




    state = env.get_state()
    state = state['q'].reshape(9).to(device=params['device'])
    actual_trajectory.append(state.clone())
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 9)
    turn_problem.T = actual_trajectory.shape[0]
    # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    final_distance_to_goal = (actual_trajectory[:, -1] - params['valve_goal']).abs

    print(f'Controller: {params["controller"]} Final distance to goal: {torch.min(final_distance_to_goal)}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
            #  constr=constraint_val.cpu().numpy(),
             d2goal=final_distance_to_goal.cpu().numpy())
    return torch.min(final_distance_to_goal).cpu().numpy()


def add_trajectories(trajectories, best_traj, chain, axes=None):
    M = len(trajectories)
    if M > 0:
        initial_state = env.get_state()['q'][:, :8]
        all_state = torch.cat((initial_state, best_traj[:-1, :8]), dim=0)
        desired_state = all_state + best_traj[:, 9:17]

        desired_index_best_traj_ee = state2ee_pos(desired_state, index_ee_name)
        desired_thumb_best_traj_ee = state2ee_pos(desired_state, thumb_ee_name)

        index_best_traj_ee = state2ee_pos(best_traj[:, :8], index_ee_name)
        thumb_best_traj_ee = state2ee_pos(best_traj[:, :8], thumb_ee_name)

        traj_line_colors = np.random.random((3, M)).astype(np.float32)
        thumb_colors = np.array([0, 1, 0]).astype(np.float32)
        index_colors = np.array([0, 0, 1]).astype(np.float32)
        force_colors = np.array([0, 1, 1]).astype(np.float32)
        
        for e in env.envs:
            T = best_traj.shape[0]
            for t in range(T):
                if t == 0:
                    initial_thumb_ee = state2ee_pos(initial_state, thumb_ee_name)
                    thumb_state_traj = torch.stack((initial_thumb_ee, thumb_best_traj_ee[0]), dim=0).cpu().numpy()
                    thumb_action_traj = torch.stack((initial_thumb_ee, desired_thumb_best_traj_ee[0]), dim=0).cpu().numpy()
                    axes[1].plot3D(thumb_state_traj[:, 0], thumb_state_traj[:, 1], thumb_state_traj[:, 2], 'blue', label='desired next state')
                    axes[1].plot3D(thumb_action_traj[:, 0], thumb_action_traj[:, 1], thumb_action_traj[:, 2], 'green', label='raw commanded position')
                    initial_index_ee = state2ee_pos(initial_state, index_ee_name)
                    index_state_traj = torch.stack((initial_index_ee, index_best_traj_ee[0]), dim=0).cpu().numpy()
                    index_action_traj = torch.stack((initial_index_ee, desired_index_best_traj_ee[0]), dim=0).cpu().numpy()
                    axes[0].plot3D(index_state_traj[:, 0], index_state_traj[:, 1], index_state_traj[:, 2], 'blue', label='desired next state')
                    axes[0].plot3D(index_action_traj[:, 0], index_action_traj[:, 1], index_action_traj[:, 2], 'green', label='raw commanded position')
                else:
                    thumb_state_traj = torch.stack((thumb_best_traj_ee[t - 1, :3], thumb_best_traj_ee[t, :3]), dim=0).cpu().numpy()
                    thumb_action_traj = torch.stack((thumb_best_traj_ee[t - 1, :3], desired_thumb_best_traj_ee[t, :3]), dim=0).cpu().numpy()
                    index_state_traj = torch.stack((index_best_traj_ee[t - 1, :3], index_best_traj_ee[t, :3]), dim=0).cpu().numpy()
                    index_action_traj = torch.stack((index_best_traj_ee[t - 1, :3], desired_index_best_traj_ee[t, :3]), dim=0).cpu().numpy()
                
                thumb_state_traj = thumb_state_traj.reshape(2, 3)
                thumb_action_traj = thumb_action_traj.reshape(2, 3)
                index_state_traj = index_state_traj.reshape(2, 3)
                index_action_traj = index_action_traj.reshape(2, 3)
                
                gym.add_lines(viewer, e, 1, index_state_traj, index_colors)
                gym.add_lines(viewer, e, 1, thumb_state_traj, thumb_colors)
                gym.add_lines(viewer, e, 1, index_action_traj, np.array([0, 1, 1]).astype(np.float32))
                gym.add_lines(viewer, e, 1, thumb_action_traj, np.array([1, 1, 1]).astype(np.float32))
                # gym.add_lines(viewer, e, M, p, traj_line_colors)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)

def add_trajectories_hardware(trajectories, best_traj, chain, axes=None):
    M = len(trajectories)
    if M > 0:
        initial_state = env.get_state()['q'][:, :8]
        all_state = torch.cat((initial_state, best_traj[:-1, :8]), dim=0)
        desired_state = all_state + best_traj[:, 9:17]

        desired_index_best_traj_ee = state2ee_pos(desired_state, index_ee_name)
        desired_thumb_best_traj_ee = state2ee_pos(desired_state, thumb_ee_name)

        index_best_traj_ee = state2ee_pos(best_traj[:, :8], index_ee_name)
        thumb_best_traj_ee = state2ee_pos(best_traj[:, :8], thumb_ee_name)

        initial_thumb_ee = state2ee_pos(initial_state, thumb_ee_name)
        thumb_state_traj = torch.stack((initial_thumb_ee, thumb_best_traj_ee[0]), dim=0).cpu().numpy()
        thumb_action_traj = torch.stack((initial_thumb_ee, desired_thumb_best_traj_ee[0]), dim=0).cpu().numpy()
        # axes[1].plot3D(thumb_state_traj[:, 0], thumb_state_traj[:, 1], thumb_state_traj[:, 2], 'blue', label='desired next state')
        axes[1].plot3D(thumb_action_traj[:, 0], thumb_action_traj[:, 1], thumb_action_traj[:, 2], 'green', label='raw commanded position')
        initial_index_ee = state2ee_pos(initial_state, index_ee_name)
        index_state_traj = torch.stack((initial_index_ee, index_best_traj_ee[0]), dim=0).cpu().numpy()
        index_action_traj = torch.stack((initial_index_ee, desired_index_best_traj_ee[0]), dim=0).cpu().numpy()
        # axes[0].plot3D(index_state_traj[:, 0], index_state_traj[:, 1], index_state_traj[:, 2], 'blue', label='desired next state')
        axes[0].plot3D(index_action_traj[:, 0], index_action_traj[:, 1], index_action_traj[:, 2], 'green', label='raw commanded position')


if __name__ == "__main__":
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro.yaml').read_text())
    from tqdm import tqdm

    env = AllegroValveTurningEnv(1, control_mode='joint_impedance',
                                 use_cartesian_controller=False,
                                 viewer=True,
                                 steps_per_action=60,
                                 valve_velocity_in_state=False,
                                 friction_coefficient=1.0,
                                 device=config['sim_device'],
                                 valve=config['valve_type'],
                                 video_save_path=img_save_dir,
                                 configuration='screw_driver')

    sim, gym, viewer = env.get_sim()

    """
    state = env.get_state()
    ee_pos, ee_ori = state['ee_pos'], state['ee_ori']
    try:
        while True:
            start = torch.cat((ee_pos, ee_ori), dim=-1).reshape(1, 7)
            env.step(start)
            print('waiting for you to finish camera adjustment, ctrl-c when done')
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    """

    if config['hardware']:
        sim_env = env
        from hardware.hardware_env import HardwareEnv
        env = HardwareEnv(sim_env.default_dof_pos, finger_list=['index', 'thumb'])
        env.world_trans = sim_env.world_trans
        env.joint_stiffness = sim_env.joint_stiffness
        env.device = sim_env.device
        env.valve_pose = sim_env.valve_pose

        del sim_env



    results = {}

    # set up the kinematic chain
    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    index_ee_name = 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link'
    thumb_ee_name = 'allegro_hand_oya_finger_3_aftc_base_link'

    # combined chain
    chain = pk.build_chain_from_urdf(open(asset).read())
    index_ee_link = chain.frame_to_idx[index_ee_name]
    thumb_ee_link = chain.frame_to_idx[thumb_ee_name]
    frame_indices = torch.tensor([index_ee_link, thumb_ee_link])
    state2ee_pos = partial(state2ee_pos, chain=chain, frame_indices=frame_indices, world_trans=env.world_trans)


    for i in tqdm(range(config['num_trials'])):
        goal = 0.5 * torch.tensor([np.pi])
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
            valve_location = torch.tensor([0.85, 0.70, 1.405]).to(device) # the root of the valve
            params['valve_location'] = valve_location
            final_distance_to_goal = do_trial(env, params, fpath)
            # final_distance_to_goal = turn(env, params, fpath)

            if controller not in results.keys():
                results[controller] = [final_distance_to_goal]
            else:
                results[controller].append(final_distance_to_goal)
        print(results)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
