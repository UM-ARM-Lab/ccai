"""
This script turns the valve assuming the contact point does not change, which means it does not considering rolling
"""
import os
import numpy as np
from isaacgym.torch_utils import quat_apply
from isaac_victor_envs.tasks.allegro import AllegroValveTurningEnv, orientation_error, quat_change_convention
from isaac_victor_envs.utils import get_assets_dir

import torch
import time
import yaml
import pathlib
from functools import partial
from torch.func import vmap, jacrev, hessian, jacfwd
# from functorch import vmap, jacrev, hessian, jacfwd

from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
from ccai.kernels import rbf_kernel, structured_rbf_kernel

from ccai.problem import ConstrainedSVGDProblem
from ccai.mpc.csvgd import Constrained_SVGD_MPC
from ccai.valve import ValveDynamics
from ccai.utils import rotate_jac
# from ccai.mpc.mppi import MPPI
# from ccai.mpc.svgd import SVMPC
# from ccai.mpc.ipopt import IpoptMPC
import time
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
# import pytorch3d.transforms as tf

from pytorch_volumetric import RobotSDF, RobotScene, MeshSDF
import matplotlib.pyplot as plt
import pickle as pkl

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

# asset = f'{get_assets_dir()}/victor/allegro.urdf'
# index_ee_name = 'index_ee'
# thumb_ee_name = 'thumb_ee'
asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
# thumb_ee_name = 'allegro_hand_oya_finger_link_15'
# index_ee_name = 'allegro_hand_hitosashi_finger_finger_link_3'
index_ee_name = 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link'
thumb_ee_name = 'allegro_hand_oya_finger_3_aftc_base_link'

# combined chain
chain = pk.build_chain_from_urdf(open(asset).read())
index_ee_link = chain.frame_to_idx[index_ee_name]
thumb_ee_link = chain.frame_to_idx[thumb_ee_name]
frame_indices = torch.tensor([index_ee_link, thumb_ee_link])

device = 'cuda:0'
device = 'cpu'
valve_location = torch.tensor([0.85, 0.70, 1.405]).to(device)  # the root of the valve
# instantiate environment
friction_coefficient = 0.95  # this one is used for planning, not simulation
valve_type = 'cylinder'  # 'cuboid' or 'cylinder
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')
env = AllegroValveTurningEnv(1, control_mode='joint_impedance', use_cartesian_controller=False,
                             viewer=True, steps_per_action=60, valve_velocity_in_state=False,
                             friction_coefficient=1.0,
                             device=device,
                             valve=valve_type,
                             video_save_path=img_save_dir,
                             configuration='screw_driver')
world_trans = env.world_trans


def partial_to_full_state(partial):
    """
    :params partial: B x 8 joint configurations for index and thumb
    :return full: B x 16 joint configuration for full hand

    # assume that default is zeros, but could change
    """
    index, thumb = torch.chunk(partial, chunks=2, dim=-1)
    full = torch.cat((
        index,
        torch.zeros_like(index),
        torch.zeros_like(index),
        thumb
    ), dim=-1)
    return full


def full_to_partial_state(full):
    """
    :params partial: B x 8 joint configurations for index and thumb
    :return full: B x 16 joint configuration for full hand

    # assume that default is zeros, but could change
    """
    index, mid, ring, thumb = torch.chunk(full, chunks=4, dim=-1)
    partial = torch.cat((
        index,
        thumb
    ), dim=-1)
    return partial


def state2ee_pos(state, finger_name):
    """
    :params state: B x 8 joint configuration for full hand
    :return ee_pos: B x 3 position of ee

    """
    fk_dict = chain.forward_kinematics(partial_to_full_state(state), frame_indices=frame_indices)
    m = world_trans.compose(fk_dict[finger_name])
    points_finger_frame = torch.tensor([0.00, 0.03, 0.00], device=m.device).unsqueeze(0)
    ee_p = m.transform_points(points_finger_frame)
    return ee_p


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


class PositionControlConstrainedSVGDMPC(Constrained_SVGD_MPC):

    def __init__(self, problem, params):
        super().__init__(problem, params)
        self.solver = PositionControlConstrainedSteinTrajOpt(problem, params)


class AllegroValveProblem(ConstrainedSVGDProblem):

    def __init__(self,
                 start,
                 goal,
                 T,
                 chain,
                 valve_location,
                 finger_name,
                 valve_asset_pos,
                 initial_valve_angle=0,
                 collision_checking=False,
                 plan_in_contact=True,
                 device='cuda:0'):
        """
        valve location: the root location of the valve
        initial_valve_angle: it is designed for continuously turning the valve. For each turn, 
        the valve might not be exactly at 0 degree, we need to subtract that out. 
        If we only care about one turn, we can leave it to 0
        plan_to_contact: If False, only the terminal state has to be in contact, rather than the entire trajectory
        """
        super().__init__(start, goal, T, device)
        self.friction_polytope_k = 4
        if collision_checking:
            self.dz = (3 + 1) * 2  # 1 means the collision checking
        else:
            self.dz = 2 * 2

        if plan_in_contact:
            self.dg = 2 * (1 + 3 + 2) * T  # true for every state
            self.dz += 2 * (self.friction_polytope_k)  # two friction constraints per finger
        else:
            self.dg = 2

        self.dh = self.dz * T  # inequality
        self.dx = 9  # position of finger joints and theta and theta dot.
        self.du = 8  # finger joints delta action.
        self.plan_in_contact = plan_in_contact
        # NOTE: the decision variable x means x_1 to x_T, but for the action u, it means u_0 to u_{T-1}. 
        # NOTE: The timestep of x and u doesn't match
        self.dt = 0.1
        self.T = T
        self.start = start
        self.goal = goal
        self.K = rbf_kernel
        self.squared_slack = True
        self.compute_hess = True
        self._collision_checking_flag = collision_checking

        self.finger_name = finger_name
        self.valve_location = valve_location
        self.initial_valve_angle = initial_valve_angle

        self.chain = chain
        self.grad_kernel = jacrev(rbf_kernel, argnums=0)
        self.alpha = 10

        self._equality_constraints = JointConstraint(
            partial(get_joint_equality_constraint, chain=self.chain, start_q=self.start,
                    initial_valve_angle=initial_valve_angle)
        )

        self._inequality_constraints = JointConstraint(
            partial(get_joint_inequality_constraint, chain=self.chain, start_q=self.start)
        )

        # add collision checking
        # collision check all of the non-finger tip links
        collision_check_oya = ['allegro_hand_oya_finger_link_13',
                               'allegro_hand_oya_finger_link_14',
                               # 'allegro_hand_oya_finger_link_15'
                               ]
        collision_check_hitosashi = [  # 'allegro_hand_hitosashi_finger_finger_link_3',
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

        # rob_trans = pk.Transform3d(pos=torch.tensor(robot_p, device=device),
        #                            rot=torch.tensor([robot_r[3], robot_r[0], robot_r[1], robot_r[2]], device=device),
        #                            device=device)
        # TODO: retrieve transformations from environment rather than hard-coded

        # scene_trans = rob_trans.inverse().compose(pk.Transform3d(device=device).translate(0.85, 0.75, 1.405))
        scene_trans = world_trans.inverse().compose(
            pk.Transform3d(device=device).translate(valve_asset_pos[0], valve_asset_pos[1], valve_asset_pos[2]))

        # TODO: right now we are using seperate collision checkers for each finger to avoid gradients swapping
        # between fingers - alteratively we can get the collision checker to return a list of collisions and gradients batched
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
                                                 points_per_link=1000)
        self.thumb_contact_scene = pv.RobotScene(robot_sdf, valve_sdf, scene_trans,
                                                 collision_check_links=contact_check_oya,
                                                 softmin_temp=1.0e3,
                                                 points_per_link=1000)
        # self.index_contact_scene.visualize_robot(partial_to_full_state(self.start[:8]), self.start[-1])

        self.friction_constr = vmap(self._friction_constr, randomness='same')
        self.grad_friction_constr = vmap(jacrev(self._friction_constr, argnums=(0, 1, 2)))

        self.dynamics_constr = vmap(self._dynamics_constr)
        self.grad_dynamics_constr = vmap(jacrev(self._dynamics_constr, argnums=(0, 1, 2, 3, 4)))

        # for honda hand
        index_x_max = torch.tensor([0.47, 1.6099999999, 1.7089999, 1.61799999]) - 0.05
        index_x_min = torch.tensor([-0.47, -0.195999999999, -0.174000000, -0.227]) + 0.05
        thumb_x_max = torch.tensor([1.396, 1.1629999999999, 1.644, 1.71899999]) - 0.05
        thumb_x_min = torch.tensor([0.26, -0.1049999999, -0.1889999999, -0.162]) + 0.05

        valve_x_max = torch.tensor([10.0 * np.pi])
        valve_x_min = torch.tensor([-10.0 * np.pi])
        self.x_max = torch.cat((index_x_max, thumb_x_max, valve_x_max))
        self.x_min = torch.cat((index_x_min, thumb_x_min, valve_x_min))
        self.robot_joint_x_max = torch.cat([index_x_max, thumb_x_max])
        self.robot_joint_x_min = torch.cat([index_x_min, thumb_x_min])

        # TODO: this is not perfect, need a better way to define the limit of u
        self.u_max = torch.tensor([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]) / 5
        self.u_min = torch.tensor([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]) / 5

        self.x_max = torch.cat((self.x_max, self.u_max))
        self.x_min = torch.cat((self.x_min, self.u_min))

        self.cost = vmap(partial(cost, start=self.start, goal=self.goal))
        self.grad_cost = vmap(jacrev(partial(cost, start=self.start, goal=self.goal)))
        self.hess_cost = vmap(hessian(partial(cost, start=self.start, goal=self.goal)))

        self.data = {}

    def _preprocess(self, xu):
        self._preprocess_finger(xu, self.index_contact_scene, index_ee_link, 'index')
        self._preprocess_finger(xu, self.thumb_contact_scene, thumb_ee_link, 'thumb')

    def _preprocess_finger(self, xu, finger_scene, finger_ee_link, finger_name):
        N = xu.shape[0]
        xu = xu.reshape(N, self.T, -1)[:, :, :self.dx + self.du]
        x = xu[:, :, :self.dx]
        N, T, _ = x.shape

        # we want to add the start state to x, this x is now T + 1
        x = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)

        q = x[:, :, :8]
        theta = x[:, :, -1].unsqueeze(-1)

        # reshape to batch across time
        q_b = q.reshape(-1, 8)
        theta_b = theta.reshape(-1, 1)
        full_q = partial_to_full_state(q_b)
        ret_scene = finger_scene.scene_collision_check(full_q, theta_b,
                                                       compute_gradient=True,
                                                       compute_hessian=False)

        link_indices = torch.ones(full_q.shape[0], dtype=torch.int64, device=full_q.device) * finger_ee_link
        djacobian_dcontact = finger_scene.robot_sdf.chain.calc_djacobian_dtool(full_q,
                                                                               link_indices=link_indices)

        # let's check the jacobian gradient wrt contact point

        djacobian_dcontact = djacobian_dcontact.reshape(N, -1, 3, 6, 16)[:, :, :, :3,
                             (0, 1, 2, 3, 12, 13, 14, 15)]
        contact_jacobian = ret_scene.get('contact_jacobian', None)
        contact_jacobian = contact_jacobian.reshape(N, T + 1, 3, 16)[:, :, :, (0, 1, 2, 3, 12, 13, 14, 15)]
        JTinv = torch.linalg.pinv(contact_jacobian.transpose(-1, -2))
        # print('--')
        # R = self.get_rotation_from_normal(ret_scene['contact_normal'].reshape(-1, 3)).reshape(N, T + 1, 3, 3)
        # if finger_name == 'thumb':
        #     print('thumb')
        #     Jr = R[0, 0] @ JTinv[0, 0, :, 4:]
        #     print(contact_jacobian.transpose(-1, -2)[0, 0, 4:, :] @ torch.tensor([0.0, 0.0, -1.0], device=JTinv.device))
        #     print(contact_jacobian.transpose(-1, -2)[0, 0, 4:, :] @ torch.tensor([1.0, 0.0, 0.0], device=JTinv.device))
        # else:
        #     print('index')
        #     Jr = R[0, 0] @ JTinv[0, 0, :, :4]
        #     print(contact_jacobian.transpose(-1, -2)[0, 0, :4, :] @ torch.tensor([0.0, 0.0, -1.0], device=JTinv.device))
        #     print(contact_jacobian.transpose(-1, -2)[0, 0, :4, :] @ torch.tensor([1.0, 0.0, 0.0], device=JTinv.device))
        ret_scene['contact_jacobian'] = contact_jacobian
        contact_hessian = ret_scene.get('contact_hessian', None)
        contact_hessian = contact_hessian.reshape(N, T + 1, 3, 16, 16)[:, :, :, (0, 1, 2, 3, 12, 13, 14, 15)]
        contact_hessian = contact_hessian[:, :, :, :, (0, 1, 2, 3, 12, 13, 14, 15)]  # shape (N, T+1, 3, 8, 8)
        d_contact_loc_dq = ret_scene.get('closest_pt_q_grad', None)
        d_contact_loc_dq = d_contact_loc_dq.reshape(N, T + 1, 3, 16)[:, :, :, (0, 1, 2, 3, 12, 13, 14, 15)]
        ret_scene['closest_pt_q_grad'] = d_contact_loc_dq
        ret_scene['contact_hessian'] = contact_hessian
        ret_scene['dnormal_dq'] = ret_scene.get('dnormal_dq').reshape(N, T + 1, 3, 16)[:, :, :,
                                  (0, 1, 2, 3, 12, 13, 14, 15)]

        #dJ_dq = djacobian_dcontact.reshape(N, T + 1, 3, 3, 8, 1) * d_contact_loc_dq.reshape(N, T + 1, 3, 1, 1, 8)
        ## print(dJ_dq.shape)
        #dJ_dq = dJ_dq.sum(dim=2)  # should be N x T x 3 x 8 x 8
        #dJ_dq = dJ_dq + contact_hessian
        dJ_dq = contact_hessian
        ret_scene['dJ_dq'] = dJ_dq
        ret_scene['djacobian_dcontact'] = djacobian_dcontact
        self.data[finger_name] = ret_scene

    def _objective(self, x):
        x = x[:, :, :self.dx + self.du]
        N = x.shape[0]
        J, grad_J, hess_J = self.cost(x), self.grad_cost(x), self.hess_cost(x)

        N = x.shape[0]
        return (self.alpha * J.reshape(N),
                self.alpha * grad_J.reshape(N, -1),
                self.alpha * hess_J.reshape(N, self.T * (self.dx + self.du), self.T * (self.dx + self.du)))

    def _contact_constraints(self, xu, finger, finger_name, compute_grads=True, compute_hess=False):
        """ Computes contact constraints
            constraint that sdf value is zero
            also constraint on contact kinematics to get the valve dynamics
        """
        x = xu[:, :, :self.dx]
        N, T, _ = x.shape
        # we want to add the start state to x, this x is now T + 1
        x = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)

        q = x[:, :, :8]
        theta = x[:, :, -1].unsqueeze(-1)

        # reshape to batch across time
        q_b = q.reshape(-1, 8)
        ret_scene = self.data[finger_name]
        sdf = ret_scene.get('sdf').reshape(N, T + 1, 1)  # - 0.0025
        grad_sdf_q = ret_scene.get('grad_sdf', None)
        hess_sdf_q = ret_scene.get('hess_sdf', None)
        grad_sdf_theta = ret_scene.get('grad_env_sdf', None)
        hess_sdf_theta = ret_scene.get('hess_env_sdf', None)
        djacobian_dcontact = ret_scene.get('djacobian_dcontact', None)

        # print(sdf[:, 1:].abs().mean(), sdf[:, 1:].abs().max())
        gain = 1

        if self.plan_in_contact:
            # approximate q dot and theta dot
            dq = (x[:, 1:, :8] - x[:, :-1, :8])
            dtheta = (x[:, 1:, 8] - x[:, :-1, 8])

            # angular velocity of the valve
            valve_omega = torch.stack((torch.zeros_like(dtheta), dtheta, torch.zeros_like(dtheta)),
                                      -1)  # should be N x T-1 x 3

            contact_jacobian = ret_scene.get('contact_jacobian', None)
            contact_hessian = ret_scene.get('contact_hessian', None)
            contact_loc = ret_scene.get('closest_pt_world', None)
            # print(contact_loc.shape)
            # print(finger_name)
            # print(contact_loc.reshape(N, T + 1, 3)[:, 0])
            d_contact_loc_dq = ret_scene.get('closest_pt_q_grad', None)

            # contact_jacobian = contact_jacobian.reshape(N, T + 1, 3, 16)[:, :, :, (0, 1, 2, 3, 12, 13, 14, 15)]
            # contact_hessian = contact_hessian.reshape(N, T + 1, 3, 16, 16)[:, :, :, (0, 1, 2, 3, 12, 13, 14, 15)]
            # contact_hessian = contact_hessian[:, :, :, :, (0, 1, 2, 3, 12, 13, 14, 15)]  # shape (N, T+1, 3, 8, 8)
            # d_contact_loc_dq = d_contact_loc_dq.reshape(N, T + 1, 3, 16)[:, :, :, (0, 1, 2, 3, 12, 13, 14, 15)]

            contact_point_v = (contact_jacobian[:, :-1] @ dq.reshape(N, T, 8, 1)).squeeze(-1)  # should be N x T x 3
            # valve location in robot frame
            valve_robot_frame = world_trans.inverse().transform_points(valve_location.reshape(1, 3))
            contact_point_r_valve = contact_loc.reshape(N, T + 1, 3) - valve_robot_frame.reshape(1, 1, 3)

            # valve omega in robot frame
            valve_omega_robot_frame = world_trans.inverse().transform_normals(valve_omega)
            # object_contact_point_v = torch.cross(contact_point_r_valve[:, :-1],
            #                                     valve_omega_robot_frame)
            object_contact_point_v = torch.cross(valve_omega_robot_frame, contact_point_r_valve[:, :-1])
            # print(finger_name)
            # print('contact point velocities in robot frame')
            # print('object')
            # print(object_contact_point_v[:, 0])
            # print('robot')
            # print(contact_point_v[:, 0])
            # kinematics constraint, should be 3-dimensional
            kinematics_constraint = gain * (
                    contact_point_v - object_contact_point_v)  # we actually ended up computing T+1 contact constraints, but start state is fixed so we throw that away

            # print('kinematics constraint', kinematics_constraint.abs().mean(), kinematics_constraint.abs().max())
            # for T kinematic constraints and T sdf constraints
            g = torch.cat((sdf[:, 1:].reshape(N, -1),
                           kinematics_constraint.reshape(N, -1)), dim=-1)  # should be N x T*4
        else:
            g = sdf[:, -1].reshape(N, -1)

        if compute_grads:
            T_range = torch.arange(T, device=x.device)
            T_range_minus = torch.arange(T - 1, device=x.device)
            T_range_plus = torch.arange(1, T, device=x.device)
            # compute gradient of sdf
            grad_sdf = torch.zeros(N, T, T, self.dx + self.du, device=x.device)
            grad_sdf_q = grad_sdf_q[:, (0, 1, 2, 3, 12, 13, 14, 15)].reshape(N, T + 1, 8)
            grad_sdf[:, T_range, T_range, :8] = grad_sdf_q[:, 1:]
            grad_sdf[:, T_range, T_range, 8] = grad_sdf_theta.reshape(N, T + 1)[:, 1:]
            grad_g = grad_sdf.reshape(N, -1, T, self.dx + self.du)

            if self.plan_in_contact:

                # retrieve and make shape (N, T+1, 3, 3, 8)
                # djacobian_dcontact = djacobian_dcontact.reshape(N, -1, 3, 6, 16)[:, :, :, :3,
                #                     (0, 1, 2, 3, 12, 13, 14, 15)]

                # dJ_dq = djacobian_dcontact.reshape(N, T + 1, 3, 3, 8, 1) * d_contact_loc_dq.reshape(N, T + 1, 3, 1, 1,
                #                                                                                    8)
                # dJ_dq = dJ_dq.sum(dim=2)  # should be N x T x 3 x 8 x 8

                # dJ_dq = dJ_dq + contact_hessian
                dJ_dq = ret_scene['dJ_dq']
                dcontact_v_dq = (dJ_dq[:, 1:] @ dq.reshape(N, T, 1, 8, 1)).squeeze(-1) - contact_jacobian[:,
                                                                                         1:]  # should be N x T x 3 x 8
                tmp = torch.cross(d_contact_loc_dq[:, 1:], valve_omega.reshape(N, T, 3, 1), dim=2)  # N x T x 3 x 8
                dkinematics_constraint_dq = dcontact_v_dq - tmp

                d_omega_dtheta = torch.stack((torch.zeros_like(dtheta),
                                              torch.ones_like(dtheta),
                                              torch.zeros_like(dtheta)), dim=-1)  # N x T x 3
                d_omega_dtheta = world_trans.inverse().transform_normals(d_omega_dtheta)

                dkinematics_constraint_dtheta = torch.cross(d_omega_dtheta, contact_point_r_valve[:, :-1],
                                                            dim=-1)  # N x T x 3

                # print('--')
                # print(world_trans.transform_normals(contact_point_v))
                # print(world_trans.transform_normals(object_contact_point_v))
                grad_kinematics_constraint = torch.zeros((N, T, T, 3, self.dx + self.du), device=x.device)
                grad_kinematics_constraint[:, T_range_plus, T_range_minus, :, :8] = dkinematics_constraint_dq[:, 1:]
                grad_kinematics_constraint[:, T_range_plus, T_range_minus, :, 8] = dkinematics_constraint_dtheta[:, 1:]
                grad_kinematics_constraint[:, T_range, T_range, :, :8] = contact_jacobian[:, :-1]  # looks correct
                grad_kinematics_constraint[:, T_range, T_range, :, 8] = -dkinematics_constraint_dtheta  # looks correct
                grad_kinematics_constraint = gain * grad_kinematics_constraint.permute(0, 1, 3, 2, 4)

                grad_g = torch.cat((grad_g,
                                    grad_kinematics_constraint.reshape(N, -1, T, self.dx + self.du)), dim=1)
            else:
                grad_g = grad_g[:, -1].unsqueeze(1)

            grad_g = grad_g.reshape(N, -1, T * (self.dx + self.du))

        else:
            return g, None, None

        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * (self.dx + self.du), T * (self.dx + self.du), device=self.device)
            return g, grad_g, hess

        return g, grad_g, None

    def contact_constraints(self, x, compute_grads=True, compute_hess=False):
        # if self.plan_in_contact:
        #     N, T, d = x.shape
        #     numerical_grad = torch.zeros(N, T, 3, T, 9, device=self.device)
        #     g, grad_g, hess_g = self._contact_constraints(x, self.index_contact_scene,
        #                                                   index_ee_link, compute_grads, compute_hess)
        #     grad_g = grad_g.reshape(N, T, 4, T, d)[:, :, 1:, :, :9]
        #     delta = 1e-4
        #     for i in range(T):
        #         for j in range(9):
        #             dx = torch.zeros_like(x)
        #             dx[:, i, j] = delta
        #             # finite difference test
        #             g_plus, _, _ = self._contact_constraints(x + dx, self.index_contact_scene,
        #                                                      index_ee_link, False, False)
        #             g_neg, _, _ = self._contact_constraints(x - dx, self.index_contact_scene,
        #                                                     index_ee_link, False, False)
        #             print(g_plus.shape)
        #             print(g_neg.shape)
        #             numerical_grad[:, :, :, i, j] = (g_plus.reshape(N, T, 4)[:, :, 1:] - g_neg.reshape(N, T, 4)[:, :, 1:]) / (2 * delta)
        #     print('done')

        # compute contact constraints for index finger
        g_i, grad_g_i, hess_g_i = self._contact_constraints(x, self.index_contact_scene, 'index',
                                                            compute_grads, compute_hess)
        g_t, grad_g_t, hess_g_t = self._contact_constraints(x, self.thumb_contact_scene, 'thumb',
                                                            compute_grads, compute_hess)
        g = torch.cat((g_i, g_t), dim=1)
        if compute_grads:
            grad_g = torch.cat((grad_g_i, grad_g_t), dim=1)
        else:
            return g, None, None

        if compute_hess:
            hess_g = torch.cat((hess_g_i, hess_g_t), dim=1)
            return g, grad_g, hess_g

        return g, grad_g, None

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

    def get_friction_polytope(self):
        """
        :param k: the number of faces of the friction cone
        :return: a list of normal vectors of the faces of the friction cone
        """

        normal_vectors = []
        for i in range(self.friction_polytope_k):
            theta = 2 * np.pi * i / self.friction_polytope_k
            normal_vector = torch.tensor([np.cos(theta), np.sin(theta), friction_coefficient]).to(device=self.device,
                                                                                                   dtype=torch.float32)
            normal_vectors.append(normal_vector)
        normal_vectors = torch.stack(normal_vectors, dim=0)
        return normal_vectors

    def _friction_constr(self, dq, contact_normal, contact_jacobian):
        # this will be vmapped, so takes in a 3 vector and a 3 x 8 jacobian and a dq vector
        # intervene on contact normal to break gradients
        # contact_normal = torch.ones_like(contact_normal)
        # contact_normal = contact_normal / torch.norm(contact_normal, dim=-1, keepdim=True)
        # compute the force in robot frame
        # force = (torch.linalg.lstsq(contact_jacobian.transpose(-1, -2),
        #                            dq.unsqueeze(-1))).solution.squeeze(-1)
        # force_world_frame = world_trans.transform_normals(force.unsqueeze(0)).squeeze(0)

        # transform contact normal to world frame
        contact_normal_world = world_trans.transform_normals(contact_normal.unsqueeze(0)).squeeze(0)

        # print(torch.linalg.matrix_rank(torch.linalg.pinv(contact_jacobian.transpose(-1, -2))))
        # print(torch.linalg.svdvals(torch.linalg.pinv(contact_jacobian.transpose(-1, -2))))
        # compute projection of force onto normal
        # normal_projection = contact_normal.unsqueeze(-1) @ contact_normal.unsqueeze(-2)
        # force_normal = (normal_projection @ force_world_frame.unsqueeze(-1)).squeeze(-1)
        # force_tan = force_world_frame - force_normal

        # TODO: I think a linearized friction cone may perform better
        # A = torch.tensor([[0.0, 1.0, 0.0],
        #                  [0.707, 0.0, 0.707]], device=self.device)
        # force_tan = A @ force_tan.unsqueeze(-1)
        R = self.get_rotation_from_normal(contact_normal_world.unsqueeze(0)).squeeze(0)
        # force_contact_frame = R.transpose(0, 1) @ force_world_frame.unsqueeze(-1)
        # force_normal = torch.sum(force_contact_frame[2]).reshape(1, 1)

        # print(torch.sum(force_world_frame.reshape(-1) * contact_normal_world.reshape(-1)))
        B = self.get_friction_polytope().detach()
        # force = torch.cat((force_tan, force_normal), dim=0)  # should be 3 x 1
        # cone_constraint = B @ force_contact_frame
        contact_v_contact_frame = R.transpose(0, 1) @ world_trans.transform_normals(
            (contact_jacobian @ dq).unsqueeze(0)).squeeze(0)
        return B @ contact_v_contact_frame

        # TODO: there are two different ways of doing a friction cone
        # either geometrically based on the contact_v or using the pinv(J^T) method, the first seems to work better
        # though it is less 'correct'

        # geometrically using contact_v
        return (torch.linalg.norm(contact_v_contact_frame[:2]) + friction_coefficient * contact_v_contact_frame[
            2]).reshape(-1)

        # using pinv(J^T) dq
        # return (torch.linalg.norm(force_contact_frame[:2]) + friction_coefficient * force_contact_frame[2])

    def _friction_cone_constraint(self, xu, finger, compute_grads=True, compute_hess=False):

        # assume access to class member variables which have already done some of the computation
        N, T, d = xu.shape
        q = xu[:, :, :8]
        theta = xu[:, :, 8].unsqueeze(-1)

        theta = torch.cat((self.start[-1].reshape(1, 1, 1).repeat(N, 1, 1), theta), dim=1)
        dtheta = theta[:, 1:] - theta[:, :-1]  # N x T x 1

        u = xu[:, :, self.dx:]

        # u is the delta q commanded
        # retrieved cached values
        contact_jac = self.data[finger]['contact_jacobian'].reshape(N, T + 1, 3, 8)[:, :-1]
        contact_normal = self.data[finger]['contact_normal'].reshape(N, T + 1, 3)[:, :-1]

        dnormal_dq = self.data[finger]['dnormal_dq'].reshape(N, T + 1, 3, 8)[:, :-1]
        dnormal_dtheta = self.data[finger]['dnormal_denv_q'].reshape(N, T + 1, 3, 1)[:, :-1]

        # compute constraint value
        # u = torch.zeros_like(u)
        h = self.friction_constr(u.reshape(-1, 8),
                                 contact_normal.reshape(-1, 3),
                                 contact_jac.reshape(-1, 3, 8)).reshape(N, -1)
        # compute the gradient

        if compute_grads:
            dh_du, dh_dnormal, dh_djac = self.grad_friction_constr(u.reshape(-1, 8),
                                                                   contact_normal.reshape(-1, 3),
                                                                   contact_jac.reshape(-1, 3, 8))

            # dnormal_dq = self.data['grad_contact_normal'].reshape(N, T + 1, 3, 3)[:, :-1]
            # dnormal_dq = torch.zeros(N, T, 3, 8, device=self.device)  # assume zero SDF hessian
            djac_dq = self.data[finger]['dJ_dq'].reshape(N, T + 1, 3, 8, 8)[:, :-1]

            dh = dh_dnormal.shape[1]
            dh_dq = (dh_dnormal.reshape(N, T, dh, -1) @ dnormal_dq)
            dh_dq = dh_dq + dh_djac.reshape(N, T, dh, -1) @ djac_dq.reshape(N, T, -1, 8)

            dh_theta = dh_dnormal.reshape(N, T, dh, -1) @ dnormal_dtheta
            grad_h = torch.zeros(N, dh, T, T, d, device=self.device)
            T_range = torch.arange(T, device=self.device)
            T_range_minus = torch.arange(T - 1, device=self.device)
            T_range_plus = torch.arange(1, T, device=self.device)
            grad_h[:, :, T_range_plus, T_range_minus, :8] = dh_dq[:, 1:].transpose(1, 2)
            grad_h[:, :, T_range_plus, T_range_minus, 8] = dh_theta[:, 1:].squeeze(-1).transpose(1, 2)
            grad_h[:, :, T_range, T_range, self.dx:] = dh_du.reshape(N, T, dh, self.du).transpose(1, 2)
            grad_h = grad_h.transpose(1, 2).reshape(N, -1, T * d)
        else:
            return h, None, None

        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], T * d, T * d, device=self.device)
            return h, grad_h, hess_h

        return h, grad_h, None

    def _dynamics_constr(self, q, u, next_q, contact_jacobian, contact_normal):
        # this will be vmapped, so takes in a 3 vector and a 3 x 8 jacobian and a dq vector
        dq = next_q - q
        contact_v = (contact_jacobian @ dq.unsqueeze(-1)).squeeze(-1)  # should be 3 vector
        # from commanded
        contact_v_u = (contact_jacobian @ u.unsqueeze(-1)).squeeze(-1)  # should be 3 vector

        # convert to world frame
        contact_v_world = world_trans.transform_normals(contact_v.unsqueeze(0)).squeeze(0)
        contact_v_u_world = world_trans.transform_normals(contact_v_u.unsqueeze(0)).squeeze(0)
        contact_normal_world = world_trans.transform_normals(contact_normal.unsqueeze(0)).squeeze(0)

        # compute projection onto normal
        normal_projection = contact_normal_world.unsqueeze(-1) @ contact_normal_world.unsqueeze(-2)
        # must find a lower dimensional representation of the constraint to avoid numerical issues
        # TODO for now hand coded, but need to find a better solution
        R = self.get_rotation_from_normal(contact_normal_world.unsqueeze(0)).squeeze(0).permute(0, 1)  # .detach(
        R = R[:2]

        # compute contact v tangential to surface
        contact_v_tan = contact_v_world - (normal_projection @ contact_v_world.unsqueeze(-1)).squeeze(-1)
        contact_v_u_tan = contact_v_u_world - (normal_projection @ contact_v_u_world.unsqueeze(-1)).squeeze(-1)

        # should have same tangential components
        # TODO: this constraint value is super small
        return (R @ (contact_v_tan - contact_v_u_tan).unsqueeze(-1)).squeeze(-1)

    def _dynamics_constraints(self, xu, finger, compute_grads=True, compute_hess=False):
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
        contact_jac = self.data[finger]['contact_jacobian'].reshape(N, T + 1, 3, 8)[:, :-1]
        contact_normal = self.data[finger]['contact_normal'].reshape(N, T + 1, 3)[:, :-1]
        dnormal_dq = self.data[finger]['dnormal_dq'].reshape(N, T + 1, 3, 8)[:, :-1]
        dnormal_dtheta = self.data[finger]['dnormal_denv_q'].reshape(N, T + 1, 3, 1)[:, :-1]

        g = self.dynamics_constr(q.reshape(-1, 8), u.reshape(-1, 8), next_q.reshape(-1, 8),
                                 contact_jac.reshape(-1, 3, 8),
                                 contact_normal.reshape(-1, 3)).reshape(N, T, -1)

        if compute_grads:
            T_range = torch.arange(T, device=x.device)
            T_plus = torch.arange(1, T, device=x.device)
            T_minus = torch.arange(T - 1, device=x.device)
            grad_g = torch.zeros(N, g.shape[2], T, T, self.dx + self.du, device=self.device)
            # dnormal_dq = torch.zeros(N, T, 3, 8, device=self.device)  # assume zero SDF hessian
            djac_dq = self.data[finger]['dJ_dq'].reshape(N, T + 1, 3, 8, 8)[:, :-1]
            dg_dq, dg_du, dg_dnext_q, dg_djac, dg_dnormal = self.grad_dynamics_constr(q.reshape(-1, 8),
                                                                                      u.reshape(-1, 8),
                                                                                      next_q.reshape(-1, 8),
                                                                                      contact_jac.reshape(-1, 3, 8),
                                                                                      contact_normal.reshape(-1, 3))
            if torch.any(torch.isnan(dg_dq)):
                print('nan')
            if torch.any(torch.isnan(dnormal_dq)):
                print('nan')
            dg_dq = dg_dq.reshape(N, T, g.shape[2], -1) + dg_dnormal.reshape(N, T, g.shape[2], -1) @ dnormal_dq  #
            dg_dq = dg_dq + dg_djac.reshape(N, T, g.shape[2], -1) @ djac_dq.reshape(N, T, -1, 8)
            dg_theta = dg_dnormal.reshape(N, T, g.shape[2], -1) @ dnormal_dtheta

            grad_g[:, :, T_plus, T_minus, :8] = dg_dq[:, 1:].transpose(1, 2)  # first q is the start
            grad_g[:, :, T_range, T_range, self.dx:] = dg_du.reshape(N, T, -1, self.du).transpose(1, 2)
            grad_g[:, :, T_plus, T_minus, 8] = dg_theta[:, 1:].squeeze(-1).transpose(1, 2)
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

    def dynamics_constraints(self, xu, compute_grads=True, compute_hess=True):
        # compute contact constraints for index finger
        g_i, grad_g_i, hess_g_i = self._dynamics_constraints(xu, 'index',
                                                             compute_grads, compute_hess)
        g_t, grad_g_t, hess_g_t = self._dynamics_constraints(xu, 'thumb',
                                                             compute_grads, compute_hess)
        g = torch.cat((g_i, g_t), dim=1)
        if compute_grads:
            grad_g = torch.cat((grad_g_i, grad_g_t), dim=1)
        else:
            return g, None, None

        if compute_hess:
            hess_g = torch.cat((hess_g_i, hess_g_t), dim=1)
            return g, grad_g, hess_g

        return g, grad_g, None

    def friction_constraints(self, xu, compute_grads=True, compute_hess=False):
        # compute contact constraints for index finger
        g_i, grad_g_i, hess_g_i = self._friction_cone_constraint(xu, 'index',
                                                                 compute_grads, compute_hess)

        g_t, grad_g_t, hess_g_t = self._friction_cone_constraint(xu, 'thumb',
                                                                 compute_grads, compute_hess)
        g = torch.cat((g_i, g_t), dim=1)
        if compute_grads:
            grad_g = torch.cat((grad_g_i, grad_g_t), dim=1)
        else:
            return g, None, None

        if compute_hess:
            hess_g = torch.cat((hess_g_i, hess_g_t), dim=1)
            return g, grad_g, hess_g

        return g, grad_g, None

    def _con_eq(self, xu, compute_grads=True, compute_hess=False):
        N = xu.shape[0]
        T = xu.shape[1]
        g_contact, grad_g_contact, hess_g_contact = self.contact_constraints(xu.reshape(N, T, self.dx + self.du),
                                                                             compute_grads=compute_grads,
                                                                             compute_hess=compute_hess)

        if self.plan_in_contact:
            # if compute_grads:
            #     eps = 1e-6
            #     delta = 1e-4
            #     # compute derivative of contact jacobian numerically
            #     self._preprocess(xu)
            #     dJ_dq = {
            #         'thumb': self.data['thumb']['dJ_dq'],
            #         'index': self.data['index']['dJ_dq']
            #     }
            #
            #     dJ_dq_numerical = {
            #         'thumb': torch.zeros_like(dJ_dq['thumb']),
            #         'index': torch.zeros_like(dJ_dq['index'])
            #     }
            #
            #     dnormal_dq = {
            #         'thumb': self.data['thumb']['dnormal_dq'].clone(),
            #         'index': self.data['index']['dnormal_dq'].clone()
            #     }
            #
            #     dnormal_dq_numerical = {
            #         'thumb': torch.zeros_like(dnormal_dq['thumb']),
            #         'index': torch.zeros_like(dnormal_dq['index'])
            #     }
            #
            #     dnormal_denv_q = {
            #         'thumb': self.data['thumb']['dnormal_denv_q'].clone(),
            #         'index': self.data['index']['dnormal_denv_q'].clone()
            #     }
            #
            #     dnormal_denv_q_numerical = {
            #         'thumb': torch.zeros_like(dnormal_denv_q['thumb']),
            #         'index': torch.zeros_like(dnormal_denv_q['index'])
            #     }
            #
            #     dcontact_dq = {
            #         'thumb': self.data['thumb']['closest_pt_q_grad'],
            #         'index': self.data['index']['closest_pt_q_grad']
            #     }
            #
            #     dcontact_dq_numerical = {
            #         'thumb': torch.zeros_like(dcontact_dq['thumb']),
            #         'index': torch.zeros_like(dcontact_dq['index'])
            #     }
            #
            #     # print(dnormal_denv_q['thumb'].shape)
            #     # print(dnormal_denv_q['thumb'].shape)
            #     #print(self.data['thumb']['closest_pt_q_grad'].shape)
            #     #print(self.data['index']['closest_pt_world'].shape)
            #
            #     for d in range(self.dx):
            #         dx = torch.zeros_like(xu)
            #         dx = dx.reshape(N * T, -1)
            #         dx[:, d] = delta
            #         dx = dx.reshape(N, T, -1)
            #         # finite difference test
            #         self._preprocess(xu + dx)
            #         Jplus_t = self.data['thumb']['contact_jacobian']
            #         Jplus_i = self.data['index']['contact_jacobian']
            #
            #         normalplus_t = self.data['thumb']['contact_normal']
            #         normalplus_i = self.data['index']['contact_normal']
            #
            #         contact_plus_t = self.data['thumb']['closest_pt_world'].reshape(N, -1, 3)
            #         contact_plus_i = self.data['index']['closest_pt_world'].reshape(N, -1, 3)
            #
            #         self._preprocess(xu - dx)
            #         Jneg_t = self.data['thumb']['contact_jacobian']
            #         Jneg_i = self.data['index']['contact_jacobian']
            #
            #         normalneg_t = self.data['thumb']['contact_normal']
            #         normalneg_i = self.data['index']['contact_normal']
            #
            #         contact_neg_t = self.data['thumb']['closest_pt_world'].reshape(N, -1, 3)
            #         contact_neg_i = self.data['index']['closest_pt_world'].reshape(N, -1, 3)
            #
            #         if d < 8:
            #             dJ_dq_numerical['thumb'][:, :, :, :, d] = (Jplus_t - Jneg_t) / (2 * delta)
            #             dJ_dq_numerical['index'][:, :, :, :, d] = (Jplus_i - Jneg_i) / (2 * delta)
            #             dnormal_dq_numerical['thumb'][:, :, :, d] = (normalplus_t - normalneg_t).reshape(N, T + 1,
            #                                                                                              -1) / (
            #                                                                 2 * delta)
            #             dnormal_dq_numerical['index'][:, :, :, d] = (normalplus_i - normalneg_i).reshape(N, T + 1,
            #                                                                                              -1) / (
            #                                                                 2 * delta)
            #
            #             dcontact_dq_numerical['thumb'][:, :, :, d] = (contact_plus_t - contact_neg_t) / (2 * delta)
            #             dcontact_dq_numerical['index'][:, :, :, d] = (contact_plus_i - contact_neg_i) / (2 * delta)
            #
            #         elif d == 8:
            #             dnormal_denv_q_numerical['thumb'][:, :, 0] = (normalplus_t - normalneg_t) / (2 * delta)
            #             dnormal_denv_q_numerical['index'][:, :, 0] = (normalplus_i - normalneg_i) / (2 * delta)
            #
            #     print('index_dnormal_dq', torch.cosine_similarity(dnormal_dq_numerical['index'][:, 1:].reshape(-1, 8) + eps,
            #                                   dnormal_dq['index'][:, 1:].reshape(-1, 8) + eps, dim=-1).mean())
            #
            #     print('thumb_dnormal_dq', torch.cosine_similarity(dnormal_dq_numerical['thumb'][:, 1:, :].reshape(-1, 8) + eps,
            #                                   dnormal_dq['thumb'][:, 1:].reshape(-1, 8) + eps, dim=-1).mean())
            #
            #     print('index_dJ_dq', torch.cosine_similarity(dJ_dq_numerical['index'][:, 1:].reshape(-1, 8) + eps,
            #                                   dJ_dq['index'][:, 1:].reshape(-1, 8) + eps, dim=-1).mean())
            #     print('thumb_dJ_dq', torch.cosine_similarity(dJ_dq_numerical['thumb'][:, 1:].reshape(-1, 8) + eps,
            #                                                  dJ_dq['thumb'][:, 1:].reshape(-1, 8) + eps, dim=-1).mean())
            #     print('thumb_dcontact_dq', torch.cosine_similarity(dcontact_dq_numerical['thumb'][:, 1:, :].reshape(-1, 8) + eps,
            #                               dcontact_dq['thumb'][:, 1:].reshape(-1, 8) + eps, dim=-1).mean())
            #     print('index_dcontact_dq',
            #           torch.cosine_similarity(dcontact_dq_numerical['index'][:, 1:, :].reshape(-1, 8) + eps,
            #                                   dcontact_dq['index'][:, 1:].reshape(-1, 8) + eps, dim=-1).mean())
            #     #print(torch.cosine_similarity(dJ_dq_numerical['index'][:, 1:].reshape(-1, 8) + eps,
            #     #                              dJ_dq['index'][:, 1:].reshape(-1, 8) + eps, dim=-1).mean())
            #
            #     #print(torch.cosine_similarity(dJ_dq_numerical['thumb'][:, 1:, :].reshape(-1, 8) + eps,
            #     #                              self.data['thumb']['contact_hessian'][:, 1:].reshape(-1, 8) + eps, dim=-1).mean())
            #     #print(torch.cosine_similarity(dJ_dq_numerical['index'][:, 1:, :].reshape(-1, 8) + eps,
            #     #                              self.data['index']['contact_hessian'][:, 1:].reshape(-1, 8) + eps,
            #     #                              dim=-1).mean())
            #
            #     #print('dnormal_dtheta', torch.cosine_similarity(dnormal_denv_q_numerical['index'][:, 1:].reshape(-1, 8),
            #     #                              dnormal_denv_q['index'][:, 1:].reshape(-1, 8), dim=-1).mean())
            #
            #     #print(torch.cosine_similarity(dJ_dq_numerical['thumb'][:, 1:, :].reshape(-1, 8),
            #     #                              dJ_dq['thumb'][:, 1:].reshape(-1, 8), dim=-1).mean())
            #
            #     self.data['thumb']['dJ_dq'] = dJ_dq_numerical['thumb']
            #     self.data['index']['dJ_dq'] = dJ_dq_numerical['index']
            #     self.data['thumb']['dnormal_dq'] = dnormal_dq_numerical['thumb']
            #     self.data['index']['dnormal_dq'] = dnormal_dq_numerical['index']
            #     self.data['thumb']['dnormal_denv_q'] = dnormal_denv_q_numerical['thumb']
            #     self.data['index']['dnormal_denv_q'] = dnormal_denv_q_numerical['index']
            #     self.data['thumb']['closest_pt_q_grad'] = dcontact_dq_numerical['thumb']
            #     self.data['index']['closest_pt_q_grad'] = dcontact_dq_numerical['index']
            #
            #     numerical_grad = torch.zeros_like(grad_g_contact)
            #     delta = 1e-4
            #     for d in range(grad_g_contact.shape[2]):
            #         print(d, 'contact')
            #         dx = torch.zeros_like(xu)
            #         dx = dx.reshape(N, -1)
            #         dx[:, d] = delta
            #         dx = dx.reshape(N, T, -1)
            #         # finite difference test
            #         self._preprocess(xu + dx)
            #         g_plus, _, _ = self.contact_constraints(xu + dx, False, False)
            #         self._preprocess(xu - dx)
            #         g_neg, _, _ = self.contact_constraints(xu - dx, False, False)
            #         numerical_grad[:, :, d] = (g_plus.reshape(N, -1) - g_neg.reshape(N, -1)) / (2 * delta)
            #
            #
            #     # print(torch.max(torch.abs(dJ_dq_numerical - dJ_dq)))
            #     # split into index and thumb
            #     numerical_grad_thumb = numerical_grad[:, 4*self.T:]#.reshape(N, self.T, 4, self.T, -1)
            #     numerical_grad_index = numerical_grad[:, :4*self.T]#.reshape(N, self.T, 4, self.T, -1)
            #     grad_h_thumb = grad_g_contact[:, 4*self.T:]#.reshape(N, self.T, 4, self.T, -1)
            #     grad_h_index = grad_g_contact[:, :4*self.T]#.reshape(N, self.T, 4, self.T, -1)
            #
            #     numerical_grad_thumb_sdf = numerical_grad_thumb[:, :self.T].reshape(N, self.T, self.T, -1)
            #     grad_thumb_sdf = grad_h_thumb[:, :self.T].reshape(N, self.T, self.T, -1)
            #     numerical_grad_index_sdf = numerical_grad_index[:, :self.T].reshape(N, self.T, self.T, -1)
            #     grad_index_sdf = grad_h_index[:, :self.T].reshape(N, self.T, self.T, -1)
            #
            #     numerical_grad_thumb_kin = numerical_grad_thumb[:, self.T:].reshape(N, self.T, 3, self.T, -1)
            #     numerical_grad_index_kin = numerical_grad_index[:, self.T:].reshape(N, self.T, 3, self.T, -1)
            #     grad_thumb_kin = grad_h_thumb[:, self.T:].reshape(N, self.T, 3, self.T, -1)
            #     grad_index_kin = grad_h_index[:, self.T:].reshape(N, self.T, 3, self.T, -1)
            #
            #     print(torch.max(torch.abs(numerical_grad - grad_g_contact)))
            #     print(torch.mean(torch.abs(numerical_grad - grad_g_contact)))
            #     print('')
            #
            g_dynamics, grad_g_dynamics, hess_g_dynamics = self.dynamics_constraints(
                xu.reshape(N, T, self.dx + self.du),
                compute_grads=compute_grads,
                compute_hess=compute_hess)

            print('g_dynamics', g_dynamics.abs().mean(), g_dynamics.abs().max())
            print('g_contact', g_contact.abs().mean(), g_contact.abs().max())
            g_contact = torch.cat((g_contact, g_dynamics), dim=1)

            if grad_g_contact is not None:

                if torch.any(torch.isinf(grad_g_contact)) or torch.any(torch.isnan(grad_g_contact)):
                    print('invalid grad g')
                if torch.any(torch.isinf(grad_g_dynamics)) or torch.any(
                        torch.isnan(grad_g_dynamics)):
                    print('invalid grad g')

                grad_g_contact = torch.cat((grad_g_contact, grad_g_dynamics), dim=1)
            if hess_g_contact is not None:
                hess_g_contact = torch.cat((hess_g_contact, hess_g_dynamics), dim=1)

        return g_contact, grad_g_contact, hess_g_contact
        # print(g_contact.shape, grad_g_contact.shape, hess_g_contact.shape)
        ## # TODO: hard code the hessian for collision check for now, need to fix it in the future
        # g_contact, grad_g_contact, hess_g_contact = self._collision_check(xu.reshape(-1, self.dx + self.du),
        #                                                                  check_flag='contact',
        #                                                                  compute_grads=compute_grads,
        #                                                                  compute_hess=False)
        # g_contact = g_contact
        # if grad_g_contact is not None:
        #    grad_g_contact = grad_g_contact
        # if hess_g_contact is not None:
        #    hess_g_contact = -hess_g_contact

        g_contact = g_contact.reshape(N, T, -1)
        num_contacts = g_contact.shape[2]
        g_contact = g_contact.reshape(N, -1)
        g = torch.cat((g, g_contact), dim=1)
        g = g.reshape(N, -1)

        # g = g_contact

        if compute_grads:
            # grad_g = grad_g.reshape(N, non_contact_con_dim, self.T * (self.dx + self.du))
            # grad_h_collision = torch.cat((grad_h_collision, torch.zeros((grad_h_collision.shape[0], grad_h_collision.shape[1], self.du)).to(grad_h_collision.device)), dim=-1)
            # grad_g_contact = grad_g_contact.reshape(N, self.T, -1, self.dx + self.du).permute(0, 2, 3, 1)
            # grad_g_contact = torch.diag_embed(grad_g_contact)  # (N, n_constraints, dx + du, T, T)
            # grad_g_contact = grad_g_contact.permute(0, 3, 1, 4, 2).reshape(N, -1, self.T * (self.dx + self.du))
            grad_g = torch.cat((grad_g.reshape(N, -1, self.T * (self.dx + self.du)), grad_g_contact), dim=1)
            # grad_g = grad_g_contact
            grad_g = grad_g.reshape(N, g.shape[1], self.T * (self.dx + self.du))

        else:
            return g, None, None

        if hess_g_contact is None:
            hess_g_contact = torch.zeros(
                (N, num_contacts * self.T, self.T, self.dx + self.du, self.T, self.dx + self.du)).to(
                hess_g.device)
        hess_g = hess_g.reshape(N, -1, self.T * (self.dx + self.du), self.T * (self.dx + self.du))
        hess_g = torch.cat((hess_g, hess_g_contact), dim=1)
        # hess_g = hess_g_contact
        return g, grad_g, hess_g

    def _collision_check_finger(self, x, finger_scene, compute_grads=True, compute_hess=True):
        N = x.shape[0]
        q = x[:, :8]
        assert x.shape[1] == 9
        theta = x[:, -1].unsqueeze(-1)
        full_q = partial_to_full_state(q)
        ret_scene = finger_scene.scene_collision_check(full_q, theta,
                                                       compute_gradient=compute_grads,
                                                       compute_hessian=compute_hess)
        h = -ret_scene.get('sdf').unsqueeze(1)
        grad_h_q = ret_scene.get('grad_sdf', None)
        hess_h_q = ret_scene.get('hess_sdf', None)
        grad_h_theta = ret_scene.get('grad_env_sdf', None)
        hess_h_theta = ret_scene.get('hess_env_sdf', None)
        grad_h = None
        hess_h = None
        if grad_h_q is not None:
            grad_h_action = torch.zeros((h.shape[0], self.du)).to(grad_h_q.device)
            grad_h = -torch.cat((grad_h_q[:, (0, 1, 2, 3, 12, 13, 14, 15)], grad_h_theta, grad_h_action),
                                dim=-1).unsqueeze(1)
        if hess_h_q is not None:
            hess_h = torch.zeros(N, 1, self.dx + self.du, self.dx + self.du, device=x.device)
            hess_h_q = hess_h_q[:, (0, 1, 2, 3, 12, 13, 14, 15)]
            hess_h_q = hess_h_q[:, :, (0, 1, 2, 3, 12, 13, 14, 15)]
            hess_h[:, :, :8, :8] = -hess_h_q.unsqueeze(1)
            hess_h[:, :, 8, 8] = -hess_h_theta.reshape(N, -1)

        if grad_h is not None:
            print(h.abs().mean(), h.abs().max(), torch.any(torch.isnan(grad_h)), torch.any(torch.isinf(grad_h)))
        # print(h)
        return h, grad_h, hess_h

    def _collision_check(self, x, check_flag, compute_grads=True, compute_hess=True):
        # this returns NEGATIVE sdf value
        if check_flag == 'collision':
            thumb_scene = self.thumb_collision_scene
            index_scene = self.index_collision_scene
        elif check_flag == 'contact':
            thumb_scene = self.thumb_contact_scene
            index_scene = self.index_contact_scene

        h_thumb, grad_h_thumb, hess_h_thumb = self._collision_check_finger(x,
                                                                           index_scene,
                                                                           compute_grads=compute_grads,
                                                                           compute_hess=compute_hess)

        h_index, grad_h_index, hess_h_index = self._collision_check_finger(x,
                                                                           thumb_scene,
                                                                           compute_grads=compute_grads,
                                                                           compute_hess=compute_hess)

        h = torch.cat((h_thumb, h_index), dim=1)
        if not compute_grads:
            return h, None, None

        grad_h = torch.cat((grad_h_thumb, grad_h_index), dim=1)

        if not compute_hess:
            return h, grad_h, None

        hess_h = torch.cat((hess_h_thumb, hess_h_index), dim=1)

        return h, grad_h, hess_h

    def _con_ineq(self, xu, compute_grads=True, compute_hess=False):
        N = xu.shape[0]
        T = xu.shape[1]

        h, grad_h, hess_h = self._inequality_constraints.eval(xu.reshape(-1, T, self.dx + self.du), compute_grads,
                                                              compute_hess=compute_hess)
        #
        # h, grad_h, hess_h = self._dynamics_constraints(xu, 'index', True, compute_hess)
        # grad_h = grad_h.reshape(N, -1, T, self.dx + self.du)[:, :, :, :8]
        #
        if self.plan_in_contact:
            # if compute_grads:
            # eps = 1e-6
            # delta = 1e-4
            # # compute derivative of contact jacobian numerically
            # self._preprocess(xu)
            # dJ_dq = {
            #     'thumb': self.data['thumb']['dJ_dq'],
            #     'index': self.data['index']['dJ_dq']
            # }
            #
            # dJ_dq_numerical = {
            #     'thumb': torch.zeros_like(dJ_dq['thumb']),
            #     'index': torch.zeros_like(dJ_dq['index'])
            # }
            #
            # dnormal_dq = {
            #     'thumb': self.data['thumb']['dnormal_dq'].clone(),
            #     'index': self.data['index']['dnormal_dq'].clone()
            # }
            #
            # dnormal_dq_numerical = {
            #     'thumb': torch.zeros_like(dnormal_dq['thumb']),
            #     'index': torch.zeros_like(dnormal_dq['index'])
            # }
            #
            # dnormal_denv_q = {
            #     'thumb': self.data['thumb']['dnormal_denv_q'].clone(),
            #     'index': self.data['index']['dnormal_denv_q'].clone()
            # }
            #
            # dnormal_denv_q_numerical = {
            #     'thumb': torch.zeros_like(dnormal_denv_q['thumb']),
            #     'index': torch.zeros_like(dnormal_denv_q['index'])
            # }
            # print(dnormal_denv_q['thumb'].shape)
            # print(dnormal_denv_q['thumb'].shape)
            #
            # for d in range(self.dx):
            #     print(d)
            #     dx = torch.zeros_like(xu)
            #     dx = dx.reshape(N * T, -1)
            #     dx[:, d] = delta
            #     dx = dx.reshape(N, T, -1)
            #     # finite difference test
            #     self._preprocess(xu + dx)
            #     Jplus_t = self.data['thumb']['contact_jacobian']
            #     Jplus_i = self.data['index']['contact_jacobian']
            #
            #     normalplus_t = self.data['thumb']['contact_normal']
            #     normalplus_i = self.data['index']['contact_normal']
            #
            #     self._preprocess(xu - dx)
            #     Jneg_t = self.data['thumb']['contact_jacobian']
            #     Jneg_i = self.data['index']['contact_jacobian']
            #
            #     normalneg_t = self.data['thumb']['contact_normal']
            #     normalneg_i = self.data['index']['contact_normal']
            #
            #     #print(Jplus_t.shape, Jneg_t.shape, Jplus_i.shape, Jneg_i.shape)
            #     #print(normalplus_t.shape, normalneg_t.shape, normalplus_i.shape, normalneg_i.shape)
            #     if d < 8:
            #         dJ_dq_numerical['thumb'][:, :, :, :, d] = (Jplus_t - Jneg_t) / (2 * delta)
            #         dJ_dq_numerical['index'][:, :, :, :, d] = (Jplus_i - Jneg_i) / (2 * delta)
            #         dnormal_dq_numerical['thumb'][:, :, :, d] = (normalplus_t - normalneg_t).reshape(N, T + 1,
            #                                                                                          -1) / (
            #                                                             2 * delta)
            #         dnormal_dq_numerical['index'][:, :, :, d] = (normalplus_i - normalneg_i).reshape(N, T + 1,
            #                                                                                          -1) / (
            #                                                             2 * delta)
            #     elif d == 8:
            #         dnormal_denv_q_numerical['thumb'][:, :, 0] = (normalplus_t - normalneg_t) / (2 * delta)
            #         dnormal_denv_q_numerical['index'][:, :, 0] = (normalplus_i - normalneg_i) / (2 * delta)
            #
            # self.data['thumb']['dJ_dq'] = dJ_dq_numerical['thumb']
            # self.data['index']['dJ_dq'] = dJ_dq_numerical['index']
            # self.data['thumb']['dnormal_dq'] = dnormal_dq_numerical['thumb']
            # self.data['index']['dnormal_dq'] = dnormal_dq_numerical['index']
            # self.data['thumb']['dnormal_denv_q'] = dnormal_denv_q_numerical['thumb']
            # self.data['index']['dnormal_denv_q'] = dnormal_denv_q_numerical['index']
            #
            # print('index_dnormal_dq', torch.cosine_similarity(dnormal_dq_numerical['index'][:, 1:].reshape(-1, 8) + eps,
            #                               dnormal_dq['index'][:, 1:].reshape(-1, 8) + eps, dim=-1).mean())
            #
            # print('thumb_dnormal_dq', torch.cosine_similarity(dnormal_dq_numerical['thumb'][:, 1:, :].reshape(-1, 8) + eps,
            #                               dnormal_dq['thumb'][:, 1:].reshape(-1, 8) + eps, dim=-1).mean())
            #
            # print('index_dJ_dq', torch.cosine_similarity(dJ_dq_numerical['index'][:, 1:].reshape(-1, 8) + eps,
            #                               dJ_dq['index'][:, 1:].reshape(-1, 8) + eps, dim=-1).mean())
            #
            # print('thumb_dJ_dq', torch.cosine_similarity(dJ_dq_numerical['thumb'][:, 1:, :].reshape(-1, 8) + eps,
            #                               dJ_dq['thumb'][:, 1:].reshape(-1, 8) + eps, dim=-1).mean())
            # #print(torch.cosine_similarity(dJ_dq_numerical['index'][:, 1:].reshape(-1, 8) + eps,
            # #                              dJ_dq['index'][:, 1:].reshape(-1, 8) + eps, dim=-1).mean())
            #
            # #print(torch.cosine_similarity(dJ_dq_numerical['thumb'][:, 1:, :].reshape(-1, 8) + eps,
            # #                              self.data['thumb']['contact_hessian'][:, 1:].reshape(-1, 8) + eps, dim=-1).mean())
            # #print(torch.cosine_similarity(dJ_dq_numerical['index'][:, 1:, :].reshape(-1, 8) + eps,
            # #                              self.data['index']['contact_hessian'][:, 1:].reshape(-1, 8) + eps,
            # #                              dim=-1).mean())
            #
            # print(torch.cosine_similarity(dnormal_denv_q_numerical['index'][:, 1:].reshape(-1, 8),
            #                               dnormal_denv_q['index'][:, 1:].reshape(-1, 8), dim=-1).mean())
            #
            # print(torch.cosine_similarity(dJ_dq_numerical['thumb'][:, 1:, :].reshape(-1, 8),
            #                               dJ_dq['thumb'][:, 1:].reshape(-1, 8), dim=-1).mean())
            #
            # print('')
            h_friction, grad_h_friction, hess_h_friction = self.friction_constraints(
                xu.reshape(-1, T, self.dx + self.du),
                compute_grads=compute_grads,
                compute_hess=compute_hess)
            print('h_friction', h_friction.mean(), h_friction.max())
            print('h_y', h.mean(), h.max())
            h = torch.cat((h, h_friction), dim=1)

            if grad_h_friction is not None:
                #
                # numerical_grad = torch.zeros_like(grad_h_friction)
                #
                # for d in range(grad_h_friction.shape[2]):
                #     print(d, 'friction')
                #     dx = torch.zeros_like(xu)
                #     dx = dx.reshape(N, -1)
                #     dx[:, d] = delta
                #     dx = dx.reshape(N, T, -1)
                #     # finite difference test
                #     self._preprocess(xu + dx)
                #     g_plus, _, _ = self.friction_constraints(xu + dx, False, False)
                #     self._preprocess(xu - dx)
                #     g_neg, _, _ = self.friction_constraints(xu - dx, False, False)
                #     print(g_neg)
                #     print(g_plus)
                #     print('')
                #     numerical_grad[:, :, d] = (g_plus.reshape(N, -1) - g_neg.reshape(N, -1)) / (2 * delta)
                #
                # print(torch.max(torch.abs(numerical_grad - grad_h_friction)))
                # print(torch.mean(torch.abs(numerical_grad - grad_h_friction)))
                #
                # # print(torch.max(torch.abs(dJ_dq_numerical - dJ_dq)))
                # # split into index and thumb
                # numerical_grad_thumb = numerical_grad[:, self.T:].reshape(N, self.T, self.T, -1)
                # numerical_grad_index = numerical_grad[:, :self.T].reshape(N, self.T, self.T, -1)
                # grad_h_thumb = grad_h_friction[:, self.T:].reshape(N, self.T, self.T, -1)
                # grad_h_index = grad_h_friction[:, :self.T].reshape(N, self.T, self.T, -1)
                # print('')
                if torch.any(torch.isinf(grad_h)) or torch.any(torch.isnan(grad_h)):
                    print('invalid grad g')
                if torch.any(torch.isinf(grad_h_friction)) or torch.any(
                        torch.isnan(grad_h_friction)):
                    print('invalid grad g')

                grad_h = torch.cat((grad_h.reshape(N, -1, self.T * (self.dx + self.du)), grad_h_friction), dim=1)
            if hess_h_friction is not None:
                hess_h = torch.cat((hess_h.reshape(N, -1,
                                                   self.T * (self.dx + self.du),
                                                   self.T * (self.dx + self.du)), hess_h_friction), dim=1)

        non_collision_con_dim = h.shape[1]

        if self._collision_checking_flag:
            # TODO: hard code the hessian for collision check for now, need to fix it in the future
            h_collision, grad_h_collision, hess_h_collision = self._collision_check(xu.reshape(-1, self.dx + self.du),
                                                                                    check_flag='collision',
                                                                                    compute_grads=compute_grads,
                                                                                    compute_hess=False)
            h_collision = h_collision.reshape(N, T, 2)
            h_collision = h_collision.reshape(N, T * 2)
            h = torch.cat((h, h_collision), dim=1)

        h = h.reshape(N, -1)

        if compute_grads:
            grad_h = grad_h.reshape(N, non_collision_con_dim, self.T * (self.dx + self.du))
            if self._collision_checking_flag:
                # grad_h_collision = torch.cat((grad_h_collision, torch.zeros((grad_h_collision.shape[0], grad_h_collision.shape[1], self.du)).to(grad_h_collision.device)), dim=-1)
                grad_h_collision = grad_h_collision.reshape(N, self.T, -1, self.dx + self.du).permute(0, 2, 3, 1)
                grad_h_collision = torch.diag_embed(grad_h_collision)  # (N, n_constraints, dx + du, T, T)
                grad_h_collision = grad_h_collision.permute(0, 3, 1, 4, 2).reshape(N, -1, self.T * (self.dx + self.du))
                grad_h = torch.cat((grad_h, grad_h_collision), dim=1)
                grad_h = grad_h.reshape(N, h.shape[1], self.T * (self.dx + self.du))
        else:
            return h, None, None
        # TODO: use zero for now, we might need to fix it if we want to use hessian
        # hess_h_collision = hess_h_collision.reshape(N, self.T, -1, self.dx, self.dx).permute(0, 2, 3, 4, 1)
        # hess_h_collision = torch.diag_embed(torch.diag_embed(hess_h_collision))  # (N, n_constraints, dx + du, dx + du, T, T, T)
        # hess_h_collision = hess_h_collision.permute(0, 4, 1, 5, 2, 6, 3).reshape(N, -1,
        #                                                     self.T * (self.dx),
        #                                                     self.T * (self.dx))
        # if self._collision_checking_flag:
        #    if hess_h_collision is None:
        #        hess_h_collision = torch.zeros(
        #            (N, 2 * self.T, self.T, self.dx + self.du, self.T, self.dx + self.du)).to(hess_h.device)
        #    hess_h = torch.cat((hess_h, hess_h_collision), dim=1)
        # hess_h = hess_h.reshape(N, h.shape[1], self.T * (self.dx + self.du), self.T * (self.dx + self.du))
        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], self.T * (self.dx + self.du), self.T * (self.dx + self.du),
                                 device=self.device)
            return h, grad_h, hess_h
        return h, grad_h, None

    def eval(self, augmented_trajectory):
        self._preprocess(augmented_trajectory)
        N = augmented_trajectory.shape[0]
        augmented_trajectory = augmented_trajectory.clone().reshape(N, self.T, -1)
        x = augmented_trajectory[:, :, :self.dx + self.du]

        J, grad_J, hess_J = self._objective(x)
        hess_J = hess_J + 0.1 * torch.eye(self.T * (self.dx + self.du), device=self.device).unsqueeze(0)
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

        # eps = 1e-4
        # dG = torch.zeros_like(dG)
        # print(dG.shape)

        # for i in range(dG.shape[2]):
        #     x = augmented_trajectory.clone().reshape(N, -1)
        #     dx = torch.zeros_like(augmented_trajectory.reshape(N, -1))
        #     dx[:, i] = eps
        #     dx = dx.reshape(N, self.T, -1)
        #
        #     if (i % self.T) in range(0, 10):
        #         self._preprocess(augmented_trajectory + dx)
        #
        #     Gplus, _, _ = self.combined_constraints(augmented_trajectory + dx,
        #                                             compute_grads=False,
        #                                             compute_hess=self.compute_hess)
        #     if (i % self.T) in range(0, 21):
        #         self._preprocess(augmented_trajectory - dx)
        #
        #     Gneg, _, _ = self.combined_constraints(augmented_trajectory - dx,
        #                                            compute_grads=False,
        #                                            compute_hess=self.compute_hess)
        #     dG[:, :, i] = (Gplus - Gneg) / (2 * eps)
        #     # print(Gplus[0, -2:])
        #     # print(Gneg[0, -2:])
        print(J.mean(), G.abs().mean(), G.abs().max())

        if hess_J is not None:
            hess_J_ext = torch.zeros(N, self.T, self.dx + self.du + self.dz, self.T, self.dx + self.du + self.dz,
                                     device=x.device)
            hess_J_ext[:, :, :self.dx + self.du, :, :self.dx + self.du] = hess_J.reshape(N, self.T, self.dx + self.du,
                                                                                         self.T, self.dx + self.du)
            hess_J = hess_J_ext.reshape(N, self.T * (self.dx + self.du + self.dz),
                                        self.T * (self.dx + self.du + self.dz))

        if hessG is not None:
            hessG.detach_()
        return grad_J.detach(), hess_J, K.detach(), grad_K.detach(), G.detach(), dG.detach(), hessG

    def update(self, start, goal=None, T=None):
        self.start = start
        if goal is not None:
            self.goal = goal

        # update functions that require start
        self.cost = vmap(partial(cost, start=self.start, goal=self.goal))
        self.grad_cost = vmap(jacrev(partial(cost, start=self.start, goal=self.goal)))
        self.hess_cost = vmap(hessian(partial(cost, start=self.start, goal=self.goal)))

        # want the offset between the current angle and the angle according to end effectors
        # we will use position of index fingertip
        fk_dict = chain.forward_kinematics(partial_to_full_state(self.start[:8]),
                                           frame_indices=torch.tensor([index_ee_link]))

        points_finger_frame = torch.tensor([0.005, 0.005, 0.0], device=self.start.device).unsqueeze(0)
        p = world_trans.compose(fk_dict[index_ee_name]).transform_points(points_finger_frame).reshape(
            -1) - valve_location
        if torch.sqrt(p[0] ** 2 + p[2] ** 2) < 0.05:
            # we are probably close enough to valve to have begun rask
            geometric_theta = torch.atan2(p[0], p[2])
            actual_theta = self.start[-1]
            theta_offset = geometric_theta - actual_theta

        else:
            theta_offset = 0
        self._equality_constraints = JointConstraint(
            partial(get_joint_equality_constraint, chain=self.chain, start_q=self.start)
        )

        self._inequality_constraints = JointConstraint(
            partial(get_joint_inequality_constraint, chain=self.chain, start_q=self.start)
        )

        if goal is not None:
            self.goal = goal

        if T is not None:
            self.T = T
            self.dh = self.dz * T
            self.dg = 2 * T

    def get_initial_xu(self, N):
        """
        use delta joint movement to get the initial trajectory
        the action (force at the finger tip) is not used. it is randomly intiailized
        the actual dynamics model is not used
        """
        # u = torch.randn(N, self.T, self.du, device=self.device) / 20
        u = 0.025 * torch.randn(N, self.T, self.du, device=self.device)
        # u = torch.zeros((N, self.T, self.du), device=self.device)
        x = [self.start.reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            next_q = x[-1][:, :8] + u[:, t]
            x.append(next_q)
        theta = torch.linspace(self.start[-1], self.goal.item(), self.T)
        theta = theta.repeat((N, 1)).unsqueeze(-1).to(self.start.device)
        x = torch.stack(x[1:], dim=1)
        x = torch.cat((x, theta), dim=-1)
        xu = torch.cat((x, u), dim=2)
        return xu


def cost(xu, start, goal):
    # TODO: check if the addtional term of the smoothness cost and running goal cost is necessary
    state = xu[:, :9]  # state dim = 9
    state = torch.cat((start.reshape(1, 9), state), dim=0)  # combine the first time step into it

    action = xu[:, 9: 9 + 8]  # action dim = 8
    next_q = state[:-1, :-1] + action
    action_cost = torch.sum((state[1:, :-1] - next_q) ** 2)
    action_cost = action_cost + 10 * torch.sum(action ** 2)

    smoothness_cost = 10 * torch.sum((state[1:] - state[:-1]) ** 2)
    smoothness_cost += 50 * torch.sum((state[1:, -1] - state[:-1, -1]) ** 2)  # weight the smoothness of theta more

    goal_cost = torch.sum((10 * (state[:, -1] - goal) ** 2), dim=0) + (10 * (state[-1, -1] - goal) ** 2).reshape(-1)

    # this goal cost is not very informative and takes a long time for the gradients to back propagate to each state.
    # thus, a better initialization is necessary for faster convergence
    return smoothness_cost + 10 * action_cost + goal_cost


class JointConstraint:

    def __init__(self, joint_constraint_function):
        self._fn = joint_constraint_function
        self.joint_constraint_fn = joint_constraint_function

        self._grad_fn = jacrev(joint_constraint_function, argnums=(0))

        self.grad_constraint = self._grad_g
        self.hess_constraint = jacfwd(self._grad_g, argnums=0)

        self._J, self._H, self._dH = None, None, None

    def _grad_g(self, q):
        dq = self._grad_fn(q)
        return dq

    def eval(self, qu, compute_grads=True, compute_hess=False):
        """
        :param qu: torch.Tensor of shape (T, 9 + 8) containing set of state + acton
        :return g: constraint values
        :return Dg: constraint gradient
        :return DDg: constraint hessian
        """
        N = qu.shape[0]
        T = qu.shape[1]
        constraints = self.joint_constraint_fn(qu)
        if not compute_grads:
            return constraints, None, None
        dq = self.grad_constraint(qu)
        dq = torch.stack([dq[i, :, i] for i in range(N)],
                         dim=0)  # data from different batches should not affect each other
        # dumb_ddq = torch.eye(qu.shape[-1]).repeat((constraints.shape[0], constraints.shape[1], 1, 1))
        # if compute_hess:
        #     ddq = self.hess_constraint(qu)
        # else:
        #     ddq = torch.zeros(dq.shape + dq.shape[2:])
        # TODO: this is a hack for now to make it work. We should compute hessian in the future
        ddq = torch.zeros(dq.shape + dq.shape[2:], device=qu.device)

        return constraints, dq, ddq

    def reset(self):
        self._J, self._h, self._dH = None, None, None


def get_joint_equality_constraint(qu, chain, start_q, initial_valve_angle=0, report=False):
    """

    :param qu: torch.Tensor (T, DoF) joint positions
    :param chain: pytorch kinematics chain

    :return constraints: torch.Tensor(N, 1) contsraints as specified above

    """
    N = qu.shape[0]
    T = qu.shape[1]
    valve_offset_for_index = -np.pi / 2.0
    valve_offset_for_thumb = np.pi / 2.0  # * 0.99
    theta = qu[:, :, 8]
    index_q = qu[:, :, :4]
    thumb_q = qu[:, :, 4:8]
    action = qu[:, :, 9:17]
    index_action = action[:, :, :4]
    thumb_action = action[:, :, 4:]
    finger_qs = [index_q, thumb_q]
    finger_names = [index_ee_name, thumb_ee_name]
    action_list = [index_action, thumb_action]
    # radii = [0.040, 0.038]
    # radii = [0.033, 0.035]
    radii = [0.0445, 0.038]
    constraint = {}
    q = torch.cat((start_q.repeat((N, 1, 1)), qu[:, :, :9]), dim=1)  # add the current time step in
    fk_dict = chain.forward_kinematics(partial_to_full_state(q[:, :, :8].reshape(-1, 8)),
                                       frame_indices=frame_indices)  # pytorch_kinematics only supprts one additional dim
    fk_desired_dict = chain.forward_kinematics(partial_to_full_state((q[:, :-1, :8] + action).reshape(-1, 8)),
                                               frame_indices=frame_indices)
    for finger_q, finger_name, r, ee_index, finger_action in zip(finger_qs, finger_names, radii, frame_indices,
                                                                 action_list):
        m = world_trans.compose(fk_dict[finger_name])
        points_finger_frame = torch.tensor([0.00, 0.03, 0.00], device=qu.device).unsqueeze(0)
        normals_finger_tip_frame = torch.tensor([0.0, 0.0, 1.0], device=qu.device).unsqueeze(0)
        # it is just the thrid column of the rotation matrix
        normals_finger_tip_frame = normals_finger_tip_frame / torch.norm(normals_finger_tip_frame, dim=1, keepdim=True)
        ee_p = m.transform_points(points_finger_frame).reshape((N, (T + 1), 1, 3))  # TODO: check if the dimension match
        ee_p = ee_p.squeeze(-2)
        # normals_world_frame = m.transform_normals(normals_finger_tip_frame).reshape(3)
        ee_mat = m.get_matrix().reshape(N, (T + 1), 4, 4)[:, :, :3, :3]
        # jac = chain.jacobian(partial_to_full_state(q[:, :, :8].reshape(-1, 8)), link_indices=ee_index) # pytorch_kinematics only supprts one additional dim
        # jac = jac.reshape((N, (T+1), jac.shape[-2], jac.shape[-1]))
        # jac = full_to_partial_state(jac)
        # jac = rotate_jac(jac, env.world_trans.get_matrix()[0, :3, :3]) # rotate jac into world frame 

        "1 st equality constraints, the finger tip movement in the tangential direction should match with the commanded movement"
        # jac = jac[:, :T] # do not consider the last state, since no action matches the last state
        # translation_jac = jac[:, :, :3, :]
        # delta_p_action = torch.matmul(translation_jac, action.unsqueeze(-1)).squeeze(-1) # FK(q_{i,t) + delta q_{i,t}} - FK(q_{i,t})
        # TODO: need to look at the actual contact point really, what is the jacobian on the contact point
        # NOTE: we could use jacobian to simulate the delta action to speed up, but we use ground truth now
        m_desired = world_trans.compose(fk_desired_dict[finger_name])
        ee_p_desired = m_desired.transform_points(points_finger_frame).reshape((N, T, 1, 3)).squeeze(-2)
        delta_p_desired = ee_p_desired - ee_p[:, :-1]

        delta_p = ee_p[:, 1:] - ee_p[:, :-1]
        # get the surface normal
        radial_vec = ee_p[:, :-1] - valve_location.unsqueeze(0).unsqueeze(
            0)  # do not consider action at the final timestep
        radial_vec[:, :, 1] = 0  # do not consider y axis
        # in our simple task, might need to ignore one dimension
        # surface_normal = - radial_vec / torch.linalg.norm(radial_vec, dim=-1).unsqueeze(-1) # the normal goes inwards for friction cone computation
        # tan_delta_p_desired = delta_p_desired - (delta_p_desired.unsqueeze(-2)@surface_normal.unsqueeze(-1)).squeeze(-1) * surface_normal
        # tan_delta_p = delta_p - (delta_p.unsqueeze(-2) @ surface_normal.unsqueeze(-1)).squeeze(-1) * surface_normal
        # constraint_list.append((tan_delta_p_desired - tan_delta_p).reshape(N, -1))
        # This version specifies a redundant constraint, which might not fit well with CSVTO
        surface_normal = - radial_vec / torch.linalg.norm(radial_vec, dim=-1).unsqueeze(
            -1)  # the normal goes inwards for friction cone computation
        tan_delta_p = delta_p - (delta_p.unsqueeze(-2) @ surface_normal.unsqueeze(-1)).squeeze(-1) * surface_normal
        base_x = torch.tensor([1, 0, 0]).float().to(surface_normal.device)
        base_x_hat = base_x - (base_x.repeat((surface_normal.shape[0], surface_normal.shape[1], 1)).unsqueeze(
            -2) @ surface_normal.unsqueeze(-1)).squeeze(-1) * surface_normal
        # base_x_hat = tan_delta_p / torch.linalg.norm(tan_delta_p, dim=-1).unsqueeze(-1)
        base_y_hat = torch.cross(surface_normal, base_x_hat, dim=-1)
        tan_delta_p_desired = delta_p_desired - (delta_p_desired.unsqueeze(-2) @ surface_normal.unsqueeze(-1)).squeeze(
            -1) * surface_normal
        tan_delta_p_desired_x_hat = (base_x_hat.unsqueeze(-2) @ tan_delta_p_desired.unsqueeze(-1)).squeeze((-1, -2))
        tan_delta_p_desired_y_hat = (base_y_hat.unsqueeze(-2) @ tan_delta_p_desired.unsqueeze(-1)).squeeze((-1, -2))
        tan_delta_p_x_hat = (base_x_hat.unsqueeze(-2) @ tan_delta_p.unsqueeze(-1)).squeeze((-1, -2))
        tan_delta_p_y_hat = (base_y_hat.unsqueeze(-2) @ tan_delta_p.unsqueeze(-1)).squeeze((-1, -2))
        constraint[f'{finger_name}_dynamics_x'] = tan_delta_p_x_hat - tan_delta_p_desired_x_hat
        constraint[f'{finger_name}_dynamics_y'] = tan_delta_p_y_hat - tan_delta_p_desired_y_hat

    if report:
        return constraint
    else:
        return torch.cat(list(constraint.values()), dim=-1)


def get_joint_inequality_constraint(qu, chain, start_q, report=False):
    """

    :param qu: torch.Tensor (Batch, T, DoF) joint positions
    :param chain: pytorch kinematics chain

    :return constraints: torch.Tensor(N, 1) contsraints as specified above

    """
    N = qu.shape[0]
    T = qu.shape[1]
    valve_offset_for_index = -np.pi / 2.0
    valve_offset_for_thumb = np.pi / 2.0  # * 0.99
    theta = qu[:, :, 8]
    index_q = qu[:, :, :4]
    thumb_q = qu[:, :, 4:8]
    action = qu[:, :, 9:17]
    index_action = action[:, :, :4]
    thumb_action = action[:, :, 4:]
    finger_qs = [index_q, thumb_q]
    finger_names = [index_ee_name, thumb_ee_name]
    action_list = [index_action, thumb_action]
    # radii = [0.040, 0.038]
    # radii = [0.033, 0.035]
    radii = [0.0445, 0.038]
    constraint = {}
    q = torch.cat((start_q.repeat((N, 1, 1)), qu[:, :, :9]), dim=1)  # add the current time step in
    fk_dict = chain.forward_kinematics(partial_to_full_state(q[:, :, :8].reshape(-1, 8)),
                                       frame_indices=frame_indices)  # pytorch_kinematics only supprts one additional dim
    fk_desired_dict = chain.forward_kinematics(partial_to_full_state((q[:, :-1, :8] + action).reshape(-1, 8)),
                                               frame_indices=frame_indices)

    for finger_q, finger_name, r, ee_index, finger_action in zip(finger_qs, finger_names, radii, frame_indices,
                                                                 action_list):
        m = world_trans.compose(fk_dict[finger_name])
        points_finger_frame = torch.tensor([0.00, 0.03, 0.00], device=qu.device).unsqueeze(0)
        normals_finger_tip_frame = torch.tensor([0.0, 0.0, 1.0], device=qu.device).unsqueeze(0)
        # it is just the thrid column of the rotation matrix
        normals_finger_tip_frame = normals_finger_tip_frame / torch.norm(normals_finger_tip_frame, dim=1, keepdim=True)
        ee_p = m.transform_points(points_finger_frame).reshape((N, (T + 1), 3))
        # ee_p = ee_p.squeeze(-2)
        # normals_world_frame = m.transform_normals(normals_finger_tip_frame).reshape(3)
        ee_mat = m.get_matrix().reshape(N, (T + 1), 4, 4)[:, :, :3, :3]
        jac = chain.jacobian(partial_to_full_state(q[:, :, :8].reshape(-1, 8)),
                             link_indices=ee_index)  # pytorch_kinematics only supprts one additional dim
        jac = jac.reshape((N, (T + 1), jac.shape[-2], jac.shape[-1]))
        jac = full_to_partial_state(jac)
        jac = rotate_jac(jac, env.world_trans.get_matrix()[0, :3, :3])  # rotate jac into world frame

        " 1st constraint: friction cone constraints"
        jac = jac[:, :T]  # do not consider the last state, since no action matches the last state
        translation_jac = jac[:, :, :3, :]
        translation_jac_T = translation_jac.permute(0, 1, 3, 2)
        force = (torch.linalg.inv(torch.matmul(translation_jac, translation_jac_T)) @ translation_jac) @ (
                env.joint_stiffness * torch.eye(8).to(jac.device)) @ action.unsqueeze(-1)
        radial_vec = ee_p[:, :-1] - valve_location.unsqueeze(0).unsqueeze(
            0)  # do not consider action at the final timestep
        radial_vec[:, :, 1] = 0  # do not consider y axis
        # # in our simple task, might need to ignore one dimension
        surface_normal = - radial_vec / torch.linalg.norm(radial_vec, dim=-1).unsqueeze(
            -1)  # the normal goes inwards for friction cone computation
        f_normal = (surface_normal.unsqueeze(-2) @ force).squeeze(-1) * surface_normal
        f_tan = force.squeeze(-1) - f_normal
        # TODO: f_tan @ f_normal is not 0, but a relatively small value.  might need to double check
        # constraint[f'{finger_name}_friction_cone'] = (torch.linalg.norm(f_tan, dim=-1) - friction_coefficient * torch.linalg.norm(f_normal, dim=-1)) / 1000
        # NOTE: we need to scale it down, since this constraint is not of the same magnitute as the other constraints

        "2nd constraint y range of the finger tip should be within a range"
        constraint_y_1 = -(valve_location[1] - ee_p[:, 1:, 1]) + 0.02  # do not consider the current time step
        constraint_y_2 = (valve_location[1] - ee_p[:, 1:, 1]) - 0.2
        constraint[f'{finger_name}_y_range_1'] = constraint_y_1
        constraint[f'{finger_name}_y_range_2'] = constraint_y_2

        # "3rd constraint: action should push in more than the actual movement"
        # m_desired = world_trans.compose(fk_desired_dict[finger_name])
        # ee_p_desired = m_desired.transform_points(points_finger_frame).reshape((N, T, 1, 3)).squeeze(-2)
        # delta_p_desired = ee_p_desired - ee_p[:, :-1]
        # delta_p = ee_p[:, 1:] - ee_p[:, :-1]
        # normal_delta_p = (delta_p.unsqueeze(-2) @ surface_normal.unsqueeze(-1)).squeeze((-1, -2))
        # normal_delta_p_desired = (delta_p_desired.unsqueeze(-2) @ surface_normal.unsqueeze(-1)).squeeze((-1,-2))
        # constraint[f'{finger_name}_push_in'] = (normal_delta_p - normal_delta_p_desired)

        "4th constraint: the surface normal should align"
        # for state
        # only consider x and z axis
        normals_world_frame = m.transform_normals(normals_finger_tip_frame).reshape((N, (T + 1), 3))
        normals_world_frame = normals_world_frame / ((1.0 - normals_world_frame[:, :, 1] ** 2) ** 0.5).unsqueeze(-1)
        normals_world_frame = normals_world_frame[:, 1:]  # do not consider the current time step
        normal_alignment = torch.sin(theta) * normals_world_frame[:, :, 0] + torch.cos(theta) * normals_world_frame[:,
                                                                                                :, 2]
        if finger_name == index_ee_name:
            normal_alignment += 0.0
        elif finger_name == thumb_ee_name:
            normal_alignment += 0.0
        else:
            raise ValueError('Invalid finger name')
        # for state + action
        # only consider x and z axis
        # normals_world_frame_desired = m_desired.transform_normals(normals_finger_tip_frame).reshape((N, T, 3))
        # normals_world_frame_desired = normals_world_frame_desired / ((1.0 - normals_world_frame_desired[:, :, 1] ** 2)**0.5).unsqueeze(-1)
        # normal_desired_alignment = torch.sin(theta) * normals_world_frame_desired[:, :, 0] + torch.cos(theta) * normals_world_frame_desired[:, :, 2]
        # if finger_name == index_ee_name:
        #     normal_desired_alignment += 0.0
        # elif finger_name == thumb_ee_name:
        #     normal_desired_alignment += 0.0
        # else:
        #     raise ValueError('Invalid finger name')

        # constraint[f'{finger_name}_normal_alignment'] = normal_alignment
        # constraint[f'{finger_name}_normal_desired_alignment'] = normal_desired_alignment

    if report:
        return constraint
    else:
        return torch.cat(list(constraint.values()), dim=-1)


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
    chain.to(device=params['device'])

    if params['controller'] == 'csvgd':
        problem_to_grasp = AllegroValveProblem(start,
                                               params['goal'] * 0,
                                               4,
                                               device=params['device'],
                                               chain=chain,
                                               finger_name='index',
                                               valve_asset_pos=env.valve_pose,
                                               valve_location=valve_location,
                                               collision_checking=params['collision_checking'],
                                               plan_in_contact=False)
        controller_to_grasp = PositionControlConstrainedSVGDMPC(problem_to_grasp, params)

        problem = AllegroValveProblem(start,
                                      params['goal'],
                                      params['T'],
                                      device=params['device'],
                                      chain=chain,
                                      finger_name='index',
                                      valve_asset_pos=env.valve_pose,
                                      valve_location=valve_location,
                                      collision_checking=params['collision_checking'],
                                      plan_in_contact=True)
        controller = PositionControlConstrainedSVGDMPC(problem, params)
    else:
        raise ValueError('Invalid controller')

    # first we move the hand to grasp the valve
    start = state['q'].reshape(9).to(device=params['device'])
    best_traj, _ = controller_to_grasp.step(start)

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
    # ax.set_xlim((0.8, 0.8))
    # ax.set_ylim((0.6, 0.7))
    # ax.set_zlim((1.35, 1.45))
    thumb_traj_history = []
    index_traj_history = []
    state = env.get_state()
    start = state['q'].reshape(9).to(device=params['device'])
    thumb_ee = state2ee_pos(start[:8], thumb_ee_name).squeeze(0)
    thumb_traj_history.append(thumb_ee.detach().cpu().numpy())
    index_ee = state2ee_pos(start[:8], index_ee_name).squeeze(0)
    index_traj_history.append(index_ee.detach().cpu().numpy())

    info_list = []

    for k in range(params['num_steps']):
        state = env.get_state()
        start = state['q'].reshape(9).to(device=params['device'])
        # for debugging
        current_theta = start[8]
        thumb_radial_vec = thumb_ee - valve_location  # do not consider action at the final timestep
        thumb_radial_vec[1] = 0  # do not consider y axis
        # in our simple task, might need to ignore one dimension
        thumb_surface_normal = - thumb_radial_vec / torch.linalg.norm(
            thumb_radial_vec) / 70  # the normal goes inwards for friction cone computation
        temp_for_plot = torch.stack((thumb_ee, thumb_ee + thumb_surface_normal), dim=0).detach().cpu().numpy()
        # ax.plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'red')
        index_radial_vec = index_ee - valve_location  # do not consider action at the final timestep
        index_radial_vec[1] = 0  # do not consider y axis
        # in our simple task, might need to ignore one dimension
        index_surface_normal = - index_radial_vec / torch.linalg.norm(
            index_radial_vec) / 70  # the normal goes inwards for friction cone computation
        temp_for_plot = torch.stack((index_ee, index_ee + index_surface_normal), dim=0).detach().cpu().numpy()
        # ax.plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'red')

        fk_start = chain.forward_kinematics(partial_to_full_state(start[:8]), frame_indices=frame_indices)
        # print(f"current theta: {current_theta}")

        actual_trajectory.append(state['q'].reshape(9).clone())
        # if k > 0:
        #     torch.cuda.synchronize()
        #     start_time = time.time()
        # if k == 0:
        # for debugging
        # if controller.x is not None:
        #     initial_theta = controller.x[0, 0, 8]
        #     print(f"initial theta: {initial_theta}")
        # fk_initial = chain.forward_kinematics(partial_to_full_state(controller.x[0, :8]), frame_indices=frame_indices)
        start_time = time.time()
        best_traj, trajectories = controller.step(start)
        print(f"solve time: {time.time() - start_time}")
        # for debugging
        # planned_theta = best_traj[0, 8]
        # print(f"planned theta: {planned_theta}")

        # add trajectory lines to sim
        if k >= 1:
            add_trajectories(trajectories, best_traj, chain, axes)

        # process the action
        ## end effector force to torque
        x = best_traj[0, :problem.dx + problem.du]
        x = x.reshape(1, problem.dx + problem.du)
        action = x[:, problem.dx:problem.dx + problem.du].to(device=env.device)
        if params['joint_friction'] > 0:
            action = action + params['joint_friction'] / env.joint_stiffness * torch.sign(action)
        action = action + start.unsqueeze(0)[:, :8]  # NOTE: this is required since we define action as delta action
        env.step(action)
        if params['hardware']:
            # ros_node.apply_action(action[0].detach().cpu().numpy())
            ros_node.apply_action(partial_to_full_state(action[0]).detach().cpu().numpy())
        # TODO: need to fix the compute hessian part
        # plan_thumb_ee = state2ee_pos(x[0, :8], thumb_ee_name).squeeze(0)
        # plan_index_ee = state2ee_pos(x[0, :8], index_ee_name).squeeze(0)

        # actual_thumb_ee = state2ee_pos(env.get_state()['q'][0, :8], thumb_ee_name).squeeze(0)
        # actual_index_ee = state2ee_pos(env.get_state()['q'][0, :8], index_ee_name).squeeze(0)

        # print(f'index_ee_diff: {torch.linalg.norm(plan_index_ee - actual_index_ee):.4f}, thumb_ee_diff: {torch.linalg.norm(plan_thumb_ee - actual_thumb_ee):.4f}')
        problem._preprocess(best_traj.unsqueeze(0))

        equality_eval_dict = problem._equality_constraints.joint_constraint_fn(best_traj.unsqueeze(0), report=True)
        equality_eval = torch.stack(list(equality_eval_dict.values()), dim=2)
        # distance = problem._con_eq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False)
        # print(f'1st step distance {distance[:, 0]}')
        # print(f'max_distance {torch.max(distance)}, min_distance {torch.min(distance)}')

        # TODO: this is not correct, this one does not include sdf constraint
        print(f"max equality constraint violation: 1st step: {equality_eval[0][0].max()}, all: {equality_eval.max()}")
        inequality_eval_dict = problem._inequality_constraints.joint_constraint_fn(best_traj.unsqueeze(0), report=True)
        # index_friction_constraint = inequality_eval_dict['allegro_hand_hitosashi_finger_finger_0_aftc_base_link_friction_cone']
        # thumb_friction_constraint = inequality_eval_dict['allegro_hand_oya_finger_3_aftc_base_link_friction_cone']
        # print(f"1st step friction cone constraint: index: {index_friction_constraint[0,0]}, thumb: {thumb_friction_constraint[0,0]}")
        inequality_eval = torch.stack(list(inequality_eval_dict.values()), dim=2)
        print(
            f"max inequality constraint violation: 1st step: {inequality_eval[0][0].max()}, all: {inequality_eval.max()}")
        print(problem.thumb_contact_scene.scene_collision_check(partial_to_full_state(x[:, :8]), x[:, 8],
                                                                compute_gradient=False, compute_hessian=False))
        # distance2surface = torch.sqrt((best_traj_ee[:, 2] - valve_location[2].unsqueeze(0)) ** 2 + (best_traj_ee[:, 0] - valve_location[0].unsqueeze(0))**2)
        distance2goal = (env.get_state()['q'][:, -1] - params['goal']).detach().cpu().item()
        print(distance2goal, env.get_state()['q'][-1])
        info_list.append({
            #   'distance': distance,
            'distance2goal': distance2goal,
            'equality_eval': equality_eval_dict,
            'inequality_eval': inequality_eval_dict
        })

        gym.clear_lines(viewer)
        # for debugging
        state = env.get_state()
        start = state['q'].reshape(9).to(device=params['device'])
        thumb_ee = state2ee_pos(start[:8], thumb_ee_name).squeeze(0)
        thumb_traj_history.append(thumb_ee.detach().cpu().numpy())
        temp_for_plot = np.stack(thumb_traj_history, axis=0)
        if k >= 2:
            ax_thumb.plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'gray', label='actual')
        index_ee = state2ee_pos(start[:8], index_ee_name).squeeze(0)
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
    problem.T = actual_trajectory.shape[0]
    # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    final_distance_to_goal = (actual_trajectory[:, -1] - params['goal']).abs()
    # final_distance_to_goal = torch.linalg.norm(
    #     chain.forward_kinematics(actual_trajectory[:, :7].reshape(-1, 7)).reshape(-1, 4, 4)[:, :2, 3] - params['goal'].unsqueeze(0),
    #     dim=1
    # )

    print(f'Controller: {params["controller"]} Final distance to goal: {torch.min(final_distance_to_goal)}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
             #  constr=constraint_val.cpu().numpy(),
             d2goal=final_distance_to_goal.cpu().numpy())
    return torch.min(final_distance_to_goal).cpu().numpy()


def add_trajectories(trajectories, best_traj, chain, axes=None):
    M = len(trajectories)
    if M > 0:
        # print(partial_to_full_state(best_traj[:, :8]).shape)
        # frame_indices = torch.tensor([index_ee_link, thumb_ee_link])
        best_traj_ee_fk_dict = chain.forward_kinematics(partial_to_full_state(best_traj[:, :8]),
                                                        frame_indices=frame_indices)
        initial_state = env.get_state()['q'][:, :8]
        whole_state = torch.cat((initial_state, best_traj[:-1, :8]), dim=0)
        desired_state = whole_state + best_traj[:, 9:17]
        desired_traj_ee_fk_dict = chain.forward_kinematics(partial_to_full_state(desired_state),
                                                           frame_indices=frame_indices)

        index_best_traj_ee = world_trans.compose(best_traj_ee_fk_dict[index_ee_name])  # .get_matrix()
        thumb_best_traj_ee = world_trans.compose(best_traj_ee_fk_dict[thumb_ee_name])  # .get_matrix()

        index_desired_best_traj_ee = world_trans.compose(desired_traj_ee_fk_dict[index_ee_name])
        thumb_desired_best_traj_ee = world_trans.compose(desired_traj_ee_fk_dict[thumb_ee_name])
        # index_best_traj_ee = best_traj_ee_fk_dict[index_ee_name].compose(world_trans).get_matrix()
        # thumb_best_traj_ee = best_traj_ee_fk_dict[thumb_ee_name].compose(world_trans).get_matrix()
        points_finger_frame = torch.tensor([0.00, 0.03, 0.00], device=best_traj.device).unsqueeze(0)
        index_best_traj_ee = index_best_traj_ee.transform_points(points_finger_frame).squeeze(1)
        thumb_best_traj_ee = thumb_best_traj_ee.transform_points(points_finger_frame).squeeze(1)
        desired_index_best_traj_ee = index_desired_best_traj_ee.transform_points(points_finger_frame).squeeze(1)
        desired_thumb_best_traj_ee = thumb_desired_best_traj_ee.transform_points(points_finger_frame).squeeze(1)
        # index_best_traj_ee = index_best_traj_ee[:, :3, 3]
        # thumb_best_traj_ee = thumb_best_traj_ee[:, :3, 3]

        traj_line_colors = np.random.random((3, M)).astype(np.float32)
        thumb_colors = np.array([0, 1, 0]).astype(np.float32)
        index_colors = np.array([0, 0, 1]).astype(np.float32)
        force_colors = np.array([0, 1, 1]).astype(np.float32)
        for e in env.envs:
            index_p = env.get_state()['index_pos'].reshape(1, 3).to(device=params['device'])
            thumb_p = env.get_state()['thumb_pos'].reshape(1, 3).to(device=params['device'])
            # p = torch.stack((s[:3].reshape(1, 3).repeat(M, 1),
            #                  trajectories[:, 0, :3]), dim=1).reshape(2 * M, 3).cpu().numpy()
            # p_best = torch.stack((s[:3].reshape(1, 3).repeat(1, 1), best_traj_ee[0, :3].unsqueeze(0)), dim=1).reshape(2, 3).cpu().numpy()
            # p[:, 2] += 0.005
            # gym.add_lines(viewer, e, 1, p_best, best_traj_line_colors)
            # gym.add_lines(viewer, e, M, p, traj_line_colors)
            # gym.add_lines(viewer, e, 1, index_best_force, force_colors)
            # gym.add_lines(viewer, e, 1, thumb_best_force, force_colors)

            T = best_traj.shape[0]
            for t in range(T - 1):
                # p = torch.stack((trajectories[:, t, :3], trajectories[:, t + 1, :3]), dim=1).reshape(2 * M, 3)
                # p = p.cpu().numpy()
                # p[:, 2] += 0.01
                index_p_best = torch.stack((index_best_traj_ee[t, :3], index_best_traj_ee[t + 1, :3]), dim=0).reshape(2,
                                                                                                                      3).cpu().numpy()
                thumb_p_best = torch.stack((thumb_best_traj_ee[t, :3], thumb_best_traj_ee[t + 1, :3]), dim=0).reshape(2,
                                                                                                                      3).cpu().numpy()
                desired_index_p_best = torch.stack((index_best_traj_ee[t, :3], desired_index_best_traj_ee[t + 1, :3]),
                                                   dim=0).reshape(2,
                                                                  3).cpu().numpy()
                desired_thumb_p_best = torch.stack((thumb_best_traj_ee[t, :3], desired_thumb_best_traj_ee[t + 1, :3]),
                                                   dim=0).reshape(2,
                                                                  3).cpu().numpy()
                if t == 0:
                    initial_thumb_ee = state2ee_pos(initial_state, thumb_ee_name).squeeze(0)
                    thumb_state_traj = torch.stack((initial_thumb_ee, thumb_best_traj_ee[0]), dim=0).cpu().numpy()
                    thumb_action_traj = torch.stack((initial_thumb_ee, desired_thumb_best_traj_ee[0]),
                                                    dim=0).cpu().numpy()
                    axes[1].plot3D(thumb_state_traj[:, 0], thumb_state_traj[:, 1], thumb_state_traj[:, 2], 'blue',
                                   label='desired next state')
                    axes[1].plot3D(thumb_action_traj[:, 0], thumb_action_traj[:, 1], thumb_action_traj[:, 2], 'green',
                                   label='raw commanded position')
                    initial_index_ee = state2ee_pos(initial_state, index_ee_name).squeeze(0)
                    index_state_traj = torch.stack((initial_index_ee, index_best_traj_ee[0]), dim=0).cpu().numpy()
                    index_action_traj = torch.stack((initial_index_ee, desired_index_best_traj_ee[0]),
                                                    dim=0).cpu().numpy()
                    axes[0].plot3D(index_state_traj[:, 0], index_state_traj[:, 1], index_state_traj[:, 2], 'blue',
                                   label='desired next state')
                    axes[0].plot3D(index_action_traj[:, 0], index_action_traj[:, 1], index_action_traj[:, 2], 'green',
                                   label='raw commanded position')
                gym.add_lines(viewer, e, 1, index_p_best, index_colors)
                gym.add_lines(viewer, e, 1, thumb_p_best, thumb_colors)
                gym.add_lines(viewer, e, 1, desired_index_p_best, np.array([0, 1, 1]).astype(np.float32))
                gym.add_lines(viewer, e, 1, desired_thumb_p_best, np.array([1, 1, 1]).astype(np.float32))
                # gym.add_lines(viewer, e, M, p, traj_line_colors)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)


if __name__ == "__main__":
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro.yaml').read_text())
    from tqdm import tqdm

    sim, gym, viewer = env.get_sim()

    # try:
    #     while True:
    #         state = env.get_state()
    #         state = state['q'].reshape(-1, 9)[:, :8]
    #         env.step(state)
    #         print('waiting for you to finish camera adjustment, ctrl-c when done')
    #         time.sleep(0.1)
    # except KeyboardInterrupt:
    #     pass
    results = {}

    for i in tqdm(range(config['num_trials'])):
        goal = -0.5 * torch.tensor([np.pi])
        # goal = goal + 0.025 * torch.randn(1) + 0.2
        for controller in config['controllers'].keys():
            env.reset()
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
            # set up params
            params = config.copy()
            if params['hardware']:
                from hardware.allegro_ros import Ros_Node

                ros_node = Ros_Node()
            params.pop('controllers')
            params.update(config['controllers'][controller])
            params['controller'] = controller
            params['goal'] = goal.to(device=params['device'])
            final_distance_to_goal = do_trial(env, params, fpath)
            # final_distance_to_goal = turn(env, params, fpath)

            if controller not in results.keys():
                results[controller] = [final_distance_to_goal]
            else:
                results[controller].append(final_distance_to_goal)
        print(results)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
