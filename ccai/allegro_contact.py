from isaac_victor_envs.utils import get_assets_dir
# from isaac_victor_envs.tasks.allegro import AllegroValveTurningEnv

import numpy as np
import pickle as pkl

import torch
import time
import yaml
import copy
import pathlib
from functools import partial
import itertools
from torch.func import vmap, jacrev, hessian, jacfwd

from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
from ccai.kernels import rbf_kernel, structured_rbf_kernel

from ccai.problem import ConstrainedSVGDProblem, IpoptProblem
from ccai.mpc.csvgd import Constrained_SVGD_MPC
# from ccai.mpc.ipopt import IpoptMPC

import time
import pytorch_volumetric as pv
import pytorch_kinematics as pk
# import pytorch3d.transforms as tf

import matplotlib.pyplot as plt
from ccai.utils.allegro_utils import *
import pytorch_kinematics.transforms as tf

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')


def euler_to_quat(euler):
    matrix = tf.euler_angles_to_matrix(euler, convention='XYZ')
    quat = tf.matrix_to_quaternion(matrix)
    return quat


def euler_to_angular_velocity(current_euler, next_euler):
    # using matrix
    # quaternion
    current_quat = euler_to_quat(current_euler)
    next_quat = euler_to_quat(next_euler)
    dquat = next_quat - current_quat
    con_quat = - current_quat  # conjugate
    con_quat[..., 0] = current_quat[..., 0]
    omega = 2 * tf.quaternion_raw_multiply(dquat, con_quat)[..., 1:]
    # TODO: quaternion and its negative are the same, but it is not true for angular velocity. Might have some bug here 
    return omega


class PositionControlConstrainedSteinTrajOpt(ConstrainedSteinTrajOpt):
    def __init__(self, problem, params):
        super().__init__(problem, params)
        self.torque_limit = params.get('torque_limit', 1)
        self.kp = params['kp']
        self.fingers = problem.fingers
        self.num_fingers = len(self.fingers)

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
            robot_joint_angles = xuz_copy[:, :-1, :4 * self.num_fingers]
            robot_joint_angles = torch.cat(
                (self.problem.start[:4 * self.num_fingers].reshape((1, 1, 4 * self.num_fingers)).repeat((N, 1, 1)),
                 robot_joint_angles), dim=1)

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
            min_x[:, :, self.problem.dx:self.problem.dx + 4 * self.problem.num_fingers] = min_u
            max_x[:, :, self.problem.dx:self.problem.dx + 4 * self.problem.num_fingers] = max_u
            torch.clamp_(xuz, min=min_x.reshape((N, -1)), max=max_x.reshape((N, -1)))

            # if self.problem.optimize_force and self.problem.turn:
            #     xuz = xuz.reshape((N, self.problem.T, -1))
            #     forces = xuz[:, :, -3 * self.problem.num_contacts:]
            #     for i, finger in enumerate(self.problem.contact_fingers):
            #         this_finger_forces = forces[:, :, 3 * i: 3 * i + 3]
            #         this_finger_force_mag = torch.norm(this_finger_forces, dim=-1)
            #         min_force_mag = self.min_force_mag[finger]
            #         force_mag_mult = torch.clamp(min_force_mag / this_finger_force_mag, min=1.0)
            #         #Replace nans with 1
            #         force_mag_mult[torch.isnan(force_mag_mult)] = 1.0
            #         # Print the number of entries > 1
            #         num_force_overrides = torch.sum(force_mag_mult > 1).item()
            #         # if num_force_overrides > 0:
            #         #     num_entries = force_mag_mult.numel()
            #         #     pct_override = num_force_overrides / num_entries
            #             # print(f'Finger {finger} Num force overrides:{pct_override:.3f}', )
            #         xuz[:, :, -3 * (self.problem.num_contacts-i): -3 * (self.problem.num_contacts-i) + 3] *= force_mag_mult.unsqueeze(-1)
            #     xuz = xuz.reshape((N, -1))
    def resample(self, xuz):
        xuz = xuz.to(dtype=torch.float32)
        self.problem._preprocess(xuz)
        return super().resample(xuz)


class PositionControlConstrainedSVGDMPC(Constrained_SVGD_MPC):

    def __init__(self, problem, params):
        super().__init__(problem, params)
        self.solver = PositionControlConstrainedSteinTrajOpt(problem, params)


class AllegroObjectProblem(ConstrainedSVGDProblem):

    def __init__(self,
                 dx,
                 du,
                 start,
                 goal,
                 T,
                 chain,
                 object_location,
                 object_type,
                 object_asset_pos,
                 world_trans,
                 fingers=['index', 'middle', 'ring', 'thumb'],
                 contact_fingers=['index', 'middle', 'ring', 'thumb'],
                 regrasp_fingers=[],
                 obj_dof=1,
                 obj_ori_rep='euler',
                 obj_joint_dim=0,
                 device='cuda:0',
                 moveable_object=False,
                 min_force_dict=None,
                 *args, **kwargs):
        """
        obj_dof: DoF of the object, The max number is 6, It's the DoF for the rigid body, not including any joints within the object. 
        obj_joint_dim: It's the DoF of the joints within the object, excluding those are rigid body DoF.
        """

        super().__init__(start, goal, T, device)
        self.dx, self.du = dx, du
        self.dg_per_t = 0
        self.dg_constant = 0
        self.device = device
        self.dt = 0.1
        self.T = T
        self.start = start
        self.goal = goal
        self.K = rbf_kernel
        self.squared_slack = True
        self.compute_hess = False

        # make sure fingers is the wright order
        all_fingers = ['index', 'middle', 'ring', 'thumb']
        self.fingers = [f for f in all_fingers if f in fingers]
        self.num_fingers = len(self.fingers)
        self.obj_dof = obj_dof
        self.obj_ori_rep = obj_ori_rep
        self.obj_joint_dim = obj_joint_dim
        self.object_location = object_location
        self.alpha = 10
        self.d = 32 + self.obj_dof
        self._base_dz = self.num_fingers * 8

        self.data = {}

        ##### Allegro hand kinematics ######
        self.chain = chain
        self.joint_index = {
            'index': [0, 1, 2, 3],
            'middle': [4, 5, 6, 7],
            'ring': [8, 9, 10, 11],
            'thumb': [12, 13, 14, 15]
        }

        self.all_joint_index = sum([self.joint_index[finger] for finger in self.fingers], [])
        self.obj_pos_index = [16 + idx for idx in range(self.obj_dof)]
        self.control_index = [16 + obj_dof + idx for idx in self.all_joint_index]
        self.all_var_index = self.all_joint_index + self.obj_pos_index + self.control_index
        if kwargs.get('optimize_force', False):
            self.d += 12
            if kwargs.get('env_force', False):
                self.d += 3
            # indices for contact forces
            self._contact_force_indices_dict = {
                'index': [0, 1, 2],
                'middle': [3, 4, 5],
                'ring': [6, 7, 8],
                'thumb': [9, 10, 11]
            }
            self.contact_force_indices_dict = {
                'index': [-9, -8, -7],
                'middle': [-6, -5, -4],
                'thumb': [-3, -2, -1]
            }
            self._contact_force_indices = [self._contact_force_indices_dict[finger] for finger in contact_fingers]
            self._contact_force_indices = list(itertools.chain.from_iterable(self._contact_force_indices))
            self.contact_force_indices = [32 + self.obj_dof + idx for idx in self._contact_force_indices]
            if kwargs.get('env_force', False):
                self.contact_force_indices += [32 + self.obj_dof + 12, 32 + self.obj_dof + 13, 32 + self.obj_dof + 14]
            self.all_var_index = self.all_var_index + self.contact_force_indices

        self.ee_names = {
            'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
        }
        self.ee_link_idx = {finger: chain.frame_to_idx[ee_name] for finger, ee_name in self.ee_names.items()}
        self.frame_indices = torch.tensor([self.ee_link_idx[finger] for finger in self.fingers])

        ##### SDF for robot and environment ######
        self.world_trans = world_trans.to(device=device)
        if object_type == 'cuboid_valve':
            asset_object = get_assets_dir() + '/valve/valve_cuboid.urdf'
        elif object_type == 'cylinder_valve':
            asset_object = get_assets_dir() + '/valve/valve_cylinder.urdf'
        elif object_type == 'screwdriver':
            asset_object = get_assets_dir() + '/screwdriver/screwdriver.urdf'
        elif object_type == "card":
            asset_object = get_assets_dir() + '/card/card.urdf'
        self.object_asset_pos = object_asset_pos
        self.moveable_object = moveable_object
        chain_object = pk.build_chain_from_urdf(open(asset_object).read())
        chain_object = chain_object.to(device=device)
        if 'valve' in object_type:
            object_sdf = pv.RobotSDF(chain_object, path_prefix=get_assets_dir() + '/valve',
                                     use_collision_geometry=False)
        elif 'screwdriver' in object_type:
            object_sdf = pv.RobotSDF(chain_object, path_prefix=get_assets_dir() + '/screwdriver',
                                     use_collision_geometry=False)
        elif 'card' in object_type:
            object_sdf = pv.RobotSDF(chain_object, path_prefix=get_assets_dir() + '/card',
                                     use_collision_geometry=False)
        self.object_sdf = object_sdf
        robot_sdf = pv.RobotSDF(chain, path_prefix=get_assets_dir() + '/xela_models',
                                use_collision_geometry=False)

        scene_trans = world_trans.inverse().compose(
            pk.Transform3d(device=device).translate(object_asset_pos[0], object_asset_pos[1], object_asset_pos[2]))

        # contact checking
        collision_check_links = [self.ee_names[finger] for finger in self.fingers]
        self.contact_scenes = pv.RobotScene(robot_sdf, object_sdf, scene_trans,
                                            collision_check_links=collision_check_links,
                                            softmin_temp=1.0e3,
                                            points_per_link=1000,
                                            )

        # self.contact_scenes.visualize_robot(torch.zeros(16).to(self.device), self.start[4*self.num_fingers:].to(self.device))
        ###### Joint limits ########
        # NOTE: DEBUG only, set the joint limit to be very large for now.
        index_x_max = torch.tensor([0.47, 1.6099999999, 1.7089999, 1.61799999]) + 10.05
        index_x_min = torch.tensor([-0.47, -0.195999999999, -0.174000000, -0.227]) - 10.05
        thumb_x_max = torch.tensor([1.396, 1.1629999999999, 1.644, 1.71899999]) + 10.05
        thumb_x_min = torch.tensor([0.26, -0.1049999999, -0.1889999999, -0.162]) - 10.05
        joint_min = {'index': index_x_min, 'middle': index_x_min, 'ring': index_x_min, 'thumb': thumb_x_min}
        joint_max = {'index': index_x_max, 'middle': index_x_max, 'ring': index_x_max, 'thumb': thumb_x_max}
        self.x_max = torch.cat([joint_max[finger] for finger in self.fingers])
        self.x_min = torch.cat([joint_min[finger] for finger in self.fingers])
        self.robot_joint_x_max = torch.cat([joint_max[finger] for finger in self.fingers])
        self.robot_joint_x_min = torch.cat([joint_min[finger] for finger in self.fingers])
        # update x_max with valve angle
        if self.moveable_object:
            obj_x_max = 10.0 * np.pi * torch.ones(self.obj_dof)
            obj_x_min = -10.0 * np.pi * torch.ones(self.obj_dof)
        else:
            obj_x_max = start[-self.obj_dof:].cpu() + 1e-3
            obj_x_min = start[-self.obj_dof:].cpu() - 1e-3

        self.x_max = torch.cat((self.x_max, obj_x_max))
        self.x_min = torch.cat((self.x_min, obj_x_min))
        if self.du > 0:
            self.u_max = torch.ones(4 * self.num_fingers) * np.pi / 5
            self.u_min = - torch.ones(4 * self.num_fingers) * np.pi / 5
            self.x_max = torch.cat((self.x_max, self.u_max))
            self.x_min = torch.cat((self.x_min, self.u_min))

        if kwargs.get('optimize_force', False):
            max_f = torch.ones(3 * len(contact_fingers)) * 10
            min_f = torch.ones(3 * len(contact_fingers)) * -10
            self.x_max = torch.cat((self.x_max, max_f))
            self.x_min = torch.cat((self.x_min, min_f))
            if kwargs.get('env_force', False):
                self.x_max = torch.cat((self.x_max, torch.ones(3) * 10))
                self.x_min = torch.cat((self.x_min, torch.ones(3) * -10))

        #### functorch functions ######
        self.grad_kernel = jacrev(rbf_kernel, argnums=0)
        self.cost = vmap(partial(self._cost, start=self.start, goal=self.goal))
        self.grad_cost = vmap(jacrev(partial(self._cost, start=self.start, goal=self.goal)))
        self.hess_cost = vmap(hessian(partial(self._cost, start=self.start, goal=self.goal)))
        self.singularity_constr = vmap(self._singularity_constr)
        self.grad_singularity_constr = vmap(jacrev(self._singularity_constr))
        self.grad_euler_to_angular_velocity = jacrev(euler_to_angular_velocity, argnums=(0, 1))

    def _preprocess(self, xu, projected_diffusion=False):
        N = xu.shape[0]
        if not projected_diffusion:
            xu = xu.reshape(N, self.T, -1)
            x = xu[:, :, :self.dx]
            # expand to include start
            x_expanded = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)
        else:
            x_expanded = xu[:, :, :self.dx]
        q = x_expanded[:, :, :4 * self.num_fingers]
        theta = x_expanded[:, :, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof]
        if self.obj_ori_rep == 'axis_angle':
            theta = axis_angle_to_euler(theta).float()
        elif self.obj_ori_rep == 'euler':
            pass
        else:
            raise NotImplementedError
        self._preprocess_fingers(q, theta)

    def _preprocess_fingers(self, q, theta):
        N, _, _ = q.shape
        obj_dof = theta.shape[-1]

        # reshape to batch across time
        q_b = q.reshape(-1, 4 * self.num_fingers)
        theta_b = theta.reshape(-1, obj_dof)
        if self.obj_joint_dim > 0:
            theta_obj_joint = torch.zeros((theta_b.shape[0], self.obj_joint_dim),
                                          device=theta_b.device)  # add an additional dimension for the cap of the screw driver
            # the cap does not matter for the task, but needs to be included in the state for the model
            theta_b = torch.cat((theta_b, theta_obj_joint), dim=1)
        full_q = partial_to_full_state(q_b, fingers=self.fingers)
        ret_scene = self.contact_scenes.scene_collision_check(full_q, theta_b,
                                                              compute_gradient=True,
                                                              compute_hessian=False)
        for i, finger in enumerate(self.fingers):
            self.data[finger] = {}
            self.data[finger]['sdf'] = ret_scene['sdf'][:, i].reshape(N, self.T + 1)
            # reshape and throw away data for unused fingers
            grad_g_q = ret_scene.get('grad_sdf', None)
            self.data[finger]['grad_sdf'] = grad_g_q[:, i].reshape(N, self.T + 1, 16)

            # contact jacobian
            contact_jacobian = ret_scene.get('contact_jacobian', None)
            self.data[finger]['contact_jacobian'] = contact_jacobian[:, i].reshape(N, self.T + 1, 3, 16)

            # contact hessian
            contact_hessian = ret_scene.get('contact_hessian', None)
            contact_hessian = contact_hessian[:, i].reshape(N, self.T + 1, 3, 16, 16)  # [:, :, :, self.all_joint_index]
            # contact_hessian = contact_hessian[:, :, :, :, self.all_joint_index]  # shape (N, T+1, 3, 8, 8)

            # gradient of contact point
            d_contact_loc_dq = ret_scene.get('closest_pt_q_grad', None)
            d_contact_loc_dq = d_contact_loc_dq[:, i].reshape(N, self.T + 1, 3, 16)  # [:, :, :, self.all_joint_index]
            self.data[finger]['closest_pt_q_grad'] = d_contact_loc_dq
            self.data[finger]['contact_hessian'] = contact_hessian
            self.data[finger]['closest_pt_world'] = ret_scene['closest_pt_world'][:, i]
            self.data[finger]['contact_normal'] = ret_scene['contact_normal'][:, i]

            # gradient of contact normal
            self.data[finger]['dnormal_dq'] = ret_scene['dnormal_dq'][:, i].reshape(N, self.T + 1, 3, 16)  # [:, :, :,
            # self.all_joint_index]

            self.data[finger]['dnormal_denv_q'] = ret_scene['dnormal_denv_q'][:, i, :, :obj_dof]
            self.data[finger]['grad_env_sdf'] = ret_scene['grad_env_sdf'][:, i, :obj_dof]
            dJ_dq = contact_hessian
            self.data[finger]['dJ_dq'] = dJ_dq  # Jacobian of the contact point

    def _cost(self, xu, start, goal):
        start_q = partial_to_full_state(start[None, :self.num_fingers * 4], self.fingers)[:, self.all_joint_index]
        q = partial_to_full_state(xu[:, :self.num_fingers * 4], self.fingers)[:, self.all_joint_index]
        q = torch.cat((start_q, q), dim=0)
        delta_q = partial_to_full_state(xu[:, self.dx:self.dx + 4 * self.num_fingers], self.fingers)

        smoothness_cost = 10 * torch.sum((q[1:] - q[-1]) ** 2)
        action_cost = 10 * torch.sum(delta_q ** 2)
        return smoothness_cost + action_cost

    def _objective(self, x):
        x = x[:, :, :self.dx + self.du]
        N = x.shape[0]
        J, grad_J = self.cost(x), self.grad_cost(x)

        N = x.shape[0]
        return (self.alpha * J.reshape(N),
                self.alpha * grad_J.reshape(N, -1),
                None)
        # self.alpha * hess_J.reshape(N, self.T * (self.dx + self.du), self.T * (self.dx + self.du)))

    def _step_size_limit(self, xu):
        N, T, _ = xu.shape
        d_steps = self.num_fingers * 4
        u = xu[:, :, self.dx:self.dx + d_steps]
        # full_u = partial_to_full_state(u, fingers=self.fingers)

        max_step_size = 0.05
        h_plus = u - max_step_size
        h_minus = -u - max_step_size
        h = torch.stack((h_plus, h_minus), dim=2)  # N x T x 2 x du

        grad_h = torch.zeros(N, T, 2, d_steps, T, self.d, device=xu.device)
        hess_h = torch.zeros(N, T * 2 * d_steps, T * self.d, device=xu.device)
        eye = torch.eye(16, device=xu.device)

        # assign gradients
        T_range = torch.arange(0, T, device=xu.device)
        lower = 16 + self.obj_dof
        upper = lower + 16
        grad_h[:, T_range, 0, :, T_range, lower:upper] = eye[self.all_joint_index]
        grad_h[:, T_range, 1, :, T_range, lower:upper] = -eye[self.all_joint_index]

        return h.reshape(N, -1), grad_h.reshape(N, -1, T * self.d), hess_h

    def _singularity_constr(self, contact_jac):
        # this will be vmapped
        A = contact_jac @ contact_jac.transpose(-1, -2)
        eig = torch.linalg.eigvals(A).abs()
        eig = torch.topk(eig, 2, dim=-1).values
        manipulability = eig[0] / eig[1] - 50
        return manipulability
        # manipulability = torch.sqrt(torch.prod(eig, dim=-1))
        # return 0.0001 - manipulability

    @all_finger_constraints
    def _singularity_constraint(self, xu, finger_name, compute_grads=True, compute_hess=False):

        # assume access to class member variables which have already done some of the computation
        N, T, d = xu.shape
        q = xu[:, :, :4 * self.num_fingers]
        contact_jac = self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, 16)[
                      :, 1:, :, self.all_joint_index]

        # compute constraint value
        h = self.singularity_constr(contact_jac.reshape(-1, 3, self.num_fingers * 4))
        h = h.reshape(N, -1)
        dh = 1
        # compute the gradient
        if compute_grads:
            dh_djac = self.grad_singularity_constr(contact_jac.reshape(-1, 3, self.num_fingers * 4))

            djac_dq = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, 4 * self.num_fingers, 4 * self.num_fingers)[
                      :, 1:]

            dh_dq = dh_djac.reshape(N, T, dh, -1) @ djac_dq.reshape(N, T, -1, 4 * self.num_fingers)
            grad_h = torch.zeros(N, dh, T, T, d, device=self.device)
            T_range = torch.arange(T, device=self.device)
            T_range_minus = torch.arange(T - 1, device=self.device)
            T_range_plus = torch.arange(1, T, device=self.device)
            grad_h[:, :, T_range_plus, T_range_minus, :4 * self.num_fingers] = dh_dq[:, 1:].transpose(1, 2)
            grad_h = grad_h.transpose(1, 2).reshape(N, -1, T * d)
        else:
            return h, None, None

        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], T * d, T * d, device=self.device)
            return h, grad_h, hess_h

        return h, grad_h, None

    def _contact_constraints(self, q, finger_name, compute_grads=True, compute_hess=False, terminal=False, projected_diffusion=False):
        """
            Computes contact constraints
            constraint that sdf value is zero
        """
        N, T, _ = q.shape
        T_offset = 0 if projected_diffusion else 1
        d = self.d
        # Retrieve pre-processed data
        ret_scene = self.data[finger_name]
        g = ret_scene.get('sdf').reshape(N, T if projected_diffusion else T + 1, 1)  # - 1.0e-3
        # for some reason the thumb penetrates the object
        # if finger_name == 'thumb':
        #    g = g - 1.0e-3
        grad_g_q = ret_scene.get('grad_sdf', None)
        hess_g_q = ret_scene.get('hess_sdf', None)
        grad_g_theta = ret_scene.get('grad_env_sdf', None)
        hess_g_theta = ret_scene.get('hess_env_sdf', None)

        # Ignore first value (if not doing projected diffusion), as it is the start state
        g = g[:, T_offset:].reshape(N, -1)

        # If terminal, only consider last state
        if terminal:
            g = g[:, -1].reshape(N, 1)

        if not self.moveable_object:
            grad_g_theta = torch.zeros_like(grad_g_theta)

        # print(g[:, -1])
        if compute_grads:
            T_range = torch.arange(T, device=q.device)
            # compute gradient of sdf
            grad_g = torch.zeros(N, T, T, d, device=q.device)
            grad_g[:, T_range, T_range, :16] = grad_g_q[:, T_offset:]
            grad_g[:, T_range, T_range, 16: 16 + self.obj_dof] = grad_g_theta.reshape(N, T + T_offset, self.obj_dof)[:, T_offset:]
            grad_g = grad_g.reshape(N, -1, T, d)
            grad_g = grad_g.reshape(N, -1, T * d)
            if terminal:
                grad_g = grad_g[:, -1].reshape(N, 1, T * d)
        else:
            return g, None, None

        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * d, T * d, device=q.device)
            return g, grad_g, hess

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
        x_axis = torch.linalg.cross(y_axis, z_axis, dim=-1)
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
        initialize with object not moving at all
        """

        u = 0.025 * torch.randn(N, self.T, self.du, device=self.device)

        x = [self.start.reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            next_q = x[-1][:, :4 * self.num_fingers] + u[:, t, :4 * self.num_fingers]
            x.append(next_q)

        x = torch.stack(x[1:], dim=1)

        # if valve angle in state
        if self.dx == (4 * self.num_fingers + self.obj_dof):
            theta = self.start[-self.obj_dof:].unsqueeze(0).repeat((N, self.T, 1))
            x = torch.cat((x, theta), dim=-1)

        xu = torch.cat((x, u), dim=2)
        return xu


class AllegroRegraspProblem(AllegroObjectProblem):

    def __init__(self,
                 start,
                 goal,
                 T,
                 chain,
                 object_location,
                 object_type,
                 world_trans,
                 object_asset_pos,
                 regrasp_fingers=['index', 'middle', 'ring', 'thumb'],
                 contact_fingers=[],
                 obj_dof=1,
                 obj_ori_rep='euler',
                 obj_joint_dim=0,
                 device='cuda:0',
                 moveable_object=False,
                 optimize_force=False,
                 default_dof_pos=None,
                 desired_ee_in_world_frame=False,
                 obj_dof_type=None,
                 *args, **kwargs):

        # object_location is different from object_asset_pos. object_asset_pos is 
        # used for pytorch volumetric. The asset of valve might contain something else such as a wall, a table
        # object_location is the location of the object joint, which is what we care for motion planning
        self.obj_dof_type = obj_dof_type
        if kwargs.get('fingers', None) is None:
            kwargs['fingers'] = regrasp_fingers + contact_fingers

        if kwargs.get('dx', None) is None:
            num_fingers = len(kwargs['fingers'])
            dx = 4 * num_fingers + obj_dof
            du = 4 * num_fingers
            if optimize_force:
                du += 3 * len(contact_fingers)
            if kwargs.get("env_force", False):
                du += 3
            kwargs['dx'] = dx
            kwargs['du'] = du

        super().__init__(start=start, goal=goal, T=T, chain=chain, object_location=object_location,
                         world_trans=world_trans, obj_dof=obj_dof,
                         obj_ori_rep=obj_ori_rep,
                         obj_joint_dim=obj_joint_dim, device=device,
                         object_asset_pos=object_asset_pos,
                         object_type=object_type,
                         moveable_object=moveable_object, optimize_force=optimize_force,
                         contact_fingers=contact_fingers,
                         regrasp_fingers=regrasp_fingers, default_dof_pos=default_dof_pos, *args, **kwargs)

        self.regrasp_fingers = regrasp_fingers
        self.num_regrasps = len(regrasp_fingers)
        self.regrasp_idx = [self.joint_index[finger] for finger in regrasp_fingers]
        self.regrasp_idx = list(itertools.chain.from_iterable(self.regrasp_idx))
        self._regrasp_dg_per_t = self.num_regrasps * 4
        self._regrasp_dg_constant = self.num_regrasps
        self._regrasp_dg = self._regrasp_dg_per_t * T + self._regrasp_dg_constant
        self._regrasp_dz = self.num_regrasps  # one contact constraints per finger
        self._regrasp_dh = self._regrasp_dz * T  # inequality

        if default_dof_pos is None:
            self.default_dof_pos = torch.cat((torch.tensor([[0.1, 0.5, 0.5, 0.5]]).float().to(device=self.device),
                                              torch.tensor([[-0.1, 0.5, 0.65, 0.65]]).float().to(device=self.device),
                                              torch.tensor([[0., 0.5, 0.65, 0.65]]).float().to(device=self.device),
                                              torch.tensor([[1.2, 0.3, 0.2, 1.]]).float().to(device=self.device)),
                                             dim=1).to(self.device).reshape(-1)
        else:
            self.default_dof_pos = default_dof_pos.to(self.device)

        self.desired_ee_in_world_frame = desired_ee_in_world_frame
        if self.num_regrasps > 0:
            if desired_ee_in_world_frame:
                self.default_ee_locs = self._ee_locations_in_world(self.default_dof_pos)
            else:
                self.default_ee_locs = self._ee_locations_in_screwdriver(self.default_dof_pos,
                                                                     torch.zeros(self.obj_dof, device=self.device))

            # add a small amount of noise to ee loc default
            #self.default_ee_locs = self.default_ee_locs #+ 0.01 * torch.randn_like(self.default_ee_locs)
        else:
            self.default_ee_locs = None
        if self.obj_dof == 3:
            object_link_name = 'screwdriver_body'
        elif self.obj_dof == 1:
            object_link_name = 'valve'
        elif self.obj_dof == 6:
            object_link_name = 'card'
        self.obj_link_name = object_link_name

    def _ee_locations_in_screwdriver(self, q_rob, q_env):

        assert q_rob.shape[-1] == 16
        assert q_env.shape[-1] == self.obj_dof

        _q_env = q_env.clone()
        if self.obj_dof == 3:
            _q_env = torch.cat((q_env, torch.zeros_like(q_env[..., :1])), dim=-1)

        robot_trans = self.contact_scenes.robot_sdf.chain.forward_kinematics(q_rob.reshape(-1, 16))
        ee_locs = []

        for finger in self.regrasp_fingers:
            ee_locs.append(robot_trans[self.ee_names[finger]].get_matrix()[:, :3, -1])

        ee_locs = torch.stack(ee_locs, dim=1)

        # convert to scene base frame
        ee_locs = self.contact_scenes.scene_transform.inverse().transform_points(ee_locs)

        # convert to scene ee frame
        object_trans = self.contact_scenes.scene_sdf.chain.forward_kinematics(
            _q_env.reshape(-1, _q_env.shape[-1]))

        ee_locs = object_trans[self.obj_link_name].inverse().transform_points(ee_locs)

        return ee_locs.reshape(q_rob.shape[:-1] + (self.num_regrasps, 3))
    
    def _ee_locations_in_world(self, q_rob):
        assert q_rob.shape[-1] == 16
        robot_trans = self.contact_scenes.robot_sdf.chain.forward_kinematics(q_rob.reshape(-1, 16))
        ee_locs = []

        for finger in self.regrasp_fingers:
            ee_locs.append(robot_trans[self.ee_names[finger]].get_matrix()[:, :3, -1])

        ee_locs = torch.stack(ee_locs, dim=1)
        return ee_locs.reshape(q_rob.shape[:-1] + (self.num_regrasps, 3))

    def _cost(self, xu, start, goal):
        if self.num_regrasps == 0:
            return 0.0
        q = partial_to_full_state(xu[:, :self.num_fingers * 4], self.fingers)  # [:, self.regrasp_idx]
        theta = xu[:, self.num_fingers * 4:self.num_fingers * 4 + self.obj_dof]

        # ignore the rotation of the screwdriver
        if self.obj_dof == 3:
            mask = torch.tensor([1.0, 1.0, 0.0], device=xu.device)
            theta = theta * mask.reshape(1, 3)
        else:
            theta = theta * 0
        # print('--')
        # print(self._ee_locations_in_screwdriver(q, theta))
        # print(self.default_ee_locs)
        if self.desired_ee_in_world_frame:
            return 1000 * torch.sum((self.default_ee_locs - self._ee_locations_in_world(q)) ** 2)
        else:
            return 1000 * torch.sum((self.default_ee_locs - self._ee_locations_in_screwdriver(q, theta)) ** 2)

        # dof_pos = self.default_dof_pos[None, self.regrasp_idx]
        # return 10 * torch.sum((q - dof_pos) ** 2)

    @regrasp_finger_constraints
    def _contact_avoidance(self, xu, finger_name, compute_grads=True, compute_hess=False, projected_diffusion=False):
        h, grad_h, hess_h = self._contact_constraints(xu, finger_name, compute_grads, compute_hess, terminal=False, projected_diffusion=projected_diffusion)
        eps = torch.zeros_like(h)
        # eps[:, :-1] = 5e-3
        if self.obj_link_name == 'card':
            eps[:, :-1] = 1.5e-2
        else:
            eps[:, :-1] = 1.5e-2
        h = -h + eps
        if grad_h is not None:
            grad_h = -grad_h
        if hess_h is not None:
            hess_h = -hess_h
        return h, grad_h, hess_h

    @regrasp_finger_constraints
    def _terminal_contact_constraint(self, xu, finger_name, compute_grads=True, compute_hess=False, projected_diffusion=False):
        return self._contact_constraints(xu, finger_name, compute_grads, compute_hess, terminal=True, projected_diffusion=projected_diffusion)

    @regrasp_finger_constraints
    def _free_dynamics_constraints(self, q, delta_q, finger_name, compute_grads=True, compute_hess=False):
        N, T, _ = q.shape
        d = self.d

        x = q[:, :, self.joint_index[finger_name]]
        u = delta_q[:, :, self.joint_index[finger_name]]

        start_q = partial_to_full_state(self.start[:self.num_fingers * 4], self.fingers)[self.joint_index[finger_name]]
        # add start
        x = torch.cat((start_q.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)
        next_x = x[:, 1:]
        x = x[:, :-1]

        # compute constraint value - just a linear constraint
        g = (next_x - x - u).reshape(N, -1)
        if compute_grads:
            # all gradients are just multiples of identity
            eye = torch.eye(4, device=x.device).reshape(1, 1, 4, 4).repeat(N, T, 1, 1)
            eye = eye.permute(0, 2, 1, 3)
            grad_g = torch.zeros(N, 4, T, T, d, device=x.device)
            T_range = torch.arange(T, device=x.device)
            T_range_minus = torch.arange(T - 1, device=x.device)
            T_range_plus = torch.arange(1, T, device=x.device)

            # for masking out for assigning
            mask = torch.zeros_like(grad_g).bool()
            mask[:, :, T_range, T_range] = True
            mask_joint_index = torch.zeros_like(grad_g).bool()
            mask_joint_index[:, :, :, :, self.joint_index[finger_name]] = True

            mask_control_index = torch.zeros_like(grad_g).bool()
            mask_control_index[:, :, :, :, [idx + 16 + self.obj_dof for idx in self.joint_index[finger_name]]] = True

            grad_g[torch.logical_and(mask, mask_joint_index)] = eye.reshape(-1)
            grad_g[torch.logical_and(mask, mask_control_index)] = -eye.reshape(-1)

            mask = torch.zeros_like(grad_g).bool()
            mask[:, :, T_range_plus, T_range_minus] = True
            grad_g[torch.logical_and(mask, mask_joint_index)] = -eye[:, :, 1:].reshape(-1)
            grad_g = grad_g.permute(0, 2, 1, 3, 4).reshape(N, -1, T * d)
        else:
            return g, None, None

        if compute_hess:
            hess_g = torch.zeros(N, g.shape[1], T * d, T * d, device=self.device)
            return g, grad_g, hess_g
        return g, grad_g, None

    def _con_eq(self, xu, compute_grads=True, compute_hess=False, projected_diffusion=False):
        N, T = xu.shape[:2]
        q = xu[:, :, :self.num_fingers * 4]
        delta_q = xu[:, :, self.num_fingers * 4 + self.obj_dof:self.num_fingers * 8 + self.obj_dof]

        q = partial_to_full_state(q, fingers=self.fingers)
        delta_q = partial_to_full_state(delta_q, fingers=self.fingers)
        g_contact, grad_g_contact, hess_g_contact = self._terminal_contact_constraint(
            xu=xu.reshape(N, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess,
            projected_diffusion=projected_diffusion)

        g_dynamics, grad_g_dynamics, hess_g_dynamics = self._free_dynamics_constraints(
            q=q,
            delta_q=delta_q,
            compute_grads=compute_grads,
            compute_hess=compute_hess)

        g = torch.cat((g_contact, g_dynamics), dim=1)
        grad_g, hess_g = None, None
        if compute_grads:
            grad_g = torch.cat((grad_g_contact, grad_g_dynamics), dim=1)
        if compute_hess:
            hess_g = torch.cat((hess_g_contact, hess_g_dynamics), dim=1)

        return g, grad_g, hess_g

    def _con_ineq(self, xu, compute_grads=True, compute_hess=False, projected_diffusion=False):
        N, T = xu.shape[:2]
        h_contact, grad_h_contact, hess_h_contact = self._contact_avoidance(xu=xu.reshape(N, T, -1)[:, :, :self.dx + self.du],
                                                                            compute_grads=compute_grads,
                                                                            compute_hess=compute_hess,
                                                                            projected_diffusion=projected_diffusion)
        # print("avoidance", h_contact.max())
        return h_contact, grad_h_contact, hess_h_contact


class AllegroContactProblem(AllegroObjectProblem):

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
                 obj_dof_type=None,
                 obj_joint_dim=0,
                 optimize_force=False,
                 env_force=False,
                 turn=False,
                 obj_gravity=False,
                 device='cuda:0',
                 min_force_dict=None,
                 **kwargs):
        self.obj_dof_type = obj_dof_type
        self.obj_gravity = obj_gravity
        self.optimize_force = optimize_force
        self.turn = turn
        self.num_contacts = len(contact_fingers)
        self.contact_fingers = contact_fingers
        self.env_force = env_force
        num_fingers = self.num_contacts + len(regrasp_fingers)
        dx = 4 * num_fingers + obj_dof
        if optimize_force:
            du = (4 + 3) * num_fingers
            if env_force:
                du += 3
        else:
            du = 4 * num_fingers

        super().__init__(dx=dx, du=du, start=start, goal=goal,
                         T=T, chain=chain, object_location=object_location,
                         object_type=object_type, world_trans=world_trans, object_asset_pos=object_asset_pos,
                         fingers=regrasp_fingers + contact_fingers, obj_dof=obj_dof, obj_ori_rep=obj_ori_rep,
                         obj_joint_dim=obj_joint_dim, device=device, moveable_object=True,
                         contact_fingers=contact_fingers, env_force=env_force, optimize_force=optimize_force,
                         regrasp_fingers=regrasp_fingers, min_force_dict=min_force_dict,
                         **kwargs)
        self.min_force_dict = min_force_dict

        self.friction_coefficient = friction_coefficient
        self.dynamics_constr = vmap(self._dynamics_constr)
        self.grad_dynamics_constr = vmap(jacrev(self._dynamics_constr, argnums=(0, 1, 2, 3, 4)))
        if not optimize_force:
            self.force_equlibrium_constr = vmap(self._force_equlibrium_constr)
            self.grad_force_equlibrium_constr = vmap(jacrev(self._force_equlibrium_constr, argnums=(0, 1, 2, 3, 4)))
        else:
            self.force_equlibrium_constr = vmap(self._force_equlibrium_constr_w_force)
            self.grad_force_equlibrium_constr = vmap(
                jacrev(self._force_equlibrium_constr_w_force, argnums=(0, 1, 2, 3, 4, 5)))
            self.min_force_constr = vmap(self._min_force_constr, randomness='same')
            self.grad_min_force_constr = vmap(jacrev(self._min_force_constr, argnums=(0,)))

        self.friction_constr = vmap(self._friction_constr, randomness='same')
        self.grad_friction_constr = vmap(jacrev(self._friction_constr, argnums=(0, 1, 2)))

        self.friction_constr_force = vmap(partial(self._friction_constr, use_force=True), randomness='same')
        self.grad_friction_constr_force = vmap(
            jacrev(partial(self._friction_constr, use_force=True), argnums=(0, 1, 2)))

        self.kinematics_constr = vmap(vmap(self._kinematics_constr))
        self.grad_kinematics_constr = vmap(vmap(jacrev(self._kinematics_constr, argnums=(0, 1, 2, 3, 4, 5, 6))))

        self.contact_state_indices = [self.joint_index[finger] for finger in contact_fingers]
        self.contact_state_indices = list(itertools.chain.from_iterable(self.contact_state_indices))
        self.contact_control_indices = [16 + self.obj_dof + idx for idx in self.contact_state_indices]
        self.friction_polytope_k = 8

        if self.optimize_force:
            self._contact_dg_per_t = self.num_contacts * (1 + 2 + 4) + 3
            # self._contact_dg_per_t = self.num_contacts * (1 + 2)

        else:
            self._contact_dg_per_t = self.num_contacts * (1 + 2) + 3
            # self.dg_per_t = self.num_fingers * (1 + 3 + 2) + 3
            # self.dg_per_t = self.num_fingers * (1 + 3 + 2)
        self._contact_dg_constant = 0
        self._contact_dg = self._contact_dg_per_t * T + self._contact_dg_constant  # terminal contact points, terminal sdf=0, and dynamics
        self._contact_dz = 2 * (self.friction_polytope_k) * self.num_contacts  # one friction constraints per finger
        # self._contact_dz = (self.friction_polytope_k) * self.num_contacts 


        if self.min_force_dict is not None:
            self.min_force_dict = {k:v for k, v in self.min_force_dict.items() if k in self.contact_fingers}
            self._contact_dz += 1 * len(self.min_force_dict)
        self._contact_dh = self._contact_dz * T  # inequality

    def get_initial_xu(self, N):
        """
        use delta joint movement to get the initial trajectory
        the action (force at the finger tip) is not used. it is randomly intiailized
        the actual dynamics model is not used
        """

        u = 0.025 * torch.randn(N, self.T, self.du, device=self.device)
        if self.optimize_force and self.turn:
            for i, finger in enumerate(self.contact_fingers):
                idx = self.contact_force_indices_dict[finger]
                if finger != 'index':
                    u[..., idx] = 1.5 * torch.randn(N, self.T, 3, device=self.device)
                else:
                    u[..., idx] = 1.5 *.01 * torch.randn(N, self.T, 3, device=self.device)

        x = [self.start.reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            next_q = x[-1][:, :4 * self.num_fingers] + u[:, t, :4 * self.num_fingers]
            x.append(next_q)

        x = torch.stack(x[1:], dim=1)

        # if valve angle in state
        if self.dx == (4 * self.num_fingers + self.obj_dof):
            theta = np.linspace(self.start[-self.obj_dof:].cpu().numpy(), self.goal.cpu().numpy(), self.T + 1)[:-1]
            theta = torch.tensor(theta, device=self.device, dtype=torch.float32)
            theta = theta.unsqueeze(0).repeat((N, 1, 1))
            # theta = self.start[-self.obj_dof:].unsqueeze(0).repeat((N, self.T, 1))
            theta = torch.ones((N, self.T, self.obj_dof)).to(self.device) * self.start[-self.obj_dof:]
            x = torch.cat((x, theta), dim=-1)

        xu = torch.cat((x, u), dim=2)
        return xu

    def _cost(self, xu, start, goal):
        if self.num_contacts == 0:
            return 0.0
        # cost function for valve turning task
        # separate out q, theta, delta_q, force
        theta = xu[:, self.dx - self.obj_dof:self.dx]

        # goal cost
        cost = 10 * torch.sum((theta[-1] - goal) ** 2)
        cost += torch.sum((3 * (theta[:-1] - goal) ** 2))

        if self.optimize_force:
            force = xu[:, -self.num_contacts * 3:]
            sq = force ** 2
            cost += torch.sum(sq) * 1#.001
        
            # if not self.turn:
            #     cost += torch.sum(sq)
            # else:
            #     sq[:, -6:] *= .01
            #     cost += torch.sum(sq)
        return cost

    def get_friction_polytope(self):
        """
        :param k: the number of faces of the friction cone
        :return: a list of normal vectors of the faces of the friction cone
        """
        normal_vectors = []
        for i in range(self.friction_polytope_k):
            theta = 2 * np.pi * i / self.friction_polytope_k
            # might be -cos(theta), -sin(theta), mu
            normal_vector = torch.tensor([np.cos(theta), np.sin(theta), self.friction_coefficient]).to(
                device=self.device,
                dtype=torch.float32)
            normal_vectors.append(normal_vector)
        normal_vectors = torch.stack(normal_vectors, dim=0)
        return normal_vectors

    def _force_equlibrium_constr(self, q, u, next_q, contact_jac_list, contact_point_list):
        # NOTE: the constriant is defined in the robot frame
        # the contact jac an contact points are all in the robot frame
        # this will be vmapped, so takes in a 3 vector and a [num_finger x 3 x 8] jacobian and a dq vector
        obj_robot_frame = self.world_trans.inverse().transform_points(self.object_location.reshape(1, 3))
        delta_q = q + u - next_q
        torque_list = []
        residual_list = []
        for i, finger_name in enumerate(self.fingers):
            # TODO: Assume that all the fingers are having an equlibrium, maybe we should change so that index finger is not considered
            contact_jacobian = contact_jac_list[i]
            # pseudo inverse form
            # TODO: check why this is not stable
            # least_square_problem = torch.linalg.lstsq(contact_jacobian.transpose(-1, -2),
            #                            delta_q.unsqueeze(-1))
            # force = least_square_problem.solution.squeeze(-1)  # in the robot frame
            # residual = least_square_problem.residuals.squeeze(-1)
            # residual_list.append(residual)
            # approximated with contact velocity
            force = contact_jacobian @ delta_q
            # left pseudo inverse form
            # print(torch.linalg.det(contact_jacobian @ contact_jacobian.T))
            contact_point_r_valve = contact_point_list[i] - obj_robot_frame[0]
            torque = torch.linalg.cross(contact_point_r_valve, force, dim=-1)
            torque_list.append(torque)
            # Force is in the robot frame instead of the world frame. 
            # It does not matter for comuputing the force equilibrium constraint
        # force_world_frame = self.world_trans.transform_normals(force.unsqueeze(0)).squeeze(0)
        torque_list = torch.stack(torque_list, dim=0)
        torque_list = torch.sum(torque_list, dim=0)
        g = torque_list
        # residual_list = torch.stack(residual_list, dim=0) * 100
        # g = torch.cat((torque_list, residual_list), dim=-1)
        return g

    def _force_equlibrium_constraints(self, q, delta_q, compute_grads=True, compute_hess=False,
                                      projected_diffusion=False):
        N, T = q.shape[:2]
        T_offset = 1 if not projected_diffusion else 0
        device = q.device
        d = 32 + self.obj_dof
        # we want to add the start state to x, this x is now T + 1
        # think we need to assume that start consists of all 4 fingers
        q = torch.cat((self.start[:16].reshape(1, 1, -1).repeat(N, 1, 1), q), dim=1)
        next_q = q[:, 1:, self.contact_state_indices]
        q = q[:, :-1, self.contact_state_indices]
        u = delta_q[:, :, self.contact_state_indices]

        # retrieve contact jacobians and points
        contact_jac_list = []
        contact_point_list = []
        for finger in self.contact_fingers:
            jac = self.data[finger]['contact_jacobian'].reshape(N, T + T_offset, 3, -1)[:, T_offset:, :, self.contact_state_indices]
            contact_points = self.data[finger]['closest_pt_world'].reshape(N, T + T_offset, 3)[:, T_offset:].reshape(-1, 3)
            contact_jac_list.append(jac.reshape(N * (T + T_offset), 3, -1))
            contact_point_list.append(contact_points)

        contact_jac_list = torch.stack(contact_jac_list, dim=1).to(device=device)
        contact_point_list = torch.stack(contact_point_list, dim=1).to(device=device)

        g = self.force_equlibrium_constr(q.reshape(-1, 4 * self.num_contacts),
                                         u.reshape(-1, 4 * self.num_contacts),
                                         next_q.reshape(-1, 4 * self.num_contacts),
                                         contact_jac_list,
                                         contact_point_list).reshape(N, T, -1)
        if compute_grads:
            dg_dq, dg_du, dg_dnext_q, dg_djac, dg_dcontact = self.grad_force_equlibrium_constr(
                q.reshape(-1, 4 * self.num_contacts),
                u.reshape(-1, 4 * self.num_contacts),
                next_q.reshape(-1, 4 * self.num_contacts),
                contact_jac_list,
                contact_point_list)

            T_range = torch.arange(T, device=device)
            T_plus = torch.arange(1, T, device=device)
            T_minus = torch.arange(T - 1, device=device)

            grad_g = torch.zeros(N, g.shape[2], T, T, 32 + self.obj_dof, device=self.device)
            dg_dq = dg_dq.reshape(N, T, g.shape[2], 4 * self.num_contacts)
            dg_dnext_q = dg_dnext_q.reshape(N, T, g.shape[2], 4 * self.num_contacts)

            for i, finger_name in enumerate(self.fingers):
                # NOTE: assume fingers have joints independent of each other
                # TODO: check if we should use the jacobian of the current time steps or the jacobian of the next time steps.
                djac_dnext_q = self.data[finger_name]['dJ_dq'].reshape(N, T + T_offset, 3, 16, 16)[:, T_offset:, :,
                               self.contact_state_indices][:, :, :, :, self.contact_state_indices]

                dg_dnext_q = dg_dnext_q + dg_djac[:, :, i].reshape(N, T, g.shape[2], -1) @ \
                             djac_dnext_q.reshape(N, T, -1, 4 * self.num_contacts)

                d_contact_loc_dq = self.data[finger_name]['closest_pt_q_grad']
                d_contact_loc_dq = d_contact_loc_dq.reshape(N, T + T_offset, 3, 16)[:, :-1, :, self.contact_state_indices]
                dg_dq = dg_dq + dg_dcontact[:, :, i].reshape(N, T, g.shape[2], 3) @ d_contact_loc_dq
            mask_t = torch.zeros_like(grad_g).bool()
            mask_t[:, :, T_range, T_range] = True
            mask_t_p = torch.zeros_like(grad_g).bool()
            mask_t_p[:, :, T_plus, T_minus] = True
            mask_state = torch.zeros_like(grad_g).bool()
            mask_state[:, :, :, :, self.contact_state_indices] = True
            mask_control = torch.zeros_like(grad_g).bool()
            mask_control[:, :, :, :, self.contact_control_indices] = True
            # first q is the start
            grad_g[torch.logical_and(mask_t_p, mask_state)] = dg_dq.reshape(N, T,
                                                                            g.shape[2],
                                                                            -1)[:, 1:].transpose(1, 2).reshape(-1)

            grad_g[torch.logical_and(mask_t, mask_control)] = dg_du.reshape(N, T, -1,
                                                                            4 * self.num_contacts
                                                                            ).transpose(1, 2).reshape(-1)
            grad_g[torch.logical_and(mask_t, mask_state)] = dg_dnext_q.reshape(N, T, -1,
                                                                               4 * self.num_contacts
                                                                               ).transpose(1, 2).reshape(-1)
            grad_g = grad_g.transpose(1, 2)
        else:
            return g.reshape(N, -1), None, None

        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * d, T * d, device=self.device)
            return g.reshape(N, -1), grad_g.reshape(N, -1, T * d), hess
        else:
            return g.reshape(N, -1), grad_g.reshape(N, -1, T * d), None

    def _min_force_constr(self, force_list):
        force_mag_offset = []
        # force_mag = []
        for i, finger_name in enumerate(self.contact_fingers):
            min_force = self.min_force_dict[finger_name]
            force = force_list[i]
            force_norm = torch.linalg.norm(force, dim=-1)
            # force_mag.append(force / force_norm)
            force_mag_offset.append(min_force - force_norm)
            # print(force_norm.min(), min_force)
        return torch.stack(force_mag_offset, dim=0)
    
    def _min_force_constraints(self, q, force, compute_grads=True, compute_hess=False, projected_diffusion=False):
        N, T = q.shape[:2]
        T_offset = 1 if not projected_diffusion else 0
        device = q.device
        d = self.d

        force_list = force[:, :, self._contact_force_indices].reshape(force.shape[0], force.shape[1], self.num_contacts,
                                                                3)
        if self.env_force:
            num_forces = self.num_contacts + 1
        else:
            num_forces = self.num_contacts

        h = self.min_force_constr(force_list.reshape(-1, num_forces, 3),)
        h = h.reshape(N, T, -1)
        # dh_dforce = dh_dforce.reshape(N, T, -1, 3)
        if compute_grads:

            dh_dforce, = self.grad_min_force_constr(force_list.reshape(-1, num_forces, 3))
            dh_dforce = dh_dforce.reshape(dh_dforce.shape[0], dh_dforce.shape[1], num_forces* 3)

            T_range = torch.arange(T, device=device)
            T_plus = torch.arange(1, T, device=device)
            T_minus = torch.arange(T - 1, device=device)

            grad_g = torch.zeros(N, h.shape[2], T, T, self.d, device=self.device)

            mask_t = torch.zeros_like(grad_g).bool()
            mask_t[:, :, T_range, T_range] = True
            mask_t_p = torch.zeros_like(grad_g).bool()
            mask_t_p[:, :, T_plus, T_minus] = True
            mask_state = torch.zeros_like(grad_g).bool()
            mask_state[:, :, :, :, self.contact_state_indices] = True
            mask_force = torch.zeros_like(grad_g).bool()
            mask_force[:, :, :, :, self.contact_force_indices] = True
            mask_control = torch.zeros_like(grad_g).bool()
            mask_control[:, :, :, :, self.contact_control_indices] = True

            grad_g[torch.logical_and(mask_t, mask_force)] = dh_dforce.reshape(N, T, -1,
                                                                              num_forces * 3
                                                                              ).transpose(1, 2).reshape(-1)
            grad_h = grad_g.transpose(1, 2)
            grad_h = grad_h.reshape(N, -1, T * d)

        else:
            return h.reshape(N, -1), None, None
        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], T * 3, T * 3, device=self.device)
            return h.reshape(N, -1), grad_h, hess_h
        return h.reshape(N, -1), grad_h, None
                
    def _force_equlibrium_constr_w_force(self, q, u, next_q, force_list, contact_jac_list, contact_point_list):
        # NOTE: the constriant is defined in the robot frame
        # NOTE: the constriant is defined in the robot frame
        # the contact jac an contact points are all in the robot frame
        # this will be vmapped, so takes in a 3 vector and a [num_finger x 3 x 8] jacobian and a dq vector
        obj_robot_frame = self.world_trans.inverse().transform_points(self.object_location.reshape(1, 3))
        delta_q = q + u - next_q
        torque_list = []
        reactional_torque_list = []
        for i, finger_name in enumerate(self.contact_fingers):
            # TODO: Assume that all the fingers are having an equlibrium, maybe we should change so that index finger is not considered
            contact_jacobian = contact_jac_list[i]
            force_robot_frame = self.world_trans.inverse().transform_normals(force_list[i].unsqueeze(0)).squeeze(0)
            reactional_torque_list.append(contact_jacobian.T @ -force_robot_frame)
            # pseudo inverse form
            contact_point_r_valve = contact_point_list[i] - obj_robot_frame[0]
            torque = torch.linalg.cross(contact_point_r_valve, force_robot_frame)
            torque_list.append(torque)
            # Force is in the robot frame instead of the world frame.
            # It does not matter for comuputing the force equilibrium constraint
        # force_world_frame = self.world_trans.transform_normals(force.unsqueeze(0)).squeeze(0)
        torque_list = torch.stack(torque_list, dim=0)
        torque_list = torch.sum(torque_list, dim=0)
        reactional_torque_list = torch.stack(reactional_torque_list, dim=0)
        sum_reactional_torque = torch.sum(reactional_torque_list, dim=0)
        g_force_torque_balance = (sum_reactional_torque + 3.0 * delta_q)
        # print(g_force_torque_balance.max(), torque_list.max())
        g = torch.cat((torque_list, g_force_torque_balance.reshape(-1)), dim=-1)
        # residual_list = torch.stack(residual_list, dim=0) * 100
        # g = torch.cat((torque_list, residual_list), dim=-1)
        return g

    def _force_equlibrium_constraints_w_force(self, q, delta_q, force, compute_grads=True, compute_hess=False,
                                              projected_diffusion=False):
        N, T = q.shape[:2]
        T_offset = 1 if not projected_diffusion else 0
        device = q.device
        d = self.d

        full_start = partial_to_full_state(self.start[None, :self.num_fingers * 4], self.fingers)
        q = torch.cat((full_start.reshape(1, 1, -1).repeat(N, 1, 1), q), dim=1)
        next_q = q[:, 1:, self.contact_state_indices]
        q = q[:, :-1, self.contact_state_indices]
        u = delta_q[:, :, self.contact_state_indices]
        force_list = force[:, :, self._contact_force_indices].reshape(force.shape[0], force.shape[1], self.num_contacts,
                                                                      3)
        if self.env_force:
            env_force = force[:, :, -3:]
            force_list = torch.cat((force_list, env_force.unsqueeze(2)), dim=2)
# torch.Size([118, 8]) torch.Size([118, 8]) torch.Size([118, 8]) torch.Size([118, 2, 3, 4]) torch.Size([59, 2, 3])
# torch.Size([118, 8]) torch.Size([118, 8]) torch.Size([118, 8]) torch.Size([118, 2, 3, 4]) torch.Size([59, 2, 3])
        # retrieve contact jacobians and points
        contact_jac_list = []
        contact_point_list = []
        for finger in self.contact_fingers:
            jac = self.data[finger]['contact_jacobian'].reshape(N, T + T_offset, 3, -1)[:, T_offset:, :, self.contact_state_indices]
            contact_points = self.data[finger]['closest_pt_world'].reshape(N, T + T_offset , 3)[:, T_offset:].reshape(-1, 3)
            contact_jac_list.append(jac.reshape(N * T, 3, -1))
            contact_point_list.append(contact_points)

        contact_jac_list = torch.stack(contact_jac_list, dim=1).to(device=device)
        contact_point_list = torch.stack(contact_point_list, dim=1).to(device=device)

        if self.env_force:
            num_forces = self.num_contacts + 1
        else:
            num_forces = self.num_contacts
        g = self.force_equlibrium_constr(q.reshape(-1, 4 * self.num_contacts),
                                         u.reshape(-1, 4 * self.num_contacts),
                                         next_q.reshape(-1, 4 * self.num_contacts),
                                         force_list.reshape(-1, num_forces, 3),
                                         contact_jac_list,
                                         contact_point_list).reshape(N, T, -1)

        if compute_grads:
            dg_dq, dg_du, dg_dnext_q, dg_dforce, dg_djac, dg_dcontact = self.grad_force_equlibrium_constr(
                q.reshape(-1, 4 * self.num_contacts),
                u.reshape(-1, 4 * self.num_contacts),
                next_q.reshape(-1, 4 * self.num_contacts),
                force_list.reshape(-1, num_forces, 3),
                contact_jac_list,
                contact_point_list)
            dg_dforce = dg_dforce.reshape(dg_dforce.shape[0], dg_dforce.shape[1], num_forces* 3)

            T_range = torch.arange(T, device=device)
            T_plus = torch.arange(1, T, device=device)
            T_minus = torch.arange(T - 1, device=device)

            grad_g = torch.zeros(N, g.shape[2], T, T, self.d, device=self.device)
            dg_dq = dg_dq.reshape(N, T, g.shape[2], 4 * self.num_contacts)
            dg_dnext_q = dg_dnext_q.reshape(N, T, g.shape[2], 4 * self.num_contacts)

            for i, finger_name in enumerate(self.contact_fingers):
                # NOTE: assume fingers have joints independent of each other
                # TODO: check if we should use the jacobian of the current time steps or the jacobian of the next time steps.
                djac_dnext_q = self.data[finger_name]['dJ_dq'].reshape(N, T + T_offset, 3, 16, 16)[:, T_offset:, :,
                               self.contact_state_indices][:, :, :, :, self.contact_state_indices]

                dg_dnext_q = dg_dnext_q + dg_djac[:, :, i].reshape(N, T, g.shape[2], -1) @ \
                             djac_dnext_q.reshape(N, T, -1, 4 * self.num_contacts)

                d_contact_loc_dq = self.data[finger_name]['closest_pt_q_grad']
                d_contact_loc_dq = d_contact_loc_dq.reshape(N, T + T_offset, 3, 16)[:, :-1, :, self.contact_state_indices]
                dg_dq = dg_dq + dg_dcontact[:, :, i].reshape(N, T, g.shape[2], 3) @ d_contact_loc_dq

            mask_t = torch.zeros_like(grad_g).bool()
            mask_t[:, :, T_range, T_range] = True
            mask_t_p = torch.zeros_like(grad_g).bool()
            mask_t_p[:, :, T_plus, T_minus] = True
            mask_state = torch.zeros_like(grad_g).bool()
            mask_state[:, :, :, :, self.contact_state_indices] = True
            mask_force = torch.zeros_like(grad_g).bool()
            mask_force[:, :, :, :, self.contact_force_indices] = True
            mask_control = torch.zeros_like(grad_g).bool()
            mask_control[:, :, :, :, self.contact_control_indices] = True

            # first q is the start
            grad_g[torch.logical_and(mask_t_p, mask_state)] = dg_dq.reshape(N, T,
                                                                            g.shape[2],
                                                                            -1)[:, 1:].transpose(1, 2).reshape(-1)

            grad_g[torch.logical_and(mask_t, mask_control)] = dg_du.reshape(N, T, -1,
                                                                            4 * self.num_contacts
                                                                            ).transpose(1, 2).reshape(-1)
            grad_g[torch.logical_and(mask_t, mask_state)] = dg_dnext_q.reshape(N, T, -1,
                                                                               4 * self.num_contacts
                                                                               ).transpose(1, 2).reshape(-1)

            grad_g[torch.logical_and(mask_t, mask_force)] = dg_dforce.reshape(N, T, -1,
                                                                              num_forces * 3
                                                                              ).transpose(1, 2).reshape(-1)
            grad_g = grad_g.transpose(1, 2)

        else:
            return g.reshape(N, -1), None, None
        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * d, T * d, device=self.device)
            return g.reshape(N, -1), grad_g.reshape(N, -1, T * d), hess
        else:
            return g.reshape(N, -1), grad_g.reshape(N, -1, T * d), None

    def _kinematics_constr(self, current_q,
                           next_q,
                           current_theta,
                           next_theta,
                           contact_jac,
                           contact_loc,
                           contact_normal):
        # N, _, _ = current_q.shape
        # T = self.T
        # approximate q dot and theta dot
        dq = next_q - current_q
        if self.obj_dof == 3:
            if self.obj_dof_type == 'x_y_theta':
                d_omega = next_theta[-1:] - current_theta[-1:]
                obj_omega = torch.cat((torch.zeros_like(d_omega),
                                       torch.zeros_like(d_omega),
                                       d_omega), -1)
                obj_v = next_theta[:2] - current_theta[:2]
                tmp = torch.zeros_like(obj_v[-1:])
                obj_v = torch.cat((obj_v, tmp), -1)
            else:
                obj_omega = euler_to_angular_velocity(current_theta, next_theta)
                obj_v = torch.zeros_like(obj_omega)
        elif self.obj_dof == 1:
            dtheta = next_theta - current_theta
            obj_omega = torch.concat((torch.zeros_like(dtheta),
                                      dtheta,
                                      torch.zeros_like(dtheta)), -1)  # should be N x T-1 x 3
            obj_v = torch.zeros_like(obj_omega)
        elif self.obj_dof == 6:
            obj_omega = euler_to_angular_velocity(current_theta[3:], next_theta[3:])
            obj_v = next_theta[:3] - current_theta[:3]

        contact_point_v = (contact_jac @ dq.reshape(4 * self.num_contacts, 1)).squeeze(-1)

        # compute valve contact point velocity
        if self.obj_dof_type == "x_y_theta":
            obj_center = current_theta[:2]
            obj_center = torch.cat((obj_center, torch.zeros_like(obj_center[:1])), -1)
            obj_center += torch.tensor(self.object_asset_pos, device=self.device)
            valve_robot_frame = self.world_trans.inverse().transform_points(obj_center.reshape(1, 3))
        else:
            valve_robot_frame = self.world_trans.inverse().transform_points(self.object_location.reshape(1, 3))
        contact_point_r_valve = contact_loc.reshape(3) - valve_robot_frame.reshape(3)
        obj_omega_robot_frame = self.world_trans.inverse().transform_normals(obj_omega.reshape(1, 3)).reshape(-1)
        obj_v_robot_frame = self.world_trans.inverse().transform_normals(obj_v.reshape(1, 3)).reshape(-1)
        object_contact_point_v = torch.cross(obj_omega_robot_frame, contact_point_r_valve, dim=-1) + obj_v_robot_frame

        # project the constraint into the tangential plane
        normal_projection = contact_normal.unsqueeze(-1) @ contact_normal.unsqueeze(-2)
        R = self.get_rotation_from_normal(contact_normal.reshape(-1, 3)).reshape(3, 3).detach().transpose(1, 0)
        R = R[:2]

        # compute contact v tangential to surface
        # TODO: might not need this part 
        contact_point_v_tan = contact_point_v - (normal_projection @ contact_point_v.unsqueeze(-1)).squeeze(-1)
        object_contact_point_v_tan = object_contact_point_v - (
                normal_projection @ object_contact_point_v.unsqueeze(-1)).squeeze(-1)

        # we actually ended up computing T+1 contact constraints, but start state is fixed so we throw that away
        # g = (contact_point_v - object_contact_point_v).reshape(N, -1) # DEBUG ONLY
        g = (R @ (contact_point_v_tan - object_contact_point_v_tan).unsqueeze(-1)).reshape(-1)

        return g

    @contact_finger_constraints
    def _kinematics_constraints(self, q, delta_q, theta, finger_name, compute_grads=True, compute_hess=False,
                                projected_diffusion=False):
        """
            Computes on the kinematics of the valve and the finger being consistant
        """
        N, T, _ = q.shape
        T_offset = 1 if not projected_diffusion else 0
        d = self.d
        device = q.device
        full_start = partial_to_full_state(self.start[None, :self.num_fingers * 4], self.fingers)
        q = torch.cat((full_start.reshape(1, 1, -1).repeat(N, 1, 1), q), dim=1)
        theta = torch.cat((self.start[-self.obj_dof:].reshape(1, 1, -1).repeat(N, 1, 1), theta), dim=1)

        # Retrieve pre-processed data
        ret_scene = self.data[finger_name]
        contact_jacobian = ret_scene.get('contact_jacobian', None).reshape(
            N, T + T_offset, 3, 16)[:, :T, :, self.contact_state_indices]
        contact_loc = ret_scene.get('closest_pt_world', None).reshape(N, T + T_offset, 3)[:, :T]
        d_contact_loc_dq = ret_scene.get('closest_pt_q_grad', None)[:, :T, :, self.contact_state_indices]
        dJ_dq = ret_scene.get('dJ_dq', None)[:, :T, :, self.contact_state_indices]
        dJ_dq = dJ_dq[:, :, :, :, self.contact_state_indices]
        contact_normal = ret_scene.get('contact_normal', None).reshape(N, T + T_offset, 3)[:, :T]
        dnormal_dq = self.data[finger_name]['dnormal_dq'].reshape(N, T + T_offset, 3, 16)[:, :T, :,
                     self.contact_state_indices]
        dnormal_dtheta = self.data[finger_name]['dnormal_denv_q'].reshape(N, T + T_offset, 3, self.obj_dof)[:, :T]

        current_q = q[:, :-1, self.contact_state_indices]
        next_q = q[:, 1:, self.contact_state_indices]
        current_theta = theta[:, :-1]
        next_theta = theta[:, 1:]
        
        g = self.kinematics_constr(current_q,
                                   next_q,
                                   current_theta,
                                   next_theta,
                                   contact_jacobian, contact_loc,
                                   contact_normal).reshape(N, -1)
        g_dim = g.reshape(N, T, -1).shape[-1]

        if compute_grads:
            T_range = torch.arange(T, device=device)
            T_range_minus = torch.arange(T - 1, device=device)
            T_range_plus = torch.arange(1, T, device=device)

            dg_d_current_q, dg_d_next_q, dg_d_current_theta, dg_d_next_theta, dg_d_contact_jac, dg_d_contact_loc, dg_d_normal \
                = self.grad_kinematics_constr(current_q, next_q, current_theta, next_theta, contact_jacobian,
                                              contact_loc, contact_normal)
            with torch.no_grad():
                dg_d_current_q = dg_d_current_q + dg_d_contact_jac.reshape(N, T, g_dim, -1) @ \
                                 dJ_dq.reshape(N, T, -1, 4 * self.num_contacts)
                dg_d_current_q = dg_d_current_q + dg_d_contact_loc @ d_contact_loc_dq

                # add constraints related to normal 
                dg_d_current_q = dg_d_current_q + dg_d_normal @ dnormal_dq
                dg_d_current_theta = dg_d_current_theta + dg_d_normal @ dnormal_dtheta
                grad_g = torch.zeros((N, T, T, g_dim, d), device=device)

                mask_t = torch.zeros_like(grad_g).bool()
                mask_t[:, T_range, T_range] = True
                mask_t_p = torch.zeros_like(grad_g).bool()
                mask_t_p[:, T_range_plus, T_range_minus] = True
                mask_state = torch.zeros_like(grad_g).bool()
                mask_state[:, :, :, :, self.contact_state_indices] = True

                grad_g[torch.logical_and(mask_t_p, mask_state)] = dg_d_current_q[:, 1:].reshape(-1)
                grad_g[:, T_range_plus, T_range_minus, :, 16: 16 + self.obj_dof] = dg_d_current_theta[:, 1:]
                grad_g[torch.logical_and(mask_t, mask_state)] = dg_d_next_q.reshape(-1)
                grad_g[:, T_range, T_range, :, 16:16 + self.obj_dof] = dg_d_next_theta
                grad_g = grad_g.permute(0, 1, 3, 2, 4).reshape(N, -1, T * d)

                if torch.any(torch.isnan(grad_g)):
                    print('hello')

        else:
            return g, None, None

        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * d, T * d, device=device)
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
        # R = self.get_rotation_from_normal(contact_normal_world.unsqueeze(0)).squeeze(0).detach().permute(0, 1)
        R = self.get_rotation_from_normal(contact_normal_world.unsqueeze(0)).squeeze(0).detach().permute(1, 0)
        R = R[:2]
        # compute contact v tangential to surface
        contact_v_tan = contact_v_world - (normal_projection @ contact_v_world.unsqueeze(-1)).squeeze(-1)
        contact_v_u_tan = contact_v_u_world - (normal_projection @ contact_v_u_world.unsqueeze(-1)).squeeze(-1)

        # should have same tangential components
        # TODO: this constraint value is super small
        return (R @ (contact_v_tan - contact_v_u_tan).unsqueeze(-1)).squeeze(-1)

    @contact_finger_constraints
    def _dynamics_constraints(self, q, delta_q, finger_name, compute_grads=True, compute_hess=False):
        """ Computes dynamics constraints
            constraint that sdf value is zero
            also constraint on contact kinematics to get the valve dynamics
        """
        N, T, _ = q.shape
        device = q.device

        d = self.d

        # we want to add the start state to x, this x is now T + 1
        q = torch.cat((self.start[:16].reshape(1, 1, -1).repeat(N, 1, 1), q), dim=1)

        q = q[:, :-1, self.contact_state_indices]
        next_q = q[:, 1:, self.contact_state_indices]
        u = delta_q[:, :, self.contact_state_indices]

        contact_jac = self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, 16)[
                      :, :-1, :, self.contact_state_indices]
        contact_normal = self.data[finger_name]['contact_normal'].reshape(N, T + 1, 3)[:, :-1]
        dnormal_dq = self.data[finger_name]['dnormal_dq'].reshape(N, T + 1, 3, 16)[
                     :, :-1, :, self.contact_state_indices]
        dnormal_dtheta = self.data[finger_name]['dnormal_denv_q'].reshape(N, T + 1, 3, self.obj_dof)[:, :-1]

        g = self.dynamics_constr(q.reshape(-1, 4 * self.num_contacts), u.reshape(-1, 4 * self.num_contacts),
                                 next_q.reshape(-1, 4 * self.num_contacts),
                                 contact_jac.reshape(-1, 3, 4 * self.num_contacts),
                                 contact_normal.reshape(-1, 3)).reshape(N, T, -1)

        if compute_grads:
            T_range = torch.arange(T, device=device)
            T_plus = torch.arange(1, T, device=device)
            T_minus = torch.arange(T - 1, device=device)
            grad_g = torch.zeros(N, g.shape[2], T, T, self.dx + self.du, device=self.device)
            # dnormal_dq = torch.zeros(N, T, 3, 8, device=self.device)  # assume zero SDF hessian
            djac_dq = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, 4 * self.num_contacts,
                                                              4 * self.num_contacts)[
                      :, :-1]
            dg_dq, dg_du, dg_dnext_q, dg_djac, dg_dnormal = self.grad_dynamics_constr(
                q.reshape(-1, 4 * self.num_contacts),
                u.reshape(-1, 4 * self.num_contacts),
                next_q.reshape(-1, 4 * self.num_contacts),
                contact_jac.reshape(-1, 3, 4 * self.num_contacts),
                contact_normal.reshape(-1, 3))

            mask_t = torch.zeros_like(grad_g).bool()
            mask_t[:, :, T_range, T_range] = True
            mask_t_p = torch.zeros_like(grad_g).bool()
            mask_t_p[:, :, T_plus, T_minus] = True
            mask_state = torch.zeros_like(grad_g).bool()
            mask_state[:, :, :, :, self.contact_state_indices] = True
            mask_force = torch.zeros_like(grad_g).bool()
            mask_force[:, :, :, :, self.contact_force_indices] = True
            mask_control = torch.zeros_like(grad_g).bool()
            mask_control[:, :, :, :, self.contact_control_indices] = True

            dg_dq = dg_dq.reshape(N, T, g.shape[2], -1) + dg_dnormal.reshape(N, T, g.shape[2], -1) @ dnormal_dq  #
            dg_dq = dg_dq + dg_djac.reshape(N, T, g.shape[2], -1) @ djac_dq.reshape(N, T, -1, 4 * self.num_contacts)
            dg_dtheta = dg_dnormal.reshape(N, T, g.shape[2], -1) @ dnormal_dtheta
            # first q is the start
            grad_g[torch.logical_and(mask_t_p, mask_state)] = dg_dq[:, 1:].transpose(1, 2)
            grad_g[:, :, T_range, T_range, self.contact_control_indices] = \
                dg_du.reshape(N, T, -1, 4 * self.num_contacts).transpose(1, 2)
            grad_g[:, :, T_plus, T_minus, 16:16 + self.obj_dof] = dg_dtheta[:, 1:].transpose(1, 2)
            grad_g[:, :, T_range, T_range, self.contact_state_indices] = \
                dg_dnext_q.reshape(N, T, -1, 4 * self.num_contacts).transpose(1, 2)
            grad_g = grad_g.transpose(1, 2)
        else:
            return g.reshape(N, -1), None, None

        if compute_hess:
            hess_g = torch.zeros(N, T * 2,
                                 T * d,
                                 T * d, device=self.device)

            return g.reshape(N, -1), grad_g.reshape(N, -1, T * d), hess_g

        return g.reshape(N, -1), grad_g.reshape(N, -1, T * d), None

    def _friction_constr(self, dq, contact_normal, contact_jacobian, use_force=False):
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
        if use_force:  # self.optimize_force:
            # here dq means the force in the world frame
            contact_v_contact_frame = R.transpose(0, 1) @ dq
        else:
            contact_v_contact_frame = R.transpose(0, 1) @ self.world_trans.transform_normals(
                (contact_jacobian @ dq).unsqueeze(0)).squeeze(0)
            # if min_force is not None:
            #     min_normal_force_con = min_force - contact_v_contact_frame[2]
            #     print(min_force, min_normal_force_con)
        # TODO: there are two different ways of doing a friction cone
        # Linearized friction cone - but based on the contact point velocity
        # force is defined as the force of robot pushing the object
        constr = B @ contact_v_contact_frame
        # if min_force is not None:
        #     constr = torch.cat((constr, min_normal_force_con.unsqueeze(0)))

        return constr

    @contact_finger_constraints
    def _friction_constraint(self, q, delta_q, finger_name, force=None, compute_grads=True, compute_hess=False, projected_diffusion=False):
        # assume access to class member variables which have already done some of the computation
        N, T, _ = q.shape
        T_offset = 1 if not projected_diffusion else 0
        d = self.d
        u = delta_q[:, :, self.contact_state_indices]

        if force is not None:
            force = force[:, :, self._contact_force_indices].reshape(-1, self.num_contacts, 3)
            for i, finger_candidate in enumerate(self.contact_fingers):
                if finger_candidate == finger_name:
                    # force_index = [i * 3 + j for j in range(3)]
                    u = force[:, i]
                    break
        else:
            u = delta_q[:, :, self.contact_state_indices].reshape(-1, 4 * self.num_contacts)

        # u is the delta q commanded
        # retrieved cached values
        contact_jac = self.data[finger_name]['contact_jacobian'].reshape(N, T + T_offset, 3, 16)[
                      :, :T, :, self.contact_state_indices]
        contact_normal = self.data[finger_name]['contact_normal'].reshape(N, T + T_offset, 3)[:,
                         :T]  # contact normal is pointing out
        dnormal_dq = self.data[finger_name]['dnormal_dq'].reshape(N, T + T_offset, 3, 16)[
                     :, :T, :, self.contact_state_indices]
        dnormal_dtheta = self.data[finger_name]['dnormal_denv_q'].reshape(N, T + T_offset, 3, self.obj_dof)[:, :T]

        if self.min_force_dict is not None and finger_name in self.min_force_dict:
            min_force = self.min_force_dict[finger_name]
        else:
            min_force = None
        
        if force is None:
            # compute constraint value
            h = self.friction_constr(u,
                                     contact_normal.reshape(-1, 3),
                                     contact_jac.reshape(-1, 3, 4 * self.num_contacts),
                                     ).reshape(N, -1)
        else:
            # compute constraint value
            h = self.friction_constr_force(u,
                                           contact_normal.reshape(-1, 3),
                                           contact_jac.reshape(-1, 3, 4 * self.num_contacts),
                                           ).reshape(N, -1)

        # compute the gradient
        if compute_grads:
            if force is None:
                dh_du, dh_dnormal, dh_djac = self.grad_friction_constr(u,
                                                                       contact_normal.reshape(-1, 3),
                                                                       contact_jac.reshape(-1, 3,
                                                                                           4 * self.num_contacts))
            else:
                dh_du, dh_dnormal, dh_djac = self.grad_friction_constr_force(u,
                                                                             contact_normal.reshape(-1, 3),
                                                                             contact_jac.reshape(-1, 3,
                                                                                                 4 * self.num_contacts))

            djac_dq = self.data[finger_name]['dJ_dq'].reshape(N, T + T_offset, 3, 16, 16)[
                      :, :T, :, self.contact_state_indices][:, :, :, :, self.contact_state_indices]

            dh = dh_dnormal.shape[1]
            dh_dq = dh_dnormal.reshape(N, T, dh, -1) @ dnormal_dq
            dh_dq = dh_dq + dh_djac.reshape(N, T, dh, -1) @ djac_dq.reshape(N, T, -1, 4 * self.num_contacts)
            dh_dtheta = dh_dnormal.reshape(N, T, dh, -1) @ dnormal_dtheta
            grad_h = torch.zeros(N, dh, T, T, d, device=self.device)

            T_range = torch.arange(T, device=self.device)
            T_range_minus = torch.arange(T - 1, device=self.device)
            T_range_plus = torch.arange(1, T, device=self.device)

            # create masks
            mask_t = torch.zeros_like(grad_h).bool()
            mask_t[:, :, T_range, T_range] = True
            mask_t_p = torch.zeros_like(grad_h).bool()
            mask_t_p[:, :, T_range_plus, T_range_minus] = True
            mask_state = torch.zeros_like(grad_h).bool()
            mask_state[:, :, :, :, self.contact_state_indices] = True

            mask_control = torch.zeros_like(grad_h).bool()
            mask_control[:, :, :, :, self.contact_control_indices] = True

            grad_h[torch.logical_and(mask_t_p, mask_state)] = dh_dq[:, 1:].transpose(1, 2).reshape(-1)
            grad_h[:, :, T_range_plus, T_range_minus, 16: 16 + self.obj_dof] = dh_dtheta[:, 1:].transpose(1, 2)

            if force is not None:
                mask_force = torch.zeros_like(grad_h).bool()
                contact_force_indices = self.contact_force_indices[i * 3: (i + 1) * 3]
                mask_force[:, :, :, :, contact_force_indices] = True
                grad_h[torch.logical_and(mask_t, mask_force)] = dh_du.reshape(N, T, dh, 3).transpose(1, 2).reshape(-1)
            else:
                grad_h[torch.logical_and(mask_t, mask_control)] = dh_du.reshape(
                    N, T, dh, 4 * self.num_contacts).transpose(1, 2).reshape(-1)
            grad_h = grad_h.transpose(1, 2).reshape(N, -1, T * d)
        else:
            return h, None, None

        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], T * d, T * d, device=q.device)
            return h, grad_h, hess_h

        return h, grad_h, None

    def _con_ineq(self, xu, compute_grads=True, compute_hess=False, verbose=False, projected_diffusion=False):
        N = xu.shape[0]
        T = xu.shape[1]
        q = xu[:, :, :self.num_fingers * 4]
        delta_q = xu[:, :, self.num_fingers * 4 + self.obj_dof:self.num_fingers * 8 + self.obj_dof]
        q = partial_to_full_state(q, fingers=self.fingers)
        delta_q = partial_to_full_state(delta_q, fingers=self.fingers)
        force = None
        h, grad_h, hess_h = self._friction_constraint(
            q=q, delta_q=delta_q, force=None,
            compute_grads=compute_grads,
            compute_hess=compute_hess,
            projected_diffusion=projected_diffusion)

        if self.optimize_force:
            force = torch.zeros(N, T, 12, device=self.device)
            force[:, :, self._contact_force_indices] = xu[:, :, -self.num_contacts * 3:]
            h2, grad_h2, hess_h2 = self._friction_constraint(
                q=q, delta_q=delta_q, force=None,
                compute_grads=compute_grads,
                compute_hess=compute_hess,
                projected_diffusion=projected_diffusion)
            h = torch.cat((h, h2), dim=1)
            if grad_h is not None:
                grad_h = torch.cat((grad_h, grad_h2), dim=1)
            if self.min_force_dict is not None:
                h_min, grad_h_min, hess_h_min = self._min_force_constraints(
                    q, force, compute_grads=compute_grads, compute_hess=compute_hess,
                    projected_diffusion=projected_diffusion)
                h = torch.cat((h, h_min), dim=1)
                if grad_h is not None and grad_h_min is not None:
                    grad_h = torch.cat((grad_h, grad_h_min), dim=1)
            # if grad_h is not None:
            #     grad_h = torch.cat((grad_h, grad_h2, grad_h_min), dim=1)
            if hess_h is not None:
                hess_h = torch.cat((hess_h, hess_h2, hess_h_min), dim=1)
        # print("friction", h.max())

        if verbose:
            print(f"max friction constraint: {torch.max(h)}")
            # print(f"max step size constraint: {torch.max(h_step_size)}")
            # print(f"max singularity constraint: {torch.max(h_sin)}")
            result_dict = {}
            result_dict['friction'] = torch.max(h).item()
            # result_dict['singularity'] = torch.max(h_sin).item()
            return result_dict

        if compute_grads:
            grad_h = grad_h.reshape(N, -1, (self.T + 1 if projected_diffusion else self.T) * self.d)
        else:
            return h, None, None
        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], self.T * (self.dx + self.du), self.T * (self.dx + self.du),
                                 device=self.device)
            return h, grad_h, hess_h
        return h, grad_h, None

    @contact_finger_constraints
    def _running_contact_constraints(self, q, finger_name, compute_grads=True, compute_hess=False, 
                                                                        projected_diffusion=False):
        return self._contact_constraints(q=q, finger_name=finger_name, compute_grads=compute_grads,
                                         compute_hess=compute_hess, 
                                         projected_diffusion=projected_diffusion)

    def _con_eq(self, xu, compute_grads=True, compute_hess=False, verbose=False, projected_diffusion=False):
        N = xu.shape[0]
        T = xu.shape[1]
        q = xu[:, :, :self.num_fingers * 4]
        delta_q = xu[:, :, self.num_fingers * 4 + self.obj_dof:self.num_fingers * 8 + self.obj_dof]
        q = partial_to_full_state(q, fingers=self.fingers)
        delta_q = partial_to_full_state(delta_q, fingers=self.fingers)
        theta = xu[:, :, self.num_fingers * 4:self.num_fingers * 4 + self.obj_dof]
        g_contact, grad_g_contact, hess_g_contact = self._running_contact_constraints(q=q,
                                                                                      compute_grads=compute_grads,
                                                                                      compute_hess=compute_hess, 
                                                                                      projected_diffusion=projected_diffusion)
        force = None
        if self.optimize_force:
            if self.env_force:
                force = torch.zeros(N, T, 12 + 3, device=self.device) # assume one env contact
                force[:, :, self._contact_force_indices] = xu[:, :, -(self.num_contacts + 1) * 3: -3]
                force[:, :, -3:] = xu[:, :, -3:]
            else:
                force = torch.zeros(N, T, 12, device=self.device)
                force[:, :, self._contact_force_indices] = xu[:, :, -self.num_contacts * 3:]

            g_equil, grad_g_equil, hess_g_equil = self._force_equlibrium_constraints_w_force(
                q=q,
                delta_q=delta_q,
                force=force,
                compute_grads=compute_grads,
                compute_hess=compute_hess,
                projected_diffusion=projected_diffusion)
        else:
            g_equil, grad_g_equil, hess_g_equil = self._force_equlibrium_constraints(
                q=q, delta_q=delta_q,
                compute_grads=compute_grads,
                compute_hess=compute_hess,
                projected_diffusion=projected_diffusion)

        g_valve, grad_g_valve, hess_g_valve = self._kinematics_constraints(
            q=q, delta_q=delta_q, theta=theta,
            compute_grads=compute_grads,
            compute_hess=compute_hess,
            projected_diffusion=projected_diffusion)
        # print("contact", g_contact.max())
        # print("equil", g_equil.max())
        # print("kinematics", g_valve.max())

        g_contact = torch.cat((g_contact,
                               #    g_dynamics,
                               g_equil,
                               g_valve,
                               ), dim=1)

        if grad_g_contact is not None:
            grad_g_contact = torch.cat((grad_g_contact,
                                        # grad_g_dynamics,
                                        grad_g_equil,
                                        grad_g_valve,
                                        ), dim=1)
            if torch.any(torch.isinf(grad_g_contact)) or torch.any(torch.isnan(grad_g_contact)):
                print('hello')
        if hess_g_contact is not None:
            hess_g_contact = torch.cat((hess_g_contact,
                                        # hess_g_dynamics,
                                        hess_g_equil,
                                        hess_g_valve,
                                        ), dim=1)

        return g_contact, grad_g_contact, hess_g_contact

class AllegroContactWithEnvProblem(AllegroContactProblem):
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
                 obj_joint_dim=0,
                 optimize_force=False,
                 obj_dof_type=None,
                 device='cuda:0', **kwargs):
        self.obj_dof_type = obj_dof_type
        super(AllegroContactWithEnvProblem, self).__init__(start=start, goal=goal, T=T, chain=chain,
                                                           object_location=object_location, object_type=object_type,
                                                           world_trans=world_trans, object_asset_pos=object_asset_pos,
                                                           regrasp_fingers=regrasp_fingers, contact_fingers=contact_fingers,
                                                           friction_coefficient=friction_coefficient, obj_dof=obj_dof,
                                                           obj_ori_rep=obj_ori_rep, obj_joint_dim=obj_joint_dim,
                                                           optimize_force=optimize_force, device=device, env_force=True, 
                                                           obj_dof_type=obj_dof_type, **kwargs)
        self._contact_dg_per_t = self.num_contacts * (1 + 2 + 4) + 3 + 1 # 1 for env friction constraint
        self._contact_dz = (self.friction_polytope_k) * self.num_contacts 
        self._contact_dh = self._contact_dz * T  # inequality
        self.force_equlibrium_constr = vmap(self._force_equlibrium_constr_w_force)
        self.grad_force_equlibrium_constr = vmap(
            jacrev(self._force_equlibrium_constr_w_force, argnums=(0, 1, 2, 3, 4, 5)))
        self.env_friction_constr = vmap(self._env_friction_constr, randomness='same')
        self.grad_env_friction_constr = vmap(jacrev(self._env_friction_constr, argnums=(0)))
        # table_dims = np.array([0.4, 0.4, 0.05], dtype=np.float32)
        # table_sdf = pv.BoxSDF([*table_dims], device=self.device)
        # world2obj = tf.Transform3d(pos=torch.tensor(self.object_asset_pos, device=self.device).float(),
        #                                   rot=torch.tensor(
        #                                       [1, 0, 0, 0],
        #                                       device=self.device).float(), device=self.device)
        # obj2table = world2obj.inverse().compose(
        #     pk.Transform3d(device=self.device).translate(*table_asset_pos))
        
        # self.obj_table_scenes = pv.RobotScene(self.object_sdf, table_sdf, obj2table,
        #                                     collision_check_links=['card'],
        #                                     softmin_temp=1.0e3,
        #                                     points_per_link=2000,
        #                                     partial_patch=False,
        #                                     )
    def _preprocess_fingers(self, q, theta):
        N, T, _ = q.shape
        if self.obj_dof_type == "x_y_theta":
            tmp = torch.zeros(N, T, 1, device=self.device)
            theta = torch.cat((theta[:, :, :2], tmp, tmp, tmp, theta[:, :, -1:]), dim=2)
        super()._preprocess_fingers(q, theta)
        if self.obj_dof_type == "x_y_theta":
            for finger in self.fingers:
                dnormal_denv_q = self.data[finger]['dnormal_denv_q']
                self.data[finger]['dnormal_denv_q'] = torch.cat((dnormal_denv_q[:, :, :2], dnormal_denv_q[:, :, -1:]), dim=2)
                grad_env_sdf = self.data[finger]['grad_env_sdf']
                self.data[finger]['grad_env_sdf'] = torch.cat((grad_env_sdf[:, :2], grad_env_sdf[:, -1:]), dim=-1)
        # theta_b = theta.reshape(-1, 6)
        # ret_scene_obj_env = self.obj_table_scenes.scene_collision_check(theta_b, None,
        #                                                         compute_gradient=True,
        #                                                         compute_hessian=False)
        # self.data['obj'] = {}
        # self.data['obj']['sdf_obj_table'] = ret_scene_obj_env['sdf'].reshape(N, self.T + 1)
        # self.data['obj']['grad_sdf_obj_table'] = ret_scene_obj_env['grad_sdf'].reshape(N, self.T + 1, 6)
        # d_contact_loc_dq = ret_scene_obj_env.get('closest_pt_q_grad', None)
        # d_contact_loc_dq = d_contact_loc_dq.reshape(N, self.T + 1, 3, 6)
        # self.data['obj']['closest_pt_q_grad'] = d_contact_loc_dq
        # # self.data['peg']['contact_hessian'] = contact_hessian
        # self.data['obj']['closest_pt_world'] = ret_scene_obj_env['closest_pt_world']
        # self.data['obj']['contact_normal'] = ret_scene_obj_env['contact_normal']

        # # gradient of contact normal
        # self.data['obj']['dnormal_dq'] = ret_scene_obj_env['dnormal_dq'].reshape(N, self.T + 1, 3, 6)
    
    # def _obj_wall_contact_constraint(self, theta, compute_grads=True, compute_hess=False):
    #     N, T, _ = theta.shape
    #     # Retrieve pre-processed data
    #     g = self.data['obj'].get('sdf_obj_table').reshape(N, T + 1, 1)  # - 0.0025
    #     grad_g_theta = self.data['obj'].get('grad_sdf_obj_table', None)
    #     if self.obj_dof_type == "x_y_theta":
    #         grad_g_theta = torch.cat((grad_g_theta[:, :, :2], grad_g_theta[:, :, -1:]), dim=2)

    #     # Ignore first value, as it is the start state
    #     g = g[:, 1:].reshape(N, -1)

    #     if compute_grads:
    #         T_range = torch.arange(T, device=theta.device)
    #         # compute gradient of sdf
    #         grad_g = torch.zeros(N, T, T, self.d, device=theta.device)
    #         grad_g[:, T_range, T_range, 16:16 + self.obj_dof] = grad_g_theta.reshape(N, T + 1, self.obj_dof)[:, 1:]
    #         grad_g = grad_g.reshape(N, -1, T, self.d)
    #         grad_g = grad_g.reshape(N, -1, T * (self.d))
    #     else:
    #         return g, None, None

    #     if compute_hess:
    #         hess = torch.zeros(N, g.shape[1], T * (self.d), T * (self.d), device=self.device)
    #         return g, grad_g, hess

    #     return g, grad_g, None
    def _env_friction_constr(self, env_force):
        env_z_force = env_force[-1]
        env_tangent_force = torch.linalg.norm(env_force[:-1])
        return env_tangent_force - (self.friction_coefficient / 2) * env_z_force # use a half since isaac gym use avg friction
    def _env_friction_constraint(self, q, force, compute_grads=True, compute_hess=False, projected_diffusion=False):
        # assume access to class member variables which have already done some of the computation
        N, T, _ = q.shape
        d = self.d

        g = self.env_friction_constr(force.reshape(-1, 3)).reshape(N, -1)

        # compute the gradient
        if compute_grads:
            T_range = torch.arange(0, T, device=q.device)
            dg_dforce = self.grad_env_friction_constr(force.reshape(-1, 3))
            dg = 1
            grad_g = torch.zeros(N, dg, T, T, d, device=self.device)
            env_force_indices = self.contact_force_indices[-3:]
            mask_t = torch.zeros_like(grad_g).bool()  
            mask_t[:, :, T_range, T_range] = True
            mask_force = torch.zeros_like(grad_g).bool()
            mask_force[:, :, :, :, env_force_indices] = True
            grad_g[torch.logical_and(mask_t, mask_force)] = dg_dforce.reshape(N, T, dg, 3).transpose(1, 2).reshape(-1)


            # if force is not None:
            #     mask_force = torch.zeros_like(grad_h).bool()
            #     contact_force_indices = self.contact_force_indices[i * 3: (i + 1) * 3]
            #     mask_force[:, :, :, :, contact_force_indices] = True
            #     grad_h[torch.logical_and(mask_t, mask_force)] = dh_du.reshape(N, T, dh, 3).transpose(1, 2).reshape(-1)
            grad_g = grad_g.transpose(1, 2).reshape(N, -1, T * d)
        else:
            return g, None, None

        if compute_hess:
            hess_g = torch.zeros(N, g.shape[1], T * d, T * d, device=q.device)
            return g, grad_g, hess_g

        return g, grad_g, None
    def _force_equlibrium_constr_w_force(self, q, u, next_q, force_list, contact_jac_list, contact_point_list):
        # NOTE: the constriant is defined in the robot frame
        # the contact jac an contact points are all in the robot frame
        # this will be vmapped, so takes in a 3 vector and a [num_finger x 3 x 8] jacobian and a dq vector
        delta_q = q + u - next_q
        torque_list = []
        reactional_torque_list = []
        for i, finger_name in enumerate(self.contact_fingers):
            # TODO: Assume that all the fingers are having an equlibrium, maybe we should change so that index finger is not considered
            contact_jacobian = contact_jac_list[i]
            force_robot_frame = self.world_trans.inverse().transform_normals(force_list[i].unsqueeze(0)).squeeze(0)
            reactional_torque_list.append(contact_jacobian.T @ -force_robot_frame)
            # pseudo inverse form
        # force_world_frame = self.world_trans.transform_normals(force.unsqueeze(0)).squeeze(0)
        reactional_torque_list = torch.stack(reactional_torque_list, dim=0)
        sum_reactional_torque = torch.sum(reactional_torque_list, dim=0)
        g_force_torque_balance = (sum_reactional_torque + 3.0 * delta_q)
        
        sum_force = torch.sum(force_list, dim=0)
        # print(g_force_torque_balance.max(), torque_list.max())
        g = torch.cat((sum_force, g_force_torque_balance.reshape(-1)), dim=-1)
        # residual_list = torch.stack(residual_list, dim=0) * 100
        # g = torch.cat((torque_list, residual_list), dim=-1)
        return g
    def _con_eq(self, xu, compute_grads=True, compute_hess=False, verbose=False, projected_diffusion=False):
        g, grad_g, hess_g = super()._con_eq(xu, compute_grads, compute_hess, verbose, projected_diffusion)
        # g_obj_contact, grad_g_obj_contact, hess_g_obj_contact \
        #     = self._obj_wall_contact_constraint(xu[:, :, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof],
        #                                                                                 compute_grads=compute_grads,
        #                                                                                 compute_hess=compute_hess)
        # g = torch.cat((g, g_obj_contact), dim=1)
        # if grad_g is not None:
        #     grad_g = torch.cat((grad_g, grad_g_obj_contact), dim=1)
        # if hess_g is not None:
        #     hess_g = torch.cat((hess_g, hess_g_obj_contact), dim=1)
        N = xu.shape[0]
        T = xu.shape[1]
        q = xu[:, :, :self.num_fingers * 4]
        delta_q = xu[:, :, self.num_fingers * 4 + self.obj_dof:self.num_fingers * 8 + self.obj_dof]
        q = partial_to_full_state(q, fingers=self.fingers)
        delta_q = partial_to_full_state(delta_q, fingers=self.fingers)
        env_force = torch.zeros(N, T, 3, device=self.device)
        env_force[:, :, :] = xu[:, :, -3:]
        g_friction, grad_g_friction, hess_g_friction = self._env_friction_constraint(
                q=q, force=env_force,
                compute_grads=compute_grads,
                compute_hess=compute_hess,
                projected_diffusion=projected_diffusion)
        # print("env friction", g_friction.max())
        g = torch.cat((g, g_friction), dim=1)
        if grad_g is not None:
            grad_g = torch.cat((grad_g, grad_g_friction), dim=1)
        if hess_g is not None:
            hess_g = torch.cat((hess_g, hess_g_friction), dim=1)
        return g, grad_g, hess_g
    
    def _con_ineq(self, xu, compute_grads=True, compute_hess=False, verbose=False, projected_diffusion=False):
        N = xu.shape[0]
        T = xu.shape[1]
        q = xu[:, :, :self.num_fingers * 4]
        delta_q = xu[:, :, self.num_fingers * 4 + self.obj_dof:self.num_fingers * 8 + self.obj_dof]
        q = partial_to_full_state(q, fingers=self.fingers)
        delta_q = partial_to_full_state(delta_q, fingers=self.fingers)

        if self.optimize_force:
            force = torch.zeros(N, T, 12, device=self.device)
            if self.env_force:
                force[:, :, self._contact_force_indices] = xu[:, :, -(self.num_contacts + 1) * 3: -3]
            else:
                force[:, :, self._contact_force_indices] = xu[:, :, -self.num_contacts * 3:]
            h, grad_h, hess_h = self._friction_constraint(
                q=q, delta_q=delta_q, force=force,
                compute_grads=compute_grads,
                compute_hess=compute_hess,
                projected_diffusion=projected_diffusion)
        # print("friction", h.max())

        if verbose:
            print(f"max friction constraint: {torch.max(h)}")
            # print(f"max step size constraint: {torch.max(h_step_size)}")
            # print(f"max singularity constraint: {torch.max(h_sin)}")
            result_dict = {}
            result_dict['friction'] = torch.max(h).item()
            # result_dict['singularity'] = torch.max(h_sin).item()
            return result_dict

        if compute_grads:
            grad_h = grad_h.reshape(N, -1, (self.T + 1 if projected_diffusion else self.T) * self.d)
        else:
            return h, None, None
        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], self.T * (self.dx + self.du), self.T * (self.dx + self.du),
                                 device=self.device)
            return h, grad_h, hess_h
        return h, grad_h, None

class AllegroManipulationProblem(AllegroContactProblem, AllegroRegraspProblem):
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
                 desired_ee_in_world_frame=False,
                 env_contact=False,
                 min_force_dict=None,
                 device='cuda:0', **kwargs):

        # super(AllegroManipulationProblem, self).__init__(start=start, goal=goal, T=T, chain=chain,
        #                                                  object_location=object_location, object_type=object_type,
        #                                                  world_trans=world_trans, object_asset_pos=object_asset_pos,
        #                                                  regrasp_fingers=regrasp_fingers,
        #                                                  contact_fingers=contact_fingers,
        #                                                  friction_coefficient=friction_coefficient, obj_dof=obj_dof,
        #                                                  obj_ori_rep=obj_ori_rep, obj_joint_dim=obj_joint_dim,
        #                                                  optimize_force=optimize_force, device=device, **kwargs)
        moveable_object = True if len(contact_fingers) > 0 else False
        AllegroContactProblem.__init__(self, start=start, goal=goal, T=T, chain=chain,
                                        object_location=object_location, object_type=object_type,
                                        world_trans=world_trans, object_asset_pos=object_asset_pos,
                                        regrasp_fingers=regrasp_fingers,
                                        contact_fingers=contact_fingers,
                                        friction_coefficient=friction_coefficient, obj_dof=obj_dof,
                                        obj_ori_rep=obj_ori_rep, obj_joint_dim=obj_joint_dim,
                                        optimize_force=optimize_force, device=device, 
                                        desired_ee_in_world_frame=desired_ee_in_world_frame,
                                        min_force_dict=min_force_dict,
                                        **kwargs)

        AllegroRegraspProblem.__init__(self, start=start, goal=goal, T=T, chain=chain,
                                       object_location=object_location, object_type=object_type,
                                       world_trans=world_trans, object_asset_pos=object_asset_pos,
                                       regrasp_fingers=regrasp_fingers,
                                       contact_fingers=contact_fingers,
                                       obj_dof=obj_dof,
                                       obj_ori_rep=obj_ori_rep, obj_joint_dim=obj_joint_dim,
                                       device=device, optimize_force=optimize_force, moveable_object=moveable_object,
                                       desired_ee_in_world_frame=desired_ee_in_world_frame,
                                       **kwargs)
        self.dg, self.dz, self.dh = 0, 0, 0
        if self.num_regrasps > 0:
            self.dg = self._regrasp_dg
            self.dz = self._regrasp_dz
            self.dh = self._regrasp_dh
            self.dg_constant = self._regrasp_dg_constant
            self.dg_per_t = self._regrasp_dg_per_t
        if self.num_contacts > 0:
            self.dg += self._contact_dg
            self.dz += self._contact_dz
            self.dh += self._contact_dh
            self.dg_constant += self._contact_dg_constant
            self.dg_per_t += self._contact_dg_per_t

        # self.dz += self._base_dz
        # self.dh += self._base_dz * T

    def _cost(self, xu, start, goal):
        return AllegroContactProblem._cost(self, xu, start, goal) + \
            AllegroObjectProblem._cost(self, xu, start, goal) + AllegroRegraspProblem._cost(self, xu, start, goal)

    def _con_eq(self, xu, compute_grads=True, compute_hess=False, verbose=False, projected_diffusion=False):
        N, T = xu.shape[:2]
        g, grad_g, hess_g = None, None, None
        if self.num_regrasps > 0:
            g_regrasp, grad_g_regrasp, hess_g_regrasp = AllegroRegraspProblem._con_eq(self, xu, compute_grads,
                                                                                      compute_hess, projected_diffusion=projected_diffusion)
        if self.num_contacts > 0:
            g_contact, grad_g_contact, hess_g_contact = AllegroContactProblem._con_eq(self, xu, compute_grads,
                                                                                      compute_hess, projected_diffusion=projected_diffusion)
        else:
            g, grad_g, hess_g = g_regrasp, grad_g_regrasp, hess_g_regrasp

        if self.num_regrasps == 0:
            g, grad_g, hess_g = g_contact, grad_g_contact, hess_g_contact

        if g is None:
            g = torch.cat((g_regrasp, g_contact), dim=1)

        if compute_grads:
            if grad_g is None:
                grad_g = torch.cat((grad_g_regrasp, grad_g_contact), dim=1)
            grad_g = grad_g.reshape(grad_g.shape[0], grad_g.shape[1], T, -1)[:, :, :, self.all_var_index]
            grad_g = grad_g.reshape(N, -1, T * (self.dx + self.du))
            if torch.any(torch.isinf(grad_g)) or torch.any(torch.isnan(grad_g)):
                print('hello')
        if compute_hess:
            if hess_g is None:
                hess_g = torch.cat((hess_g_regrasp, hess_g_contact), dim=1)
                hess_g = hess_g.reshape(hess_g.shape[0], hess_g.shape[1], T, self.d, T, self.d)[:, :, :,
                         self.all_var_index]
                hess_g = hess_g[:, :, :, :, self.all_var_index].reshape(N, -1,
                                                                        T * (self.dx + self.du),
                                                                        T * (self.dx + self.du))

        return g, grad_g, hess_g

    def _con_ineq(self, xu, compute_grads=True, compute_hess=False, verbose=False, projected_diffusion=False):
        N, T = xu.shape[:2]
        h, grad_h, hess_h = None, None, None
        # h, grad_h, hess_h = AllegroObjectProblem._con_ineq(self, xu, compute_grads, compute_hess)

        if not compute_grads:
            grad_h = None

        if not compute_hess:
            hess_h = None

        if self.num_regrasps > 0:
            h_regrasp, grad_h_regrasp, hess_h_regrasp = AllegroRegraspProblem._con_ineq(self, xu, compute_grads,
                                                                                        compute_hess, projected_diffusion=projected_diffusion)
            if h is not None:
                h = torch.cat((h_regrasp, h), dim=1)
                if grad_h is not None:
                    grad_h = torch.cat((grad_h_regrasp, grad_h), dim=1)
                if hess_h is not None:
                    hess_h = torch.cat((hess_h_regrasp, hess_h), dim=1)
            else:
                h, grad_h, hess_h = h_regrasp, grad_h_regrasp, hess_h_regrasp

        if self.num_contacts > 0:
            h_contact, grad_h_contact, hess_h_contact = AllegroContactProblem._con_ineq(self, xu, compute_grads,
                                                                                        compute_hess, projected_diffusion=projected_diffusion)
            if h is not None:
                h = torch.cat((h, h_contact), dim=1)
                if grad_h is not None:
                    grad_h = torch.cat((grad_h, grad_h_contact), dim=1)
                if hess_h is not None:
                    hess_h = torch.cat((hess_h, hess_h_contact), dim=1)
            else:
                h, grad_h, hess_h = h_contact, grad_h_contact, hess_h_contact

        # if h is None:
        #    h = torch.cat((h_regrasp, h_contact), dim=1)

        if compute_grads:
            grad_h = grad_h.reshape(grad_h.shape[0], grad_h.shape[1], T, -1)[:, :, :, self.all_var_index]
            grad_h = grad_h.reshape(N, -1, T * (self.dx + self.du))
        if compute_hess:
            hess_h = hess_h.reshape(hess_h.shape[0], hess_h.shape[1], T, self.d, T, self.d)[:, :, :,
                     self.all_var_index]
            hess_h = hess_h[:, :, :, :, self.all_var_index].reshape(N, -1,
                                                                    T * (self.dx + self.du),
                                                                    T * (self.dx + self.du))

        return h, grad_h, hess_h

class AllegroRegraspWithEnvProblem(AllegroRegraspProblem):

    def __init__(self,
                 start,
                 goal,
                 T,
                 chain,
                 object_location,
                 object_type,
                 world_trans,
                 object_asset_pos,
                 regrasp_fingers=['index', 'middle', 'ring', 'thumb'],
                 contact_fingers=[],
                 obj_dof=1,
                 obj_ori_rep='euler',
                 obj_joint_dim=0,
                 device='cuda:0',
                 moveable_object=False,
                 optimize_force=False,
                 default_dof_pos=None,
                 desired_ee_in_world_frame=False,
                 obj_dof_type='x_y_theta',
                 *args, **kwargs):
        super(AllegroRegraspWithEnvProblem, self).__init__(start=start, goal=goal, T=T, chain=chain,
                                                           object_location=object_location, object_type=object_type,
                                                           world_trans=world_trans, object_asset_pos=object_asset_pos,
                                                           regrasp_fingers=regrasp_fingers,
                                                           contact_fingers=contact_fingers,
                                                           obj_dof=obj_dof, obj_ori_rep=obj_ori_rep,
                                                           obj_joint_dim=obj_joint_dim, device=device,
                                                           moveable_object=moveable_object,
                                                           optimize_force=optimize_force,
                                                           default_dof_pos=default_dof_pos,
                                                           desired_ee_in_world_frame=desired_ee_in_world_frame,
                                                           obj_dof_type=obj_dof_type,
                                                           *args, **kwargs)
    def _preprocess_fingers(self, q, theta):
        N, T, _ = q.shape
        if self.obj_dof_type == "x_y_theta":
            tmp = torch.zeros(N, T, 1, device=self.device)
            theta = torch.cat((theta[:, :, :2], tmp, tmp, tmp, theta[:, :, -1:]), dim=2)
        super()._preprocess_fingers(q, theta)
        if self.obj_dof_type == "x_y_theta":
            for finger in self.fingers:
                dnormal_denv_q = self.data[finger]['dnormal_denv_q']
                self.data[finger]['dnormal_denv_q'] = torch.cat((dnormal_denv_q[:, :, :2], dnormal_denv_q[:, :, -1:]), dim=2)
                grad_env_sdf = self.data[finger]['grad_env_sdf']
                self.data[finger]['grad_env_sdf'] = torch.cat((grad_env_sdf[:, :2], grad_env_sdf[:, -1:]), dim=-1)
        

class AllegroManipulationExternalContactProblem(AllegroContactWithEnvProblem, AllegroRegraspWithEnvProblem):
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
                 obj_joint_dim=0,
                 optimize_force=False,
                 desired_ee_in_world_frame=False,
                 device='cuda:0', **kwargs):

        moveable_object = True if len(contact_fingers) > 0 else False
        AllegroContactWithEnvProblem.__init__(self, start=start, goal=goal, T=T, chain=chain,
                                        object_location=object_location, object_type=object_type,
                                        world_trans=world_trans, object_asset_pos=object_asset_pos,
                                        table_asset_pos=table_asset_pos,
                                        regrasp_fingers=regrasp_fingers,
                                        contact_fingers=contact_fingers,
                                        friction_coefficient=friction_coefficient, obj_dof=obj_dof,
                                        obj_ori_rep=obj_ori_rep, obj_joint_dim=obj_joint_dim,
                                        optimize_force=optimize_force, device=device, 
                                        desired_ee_in_world_frame=desired_ee_in_world_frame,
                                        obj_dof_type='x_y_theta',
                                        **kwargs)

        AllegroRegraspWithEnvProblem.__init__(self, start=start, goal=goal, T=T, chain=chain,
                                       object_location=object_location, object_type=object_type,
                                       world_trans=world_trans, object_asset_pos=object_asset_pos,
                                       regrasp_fingers=regrasp_fingers,
                                       contact_fingers=contact_fingers,
                                       obj_dof=obj_dof,
                                       obj_ori_rep=obj_ori_rep, obj_joint_dim=obj_joint_dim,
                                       device=device, optimize_force=optimize_force, moveable_object=moveable_object,
                                       desired_ee_in_world_frame=desired_ee_in_world_frame, env_force=True,
                                       **kwargs)
        self.dg, self.dz, self.dh = 0, 0, 0
        if self.num_regrasps > 0:
            self.dg = self._regrasp_dg
            self.dz = self._regrasp_dz
            self.dh = self._regrasp_dh
            self.dg_constant = self._regrasp_dg_constant
            self.dg_per_t = self._regrasp_dg_per_t
        if self.num_contacts > 0:
            self.dg += self._contact_dg
            self.dz += self._contact_dz
            self.dh += self._contact_dh
            self.dg_constant += self._contact_dg_constant
            self.dg_per_t += self._contact_dg_per_t
            
            # self.dg_per_t += 1 # add the obj env contact constraint
            # self.dg += 1 * self.T

        # self.dz += self._base_dz
        # self.dh += self._base_dz * T

    def _cost(self, xu, start, goal):
        return AllegroContactWithEnvProblem._cost(self, xu, start, goal) + \
            AllegroObjectProblem._cost(self, xu, start, goal) + AllegroRegraspWithEnvProblem._cost(self, xu, start, goal)

    def _con_eq(self, xu, compute_grads=True, compute_hess=False, verbose=False, projected_diffusion=False):
        N, T = xu.shape[:2]
        g, grad_g, hess_g = None, None, None
        if self.num_regrasps > 0:
            g_regrasp, grad_g_regrasp, hess_g_regrasp = AllegroRegraspWithEnvProblem._con_eq(self, xu, compute_grads,
                                                                                      compute_hess, projected_diffusion=projected_diffusion)
        if self.num_contacts > 0:
            g_contact, grad_g_contact, hess_g_contact = AllegroContactWithEnvProblem._con_eq(self, xu, compute_grads,
                                                                                        compute_hess, projected_diffusion=projected_diffusion)
        else:
            g, grad_g, hess_g = g_regrasp, grad_g_regrasp, hess_g_regrasp

        if self.num_regrasps == 0:
            g, grad_g, hess_g = g_contact, grad_g_contact, hess_g_contact

        if g is None:
            g = torch.cat((g_regrasp, g_contact), dim=1)

        if compute_grads:
            if grad_g is None:
                grad_g = torch.cat((grad_g_regrasp, grad_g_contact), dim=1)
            grad_g = grad_g.reshape(grad_g.shape[0], grad_g.shape[1], T, -1)[:, :, :, self.all_var_index]
            grad_g = grad_g.reshape(N, -1, T * (self.dx + self.du))
            if torch.any(torch.isinf(grad_g)) or torch.any(torch.isnan(grad_g)):
                print('hello')
        if compute_hess:
            if hess_g is None:
                hess_g = torch.cat((hess_g_regrasp, hess_g_contact), dim=1)
                hess_g = hess_g.reshape(hess_g.shape[0], hess_g.shape[1], T, self.d, T, self.d)[:, :, :,
                         self.all_var_index]
                hess_g = hess_g[:, :, :, :, self.all_var_index].reshape(N, -1,
                                                                        T * (self.dx + self.du),
                                                                        T * (self.dx + self.du))

        return g, grad_g, hess_g

    def _con_ineq(self, xu, compute_grads=True, compute_hess=False, verbose=False, projected_diffusion=False):
        N, T = xu.shape[:2]
        h, grad_h, hess_h = None, None, None
        # h, grad_h, hess_h = AllegroObjectProblem._con_ineq(self, xu, compute_grads, compute_hess)

        if not compute_grads:
            grad_h = None

        if not compute_hess:
            hess_h = None

        if self.num_regrasps > 0:
            h_regrasp, grad_h_regrasp, hess_h_regrasp = AllegroRegraspWithEnvProblem._con_ineq(self, xu, compute_grads,
                                                                                        compute_hess, projected_diffusion=projected_diffusion)
            if h is not None:
                h = torch.cat((h_regrasp, h), dim=1)
                if grad_h is not None:
                    grad_h = torch.cat((grad_h_regrasp, grad_h), dim=1)
                if hess_h is not None:
                    hess_h = torch.cat((hess_h_regrasp, hess_h), dim=1)
            else:
                h, grad_h, hess_h = h_regrasp, grad_h_regrasp, hess_h_regrasp

        if self.num_contacts > 0:
            h_contact, grad_h_contact, hess_h_contact = AllegroContactWithEnvProblem._con_ineq(self, xu, compute_grads,
                                                                                        compute_hess, projected_diffusion=projected_diffusion)
            if h is not None:
                h = torch.cat((h, h_contact), dim=1)
                if grad_h is not None:
                    grad_h = torch.cat((grad_h, grad_h_contact), dim=1)
                if hess_h is not None:
                    hess_h = torch.cat((hess_h, hess_h_contact), dim=1)
            else:
                h, grad_h, hess_h = h_contact, grad_h_contact, hess_h_contact

        # if h is None:
        #    h = torch.cat((h_regrasp, h_contact), dim=1)

        if compute_grads:
            grad_h = grad_h.reshape(grad_h.shape[0], grad_h.shape[1], T, -1)[:, :, :, self.all_var_index]
            grad_h = grad_h.reshape(N, -1, T * (self.dx + self.du))
        if compute_hess:
            hess_h = hess_h.reshape(hess_h.shape[0], hess_h.shape[1], T, self.d, T, self.d)[:, :, :,
                     self.all_var_index]
            hess_h = hess_h[:, :, :, :, self.all_var_index].reshape(N, -1,
                                                                    T * (self.dx + self.du),
                                                                    T * (self.dx + self.du))

        return h, grad_h, hess_h


class IpoptManipulationProblem(AllegroManipulationProblem, IpoptProblem):

    def __init__(self, *args, **kwargs):
        device = kwargs.get('device', None)
        if device is not None:
            kwargs.pop('device')
        super().__init__(*args, **kwargs, N=1, device='cpu')


def do_trial(env, params, fpath, sim_viz_env=None, ros_copy_node=None):
    "only turn the valve once"
    num_contacts = len(params['fingers'])
    state = env.get_state()
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None

    start = state['q'].reshape(4 * num_contacts + 1).to(device=params['device'])
    # start = torch.cat((state['q'].reshape(10), torch.zeros(1).to(state['q'].device))).to(device=params['device'])
    if params['controller'] == 'csvgd':
        pregrasp_problem = AllegroManipulationProblem(
            start=start[:4 * num_contacts + 1],
            goal=params['valve_goal'] * 0,
            T=4,
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.valve_pose,
            object_location=params['object_location'],
            object_type=params['object_type'],
            world_trans=env.world_trans,
            regrasp_fingers=params['fingers'],
            contact_fingers=[],
        )

        pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, params)
        pregrasp_planner.warmup_iters = 40

        # turn_problem = AllegroValveTurning(
        #     start=start,
        #     goal=params['valve_goal'],
        #     T=params['T'],
        #     chain=params['chain'],
        #     device=params['device'],
        #     object_asset_pos=env.valve_pose,
        #     object_location=params['object_location'],
        #     object_type=params['object_type'],
        #     friction_coefficient=params['friction_coefficient'],
        #     world_trans=env.world_trans,
        #     fingers=params['fingers'],
        #     optimize_force=params['optimize_force']
        # )

        # turn_planner = PositionControlConstrainedSVGDMPC(turn_problem, params)
    elif params['controller'] == 'ipopt':
        pregrasp_problem = IpoptManipulationProblem(
            start=start[:4 * num_contacts + 1],
            goal=params['valve_goal'] * 0,
            T=4,
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.valve_pose,
            object_location=params['object_location'],
            object_type=params['object_type'],
            world_trans=env.world_trans,
            regrasp_fingers=params['fingers'],
            contact_fingers=[],
        )
        pregrasp_planner = IpoptMPC(pregrasp_problem, params)
        pregrasp_planner.warmup_iters = 40
    else:
        raise ValueError('Invalid controller')

    # first we move the hand to grasp the valve
    start = state['q'].reshape(4 * num_contacts + 1).to(device=params['device'])
    best_traj, _ = pregrasp_planner.step(start[:4 * num_contacts + 1])

    # we will just execute this open loop
    for x in best_traj[:, :4 * num_contacts]:
        if params['mode'] == 'hardware':
            sim_viz_env.set_pose(env.get_state()['all_state'].to(device=env.device))
            sim_viz_env.step(x.reshape(-1, 4 * num_contacts).to(device=env.device))
        env.step(x.reshape(-1, 4 * num_contacts).to(device=env.device))
        if params['mode'] == 'hardware_copy':
            ros_copy_node.apply_action(partial_to_full_state(x.reshape(-1, 4 * num_contacts)[0], params['fingers']))
            # ros_node.apply_action(action[0].detach().cpu().numpy())

    actual_trajectory = []
    duration = 0

    fig = plt.figure()
    axes = {params['fingers'][i]: fig.add_subplot(int(f'1{num_contacts}{i + 1}'), projection='3d') for i in
            range(num_contacts)}
    for finger in params['fingers']:
        axes[finger].set_title(finger)
        axes[finger].set_aspect('equal')
        axes[finger].set_xlabel('x', labelpad=20)
        axes[finger].set_ylabel('y', labelpad=20)
        axes[finger].set_zlabel('z', labelpad=20)
        axes[finger].set_xlim3d(0.8, 0.87)
        axes[finger].set_ylim3d(0.52, 0.58)
        axes[finger].set_zlim3d(1.36, 1.46)
    finger_traj_history = {}
    for finger in params['fingers']:
        finger_traj_history[finger] = []
    state = env.get_state()
    start = state['q'].reshape(4 * num_contacts + 1).to(device=params['device'])
    if 'csvgd' in params['controller']:
        turn_problem = AllegroManipulationProblem(
            start=start,
            goal=params['valve_goal'],
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.valve_pose,
            object_location=params['object_location'],
            object_type=params['object_type'],
            friction_coefficient=params['friction_coefficient'],
            world_trans=env.world_trans,
            contact_fingers=params['fingers'],
            regrasp_fingers=[],
            optimize_force=params['optimize_force']
        )
        turn_planner = PositionControlConstrainedSVGDMPC(turn_problem, params)
    elif 'ipopt' in params['controller']:
        turn_problem = IpoptManipulationProblem(
            start=start,
            goal=params['valve_goal'],
            T=params['T'],
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.valve_pose,
            object_location=params['object_location'],
            object_type=params['object_type'],
            friction_coefficient=params['friction_coefficient'],
            world_trans=env.world_trans,
            contact_fingers=params['fingers'],
            regrasp_fingers=[],
            optimize_force=params['optimize_force']
        )
        turn_planner = IpoptMPC(turn_problem, params)

    else:
        raise ValueError('Invalid controller')

    for finger in params['fingers']:
        ee = state2ee_pos(start[:4 * num_contacts], turn_problem.ee_names[finger])
        finger_traj_history[finger].append(ee.detach().cpu().numpy())

    info_list = []

    for k in range(params['num_steps']):
        state = env.get_state()
        start = state['q'].reshape(4 * num_contacts + 1).to(device=params['device'])

        actual_trajectory.append(state['q'].reshape(4 * num_contacts + 1).clone())
        start_time = time.time()
        best_traj, trajectories = turn_planner.step(start)

        print(f"solve time: {time.time() - start_time}")
        planned_theta_traj = best_traj[:, 4 * num_contacts].detach().cpu().numpy()
        print(f"current theta: {state['q'][0, -1].detach().cpu().numpy()}")
        print(f"planned theta: {planned_theta_traj}")
        # add trajectory lines to sim
        if params['mode'] == 'hardware':
            add_trajectories_hardware(trajectories, best_traj, axes, env, config=params, state2ee_pos_func=state2ee_pos)
        else:
            if best_traj.shape[0] > 1 and False:
                add_trajectories(trajectories.to(device=env.device),
                                 best_traj.to(device=env.device), axes, env, sim=sim, gym=gym, viewer=viewer,
                                 config=params, state2ee_pos_func=state2ee_pos)

        if params['visualize_plan']:
            traj_for_viz = best_traj[:, :turn_problem.dx]
            viz_fpath = pathlib.PurePath.joinpath(fpath, f"timestep_{k}")
            img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
            gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
            pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
            pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
            visualize_trajectory(traj_for_viz, turn_problem.contact_scenes, viz_fpath, turn_problem.fingers,
                                 turn_problem.obj_dof + 1)

        # process the action
        x = best_traj[0, :turn_problem.dx + turn_problem.du]
        x = x.reshape(1, turn_problem.dx + turn_problem.du)
        print("--------------------------------------")

        action = x[:, turn_problem.dx:turn_problem.dx + 4 * turn_problem.num_contacts].to(device=env.device)
        start = start.to(device=env.device)
        action = action + start.unsqueeze(0)[:,
                          :4 * num_contacts]  # NOTE: this is required since we define action as delta action
        if params['mode'] == 'hardware':
            sim_viz_env.set_pose(env.get_state()['all_state'].to(device=env.device))
            sim_viz_env.step(action)
        elif params['mode'] == 'hardware_copy':
            ros_copy_node.apply_action(partial_to_full_state(action[0], params['fingers']))
        # DEBUG ONLY, execute the state sequence
        # action = best_traj[1, :4 * turn_problem.num_contacts].unsqueeze(0)
        env.step(action)
        # time.sleep(3.0)
        # current_theta = env.get_state()['q'][0, -1]
        # if current_theta.detach().item() > best_traj[0, 8].detach().item() - 0.1:
        #     print("satisfied")
        #     break
        # time.sleep(0.1)
        # hardware_joint_q = ros_copy_node.current_joint_pose.position
        # hardware_joint_q = torch.tensor(hardware_joint_q).to(env.device)
        # hardware_joint_q = full_to_partial_state(hardware_joint_q).unsqueeze(0)
        # print("ee index commanded-----------------")
        # print(state2ee_pos(action[:, :8], turn_problem.index_ee_name))
        # print("ee index actual-----------------")
        # print(state2ee_pos(hardware_joint_q[:, :8], turn_problem.index_ee_name))
        # print('ee index error-----------------')
        # print(state2ee_pos(action[:, :8], turn_problem.index_ee_name) - state2ee_pos(hardware_joint_q[:, :8], turn_problem.index_ee_name))

        # print("ee thumb commanded-----------------")
        # print(state2ee_pos(action[:, :8], turn_problem.thumb_ee_name))
        # print("ee thumb actual-----------------")
        # print(state2ee_pos(hardware_joint_q[:, :8], turn_problem.thumb_ee_name))
        # print('ee thumb error-----------------')
        # print(state2ee_pos(action[:, :8], turn_problem.thumb_ee_name) - state2ee_pos(hardware_joint_q[:, :8], turn_problem.thumb_ee_name))

        # if params['hardware']:
        #     # ros_node.apply_action(action[0].detach().cpu().numpy())
        #     ros_node.apply_action(partial_to_full_state(action[0]).detach().cpu().numpy())
        turn_problem._preprocess(best_traj.unsqueeze(0))
        equality_constr_dict = turn_problem._con_eq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False,
                                                    verbose=True)
        inequality_constr_dict = turn_problem._con_ineq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False,
                                                        verbose=True)

        # print(turn_problem.thumb_contact_scene.scene_collision_check(partial_to_full_state(x[:, :8]), x[:, 8],
        #                                                         compute_gradient=False, compute_hessian=False))
        # distance2surface = torch.sqrt((best_traj_ee[:, 2] - object_location[2].unsqueeze(0)) ** 2 + (best_traj_ee[:, 0] - object_location[0].unsqueeze(0))**2)
        distance2goal = (params['valve_goal'].cpu() - env.get_state()['q'][:, -1].cpu()).detach().cpu().item()
        print(f"distance to goal: {distance2goal}")
        # info = {**equality_constr_dict, **inequality_constr_dict, **{'distance2goal': distance2goal}}
        # info_list.append(info)

        gym.clear_lines(viewer)
        # for debugging
        state = env.get_state()
        start = state['q'].reshape(4 * num_contacts + 1).to(device=params['device'])
        for finger in params['fingers']:
            ee = state2ee_pos(start[:4 * num_contacts], turn_problem.ee_names[finger])
            finger_traj_history[finger].append(ee.detach().cpu().numpy())
        for finger in params['fingers']:
            traj_history = finger_traj_history[finger]
            temp_for_plot = np.stack(traj_history, axis=0)
            if k >= 2:
                axes[finger].plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'gray',
                                    label='actual')
    # with open(f'{fpath.resolve()}/info.pkl', 'wb') as f:
    #    pkl.dump(info_list, f)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # newLabels, newHandles = [], []
    # for handle, label in zip(handles, labels):
    #     if label not in newLabels:
    #         newLabels.append(label)
    #         newHandles.append(handle)
    # fig.tight_layout()
    # fig.legend(newHandles, newLabels, loc='lower center', ncol=3)
    # # plt.savefig(f'{fpath.resolve()}/traj.png')
    # # plt.close()
    # plt.show()

    env.reset()
    state = env.get_state()
    state = state['q'].reshape(4 * num_contacts + 1).to(device=params['device'])
    actual_trajectory.append(state.clone())
    actual_trajectory = torch.stack([a.to(device=params['device']) for a in actual_trajectory], dim=0).reshape(-1, 4 * num_contacts + 1)
    turn_problem.T = actual_trajectory.shape[0]
    # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    final_distance_to_goal = (actual_trajectory[:, -1] - params['valve_goal']).abs()

    print(f'Controller: {params["controller"]} Final distance to goal: {torch.min(final_distance_to_goal)}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')
    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
             #  constr=constraint_val.cpu().numpy(),
             d2goal=final_distance_to_goal.cpu().numpy())
    return torch.min(final_distance_to_goal).cpu().numpy()


def add_trajectories(trajectories, best_traj, axes, env, sim, gym, viewer, config, state2ee_pos_func, show_force=False):
    M = len(trajectories)
    T = len(best_traj)
    print(trajectories.shape, best_traj.shape)
    fingers = copy.copy(config['fingers'])
    if 'exclude_index' in config.keys() and config['exclude_index']:
        fingers.remove('index')
    num_contacts = len(fingers)
    obj_dof = config['obj_dof']
    if M > 0:
        initial_state = env.get_state()['q']
        # num_contacts = initial_state.shape[1] // 4
        if 'exclude_index' in config.keys() and config['exclude_index']:
            initial_state = initial_state[:, 4: 4 * (num_contacts + 1)]
        else:
            initial_state = initial_state[:, :4 * num_contacts]
        if show_force:
            force = best_traj[:, (4 + 4) * num_contacts + obj_dof:].reshape(T, num_contacts, 3) / 2
            # force = env.world_trans.transform_normals(force) / 10 # add a scaling factor since the force is too large now
        all_state = torch.cat((initial_state, best_traj[:-1, :4 * num_contacts]), dim=0)
        desired_state = all_state + best_traj[:,
                                    4 * num_contacts + obj_dof: 4 * num_contacts + obj_dof + 4 * num_contacts]

        desired_best_traj_ee = [state2ee_pos_func(desired_state, config['ee_names'][finger], fingers=fingers) for finger
                                in fingers]
        best_traj_ee = [state2ee_pos_func(best_traj[:, :4 * num_contacts], config['ee_names'][finger], fingers=fingers)
                        for finger in fingers]

        state_colors = np.array([0, 0, 1]).astype(np.float32)
        desired_state_colors = np.array([0, 1, 1]).astype(np.float32)
        force_colors = np.array([1, 1, 1]).astype(np.float32)

        for e in env.envs:
            T = best_traj.shape[0]
            for t in range(T):
                for i, finger in enumerate(fingers):
                    if t == 0:
                        initial_ee = state2ee_pos_func(initial_state, config['ee_names'][finger], fingers=fingers)
                        state_traj = torch.stack((initial_ee, best_traj_ee[i][0]), dim=0).cpu().numpy()
                        action_traj = torch.stack((initial_ee, desired_best_traj_ee[i][0]), dim=0).cpu().numpy()
                        axes[finger].plot3D(state_traj[:, 0], state_traj[:, 1], state_traj[:, 2], 'blue',
                                            label='desired next state')
                        axes[finger].plot3D(action_traj[:, 0], action_traj[:, 1], action_traj[:, 2], 'green',
                                            label='raw commanded position')
                        if show_force:
                            force_traj = torch.stack((initial_ee, initial_ee + force[t, i]), dim=0).cpu().numpy()
                            axes[finger].plot3D(force_traj[:, 0], force_traj[:, 1], force_traj[:, 2], 'red',
                                                label='force')
                    else:
                        state_traj = torch.stack((best_traj_ee[i][t - 1, :3], best_traj_ee[i][t, :3]),
                                                 dim=0).cpu().numpy()
                        action_traj = torch.stack((best_traj_ee[i][t - 1, :3], desired_best_traj_ee[i][t, :3]),
                                                  dim=0).cpu().numpy()
                        if show_force:
                            force_traj = torch.stack(
                                (best_traj_ee[i][t - 1, :3], best_traj_ee[i][t - 1, :3] + force[t, i]),
                                dim=0).cpu().numpy()
                    state_traj = state_traj.reshape(2, 3)
                    action_traj = action_traj.reshape(2, 3)

                    gym.add_lines(viewer, e, 1, state_traj, state_colors)
                    gym.add_lines(viewer, e, 1, action_traj, desired_state_colors)
                    if show_force:
                        gym.add_lines(viewer, e, 1, force_traj, force_colors)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)


def add_trajectories_hardware(trajectories, best_traj, axes, env, config, state2ee_pos_func):
    M = len(trajectories)
    fingers = config['fingers']
    if M > 0:
        initial_state = env.get_state()['q']
        num_contacts = initial_state.shape[1] // 4
        initial_state = initial_state[:, :4 * num_contacts]
        all_state = torch.cat((initial_state, best_traj[:-1, :4 * num_contacts]), dim=0)
        desired_state = all_state + best_traj[:, 9:17]

        desired_best_traj_ee = [state2ee_pos_func(desired_state, ee_names[finger], fingers=fingers) for finger in
                                fingers]
        best_traj_ee = [state2ee_pos_func(best_traj[:, :4 * num_contacts], ee_names[finger], fingers=fingers) for finger
                        in fingers]

        for i, finger in enumerate(fingers):
            initial_ee = state2ee_pos_func(initial_state, ee_names[finger], fingers=fingers)
            state_traj = torch.stack((initial_ee, best_traj_ee[i][0]), dim=0).cpu().numpy()
            action_traj = torch.stack((initial_ee, desired_best_traj_ee[i][0]), dim=0).cpu().numpy()
            axes[finger].plot3D(action_traj[:, 0], action_traj[:, 1], action_traj[:, 2], 'green',
                                label='raw commanded position')


if __name__ == "__main__":
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_valve.yaml').read_text())
    from tqdm import tqdm

    if config['object_type'] == 'cuboid_valve':
        valve_type = 'cuboid'
    elif config['object_type'] == 'cylinder_valve':
        valve_type = 'cylinder'
    config['object_dof'] = 1

    if config['mode'] == 'hardware':
        env = RosAllegroValveTurningEnv(1, control_mode='joint_impedance',
                                        use_cartesian_controller=False,
                                        viewer=True,
                                        steps_per_action=60,
                                        friction_coefficient=1.0,
                                        device=config['sim_device'],
                                        valve=valve_type,
                                        video_save_path=img_save_dir,
                                        joint_stiffness=config['kp'],
                                        fingers=config['fingers'],
                                        )
    else:
        env = AllegroValveTurningEnv(1, control_mode='joint_impedance',
                                     use_cartesian_controller=False,
                                     viewer=True,
                                     steps_per_action=60,
                                     friction_coefficient=1.0,
                                     device=config['sim_device'],
                                     valve=valve_type,
                                     video_save_path=img_save_dir,
                                     joint_stiffness=config['kp'],
                                     fingers=config['fingers'],
                                     random_robot_pose=config['random_robot_pose'],
                                     )

    sim, gym, viewer = env.get_sim()

    # state = env.get_state()
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
    index_ee_name = 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link'
    thumb_ee_name = 'allegro_hand_oya_finger_3_aftc_base_link'
    config['ee_names'] = ee_names
    config['obj_dof'] = 1

    chain = pk.build_chain_from_urdf(open(asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]  # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices,
                           world_trans=env.world_trans)
    # full_to_partial_state = partial(full_to_partial_state, fingers=config['fingers'])
    # partial_to_full_state = partial(partial_to_full_state, fingers=config['fingers'])

    for i in tqdm(range(config['num_trials'])):
        goal = 0.5 * torch.tensor([np.pi])
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
            object_location = torch.tensor([0.85, 0.70, 1.405]).to(params['device'])  # the root of the valve
            params['object_location'] = object_location
            final_distance_to_goal = do_trial(env, params, fpath, sim_env, ros_copy_node)
            # final_distance_to_goal = turn(env, params, fpath)

            if controller not in results.keys():
                results[controller] = [final_distance_to_goal]
            else:
                results[controller].append(final_distance_to_goal)
        print(results)

    csvdg_final_distance_to_goal = np.array(results['csvgd'])
    print(f"mean csvdg final dist to goal: {csvdg_final_distance_to_goal.mean()}")
    print(f"std csvdg final dist to goal: {csvdg_final_distance_to_goal.std()}")
    if config['random_robot_pose']:
        print(env.random_bias)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
