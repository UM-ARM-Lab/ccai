from isaac_victor_envs.utils import get_assets_dir

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

from ccai.kernels import rbf_kernel, structured_rbf_kernel

from ccai.problem import ConstrainedSVGDProblem
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


class AllegroObjectProblemDiff(ConstrainedSVGDProblem):

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
            # indices for contact forces
            self._contact_force_indices_dict = {
                'index': [0, 1, 2],
                'middle': [3, 4, 5],
                'ring': [6, 7, 8],
                'thumb': [9, 10, 11]
            }
            self._contact_force_indices = [self._contact_force_indices_dict[finger] for finger in contact_fingers]
            self._contact_force_indices = list(itertools.chain.from_iterable(self._contact_force_indices))
            self.contact_force_indices = [32 + self.obj_dof + idx for idx in self._contact_force_indices]
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
        robot_sdf = pv.RobotSDF(chain, path_prefix=get_assets_dir() + '/xela_models',
                                use_collision_geometry=False)

        scene_trans = world_trans.inverse().compose(
            pk.Transform3d(device=device).translate(object_asset_pos[0], object_asset_pos[1], object_asset_pos[2]))

        # contact checking
        self.contact_scenes = {}
        collision_check_links = [self.ee_names[finger] for finger in self.fingers]
        self.contact_scenes = pv.RobotScene(robot_sdf, object_sdf, scene_trans,
                                            collision_check_links=collision_check_links,
                                            softmin_temp=1.0e3,
                                            points_per_link=1000,
                                            )

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
        self._preprocess_fingers(q, theta, self.contact_scenes)

    def _preprocess_fingers(self, q, theta, finger_ee_link):
        N, _, _ = q.shape

        # reshape to batch across time
        q_b = q.reshape(-1, 4 * self.num_fingers)
        theta_b = theta.reshape(-1, self.obj_dof)
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

            self.data[finger]['dnormal_denv_q'] = ret_scene['dnormal_denv_q'][:, i, :, :self.obj_dof]
            self.data[finger]['grad_env_sdf'] = ret_scene['grad_env_sdf'][:, i, :self.obj_dof]
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


class AllegroRegraspProblemDiff(AllegroObjectProblemDiff):

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
                 *args, **kwargs):

        # object_location is different from object_asset_pos. object_asset_pos is 
        # used for pytorch volumetric. The asset of valve might contain something else such as a wall, a table
        # object_location is the location of the object joint, which is what we care for motion planning
        if kwargs.get('fingers', None) is None:
            kwargs['fingers'] = regrasp_fingers + contact_fingers

        if kwargs.get('dx', None) is None:
            num_fingers = len(kwargs['fingers'])
            dx = 4 * num_fingers + obj_dof
            du = 4 * num_fingers
            if optimize_force:
                du += 3 * len(contact_fingers)
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
        self._regrasp_dg_per_t = self.num_regrasps * 1
        self._regrasp_dg_constant = self.num_regrasps
        self._regrasp_dg = self._regrasp_dg_per_t * T + self._regrasp_dg_constant
        self._regrasp_dz = 0  # one contact constraints per finger
        self._regrasp_dh = self._regrasp_dz * T  # inequality

        if default_dof_pos is None:
            self.default_dof_pos = torch.cat((torch.tensor([[0.1, 0.5, 0.5, 0.5]]).float().to(device=self.device),
                                              torch.tensor([[-0.1, 0.5, 0.65, 0.65]]).float().to(device=self.device),
                                              torch.tensor([[0., 0.5, 0.65, 0.65]]).float().to(device=self.device),
                                              torch.tensor([[1.2, 0.3, 0.2, 1.]]).float().to(device=self.device)),
                                             dim=1).to(self.device).reshape(-1)
        else:
            self.default_dof_pos = default_dof_pos

        if self.num_regrasps > 0:
            self.default_ee_locs = self._ee_locations_in_screwdriver(self.default_dof_pos,
                                                                     torch.zeros(self.obj_dof, device=self.device))

            # add a small amount of noise to ee loc default
            #self.default_ee_locs = self.default_ee_locs #+ 0.01 * torch.randn_like(self.default_ee_locs)
        else:
            self.default_ee_locs = None

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
        if self.obj_dof == 3:
            object_link_name = 'screwdriver_body'
        else:
            object_link_name = 'valve'

        ee_locs = object_trans[object_link_name].inverse().transform_points(ee_locs)

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
        return 1000 * torch.sum((self.default_ee_locs - self._ee_locations_in_screwdriver(q, theta)) ** 2)

        # dof_pos = self.default_dof_pos[None, self.regrasp_idx]
        # return 10 * torch.sum((q - dof_pos) ** 2)

    @regrasp_finger_constraints
    def _contact_avoidance(self, xu, finger_name, compute_grads=True, compute_hess=False, projected_diffusion=False):
        h, grad_h, hess_h = self._contact_constraints(xu, finger_name, compute_grads, compute_hess, terminal=False, projected_diffusion=projected_diffusion)
        eps = torch.zeros_like(h)
        eps[:, :-1] = 5e-3
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
        g, grad_g, hess_g = self._terminal_contact_constraint(
            xu=xu.reshape(N, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess,
            projected_diffusion=projected_diffusion)

        return g, grad_g, hess_g

    def _con_ineq(self, xu, compute_grads=True, compute_hess=False, projected_diffusion=False):
        return None, None, None


class AllegroContactProblemDiff(AllegroObjectProblemDiff):

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
        self.optimize_force = optimize_force
        self.num_contacts = len(contact_fingers)
        self.contact_fingers = contact_fingers
        num_fingers = self.num_contacts + len(regrasp_fingers)
        dx = 4 * num_fingers + obj_dof
        if optimize_force:
            du = (4 + 3) * num_fingers
        else:
            du = 4 * num_fingers
        super().__init__(dx=dx, du=du, start=start, goal=goal,
                         T=T, chain=chain, object_location=object_location,
                         object_type=object_type, world_trans=world_trans, object_asset_pos=object_asset_pos,
                         fingers=regrasp_fingers + contact_fingers, obj_dof=obj_dof, obj_ori_rep=obj_ori_rep,
                         obj_joint_dim=obj_joint_dim, device=device, moveable_object=True,
                         contact_fingers=contact_fingers,
                         regrasp_fingers=regrasp_fingers, **kwargs)

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

        self._contact_dg_per_t = self.num_contacts * (1) + 3
            # self.dg_per_t = self.num_fingers * (1 + 3 + 2) + 3
            # self.dg_per_t = self.num_fingers * (1 + 3 + 2)
        self._contact_dg_constant = 0
        self._contact_dg = self._contact_dg_per_t * T + self._contact_dg_constant  # terminal contact points, terminal sdf=0, and dynamics
        self._contact_dz = 0  # one friction constraints per finger
        self._contact_dh = self._contact_dz * T  # inequality

    def get_initial_xu(self, N):
        """
        use delta joint movement to get the initial trajectory
        the action (force at the finger tip) is not used. it is randomly intiailized
        the actual dynamics model is not used
        """

        u = 0.025 * torch.randn(N, self.T, self.du, device=self.device)

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
            cost += torch.sum(force ** 2)

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

    def _force_equlibrium_constraints(self, q, delta_q, compute_grads=True, compute_hess=False):
        N, T = q.shape[:2]
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
            jac = self.data[finger]['contact_jacobian'].reshape(N, T + 1, 3, -1)[:, 1:, :, self.contact_state_indices]
            contact_points = self.data[finger]['closest_pt_world'].reshape(N, T + 1, 3)[:, 1:].reshape(-1, 3)
            contact_jac_list.append(jac.reshape(N * (T + 1), 3, -1))
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
                djac_dnext_q = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, 16, 16)[:, 1:, :,
                               self.contact_state_indices][:, :, :, :, self.contact_state_indices]

                dg_dnext_q = dg_dnext_q + dg_djac[:, :, i].reshape(N, T, g.shape[2], -1) @ \
                             djac_dnext_q.reshape(N, T, -1, 4 * self.num_contacts)

                d_contact_loc_dq = self.data[finger_name]['closest_pt_q_grad']
                d_contact_loc_dq = d_contact_loc_dq.reshape(N, T + 1, 3, 16)[:, :-1, :, self.contact_state_indices]
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

    def _force_equlibrium_constraints_w_force(self, q, delta_q, force, compute_grads=True, compute_hess=False):
        N, T = q.shape[:2]
        device = q.device
        d = self.d

        full_start = partial_to_full_state(self.start[None, :self.num_fingers * 4], self.fingers)
        q = torch.cat((full_start.reshape(1, 1, -1).repeat(N, 1, 1), q), dim=1)
        next_q = q[:, 1:, self.contact_state_indices]
        q = q[:, :-1, self.contact_state_indices]
        u = delta_q[:, :, self.contact_state_indices]
        force_list = force[:, :, self._contact_force_indices].reshape(force.shape[0], force.shape[1], self.num_contacts,
                                                                      3)

        # retrieve contact jacobians and points
        contact_jac_list = []
        contact_point_list = []
        for finger in self.contact_fingers:
            jac = self.data[finger]['contact_jacobian'].reshape(N, T + 1, 3, -1)[:, 1:, :, self.contact_state_indices]
            contact_points = self.data[finger]['closest_pt_world'].reshape(N, T + 1, 3)[:, 1:].reshape(-1, 3)
            contact_jac_list.append(jac.reshape(N * T, 3, -1))
            contact_point_list.append(contact_points)

        contact_jac_list = torch.stack(contact_jac_list, dim=1).to(device=device)
        contact_point_list = torch.stack(contact_point_list, dim=1).to(device=device)

        g = self.force_equlibrium_constr(q.reshape(-1, 4 * self.num_contacts),
                                         u.reshape(-1, 4 * self.num_contacts),
                                         next_q.reshape(-1, 4 * self.num_contacts),
                                         force_list.reshape(-1, self.num_contacts, 3),
                                         contact_jac_list,
                                         contact_point_list).reshape(N, T, -1)

        if compute_grads:
            dg_dq, dg_du, dg_dnext_q, dg_dforce, dg_djac, dg_dcontact = self.grad_force_equlibrium_constr(
                q.reshape(-1, 4 * self.num_contacts),
                u.reshape(-1, 4 * self.num_contacts),
                next_q.reshape(-1, 4 * self.num_contacts),
                force_list.reshape(-1, self.num_contacts, 3),
                contact_jac_list,
                contact_point_list)
            dg_dforce = dg_dforce.reshape(dg_dforce.shape[0], dg_dforce.shape[1], self.num_contacts * 3)

            T_range = torch.arange(T, device=device)
            T_plus = torch.arange(1, T, device=device)
            T_minus = torch.arange(T - 1, device=device)

            grad_g = torch.zeros(N, g.shape[2], T, T, self.d, device=self.device)
            dg_dq = dg_dq.reshape(N, T, g.shape[2], 4 * self.num_contacts)
            dg_dnext_q = dg_dnext_q.reshape(N, T, g.shape[2], 4 * self.num_contacts)

            for i, finger_name in enumerate(self.contact_fingers):
                # NOTE: assume fingers have joints independent of each other
                # TODO: check if we should use the jacobian of the current time steps or the jacobian of the next time steps.
                djac_dnext_q = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, 16, 16)[:, 1:, :,
                               self.contact_state_indices][:, :, :, :, self.contact_state_indices]

                dg_dnext_q = dg_dnext_q + dg_djac[:, :, i].reshape(N, T, g.shape[2], -1) @ \
                             djac_dnext_q.reshape(N, T, -1, 4 * self.num_contacts)

                d_contact_loc_dq = self.data[finger_name]['closest_pt_q_grad']
                d_contact_loc_dq = d_contact_loc_dq.reshape(N, T + 1, 3, 16)[:, :-1, :, self.contact_state_indices]
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
                                                                              self.num_contacts * 3
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
            obj_omega = euler_to_angular_velocity(current_theta, next_theta)
        elif self.obj_dof == 1:
            dtheta = next_theta - current_theta
            obj_omega = torch.concat((torch.zeros_like(dtheta),
                                      dtheta,
                                      torch.zeros_like(dtheta)), -1)  # should be N x T-1 x 3

        contact_point_v = (contact_jac @ dq.reshape(4 * self.num_contacts, 1)).squeeze(-1)

        # compute valve contact point velocity
        valve_robot_frame = self.world_trans.inverse().transform_points(self.object_location.reshape(1, 3))
        contact_point_r_valve = contact_loc.reshape(3) - valve_robot_frame.reshape(3)
        obj_omega_robot_frame = self.world_trans.inverse().transform_normals(obj_omega.reshape(1, 3)).reshape(-1)
        object_contact_point_v = torch.cross(obj_omega_robot_frame, contact_point_r_valve, dim=-1)

        # project the constraint into the tangential plane
        normal_projection = contact_normal.unsqueeze(-1) @ contact_normal.unsqueeze(-2)
        R = self.get_rotation_from_normal(contact_normal.reshape(-1, 3)).reshape(3, 3).detach().transpose(1, 0)
        R = R[:2]

        # compute contact v tangential to surface
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

    def _friction_constr_force(self, dq, contact_normal, contact_jacobian):
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
        if False:  # self.optimize_force:
            # here dq means the force in the world frame
            contact_v_contact_frame = R.transpose(0, 1) @ dq
        else:
            contact_v_contact_frame = R.transpose(0, 1) @ self.world_trans.transform_normals(
                (contact_jacobian @ dq).unsqueeze(0)).squeeze(0)
        # TODO: there are two different ways of doing a friction cone
        # Linearized friction cone - but based on the contact point velocity
        # force is defined as the force of robot pushing the object
        return B @ contact_v_contact_frame

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
        # TODO: there are two different ways of doing a friction cone
        # Linearized friction cone - but based on the contact point velocity
        # force is defined as the force of robot pushing the object
        return B @ contact_v_contact_frame

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
                      :, :-1, :, self.contact_state_indices]
        contact_normal = self.data[finger_name]['contact_normal'].reshape(N, T + T_offset, 3)[:,
                         :-1]  # contact normal is pointing out
        dnormal_dq = self.data[finger_name]['dnormal_dq'].reshape(N, T + T_offset, 3, 16)[
                     :, :-1, :, self.contact_state_indices]
        dnormal_dtheta = self.data[finger_name]['dnormal_denv_q'].reshape(N, T + T_offset, 3, self.obj_dof)[:, :-1]

        if force is None:
            # compute constraint value
            h = self.friction_constr(u,
                                     contact_normal.reshape(-1, 3),
                                     contact_jac.reshape(-1, 3, 4 * self.num_contacts)).reshape(N, -1)
        else:
            # compute constraint value
            h = self.friction_constr_force(u,
                                           contact_normal.reshape(-1, 3),
                                           contact_jac.reshape(-1, 3, 4 * self.num_contacts)).reshape(N, -1)

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
                      :, :-1, :, self.contact_state_indices][:, :, :, :, self.contact_state_indices]

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
        return None, None, None

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

        g_valve, grad_g_valve, hess_g_valve = self._kinematics_constraints(
            q=q, delta_q=delta_q, theta=theta,
            compute_grads=compute_grads,
            compute_hess=compute_hess,
            projected_diffusion=projected_diffusion)

        g_contact = torch.cat((g_contact,
                               #    g_dynamics,
                               g_valve,
                               ), dim=1)

        if grad_g_contact is not None:
            grad_g_contact = torch.cat((grad_g_contact,
                                        # grad_g_dynamics,
                                        grad_g_valve,
                                        ), dim=1)
            if torch.any(torch.isinf(grad_g_contact)) or torch.any(torch.isnan(grad_g_contact)):
                print('hello')
        if hess_g_contact is not None:
            hess_g_contact = torch.cat((hess_g_contact,
                                        # hess_g_dynamics,
                                        hess_g_valve,
                                        ), dim=1)

        return g_contact, grad_g_contact, hess_g_contact


class AllegroManipulationProblemDiff(AllegroContactProblemDiff, AllegroRegraspProblemDiff):
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

        # super(AllegroManipulationProblem, self).__init__(start=start, goal=goal, T=T, chain=chain,
        #                                                  object_location=object_location, object_type=object_type,
        #                                                  world_trans=world_trans, object_asset_pos=object_asset_pos,
        #                                                  regrasp_fingers=regrasp_fingers,
        #                                                  contact_fingers=contact_fingers,
        #                                                  friction_coefficient=friction_coefficient, obj_dof=obj_dof,
        #                                                  obj_ori_rep=obj_ori_rep, obj_joint_dim=obj_joint_dim,
        #                                                  optimize_force=optimize_force, device=device, **kwargs)
        moveable_object = True if len(contact_fingers) > 0 else False
        AllegroContactProblemDiff.__init__(self, start=start, goal=goal, T=T, chain=chain,
                                       object_location=object_location, object_type=object_type,
                                       world_trans=world_trans, object_asset_pos=object_asset_pos,
                                       regrasp_fingers=regrasp_fingers,
                                       contact_fingers=contact_fingers,
                                       friction_coefficient=friction_coefficient, obj_dof=obj_dof,
                                       obj_ori_rep=obj_ori_rep, obj_joint_dim=obj_joint_dim,
                                       optimize_force=optimize_force, device=device,
                                       **kwargs)

        AllegroRegraspProblemDiff.__init__(self, start=start, goal=goal, T=T, chain=chain,
                                       object_location=object_location, object_type=object_type,
                                       world_trans=world_trans, object_asset_pos=object_asset_pos,
                                       regrasp_fingers=regrasp_fingers,
                                       contact_fingers=contact_fingers,
                                       obj_dof=obj_dof,
                                       obj_ori_rep=obj_ori_rep, obj_joint_dim=obj_joint_dim,
                                       device=device, optimize_force=optimize_force, moveable_object=moveable_object,
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
        return AllegroContactProblemDiff._cost(self, xu, start, goal) + \
            AllegroObjectProblemDiff._cost(self, xu, start, goal) + AllegroRegraspProblemDiff._cost(self, xu, start, goal)

    def _con_eq(self, xu, compute_grads=True, compute_hess=False, verbose=False, projected_diffusion=False):
        N, T = xu.shape[:2]
        g, grad_g, hess_g = None, None, None
        if self.num_regrasps > 0:
            g_regrasp, grad_g_regrasp, hess_g_regrasp = AllegroRegraspProblemDiff._con_eq(self, xu, compute_grads,
                                                                                      compute_hess, projected_diffusion=projected_diffusion)
        if self.num_contacts > 0:
            g_contact, grad_g_contact, hess_g_contact = AllegroContactProblemDiff._con_eq(self, xu, compute_grads,
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
        # h, grad_h, hess_h = AllegroObjectProblemDiff._con_ineq(self, xu, compute_grads, compute_hess)

        if not compute_grads:
            grad_h = None

        if not compute_hess:
            hess_h = None

        if self.num_regrasps > 0:
            h_regrasp, grad_h_regrasp, hess_h_regrasp = AllegroRegraspProblemDiff._con_ineq(self, xu, compute_grads,
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
            h_contact, grad_h_contact, hess_h_contact = AllegroContactProblemDiff._con_ineq(self, xu, compute_grads,
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

        if compute_grads and grad_h is not None:
            grad_h = grad_h.reshape(grad_h.shape[0], grad_h.shape[1], T, -1)[:, :, :, self.all_var_index]
            grad_h = grad_h.reshape(N, -1, T * (self.dx + self.du))
        if compute_hess and hess_h is not None:
            hess_h = hess_h.reshape(hess_h.shape[0], hess_h.shape[1], T, self.d, T, self.d)[:, :, :,
                     self.all_var_index]
            hess_h = hess_h[:, :, :, :, self.all_var_index].reshape(N, -1,
                                                                    T * (self.dx + self.du),
                                                                    T * (self.dx + self.du))

        return h, grad_h, hess_h