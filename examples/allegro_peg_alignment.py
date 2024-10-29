from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroPegInsertionEnv

import numpy as np
import pickle as pkl

import torch
import time
import copy
import yaml
import pathlib
from functools import partial

import time
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
from torch.func import vmap, jacrev, hessian, jacfwd
# import pytorch3d.transforms as tf

import matplotlib.pyplot as plt
from utils.allegro_utils import *
from allegro_valve_roll import AllegroValveTurning, AllegroContactProblem, PositionControlConstrainedSVGDMPC, add_trajectories, add_trajectories_hardware
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp



CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

device = 'cuda:0'
obj_dof = 6
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

class AllegroPegInsertion(AllegroValveTurning):
    def get_constraint_dim(self, T):
        self.friction_polytope_k = 4
        wrench_dim = 0
        if self.obj_translational_dim > 0:
            wrench_dim += 3
        if self.obj_rotational_dim > 0:
            wrench_dim += 3
        if self.optimize_force:
            self.dg_per_t = self.num_fingers * (1 + 2 + 4) + wrench_dim + 1
        else:
            self.dg_per_t = self.num_fingers * (1 + 2) + wrench_dim + 1
        self.dg_constant = 0
        self.dg = self.dg_per_t * T + self.dg_constant  # terminal contact points, terminal sdf=0, and dynamics
        self.dz = (self.friction_polytope_k) * self.num_fingers # one friction constraints per finger
        # self.dz = 0 # DEBUG ONLY
        self.dh = self.dz * T  # inequality
    def __init__(self,
                 start,
                 goal,
                 T,
                 chain,
                 object_location,
                 object_type,
                 world_trans,
                 peg_asset_pos,
                 wall_asset_pos,
                 wall_dims,
                 fingers=['index', 'middle', 'ring', 'thumb'],
                 friction_coefficient=0.95,
                 optimize_force=False,
                 obj_dof_code=[1, 1, 1, 1, 1, 1],
                 obj_gravity=False,
                 device='cuda:0', **kwargs):
        self.num_fingers = len(fingers)
        self.optimize_force = optimize_force
        self.peg_asset_pos = peg_asset_pos
        self.wall_asset_pos = wall_asset_pos
        self.wall_dims = wall_dims.astype('float32')
        self.obj_mass = 0.3

        super(AllegroPegInsertion, self).__init__(start=start, goal=goal, T=T, chain=chain, object_location=object_location,
                                                 object_type=object_type, world_trans=world_trans, object_asset_pos=peg_asset_pos,
                                                 fingers=fingers, friction_coefficient=friction_coefficient, obj_dof_code=obj_dof_code, 
                                                 obj_joint_dim=0, optimize_force=optimize_force, obj_gravity=obj_gravity, device=device)
        self.friction_coefficient = friction_coefficient
    
    def _cost(self, xu, start, goal):
        # TODO: consider using quaternion difference for the orientation.
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it
        
        action = xu[:, self.dx:self.dx + 4 * self.num_fingers]  # action dim = 8
        next_q = state[:-1, :-self.obj_dof] + action
        if self.optimize_force:
            action_cost = 0
        else:
            action_cost = torch.sum((state[1:, :-self.obj_dof] - next_q) ** 2)

        smoothness_cost = 1 * torch.sum((state[1:] - state[:-1]) ** 2)

        goal_cost = 0
        if self.obj_translational_dim:
            obj_position = state[:, -self.obj_dof:-self.obj_dof+self.obj_translational_dim]
            # terminal cost
            goal_cost = goal_cost + torch.sum((100 * (obj_position[-1] - goal[:self.obj_translational_dim]) ** 2))
            # running cost
            goal_cost = goal_cost + torch.sum((1 * (obj_position - goal[:self.obj_translational_dim]) ** 2))
            smoothness_cost = smoothness_cost + 100 * torch.sum((obj_position[1:] - obj_position[:-1]) ** 2)
        if self.obj_rotational_dim:
            obj_orientation = state[:, -self.obj_dof+self.obj_translational_dim:]
            obj_orientation = tf.euler_angles_to_matrix(obj_orientation, convention='XYZ')
            obj_orientation = tf.matrix_to_rotation_6d(obj_orientation)
            goal_orientation = tf.euler_angles_to_matrix(goal[-self.obj_rotational_dim:], convention='XYZ')
            goal_orientation = tf.matrix_to_rotation_6d(goal_orientation)
            # terminal cost
            goal_cost = goal_cost + torch.sum((20 * (obj_orientation[-1] - goal_orientation) ** 2))
            # running cost 
            goal_cost = goal_cost + torch.sum((10 * (obj_orientation - goal_orientation) ** 2))
            smoothness_cost = smoothness_cost + 50 * torch.sum((obj_orientation[1:] - obj_orientation[:-1]) ** 2)
        # goal_cost = torch.sum((1000 * (state[-1, -self.obj_dof:] - goal) ** 2)).reshape(-1)
        # goal_cost += torch.sum((10 * (state[:, -self.obj_dof:] - goal.unsqueeze(0)) ** 2))


        # if self.optimize_force:
        #     force = xu[:, self.dx + 4 * self.num_fingers: self.dx + (4 + 3) * self.num_fingers]
        #     force = force.reshape(force.shape[0], self.num_fingers, 3)
        #     force_norm = torch.norm(force, dim=-1)
        #     force_norm = force_norm - 0.3 # desired maginitute
        #     force_cost = 10 * torch.sum(force_norm ** 2)
        #     action_cost += force_cost
        return smoothness_cost + action_cost + goal_cost 
    def _init_contact_scenes(self, asset_object, collision_checking):
        # robot and peg
        peg_chain = pk.build_chain_from_urdf(open(asset_object).read())
        peg_chain = peg_chain.to(device=self.device)
        peg_sdf = pv.RobotSDF(peg_chain, path_prefix=None, use_collision_geometry=True) # since we are using primitive shapes for the object, there's no need to define path for stl
        robot_sdf = pv.RobotSDF(self.chain, path_prefix=get_assets_dir() + '/xela_models', use_collision_geometry=True)

        robot2peg = self.world_trans.inverse().compose(
            pk.Transform3d(device=self.device).translate(self.peg_asset_pos[0], self.peg_asset_pos[1], self.peg_asset_pos[2]))

        # contact checking
        collision_check_links = [self.ee_names[finger] for finger in self.fingers]
        self.robot_peg_scenes = pv.RobotScene(robot_sdf, peg_sdf, robot2peg,
                                            collision_check_links=collision_check_links,
                                            softmin_temp=1.0e3,
                                            points_per_link=1000,
                                            partial_patch=False,
                                            )
        # peg and wall
        wall_sdf = pv.BoxSDF([self.wall_dims[0], self.wall_dims[1], self.wall_dims[2]], device=self.device)
        world2peg = tf.Transform3d(pos=torch.tensor(self.peg_asset_pos, device=self.device).float(),
                                          rot=torch.tensor(
                                              [1, 0, 0, 0],
                                              device=self.device).float(), device=self.device)
        peg2wall = world2peg.inverse().compose(
            pk.Transform3d(device=self.device).translate(self.wall_asset_pos[0], self.wall_asset_pos[1], self.wall_asset_pos[2]))
        self.peg_wall_scenes = pv.RobotScene(peg_sdf, wall_sdf, peg2wall,
                                            collision_check_links=['peg'],
                                            softmin_temp=1.0e3,
                                            points_per_link=2000,
                                            partial_patch=False,
                                            )
        peg_wall_transforms = torch.tensor([[[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0.1],
                                            [0, 0, 0, 1]]], device=self.device).float()
        peg2wall = torch.tensor(self.wall_asset_pos - self.peg_asset_pos).float().to(self.device)
        peg_wall_transforms[:, :3, 3] = peg2wall
        identity = torch.eye(4).unsqueeze(0).to(self.device).float()
        peg_wall_transforms = torch.cat((identity, peg_wall_transforms), dim=0)
        peg_wall_transforms = pk.Transform3d(matrix=peg_wall_transforms)
        # Compose SDFs
        composed_sdf = pv.ComposedSDF([peg_sdf, wall_sdf], peg_wall_transforms)
        # self.viz_scenes = pv.RobotScene(robot_sdf, composed_sdf, robot2peg, 
        #                             collision_check_links=collision_check_links, 
        #                             softmin_temp=1.0e3,
        #                             points_per_link=1000,
        #                             partial_patch=False)
        # self.robot_peg_scenes.visualize_robot(partial_to_full_state(self.start[:4*self.num_fingers], fingers=self.fingers), self.start[4*self.num_fingers:].to(self.device))
        # self.viz_scenes.visualize_robot(partial_to_full_state(self.start[:4*self.num_fingers], fingers=self.fingers), self.start[4*self.num_fingers:].to(self.device))
        # self.peg_table_scenes.visualize_robot(torch.zeros(6).float().to(self.device))                
        # self.peg_wall_scenes.visualize_robot(self.start[4*self.num_fingers:].to(self.device))        
    def _preprocess(self, xu):
        N = xu.shape[0]
        xu = xu.reshape(N, self.T, -1)
        x = xu[:, :, :self.dx]
        # expand to include start
        x_expanded = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)

        q = x_expanded[:, :, :4 * self.num_fingers]
        if self.fixed_obj:
            theta = self.start_obj_pose.unsqueeze(0).repeat((N, self.T + 1, 1))
        else:
            theta = x_expanded[:, :, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof]
        # theta = x_expanded[:, :, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof]
        self._preprocess_fingers(q, theta)

    def _preprocess_fingers(self, q, theta):
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
        ret_scene = self.robot_peg_scenes.scene_collision_check(full_q, theta_b,
                                                              compute_gradient=True,
                                                              compute_hessian=False)
        for i, finger in enumerate(self.fingers):
            self.data[finger] = {}
            self.data[finger]['sdf'] = ret_scene['sdf'][:, i].reshape(N, self.T + 1)
            # reshape and throw away data for unused fingers
            grad_g_q = ret_scene.get('grad_sdf', None)
            self.data[finger]['grad_sdf'] = grad_g_q[:, i].reshape(N, self.T + 1, 16)[:, :, self.all_joint_index]

            # contact jacobian
            contact_jacobian = ret_scene.get('contact_jacobian', None)
            self.data[finger]['contact_jacobian'] = contact_jacobian[:, i].reshape(N, self.T + 1, 3, 16)[:, :, :, self.all_joint_index]

            # contact hessian
            contact_hessian = ret_scene.get('contact_hessian', None)
            contact_hessian = contact_hessian[:, i].reshape(N, self.T + 1, 3, 16, 16)[:, :, :, self.all_joint_index]
            contact_hessian = contact_hessian[:, :, :, :, self.all_joint_index]  # [:, :, :, self.all_joint_index]
            # contact_hessian = contact_hessian[:, :, :, :, self.all_joint_index]  # shape (N, T+1, 3, 8, 8)

            # gradient of contact point
            d_contact_loc_dq = ret_scene.get('closest_pt_q_grad', None)
            d_contact_loc_dq = d_contact_loc_dq[:, i].reshape(N, self.T + 1, 3, 16)[:, :, :, self.all_joint_index]  # [:, :, :, self.all_joint_index]
            self.data[finger]['closest_pt_q_grad'] = d_contact_loc_dq
            self.data[finger]['contact_hessian'] = contact_hessian
            self.data[finger]['closest_pt_world'] = ret_scene['closest_pt_world'][:, i]
            self.data[finger]['contact_normal'] = ret_scene['contact_normal'][:, i]

            # gradient of contact normal
            self.data[finger]['dnormal_dq'] = ret_scene['dnormal_dq'][:, i].reshape(N, self.T + 1, 3, 16)[:, :, :, self.all_joint_index]  # [:, :, :,
            # self.all_joint_index]

            self.data[finger]['dnormal_denv_q'] = ret_scene['dnormal_denv_q'][:, i, :, :self.obj_dof]
            self.data[finger]['grad_env_sdf'] = ret_scene['grad_env_sdf'][:, i, :self.obj_dof]
            dJ_dq = contact_hessian
            self.data[finger]['dJ_dq'] = dJ_dq  # Jacobian of the contact point

        ret_scene_peg_wall = self.peg_wall_scenes.scene_collision_check(theta_b, None,
                                                                compute_gradient=True,
                                                                compute_hessian=False)
        self.data['sdf_peg_wall'] = ret_scene_peg_wall['sdf'].reshape(N, self.T + 1)
        self.data['grad_sdf_peg_wall'] = ret_scene_peg_wall['grad_sdf'].reshape(N, self.T + 1, 6)

    def get_initial_xu(self, N):
        # TODO: fix the initialization, for 6D movement, the angle is not supposed to be the linear interpolation of the euler angle. 
        """
        use delta joint movement to get the initial trajectory
        the action (force at the finger tip) is not used. it is randomly intiailized
        the actual dynamics model is not used
        """

        # u = 0.025 * torch.randn(N, self.T, self.du, device=self.device)
        if self.optimize_force:
            u = 0.025 * torch.randn(N, self.T, 4 * self.num_fingers, device=self.device)
            force = 0.025 * torch.randn(N, self.T, 3 * self.num_fingers, device=self.device)
            u = torch.cat((u, force), dim=-1)
        else:
            u = 0.025 * torch.randn(N, self.T, self.du, device=self.device)

        x = [self.start.reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            next_q = x[-1][:, :4 * self.num_fingers] + u[:, t, :4 * self.num_fingers]
            x.append(next_q)

        x = torch.stack(x[1:], dim=1)

        # if valve angle in state
        current_obj_position = self.start[4 * self.num_fingers: 4 * self.num_fingers + self.obj_translational_dim]
        current_obj_orientation = self.start[4 * self.num_fingers + self.obj_translational_dim:4 * self.num_fingers + self.obj_dof]
        current_obj_R = R.from_euler('XYZ', current_obj_orientation.cpu().numpy())
        goal_obj_R = R.from_euler('XYZ', self.goal[self.obj_translational_dim:self.obj_translational_dim + self.obj_rotational_dim].cpu().numpy())
        key_times = [0, self.T]
        times = np.linspace(0, self.T, self.T + 1)
        slerp = Slerp(key_times, R.concatenate([current_obj_R, goal_obj_R]))
        interp_rots = slerp(times)
        interp_rots = interp_rots.as_euler('XYZ')[1:]
        # current_obj_orientation = tf.euler_angles_to_matrix(current_obj_orientation, convention='XYZ')
        # current_obj_orientation = tf.matrix_to_rotation_6d(current_obj_orientation)

        theta_position = np.linspace(current_obj_position.cpu().numpy(), self.goal[:self.obj_translational_dim].cpu().numpy(), self.T + 1)[1:]
        theta = np.concatenate((theta_position, interp_rots), axis=-1)
        theta = torch.tensor(theta, device=self.device, dtype=torch.float32)
        theta = theta.unsqueeze(0).repeat((N,1,1))

        # DEBUG ONLY, use initial state as the initialization
        # theta = self.start[-self.obj_dof:].unsqueeze(0).repeat((N, self.T, 1))
        # theta = torch.ones((N, self.T, self.obj_dof)).to(self.device) * self.start[-self.obj_dof:]
        x = torch.cat((x, theta), dim=-1)

        xu = torch.cat((x, u), dim=2)
        return xu

    def _peg_wall_contact_constraint(self, theta, compute_grads=True, compute_hess=False):
        N, T, _ = theta.shape
        # Retrieve pre-processed data
        g = self.data.get('sdf_peg_wall').reshape(N, T + 1, 1)  # - 0.0025
        grad_g_theta = self.data.get('grad_sdf_peg_wall', None)

        # Ignore first value, as it is the start state
        g = g[:, 1:].reshape(N, -1)

        if compute_grads:
            T_range = torch.arange(T, device=theta.device)
            # compute gradient of sdf
            grad_g = torch.zeros(N, T, T, self.dx + self.du, device=theta.device)
            grad_g[:, T_range, T_range, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof] = grad_g_theta.reshape(N, T + 1, self.obj_dof)[:, 1:]
            grad_g = grad_g.reshape(N, -1, T, self.dx + self.du)
            grad_g = grad_g.reshape(N, -1, T * (self.dx + self.du))
        else:
            return g, None, None

        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * (self.dx + self.du), T * (self.dx + self.du), device=self.device)
            return g, grad_g, hess

        return g, grad_g, None

    def _cost(self, xu, start, goal):
        # TODO: consider using quaternion difference for the orientation.
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it
        
        action = xu[:, self.dx:self.dx + 4 * self.num_fingers]  # action dim = 8
        next_q = state[:-1, :-self.obj_dof] + action
        if self.optimize_force:
            action_cost = 0
        else:
            action_cost = torch.sum((state[1:, :-self.obj_dof] - next_q) ** 2)

        smoothness_cost = 1 * torch.sum((state[1:] - state[:-1]) ** 2)
        # smoothness_cost += 10 * torch.sum((state[1:, -self.obj_dof:] - state[:-1, -self.obj_dof:]) ** 2)
        # smoothness_cost += 5000 * torch.sum((state[1:, -self.obj_dof:-self.obj_dof+3] - state[:-1, -self.obj_dof:-self.obj_dof+3]) ** 2) # the position should stay smooth
        # upright_cost = 1000 * torch.sum((state[:, -self.obj_dof:-self.obj_dof+3]) ** 2) # the screwdriver should only rotate in z direction

        # goal_cost = torch.sum((10 * (state[-1, -self.obj_dof:] - goal) ** 2)).reshape(-1)
        # goal_cost += torch.sum((0.1 * (state[:, -self.obj_dof:] - goal.unsqueeze(0)) ** 2))

        goal_cost = 0
        if self.obj_translational_dim:
            obj_position = state[:, -self.obj_dof:-self.obj_dof+self.obj_translational_dim]
            # terminal cost
            goal_cost = goal_cost + torch.sum((100 * (obj_position[-1] - goal[:self.obj_translational_dim]) ** 2))
            # running cost
            goal_cost = goal_cost + torch.sum((1 * (obj_position - goal[:self.obj_translational_dim]) ** 2))
            smoothness_cost = smoothness_cost + 100 * torch.sum((obj_position[1:] - obj_position[:-1]) ** 2)
        if self.obj_rotational_dim:
            obj_orientation = state[:, -self.obj_dof+self.obj_translational_dim:]
            obj_orientation = tf.euler_angles_to_matrix(obj_orientation, convention='XYZ')
            obj_orientation = tf.matrix_to_rotation_6d(obj_orientation)
            goal_orientation = tf.euler_angles_to_matrix(goal[-self.obj_rotational_dim:], convention='XYZ')
            goal_orientation = tf.matrix_to_rotation_6d(goal_orientation)
            # terminal cost
            goal_cost = goal_cost + torch.sum((50 * (obj_orientation[-1] - goal_orientation) ** 2))
            # running cost 
            goal_cost = goal_cost + torch.sum((3 * (obj_orientation - goal_orientation) ** 2))
            smoothness_cost = smoothness_cost + 50 * torch.sum((obj_orientation[1:] - obj_orientation[:-1]) ** 2)
        # goal_cost = torch.sum((1000 * (state[-1, -self.obj_dof:] - goal) ** 2)).reshape(-1)
        # goal_cost += torch.sum((10 * (state[:, -self.obj_dof:] - goal.unsqueeze(0)) ** 2))


        # if self.optimize_force:
        #     force = xu[:, self.dx + 4 * self.num_fingers: self.dx + (4 + 3) * self.num_fingers]
        #     force = force.reshape(force.shape[0], self.num_fingers, 3)
        #     force_norm = torch.norm(force, dim=-1)
        #     force_norm = force_norm - 0.3 # desired maginitute
        #     force_cost = 10 * torch.sum(force_norm ** 2)
        #     action_cost += force_cost
        return smoothness_cost + action_cost + goal_cost 

    def _con_eq(self, xu, compute_grads=True, compute_hess=False, verbose=False):
        N = xu.shape[0]
        T = xu.shape[1]
        g_contact, grad_g_contact, hess_g_contact = self._contact_constraints(xu=xu.reshape(N, T, self.dx + self.du),
                                                                              compute_grads=compute_grads,
                                                                              compute_hess=compute_hess)
        g_peg_contact, grad_g_peg_contact, hess_g_peg_contact \
            = self._peg_wall_contact_constraint(xu[:, :, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof],
                                                                                                  compute_grads=compute_grads,
                                                                                                  compute_hess=compute_hess)
        if self.optimize_force:
            g_equil, grad_g_equil, hess_g_equil = self._force_equlibrium_constraints_w_force(
                xu=xu.reshape(N, T, self.dx + self.du),
                compute_grads=compute_grads,
                compute_hess=compute_hess)
        else:
            g_equil, grad_g_equil, hess_g_equil = self._force_equlibrium_constraints(
                xu=xu.reshape(N, T, self.dx + self.du),
                compute_grads=compute_grads,
                compute_hess=compute_hess)

        g_valve, grad_g_valve, hess_g_valve = self._kinematics_constraints(
            xu=xu.reshape(N, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)
        

        if verbose:
            print(f"max contact constraint: {torch.max(torch.abs(g_contact))}")
            print(f"max peg wall contact constraint: {torch.max(torch.abs(g_peg_contact))}")
            print(f"max valve kinematics constraint: {torch.max(torch.abs(g_valve))}")
            print(f"max force equilibrium constraint: {torch.max(torch.abs(g_equil))}")
            result_dict = {}
            result_dict['contact'] = torch.max(torch.abs(g_contact)).item()
            result_dict['peg_wall_contact'] = torch.max(torch.abs(g_peg_contact)).item()
            result_dict['kinematics'] = torch.max(torch.abs(g_valve)).item()
            result_dict['force'] = torch.max(torch.abs(g_equil)).item()
            result_dict['contact_mean'] = torch.mean(torch.abs(g_contact)).item()
            result_dict['peg_wall_contact_mean'] = torch.mean(torch.abs(g_peg_contact)).item()
            result_dict['kinematics_mean'] = torch.mean(torch.abs(g_valve)).item()
            result_dict['force_mean'] = torch.mean(torch.abs(g_equil)).item()

            return result_dict
        g_contact = torch.cat((
                                g_contact, 
                                g_peg_contact,
                                g_equil,
                                g_valve,
                               ), dim=1)

        if grad_g_contact is not None:
            grad_g_contact = torch.cat((
                                        grad_g_contact, 
                                        grad_g_peg_contact,
                                        grad_g_equil,
                                        grad_g_valve,
                                        ), dim=1)
        if hess_g_contact is not None:
            hess_g_contact = torch.cat((
                                        hess_g_contact, 
                                        hess_g_peg_contact,
                                        hess_g_equil,
                                        hess_g_valve,
                                        ), dim=1)

        return g_contact, grad_g_contact, hess_g_contact  
    
def do_trial(env, params, fpath, sim_viz_env=None, ros_copy_node=None):
    # step multiple times untile it's stable
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None
    for i in range(1):
        if len(params['fingers']) == 3:
            action = torch.cat((env.default_dof_pos[:,:8], env.default_dof_pos[:, 12:16]), dim=-1)
        elif len(params['fingers']) == 4:
            action = env.default_dof_pos[:, :16]
        state = env.step(action)

    num_fingers = len(params['fingers'])
    state = env.get_state()
    action_list = []

    start = state['q'].reshape(4 * num_fingers + obj_dof).to(device=params['device'])

    if params['controller'] == 'csvgd':
        # index finger is used for stability
        if 'index' in params['fingers']:
            contact_fingers = params['fingers']
        else:
            contact_fingers = ['index'] + params['fingers']        
        pregrasp_problem = AllegroContactProblem(
            dx=4 * num_fingers,
            du=4 * num_fingers,
            start=start[:4 * num_fingers + obj_dof],
            goal=None,
            T=4,
            chain=params['chain'],
            device=params['device'],
            object_asset_pos=env.peg_pose,
            object_type='peg',
            world_trans=env.world_trans,
            fingers=contact_fingers,
            obj_dof_code=params['obj_dof_code'],
            obj_joint_dim=0,
            fixed_obj=True
        )

        pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, params)
        pregrasp_planner.warmup_iters = 50
    else:
        raise ValueError('Invalid controller')
    
    
    start = env.get_state()['q'].reshape(4 * num_fingers + obj_dof).to(device=params['device'])
    best_traj, _ = pregrasp_planner.step(start[:4 * num_fingers])

    if params['visualize_plan']:
        traj_for_viz = best_traj[:, :pregrasp_problem.dx]
        tmp = start[4 * num_fingers:].unsqueeze(0).repeat(traj_for_viz.shape[0], 1)
        traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)    
        viz_fpath = pathlib.PurePath.joinpath(fpath, "pregrasp")
        img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
        gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
        pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
        pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
        visualize_trajectory(traj_for_viz, pregrasp_problem.contact_scenes, viz_fpath, pregrasp_problem.fingers, pregrasp_problem.obj_dof+1)

    for x in best_traj[:, :4 * num_fingers]:
        action = x.reshape(-1, 4 * num_fingers).to(device=env.device) # move the rest fingers
        if params['mode'] == 'hardware':
            sim_viz_env.set_pose(env.get_state()['all_state'].to(device=env.device))
            sim_viz_env.step(action)
        env.step(action)
        action_list.append(action)
        if params['mode'] == 'hardware_copy':
            ros_copy_node.apply_action(partial_to_full_state(x.reshape(-1, 4 * num_fingers)[0], params['fingers']))

    desired_table_pose = torch.tensor([0, 0, -1.0, 0, 0, 0, 1]).float().to(env.device)
    env.set_table_pose(env.handles['table'][0], desired_table_pose)

    state = env.get_state()
    state = env.step(state['q'][:, :4 * num_fingers])
    start = state['q'].reshape(4 * num_fingers + obj_dof).to(device=params['device'])
    turn_problem_fingers = params['fingers']
    turn_problem_start = start[:4 * num_fingers + obj_dof]
    turn_problem = AllegroPegInsertion(
        start=turn_problem_start,
        goal=params['valve_goal'],
        T=params['T'],
        chain=params['chain'],
        device=params['device'],
        peg_asset_pos=env.peg_pose,
        wall_asset_pos=env.wall_pose,
        wall_dims = env.wall_dims,
        object_location=params['object_location'],
        object_type=params['object_type'],
        friction_coefficient=params['friction_coefficient'],
        world_trans=env.world_trans,
        fingers=turn_problem_fingers,
        optimize_force=params['optimize_force'],
        obj_gravity=params['obj_gravity'],
    )
    turn_planner = PositionControlConstrainedSVGDMPC(turn_problem, params)

    actual_trajectory = []
    duration = 0

    fig = plt.figure()
    axes = {params['fingers'][i]: fig.add_subplot(int(f'1{num_fingers}{i+1}'), projection='3d') for i in range(num_fingers)}
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

    num_fingers_to_plan = num_fingers
    info_list = []

    for k in range(params['num_steps']):
        state = env.get_state()
        start = state['q'].reshape(4 * num_fingers + obj_dof).to(device=params['device'])

        actual_trajectory.append(state['q'][:, :4 * num_fingers + obj_dof].squeeze(0).clone())
        start_time = time.time()

        best_traj, trajectories = turn_planner.step(start[:4 * num_fingers + obj_dof])

        print(f"solve time: {time.time() - start_time}")
        planned_theta_traj = best_traj[:, 4 * num_fingers_to_plan: 4 * num_fingers_to_plan + obj_dof].detach().cpu().numpy()
        print(f"current theta: {state['q'][0, -obj_dof:].detach().cpu().numpy()}")
        print(f"planned theta: {planned_theta_traj}")
        # add trajectory lines to sim
        if k < params['num_steps'] - 1:
            if params['mode'] == 'hardware':
                add_trajectories_hardware(trajectories, best_traj, axes, env, config=params, state2ee_pos_func=state2ee_pos)
            else:
                add_trajectories(trajectories, best_traj, axes, env, sim=sim, gym=gym, viewer=viewer,
                                config=params, state2ee_pos_func=state2ee_pos)

        if params['visualize_plan']:
            traj_for_viz = best_traj[:, :turn_problem.dx]
            traj_for_viz = torch.cat((start[:turn_problem.dx].unsqueeze(0), traj_for_viz), dim=0)
            tmp = torch.zeros((traj_for_viz.shape[0], 1), device=best_traj.device) # add the joint for the screwdriver cap
            traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)
            # traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof] = axis_angle_to_euler(traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof])
        
            viz_fpath = pathlib.PurePath.joinpath(fpath, f"timestep_{k}")
            img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
            gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
            pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
            pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
            visualize_trajectory(traj_for_viz, turn_problem.robot_peg_scenes, viz_fpath, turn_problem.fingers, turn_problem.obj_dof)
            
            img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img_obj')
            gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif_obj')
            pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
            pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
            visualize_obj_trajectory(traj_for_viz[:, 4 * num_fingers: 4 * num_fingers + obj_dof], turn_problem.peg_wall_scenes, viz_fpath)
        
        x = best_traj[0, :turn_problem.dx+turn_problem.du]
        x = x.reshape(1, turn_problem.dx+turn_problem.du)
        turn_problem._preprocess(best_traj.unsqueeze(0))
        equality_constr_dict = turn_problem._con_eq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        inequality_constr_dict = turn_problem._con_ineq(best_traj.unsqueeze(0), compute_grads=False, compute_hess=False, verbose=True)
        print("--------------------------------------")

        action = x[:, turn_problem.dx:turn_problem.dx+turn_problem.du].to(device=env.device)
        if params['optimize_force']:
            print("planned force")
            print(action[:, 4 * num_fingers_to_plan:].reshape(num_fingers_to_plan, 3)) # print out the action for debugging
        # print(action)
        action = action[:, :4 * num_fingers_to_plan]
        action = action + start.unsqueeze(0)[:, :4 * num_fingers] # NOTE: this is required since we define action as delta action
        if params['mode'] == 'hardware':
            sim_viz_env.set_pose(env.get_state()['all_state'].to(device=env.device))
            sim_viz_env.step(action)
        elif params['mode'] == 'hardware_copy':
            ros_copy_node.apply_action(partial_to_full_state(action[0], params['fingers']))
        # action = x[:, :4 * num_fingers].to(device=env.device)
        # NOTE: DEBUG ONLY
        # action = best_traj[1, :4 * turn_problem.num_fingers].unsqueeze(0)
        env.step(action)
        action_list.append(action)
        # if params['hardware']:
        #     # ros_node.apply_action(action[0].detach().cpu().numpy())
        #     ros_node.apply_action(partial_to_full_state(action[0]).detach().cpu().numpy())
        turn_problem._preprocess(best_traj.unsqueeze(0))
        
        # print(turn_problem.thumb_contact_scene.scene_collision_check(partial_to_full_state(x[:, :8]), x[:, 8],
        #                                                         compute_gradient=False, compute_hessian=False))
        # distance2surface = torch.sqrt((best_traj_ee[:, 2] - object_location[2].unsqueeze(0)) ** 2 + (best_traj_ee[:, 0] - object_location[0].unsqueeze(0))**2)
        distance2goal = (params['valve_goal'].cpu() - env.get_state()['q'][:, -obj_dof:].cpu()).detach().cpu()
        print(distance2goal)
        info = {**equality_constr_dict, **inequality_constr_dict, **{'distance2goal': distance2goal}}
        info_list.append(info)

        gym.clear_lines(viewer)
        state = env.get_state()
        start = state['q'][:,:4 * num_fingers + obj_dof].squeeze(0).to(device=params['device'])
        for finger in params['fingers']:
            ee = state2ee_pos(start[:4 * num_fingers], turn_problem.ee_names[finger])
            finger_traj_history[finger].append(ee.detach().cpu().numpy())
        for finger in params['fingers']:
            traj_history = finger_traj_history[finger]
            temp_for_plot = np.stack(traj_history, axis=0)
            if k >= 2:
                axes[finger].plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'gray', label='actual')
    with open(f'{fpath.resolve()}/info.pkl', 'wb') as f:
        pkl.dump(info_list, f)
    action_list = torch.concat(action_list, dim=0)
    with open(f'{fpath.resolve()}/action.pkl', 'wb') as f:
        pkl.dump(action_list, f)
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
      if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)
    fig.tight_layout()
    fig.legend(newHandles, newLabels, loc='lower center', ncol=3)
    plt.savefig(f'{fpath.resolve()}/traj.png')
    plt.close()
    # plt.show()



    env.reset()
    state = env.get_state()
    state = state['q'].reshape(4 * num_fingers + obj_dof).to(device=params['device'])
    actual_trajectory.append(state.clone()[: 4 * num_fingers + obj_dof])
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 4 * num_fingers + obj_dof)
    turn_problem.T = actual_trajectory.shape[0]
    # constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    final_distance_to_goal = (actual_trajectory[:, -obj_dof:] - params['valve_goal']).abs()

    print(f'Controller: {params["controller"]} Final distance to goal: {torch.min(final_distance_to_goal)}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
            #  constr=constraint_val.cpu().numpy(),
             d2goal=final_distance_to_goal.cpu().numpy())
    return torch.min(final_distance_to_goal).cpu().numpy()

if __name__ == "__main__":
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_peg_insertion.yaml').read_text())
    from tqdm import tqdm

    if config['mode'] == 'hardware':
        raise ValueError('Hardware mode is not supported for this task')
    else:
        env = AllegroPegInsertionEnv(1, control_mode='joint_impedance',
                                    use_cartesian_controller=False,
                                    viewer=True,
                                    steps_per_action=60,
                                    friction_coefficient=1.0,
                                    device=config['sim_device'],
                                    video_save_path=img_save_dir,
                                    joint_stiffness=config['kp'],
                                    fingers=config['fingers'],
                                    )

    sim, gym, viewer = env.get_sim()


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
    # if config['mode'] == 'hardware':
    #     sim_env = env
    #     from hardware.hardware_env import HardwareEnv
    #     env = HardwareEnv(sim_env.default_dof_pos[:, :16], finger_list=['index', 'thumb'], kp=config['kp'])
    #     env.world_trans = sim_env.world_trans
    #     env.joint_stiffness = sim_env.joint_stiffness
    #     env.device = sim_env.device
    #     env.valve_pose = sim_env.valve_pose
    # elif config['mode'] == 'hardware_copy':
    #     from hardware.hardware_env import RosNode
    #     ros_copy_node = RosNode()



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
    config['obj_dof_code'] = [1, 1, 1, 1, 1, 1]
    config['obj_dof'] = np.sum(config['obj_dof_code'])

    # screwdriver_asset = f'{get_assets_dir()}/screwdriver/screwdriver_6d.urdf'

    chain = pk.build_chain_from_urdf(open(asset).read())
    # screwdriver_chain = pk.build_chain_from_urdf(open(screwdriver_asset).read())
    frame_indices = [chain.frame_to_idx[ee_names[finger]] for finger in config['fingers']]    # combined chain
    frame_indices = torch.tensor(frame_indices)
    state2ee_pos = partial(state2ee_pos, fingers=config['fingers'], chain=chain, frame_indices=frame_indices, world_trans=env.world_trans)
    
    forward_kinematics = partial(chain.forward_kinematics, frame_indices=frame_indices) # full_to= _partial_state = partial(full_to_partial_state, fingers=config['fingers'])
    # partial_to_full_state = partial(partial_to_full_state, fingers=config['fingers'])


    for i in tqdm(range(config['num_trials'])):
        goal = torch.tensor([0, 0, 0, 0, 0, 0])
        for controller in config['controllers'].keys():
            env.reset()
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
            config_file_path = pathlib.PurePath.joinpath(fpath, 'config.yaml')
            with open(config_file_path, 'w') as f:
                yaml.dump(config, f)
            # set up params
            params = config.copy()
            params.pop('controllers')
            params.update(config['controllers'][controller])
            params['controller'] = controller
            params['valve_goal'] = goal.to(device=params['device'])
            params['chain'] = chain.to(device=params['device'])
            object_location = torch.tensor(env.peg_pose).to(device).float() # TODO: confirm if this is the correct location
            params['object_location'] = object_location
            final_distance_to_goal = do_trial(env, params, fpath, sim_env, ros_copy_node)
            # final_distance_to_goal = turn(env, params, fpath)

            if controller not in results.keys():
                results[controller] = [final_distance_to_goal]
            else:
                results[controller].append(final_distance_to_goal)
        print(results)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

