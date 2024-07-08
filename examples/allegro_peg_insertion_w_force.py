"this formulation considers contact forces from the wall to the peg"
from isaacgym import gymapi
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
from utils.allegro_utils import partial_to_full_state, full_to_partial_state, all_finger_constraints, state2ee_pos, visualize_trajectory, visualize_obj_trajectory
from allegro_valve_roll import AllegroValveTurning, AllegroContactProblem, PositionControlConstrainedSVGDMPC, add_trajectories_hardware
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp



CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

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
        self.dz = (self.friction_polytope_k) * (self.num_fingers + 1) # one friction constraints per finger
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
        self.peg_trans = tf.Transform3d(pos=torch.tensor(self.peg_asset_pos, device=device).float(),
                                          rot=torch.tensor(
                                        [1, 0, 0, 0],
                                        device=device).float(), device=device)
        self.wall_asset_pos = wall_asset_pos
        self.wall_dims = wall_dims.astype('float32')
        du = (4 + 3) * self.num_fingers + 3 
        self.obj_mass = 0.03

        super(AllegroPegInsertion, self).__init__(start=start, goal=goal, T=T, chain=chain, object_location=object_location,
                                                 object_type=object_type, world_trans=world_trans, object_asset_pos=peg_asset_pos,
                                                 fingers=fingers, friction_coefficient=friction_coefficient, obj_dof_code=obj_dof_code, 
                                                 obj_joint_dim=0, optimize_force=optimize_force, du=du, obj_gravity=obj_gravity, device=device)
        self.friction_coefficient = friction_coefficient
        self.force_equlibrium_constr = vmap(self._force_equlibrium_constr_w_force)
        self.grad_force_equlibrium_constr = vmap(jacrev(self._force_equlibrium_constr_w_force, argnums=(0, 1, 2, 3, 4, 5, 6)))

        self.wall_friction_constr = vmap(self._wall_friction_constr, randomness='same')
        self.grad_wall_friction_constr = vmap(jacrev(self._wall_friction_constr, argnums=(0, 1)))

        # append the additional env force 
        max_f = torch.ones(3) * 10
        min_f = torch.ones(3) * -10
        self.x_max = torch.cat((self.x_max, max_f))
        self.x_min = torch.cat((self.x_min, min_f))

    
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
        # peg_wall_transforms = torch.tensor([[[1, 0, 0, 0],
        #                                     [0, 1, 0, 0],
        #                                     [0, 0, 1, 0.1],
        #                                     [0, 0, 0, 1]]], device=self.device).float()
        # peg2wall = torch.tensor(self.wall_asset_pos - self.peg_asset_pos).float().to(self.device)
        # peg_wall_transforms[:, :3, 3] = peg2wall
        # identity = torch.eye(4).unsqueeze(0).to(self.device).float()
        # peg_wall_transforms = torch.cat((identity, peg_wall_transforms), dim=0)
        # peg_wall_transforms = pk.Transform3d(matrix=peg_wall_transforms)
        # Compose SDFs
        # composed_sdf = pv.ComposedSDF([peg_sdf, wall_sdf], peg_wall_transforms)
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
        self.data['peg'] = {}
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
        self.data['peg']['sdf_peg_wall'] = ret_scene_peg_wall['sdf'].reshape(N, self.T + 1)
        self.data['peg']['grad_sdf_peg_wall'] = ret_scene_peg_wall['grad_sdf'].reshape(N, self.T + 1, 6)
        d_contact_loc_dq = ret_scene_peg_wall.get('closest_pt_q_grad', None)
        d_contact_loc_dq = d_contact_loc_dq.reshape(N, self.T + 1, 3, 6)
        self.data['peg']['closest_pt_q_grad'] = d_contact_loc_dq
        # self.data['peg']['contact_hessian'] = contact_hessian
        self.data['peg']['closest_pt_world'] = ret_scene_peg_wall['closest_pt_world']
        self.data['peg']['contact_normal'] = ret_scene_peg_wall['contact_normal']

        # gradient of contact normal
        self.data['peg']['dnormal_dq'] = ret_scene_peg_wall['dnormal_dq'].reshape(N, self.T + 1, 3, 6)


    def get_initial_xu(self, N):
        # TODO: fix the initialization, for 6D movement, the angle is not supposed to be the linear interpolation of the euler angle. 
        """
        use delta joint movement to get the initial trajectory
        the action (force at the finger tip) is not used. it is randomly intiailized
        the actual dynamics model is not used
        """

        # u = 0.025 * torch.randn(N, self.T, self.du, device=self.device)
        u = 0.025 * torch.randn(N, self.T, 4 * self.num_fingers, device=self.device)
        force = 0.025 * torch.randn(N, self.T, 3 * (self.num_fingers + 1), device=self.device)
        u = torch.cat((u, force), dim=-1)

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
        g = self.data['peg'].get('sdf_peg_wall').reshape(N, T + 1, 1)  # - 0.0025
        grad_g_theta = self.data['peg'].get('grad_sdf_peg_wall', None)

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

    def _con_ineq(self, xu, compute_grads=True, compute_hess=False, verbose=False):
        N = xu.shape[0]
        T = xu.shape[1]
        h, grad_h, hess_h = self._friction_constraint(
            xu=xu.reshape(-1, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)
        
        h_wall, grad_h_wall, hess_h_wall = self._wall_friction_constraint(
            xu=xu.reshape(-1, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)
        
        
        if verbose:
            print(f"max friction constraint: {torch.max(h)}")
            print(f"max wall friction constraint: {torch.max(h_wall)}")
            result_dict = {}
            result_dict['friction'] = torch.max(h).item()
            result_dict['friction_mean'] = torch.mean(h).item()
            result_dict['wall_friction'] = torch.max(h_wall).item()
            result_dict['wall_friction_mean'] = torch.mean(h_wall).item()
            return result_dict

        h = torch.cat((h,
                        h_wall,
                        ), dim=1)
        if compute_grads:
            grad_h = grad_h.reshape(N, -1, self.T * (self.dx + self.du))
            grad_h = torch.cat((grad_h, 
                                grad_h_wall), dim=1)
        else:
            return h, None, None
        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], self.T * (self.dx + self.du), self.T * (self.dx + self.du),
                                device=self.device)
            return h, grad_h, hess_h
        return h, grad_h, None    
    def _force_equlibrium_constr_w_force(self, q, u, next_q, force_list, contact_jac_list, contact_point_list, next_env_q):
        # NOTE: the constriant is defined in the robot frame
        # the contact jac an contact points are all in the robot frame
        # this will be vmapped, so takes in a 3 vector and a [num_finger x 3 x 8] jacobian and a dq vector
        obj_robot_frame = self.world_trans.inverse().transform_points(self.object_location.reshape(1, 3))
        delta_q = q + u - next_q
        torque_list = []
        reactional_torque_list = []
        for i, finger_name in enumerate(self.fingers):
            # TODO: Assume that all the fingers are having an equlibrium, maybe we should change so that index finger is not considered
            contact_jacobian = contact_jac_list[i]
            force_robot_frame = self.world_trans.inverse().transform_normals(force_list[i].unsqueeze(0)).squeeze(0)
            reactional_torque_list.append(contact_jacobian.T @ -force_robot_frame)
            # pseudo inverse form
            contact_point_r_valve = contact_point_list[i] - obj_robot_frame[0]
            torque = torch.linalg.cross(contact_point_r_valve, force_robot_frame)
            torque_list.append(torque)
            # It does not matter for comuputing the force equilibrium constraint
        env_force = force_list[-1]
        peg_wall_contact_point_peg_frame = contact_point_list[-1]
        peg_wall_contact_point_robot_frame = self.world_trans.inverse().transform_points(self.peg_trans.transform_points(contact_point_list[-1].reshape(1,3)))
        peg_wall_r = peg_wall_contact_point_robot_frame[0] - obj_robot_frame[0]
        env_force_robot_frame = force_robot_frame = self.world_trans.inverse().transform_normals(env_force.unsqueeze(0)).squeeze(0)
        peg_wall_torque = torch.linalg.cross(peg_wall_r, env_force_robot_frame)
        torque_list.append(peg_wall_torque)
        # contact_point_r_valve = contact_point_list[-1] - obj_robot_frame[0]
        if self.obj_gravity:
            if self.obj_translational_dim > 0:
                g = self.obj_mass * torch.tensor([0, 0, -9.8], device=self.device, dtype=torch.float32)
                force_list = torch.cat((force_list, g.unsqueeze(0)))

        # force_world_frame = self.world_trans.transform_normals(force.unsqueeze(0)).squeeze(0)
        torque_list = torch.stack(torque_list, dim=0)
        sum_torque_list = torch.sum(torque_list, dim=0)
        sum_force_list = torch.sum(force_list, dim=0)

        g = []
        if self.obj_translational_dim > 0:
            g.append(sum_force_list)
        elif self.screwdriver_force_balance:
            # balance the force with the torque
            g.append(torch.sum(force_list, dim=0)[:2])
        if self.obj_rotational_dim > 0:
            g.append(sum_torque_list)
        g = torch.cat(g)
        reactional_torque_list = torch.stack(reactional_torque_list, dim=0)
        sum_reactional_torque = torch.sum(reactional_torque_list, dim=0)
        g_force_torque_balance = (sum_reactional_torque + 3.0 * delta_q)
        # print(g_force_torque_balance.max(), torque_list.max())
        g = torch.cat((g, g_force_torque_balance.reshape(-1)), dim=-1)
        # residual_list = torch.stack(residual_list, dim=0) * 100
        # g = torch.cat((torque_list, residual_list), dim=-1)
        return g

    def _force_equlibrium_constraints_w_force(self, xu, compute_grads=True, compute_hess=False):
        N, T, d = xu.shape
        x = xu[:, :, :self.dx]

        # we want to add the start state to x, this x is now T + 1
        x = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)
        q = x[:, :-1, :self.num_fingers * 4]
        next_q = x[:, 1:, :self.num_fingers * 4]
        next_env_q = x[:, 1:, self.num_fingers * 4: self.num_fingers * 4 + self.obj_dof]
        u = xu[:, :, self.dx: self.dx + 4 * self.num_fingers]
        force = xu[:, :, self.dx + 4 * self.num_fingers:]
        force_list = force.reshape((force.shape[0], force.shape[1], self.num_fingers + 1, 3))
        # contact_jac_list = [self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, 4 * self.num_fingers)[:, :-1].reshape(-1, 3, 4 * self.num_fingers)\
        #                      for finger_name in self.fingers]
        contact_jac_list = [self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, 4 * self.num_fingers)[:, 1:].reshape(-1, 3, 4 * self.num_fingers)\
                             for finger_name in self.fingers]
        contact_jac_list = torch.stack(contact_jac_list, dim=1).to(device=xu.device)
        contact_point_list = [self.data[finger_name]['closest_pt_world'].reshape(N, T + 1, 3)[:, :-1].reshape(-1, 3) for finger_name in self.fingers]
        contact_point_list.append(self.data['peg']['closest_pt_world'].reshape(N, T + 1, 3)[:, :-1].reshape(-1, 3))
        contact_point_list = torch.stack(contact_point_list, dim=1).to(device=xu.device)

        g = self.force_equlibrium_constr(q.reshape(-1, 4 * self.num_fingers), 
                                         u.reshape(-1, 4 * self.num_fingers), 
                                         next_q.reshape(-1, 4 * self.num_fingers), 
                                         force_list.reshape(-1, self.num_fingers + 1, 3),
                                         contact_jac_list,
                                         contact_point_list,
                                         next_env_q.reshape(-1, self.obj_dof)).reshape(N, T, -1)
        # print(g.abs().max().detach().cpu().item(), g.abs().mean().detach().cpu().item())
        if compute_grads:
            dg_dq, dg_du, dg_dnext_q, dg_dforce, dg_djac, dg_dcontact, dg_dnext_env_q = self.grad_force_equlibrium_constr(q.reshape(-1, 4 * self.num_fingers), 
                                                                                  u.reshape(-1, 4 * self.num_fingers), 
                                                                                  next_q.reshape(-1, 4 * self.num_fingers), 
                                                                                  force_list.reshape(-1, self.num_fingers + 1, 3),
                                                                                  contact_jac_list,
                                                                                  contact_point_list,
                                                                                  next_env_q.reshape(-1, self.obj_dof))
            dg_dforce = dg_dforce.reshape(dg_dforce.shape[0], dg_dforce.shape[1], (self.num_fingers + 1) * 3)
            
            T_range = torch.arange(T, device=x.device)
            T_plus = torch.arange(1, T, device=x.device)
            T_minus = torch.arange(T - 1, device=x.device)
            grad_g = torch.zeros(N, g.shape[2], T, T, self.dx + self.du, device=self.device)
            # dnormal_dq = torch.zeros(N, T, 3, 8, device=self.device)  # assume zero SDF hessian
            dg_dq = dg_dq.reshape(N, T, g.shape[2], 4 * self.num_fingers) 
            dg_dnext_q = dg_dnext_q.reshape(N, T, g.shape[2], 4 * self.num_fingers) 
            for i, finger_name in enumerate(self.fingers):
                # NOTE: assume fingers have joints independent of each other
                # djac_dq = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, 4 * self.num_fingers, 4 * self.num_fingers)[:, :-1] # jacobian is the contact jacobian
                # dg_dq = dg_dq + dg_djac[:, :, i].reshape(N, T, g.shape[2], -1) @ djac_dq.reshape(N, T, -1, 4 * self.num_fingers)
                djac_dnext_q = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, 4 * self.num_fingers, 4 * self.num_fingers)[:, 1:]
                dg_dnext_q = dg_dnext_q + dg_djac[:, :, i].reshape(N, T, g.shape[2], -1) @ djac_dnext_q.reshape(N, T, -1, 4 * self.num_fingers)
                
                d_contact_loc_dq = self.data[finger_name]['closest_pt_q_grad'].reshape(N, T + 1, 3, 4 * self.num_fingers)[:, :-1]
                dg_dq = dg_dq + dg_dcontact[:, : ,i].reshape(N, T, g.shape[2], 3) @ d_contact_loc_dq 
            grad_g[:, :, T_plus, T_minus, :4 * self.num_fingers] = dg_dq.reshape(N, T, g.shape[2], 4 * self.num_fingers)[:, 1:].transpose(1, 2)  # first q is the start
            dg_du = torch.cat((dg_du, dg_dforce), dim=-1)  # check the dim
            grad_g[:, :, T_range, T_range, self.dx:] = dg_du.reshape(N, T, -1, self.du).transpose(1, 2)
            grad_g[:, :, T_range, T_range, :4 * self.num_fingers] = dg_dnext_q.reshape(N, T, -1, 4 * self.num_fingers).transpose(1, 2)
            if self.obj_gravity:
                grad_g[:, :, T_range, T_range, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof] = dg_dnext_env_q.reshape(N, T, -1, self.obj_dof).transpose(1, 2)
            grad_g = grad_g.transpose(1, 2)
        else:
            return g.reshape(N, -1), None, None
        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * d, T * d, device=self.device)
            return g.reshape(N, -1), grad_g.reshape(N, -1, T * (self.dx + self.du)), hess
        else:
            return g.reshape(N, -1), grad_g.reshape(N, -1, T * (self.dx + self.du)), None

    def _wall_friction_constr(self, force, contact_normal):

        # compute the force in robot frame
        # transform contact normal to world frame
        contact_normal_world = self.peg_trans.transform_normals(contact_normal.unsqueeze(0)).squeeze(0)

        # transform force to contact frame
        R = self.get_rotation_from_normal(contact_normal_world.unsqueeze(0)).squeeze(0)
        # force_contact_frame = R.transpose(0, 1) @ force_world_frame.unsqueeze(-1)
        B = self.get_friction_polytope().detach()

        # here dq means the force in the world frame
        contact_v_contact_frame = R.transpose(0, 1) @ force
        return B @ contact_v_contact_frame

    def _wall_friction_constraint(self, xu, compute_grads=True, compute_hess=False):

        # assume access to class member variables which have already done some of the computation
        N, T, d = xu.shape
        u = xu[:, :, self.dx:]

        u = u[:, :, (4 + 3) * self.num_fingers: ].reshape(-1, 3)
        # retrieved cached values
        contact_normal = - self.data['peg']['contact_normal'].reshape(N, T + 1, 3)[:, :-1] # contact normal is pointing out 
        dnormal_dtheta = - self.data['peg']['dnormal_dq'].reshape(N, T + 1, 3, self.obj_dof)[:, :-1]

        # compute constraint value
        h = self.wall_friction_constr(u,
                                 contact_normal.reshape(-1, 3)).reshape(N, -1)

        # compute the gradient
        if compute_grads:
            dh_du, dh_dnormal = self.grad_wall_friction_constr(u,
                                                            contact_normal.reshape(-1, 3))

            dh = dh_dnormal.shape[1]
            dh_dtheta = dh_dnormal.reshape(N, T, dh, -1) @ dnormal_dtheta
            grad_h = torch.zeros(N, dh, T, T, d, device=self.device)
            T_range = torch.arange(T, device=self.device)
            T_range_minus = torch.arange(T - 1, device=self.device)
            T_range_plus = torch.arange(1, T, device=self.device)
            grad_h[:, :, T_range_plus, T_range_minus, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof] = dh_dtheta[:, 1:].transpose(1, 2)
            grad_h[:, :, T_range, T_range, self.dx + 7 * self.num_fingers:] = dh_du.reshape(N, T, dh, 3).transpose(1, 2)
            grad_h = grad_h.transpose(1, 2).reshape(N, -1, T * d)
        else:
            return h, None, None

        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], T * d, T * d, device=self.device)
            return h, grad_h, hess_h

        return h, grad_h, None

def add_trajectories(trajectories, best_traj, axes, env, sim, gym, viewer, config, state2ee_pos_func):
    wall_coor = torch.tensor([-0.03, 0, 0.25]).to(trajectories.device)

    M = len(trajectories)
    T = len(best_traj)
    fingers = copy.copy(config['fingers'])
    if 'exclude_index' in config.keys() and config['exclude_index']:
        fingers.remove('index')
    num_fingers = len(fingers)
    obj_dof = config['obj_dof']
    if M > 0:
        initial_state = env.get_state()['q']
        # num_fingers = initial_state.shape[1] // 4
        initial_state = initial_state[:, :4 * num_fingers]
        force = best_traj[:, (4 + 4) * num_fingers + obj_dof: -3].reshape(T, num_fingers, 3)
        env_force = best_traj[:, -3: ].reshape(T, 1, 3)

        all_state = torch.cat((initial_state, best_traj[:-1, :4 * num_fingers]), dim=0)
        desired_state = all_state + best_traj[:, 4 * num_fingers + obj_dof: 4 * num_fingers + obj_dof + 4 * num_fingers]
        
        desired_best_traj_ee = [state2ee_pos_func(desired_state, config['ee_names'][finger], fingers=fingers) for finger in fingers]
        best_traj_ee = [state2ee_pos_func(best_traj[:, :4 * num_fingers], config['ee_names'][finger], fingers=fingers) for finger in fingers]

        state_colors = np.array([0, 0, 1]).astype(np.float32)
        desired_state_colors = np.array([0, 1, 1]).astype(np.float32)
        force_colors = np.array([1, 1, 0]).astype(np.float32)
        
        for e in env.envs:
            T = best_traj.shape[0]
            for t in range(T):
                for i, finger in enumerate(fingers):
                    if t == 0:
                        initial_ee = state2ee_pos_func(initial_state, config['ee_names'][finger], fingers=fingers)
                        state_traj = torch.stack((initial_ee, best_traj_ee[i][0]), dim=0).cpu().numpy()
                        action_traj = torch.stack((initial_ee, desired_best_traj_ee[i][0]), dim=0).cpu().numpy()
                        axes[finger].plot3D(state_traj[:, 0], state_traj[:, 1], state_traj[:, 2], 'blue', label='desired next state')
                        axes[finger].plot3D(action_traj[:, 0], action_traj[:, 1], action_traj[:, 2], 'green', label='raw commanded position')
                        force_traj = torch.stack((initial_ee, initial_ee + force[t, i]), dim=0).cpu().numpy()
                        axes[finger].plot3D(force_traj[:, 0], force_traj[:, 1], force_traj[:, 2], 'red', label='force')
                    else:
                        state_traj = torch.stack((best_traj_ee[i][t - 1, :3], best_traj_ee[i][t, :3]), dim=0).cpu().numpy()
                        action_traj = torch.stack((best_traj_ee[i][t - 1, :3], desired_best_traj_ee[i][t, :3]), dim=0).cpu().numpy()
                        force_traj = torch.stack((best_traj_ee[i][t - 1, :3], best_traj_ee[i][t - 1, :3] + force[t, i]), dim=0).cpu().numpy()
                    state_traj = state_traj.reshape(2, 3)
                    action_traj = action_traj.reshape(2, 3)
                
                    gym.add_lines(viewer, e, 1, state_traj, state_colors)
                    gym.add_lines(viewer, e, 1, action_traj, desired_state_colors)
                    gym.add_lines(viewer, e, 1, force_traj, force_colors)  
                env_force_traj = torch.stack((wall_coor, wall_coor + env_force[t, 0]), dim=0).cpu().numpy()
                gym.add_lines(viewer, e, 1, env_force_traj, force_colors)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)

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
        obj_gravity = params['obj_gravity'],
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
    contact_list = [] # list of checking whether the peg is in contact with the wall
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
        if k <= params['num_steps'] - 1:
            gym.draw_env_rigid_contacts(viewer, env.envs[0], gymapi.Vec3(0,1,0), 10000, 1)
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
            print(action[:, 4 * num_fingers_to_plan:].reshape(num_fingers_to_plan + 1, 3)) # print out the action for debugging
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

        # process contact
        contacts = gym.get_env_rigid_contacts(env.envs[0])
        for body0, body1 in zip (contacts['body0'], contacts['body1']):
            if body0 == 27 and body1 == 29:
                print("contact with wall")
                contact_list.append(True)
                break
            elif body0 == 29 and body1 == 27:
                print("contact with wall")
                contact_list.append(True)
                break
        
        # print(turn_problem.thumb_contact_scene.scene_collision_check(partial_to_full_state(x[:, :8]), x[:, 8],
        #                                                         compute_gradient=False, compute_hessian=False))
        # distance2surface = torch.sqrt((best_traj_ee[:, 2] - object_location[2].unsqueeze(0)) ** 2 + (best_traj_ee[:, 0] - object_location[0].unsqueeze(0))**2)
        distance2goal = (params['valve_goal'].cpu() - env.get_state()['q'][:, -obj_dof:].cpu()).detach().cpu()
        print(distance2goal)
        info = {**equality_constr_dict, **inequality_constr_dict, 'distance2goal': distance2goal, 'contact': contact_list}
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
    print(contact_list)
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
    desired_table_pose = torch.tensor([0, 0, 0.105, 0, 0, 0, 1]).float().to(env.device)
    env.set_table_pose(env.handles['table'][0], desired_table_pose)
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
            object_location = torch.tensor(env.peg_pose).to(params['device']).float() # TODO: confirm if this is the correct location
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

