from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.allegro import AllegroPegAlignmentEnv

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
from allegro_valve_roll import AllegroValveTurning
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp



CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

device = 'cuda:0'
obj_dof = 6
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

class AllegroReorientation(AllegroValveTurning):
    def get_constraint_dim(self, T):
        self.friction_polytope_k = 4
        wrench_dim = 0
        if self.obj_translational_dim > 0:
            wrench_dim += 3
        if self.obj_rotational_dim > 0:
            wrench_dim += 3
        if self.optimize_force:
            self.dg_per_t = self.num_fingers * (1 + 2 + 4) + wrench_dim
        else:
            self.dg_per_t = self.num_fingers * (1 + 2) + wrench_dim
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
                 object_asset_pos,
                 fingers=['index', 'middle', 'ring', 'thumb'],
                 friction_coefficient=0.95,
                 optimize_force=False,
                 obj_dof_code=[1, 1, 1, 1, 1, 1],
                 obj_gravity=False,
                 device='cuda:0', **kwargs):
        self.num_fingers = len(fingers)
        self.optimize_force = optimize_force
        self.object_asset_pos = object_asset_pos
        self.obj_mass = 0.022

        super(AllegroReorientation, self).__init__(start=start, goal=goal, T=T, chain=chain, object_location=object_location,
                                                 object_type=object_type, world_trans=world_trans, object_asset_pos=object_asset_pos,
                                                 fingers=fingers, friction_coefficient=friction_coefficient, obj_dof_code=obj_dof_code, 
                                                 obj_joint_dim=0, optimize_force=optimize_force, obj_gravity=obj_gravity, device=device)
        self.friction_coefficient = friction_coefficient
    
    def _cost(self, xu, start, goal):
        # TODO: consider using quaternion difference for the orientation.
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it
        
        action = xu[:, self.dx:self.dx + 4 * self.num_fingers]  # action dim = 8
        next_q = state[:-1, :-self.obj_dof] + action
        action_cost = 0

        smoothness_cost = 100 * torch.sum((state[1:] - state[:-1]) ** 2)
        # smoothness_cost += 10 * torch.sum((state[1] - state[0]) ** 2) # penalize the 1st step smoothness

        obj_orientation = state[:, -self.obj_dof+self.obj_translational_dim:]
        peg_upright_cost = torch.sum((obj_orientation[:, 0] ** 2)) + torch.sum((obj_orientation[:, 1] ** 2))
        smoothness_cost += 1000 * peg_upright_cost
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
            goal_cost = goal_cost + torch.sum((100 * (obj_orientation[-1] - goal_orientation) ** 2))
            # running cost 
            goal_cost = goal_cost + torch.sum((0.1 * (obj_orientation - goal_orientation) ** 2))
            smoothness_cost = smoothness_cost + 50 * torch.sum((obj_orientation[1:] - obj_orientation[:-1]) ** 2)
        return smoothness_cost + action_cost + goal_cost 

    def get_initial_xu(self, N):
        # TODO: fix the initialization, for 6D movement, the angle is not supposed to be the linear interpolation of the euler angle. 
        """
        use delta joint movement to get the initial trajectory
        the action (force at the finger tip) is not used. it is randomly intiailized
        the actual dynamics model is not used
        """

        # u = 0.025 * torch.randn(N, self.T, self.du, device=self.device)
        u = 0.025 * torch.randn(N, self.T, 4 * self.num_fingers, device=self.device)
        force = 0.025 * torch.randn(N, self.T, 3 * self.num_fingers, device=self.device)
        # force[:, :, 3:] *= 10 # the catching fingers use a larger force to maintain stability
        force[:, :, 0] -= 0.2
        # force[:, :, 3] -= 0.2
        force[:, :, 6] += 0.2
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

        theta_position = np.linspace(current_obj_position.cpu().numpy(), self.goal[:self.obj_translational_dim].cpu().numpy(), self.T + 1)[1:]
        theta = np.concatenate((theta_position, interp_rots), axis=-1)
        theta = torch.tensor(theta, device=self.device, dtype=torch.float32)
        theta = theta.unsqueeze(0).repeat((N,1,1))

        theta = self.start[-self.obj_dof:].reshape(1, 1, -1).repeat(N, self.T, 1)

        x = torch.cat((x, theta), dim=-1)

        xu = torch.cat((x, u), dim=2)
        return xu
    def check_validity(self, state):
        obj_state = state[-self.obj_dof:]
        obj_z = obj_state[2]
        if obj_z < -0.1:
            return False
        else:
            return True


 