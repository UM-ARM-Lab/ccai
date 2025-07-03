import numpy as np
import torch

from allegro_optimized_wrapper import transforms as tf
# import pytorch3d.transforms as tf

class ValidityCheck:
    def __init__(self, obj_dof):
        self.obj_dof = obj_dof
    def check_validity(self, state):
        obj_state = state[0, -self.obj_dof:]
        obj_z = obj_state[2]
        if obj_z < -0.1:
            return False
        else:
            return True
        
class RunningCost:
    def __init__(self, goal, include_velocity=False):
        # self.start = start
        self.goal = goal
        self.obj_dof = 6
        self.obj_translational_dim = 3
        self.obj_rotational_dim = 3
        self.include_velocity = include_velocity
    
    def __call__(self, state, action):
        state = state.to(action.device)
        N = action.shape[0]
        if self.include_velocity:
            state = state.reshape(N, -1 ,2)
            state = state[:, :, 0] # no the cost is only on the position
                
        action_cost = torch.sum(action ** 2, dim=-1)

        goal_cost = 0
        obj_position = state[:, -self.obj_dof:-self.obj_dof+self.obj_translational_dim]
        # terminal cost
        goal_cost = goal_cost + torch.sum((100 * (obj_position - self.goal[:self.obj_translational_dim].unsqueeze(0)) ** 2), dim=-1)

        obj_orientation = state[:, -self.obj_dof+self.obj_translational_dim:]
        obj_orientation = tf.euler_angles_to_matrix(obj_orientation, convention='XYZ')
        obj_orientation = tf.matrix_to_rotation_6d(obj_orientation)
        goal_orientation = tf.euler_angles_to_matrix(self.goal[-self.obj_rotational_dim:], convention='XYZ')
        goal_orientation = tf.matrix_to_rotation_6d(goal_orientation)
        # terminal cost
        goal_cost = goal_cost + torch.sum((100 * (obj_orientation - goal_orientation.unsqueeze(0)) ** 2), dim=-1)
        # dropping cost
        dropping_cost = 1e6 * ((obj_position[:, 2] < -0.02) * obj_position[:, 2])** 2

        return action_cost + goal_cost + dropping_cost