import numpy as np
import torch

import pytorch_kinematics.transforms as tf
# import pytorch3d.transforms as tf

class ValidityCheck:
    def __init__(self, obj_dof):
        self.obj_dof = obj_dof
    def check_validity(self, state):
        return True
        
class RunningCost:
    def __init__(self, start, goal, include_velocity=False):
        self.start = start
        self.goal = goal
        self.obj_dof = 1
        self.obj_translational_dim = 0
        self.obj_rotational_dim = 1
        self.include_velocity = include_velocity
    
    def __call__(self, state, action):
        N = action.shape[0]
        if self.include_velocity:
            state = state.reshape(N, -1 ,2)
            state = state[:, :, 0] # no the cost is only on the position
                
        action_cost = torch.sum(action ** 2, dim=-1)

        obj_orientation = state[:, -self.obj_dof:]
        goal_cost = goal_cost + torch.sum((100 * (obj_orientation - self.goal.unsqueeze(0)) ** 2), dim=-1)

        return action_cost + goal_cost