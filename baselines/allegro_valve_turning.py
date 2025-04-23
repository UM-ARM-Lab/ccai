import numpy as np
import torch
from ccai.utils.allegro_utils import get_screwdriver_top_in_world, convert_sine_cosine_to_yaw, convert_yaw_to_sine_cosine, get_model_input_state
from ccai.recovery_rl.recovery_rl.model import QNetworkConstraint

import pytorch_kinematics.transforms as tf
# import pytorch3d.transforms as tf

class ValidityCheck:
    def __init__(self, obj_dof):
        self.obj_dof = obj_dof
    def check_validity(self, state):
        return True
        
class RunningCostSafeRL:
    def __init__(self, goal, include_velocity=False):
        # self.start = start
        self.goal = goal
        self.obj_dof = 1
        self.obj_translational_dim = 0
        self.obj_rotational_dim = 1
        self.include_velocity = include_velocity
    
    def __call__(self, state, action):
        state = state.to(action.device)
        N = action.shape[0]
        if self.include_velocity:
            state = state.reshape(N, -1 ,2)
            state = state[:, :, 0] # no the cost is only on the position
                
        action_cost = torch.sum(action ** 2, dim=-1)
        goal_cost = 0

        obj_orientation = state[:, -self.obj_dof:]
        goal_cost = goal_cost + torch.sum((100 * (obj_orientation - self.goal.unsqueeze(0)) ** 2), dim=-1)

        return action_cost + goal_cost
    
class TerminalCostDiffusionLikelihood:
    def __init__(self, trajectory_sampler, env, device, include_velocity=False, cosine_sine=True):
        # self.start = start
        self.obj_dof = 1
        self.obj_translational_dim = 0
        self.obj_rotational_dim = 1
        self.include_velocity = include_velocity
        
        self.cosine_sine = cosine_sine

        self.device = device

        self.env = env
        
        self.trajectory_sampler = trajectory_sampler
    
    def __call__(self, states, action):
        N = 8
        model_in_state = get_model_input_state(states[0], self.env, self.obj_dof)

        state = model_in_state[:, -1]
        state_sine_cosine = convert_yaw_to_sine_cosine(state, yaw_idx=12).to(self.device)
        with torch.no_grad():
            _, _, likelihood = self.trajectory_sampler.sample(N*state_sine_cosine.shape[0], H=self.trajectory_sampler.T, start=state_sine_cosine.repeat_interleave(N, 0))
        likelihood = likelihood.reshape(-1, N).mean(1)
        return -likelihood