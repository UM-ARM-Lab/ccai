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
        
# class RunningCostSafeRL:
#     def __init__(self, goal, include_velocity=False):
#         # self.start = start
#         self.goal = goal
#         self.obj_dof = 1
#         self.obj_translational_dim = 0
#         self.obj_rotational_dim = 1
#         self.include_velocity = include_velocity
    
#     def __call__(self, state, action):
#         state = state.to(action.device)
#         N = action.shape[0]
#         if self.include_velocity:
#             state = state.reshape(N, -1 ,2)
#             state = state[:, :, 0] # no the cost is only on the position
                
#         action_cost = torch.sum(action ** 2, dim=-1)
#         goal_cost = 0

#         obj_orientation = state[:, -self.obj_dof:]
#         goal_cost = goal_cost + torch.sum((100 * (obj_orientation - self.goal.unsqueeze(0)) ** 2), dim=-1)

#         return action_cost + goal_cost
    
class RunningCostSafeRL:
    def __init__(self, path, cutoff, env, device, include_velocity=False, cosine_sine=True):
        # self.start = start
        self.obj_dof = 1
        self.obj_translational_dim = 0
        self.obj_rotational_dim = 1
        self.include_velocity = include_velocity

        self.cosine_sine = cosine_sine

        self.device = device
        self.setup_safety_critic(path)

        self.env = env
        self.cutoff = cutoff
    
    def setup_safety_critic(self, path):
        self.safety_critic = QNetworkConstraint(14, 12, 256)
        self.safety_critic.load_state_dict(torch.load(path))
        self.safety_critic.eval()
        self.safety_critic.to(self.device)

    def query_safety_critic(self, state, action):
        state_sine_cosine = convert_yaw_to_sine_cosine(state, yaw_idx=12).to(self.device)
        # safety_critic_input = torch.cat((state_sine_cosine, action), dim=-1)
        # safety_critic_input = safety_critic_input
        with torch.no_grad():
            safety_critic_output = self.safety_critic(state_sine_cosine, action)
        return safety_critic_output
    
    def check_id(self, state, action):
        q = self.query_safety_critic(state, action)
        q1, q2 = q
        q_max = torch.maximum(q1, q2).reshape(state.shape[0])
        id_ = q_max < self.cutoff

        return id_, q_max

    def __call__(self, state, action):
        model_in_state = get_model_input_state(state, self.env, self.obj_dof)
        q1, q2 = self.query_safety_critic(model_in_state, action)
        cost = torch.maximum(q1, q2).reshape(state.shape[0])
        return cost
    
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