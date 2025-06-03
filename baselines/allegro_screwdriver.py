import torch
import numpy as np
from ccai.utils.allegro_utils import get_screwdriver_top_in_world, convert_sine_cosine_to_yaw, convert_yaw_to_sine_cosine, get_model_input_state

from ccai.recovery_rl.recovery_rl.model import QNetworkConstraint


class RunningCostSafeRL:
    def __init__(self, path, cutoff, env, device, include_velocity=False, cosine_sine=True, hardware=False):
        # self.start = start
        self.obj_dof = 3
        self.obj_translational_dim = 0
        self.obj_rotational_dim = 3
        self.include_velocity = include_velocity

        self.cosine_sine = cosine_sine

        self.device = device
        self.setup_safety_critic(path)

        self.env = env
        self.cutoff = float(cutoff)

        self.hardware = hardware
    
    def setup_safety_critic(self, path):
        self.safety_critic = QNetworkConstraint(16, 12, 256)
        self.safety_critic.load_state_dict(torch.load(path))
        self.safety_critic.eval()
        self.safety_critic.to(self.device)

    def query_safety_critic(self, state, action):
        state_sine_cosine = convert_yaw_to_sine_cosine(state).to(self.device)
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
        self.trajectory_sampler = trajectory_sampler
        self.obj_dof = 3
        self.obj_translational_dim = 0
        self.obj_rotational_dim = 3
        self.include_velocity = include_velocity

        self.cosine_sine = cosine_sine

        self.device = device

        self.env = env

    def __call__(self, states, actions):
        N = 8
        model_in_state = get_model_input_state(states[0], self.env, self.obj_dof)

        state = model_in_state[:, -1]
        state_sine_cosine = convert_yaw_to_sine_cosine(state).to(self.device)
        with torch.no_grad():
            _, _, likelihood = self.trajectory_sampler.sample(N*state_sine_cosine.shape[0], H=self.trajectory_sampler.T, start=state_sine_cosine.repeat_interleave(N, 0))
        likelihood = likelihood.reshape(-1, N).mean(1)
        return -likelihood

class ValidityCheck:
    def __init__(self, obj_chain, obj_dof, world_trans, obj_pose):
        self.nominal_screwdriver_top = np.array([0, 0, 1.405])
        self.obj_chain = obj_chain
        self.obj_dof = obj_dof
        self.world_trans = world_trans
        self.obj_pose = obj_pose

    def check_validity(self, state):
        screwdriver_top_pos = get_screwdriver_top_in_world(state[0, -self.obj_dof:], self.obj_chain, self.world_trans, self.obj_pose)
        screwdriver_top_pos = screwdriver_top_pos.detach().cpu().numpy()
        distance2nominal = np.linalg.norm(screwdriver_top_pos - self.nominal_screwdriver_top)
        if distance2nominal > 0.02:
            validity_flag = False
        else:
            validity_flag = True
        return validity_flag


   