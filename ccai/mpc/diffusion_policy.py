import torch
import numpy as np
from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
import hydra

class DummyProblem:
    def __init__(self, dx, T):
        self.dx = dx
        self.T = T
        self.data = None

class Diffusion_Policy:

    def __init__(self, problem, params):
        self.problem = problem
        self.device = params.get('device', 'cuda:0')
        self.N = params.get('N')
        self.sine_cosine = params.get('sine_cosine', False)

        self.warmed_up = False
        self.iter = 0
        self.path = torch.tensor([])

        diffusion_policy_fpath = params.get('diffusion_policy_fpath', None)
        output_dir = params.get('output_dir', None)

        payload = torch.load(open(diffusion_policy_fpath, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # get policy from workspace
        self.policy = workspace.model
        if cfg.training.use_ema:
            self.policy = workspace.ema_model
        self.policy.to(self.device)
        self.policy.eval()

        self.obs_hist = torch.zeros(1, 0, self.problem.dx + 1).to(self.device)

    def step(self, state, **kwargs):

        if self.sine_cosine:
            state = torch.cat([state[..., :-1], torch.cos(state[..., -1:]), torch.sin(state[..., -1:])], dim=-1)
            
        # Add the state to the history
        self.obs_hist = torch.cat([self.obs_hist, state.to(self.device).reshape(1, 1, -1)], dim=1)
        if self.obs_hist.shape[1] == 1:
            # Add again to make the history length 2
            self.obs_hist = torch.cat([self.obs_hist, state.to(self.device).reshape(1, 1, -1)], dim=1)
        obs_dict = {
            'obs': self.obs_hist[:, -2:].to(torch.float32)
        }

        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)

        action = action_dict['action']

        #Swap action[..., :4] and action[..., 4:8]
        return action[:, 0], action.repeat(self.N, 1, 1)

    def shift(self):
        pass

    def reset(self, start, initial_x=None, **kwargs):
        pass
        # self.policy.reset()
        # self.obs_hist = torch.zeros(1, 1, self.problem.dx + 1).to(self.device)
