import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
# import cv2

from recovery_rl.utils import soft_update, hard_update
from recovery_rl.model import QNetworkConstraint


# Process observation for CNN
def process_obs(obs):
    im = np.transpose(obs, (2, 0, 1))
    return im


'''
Wrapper for training, querying, and visualizing Q_risk for Recovery RL

Source: https://github.com/abalakrishna123/recovery-rl/tree/master
'''


class QRiskWrapper:
    def __init__(self, obs_space, ac_space, hidden_size, logdir,
                 args):
        self.logdir = logdir
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.ac_space = ac_space

        self.safety_critic = QNetworkConstraint(
            obs_space.shape[0], ac_space.shape[0],
            hidden_size).to(device=self.device)
        self.safety_critic_target = QNetworkConstraint(
            obs_space.shape[0], ac_space.shape[0],
            args.hidden_size).to(device=self.device)

        self.lr = 1e-4
        self.safety_critic_optim = Adam(self.safety_critic.parameters(),
                                        lr=args.lr)
        hard_update(self.safety_critic_target, self.safety_critic)

        self.tau = args.tau_safe
        self.gamma_safe = args.gamma_safe
        self.updates = 0
        self.target_update_interval = args.target_update_interval
        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)

        self.pos_fraction = args.pos_fraction if args.pos_fraction >= 0 else None

    def update_parameters(self,
                          memory=None,
                          policy=None,
                          batch_size=None,
                          ):
        '''
        Trains safety critic Q_risk and model-free recovery policy which performs
        gradient ascent on the safety critic

        Arguments:
            memory: Agent's replay buffer
            policy: Agent's composite policy
            critic: Safety critic (Q_risk)
        '''
        if self.pos_fraction:
            batch_size = min(batch_size,
                             int((1 - self.pos_fraction) * len(memory)))
        else:
            batch_size = min(batch_size, len(memory))
        state_batch, action_batch, constraint_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size, pos_fraction=self.pos_fraction)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        constraint_batch = torch.FloatTensor(constraint_batch).to(
            self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = policy.sample(
                next_state_batch)

            qf1_next_target, qf2_next_target = self.safety_critic_target(
                next_state_batch, next_state_action)
            min_qf_next_target = torch.max(qf1_next_target, qf2_next_target)
            next_q_value = constraint_batch + mask_batch * self.gamma_safe * (
                min_qf_next_target)

        qf1, qf2 = self.safety_critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        self.safety_critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.safety_critic_optim.step()

        if self.updates % self.target_update_interval == 0:
            soft_update(self.safety_critic_target, self.safety_critic,
                        self.tau)
        self.updates += 1

    def __call__(self, states, actions):
        return self.safety_critic(states, actions)