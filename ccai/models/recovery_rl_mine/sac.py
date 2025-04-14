'''
Built on on SAC implementation from 
https://github.com/pranz24/pytorch-soft-actor-critic
'''

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
# from dotmap import DotMap
# import cv2

from recovery_rl.utils import soft_update, hard_update
from recovery_rl.model import GaussianPolicy, QNetwork, DeterministicPolicy, QNetworkCNN, \
    GaussianPolicyCNN, QNetworkConstraint, QNetworkConstraintCNN, DeterministicPolicyCNN
from recovery_rl.qrisk import QRiskWrapper


class SAC(object):
    def __init__(self,
                 observation_space,
                 action_space,
                 config,
                 logdir,
                 im_shape=None,
                 tmp_env=None):

        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)
        # Parameters for SAC
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.alpha = config['alpha']
        self.env_name = config['env_name']
        self.logdir = logdir
        self.policy_type = config['policy']
        self.target_update_interval = config['target_update_interval']
        self.automatic_entropy_tuning = config['automatic_entropy_tuning']
        self.device = torch.device("cuda" if config['cuda'] else "cpu")
        self.updates = 0
        self.cnn = config['cnn']
        if im_shape:
            observation_space = im_shape

        # RRL specific parameters
        self.gamma_safe = config['gamma_safe']
        self.eps_safe = config['eps_safe']
        # Parameters for comparisons
        self.DGD_constraints = config['DGD_constraints']
        self.nu = config['nu']
        self.update_nu = config['update_nu']

        self.use_constraint_sampling = config['use_constraint_sampling']
        self.log_nu = torch.tensor(np.log(self.nu),
                                   requires_grad=True,
                                   device=self.device)
        self.nu_optim = Adam([self.log_nu], lr=0.1 * config['lr'])

        # SAC setup
        if config['cnn']:
            self.critic = QNetworkCNN(observation_space, action_space.shape[0],
                                      config['hidden_size'],
                                      config['env_name']).to(device=self.device)
            self.critic_target = QNetworkCNN(
                observation_space, action_space.shape[0], config['hidden_size'],
                config['env_name']).to(device=self.device)
        else:
            self.critic = QNetwork(observation_space.shape[0],
                                   action_space.shape[0],
                                   config['hidden_size']).to(device=self.device)
            self.critic_target = QNetwork(
                observation_space.shape[0], action_space.shape[0],
                config['hidden_size']).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=config['lr'])

        hard_update(self.critic_target, self.critic)
        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A)
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(
                    torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1,
                                             requires_grad=True,
                                             device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=config['lr'])

            if config['cnn']:
                self.policy = GaussianPolicyCNN(observation_space,
                                                action_space.shape[0],
                                                config['hidden_size'],
                                                config['env_name'],
                                                action_space).to(self.device)
            else:
                self.policy = GaussianPolicy(observation_space.shape[0],
                                             action_space.shape[0],
                                             config['hidden_size'],
                                             action_space).to(self.device)

            self.policy_optim = Adam(self.policy.parameters(), lr=config['lr'])

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            assert not config['cnn']
            self.policy = DeterministicPolicy(observation_space.shape[0],
                                              action_space.shape[0],
                                              config['hidden_size'],
                                              action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=config['lr'])

        # Initialize safety critic
        self.safety_critic = QRiskWrapper(observation_space,
                                          action_space,
                                          config['hidden_size'],
                                          logdir,
                                          config,
                                          tmp_env=tmp_env)

    def select_action(self, state, eval=False):
        '''
            Get action from current task policy
        '''
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # Action Sampling for SQRL
        if self.use_constraint_sampling:
            self.safe_samples = 100  # TODO: don't hardcode
            if not self.cnn:
                state_batch = state.repeat(self.safe_samples, 1)
            else:
                state_batch = state.repeat(self.safe_samples, 1, 1, 1)
            pi, log_pi, _ = self.policy.sample(state_batch)
            max_qf_constraint_pi = self.safety_critic.get_value(
                state_batch, pi)

            thresh_idxs = (max_qf_constraint_pi <= self.eps_safe).nonzero()[:,
                                                                            0]
            # Note: these are auto-normalized
            thresh_probs = torch.exp(log_pi[thresh_idxs])
            thresh_probs = thresh_probs.flatten()

            if list(thresh_probs.size())[0] == 0:
                min_q_value_idx = torch.argmin(max_qf_constraint_pi)
                action = pi[min_q_value_idx, :].unsqueeze(0)
            else:
                prob_dist = torch.distributions.Categorical(thresh_probs)
                sampled_idx = prob_dist.sample()
                action = pi[sampled_idx, :].unsqueeze(0)
        # Action Sampling for all other algorithms
        else:
            if eval is False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self,
                          memory,
                          batch_size,
                          updates,
                          nu=None,
                          safety_critic=None):
        '''
        Train task policy and associated Q function with experience in replay buffer (memory)
        '''
        if nu is None:
            nu = self.nu
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(
            self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action)
            min_qf_next_target = torch.min(
                qf1_next_target,
                qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (
                min_qf_next_target)
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        sqf1_pi, sqf2_pi = self.safety_critic(state_batch, pi)
        max_sqf_pi = torch.max(sqf1_pi, sqf2_pi)

        if self.DGD_constraints:
            policy_loss = (
                (self.alpha * log_pi) + nu * (max_sqf_pi - self.eps_safe) -
                1. * min_qf_pi
            ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        else:
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean(
            )  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        # Optimize nu for LR
        if self.update_nu:
            nu_loss = (self.log_nu *
                       (self.eps_safe - max_sqf_pi).detach()).mean()
            self.nu_optim.zero_grad()
            nu_loss.backward()
            self.nu_optim.step()
            self.nu = self.log_nu.exp()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(
        ), alpha_loss.item(), alpha_tlogs.item()