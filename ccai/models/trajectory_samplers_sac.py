from typing import Dict, Tuple, List, Optional, Union
import numpy as np
import gpytorch
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.optim import Adam
from gpytorch.optim import NGD

import random
import time
import collections
from collections import deque

from tqdm import tqdm
from ccai.models.custom_likelihoods import SigmoidBernoulliLikelihood
from ccai.models.helpers import MLP
from ccai.models.likelihood_residual_gp import LikelihoodResidualGP

# Normalizing flow stuff
# from nflows.flows.base import Flow
# from nflows.distributions.normal import ConditionalDiagonalNormal
# from nflows.transforms.base import CompositeTransform
# from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform, AffineCouplingTransform
# from nflows.transforms.lu import LULinear
# from nflows.nn.nets import ResidualNet
# from nflows.utils import torchutils

# Diffusion
from ccai.models.diffusion.diffusion import GaussianDiffusion, ConstrainedDiffusion, JointDiffusion, LatentDiffusion
from ccai.models.cnf.cnf import TrajectoryCNF

from ccai.models.helpers import MLP
from ccai.models.likelihood_residual_gp import LikelihoodResidualGP

import time
import collections

# Import fast prediction settings
from gpytorch.settings import fast_pred_var, fast_pred_samples, cg_tolerance

# Import the new PrioritizedReplayBuffer
from ccai.models.replay_buffer import PrioritizedReplayBuffer

class TrajectoryFlowModel(nn.Module):

    def __init__(self, T, dx, du, context_dim, dynamics=None):
        super().__init__()
        self.T = T
        self.dx = dx
        self.du = du
        split_prior = False
        if dynamics is not None:
            flow_dim = T * du
        else:
            flow_dim = T * (dx + du)

        self.dynamics = dynamics
        # self.flow = build_ffjord(T*(dx+du), context_dim, 1)
        # self.flow = OTFlow(T*(dx+du), 64, 2, context_dim)

        if split_prior or dynamics is not None:
            prior_dim = T * du
        else:
            prior_dim = T * (dx + du)

        def create_transform_net(in_features, out_features):
            net = ResidualNet(in_features, out_features, hidden_features=64, context_features=context_dim)
            return net

        # base_dist = StandardNormal(shape=[flow_dim])
        base_dist = ConditionalDiagonalNormal(shape=[flow_dim], context_encoder=MLP(context_dim,
                                                                                    2 * flow_dim,
                                                                                    hidden_size=64)
                                              )

        transforms = []
        for _ in range(16):
            # transforms.append(ReversePermutation(features=flow_dim))
            # transforms.append(RandomPermutation(features=flow_dim))
            transforms.append(LULinear(features=flow_dim))
            # transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=flow_dim,
            #                                                                          context_features=context_dim,
            #                                                                          hidden_features=64,
            #                                                                          tails='linear',
            #                                                                          tail_bound=4))
            mask = torchutils.create_mid_split_binary_mask(flow_dim)
            # transforms.append(PiecewiseRationalQuadraticCouplingTransform(mask=mask,
            #                                                              transform_net_create_fn=create_transform_net,
            #                                                              tails='linear', tail_bound=4))
            transforms.append(AffineCouplingTransform(mask=mask, transform_net_create_fn=create_transform_net))

        transforms = CompositeTransform(transforms)
        self.flow = Flow(transforms, base_dist)


    def sample(self, N, H=None, start=None, goal=None, constraints=None, T=1):
        B = 1
        if start is not None:
            B = start.shape[0]
            context = torch.cat((start, goal), dim=1)
        if constraints is not None:
            context = torch.cat((context, constraints), dim=1)

        samples = self.flow.sample(num_samples=N, context=context)
        trajectories = samples.reshape(B, N, self.T, -1)

        if self.dynamics is not None:
            x = [start.clone()]
            for t in range(self.T - 1):
                x.append(self.dynamics(x[-1], trajectories[:, t]))
            x = torch.stack(x, dim=1)
            trajectories = torch.cat((x, trajectories), dim=-1)

        trajectories = trajectories.reshape(B, N, self.T, self.dx + self.du)
        if B == 1:
            return trajectories[0]
        return trajectories

    def log_prob(self, trajectories, start=None, goal=None, constraints=None):
        B = trajectories.shape[0]
        context = None
        if start is not None:
            B = start.shape[0]
            context = torch.cat((start, goal), dim=1)

        if constraints is not None:
            context = torch.cat((context, constraints), dim=1)

        if self.dynamics is not None:
            trajectories = trajectories[:, :, -self.du:]

        log_prob = self.flow.log_prob(trajectories.reshape(B, -1), context=context)
        return log_prob.reshape(B)

    def loss(self, trajectories, start=None, goal=None, constraints=None):
        return -self.log_prob(trajectories, start, goal, constraints).mean()


class TrajectoryDiffusionModel(nn.Module):

    def __init__(self, T, dx, du, context_dim, problem=None, timesteps=20, hidden_dim=64, constrained=False,
                 unconditional=False, generate_context=False, score_model='conv_unet', latent_diffusion=False,
                 vae=None, inits_noise=None, noise_noise=None, guided=False):
        super().__init__()
        self.T = T
        self.dx = dx
        self.du = du
        self.context_dim = context_dim

        if latent_diffusion:
            self.diffusion_model = LatentDiffusion(vae, T, dx, du, context_dim,
                                                            timesteps=timesteps, sampling_timesteps=timesteps,
                                                            hidden_dim=hidden_dim,
                                                            unconditional=unconditional)
        else:
            if constrained:
                self.diffusion_model = ConstrainedDiffusion(T, dx, du, context_dim, problem,
                                                            timesteps=timesteps, sampling_timesteps=timesteps,
                                                            constrain=constrained,
                                                            hidden_dim=hidden_dim,
                                                            unconditional=unconditional,
                                                            inits_noise=inits_noise, noise_noise=noise_noise,
                                                            guided=guided)
            else:
                if generate_context:
                    self.diffusion_model = JointDiffusion(T, dx, du, context_dim, timesteps=timesteps,
                                                        sampling_timesteps=timesteps,
                                                        hidden_dim=hidden_dim,
                                                        model_type=score_model,
                                                        inits_noise=inits_noise, noise_noise=noise_noise,
                                                        guided=guided)
                else:
                    self.diffusion_model = GaussianDiffusion(T, dx, du, context_dim, timesteps=timesteps,
                                                            sampling_timesteps=timesteps, hidden_dim=hidden_dim,
                                                            unconditional=unconditional,
                                                            model_type=score_model)

    def construct_context(self, constraints=None):
        if constraints is not None:
            constraints = constraints.reshape(constraints.shape[0], -1, constraints.shape[-1])
            context = constraints
            # context = torch.cat((start.unsqueeze(1).repeat(1, N, 1),
            #                     goal.unsqueeze(1).repeat(1, N, 1),
            #                     constraints), dim=-1)
        else:
            context = None

        return context

    def construct_condition(self, H=None, start=None, goal=None, past=None):
        condition = {}
        time_index = 0
        if past is not None:
            condition[0] = [0, past]
            time_index = past.shape[1]
        if start is not None:
            condition[time_index] = [0, start]
        if goal is not None:
            g = torch.stack((torch.cos(goal), torch.sin(goal)), dim=1).reshape(-1, 2)
            condition[H - 1] = [14, g]
        if condition == {}:
            condition = None
        return condition
    
    def sample(self, N, H=None, start=None, goal=None, constraints=None, past=None, project=False, no_grad=True, skip_likelihood=False, context_for_likelihood=True, residual_gp=None):
        # B, N, _ = constraints.shape
        context = self.construct_context(constraints)

        condition = self.construct_condition(H, start, goal, past)

        if project:
            samples, samples_0, (all_losses, all_samples, all_likelihoods) = self.diffusion_model.project(N=N, H=H, context=context, condition=condition, residual_gp=residual_gp)
            return samples, samples_0, (all_losses, all_samples, all_likelihoods)
        else:
            samples = self.diffusion_model.sample(N=N, H=H, context=context, condition=condition, no_grad=no_grad, skip_likelihood=skip_likelihood, context_for_likelihood=context_for_likelihood)  # .reshape(-1, H#,
        #         self.dx + self.du)
            return samples

    def loss(self, trajectories, mask=None, start=None, goal=None, constraints=None):
        B = trajectories.shape[0]
        if start is not None:
            context = torch.cat((start, goal, constraints), dim=1)
        else:
            context = constraints
        # context=None
        return self.diffusion_model.loss(trajectories, context=context, mask=mask).mean()

    def classifier_loss(self, trajectories, mask=None, context=None, label=None):
        return self.diffusion_model.classifier_loss(trajectories, context=context, mask=mask, label=label)

    def set_norm_constants(self, x_mu, x_std):
        self.diffusion_model.set_norm_constants(x_mu, x_std)

    def grad(self, trajectories, t, start, goal, constraints):
        B, N, _ = constraints.shape
        context = torch.cat((start.unsqueeze(1).repeat(1, N, 1),
                             goal.unsqueeze(1).repeat(1, N, 1),
                             constraints), dim=-1)
        return self.diffusion_model.model_predictions(trajectories, t, context=context).pred_noise

    def resample(self, start, goal, constraints, initial_trajectory, past, timestep):
        # B, N, _ = constraints.shape
        N, H, _ = initial_trajectory.shape
        context = None
        if constraints is not None:
            constraints = constraints.reshape(constraints.shape[0], -1, constraints.shape[-1])
            context = constraints
        condition = {}
        time_index = 0
        if past is not None:
            condition[0] = [0, past]
            time_index = past.shape[1]
        if start is not None:
            condition[time_index] = [0, start]
        if goal is not None:
            condition[-1] = [8, goal]
        if condition == {}:
            condition = None

        samples, c = self.diffusion_model.resample(x=initial_trajectory, context=context,
                                                   condition=condition,
                                                   timestep=timestep)
        return samples.reshape(-1, self.T, self.dx + self.du), c.reshape(c.shape[0], -1, self.context_dim)

    def likelihood(self, trajectories, context):
        return self.diffusion_model.approximate_likelihood(trajectories, context)


class TrajectoryCNFModel(TrajectoryCNF):

    def __init__(self, horizon, dx, du, context_dim, problem, hidden_dim=32, state_only=False, state_control_only=False):
        super().__init__(horizon, dx, du, context_dim, problem, hidden_dim=hidden_dim, state_only=state_only,
                         state_control_only=state_control_only)

    def sample(self, N, H=None, start=None, goal=None, constraints=None, past=None, project=False):
        # B, N, _ = constraints.shape
        if constraints is not None:
            constraints = constraints.reshape(constraints.shape[0], -1, constraints.shape[-1])
            context = constraints
        else:
            context = None
        condition = {}
        time_index = 0
        mask = torch.ones(N, self.horizon, self.dx + self.du, device=self._grad_mask.device)
        if past is not None:
            condition[0] = [0, past]
            time_index = past.shape[1]
            mask[:, :time_index] = torch.zeros_like(past)
        if start is not None:
            condition[time_index] = [0, start]
            mask[:, time_index, :start.shape[1]] = torch.zeros_like(start)
        if goal is not None:
            condition[-1] = [8, goal]
        if condition == {}:
            condition = None

        # TODO: in-painting for sample generation with CNF
        # I guess just a case of intervening on the required gradient?
        # initialize trajectory to right amount, set gradient of components to be zero
        # trajectories, likelihood = self._sample(context=context, condition=condition, mask=mask, H=H)
        # return (trajectories, context, likelihood), trajectories

        if project:
            samples, samples_0, (all_losses, all_samples, all_likelihoods) = self.project(H=H, context=context, condition=condition)
        else:
            samples = self._sample(H=H, context=context, condition=condition)  # .reshape(-1, H#,
        #         self.dx + self.du)
        return samples, samples_0, (all_losses, all_samples, all_likelihoods)


    def loss(self, trajectories, mask=None, start=None, goal=None, constraints=None):
        B = trajectories.shape[0]
        if start is not None:
            context = torch.cat((start, goal, constraints), dim=1)
        else:
            context = constraints
        return self.flow_matching_loss(trajectories, context=context, mask=mask)

    def set_norm_constants(self, x_mu, x_std):
        self.x_mean.data = x_mu.to(device=self.x_mean.device, dtype=self.x_mean.dtype)
        self.x_std.data = x_std.to(device=self.x_std.device, dtype=self.x_std.dtype)

        self.model.x_mean = self.x_mean
        self.model.x_std = self.x_std

class TrajectorySampler(nn.Module):

    def __init__(self, T, dx, du, context_dim, type='nf', dynamics=None, problem=None, timesteps=50, hidden_dim=64,
                 constrain=False, unconditional=False, generate_context=False, score_model='conv_unet',
                 latent_diffusion=False, vae=None, inits_noise=None, noise_noise=None, guided=False, discriminator_guidance=False,
                 learn_inverse_dynamics=False, state_only=False, state_control_only=False,
                 rl_adjustment=False, initial_threshold=-15):
        super().__init__()
        self.T = T
        self.dx = dx
        self.du = du
        self.context_dim = context_dim
        self.inverse_dynamics = None
        self.type = type
        assert type in ['nf', 'latent_diffusion', 'diffusion', 'cnf']
        if type == 'nf':
            self.model = TrajectoryFlowModel(T, dx, du, context_dim, dynamics)
        elif type == 'cnf':
            self.model = TrajectoryCNFModel(T, dx, du, context_dim, problem, hidden_dim=hidden_dim, 
                                            state_only=state_only, state_control_only=state_control_only)
        elif type == 'latent_diffusion':
            self.model = TrajectoryDiffusionModel(T, dx, du, context_dim, problem, timesteps, hidden_dim, constrain,
                                                  unconditional, generate_context=generate_context, score_model=score_model,
                                                  latent_diffusion=True, vae=vae)
        else:
            self.model = TrajectoryDiffusionModel(T, dx, du, context_dim, problem, timesteps, hidden_dim, constrain,
                                                  unconditional, generate_context=generate_context, score_model=score_model,
                                                  inits_noise=inits_noise, noise_noise=noise_noise, guided=guided)

        self.register_buffer('x_mean', torch.zeros(dx + du))
        self.register_buffer('x_std', torch.ones(dx + du))

        self.rl_adjustment = rl_adjustment
        if self.rl_adjustment:
            # Create the actor GP (policy network)
            actor_likelihood = SigmoidBernoulliLikelihood(
                initial_bias=-initial_threshold,
                initial_temp=0.1,
                num_samples=15
            )
            self.gp = LikelihoodResidualGP(
                dx, 
                num_inducing=32,
                kernel_type="matern32",
                likelihood=actor_likelihood,
                use_whitening=True
            )
            
            # Create the critic GP (Q-function)
            self.value_function = LikelihoodResidualGP(
                dx, 
                num_inducing=32, 
                kernel_type="matern32",
                use_ard=True,
                use_whitening=True
            )
            
            # Create target critic for stability
            self.value_function_target = LikelihoodResidualGP(
                dx, 
                num_inducing=32, 
                kernel_type="matern32",
                use_ard=True,
                use_whitening=True
            )
            # Copy weights
            self.value_function_target.load_state_dict(self.value_function.state_dict())
            
            # Separate variational parameters from hyperparameters for NGD
            actor_variational_params = []
            actor_hyper_params = []
            for name, param in self.gp.gp_model.named_parameters():
                if 'variational' in name:
                    actor_variational_params.append(param)
                else:
                    actor_hyper_params.append(param)
            
            value_variational_params = []
            value_hyper_params = []
            for name, param in self.value_function.gp_model.named_parameters():
                if 'variational' in name:
                    value_variational_params.append(param)
                else:
                    value_hyper_params.append(param)
            
            # SAC hyperparameters
            self.gamma = 0.99  # Discount factor
            self.tau = 0.005   # Soft target update parameter
            self.batch_size = 64
            self.alpha = 0.2   # Temperature parameter for entropy
            self.buffer_size = 10000  # Replay buffer size
            
            # Prioritized experience replay parameters
            self.per_alpha = 0.75  # How much prioritization to use (0 = none, 1 = full)
            self.per_beta = 0.4   # Importance sampling correction (0 = none, 1 = full)
            self.per_beta_increment = 0.001  # Increment per update
            self.per_epsilon = 0.01  # Small constant to ensure non-zero priority
            
            # Using NGD for variational parameters
            self.actor_ngd_optimizer = NGD(actor_variational_params, num_data=self.batch_size, lr=0.1)
            self.value_ngd_optimizer = NGD(value_variational_params, num_data=self.batch_size, lr=0.1)
            
            # Using Adam for hyperparameters
            self.sac_optimizer = Adam([
                {'params': actor_hyper_params, 'lr': 1e-2},
                {'params': self.gp.likelihood.parameters(), 'lr': 1e-2},
                {'params': value_hyper_params, 'lr': 1e-2},
                {'params': self.value_function.likelihood.parameters(), 'lr': 1e-2}
            ])
            
            # Replace regular replay buffer with prioritized version
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.buffer_size,
                alpha=self.per_alpha,
                beta=self.per_beta,
                beta_increment=self.per_beta_increment,
                epsilon=self.per_epsilon
            )

        self.send_norm_constants_to_submodels()

    def to(self, device: torch.device) -> 'TrajectorySampler':
        """
        Move the model to the specified device, including the GP model.
        
        Args:
            device: Device to move the model to
            
        Returns:
            Self for method chaining
        """

        self.device = device
        # Move the main model
        self.model = self.model.to(device)
        
        # Move normalization constants
        self.x_mean = self.x_mean.to(device)
        self.x_std = self.x_std.to(device)
        
        # Move GP model and associated components if present
        if self.rl_adjustment:
            self.gp = self.gp.to(device)
            self.value_function = self.value_function.to(device)
        
        # Update submodels with new device
        self.send_norm_constants_to_submodels()
        
        return self


    def set_norm_constants(self, x_mean, x_std):
        self.x_mean.data = x_mean.to(device=self.x_mean.device, dtype=self.x_mean.dtype)
        self.x_std.data = x_std.to(device=self.x_std.device, dtype=self.x_std.dtype)

    def send_norm_constants_to_submodels(self):
        self.model.set_norm_constants(self.x_mean, self.x_std)
        self.gp.set_norm_constants(self.x_mean, self.x_std)
        self.value_function.set_norm_constants(self.x_mean, self.x_std)

    def convert_yaw_to_sine_cosine(self, xu):
        """
        xu is shape (N, T, 36)
        Replace the yaw in xu with sine and cosine and return the new xu
        """
        yaw = xu[14]
        sine = torch.sin(yaw)
        cosine = torch.cos(yaw)
        xu_new = torch.cat([xu[:14], cosine.unsqueeze(-1), sine.unsqueeze(-1), xu[15:]], dim=-1)
        return xu_new

    def check_id(self, state, N, threshold=None, likelihood_only=False, return_samples=False, deterministic=False):
        """
        Check if the state is in-distribution using the learned model.
        
        Args:
            state: Current state to check
            N: Number of samples for likelihood estimation
            threshold: Fallback threshold if not using GP (legacy)
            return_samples: Whether to return samples
            deterministic: Whether to use deterministic decision (True) or 
                          sample from Bernoulli (False)
            
        Returns:
            Tuple of (is_in_distribution, likelihood, [samples if return_samples])
        """
        start = state[:4 * 3 + 3]
        start_sine_cosine = self.convert_yaw_to_sine_cosine(start)
        
        # Use GPyTorch's fast prediction settings to speed up sampling
        with fast_pred_var(), fast_pred_samples(), cg_tolerance(1e-3):
            samples, _, likelihood = self.sample(N=N, H=self.T, start=start_sine_cosine.reshape(1, -1),
                    constraints=torch.ones(N, 3).to(device=state.device))
        
        likelihood = likelihood.reshape(N).mean().item()
        
        if self.rl_adjustment:
            # Use the optimized GP prediction
            with torch.no_grad():
                # Get residual mean with fast prediction
                residual, _ = self.gp.gp_model.predict(start_sine_cosine.unsqueeze(0), fast_mode=True)
                residual = residual.item()
                print('Likelihood residual:', residual)
                
                # Combine base likelihood with residual
                print('Original likelihood:', likelihood)
                adjusted_likelihood = likelihood + residual
                print('Adjusted likelihood:', adjusted_likelihood)

                if likelihood_only:
                    if return_samples:
                        return adjusted_likelihood, samples
                    else:
                        return adjusted_likelihood
                
                # Use sigmoid with bias and temperature to get ID probability
                bias = self.gp.likelihood.bias.item()
                temperature = self.gp.likelihood.temperature.item()
                id_probability = torch.sigmoid(temperature * (torch.tensor(adjusted_likelihood) + bias)).item()
                print(f'ID probability: {id_probability:.4f}')
                
                # Either use deterministic policy or sample from Bernoulli
                if deterministic:
                    # Deterministic mode (for evaluation or testing)
                    is_id = id_probability >= 0.5
                    print(f"Deterministic decision: {'ID' if is_id else 'OOD'}")
                else:
                    # Stochastic mode (for training and exploration)
                    is_id = torch.bernoulli(torch.tensor(id_probability)).bool().item()
                    print(f"Stochastic decision: {'ID' if is_id else 'OOD'} (sampled from p={id_probability:.4f})")
                
                if not is_id:
                    print('State is out of distribution')
                    if return_samples:
                        return False, adjusted_likelihood, samples
                    else:
                        return False, adjusted_likelihood
                if return_samples:
                    return True, adjusted_likelihood, samples
                else:
                    return True, adjusted_likelihood
        else:
            # Fall back to threshold based approach if rl_adjustment is False
            if threshold is None:
                threshold = -15  # Default threshold
            
            if likelihood < threshold:
                print('State is out of distribution')
                if return_samples:
                    return False, likelihood, samples
                else:
                    return False, likelihood
            if return_samples:
                return True, likelihood, samples
            else:
                return True, likelihood

    def sample(self, N, H=10, start=None, goal=None, constraints=None, past=None, project=False, no_grad=True, skip_likelihood=False, context_for_likelihood=True):
        norm_start = None
        norm_past = None
        if start is not None and self.type != 'latent_diffusion':
            norm_start = (start - self.x_mean[:self.dx]) / self.x_std[:self.dx]
        else:
            norm_start = start
        if past is not None and self.type != 'latent_diffusion':
            norm_past = (past - self.x_mean) / self.x_std
        else:
            norm_past = past
        if project:
            samples, samples_0, (all_losses, all_samples, all_likelihoods) = self.model.sample(N, H, norm_start, goal, constraints, norm_past, project, context_for_likelihood=context_for_likelihood, residual_gp=self.gp)
        else:
            samples = self.model.sample(N, H, norm_start, goal, constraints, norm_past, project, no_grad=no_grad, skip_likelihood=skip_likelihood, context_for_likelihood=context_for_likelihood)
        # if len(samples) == N:
        #     x = samples
        #     c = None
        #     likelihood = None
        if len(samples) == 2:
            x = samples[0]
            c = None
            likelihood = samples[1]
        else:
            x, c, likelihood = samples
            
        if self.type != 'latent_diffusion':
            if project:
                return x * self.x_std + self.x_mean, c, likelihood, samples_0 * self.x_std + self.x_mean, (all_losses, all_samples, all_likelihoods)
            elif not no_grad:
                return x, c, likelihood
            else:
                return x * self.x_std + self.x_mean, c, likelihood
        else:
            return x, c, likelihood

    def resample(self, start=None, goal=None, constraints=None, initial_trajectory=None, past=None, timestep=10):
        norm_start = None
        norm_initial_trajectory = None
        norm_past = None
        if start is not None:
            norm_start = (start - self.x_mean[:self.dx]) / self.x_std[:self.dx]
        if initial_trajectory is not None:
            norm_initial_trajectory = (initial_trajectory - self.x_mean) / self.x_std
        if past is not None:
            norm_past = (past - self.x_mean) / self.x_std

        x, c = self.model.resample(norm_start, goal, constraints, norm_initial_trajectory, norm_past,
                                   timestep)
        return x * self.x_std + self.x_mean, c

    def loss(self, trajectories, mask=None, start=None, goal=None, constraints=None):
        loss = self.model.loss(trajectories, mask=mask, start=start, goal=goal, constraints=constraints)
        if self.inverse_dynamics is not None and False:
            u = trajectories[:, :-1, self.dx:]
            x = trajectories[:, :-1, :self.dx]
            next_x = trajectories[:, 1:, :self.dx]
            x = torch.cat((x, next_x), dim=-1)
            pred_u = self.inverse_dynamics(x)
            loss_id = torch.mean((pred_u - u)**2)
            loss = loss + 0.1 * loss_id
        return loss

    def grad(self, trajectories, t, start, goal, constraints=None):
        return self.model.grad(trajectories, t, start, goal, constraints)

    def likelihood(self, trajectories, mask=None, start=None, goal=None, context=None):
        norm_traj = (trajectories - self.x_mean) / self.x_std
        return self.model.likelihood(norm_traj, context)

    def classifier_loss(self, trajectories, mask=None, context=None, label=None):
        return self.model.classifier_loss(trajectories, context=context, mask=mask, label=label)

    def evaluate_actions(self, states, actions):
        """
        Evaluate the log probability and value function for given states and actions,
        ensuring proper gradient flow for PPO updates.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, 1] (binary decisions: 0=OOD, 1=ID)
            
        Returns:
            action_log_probs: Log probabilities of actions
            state_values: Value function estimates
            dist_entropy: Entropy of the policy distribution
        """
        # Convert states for consistency
        state_batch = states
        
        # Sample trajectories to get base likelihoods
        with torch.no_grad(), fast_pred_var(), fast_pred_samples():
            sb_for_lb = state_batch.detach()
            batch_likelihoods = []
            _, _, likelihood = self.sample(
                N=8*sb_for_lb.shape[0],  # Small number for efficiency
                H=self.T,
                start=sb_for_lb.repeat_interleave(8, 0),
                constraints=torch.ones(8*sb_for_lb.shape[0], 3).to(device=sb_for_lb.device)
            )
            base_likelihoods = likelihood.reshape(-1, 8).mean(dim=1)
        state_batch_normalized = (state_batch - self.x_mean[:self.dx]) / self.x_std[:self.dx]
        
        # Create a new tensor that requires gradients - this is a trick to incorporate 
        # the base likelihoods into our computational graph
        base_likelihoods = base_likelihoods.clone().detach().requires_grad_(False)
        
        # Use fast_pred_var for GP predictions with gradients
        with fast_pred_var():
            output_mean = self.gp.gp_model(state_batch_normalized).mean
        
        # Combine base likelihoods with GP output
        combined_features = base_likelihoods + output_mean
        combined_features = combined_features.unsqueeze(-1)
        # Get Bernoulli distribution from the likelihood
        dist = self.gp.likelihood(combined_features)
        
        # Get log probabilities and entropy
        action_log_probs = dist.log_prob(actions.float())
        dist_entropy = dist.entropy().mean()  # Approximate entropy
        
        # Get value predictions using GP
        value_output = self.value_function.gp_model(state_batch_normalized)

        return action_log_probs, value_output, dist_entropy

    def store_transition(self, state, action: int, likelihood: float, next_state, reward: float, done: bool = False) -> None:
        """
        Store a transition in the SAC prioritized replay buffer.
        
        Args:
            state: Current state
            action: Action taken (0=OOD, 1=ID)
            likelihood: Base likelihood estimate
            next_state: Next state
            reward: Reward received
            done: Whether the episode is done
        """
        if not self.rl_adjustment:
            return
            
        # Process state with sine/cosine conversion
        state_tensor = torch.tensor(
            self.convert_yaw_to_sine_cosine(state), 
            dtype=torch.float32, 
            device=next(self.parameters()).device
        ).unsqueeze(0)
        
        next_state_tensor = torch.tensor(
            self.convert_yaw_to_sine_cosine(next_state), 
            dtype=torch.float32, 
            device=next(self.parameters()).device
        ).unsqueeze(0) if next_state is not None else None
        
        # Store likelihood for context
        likelihood_tensor = torch.tensor([likelihood], 
                                          dtype=torch.float32, 
                                          device=next(self.parameters()).device)
        
        # Create experience tuple
        experience = (
            state_tensor, 
            action, 
            likelihood_tensor,
            reward, 
            next_state_tensor, 
            done
        )
        
        # Store transition with maximum priority for new experiences
        self.replay_buffer.add(experience)
    
    def sample_replay_buffer(self, batch_size: int = None) -> Tuple:
        """
        Sample a batch from the prioritized replay buffer.
        
        Args:
            batch_size: Number of transitions to sample (defaults to self.batch_size)
            
        Returns:
            Tuple of (states, actions, likelihoods, rewards, next_states, dones, indices, is_weights)
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        batch_size = min(batch_size, len(self.replay_buffer))
        
        # Sample from prioritized replay buffer
        experiences, indices, is_weights = self.replay_buffer.sample(batch_size)
        
        # Convert is_weights to tensor
        is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
        
        # Unpack the experiences
        states, actions, likelihoods, rewards, next_states, dones = zip(*experiences)
        
        # Convert to tensors and stack
        states = torch.cat(states)
        actions = torch.tensor(actions, dtype=torch.long, device=states.device).unsqueeze(1)
        likelihoods = torch.cat(likelihoods)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=states.device)
        
        # Handle None values in next_states for terminal states
        valid_next_states = [s for s in next_states if s is not None]
        if valid_next_states:
            next_states_tensor = torch.cat(valid_next_states)
            # For None values, use zeros or the last state
            missing_shape = list(next_states_tensor.shape[1:])
            for i, ns in enumerate(next_states):
                if ns is None:
                    next_states[i] = torch.zeros([1] + missing_shape, device=states.device)
            next_states = torch.cat(next_states)
        else:
            # If all next_states are None, create zeros
            next_states = torch.zeros_like(states)
            
        dones = torch.tensor(dones, dtype=torch.float32, device=states.device)
        
        return states, actions, likelihoods, rewards, next_states, dones, indices, is_weights
    
    def get_action(self, state: torch.Tensor, likelihood: float, deterministic: bool = False) -> Tuple[int, float]:
        """
        Sample an action from the policy, adding the likelihood as context.
        
        Args:
            state: Current state tensor
            likelihood: Base likelihood from trajectory sampling
            deterministic: If True, choose most likely action
            
        Returns:
            Tuple of (action, log_prob)
        """
        with torch.no_grad(), fast_pred_var():
            # Get GP prediction
            output_mean, _ = self.gp.predict(state, fast_mode=True)
            
            # Add base likelihood for context
            combined_feature = likelihood + output_mean
            combined_feature = combined_feature.unsqueeze(-1)
            
            # Convert to Bernoulli distribution via the likelihood
            dist = self.gp.likelihood(combined_feature)
            
            # Generate probabilities for action 1 (ID)
            prob_id = dist.mean  # Probability of being in-distribution
            
            # Create a proper categorical distribution
            probs = torch.cat([1-prob_id, prob_id], dim=-1)
            
            # Sample from distribution or take most likely
            if deterministic:
                action = torch.argmax(probs).item()
            else:
                action = torch.distributions.Categorical(probs).sample().item()
            
            # Calculate log prob for the action
            log_prob = torch.log(probs.squeeze()[action] + 1e-10)
            
        return action, log_prob.item()

    def sac_update(self) -> Dict[str, float]:
        """
        Update the actor (GP) and critic (value_function) using SAC with prioritized experience replay.
        
        Returns:
            Dictionary with loss information
        """
        if len(self.replay_buffer) < self.batch_size:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "alpha_loss": 0.0,
                "mean_reward": 0.0
            }
        
        # Sample from replay buffer with importance sampling weights
        states, actions, likelihoods, rewards, next_states, dones, indices, is_weights = self.sample_replay_buffer()
        
        # Normalize states
        state_norm = (states - self.x_mean[:self.dx]) / self.x_std[:self.dx]
        next_state_norm = (next_states - self.x_mean[:self.dx]) / self.x_std[:self.dx]
        
        # Track inducing point changes
        with torch.no_grad():
            actor_ip_before = self.gp.gp_model.variational_strategy.inducing_points.clone()
            critic_ip_before = self.value_function.gp_model.variational_strategy.inducing_points.clone()
        
        # Critic update (minimize Q-value loss)
        with torch.no_grad():
            # Get next state GP output for target critic
            next_output_mean = self.gp.gp_model(next_state_norm).mean
            
            # Combine with likelihoods (assuming they're preserved across transitions)
            next_combined = next_output_mean + likelihoods
            next_combined = next_combined.unsqueeze(-1)
            
            # Get action probabilities using the target network
            next_probs = self.gp.likelihood(next_combined).mean
            next_probs = torch.cat([1-next_probs, next_probs], dim=-1)
            next_dist = Categorical(next_probs)
            
            # Calculate entropy term
            next_entropy = next_dist.entropy()
            
            # Calculate expected next Q-value with entropy regularization
            next_q_values = self.value_function_target.gp_model(next_state_norm).mean
            
            # Target Q = r + γ(Q(s',a') + α*H(π(·|s')))
            target_q = rewards + (1 - dones) * self.gamma * (next_q_values.squeeze() + self.alpha * next_entropy)
        
        # Zero gradients
        self.value_ngd_optimizer.zero_grad()
        self.sac_optimizer.zero_grad()
        
        # Current Q-value prediction
        current_q_values = self.value_function.gp_model(state_norm).mean
        
        # Compute TD errors for priority updates
        td_errors = target_q - current_q_values.squeeze()
        
        # Compute critic loss with importance sampling weights
        critic_loss = (is_weights * F.mse_loss(current_q_values.squeeze(), target_q, reduction='none')).mean()
        
        # Backpropagate critic loss
        critic_loss.backward()
        
        # Step optimizers for critic
        self.value_ngd_optimizer.step()
        self.sac_optimizer.step()
        
        # Actor update (maximize policy objective)
        # Zero gradients
        self.actor_ngd_optimizer.zero_grad()
        self.sac_optimizer.zero_grad()
        
        # Get output from GP
        output_mean = self.gp.gp_model(state_norm).mean
        
        # Combine with likelihoods
        combined_features = output_mean + likelihoods
        combined_features = combined_features.unsqueeze(-1)
        
        # Get action probabilities
        probs = self.gp.likelihood(combined_features).mean
        probs = torch.cat([1-probs, probs], dim=-1)
        dist = Categorical(probs)
        
        # Get entropy
        entropy = dist.entropy().mean()
        
        # Get log probs for each action in the batch
        log_probs = torch.log(torch.gather(probs, 1, actions) + 1e-10)
        
        # Estimate Q-value for current policy
        q_values = self.value_function.gp_model(state_norm).mean
        
        # Actor loss: E[α*log(π(a|s)) - Q(s,a)] with importance sampling
        actor_loss = (is_weights * (self.alpha * log_probs - q_values)).mean()
        
        # Backpropagate actor loss
        actor_loss.backward()
        
        # Step optimizers for actor
        self.actor_ngd_optimizer.step()
        self.sac_optimizer.step()
        
        # Soft update target networks
        with torch.no_grad():
            for param, target_param in zip(
                self.value_function.parameters(),
                self.value_function_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
                
            # Calculate inducing point changes
            actor_ip_after = self.gp.gp_model.variational_strategy.inducing_points
            critic_ip_after = self.value_function.gp_model.variational_strategy.inducing_points
            actor_ip_change = torch.norm(actor_ip_after - actor_ip_before).item()
            critic_ip_change = torch.norm(critic_ip_after - critic_ip_before).item()
            print(f"Actor IP change: {actor_ip_change:.6f} | Critic IP change: {critic_ip_change:.6f}")
        
        # Update priorities in the replay buffer
        td_errors_np = td_errors.abs().detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors_np)
        
        # Return metrics
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "mean_reward": rewards.mean().item(),
            "actor_ip_change": actor_ip_change,
            "critic_ip_change": critic_ip_change,
            "mean_td_error": td_errors.abs().mean().item()
        }
        
    def adjust_temperature(self, alpha: float) -> None:
        """
        Adjust the temperature parameter alpha.
        
        Args:
            alpha: New temperature value
        """
        self.alpha = alpha
        print(f"SAC temperature adjusted: α={alpha}")
        
    def adjust_ngd_learning_rate(self, actor_lr: float = 0.1, value_lr: float = 0.1) -> None:
        """
        Adjust the learning rates for Natural Gradient Descent optimizers.
        
        Args:
            actor_lr: Learning rate for actor GP optimizer
            value_lr: Learning rate for critic GP optimizer
        """
        for param_group in self.actor_ngd_optimizer.param_groups:
            param_group['lr'] = actor_lr
            
        for param_group in self.value_ngd_optimizer.param_groups:
            param_group['lr'] = value_lr
            
        print(f"NGD learning rates adjusted: Actor={actor_lr}, Value={value_lr}")