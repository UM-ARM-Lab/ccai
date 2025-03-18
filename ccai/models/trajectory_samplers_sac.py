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
from ccai.models.custom_likelihoods import CDFBernoulliLikelihood
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
from ccai.models.custom_predictive_log_likelihood import CustomPredictiveLogLikelihood

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
            num_inducing = 32
            actor_likelihood = CDFBernoulliLikelihood(
                initial_threshold=initial_threshold,
                num_samples=15
            )
            self.gp = LikelihoodResidualGP(
                dx, 
                0,
                num_inducing=num_inducing,
                kernel_type="rbf",
                likelihood=actor_likelihood,
                use_whitening=True
            )
            value_function_inducing_points = torch.randn(num_inducing, dx+1)
            # Create the critic GP (Q-function)
            self.value_function = LikelihoodResidualGP(
                dx,
                1, 
                inducing_points=value_function_inducing_points, 
                kernel_type="rbf",
                use_ard=True,
                use_whitening=True
            )
            
            # Create target critic for stability
            self.value_function_target = LikelihoodResidualGP(
                dx,
                1, 
                inducing_points=value_function_inducing_points, 
                kernel_type="rbf",
                use_ard=True,
                use_whitening=True
            )
            
            # Create Q-function 2 for clipped double-Q trick
            value_function_inducing_points_2 = torch.randn(num_inducing, dx+1)
            self.value_function_2 = LikelihoodResidualGP(
                dx,
                1, 
                inducing_points=value_function_inducing_points_2, 
                kernel_type="rbf",
                use_ard=True,
                use_whitening=True
            )
            
            # Create target for Q-function 2
            self.value_function_target_2 = LikelihoodResidualGP(
                dx, 
                1,
                inducing_points=value_function_inducing_points_2, 
                kernel_type="rbf",
                use_ard=True,
                use_whitening=True
            )
            
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
            
            # Value function 2 parameters
            value_variational_params_2 = []
            value_hyper_params_2 = []
            for name, param in self.value_function_2.gp_model.named_parameters():
                if 'variational' in name:
                    value_variational_params_2.append(param)
                else:
                    value_hyper_params_2.append(param)
            
            # SAC hyperparameters
            self.gamma = 0.99  # Discount factor
            self.tau = 0.005   # Soft target update parameter
            self.batch_size = 2
            self.alpha = 0.02   # Temperature parameter for entropy
            self.buffer_size = 10000  # Replay buffer size
            
            # Prioritized experience replay parameters
            self.per_alpha = 0.75  # How much prioritization to use (0 = none, 1 = full)
            self.per_beta = 0.4   # Importance sampling correction (0 = none, 1 = full)
            self.per_beta_increment = 0.001  # Increment per update
            self.per_epsilon = 0.01  # Small constant to ensure non-zero priority
            
            # Using NGD for variational parameters
            self.actor_ngd_optimizer = NGD(actor_variational_params, num_data=self.batch_size, lr=0.1)
            self.value_ngd_optimizer = NGD(value_variational_params, num_data=self.batch_size, lr=0.1)
            self.value_ngd_optimizer_2 = NGD(value_variational_params_2, num_data=self.batch_size, lr=0.1)
            
            # Using Adam for hyperparameters
            self.sac_optimizer = Adam([
                {'params': actor_hyper_params, 'lr': 1e-2},
                {'params': self.gp.likelihood.parameters(), 'lr': 1e-2},
                {'params': value_hyper_params, 'lr': 1e-2},
                {'params': value_hyper_params_2, 'lr': 1e-2},
                {'params': self.value_function.likelihood.parameters(), 'lr': 1e-2},
                {'params': self.value_function_2.likelihood.parameters(), 'lr': 1e-2}
            ])
            
            # Replace regular replay buffer with prioritized version
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.buffer_size,
                alpha=self.per_alpha,
                beta=self.per_beta,
                beta_increment=self.per_beta_increment,
                epsilon=self.per_epsilon
            )
            
            # For tracking the last transition
            self._last_transition_idx = None
            
            # Reward normalization parameters
            self.normalize_rewards = True
            self.reward_stats = {
                'mean': 0.0,
                'var': 1.0,
                'count': 0,
                'running_mean': 0.0,
                'running_var': 0.0
            }
            self.reward_norm_eps = 1e-6  # Small constant for numerical stability

        self.num_update = 5  # Multiple updates per call
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
            self.value_function_target = self.value_function_target.to(device)
            # Add second Q-function and target
            self.value_function_2 = self.value_function_2.to(device)
            self.value_function_target_2 = self.value_function_target_2.to(device)
        
        # Update submodels with new device
        self.send_norm_constants_to_submodels()
        
        return self


    def set_norm_constants(self, x_mean, x_std):
        self.x_mean.data = x_mean.to(device=self.x_mean.device, dtype=self.x_mean.dtype)
        self.x_std.data = x_std.to(device=self.x_std.device, dtype=self.x_std.dtype)

    def send_norm_constants_to_submodels(self):
        self.model.set_norm_constants(self.x_mean, self.x_std)
        if hasattr(self, 'gp'):
            self.gp.set_norm_constants(self.x_mean, self.x_std)
        if hasattr(self, 'value_function'):
            self.value_function.set_norm_constants(self.x_mean, self.x_std)
            self.value_function_target.set_norm_constants(self.x_mean, self.x_std)
        if hasattr(self, 'value_function_2'):
            self.value_function_2.set_norm_constants(self.x_mean, self.x_std)
            self.value_function_target_2.set_norm_constants(self.x_mean, self.x_std)

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
        
        likelihood = likelihood.reshape(N).mean()
        
        if self.rl_adjustment:
            # Use the optimized GP prediction
            with torch.no_grad():
                # Get residual mean with fast prediction

                bernoulli_dist, adjusted_likelihood, _ = self.gp.predict(start_sine_cosine.unsqueeze(0), likelihood, fast_mode=True)
                adjusted_likelihood = adjusted_likelihood.item()
                likelihood = likelihood.item()
                # Combine base likelihood with residual
                print('Original likelihood:', likelihood)
                print('Adjusted likelihood:', adjusted_likelihood)

                residual = adjusted_likelihood - likelihood
                print('Likelihood residual:', residual)

                if likelihood_only:
                    if return_samples:
                        return adjusted_likelihood, samples
                    else:
                        return adjusted_likelihood
    
                
                # Either use deterministic policy or sample from Bernoulli
                if deterministic:
                    # Deterministic mode (for evaluation or testing)
                    is_id = bernoulli_dist.probs.item() >= 0.5
                    print(f"Deterministic decision: {'ID' if is_id else 'OOD'}")
                else:
                    # Stochastic mode (for training and exploration)
                    is_id = (1-bernoulli_dist.sample((1,))).bool().item()
                    id_probability = 1-bernoulli_dist.probs.item()
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
        
        # Update reward statistics with the new reward
        reward_tensor = torch.tensor([reward], device=next(self.parameters()).device)
        self._update_reward_stats(reward_tensor)
        
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
        self._last_transition_idx = self.replay_buffer.add(experience)
        
    def mark_last_transition_done(self) -> bool:
        """
        Mark the last added transition as done (terminal state).
        
        Returns:
            bool: True if successfully marked, False if no last transition exists
        """
        if not self.rl_adjustment or self._last_transition_idx is None:
            return False
            
        # Check if the index is valid in the buffer
        if self._last_transition_idx >= len(self.replay_buffer):
            print("Warning: Last transition index is out of bounds")
            return False
            
        try:
            # Get the transition
            transition = self.replay_buffer._storage[self._last_transition_idx]
            if transition is None:
                print("Warning: No transition found at last index")
                return False
                
            # Unpack transition components
            state, action, likelihood, reward, next_state, _ = transition
            
            # Create updated transition with done=True
            updated_transition = (state, action, likelihood, reward, next_state, True)
            
            # Update the storage
            self.replay_buffer._storage[self._last_transition_idx] = updated_transition
            
            print(f"Successfully marked transition {self._last_transition_idx} as done")
            return True
            
        except Exception as e:
            print(f"Failed to mark last transition as done: {e}")
            return False

    def get_memory_size(self):
        """
        Get the current size of the replay buffer.
        
        Returns:
            Size of the replay buffer
        """
        return len(self.replay_buffer)
    
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
    
    def get_action(self, state: torch.Tensor, likelihood: float, deterministic: bool = False, return_entropy=False) -> Tuple[int, float]:
        """
        Sample an action from the policy, adding the likelihood as context.
        
        Args:
            state: Current state tensor
            likelihood: Base likelihood from trajectory sampling
            deterministic: If True, choose most likely action
            
        Returns:
            Tuple of (action, log_prob, entropy [if return_entropy])
        """
        with torch.no_grad(), fast_pred_var():
            # Get GP prediction
            dist = self.gp.predict(state, torch.tensor(likelihood).to(device=state.device, dtype=state.dtype), fast_mode=True)
            
            action = torch.round(dist.probs).item() if deterministic else dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action).to(device=state.device, dtype=state.dtype))
            
        if return_entropy:
            return action, log_prob.item(), dist.entropy().item()
        return action, log_prob.item()
    
    def get_action_batch(self, state: torch.Tensor, likelihood: torch.Tensor, deterministic: bool = False, return_entropy=False) -> Tuple[int, float]:
        """
        Sample an action from the policy, adding the likelihood as context.
        
        Args:
            state: Current state tensor
            likelihood: Base likelihood from trajectory sampling
            deterministic: If True, choose most likely action
            
        Returns:
            Tuple of (action, log_prob, entropy [if return_entropy])
        """
        with torch.no_grad(), fast_pred_var():
            # Get GP prediction
            dist, _, _ = self.gp.predict(state, likelihood, fast_mode=True)
            
            action = torch.round(dist.probs) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            
        if return_entropy:
            return action, log_prob, dist.entropy()
        return action, log_prob

    def sac_update(self) -> Dict[str, float]:
        """
        Update the actor (GP) and critic (value_function) using SAC with prioritized experience replay
        and clipped double-Q trick.
        
        Returns:
            Dictionary with loss information and timing statistics
        """
        update_start_time = time.time()
        timing_stats = collections.defaultdict(float)
        
        # Initialize track metrics
        aggregated_metrics = {
            "actor_loss": 0.0,
            "critic_loss_1": 0.0,
            "critic_loss_2": 0.0,
            "entropy": 0.0,
            "mean_reward": 0.0,
            "actor_ip_change": 0.0,
            "critic_ip_change_1": 0.0,
            "critic_ip_change_2": 0.0,
            "mean_td_error": 0.0
        }
        
        if len(self.replay_buffer) < self.batch_size:
            return aggregated_metrics
        
        # Set models to evaluation mode for inference
        self.gp.gp_model.eval()
        self.gp.likelihood.eval()
        self.value_function.gp_model.eval()
        self.value_function.likelihood.eval()
        self.value_function_2.gp_model.eval() 
        self.value_function_2.likelihood.eval()
        self.value_function_target.gp_model.eval()
        self.value_function_target.likelihood.eval()
        self.value_function_target_2.gp_model.eval()
        self.value_function_target_2.likelihood.eval()
        
        for update_iter in range(self.num_update):
            iter_start_time = time.time()
            
            # Sample from replay buffer with importance sampling weights
            sampling_start = time.time()
            states, actions, likelihoods, rewards, next_states, dones, indices, is_weights = self.sample_replay_buffer()
            timing_stats['sampling_time'] += time.time() - sampling_start
            
            # Normalize rewards if enabled
            raw_rewards = rewards.clone()
            normalized_rewards = self._normalize_rewards(rewards)
            
            # Normalize states
            norm_start = time.time()
            state_norm = (states - self.x_mean[:self.dx]) / self.x_std[:self.dx]
            next_state_norm = (next_states - self.x_mean[:self.dx]) / self.x_std[:self.dx]

            # Concatenate the actions to the states for Q functions
            state_action_norm = torch.cat([state_norm, actions.float()], dim=-1)
            timing_stats['norm_time'] += time.time() - norm_start
            
            # Track inducing point changes
            ip_start = time.time()
            with torch.no_grad():
                actor_ip_before = self.gp.gp_model.variational_strategy.inducing_points.clone()
                critic_ip_before_1 = self.value_function.gp_model.variational_strategy.inducing_points.clone()
                critic_ip_before_2 = self.value_function_2.gp_model.variational_strategy.inducing_points.clone()
            timing_stats['ip_tracking'] += time.time() - ip_start
            
            # ------------------------ Update Critic (Q-functions) ------------------------
            target_q_start = time.time()
            with torch.no_grad():
                # Get next state GP output and action probabilities
                action, _, next_entropy = self.get_action_batch(next_state_norm, likelihoods, deterministic=False, return_entropy=True)

                next_state_action_norm = torch.cat([next_state_norm, action.reshape(-1, 1)], dim=-1).to(dtype=next_state_norm.dtype)

                # Get both Q-values and take the minimum for the target (clipped double-Q trick)
                _, next_q_values_1, _ = self.value_function_target.predict(next_state_action_norm, fast_mode=True)
                _, next_q_values_2, _ = self.value_function_target_2.predict(next_state_action_norm, fast_mode=True)
                
                # Take minimum of the two Q-values for the target calculation
                next_q_values_min = torch.min(next_q_values_1, next_q_values_2)
                
                # Target Q = r + γ(min(Q1, Q2) + α*H(π(·|s')))
                # Use normalized rewards here
                target_q = normalized_rewards + (1 - dones) * self.gamma * (next_q_values_min + self.alpha * next_entropy)
            timing_stats['target_q_time'] += time.time() - target_q_start
            
            # Update first Q-function
            q1_update_start = time.time()
            # Set models to training mode for SGD
            self.value_function.gp_model.train()
            self.value_function.likelihood.train()
            
            # Zero gradients for first critic update
            self.value_ngd_optimizer.zero_grad()
            self.sac_optimizer.zero_grad()
            
            # Create predictive log likelihood objective for Q1
            batch_objective_1 = CustomPredictiveLogLikelihood(
                self.value_function.likelihood,
                self.value_function.gp_model,
                num_data=states.shape[0]
            )
            
            # Current Q1 prediction
            current_q_values_1 = self.value_function.gp_model(state_action_norm)
                        
            # Compute Q1 loss using CustomPredictiveLogLikelihood
            critic_loss_1 = -batch_objective_1(current_q_values_1, target_q)
            
            # Weight the loss with importance sampling weights
            critic_loss_1 = (is_weights * critic_loss_1).sum()
            
            # Backpropagate first critic loss
            critic_loss_1.backward()
            
            # Step optimizers for first critic
            self.value_ngd_optimizer.step()
            self.sac_optimizer.step()
            timing_stats['q1_update_time'] += time.time() - q1_update_start
            
            # Update second Q-function 
            q2_update_start = time.time()
            # Set models to training mode
            self.value_function_2.gp_model.train()
            self.value_function_2.likelihood.train()
            
            # Zero gradients for second critic update
            self.value_ngd_optimizer_2.zero_grad()
            self.sac_optimizer.zero_grad()
            
            # Create predictive log likelihood objective for Q2
            batch_objective_2 = CustomPredictiveLogLikelihood(
                self.value_function_2.likelihood,
                self.value_function_2.gp_model,
                num_data=states.shape[0]
            )
            
            # Current Q2 prediction
            current_q_values_2 = self.value_function_2.gp_model(state_action_norm)
            
            min_q_for_td_error = torch.min(current_q_values_1.mean.detach().squeeze(), current_q_values_2.mean.detach().squeeze())
            td_errors = (target_q - min_q_for_td_error).detach()

            # Compute Q2 loss using CustomPredictiveLogLikelihood
            critic_loss_2 = -batch_objective_2(current_q_values_2, target_q)
            
            # Weight the loss with importance sampling weights
            critic_loss_2 = (is_weights * critic_loss_2).sum()
            
            # Backpropagate second critic loss
            critic_loss_2.backward()
            
            # Step optimizers for second critic
            self.value_ngd_optimizer_2.step()
            self.sac_optimizer.step()
            timing_stats['q2_update_time'] += time.time() - q2_update_start
            
            # ------------------------ Update Actor (Policy) ------------------------
            actor_update_start = time.time()
            # Set actor to training mode
            self.gp.gp_model.train()
            self.gp.likelihood.train()
            
            # Set critics to evaluation mode for policy update
            self.value_function.gp_model.eval()
            self.value_function.likelihood.eval()
            self.value_function_2.gp_model.eval()
            self.value_function_2.likelihood.eval()
            
            # Zero gradients for actor update
            self.actor_ngd_optimizer.zero_grad()
            self.sac_optimizer.zero_grad()
            boolean_dist, _, _ = self.gp.predict_with_grad(state_norm, torch.tensor(likelihoods).to(device=state_norm.device, dtype=state_norm.dtype), fast_mode=True)
            
            # Get action probabilities
            probs = boolean_dist.probs.reshape(-1, 1)
            probs = torch.cat([1-probs, probs], dim=-1)
            dist = Categorical(probs)
            
            # Get entropy
            entropy = dist.entropy().mean()
            
            # Get log probs for each action in the batch
            log_probs = torch.log(torch.gather(probs, 1, actions) + 1e-10)

            state_pred_prob_norm = torch.cat([state_norm, probs[:, 1:2]], dim=-1).to(dtype=state_norm.dtype)
            
            # Estimate Q-value using the first critic for policy optimization
            q_values_1 = self.value_function.gp_model(state_pred_prob_norm).mean
            q_values_2 = self.value_function_2.gp_model(state_pred_prob_norm).mean
            q_values = torch.min(q_values_1, q_values_2)
            
            # Actor loss: E[α*log(π(a|s)) - Q(s,a)] with importance sampling weights
            actor_loss = (is_weights * (self.alpha * log_probs.flatten() - q_values)).mean()
            
            # Backpropagate actor loss
            actor_loss.backward()
            
            # Step optimizers for actor
            self.actor_ngd_optimizer.step()
            self.sac_optimizer.step()
            timing_stats['actor_update_time'] += time.time() - actor_update_start
            
            # ------------------------ Soft Update Target Networks ------------------------
            target_update_start = time.time()
            with torch.no_grad():
                # Soft update first Q-network target
                for param, target_param in zip(
                    self.value_function.gp_model.parameters(),
                    self.value_function_target.gp_model.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
                
                # Soft update first Q-network likelihood
                for param, target_param in zip(
                    self.value_function.likelihood.parameters(),
                    self.value_function_target.likelihood.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

                # Soft update second Q-network target
                for param, target_param in zip(
                    self.value_function_2.gp_model.parameters(),
                    self.value_function_target_2.gp_model.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

                # Soft update second Q-network likelihood
                for param, target_param in zip(
                    self.value_function_2.likelihood.parameters(),
                    self.value_function_target_2.likelihood.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
                    
                # Calculate inducing point changes
                actor_ip_after = self.gp.gp_model.variational_strategy.inducing_points
                critic_ip_after_1 = self.value_function.gp_model.variational_strategy.inducing_points
                critic_ip_after_2 = self.value_function_2.gp_model.variational_strategy.inducing_points
                actor_ip_change = torch.norm(actor_ip_after - actor_ip_before).item()
                critic_ip_change_1 = torch.norm(critic_ip_after_1 - critic_ip_before_1).item()
                critic_ip_change_2 = torch.norm(critic_ip_after_2 - critic_ip_before_2).item()

            # Set models back to evaluation mode
            self.gp.gp_model.eval()
            self.gp.likelihood.eval()
            timing_stats['target_update_time'] += time.time() - target_update_start
            
            # ------------------------ Update Replay Buffer Priorities ------------------------
            priority_update_start = time.time()
            # Update priorities in the replay buffer using TD errors
            td_errors_np = td_errors.abs().detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors_np)
            timing_stats['priority_update_time'] += time.time() - priority_update_start
            
            # ------------------------ Collect metrics for this iteration ------------------------
            # Aggregate metrics for each iteration
            aggregated_metrics["actor_loss"] += actor_loss.item()
            aggregated_metrics["critic_loss_1"] += critic_loss_1.item()
            aggregated_metrics["critic_loss_2"] += critic_loss_2.item()
            aggregated_metrics["entropy"] += entropy.item()
            aggregated_metrics["mean_reward"] += rewards.mean().item()
            aggregated_metrics["actor_ip_change"] += actor_ip_change
            aggregated_metrics["critic_ip_change_1"] += critic_ip_change_1
            aggregated_metrics["critic_ip_change_2"] += critic_ip_change_2
            aggregated_metrics["mean_td_error"] += td_errors.abs().mean().item()
            
            # Calculate iteration time
            iter_time = time.time() - iter_start_time
            timing_stats['total_iter_time'] += iter_time
            
            # Print progress for this iteration
            if update_iter % max(1, self.num_update // 5) == 0:
                print(f"SAC Update {update_iter+1}/{self.num_update}: "
                      f"Actor Loss={actor_loss.item():.4f}, "
                      f"Q1 Loss={critic_loss_1.item():.4f}, "
                      f"Q2 Loss={critic_loss_2.item():.4f}, "
                      f"Entropy={entropy.item():.4f}, "
                      f"Time={iter_time:.2f}s"
                      f"Actor IP Change={actor_ip_change:.8f}, "
                      f"Critic IP Change 1={critic_ip_change_1:.8f}, "
                      f"Critic IP Change 2={critic_ip_change_2:.8f}, "
                      f"Mean TD Error={td_errors.abs().mean().item():.4f}")
        

        
        # ------------------------ Final processing ------------------------
        # Average metrics over the number of updates
        for key in aggregated_metrics:
            aggregated_metrics[key] /= self.num_update
        
        # Add reward normalization stats to metrics
        aggregated_metrics["reward_mean"] = self.reward_stats['mean']
        aggregated_metrics["reward_std"] = np.sqrt(self.reward_stats['var'])
        aggregated_metrics["normalized_reward_mean"] = normalized_rewards.mean().item()
        aggregated_metrics["normalized_reward_std"] = normalized_rewards.std().item() if len(normalized_rewards) > 1 else 0
        
        # Add hyperparameter values to metrics
        aggregated_metrics['threshold'] = self.gp.likelihood.threshold.item()
        aggregated_metrics['global_scale_offset'] = torch.exp(self.gp.likelihood.log_global_scale_increase).item()
        
        # Calculate and add timing statistics
        total_time = time.time() - update_start_time
        timing_stats['total_time'] = total_time
        aggregated_metrics['timing'] = dict(timing_stats)
        
        # Print comprehensive timing summary
        print(f"\n------ SAC Update Summary ({self.num_update} iterations) ------")
        print(f"Total time: {total_time:.2f}s, Avg iteration: {total_time/self.num_update:.2f}s")
        print(f"Sampling: {timing_stats['sampling_time']:.2f}s ({100*timing_stats['sampling_time']/total_time:.1f}%)")
        print(f"Target Q: {timing_stats['target_q_time']:.2f}s ({100*timing_stats['target_q_time']/total_time:.1f}%)")
        print(f"Q1 update: {timing_stats['q1_update_time']:.2f}s ({100*timing_stats['q1_update_time']/total_time:.1f}%)")
        print(f"Q2 update: {timing_stats['q2_update_time']:.2f}s ({100*timing_stats['q2_update_time']/total_time:.1f}%)")
        print(f"Actor update: {timing_stats['actor_update_time']:.2f}s ({100*timing_stats['actor_update_time']/total_time:.1f}%)")
        print(f"Target update: {timing_stats['target_update_time']:.2f}s ({100*timing_stats['target_update_time']/total_time:.1f}%)")
        print(f"Priority update: {timing_stats['priority_update_time']:.2f}s ({100*timing_stats['priority_update_time']/total_time:.1f}%)")
        print(f"Actor loss: {aggregated_metrics['actor_loss']:.6f}")
        print(f"Q1 loss: {aggregated_metrics['critic_loss_1']:.6f}")
        print(f"Q2 loss: {aggregated_metrics['critic_loss_2']:.6f}")
        print(f"Current threshold: {aggregated_metrics['threshold']:.4f}")
        print(f"Global Scale Offset: {aggregated_metrics['global_scale_offset']:.4f}")
        print(f"Reward stats: mean={self.reward_stats['mean']:.4f}, "
              f"std={np.sqrt(self.reward_stats['var']):.4f}, count={self.reward_stats['count']}")
        print("----------------------------------------------")
        
        self.gp.gp_model.eval()
        self.gp.likelihood.eval()
        self.value_function.gp_model.eval()
        self.value_function.likelihood.eval()
        self.value_function_2.gp_model.eval() 
        self.value_function_2.likelihood.eval()
        self.value_function_target.gp_model.eval()
        self.value_function_target.likelihood.eval()
        self.value_function_target_2.gp_model.eval()
        self.value_function_target_2.likelihood.eval()
        # Return the aggregated metrics
        return aggregated_metrics
        
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
        
    def _update_reward_stats(self, rewards: torch.Tensor) -> None:
        """
        Update running statistics for reward normalization using Welford's algorithm.
        
        Args:
            rewards: Batch of rewards to update statistics with
        """
        if not self.normalize_rewards:
            return
            
        batch_size = rewards.shape[0]
        
        if batch_size == 0:
            return
            
        # Get statistics before update for logging
        old_mean = self.reward_stats['mean']
        
        # Update count
        new_count = self.reward_stats['count'] + batch_size
        
        # Update mean
        batch_mean = rewards.mean().item()
        delta = batch_mean - self.reward_stats['mean']
        new_mean = self.reward_stats['mean'] + delta * batch_size / new_count
        
        # Update variance using Welford's online algorithm
        batch_var = rewards.var(unbiased=False).item() if batch_size > 1 else 0.0
        m_a = self.reward_stats['var'] * self.reward_stats['count']
        m_b = batch_var * batch_size
        M2 = m_a + m_b + delta**2 * self.reward_stats['count'] * batch_size / new_count
        new_var = M2 / new_count
        
        # Store updated statistics
        self.reward_stats['count'] = new_count
        self.reward_stats['mean'] = new_mean
        self.reward_stats['var'] = new_var
        
        # Also track exponentially weighted statistics for smoother updates
        if self.reward_stats['count'] <= 1:
            self.reward_stats['running_mean'] = batch_mean
            self.reward_stats['running_var'] = batch_var if batch_var > 0 else 1.0
        else:
            momentum = 0.99 if self.reward_stats['count'] > 1000 else 0.9
            self.reward_stats['running_mean'] = momentum * self.reward_stats['running_mean'] + (1 - momentum) * batch_mean
            self.reward_stats['running_var'] = momentum * self.reward_stats['running_var'] + (1 - momentum) * batch_var
            
        print(f"Reward stats updated: mean {old_mean:.4f} -> {new_mean:.4f}, "
              f"std: {np.sqrt(self.reward_stats['var']):.4f}")
    
    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize rewards using tracked statistics.
        
        Args:
            rewards: Rewards to normalize
        
        Returns:
            Normalized rewards
        """
        if not self.normalize_rewards or self.reward_stats['count'] < 100:  
            # Don't normalize until we have enough samples
            return rewards
            
        std = torch.sqrt(torch.tensor(self.reward_stats['var'] + self.reward_norm_eps, device=rewards.device))
        mean = torch.tensor(self.reward_stats['mean'], device=rewards.device)
        
        return (rewards - mean) / std
        
    def set_reward_normalization(self, enabled: bool = True) -> None:
        """
        Enable or disable reward normalization.
        
        Args:
            enabled: Whether to normalize rewards
        """
        self.normalize_rewards = enabled
        print(f"Reward normalization {'enabled' if enabled else 'disabled'}")
        
    def reset_reward_stats(self) -> None:
        """Reset the reward normalization statistics."""
        self.reward_stats = {
            'mean': 0.0,
            'var': 1.0,
            'count': 0,
            'running_mean': 0.0,
            'running_var': 0.0
        }
        print("Reward statistics have been reset")