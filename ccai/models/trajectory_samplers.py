from typing import Dict
import numpy as np
import gpytorch
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from gpytorch.optim import NGD

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
            # We no longer need the threshold as a separate parameter
            # Instead, we'll use the bias parameter in our custom likelihood
            
            # Create the actor GP with our custom likelihood
            actor_likelihood = SigmoidBernoulliLikelihood(
                initial_bias=-initial_threshold,
                initial_temp=.25,
                num_samples=15
            )
            self.gp = LikelihoodResidualGP(
                dx, 
                num_inducing=32,
                likelihood=actor_likelihood,
                use_whitening=True  # Use whitening for better numerical stability
            )
            
            # Value function GP still uses Gaussian likelihood
            self.value_function = LikelihoodResidualGP(
                dx, 
                num_inducing=32, 
                kernel_type="matern32",
                use_ard=True,
                use_whitening=True  # Use whitening for better numerical stability
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
            
            # PPO hyperparameters
            self.clip_param = 0.2
            self.value_loss_coef = 0.5
            self.entropy_coef = 0.01
            self.max_grad_norm = 0.5
            self.ppo_epochs = 32
            self.batch_size = 16
            self.gamma = 0.99
            self.gae_lambda = 0.95
            
            # PPO memory
            self.ppo_states = []
            self.ppo_actions = []
            self.ppo_log_probs = []
            self.ppo_rewards = []
            self.ppo_values = []
            self.ppo_dones = []
            self.ppo_returns = []
            self.ppo_advantages = []

            # Use NGD for variational parameters (typically needs higher learning rate)
            self.actor_ngd_optimizer = NGD(actor_variational_params, num_data=self.batch_size, lr=0.1)
            self.value_ngd_optimizer = NGD(value_variational_params, num_data=self.batch_size, lr=0.1)
            
            # Use Adam for hyperparameters and likelihood parameters
            self.ppo_optimizer = Adam([
                {'params': actor_hyper_params, 'lr': 1e-2},
                {'params': self.gp.likelihood.parameters(), 'lr': 1e-2},
                {'params': value_hyper_params, 'lr': 1e-2},
                {'params': self.value_function.likelihood.parameters(), 'lr': 1e-2}
            ])

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
                residual, _ = self.gp.predict(start_sine_cosine.unsqueeze(0), fast_mode=True)
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
        Store a transition in the PPO memory.
        """
        if not self.rl_adjustment:
            return
            
        # Process state and evaluate policy
        state_tensor = torch.tensor(self.convert_yaw_to_sine_cosine(state), 
                                    dtype=torch.float32, device=next(self.parameters()).device).unsqueeze(0)
        
        # Get value estimate using GP. Use predict
        value_output_mean, _ = self.value_function.predict(state_tensor, fast_mode=True)
        value = value_output_mean.squeeze().item()
        
        # Get action probability using the GP and likelihood
        with torch.no_grad():
            output_mean, _ = self.gp.predict(state_tensor, fast_mode=True)
            combined_feature = likelihood + output_mean
            bernoulli_dist = self.gp.likelihood(combined_feature)
            log_prob = bernoulli_dist.log_prob(torch.tensor([action], dtype=torch.float, 
                                                            device=next(self.parameters()).device)).item()
        
        # Store in memory
        self.ppo_states.append(state_tensor)
        self.ppo_actions.append(action)
        self.ppo_log_probs.append(log_prob)
        self.ppo_rewards.append(reward)
        self.ppo_values.append(value)
        self.ppo_dones.append(done)
    
    def compute_returns_and_advantages(self) -> None:
        """
        Compute returns and advantages from stored transitions using GAE.
        This should be called before ppo_update when enough data has been collected.
        """
        if not self.ppo_states:
            return  # No data to process
        
        device = self.device
        
        # Get final value for bootstrapping
        if self.ppo_dones[-1]:
            last_value = 0.0
        else:
            # Use the last state's value as bootstrap
            final_state = torch.tensor(
                self.ppo_states[-1],
                dtype=torch.float32, 
                device=device
            ).unsqueeze(0)
            
            # Use predict to get value
            last_value = self.value_function.predict(final_state)[0].squeeze().item()
        
        # Calculate returns and advantages using GAE
        self.ppo_returns = []
        self.ppo_advantages = []
        
        next_return = last_value
        next_advantage = 0
        
        # Process transitions in reverse for easier bootstrapping
        for step in reversed(range(len(self.ppo_rewards))):
            # Calculate returns and advantages
            if self.ppo_dones[step]:
                # If terminal state, no bootstrapping
                next_return = 0.0
                next_advantage = 0.0
            
            # GAE calculations
            delta = self.ppo_rewards[step] + self.gamma * next_return * (1.0 - self.ppo_dones[step]) - self.ppo_values[step]
            current_advantage = delta + self.gamma * self.gae_lambda * next_advantage * (1.0 - self.ppo_dones[step])
            
            # Store results
            self.ppo_returns.insert(0, (delta + self.ppo_values[step]).item())
            self.ppo_advantages.insert(0, current_advantage.item())
            
            # Update for next iteration
            next_return = self.ppo_returns[0]  
            next_advantage = current_advantage
    
    def clear_memory(self) -> None:
        """Clear all stored PPO data."""
        self.ppo_states = []
        self.ppo_actions = []
        self.ppo_log_probs = []
        self.ppo_rewards = []
        self.ppo_values = []
        self.ppo_dones = []
        self.ppo_returns = []
        self.ppo_advantages = []
    
    def get_memory_size(self) -> int:
        """Return the current size of stored transitions."""
        return len(self.ppo_states)
        
    def ppo_update(self) -> Dict[str, float]:
        """
        Update the policy parameters (GP and threshold) and value function using PPO.
        Includes detailed timing statistics for performance analysis.
        
        Returns:
            Dictionary with loss information and timing statistics
        """
        self.gp.gp_model.eval()
        self.gp.likelihood.eval()
        self.value_function.gp_model.train()
        self.value_function.likelihood.train()
        update_start_time = time.time()
        timing_stats = collections.defaultdict(float)

        # Track more detailed timing metrics
        timing_stats['ngd_time'] = 0.0
        timing_stats['adam_time'] = 0.0
        timing_stats['forward_pass_time'] = 0.0
        timing_stats['loss_computation_time'] = 0.0
        timing_stats['backward_time'] = 0.0
        timing_stats['optimizer_overhead'] = 0.0
        timing_stats['memory_operations'] = 0.0
        timing_stats['policy_evaluation'] = 0.0
        timing_stats['gpu_sync_time'] = 0.0
        timing_stats['obj_creation_time'] = 0.0
        timing_stats['data_preparation'] = 0.0
        timing_stats['batch_data_prep'] = 0.0


        # Metrics to track during training
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clip_fraction = 0
        num_updates = 0
        
        # Create per-epoch timing stats
        epoch_timing_stats = []
        
        # Multiple epochs of PPO update
        for epoch in range(self.ppo_epochs):
            epoch_start = time.time()
            
            # Create epoch-specific timing stats
            epoch_timings = collections.defaultdict(float)
            
            # Track losses for this epoch
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy = 0
            epoch_clip_fraction = 0
            epoch_updates = 0
            
            # Data preparation timing
            data_prep_start = time.time()
            device = self.device
            
            # Convert memory to tensors but keep on CPU initially to avoid graph attachment
            # IMPORTANT: Convert all to numpy first to completely detach from any previous computation
            states_np = torch.cat([i.clone().detach() for i in self.ppo_states]).cpu().numpy()
            actions_np = np.array(self.ppo_actions).reshape(-1, 1)
            old_log_probs_np = np.array(self.ppo_log_probs)
            returns_np = np.array(self.ppo_returns)
            advantages_np = np.array(self.ppo_advantages)
            
            # Normalize advantages in numpy to avoid graph issues
            advantages_np = (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)
            timing_stats['data_preparation'] = time.time() - data_prep_start
            
            # Process mini-batches
            batch_indices = np.random.permutation(len(states_np))
            
            for start_idx in range(0, len(states_np), self.batch_size):
                batch_start = time.time()
                num_updates += 1
                epoch_updates += 1
                end_idx = min(start_idx + self.batch_size, len(states_np))
                batch_idx = batch_indices[start_idx:end_idx]
                
                # Get batch data and move to device here, creating fresh tensors each time
                batch_data_start = time.time()
                batch_states = torch.tensor(states_np[batch_idx], dtype=torch.float32, device=device)
                batch_actions = torch.tensor(actions_np[batch_idx], dtype=torch.long, device=device)
                batch_old_log_probs = torch.tensor(old_log_probs_np[batch_idx], dtype=torch.float32, device=device)
                batch_returns = torch.tensor(returns_np[batch_idx], dtype=torch.float32, device=device)
                batch_advantages = torch.tensor(advantages_np[batch_idx], dtype=torch.float32, device=device)
                timing_stats['batch_data_prep'] += time.time() - batch_data_start
                
                pre_obj_start = time.time()
                batch_objective = gpytorch.mlls.PredictiveLogLikelihood(
                    self.value_function.likelihood, 
                    self.value_function.gp_model, 
                    num_data=batch_returns.size(0)  # Use actual batch size
                )
                timing_stats['obj_creation_time'] += time.time() - pre_obj_start
                # Zero all gradients BEFORE forward pass to ensure clean computation
                self.actor_ngd_optimizer.zero_grad()
                self.value_ngd_optimizer.zero_grad()
                self.ppo_optimizer.zero_grad()
                
                # Evaluate current policy on batch - separate function to ensure clean graph
                policy_eval_start = time.time()
                log_probs, values, entropy = self.evaluate_actions(batch_states, batch_actions)
                timing_stats['policy_evaluation'] += time.time() - policy_eval_start
                
                # PPO policy loss calculation
                loss_calc_start = time.time()
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                
                # Calculate policy loss (negative because we're minimizing)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value function loss
                value_loss = -batch_objective(values, batch_returns)
                

                # Calculate combined loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                timing_stats['loss_computation_time'] += time.time() - loss_calc_start
                
                # Backpropagate
                backprop_start = time.time()
                loss.backward()
                timing_stats['backward_time'] += time.time() - backprop_start
                
                # GPU sync point - measure sync time if using GPU
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    sync_time = time.time() - (backprop_start + timing_stats['backward_time'])
                    timing_stats['gpu_sync_time'] += sync_time
                    epoch_timings['gpu_sync_time'] += sync_time
                
                # Measure NGD optimizer step time
                ngd_start = time.time()
                self.actor_ngd_optimizer.step()
                self.value_ngd_optimizer.step()
                ngd_time = time.time() - ngd_start
                timing_stats['ngd_time'] += ngd_time
                epoch_timings['ngd_time'] += ngd_time
                
                # Measure Adam optimizer step time
                adam_start = time.time()
                self.ppo_optimizer.step()
                adam_time = time.time() - adam_start
                timing_stats['adam_time'] += adam_time
                epoch_timings['adam_time'] += adam_time
                
                # Measure optimizer overhead (zero_grad)
                opt_overhead_start = time.time()
                self.actor_ngd_optimizer.zero_grad()
                self.value_ngd_optimizer.zero_grad()
                self.ppo_optimizer.zero_grad()
                opt_overhead = time.time() - opt_overhead_start
                timing_stats['optimizer_overhead'] += opt_overhead
                epoch_timings['optimizer_overhead'] += opt_overhead
                
                # Track metrics - detach tensors to avoid graph retention
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                
                # Calculate clipping fraction for monitoring
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                total_clip_fraction += clip_fraction
                
                # Also track for this epoch
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy.item()
                epoch_clip_fraction += clip_fraction
                
                # Clear computation graph
                del log_probs, values, entropy, ratio, surr1, surr2
                del policy_loss, value_loss, loss
                
                timing_stats['batch_total'] += time.time() - batch_start
                epoch_timings['batch_total'] += time.time() - batch_start
            
            # Calculate epoch averages
            if epoch_updates > 0:
                epoch_avg_policy_loss = epoch_policy_loss / epoch_updates
                epoch_avg_value_loss = epoch_value_loss / epoch_updates
                epoch_avg_entropy = epoch_entropy / epoch_updates
                epoch_avg_clip = epoch_clip_fraction / epoch_updates
                
                # Calculate average times per batch for this epoch
                for key in epoch_timings:
                    epoch_timings[key] /= max(1, epoch_updates)
            else:
                epoch_avg_policy_loss = epoch_avg_value_loss = epoch_avg_entropy = epoch_avg_clip = 0
            
            # Store the epoch timing stats for final summary
            epoch_timings['epoch_number'] = epoch + 1
            epoch_timings['total_time'] = time.time() - epoch_start
            epoch_timings['policy_loss'] = epoch_avg_policy_loss
            epoch_timings['value_loss'] = epoch_avg_value_loss
            epoch_timings['entropy'] = epoch_avg_entropy
            epoch_timings['clip_fraction'] = epoch_avg_clip
            epoch_timing_stats.append(dict(epoch_timings))
            
            # Calculate and store the actual threshold value from bias for easier interpretation
            current_threshold = -self.gp.likelihood.bias.item()
            
            timing_stats['epoch_time'] += time.time() - epoch_start
            epoch_time = time.time() - epoch_start
            
            # Print detailed summary for this epoch
            print(f"\n==== Epoch {epoch+1}/{self.ppo_epochs} Summary ====")
            print(f"Time: {epoch_time:.4f}s (NGD: {epoch_timings['ngd_time']:.4f}s, Adam: {epoch_timings['adam_time']:.4f}s)")
            print(f"Data preparation: {timing_stats['data_preparation']:.4f}s ({timing_stats['data_preparation']/epoch_time*100:.1f}%)")
            print(f"Batch preparation: {timing_stats['batch_data_prep']:.4f}s ({timing_stats['batch_data_prep']/epoch_time*100:.1f}%)")
            print(f"Objective creation: {timing_stats['obj_creation_time']:.4f}s ({timing_stats['obj_creation_time']/epoch_time*100:.1f}%)")
            print(f"Policy evaluation: {epoch_timings['policy_evaluation']:.4f}s ({epoch_timings['policy_evaluation']/epoch_time*100:.1f}%)")
            print(f"Forward pass: {epoch_timings['forward_pass_time']:.4f}s ({epoch_timings['forward_pass_time']/epoch_time*100:.1f}%)")
            print(f"Loss computation: {epoch_timings['loss_computation_time']:.4f}s ({epoch_timings['loss_computation_time']/epoch_time*100:.1f}%)")
            print(f"Backward pass: {epoch_timings['backward_time']:.4f}s ({epoch_timings['backward_time']/epoch_time*100:.1f}%)")
            if 'gpu_sync_time' in epoch_timings:
                print(f"GPU sync: {epoch_timings['gpu_sync_time']:.4f}s ({epoch_timings['gpu_sync_time']/epoch_time*100:.1f}%)")
            print(f"Optimizer steps: {epoch_timings['ngd_time'] + epoch_timings['adam_time']:.4f}s ({(epoch_timings['ngd_time'] + epoch_timings['adam_time'])/epoch_time*100:.1f}%)")
            print(f"Memory operations: {epoch_timings['memory_operations']:.4f}s ({epoch_timings['memory_operations']/epoch_time*100:.1f}%)")
            print(f"Avg batch time: {epoch_timings['batch_total']:.4f}s/batch")
            print(f"Policy Loss: {epoch_avg_policy_loss:.6f}")
            print(f"Value Loss: {epoch_avg_value_loss:.6f}")
            print(f"Entropy: {epoch_avg_entropy:.6f}")
            print(f"Clip Fraction: {epoch_avg_clip:.4f}")
            print(f"Current Threshold: {current_threshold:.4f}")
            print("=======================================")
            
            # Force garbage collection between epochs to free memory
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Clear memory after update
        memory_clear_start = time.time()
        self.clear_memory()
        timing_stats['memory_clear'] = time.time() - memory_clear_start
        
        # Calculate average times
        total_update_time = time.time() - update_start_time
        avg_epoch_time = timing_stats['epoch_time'] / max(1, self.ppo_epochs)
        avg_batch_time = timing_stats['batch_total'] / max(1, num_updates)
        
        # Compute time percentages
        time_percentages = {
            'forward_pass': timing_stats['forward_pass_time'] / total_update_time * 100,
            'loss_computation': timing_stats['loss_computation_time'] / total_update_time * 100,
            'backward': timing_stats['backward_time'] / total_update_time * 100,
            'ngd': timing_stats['ngd_time'] / total_update_time * 100,
            'adam': timing_stats['adam_time'] / total_update_time * 100,
            'optimizer_overhead': timing_stats['optimizer_overhead'] / total_update_time * 100,
            'memory_operations': timing_stats['memory_operations'] / total_update_time * 100,
            'data_preparation': timing_stats['data_preparation'] / total_update_time * 100,
            'memory_clear': timing_stats['memory_clear'] / total_update_time * 100
        }
        if 'gpu_sync_time' in timing_stats:
            time_percentages['gpu_sync'] = timing_stats['gpu_sync_time'] / total_update_time * 100
        
        # Log comprehensive timing statistics
        print("\n============== PPO UPDATE COMPREHENSIVE TIMING SUMMARY ==============")
        print(f"Total update time: {total_update_time:.4f}s for {self.ppo_epochs} epochs ({num_updates} updates)")
        print(f"Data preparation: {timing_stats['data_preparation']:.4f}s ({time_percentages['data_preparation']:.1f}%)")
        print(f"Average epoch time: {avg_epoch_time:.4f}s")
        print(f"Average batch time: {avg_batch_time:.4f}s")
        
        print("\nComputation breakdown (total time / percentage):")
        print(f"  Forward pass:      {timing_stats['forward_pass_time']:.4f}s ({time_percentages['forward_pass']:.1f}%)")
        print(f"  Loss computation:  {timing_stats['loss_computation_time']:.4f}s ({time_percentages['loss_computation']:.1f}%)")
        print(f"  Backward pass:     {timing_stats['backward_time']:.4f}s ({time_percentages['backward']:.1f}%)")
        if 'gpu_sync_time' in timing_stats:
            print(f"  GPU sync:          {timing_stats['gpu_sync_time']:.4f}s ({time_percentages['gpu_sync']:.1f}%)")
        print(f"  NGD optimization:  {timing_stats['ngd_time']:.4f}s ({time_percentages['ngd']:.1f}%)")
        print(f"  Adam optimization: {timing_stats['adam_time']:.4f}s ({time_percentages['adam']:.1f}%)")
        print(f"  Optimizer overhead:{timing_stats['optimizer_overhead']:.4f}s ({time_percentages['optimizer_overhead']:.1f}%)")
        print(f"  Memory operations: {timing_stats['memory_operations']:.4f}s ({time_percentages['memory_operations']:.1f}%)")
        print(f"  Memory cleanup:    {timing_stats['memory_clear']:.4f}s ({time_percentages['memory_clear']:.1f}%)")
        
        print("\nEpoch-by-epoch timing (seconds):")
        headers = ["Epoch", "Total", "Forward", "Loss", "Backward", "NGD", "Adam", "Mem-Ops", "Batch Avg"]
        row_format = "{:>6} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}"
        print(" ".join(headers))
        for epoch_stats in epoch_timing_stats:
            print(row_format.format(
                epoch_stats['epoch_number'],
                epoch_stats['total_time'],
                epoch_stats['forward_pass_time'] * epoch_updates,
                epoch_stats['loss_computation_time'] * epoch_updates,
                epoch_stats['backward_time'] * epoch_updates,
                epoch_stats['ngd_time'] * epoch_updates,
                epoch_stats['adam_time'] * epoch_updates,
                epoch_stats['memory_operations'] * epoch_updates,
                epoch_stats['batch_total']
            ))
        
        print("\nPerformance metrics:")
        print(f"  Updates per second: {num_updates / total_update_time:.2f}")
        print(f"  Samples per second: {num_updates * self.batch_size / total_update_time:.2f}")
        print(f"  NGD/Adam ratio: {timing_stats['ngd_time'] / max(timing_stats['adam_time'], 1e-6):.2f}")
        
        print("=================================================================\n")
        
        # Add the epoch timing stats to the return dict
        timing_stats['epoch_stats'] = epoch_timing_stats
        timing_stats['time_percentages'] = time_percentages
        
        # Return metrics with enhanced timing info
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'clip_fraction': total_clip_fraction / num_updates,
            'threshold': -self.gp.likelihood.bias.squeeze().item(),
            'timing': dict(timing_stats),
            'performance': {
                'updates_per_second': num_updates / total_update_time,
                'samples_per_second': num_updates * self.batch_size / total_update_time,
                'total_time': total_update_time,
                'ngd_adam_ratio': timing_stats['ngd_time'] / max(timing_stats['adam_time'], 1e-6)
            }
        }

    def adjust_ngd_learning_rate(self, actor_lr: float = 0.1, value_lr: float = 0.1) -> None:
        """
        Adjust the learning rates for Natural Gradient Descent optimizers.
        
        Args:
            actor_lr: Learning rate for actor GP optimizer
            value_lr: Learning rate for value function GP optimizer
        """
        for param_group in self.actor_ngd_optimizer.param_groups:
            param_group['lr'] = actor_lr
            
        for param_group in self.value_ngd_optimizer.param_groups:
            param_group['lr'] = value_lr
            
        print(f"NGD learning rates adjusted: Actor={actor_lr}, Value={value_lr}")