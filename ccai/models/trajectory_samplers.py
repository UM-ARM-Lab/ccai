from typing import Dict
import gpytorch
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from tqdm import tqdm

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

    def to(self, device: torch.device) -> 'TrajectorySampler':
        """
        Move the model to the specified device, including the GP model.
        
        Args:
            device: Device to move the model to
            
        Returns:
            Self for method chaining
        """
        # Move the main model
        self.model = self.model.to(device)
        
        # Move normalization constants
        self.x_mean = self.x_mean.to(device)
        self.x_std = self.x_std.to(device)
        
        # Move GP model and associated components if present
        if hasattr(self, 'rl_adjustment') and self.rl_adjustment:
            self.id_threshold = self.id_threshold.to(device)
            self.gp = self.gp.to(device)
            self.value_function = self.value_function.to(device)
            
            # Update optimizer with new parameter references
            self.ppo_optimizer = torch.optim.Adam([
                {'params': self.gp.gp_model.parameters()},
                {'params': self.gp.likelihood.parameters()},
                {'params': [self.id_threshold], 'lr': .1},
                {'params': self.value_function.parameters(), 'lr': 3e-4}
            ])
        
        # Update submodels with new device
        self.send_norm_constants_to_submodels()
        
        return self

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
    
    def sample(self, N, H=None, start=None, goal=None, constraints=None, past=None, project=False, no_grad=True, skip_likelihood=False, context_for_likelihood=True):
        # B, N, _ = constraints.shape
        context = self.construct_context(constraints)

        condition = self.construct_condition(H, start, goal, past)

        if project:
            samples, samples_0, (all_losses, all_samples, all_likelihoods) = self.diffusion_model.project(N=N, H=H, context=context, condition=condition)
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
            self.id_threshold = torch.nn.Parameter(torch.tensor([initial_threshold]))

            self.gp = LikelihoodResidualGP(dx, num_inducing=32)

            
            # Replace MLP value function with a GP
            self.value_function = LikelihoodResidualGP(dx, num_inducing=32, 
                                                      kernel_type="matern32",
                                                      use_ard=True)
            
            # Update PPO optimizer to include the new value function GP parameters
            self.ppo_optimizer = Adam([
                {'params': self.gp.gp_model.parameters(), 'lr': 1e-2},
                {'params': self.gp.likelihood.parameters(), 'lr': 1e-2},
                {'params': [self.id_threshold], 'lr': 1e-2},
                {'params': self.value_function.gp_model.parameters(), 'lr': 1e-3},
                {'params': self.value_function.likelihood.parameters(), 'lr': 1e-3}
            ])
            
            # PPO hyperparameters
            self.clip_param = 0.2
            self.value_loss_coef = 0.5
            self.entropy_coef = 0.01
            self.max_grad_norm = 0.5
            self.ppo_epochs = 10
            self.batch_size = 32
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

        self.send_norm_constants_to_submodels()

    def to(self, device: torch.device) -> 'TrajectorySampler':
        """
        Move the model to the specified device, including the GP model.
        
        Args:
            device: Device to move the model to
            
        Returns:
            Self for method chaining
        """
        # Move the main model
        self.model = self.model.to(device)
        
        # Move normalization constants
        self.x_mean = self.x_mean.to(device)
        self.x_std = self.x_std.to(device)
        
        # Move GP model and associated components if present
        if self.rl_adjustment:
            self.id_threshold.data = self.id_threshold.to(device)
            self.gp = self.gp.to(device)
            self.value_function = self.value_function.to(device)
            
            # Update optimizer with new parameter references
            self.ppo_optimizer = torch.optim.Adam([
                {'params': self.gp.gp_model.parameters()},
                {'params': self.gp.likelihood.parameters()},
                {'params': [self.id_threshold], 'lr': 1e-2},
                {'params': self.value_function.gp_model.parameters(), 'lr': 1e-3},
                {'params': self.value_function.likelihood.parameters(), 'lr': 1e-3}
            ])
        
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

    def check_id(self, state, N, threshold=None, return_samples=False):

        start = state[:4 * 3 + 3]
        start_sine_cosine = self.convert_yaw_to_sine_cosine(start)
        samples, _, likelihood = self.sample(N=N, H=self.T, start=start_sine_cosine.reshape(1, -1),
                constraints=torch.ones(N, 3).to(device=state.device))
        likelihood = likelihood.reshape(N).mean().item()
        # samples = samples.cpu().numpy()
        if self.rl_adjustment:
            likelihood_residual_mean, likelihood_residual_variance = self.gp.predict(start_sine_cosine)
            likelihood += likelihood_residual_mean
            print('Likelihood residual:', likelihood_residual_mean)
        print('Likelihood:', likelihood)
        if threshold is None:
            threshold = self.id_threshold
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
            samples, samples_0, (all_losses, all_samples, all_likelihoods) = self.model.sample(N, H, norm_start, goal, constraints, norm_past, project, context_for_likelihood=context_for_likelihood)
        else:odel.sample(N, H, norm_start, goal, constraints, norm_past, project, no_grad=no_grad, skip_likelihood=skip_likelihood, context_for_likelihood=context_for_likelihood)
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
        state_batch = torch.stack([self.convert_yaw_to_sine_cosine(state) for state in states])
        
        # Sample trajectories to get base likelihoods
        # We need to do this part without gradients because it's expensive and not directly differentiable
        with torch.no_grad():
            batch_likelihoods = []
            _, _, likelihood = self.sample(
                N=5*state_batch.shape[0],  # Small number for efficiency
                H=self.T,
                start=state_batch.repeat_interleave(5, 0),
                constraints=torch.ones(5*state_batch.shape[0], 3).to(device=state_batch.device)
            )
            base_likelihoods = likelihood.reshape(-1, 5).mean(dim=1)
        
        # Create a new tensor that requires gradients - this is a trick to incorporate 
        # the base likelihoods into our computational graph
        base_likelihoods = base_likelihoods.detach().requires_grad_(False)
        
        # Get GP predictions WITH gradients using the new method
        likelihood_residuals = self.gp.predict_with_grad(state_batch)[0]
        
        # Combine base likelihoods with residuals
        adjusted_likelihoods = base_likelihoods + likelihood_residuals.squeeze()
        
        # Calculate policy distribution
        probs = torch.sigmoid(adjusted_likelihoods - self.id_threshold)
        dist = torch.distributions.Bernoulli(probs)
        
        # Get log probabilities and entropy
        action_log_probs = dist.log_prob(actions.float())
        dist_entropy = dist.entropy().mean()
        
        # Get value predictions using GP
        state_values = self.value_function.gp_model(state_batch)[0]
        
        return action_log_probs, state_values, dist_entropy

    def store_transition(self, state, action: int, likelihood: torch.Tensor, next_state, reward: float, done: bool = False) -> None:
        """
        Store a transition in the PPO memory.
        
        Args:
            state: Current state
            action: Action taken (0 or 1)
            next_state: Resulting next state
            reward: Reward received
            done: Whether the episode is done
        """
        if not self.rl_adjustment:
            return
            
        # Process state and evaluate policy
        state_tensor = torch.tensor(self.convert_yaw_to_sine_cosine(state), 
                                    dtype=torch.float32, device=self.id_threshold.device).unsqueeze(0)
        
        # Get value estimate using GP
        with torch.no_grad():
            value, _ = self.value_function.predict(state_tensor)
            value = value.squeeze().item()
        
        # Get action probability using sigmoid and the threshold
        prob = torch.sigmoid(likelihood - self.id_threshold)
        # Calculate log probability of the action
        action_tensor = torch.tensor(action, dtype=torch.float, device=self.id_threshold.device)
        log_prob = torch.log(prob + 1e-8)
        log_prob = log_prob.item()  # Convert to scalar
        
        # Store in memory
        self.ppo_states.append(state)
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
        
        device = self.id_threshold.device
        
        # Get final value for bootstrapping
        if self.ppo_dones[-1]:
            last_value = 0.0
        else:
            # Use the last state's value as bootstrap
            final_state = torch.tensor(
                self.convert_yaw_to_sine_cosine(self.ppo_states[-1]),
                dtype=torch.float32, 
                device=device
            ).unsqueeze(0)
            
            with torch.no_grad():
                last_value = self.value_function(final_state).squeeze().item()
        
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
            self.ppo_returns.insert(0, delta + self.ppo_values[step])
            self.ppo_advantages.insert(0, current_advantage)
            
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
        Uses data stored in the PPO memory.
        
        Returns:
            Dictionary with loss information
        """

        self.value_function.train()
        if not self.ppo_returns or len(self.ppo_states) == 0:
            return {'error': 'No data available for update'}
            
        device = self.id_threshold.device
        
        # Convert memory to tensors on the correct device
        states = torch.stack(self.ppo_states).to(dtype=torch.float32, device=device)
        actions = torch.tensor(self.ppo_actions, dtype=torch.long, device=device).unsqueeze(-1)
        old_log_probs = torch.tensor(self.ppo_log_probs, dtype=torch.float32, device=device)
        returns = torch.tensor(self.ppo_returns, dtype=torch.float32, device=device)
        advantages = torch.tensor(self.ppo_advantages, dtype=torch.float32, device=device)
        
        # Normalize advantages for more stable training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Metrics to track during training
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clip_fraction = 0
        num_updates = 0
        objective_function = gpytorch.mlls.PredictiveLogLikelihood(self.value_function.likelihood, self.value_function.gp_model, num_data=batch_states.shape[0])

        # Multiple epochs of PPO update
        for _ in tqdm(range(self.ppo_epochs)):
            # Create data loader for mini-batches
            batch_indices = torch.randperm(len(states))
            
            # Process mini-batches
            for start_idx in tqdm(range(0, len(states), self.batch_size)):
                num_updates += 1
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_idx = batch_indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                
                # Evaluate current policy on batch
                log_probs, values, entropy = self.evaluate_actions(batch_states, batch_actions)
                
                # PPO policy loss calculation
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                
                # Calculate policy loss (negative because we're minimizing)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value function loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Calculate combined loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Perform gradient step
                self.ppo_optimizer.zero_grad()
                loss.backward()
                
                # Optional: gradient clipping for stability
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.gp.gp_model.parameters()) + 
                        list(self.gp.likelihood.parameters()) +
                        [self.id_threshold] +
                        list(self.value_function.parameters()),
                        self.max_grad_norm
                    )
                
                # Update parameters
                self.ppo_optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                
                # Calculate clipping fraction for monitoring
                clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                total_clip_fraction += clip_fraction
        
        # Clear memory after update
        self.clear_memory()
        self.value_function.eval()
        
        # Return average metrics
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'clip_fraction': total_clip_fraction / num_updates,
            'threshold': self.id_threshold.item()
        }