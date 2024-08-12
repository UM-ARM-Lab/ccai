import torch
from torch import nn

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

    def sample(self, N, H=None, start=None, goal=None, constraints=None, past=None):
        # B, N, _ = constraints.shape
        if constraints is not None:
            constraints = constraints.reshape(constraints.shape[0], -1, constraints.shape[-1])
            context = constraints
            # context = torch.cat((start.unsqueeze(1).repeat(1, N, 1),
            #                     goal.unsqueeze(1).repeat(1, N, 1),
            #                     constraints), dim=-1)
        else:
            context = None
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

        samples = self.diffusion_model.sample(N=N, H=H, context=context, condition=condition)  # .reshape(-1, H#,
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

    def __init__(self, horizon, dx, du, context_dim, hidden_dim=32):
        super().__init__(horizon, dx, du, context_dim, hidden_dim=hidden_dim)

    def sample(self, N, H=None, start=None, goal=None, constraints=None, past=None):
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
        trajectories, likelihood = self._sample(context=context, condition=condition, mask=mask, H=H)
        return trajectories, context, likelihood


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


class TrajectorySampler(nn.Module):

    def __init__(self, T, dx, du, context_dim, type='nf', dynamics=None, problem=None, timesteps=50, hidden_dim=64,
                 constrain=False, unconditional=False, generate_context=False, score_model='conv_unet',
                 latent_diffusion=False, vae=None, inits_noise=None, noise_noise=None, guided=False):
        super().__init__()
        self.T = T
        self.dx = dx
        self.du = du
        self.context_dim = context_dim
        self.type = type
        assert type in ['nf', 'latent_diffusion', 'diffusion', 'cnf']
        if type == 'nf':
            self.model = TrajectoryFlowModel(T, dx, du, context_dim, dynamics)
        elif type == 'cnf':
            self.model = TrajectoryCNFModel(T, dx, du, context_dim, hidden_dim=hidden_dim)
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
        self.send_norm_constants_to_submodels()

    def set_norm_constants(self, x_mean, x_std):
        self.x_mean.data = x_mean.to(device=self.x_mean.device, dtype=self.x_mean.dtype)
        self.x_std.data = x_std.to(device=self.x_std.device, dtype=self.x_std.dtype)

    def send_norm_constants_to_submodels(self):
        self.model.set_norm_constants(self.x_mean, self.x_std)

    def sample(self, N, H=10, start=None, goal=None, constraints=None, past=None):
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

        samples = self.model.sample(N, H, norm_start, goal, constraints, norm_past)
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