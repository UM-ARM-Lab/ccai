import torch
from torch import nn

# Normalizing flow stuff
from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform, AffineCouplingTransform
from nflows.transforms.lu import LULinear
from nflows.nn.nets import ResidualNet
from nflows.utils import torchutils

# Diffusion
from ccai.models.diffusion.diffusion import GaussianDiffusion
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
            #transforms.append(PiecewiseRationalQuadraticCouplingTransform(mask=mask,
            #                                                              transform_net_create_fn=create_transform_net,
            #                                                              tails='linear', tail_bound=4))
            transforms.append(AffineCouplingTransform(mask=mask, transform_net_create_fn=create_transform_net))

        transforms = CompositeTransform(transforms)
        self.flow = Flow(transforms, base_dist)

    def sample(self, start, goal, constraints=None, T=1):
        B = start.shape[0]
        context = torch.cat((start, goal), dim=1)
        if constraints is not None:
            context = torch.cat((context, constraints), dim=1)

        samples = self.flow.sample(num_samples=1, context=context)
        trajectories = samples.reshape(B, self.T, -1)

        if self.dynamics is not None:
            x = [start.clone()]
            for t in range(self.T - 1):
                x.append(self.dynamics(x[-1], trajectories[:, t]))
            x = torch.stack(x, dim=1)
            trajectories = torch.cat((x, trajectories), dim=-1)

        return trajectories.reshape(B, self.T, self.dx + self.du)

    def log_prob(self, trajectories, start, goal, constraints=None):
        B = start.shape[0]
        context = torch.cat((start, goal), dim=1)

        if constraints is not None:
            context = torch.cat((context, constraints), dim=1)

        if self.dynamics is not None:
            trajectories = trajectories[:, :, -self.du:]

        log_prob = self.flow.log_prob(trajectories.reshape(B, -1), context=context)
        return log_prob.reshape(B)

    def loss(self, trajectories, start, goal, constraints=None):
        return -self.log_prob(trajectories, start, goal, constraints).mean()


class TrajectoryDiffusionModel(nn.Module):

    def __init__(self, T, dx, du, context_dim):
        super().__init__()
        self.T = T
        self.dx = dx
        self.du = du
        self.context_dim = context_dim
        self.diffusion_model = GaussianDiffusion(T, (dx + du), context_dim, timesteps=20, sampling_timesteps=20)

    def sample(self, start, goal, constraints):
        context = torch.cat((start, goal, constraints), dim=1)
        condition = {0: start}
        samples = self.diffusion_model.sample(N=1, context=context, condition=condition).reshape(-1, self.T,
                                                                                                 self.dx + self.du)
        return samples

    def loss(self, trajectories, start, goal, constraints):
        B = trajectories.shape[0]
        context = torch.cat((start, goal, constraints), dim=1)
        return self.diffusion_model.loss(trajectories.reshape(B, -1), context=context).mean()


class TrajectoryCNFModel(TrajectoryCNF):

    def __init__(self, horizon, dx, du, context_dim):
        super().__init__(horizon, dx + du, context_dim)

    def sample(self, start, goal, constraints):
        context = torch.cat((start, goal, constraints), dim=1)
        return self._sample(context)

    def loss(self, x, start, goal, constraints):
        context = torch.cat((start, goal, constraints), dim=1)
        return self.flow_matching_loss(x, context)


class TrajectorySampler(nn.Module):

    def __init__(self, T, dx, du, context_dim, type='nf', dynamics=None):
        super().__init__()
        self.T = T
        self.dx = dx
        self.du = du
        self.context_dim = context_dim
        self.type = type
        assert type in ['nf', 'diffusion', 'cnf']
        if type == 'nf':
            self.model = TrajectoryFlowModel(T, dx, du, context_dim, dynamics)
        elif type == 'cnf':
            self.model = TrajectoryCNFModel(T, dx, du, context_dim)
        else:
            self.model = TrajectoryDiffusionModel(T, dx, du, context_dim)

        self.register_buffer('x_mean', torch.zeros(dx + du))
        self.register_buffer('x_std', torch.ones(dx + du))

    def set_norm_constants(self, x_mean, x_std):
        self.x_mean.data = torch.from_numpy(x_mean).to(device=self.x_mean.device, dtype=self.x_mean.dtype)
        self.x_std.data = torch.from_numpy(x_std).to(device=self.x_std.device, dtype=self.x_std.dtype)

    def sample(self, start, goal, constraints=None):
        samples = self.model.sample(start, goal, constraints)
        return samples * self.x_std + self.x_mean

    def loss(self, trajectories, start, goal, constraints=None):
        return self.model.loss(trajectories, start, goal, constraints)
