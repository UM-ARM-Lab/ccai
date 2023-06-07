''' Constraint generative model '''

import torch
from torch import nn

from torch.nn.functional import mse_loss


class ConditionalMLP(nn.Module):

    def __init__(self, trajectory_dim, context_dim, output_dim, hidden_units=(64, 64)):
        super().__init__()

        self.trajectory_dim = trajectory_dim
        self.context_dim = context_dim
        self.output_dim = output_dim

        layers = [nn.Linear(trajectory_dim + context_dim, hidden_units[0]), nn.ReLU()]
        for i in range(1, len(hidden_units)):
            layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, trajectory, context):
        return self.net(torch.cat([trajectory, context], dim=-1))


class ConstraintGenerativeModel(nn.Module):

    def __init__(self, horizon, xu_dim, context_dim, constraint_dim, hidden_units=(32, 32)):
        super().__init__()

        self.T = horizon
        self.xu_dim = xu_dim
        self.context_dim = context_dim
        self.constraint_dim = constraint_dim

        self.constraint_net = ConditionalMLP(xu_dim, context_dim, constraint_dim, hidden_units)
        import functorch
        self.grad_constraint = functorch.vmap(functorch.jacrev(self.constraint_net, argnums=0))

        self.register_buffer('prior_mu', torch.zeros(context_dim))
        self.register_buffer('prior_sigma', torch.ones(context_dim))

        self.register_buffer('posterior_mu', torch.zeros(context_dim))
        self.register_buffer('posterior_sigma', torch.ones(context_dim))

    @staticmethod
    def _kl_divergence(mu_1, sigma_1, mu_2, sigma_2):
        return torch.log(sigma_2 / sigma_1) + (sigma_1 ** 2 + (mu_1 - mu_2) ** 2) / (2 * sigma_2 ** 2) - 0.5

    def infer_posterior_context(self, trajectories, target_constraint, target_grad_constraint, iters=100):
        """

        :param trajectories: B x N x T x (X + U)
        :param target_constraint: B x N x T x num_constraints
        :param target_grad_constraint: B x N x T x num_constraints x (X + U)
        :param iters:
        :return: B x context_dim mu, B x context_dim sigma
        """
        posterior_mu = self.prior_mu.clone()
        posterior_log_sigma = torch.log(self.prior_sigma.clone())

        optim = torch.optim.Adam([posterior_mu, posterior_log_sigma], lr=1e-2)
        B, T, _ = trajectories.shape
        posterior_mu = posterior_mu.repeat(B, 1)
        posterior_log_sigma = posterior_log_sigma.repeat(B, 1)
        for i in range(iters):
            posterior_sigma = torch.exp(posterior_log_sigma)
            context = posterior_mu + torch.randn_like(posterior_sigma)
            context_repeated = context.reshape(B, 1, 1, -1).repeat(1, N, T, 1).reshape(B * N * T, -1)
            pred_constraint = self.constraint_net(trajectories.reshape(B * N * T, -1), context_repeated)
            pred_grad = self.grad_constraint(trajectories.reshape(B * N * T, -1), context_repeated)
            pred_grad = pred_grad.reshape(B, N, T, -1)
            pred_constraint = pred_constraint.reshape(B, N, T, -1)

            negative_likelihood = torch.mean((pred_constraint - target_constraint) ** 2)
            #negative_likelihood += torch.mean((pred_grad - target_grad_constraint) ** 2)

            prior_divergence = self._kl_divergence(posterior_mu, posterior_sigma, self.prior_mu,
                                                   self.prior_sigma).mean()

            loss = negative_likelihood + prior_divergence / N
            loss.backward()
            optim.step()
            optim.zero_grad()

        self.posterior_mu.data = posterior_mu.data
        self.posterior_sigma.data = posterior_sigma.data
        return posterior_mu, posterior_sigma

    def loss(self, trajectories, target_constraint, target_grad_constraint, context_mu, context_sigma):
        """

        :param trajectories: B x N x (T * dim_x * dim_u)
        :param target_constraint: B x N x num_constraints
        :param target_grad_constraint: B x N x num_constraints x (dim_x + dim_u)
        :param context_mu: B x context_dim
        :param context_sigma: B x context_dim
        :return:
        """
        # sample context
        context = context_mu + context_sigma * torch.randn_like(context_mu)

        B, T, _ = trajectories.shape
        # use same sampled context vs use different sampled context? TODO
        context_repeated = context.reshape(B, 1, -1).repeat(1, T, 1).reshape(B*T, -1)
        pred_constraint = self.constraint_net(trajectories.reshape(B*T, -1), context_repeated)
        pred_grad = self.grad_constraint(trajectories.reshape(B*T, -1), context_repeated)
        pred_grad = pred_grad.reshape(B, T, -1)
        pred_constraint = pred_constraint.reshape(B, T, -1)

        # TODO: add term for gradient
        negative_likelihood = mse_loss(pred_constraint, target_constraint)#torch.mean((pred_constraint - target_constraint) ** 2)
        negative_likelihood += mse_loss(pred_grad, target_grad_constraint)
        prior_divergence = self._kl_divergence(context_mu, context_sigma, self.prior_mu, self.prior_sigma).mean()

        loss = negative_likelihood + prior_divergence
        return loss, context
