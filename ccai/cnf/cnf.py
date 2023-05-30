""" CNF trained using Flow Matching """

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

from ccai.diffusion.temporal import TemporalUnet

from ccai.cnf.ffjord.layers import RegularizedODEfunc, CNF, ODEfunc, SequentialFlow
from einops.layers.torch import Rearrange

from ccai.diffusion.helpers import SinusoidalPosEmb


class MLP(nn.Module):
    def __init__(self, x_dim, context_dim, hidden_dim, num_layers=4):
        super().__init__()
        self.time_embedding = SinusoidalPosEmb(32)
        self.input_dim = x_dim + context_dim + 32
        self.hidden_dim = hidden_dim
        self.output_dim = x_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([nn.Linear(self.input_dim, hidden_dim)])
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            # self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(hidden_dim, x_dim))

    def vmapped_fwd(self, t, x, context=None):
        return self(t.reshape(1), x.unsqueeze(0), context.unsqueeze(0)).squeeze(0)

    def forward(self, t, x, context=None):
        # ensure t is a batched tensor
        B, T, d = x.shape
        t = t.reshape(-1)
        if t.shape[0] == 1:
            t = t.repeat(B)
        t = self.time_embedding(t)

        # TODO make this a resnet? Skip connections?
        # x = torch.cat((x.reshape(B, -1), t.reshape(B, -1)), dim=1)
        x = torch.cat((x.reshape(B, -1), context.reshape(B, -1), t.reshape(B, -1)), dim=1)
        for layer in self.layers:
            x = layer(x)

        return x.reshape(B, T, d)


from ccai.diffusion.temporal import ResidualTemporalBlock

import einops


class TrajectoryConvNet(nn.Module):

    def __init__(self, xu_dim, context_dim):
        super().__init__()
        kernel_size = 3

        self.time_embedding = SinusoidalPosEmb(32)
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim + 32, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
        )

        self.net = ResidualTemporalBlock(xu_dim, xu_dim, embed_dim=128, kernel_size=3)

    def vmapped_fwd(self, t, x, context=None):
        return self(t.reshape(1), x.unsqueeze(0), context.unsqueeze(0)).squeeze(0)

    def forward(self, t, x, context):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        B, T, d = x.shape
        t = t.reshape(-1)
        if t.shape[0] == 1:
            t = t.repeat(B)
        t_embed = self.time_embedding(t)
        context_t = torch.cat([context, t_embed], dim=-1)
        embed = self.context_mlp(context_t)
        x = einops.rearrange(x, 'b h t -> b t h')
        out = self.net(x, embed)
        out = einops.rearrange(out, 'b t h -> b h t')
        return out


def set_cnf_options(solver, model):
    def _set(module):
        if isinstance(module, CNF):
            # Set training settings
            module.solver = solver
            module.atol = 1e-5
            module.rtol = 1e-5

            # If using fixed-grid adams, restrict order to not be too high.
            if solver in ['fixed_adams', 'explicit_adams']:
                module.solver_options['max_order'] = 4
            module.solver_options['first_step'] = 0.2

            # Set the test settings
            module.test_solver = solver
            module.test_atol = 1e-4
            module.test_rtol = 1e-4

        if isinstance(module, ODEfunc):
            module.rademacher = False
            module.residual = False

    model.apply(_set)


class TrajectoryCNF(nn.Module):

    def __init__(
            self,
            horizon,
            xu_dim,
            context_dim,
            inflation_noise=0.0
    ):
        super().__init__()

        self.horizon = horizon
        self.xu_dim = xu_dim
        self.context_dim = context_dim

        self.model = TemporalUnet(self.horizon, self.xu_dim, cond_dim=context_dim, dim=32, dim_mults=(1, 2, 4), attention=False)

        #self.model = MLP(horizon * xu_dim, context_dim, 256, num_layers=4)
        #self.model = TrajectoryConvNet(xu_dim, context_dim)

        self.register_buffer('prior_mu', torch.zeros(horizon, xu_dim))
        self.register_buffer('prior_sigma', torch.ones(horizon, xu_dim))

        self.inflation_noise = inflation_noise
        self.sigma_min = 0.01
        noise_dist = torch.distributions.Normal(self.prior_mu, self.prior_sigma)

        odefunc = ODEfunc(
            diffeq=self.model,
            divergence_fn='approximate',
            residual=False,
            rademacher=False,
        )

        solver = 'dopri5'
        self.flow = CNF(odefunc=odefunc,
                        T=1.0,
                        train_T=False,
                        solver=solver
                        )

        set_cnf_options(solver, self.flow)

    def flow_matching_loss(self, xu, context):
        # Using OT flow matching from https://arxiv.org/pdf/2210.02747.pdf
        B, T, _ = xu.shape

        # Inflate xu with noise
        xu = xu + self.inflation_noise * torch.randn_like(xu)

        # sample t from [0, 1]
        t = torch.rand(B, 1, 1, device=xu.device)

        mu_t = t * xu
        sigma_t = 1 - (1 - self.sigma_min) * t * torch.ones_like(xu)

        # sample initial noise
        x0 = torch.randn_like(xu)

        # Re-parameterize to xt
        xut = mu_t + sigma_t * x0

        # get GT velocity
        ut = (xu - (1 - self.sigma_min) * x0)
        # ut = (xu - (1 - self.sigma_min) * xut) / (1 - t * (1 - self.sigma_min))

        vt = self.model(t.reshape(-1), xut, context)

        # compute loss
        return mse_loss(ut, vt)

    def _sample(self, context=None):
        N = context.shape[0]
        prior = torch.distributions.Normal(self.prior_mu, self.prior_sigma)
        noise = prior.sample(sample_shape=torch.Size([N])).to(context)
        log_prob = torch.zeros(N, device=noise.device)
        out = self.flow(noise, logpx=log_prob, context=context, reverse=False)
        return out[0]

    def log_prob(self, xu, context):
        # inflate with noise
        B, T, _ = xu.shape
        xu = xu + self.inflation_noise * torch.randn_like(xu)
        log_prob = torch.zeros(B, device=xu.device)
        out = self.flow(xu, logpx=log_prob, context=context, reverse=True)
        z = out[0]
        delta_log_prob = out[1]
        prior = torch.distributions.Normal(self.prior_mu, self.prior_sigma)
        logprob = prior.log_prob(z).to(xu).sum(dim=[1, 2]) - delta_log_prob  # logp(z_S) = logp(z_0) - \int_0^S trJ
        return logprob, z


import matplotlib.pyplot as plt

if __name__ == "__main__":

    # let's try a test dataset first
    N = 10000
    device = 'cuda:0'

    # generate some data - let's do a 1D uniform [-1, 1] distribution embedded in 2D
    #z = 4 * torch.rand(N, 1, device=device) - 2
    z = torch.randn(N, 1, device=device)
    #z1 = 1.2 + 0.25 * torch.randn(N // 2, 1, device=device)
    #z2 = -0.9 + 0.2 * torch.randn(N // 2, 1, device=device)
    #z = torch.cat([z1, z2], dim=0)
    A = torch.tensor([[0.8825],
                      [-0.2678]], device=device)

    x = z @ A.T  # N x 2 data
    # what if we inflate the data slightly?
    x = x + torch.randn_like(x) * 0.01
    # x = 0.25 * torch.randn(N, 2, device=device)
    # reshape so horizon is 1
    x = x.reshape(N, 1, 2)

    plt.scatter(x[:, 0, 0].cpu(), x[:, 0, 1].cpu())
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.show()

    # let's say no context
    model = TrajectoryCNF(horizon=1, xu_dim=2, context_dim=0, inflation_noise=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for i in range(2001):
        # shuffle x
        x = x[torch.randperm(len(x))]
        x_batch = x.reshape(-1, 250, 1, 2)
        total_loss = 0
        model.train()
        for xb in x_batch:
            loss = model.flow_matching_loss(xb, None)
            # loss, _ = model.log_prob(xb, None)
            # loss = -loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        model.eval()
        if i % 100 == 0:
            total_ll = 0
            all_z = []
            for xb in x_batch:
                ll, z = model.log_prob(xb, None)
                ll = ll.mean()
                all_z.append(z)
                total_ll += ll.item()
            print(f'iter {i}, loss {total_loss / len(x_batch)}, ll {total_ll / len(x_batch)}')
            all_z = torch.stack(all_z, dim=0).reshape(-1, 2).detach().cpu().numpy()
            # Now let's sample from the model and see how things look
            x_hat = model._sample(1000, None)
            x_hat = x_hat.reshape(-1, 2).detach().cpu().numpy()
            x_plot = x.reshape(-1, 2).detach().cpu().numpy()

            lims = [-3, 3]
            fig, axes = plt.subplots(2, 2, figsize=(10, 5))

            axes[0, 0].scatter(x_plot[:, 0], x_plot[:, 1], label='data', alpha=0.25)
            axes[0, 0].scatter(x_hat[:, 0], x_hat[:, 1], label='model', alpha=0.1)
            axes[0, 0].legend()
            axes[0, 0].set_xlim(*lims)
            axes[0, 0].set_ylim(*lims)
            import numpy as np

            noise = np.random.randn(1000, 2)
            axes[0, 1].scatter(all_z[:1000, 0], all_z[:1000, 1], alpha=0.25, label='data')
            axes[0, 1].scatter(noise[:, 0], noise[:, 1], alpha=0.25, label='base dist')

            axes[0, 1].set_title('Latent space')
            axes[0, 1].set_xlim(*lims)
            axes[0, 1].set_ylim(*lims)
            # now let's try and plot the density
            Nplot = 500
            x1 = torch.linspace(*lims, Nplot)
            x2 = torch.linspace(*lims, Nplot)

            X1, X2 = torch.meshgrid(x1, x2)
            X = torch.stack([X1, X2], dim=-1).reshape(-1, 2).to(device=device)
            log_prob, _ = model.log_prob(X.reshape(-1, 1, 2), None)

            density = torch.exp(log_prob)
            # clip density
            # density = torch.clamp(density, min=None, max=1000)
            density = density.reshape(Nplot, Nplot).detach().cpu().numpy()
            print(density.max(), density.min())
            axes[1, 0].imshow(density.T, extent=[*lims, *lims], origin='lower')
            axes[1, 0].set_title('density')
            log_prob = log_prob.reshape(Nplot, Nplot).detach().cpu().numpy()
            # scale log prob to be between 0 and 1
            axes[1, 1].contourf(X1, X2, log_prob, levels=20)
            # axes[1, 1].imshow(log_prob.T, extent=[*lims, *lims], origin='lower')
            axes[1, 1].set_title('log prob')
            plt.savefig(f'test_two_gaussians_inflated_{i}.png')
            plt.close()
