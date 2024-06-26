""" CNF trained using Flow Matching """

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

from ccai.models.temporal import TemporalUnet

from ccai.models.cnf.ffjord.layers import CNF, ODEfunc

from ccai.models.helpers import SinusoidalPosEmb
# import ot
import numpy as np


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


from ccai.models.temporal import ResidualTemporalBlock

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
            module.test_atol = 1e-5
            module.test_rtol = 1e-5

        if isinstance(module, ODEfunc):
            module.rademacher = False
            module.residual = False

    model.apply(_set)


class TrajectoryCNF(nn.Module):

    def __init__(
            self,
            horizon,
            x_dim,
            u_dim,
            context_dim,
            hidden_dim=32,
            inflation_noise=0.0
    ):
        super().__init__()

        self.horizon = horizon
        self.dx = x_dim
        self.du = u_dim
        self.xu_dim = x_dim + u_dim
        self.context_dim = context_dim
        self.loss_type = 'ot'

        self.model = TemporalUnet(self.horizon, self.xu_dim,
                                  cond_dim=context_dim,
                                  dim=hidden_dim, dim_mults=(1, 2, 4, 8),
                                  attention=False)
        # self.model = MLP(horizon * xu_dim, context_dim, 256, num_layers=4)
        # self.model = TrajectoryConvNet(xu_dim, context_dim)

        self.register_buffer('prior_mu', torch.zeros(horizon, self.xu_dim))
        self.register_buffer('prior_sigma', torch.ones(horizon, self.xu_dim))
        self.register_buffer('x_mean', torch.zeros(self.xu_dim))
        self.register_buffer('x_std', torch.ones(self.xu_dim))

        self.inflation_noise = inflation_noise
        self.sigma_min = 1e-4
        noise_dist = torch.distributions.Normal(self.prior_mu, self.prior_sigma)

        odefunc = ODEfunc(
            diffeq=self.masked_grad,
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

        mask = torch.ones(1, self.horizon, self.xu_dim)
        #self.register_buffer('_grad_mask', mask)
        self.register_buffer('start_mask', mask[0])
        self._grad_mask = mask

    def masked_grad(self, t, x, context):
        dx, _ = self.model(t, x, context)

        if len(self._grad_mask.shape) == 4:
            num_samples, num_sub_trajs = self._grad_mask.shape[:2]

            dx = dx.reshape(num_samples, num_sub_trajs, self.horizon, self.xu_dim)
            for i in range(num_sub_trajs - 1):
                tmp = dx[:, i, -1, :self.dx].clone()
                dx[:, i, -1, :self.dx] = (tmp + dx[:, i + 1, 0, :self.dx]) / 2
                dx[:, i + 1, 0, :self.dx] = dx[:, i, -1, :self.dx]

            dx = dx.reshape(-1, self.horizon, self.xu_dim)

        return dx * self._grad_mask.reshape(-1, self.horizon, self.xu_dim)

    def flow_matching_loss(self, xu, context, mask=None):
        if self.loss_type == 'diffusion':
            return self.flow_matching_loss_diffusion(xu, context, mask)
        elif self.loss_type == 'ot':
            return self.flow_matching_loss_ot(xu, context, mask)
        elif self.loss_type == 'conditional_ot':
            return self.conditional_flow_matching_loss_ot(xu, context, mask)
        elif self.loss_type == 'conditional_sb':
            return self.conditional_flow_matching_loss_sb(xu, context, mask)

    def flow_matching_loss_ot(self, xu, context, mask=None):

        # Using OT flow matching from https://arxiv.org/pdf/2210.02747.pdf
        B, T, _ = xu.shape

        # Inflate xu with noise
        xu = xu + self.inflation_noise * torch.randn_like(xu)

        # sample t from [0, 1]
        t = torch.rand(B, 1, 1, device=xu.device) * (1 - 1.0e-5)

        mu_t = t * xu
        sigma_t = 1 - (1 - self.sigma_min) * t * torch.ones_like(xu)

        # sample initial noise
        x0 = torch.randn_like(xu)

        # Re-parameterize to xt
        xut = mu_t + sigma_t * x0

        # get GT velocity
        ut = (xu - (1 - self.sigma_min) * x0)

        # include masking to allow inpainting
        # when in-painting, gradient should be zero of in-painted values, and model recieve non-noisy input
        if mask is not None:
            masked_idx = (mask == 0).nonzero()
            xut[masked_idx[:, 0], masked_idx[:, 1], masked_idx[:, 2]] = xu[
                masked_idx[:, 0], masked_idx[:, 1], masked_idx[:, 2]]

            ut[masked_idx[:, 0], masked_idx[:, 1], masked_idx[:, 2]] = torch.zeros_like(
                xu[masked_idx[:, 0], masked_idx[:, 1], masked_idx[:, 2]])

        # ut = (xu - (1 - self.sigma_min) * xut) / (1 - t * (1 - self.sigma_min))

        # faster for training
        vt, _ = self.model.compiled_conditional_train(t.reshape(-1), xut, context)

        # compute loss
        return mse_loss(ut, vt)

    def conditional_flow_matching_loss_ot(self, xu, context):
        # Using OT flow matching from https://arxiv.org/pdf/2210.02747.pdf
        B, T, _ = xu.shape

        # Inflate xu with noise
        xu = xu + self.inflation_noise * torch.randn_like(xu)
        # sample initial noise
        x0 = torch.randn_like(xu)

        # solve OT problem between x0 and xu
        x0_flatten = x0.reshape(B, -1)
        xu_flatten = xu.reshape(B, -1)
        M = torch.cdist(x0_flatten,
                        xu_flatten) ** 2  # in principle we can use a different distance specific to trajectories
        M = M / M.max()
        a, b = ot.unif(x0.size()[0]), ot.unif(xu.size()[0])
        pi = ot.emd(a, b, M.detach().cpu().numpy())
        # Sample random interpolations on pi
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=B)
        i, j = np.divmod(choices, pi.shape[1])
        x0 = x0[i]
        xu = xu[j]

        # sample t from [0, 1]
        t = torch.rand(B, 1, 1, device=xu.device)  # * (1 - 1.0e-5)
        sigma = 0.1
        mu_t = t * xu + (1 - t) * x0
        # sample
        xut = mu_t + sigma * torch.randn_like(xu)
        # get GT velocity
        ut = (xu - x0)

        # faster for training
        vt = self.model.compiled_conditional_train(t.reshape(-1), xut, context)

        # compute loss
        return mse_loss(ut, vt)

    def rectified_flow_matching_loss(self, xu, context):
        # Using OT flow matching from https://arxiv.org/pdf/2210.02747.pdf
        B, T, _ = xu.shape

        # Inflate xu with noise
        xu = xu + self.inflation_noise * torch.randn_like(xu)
        # sample initial noise
        x0 = torch.randn_like(xu)

        # sample t from [0, 1]
        t = torch.rand(B, 1, 1, device=xu.device)  # * (1 - 1.0e-5)
        xut = t * xu + (1 - t) * x0
        # get GT velocity
        ut = (xu - x0)

        # faster for training
        vt = self.model.compiled_conditional_train(t.reshape(-1), xut, context)

        # compute loss
        return mse_loss(ut, vt)

    def conditional_flow_matching_loss_sb(self, xu, context):
        sigma = 1
        B = xu.shape[0]
        # Inflate xu with noise
        xu = xu + self.inflation_noise * torch.randn_like(xu)
        # sample initial noise
        x0 = torch.randn_like(xu)
        t = torch.rand(B, 1, 1, device=xu.device)  # * (1 - 1.0e-5)

        a, b = ot.unif(x0.size()[0]), ot.unif(xu.size()[0])
        M = torch.cdist(x0.reshape(B, -1), xu.reshape(B, -1)) ** 2
        pi = ot.sinkhorn(a, b, M.detach().cpu().numpy(), reg=2 * (sigma ** 2))
        # Sample random interpolations on pi
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=B)
        i, j = np.divmod(choices, pi.shape[1])
        x0 = x0[i]
        xu = xu[j]
        # calculate regression loss
        mu_t = t * xu + (1 - t) * x0
        sigma_t = sigma * torch.sqrt(t - t ** 2)
        xut = mu_t + sigma_t * torch.randn_like(x0)
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t))
        ut = sigma_t_prime_over_sigma_t * (xut - mu_t) + xu - x0
        # faster for training
        vt = self.model.compiled_fwd(t.reshape(-1), xut, context)
        # compute loss
        return mse_loss(ut, vt)

    def _sample(self, context=None, condition=None, mask=None, H=None):
        N = context.shape[0]
        if H is None:
            H = self.horizon

        num_sub_trajectories = H // self.horizon
        sample_shape = torch.Size([N, num_sub_trajectories])
        assert context.shape[1] == num_sub_trajectories

        prior = torch.distributions.Normal(self.prior_mu, self.prior_sigma)
        noise = prior.sample(sample_shape=sample_shape).to(context)

        # if longer horizon, set noise of starts / ends to be same value
        for i in range(num_sub_trajectories - 1):
            noise[:, i, -1] = noise[:, i + 1, 0]

        # manually set the conditions
        if condition is not None:
            noise[:, 0] = self._apply_conditioning(noise[:, 0], condition)
            self._grad_mask = mask
            if num_sub_trajectories > 1:
                self._grad_mask = torch.cat((self._grad_mask[:, None],
                                             torch.ones(N, num_sub_trajectories - 1,
                                                        self.horizon, self.xu_dim, device=mask.device)), dim=1)
        else:
            self._grad_mask = torch.ones_like(noise)
        self._grad_mask = self._grad_mask.to(device=noise.device)
        log_prob = torch.zeros(N * num_sub_trajectories, device=noise.device)
        out = self.flow(noise.reshape(-1, self.horizon, self.xu_dim),
                        logpx=log_prob,
                        context=context.reshape(-1, context.shape[-1]),
                        reverse=False)

        trajectories, log_prob = out[:2]

        return trajectories.reshape(N, -1, self.xu_dim), log_prob.reshape(N, -1).sum(dim=1)

    def _apply_conditioning(self, x, condition=None):
        if condition is None:
            return x

        for t, (start_idx, val) in condition.items():
            val = val.reshape(val.shape[0], -1, val.shape[-1])
            n, h, d = val.shape
            x[:, t:t + h, start_idx:start_idx + d] = val.clone()
        return x

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

    def _get_Ts(self, t):
        beta_min = 0.1
        beta_max = 20
        Ts = t * beta_min + t ** 2 * (beta_max - beta_min) / 2
        grad_Ts = beta_min + t * (beta_max - beta_min)
        return Ts, grad_Ts

    def flow_matching_loss_diffusion(self, xu, context):
        B, T, _ = xu.shape

        # Inflate xu with noise
        xu = xu + self.inflation_noise * torch.randn_like(xu)

        # sample t from [0, 1]
        t = torch.rand(B, 1, 1, device=xu.device) * (1 - 1.0e-5)

        Ts, grad_Ts = self._get_Ts(1 - t)

        alpha_1_minus_t = torch.exp(-Ts / 2)

        mu_t = alpha_1_minus_t * xu
        sigma_t = torch.sqrt(1 - alpha_1_minus_t ** 2) * torch.ones_like(xu)

        # sample initial noise
        x0 = torch.randn_like(xu)

        # Re-parameterize to xt
        xut = mu_t + sigma_t * x0

        # get GT velocity
        tmp = torch.exp(-Ts)
        ut = -grad_Ts * (tmp * xut - alpha_1_minus_t * xu) / (2 * (1 - tmp))
        # faster for training
        vt = self.model.compiled_fwd(t.reshape(-1), xut, context)

        # compute loss
        return mse_loss(ut, vt)

    def diffusion_grad_wrapper(self, t, xu, context):
        s = self.model(t, xu, context)

        Ts, grad_Ts = self._get_Ts(1 - t)

        alpha_1_minus_t = torch.exp(-Ts / 2)
        sigma_t = torch.sqrt(1 - alpha_1_minus_t ** 2) * torch.ones_like(xu)
        return s / sigma_t


import matplotlib.pyplot as plt

if __name__ == "__main__":

    # let's try a test dataset first
    N = 10000
    device = 'cuda:0'

    # generate some data - let's do a 1D uniform [-1, 1] distribution embedded in 2D
    # z = 4 * torch.rand(N, 1, device=device) - 2
    z = torch.randn(N, 1, device=device)
    # z1 = 1.2 + 0.25 * torch.randn(N // 2, 1, device=device)
    # z2 = -0.9 + 0.2 * torch.randn(N // 2, 1, device=device)
    # z = torch.cat([z1, z2], dim=0)
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
