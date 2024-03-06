import torch
from torch import nn
import numpy as np

from ccai.models.temporal import TemporalUnet

from einops import reduce
import math


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, t, *args, **kwargs):
        return self.fn(x, t, *args, **kwargs) + x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64, num_layers=2):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, context):
        B = x.shape[0]
        t = t.reshape(B, 1)
        xtc = torch.cat([x, t, context], dim=-1)
        return self.net(xtc)


def ResNetBlock(input_dim, context_dim, hidden_size, num_layers):
    return Residual(MLP(input_dim + 1 + context_dim, input_dim, hidden_size, num_layers))


class ResNet(nn.Module):
    def __init__(self, input_dim, context_dim, hidden_size, num_layers, num_blocks):
        super().__init__()
        self.blocks = [ResNetBlock(input_dim, context_dim, hidden_size, num_layers) for _ in range(num_blocks)]
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x, t, context):
        for block in self.blocks:
            x = block(x, t, context)
        return x


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


from torch.nn import functional as F


def identity(t, *args, **kwargs):
    return t


from collections import namedtuple

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            horizon,
            xu_dim,
            context_dim,
            timesteps=1000,
            sampling_timesteps=20,
            loss_type='l2',
            objective='pred_noise',
            schedule_fn_kwargs=dict(),
            ddim_sampling_eta=0.,
            min_snr_loss_weight=False,  # https://arxiv.org/abs/2303.09556
            min_snr_gamma=5,
            hidden_dim=32,
            unconditional=True
    ):
        super().__init__()
        self.horizon = horizon
        self.xu_dim = xu_dim
        self.model = TemporalUnet(self.horizon, self.xu_dim, cond_dim=context_dim, dim=hidden_dim)
        self.objective = objective
        self.unconditional = unconditional
        beta_schedule_fn = cosine_beta_schedule
        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        self.sampling_timesteps = sampling_timesteps

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)
        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, context=None):
        if context is not None:
            B, N, _ = context.shape
        guidance_weights = torch.tensor([0.5, 0.5], device=x.device)
        w_total = 1.0
        unconditional = self.model.compiled_unconditional_test(t, x)
        # print(context.shape)
        if not (context is None or self.unconditional):
            num_constraints = context.shape[1]

            conditional = self.model.compiled_conditional_test(t.unsqueeze(1).expand(-1, N).reshape(-1),
                                                               x.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N,
                                                                                                            self.horizon,
                                                                                                            -1),
                                                               context.reshape(B * N, -1))
            conditional = conditional.reshape(B, N, self.horizon, -1)
            # classifier free guidance
            model_output = unconditional
            # compose multiple constraints
            if num_constraints == 1:
                model_output += w_total * (conditional.squeeze(1) - unconditional)
            else:
                diff = conditional - unconditional.unsqueeze(1)
                model_output += w_total * torch.sum(guidance_weights.reshape(1, -1, 1, 1) * diff, dim=1)
        else:
            model_output = unconditional
        pred_noise = model_output
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, context):
        preds = self.model_predictions(x, t, context)
        x_start = preds.pred_x_start
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t, context):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, context=context)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        # pred_img[:, -1, 8] += 0.01
        return pred_img, x_start

    def _apply_conditioning(self, x, condition=None):
        if condition is None:
            return x

        for t, (start_idx, val) in condition.items():
            n, d = val.shape
            x[:, t, start_idx:start_idx + d] = val.clone()
        return x

    @torch.no_grad()
    def p_sample_loop(self, shape, condition, context, return_all_timesteps=False,
                      start_timestep=None,
                      trajectory=None):
        batch, device = shape[0], self.betas.device

        if trajectory is None:
            img = torch.randn(shape, device=device)
        else:
            img = trajectory
        if start_timestep is None:
            start_timestep = self.num_timesteps
        img = self._apply_conditioning(img, condition)
        imgs = [img]
        for t in reversed(range(0, start_timestep)):
            img, x_start = self.p_sample(img, t, context)
            img = self._apply_conditioning(img, condition)
            img[:, -1, 8] += 1.0e-3
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        return ret

    @torch.no_grad()
    def p_multi_sample_loop(self, shape, condition, context, return_all_timesteps=False,
                            start_timestep=None,
                            trajectory=None):

        # B is batch dimension, N is number of sub-trajectories, H is horizon of each subtrajectory
        B, N, H, xu_dim = shape
        device = self.betas.device
        if trajectory is None:
            img = torch.randn(shape, device=device)
        else:
            img = trajectory

        # first state of each subtrajectory should be the same as the final state of the previous
        # make noises the same
        for i in range(1, N):
            img[:, i, 0] = img[:, i - 1, -1]

        if start_timestep is None:
            start_timestep = self.num_timesteps

        # TODO: currently only allow conditioning for first subtrajectory
        #  need to think how to do this for all subtrajectories
        img[:, 0] = self._apply_conditioning(img[:, 0], condition)
        imgs = [img]
        for t in reversed(range(0, start_timestep)):
            img, x_start = self.p_sample(img.reshape(B * N, H, -1), t, context)
            img = img.reshape(B, N, H, -1)

            # combine subtrajectory updates
            for i in range(1, N):
                tmp = img[:, i, 0, :9].clone()
                img[:, i, 0, :9] = (tmp + img[:, i - 1, -1, :9]) / 2
                img[:, i - 1, -1, :9] = img[:, i, 0, :9]

            img[:, 0] = self._apply_conditioning(img[:, 0], condition)
            img[:, -1, -1, 8] += 1.0e-3
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, context, return_all_timesteps=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, context)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        return ret

    def sample(self, N, H=None, condition=None, context=None, return_all_timesteps=False):
        B = 1
        if H is None:
            H = self.horizon

        if context is not None:
            B, num_constraints, dc = context.shape
            context = context.reshape(B, 1, num_constraints, dc).repeat(1, N, 1, 1).reshape(B * N, num_constraints, -1)

        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if H > self.horizon:
            # get closest multiple of horizon
            factor = math.ceil(H / self.horizon)
            #H_total = factor * self.horizon
            sample = self.p_multi_sample_loop((B * N, factor, self.horizon, self.xu_dim),
                                              condition=condition, context=context,
                                              return_all_timesteps=return_all_timesteps)
            # combine samples
            combined_samples = [sample[:, i, :-1] for i in range(0, factor-1)]
            combined_samples.append(sample[:, -1])
            # get combined trajectory
            sample = torch.cat(combined_samples, dim=1)
            return sample

        return sample_fn((B * N, H, self.xu_dim), condition=condition, context=context,
                         return_all_timesteps=return_all_timesteps)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, context, noise=None, mask=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # mask defines in-painting, model should receive un-noised copy of masked states
        masked_idx = (mask == 0).nonzero()
        x[masked_idx[:, 0], masked_idx[:, 1]] = x_start[masked_idx[:, 0], masked_idx[:, 1]]
        # predict and take gradient step
        model_out = self.model.compiled_conditional_train(t, x, context)

        loss = self.loss_fn(model_out, noise, reduction='none')

        # apply mask
        if mask is not None:
            loss = loss * mask[:, :, None]
            loss = loss.sum(dim=1) / mask.sum(dim=1).reshape(-1, 1)

        # loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # loss = loss * extract(self.loss_weight, t, loss.shape)

        return loss.mean()

    def loss(self, x, context, mask=None):
        b = x.shape[0]
        x = x.reshape(b, -1, self.xu_dim)
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, context, mask=mask)

    def set_norm_constants(self, mu, std):
        self.mu = mu
        self.std = std

    def resample(self, x, condition, context, timestep):
        B, num_constraints, dc = context.shape
        N, _, _ = x.shape
        assert (B == N)

        ##context = context.reshape(B, 1, num_constraints, dc).repeat(1, N, 1, 1).reshape(B * N, num_constraints, -1)

        # for replanning
        # takes a current data estimate x_0, samples from forward diffusion to noise to x_timestep < T
        # then runs reverse diffusion to get a new updated sample
        # what if we noise it a little less than we were supposed to
        batched_times = torch.full((B,), timestep, device=x.device, dtype=torch.long)

        x_noised = self.q_sample(x, batched_times)
        # x_noised = x
        # return resampled
        return self.p_sample_loop(x.shape, condition, context,
                                  return_all_timesteps=False,
                                  start_timestep=timestep,
                                  trajectory=x_noised)


class ConstrainedDiffusion(GaussianDiffusion):

    def __init__(
            self,
            horizon,
            xu_dim,
            context_dim,
            opt_problem,
            z_dim,
            timesteps=1000,
            sampling_timesteps=10,
            loss_type='l2',
            constrain=True,
            hidden_dim=32,
            unconditional=False,
            alpha_J=1e-4,
            alpha_C=0.1
    ):
        super().__init__(
            horizon,
            xu_dim,
            context_dim,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            loss_type=loss_type,
            hidden_dim=hidden_dim,
            unconditional=unconditional,
        )
        self.problem = opt_problem
        self.z_dim = z_dim  # dimension of auxiliary variable z (number of inequality constraints)
        self.constrain = constrain
        self.alpha_J = alpha_J
        self.alpha_C = alpha_C

        if constrain:
            self.use_gauss_newton = True

        self.skip_first = False
        self.anneal_factor = .1
        self.noise_factor = 0.707
        self.max_norm = 10

    def p_sample(self, x, t, context, anneal=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        with torch.no_grad():
            model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x[:, :, :self.xu_dim],
                                                                              t=batched_times,
                                                                              context=context)
        # we have to unnormalize first
        if self.skip_first:
            _x = x[:, 1:].clone()
            _model_mean = model_mean[:, 1:]
            H = self.horizon - 1
        else:
            _x = x.clone()
            _model_mean = model_mean
            H = self.horizon

        noise = self.noise_factor * torch.randn_like(_x)[:, :, :self.xu_dim] if t > 0 else 0.  # no noise if t == 0

        x_norm = _x.clone()
        x_norm[:, :, :self.xu_dim] *= self.std
        x_norm[:, :, :self.xu_dim] += self.mu

        J, dJ, _ = self.problem._objective(x_norm[:, :, :self.xu_dim])
        C, dC, _ = self.problem.combined_constraints(x_norm)

        dJ = dJ.reshape(b, H, self.xu_dim)
        dC = dC.reshape(b, -1, H * (self.xu_dim + self.z_dim))

        # compute unconstrained update with cost guide
        update = _model_mean + (0.5 * model_log_variance).exp() * self.noise_factor * noise - _x[:, :, :self.xu_dim]

        # make update be unnormalized
        update = update * self.std - self.alpha_J * dJ

        # add zero update for z
        update = torch.cat((update, torch.zeros(b, H, self.problem.dz, device=device)), dim=2)

        xi_C = None
        if self.constrain:
            with (torch.no_grad()):
                # convert to float64
                dtype = torch.float64
                C = C.to(dtype=dtype)
                dC = dC.to(dtype=dtype)
                eye = torch.eye(C.shape[1]).repeat(b, 1, 1).to(device=C.device, dtype=dtype)
                update = update.to(dtype=dtype)

                # Damping anneals from 1 to 0
                if anneal:
                    max_damping = 1
                    damping = min_damping + (max_damping - min_damping) * t / self.num_timesteps
                else:
                    damping = 0

                # Compute damped projection matrix
                try:
                    dCdCT_inv = torch.linalg.solve(dC @ dC.permute(0, 2, 1) +
                                                   damping * eye
                                                   , eye)
                except Exception as e:
                    dCdCT_inv = torch.linalg.pinv(dC @ dC.permute(0, 2, 1) * damping * eye)
                projection = dC.permute(0, 2, 1) @ dCdCT_inv @ dC

                # Compute constrained update
                eye2 = torch.eye(_x.shape[1] * _x.shape[2], device=x.device, dtype=dtype).unsqueeze(0)
                update = (eye2 - projection) @ update.reshape(b, -1, 1)
                update = update.squeeze(-1)

                # Update to decrease constraint violation
                if self.use_gauss_newton:
                    xi_C = dCdCT_inv @ C.unsqueeze(-1)
                    xi_C = (dC.permute(0, 2, 1) @ xi_C).squeeze(-1)
                else:
                    grad_C_sq = 2 * C.unsqueeze(-1) * dC
                    xi_C = torch.sum(grad_C_sq, dim=1)

        # Update to minimize constraint if no projection
        if xi_C is None:
            grad_C_sq = 2 * C.unsqueeze(-1) * dC
            xi_C = torch.sum(grad_C_sq, dim=1)

        # Convert back to single precision
        dtype = torch.float32
        update = update.to(dtype=dtype)
        update = update.reshape(b, -1)

        # total update
        update -= self.alpha_C * xi_C

        # maximum norm on xi_C
        norm_update = torch.linalg.norm(update, dim=1, keepdim=True)
        update = torch.where(norm_update < self.max_norm, update, update * self.max_norm / norm_update)

        # normalize update
        update = update.reshape(b, H, -1)
        update[:, :, :self.xu_dim] = update[:, :, :self.xu_dim] / self.std

        # update
        pred_x = _x + update

        if self.skip_first:
            pred_x = torch.cat((x[:, 0].unsqueeze(1), pred_x), dim=1)

        pred_x.detach_()
        return pred_x, x_start

    def p_sample_loop(self, shape, condition, context, return_all_timesteps=False,
                      start_timestep=None,
                      trajectory=None,
                      anneal=True):

        batch, T, xu_dim = shape
        device = context.device
        if trajectory is None:
            trajectory = torch.randn(shape, device=device)

        if self.skip_first:
            z = torch.zeros(batch, T, self.problem.dz, device=device)
            z[:, 1:] = self.problem.get_initial_z(trajectory[:, 1:])
        else:
            z = self.problem.get_initial_z(trajectory)

        if start_timestep is None:
            start_timestep = self.num_timesteps

        augmented_trajectory = torch.cat((trajectory, z), dim=-1)
        augmented_trajectory = self._apply_conditioning(augmented_trajectory, condition)
        trajectories = [augmented_trajectory]

        for t in reversed(range(0, start_timestep)):
            augmented_trajectory, x_start = self.p_sample(augmented_trajectory, t, context, anneal)
            augmented_trajectory = self._apply_conditioning(augmented_trajectory, condition)
            # also clamp between bounds
            self._clamp_in_bounds(augmented_trajectory[:, :, :self.xu_dim])

            trajectories.append(augmented_trajectory)

        ret = augmented_trajectory[:, :, :self.xu_dim] if not return_all_timesteps else torch.stack(trajectories, dim=1)
        return ret

    def _clamp_in_bounds(self, xuz):
        min_x = self.problem.x_min.reshape(1, 1, -1).expand(-1, self.horizon, -1)
        max_x = self.problem.x_max.reshape(1, 1, -1).expand(-1, self.horizon, -1)
        torch.clamp_(xuz[:, :, :self.xu_dim], min=min_x.to(device=xuz.device) / self.std,
                     max=max_x.to(device=xuz.device) / self.std)

    def resample(self, x, condition, context, timestep):
        B, num_constraints, dc = context.shape
        N, _, _ = x.shape
        assert (B == N)
        # for replanning
        # takes a current data estimate x_0, samples from forward diffusion to noise to x_timestep < T
        # then runs reverse diffusion to get a new updated sample
        # what if we noise it a little less than we were supposed to
        batched_times = torch.full((B,), timestep, device=x.device, dtype=torch.long)

        x_noised = self.q_sample(x, batched_times)
        # return resampled
        return self.p_sample_loop(x.shape, condition, context,
                                  return_all_timesteps=False,
                                  start_timestep=timestep,
                                  trajectory=x_noised,
                                  anneal=False)
