import torch
from torch import nn
import numpy as np

from ccai.models.temporal import TemporalUnet, TemporalUNetContext, BinaryClassifier, UnetClassifier
from ccai.models.transformer import TransformerContext, TransformerForDiffusion

from einops import reduce
import math

from tqdm import tqdm


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
            dx,
            du,
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
            unconditional=True,
            model_type='conv_unet',
            discriminator_guidance=False
    ):
        super().__init__()

        if model_type not in ['conv_unet', 'transformer']:
            raise ValueError('Invalid model type')

        # model provides score fcn for both trajectory and context
        if model_type == 'conv_unet':
            self.model = TemporalUnet(horizon, dx + du, context_dim, dim=hidden_dim)
        else:
            self.model = TransformerForDiffusion(dx + du, dx + du, horizon, 1, cond_dim=context_dim, n_emb=hidden_dim)

        self.horizon = horizon
        self.xu_dim = dx + du
        self.dx = dx
        self.du = du
        self.hidden_dim = hidden_dim
        self.model = TemporalUnet(self.horizon, self.xu_dim, cond_dim=context_dim, dim=hidden_dim)

        self.classifier = None
        # if discriminator_guidance:
        self.classifier = UnetClassifier(self.horizon, self.xu_dim, cond_dim=context_dim, dim=hidden_dim)


        self.objective = objective
        self.unconditional = unconditional
        self.context_dim = context_dim
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

        self.classifier_guidance = torch.vmap(torch.func.jacrev(self._classifier_guidance, argnums=1))

    def add_classifier(self):
        #self.classifier = BinaryClassifier()
        self.classifier = UnetClassifier(self.horizon, self.xu_dim, cond_dim=self.context_dim, dim=self.hidden_dim)

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
            context = context.reshape(context.shape[0], -1, context.shape[-1])
            B, N, _ = context.shape
        guidance_weights = torch.tensor([0.5, 0.5], device=x.device)
        w_total = 1.2
        unconditional, _ = self.model.compiled_unconditional_test(t, x)
        if not (context is None or self.unconditional):
            num_constraints = context.shape[1]

            conditional, _ = self.model.compiled_conditional_test(t.unsqueeze(1).expand(-1, N).reshape(-1),
                                                               x.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N,
                                                                                                            self.horizon,
                                                                                                            -1),
                                                               context.reshape(B * N, -1))

            if self.classifier is not None and False:
                guidance = self.classifier_guidance(t.unsqueeze(1).expand(-1, N).reshape(-1),
                                                    x.unsqueeze(1).expand(-1, N, -1, -1).reshape(B*N, self.horizon, -1),
                                                    context.reshape(B*N, -1)).squeeze(1)
                guidance = extract(torch.sqrt(self.posterior_variance), t, guidance.shape) * guidance
                conditional = conditional - 0.0 * guidance
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

    def _classifier_guidance(self, t, x, context):
        #_, h = self.model(t[None], x[None], context[None])
        pred = torch.clip(self.classifier.vmapped_fwd(t, x, context), min=1e-5, max=1 - 1e-5)
        log_odds = torch.log(pred / (1 - pred))
        return log_odds.squeeze(0)


    def p_mean_variance(self, x, t, context):
        preds = self.model_predictions(x, t, context)
        x_start = preds.pred_x_start
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t, context):
        b, *_, device = *x.shape, x.device
        alpha = 0.5
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, context=context)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + alpha * (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def _apply_conditioning(self, x, condition=None):
        if condition is None:
            return x
        self.screwdriver_bounds = False
        if self.screwdriver_bounds:
            lb = torch.tensor([-0.47, -0.196, -0.174, -0.227, -0.47, -0.196, -0.174,
                            -0.227, 0.26, -0.105, -0.199, -0.162, -0.1, -0.1])
            ub = torch.tensor([0.47, 1.61, 1.709, 1.618, 0.47, 1.61, 1.709,
                            1.618, 1.396, 1.163, 1.644, 1.719, 0.1, 0.1])
        else:
            lb = torch.tensor([-0.47, -0.196, -0.174, -0.227, -0.47, -0.196, -0.174,
                            -0.227, -5, -5, -1, -1])
            ub = torch.tensor([0.47, 1.61, 1.709, 1.618, 0.47, 1.61, 1.709,
                            1.618, 5, 5, 1, 1])

        lb, ub, = lb.to(device=x.device), ub.to(device=x.device)
        lb = (lb - self.mu[:len(lb)]) / self.std[:len(lb)]
        ub = (ub - self.mu[:len(ub)]) / self.std[:len(ub)]

        torch.clip_(x[:, :, :len(lb)], min=lb, max=ub)
        for t, (start_idx, val) in condition.items():
            val = val.reshape(val.shape[0], -1, val.shape[-1])
            n, h, d = val.shape
            x[:, t:t + h, start_idx:start_idx + d] = val.clone()
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
            # img[:, -1, 8] = img[:, 0, 8] + np.pi / 4.0
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
        if context is not None:
            context = context.reshape(B * N, 1, -1)
        for t in reversed(range(0, start_timestep)):
            img, x_start = self.p_sample(img.reshape(B * N, H, -1), t, context)
            img = img.reshape(B, N, H, -1)

            # combine subtrajectory updates
            for i in range(1, N):
                tmp = img[:, i, 0, :self.dx].clone()
                img[:, i, 0, :self.dx] = (tmp + img[:, i - 1, -1, :self.dx]) / 2
                img[:, i - 1, -1, :self.dx] = img[:, i, 0, :self.dx]

            img = self._apply_conditioning(img.reshape(B, H*N, -1), condition).reshape(B, N, H, -1)

            # add some guidance using gradient - maximise turn angle, keep upright
            eta = 0.0
            img[:, :, :, self.dx - 3:self.dx - 1] -= 0.1 * 2 * eta * img[:, :, :, self.dx - 3:self.dx - 1]
            # img[:, -1, -1, self.dx - 1] -= 10 * /eta
            img[:, :, :, self.dx - 1] -= eta

            # img[:, -1, -1, 8] += 1.0e-3
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
            N2, num_constraints, dc = context.shape
            assert N2 == N

            # context = context.reshape(B, 1, num_constraints, dc).repeat(1, N, 1, 1).reshape(B * N, num_constraints, -1)

        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if H > self.horizon:
            # get closest multiple of horizon
            factor = math.ceil(H / self.horizon)
            # H_total = factor * self.horizon
            sample = self.p_multi_sample_loop((B * N, factor, self.horizon, self.xu_dim),
                                              condition=condition, context=context,
                                              return_all_timesteps=return_all_timesteps)

            if self.classifier is not None:
                likelihood = self.classifier(
                    torch.zeros(B *N * factor, device=sample.device),
                    sample.reshape(-1, self.horizon, self.xu_dim),
                    context=context.reshape(-1, self.context_dim)
                ).reshape(B*N, factor)
            else:
                likelihood = self.approximate_likelihood(sample.reshape(-1, self.horizon, self.xu_dim)).reshape(B*N, factor)

            weight = 0.9 ** torch.arange(0, factor, device=sample.device)
            likelihood = torch.sum(likelihood * weight[None, :], dim=1)
            # combine samples
            combined_samples = [sample[:, i, :-1] for i in range(0, factor - 1)]
            combined_samples.append(sample[:, -1])
            sample = torch.cat([sample[:, i] for i in range(0, factor)], dim=1)
            # get combined trajectory
            # sample = torch.cat(combined_samples, dim=1)
            return sample, likelihood

        sample = sample_fn((B * N, H, self.xu_dim), condition=condition, context=context.squeeze(1),
                         return_all_timesteps=return_all_timesteps)

        if self.classifier is not None:
            likelihood = self.classifier(
                torch.zeros(B * N, device=sample.device),
                sample.reshape(-1, self.horizon, self.xu_dim),
                context=context.reshape(-1, self.context_dim)
            )
        else:
            return sample, None
            likelihood = self.approximate_likelihood(sample.reshape(-1, self.horizon, self.xu_dim))

        return sample, likelihood

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
        B = x_start.shape[0]
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        mask = mask[:, :, :self.dx+self.du]
        # mask defines in-painting, model should receive un-noised copy of masked states
        # note -- only mask states, not controls
        masked_idx = (mask == 0).nonzero()
        x[masked_idx[:, 0], masked_idx[:, 1], masked_idx[:, 2]] = x_start[
            masked_idx[:, 0], masked_idx[:, 1], masked_idx[:, 2]]

        # ensure angle between [-1, 1]
        # x[..., 8] = ((x[..., 8] + 1.0) % 2.0) - 1.0

        # predict and take gradient step
        model_out, _ = self.model.compiled_conditional_train(t, x, context)
        # TODO need to figure out how to do loss that respects wrap-around
        loss = self.loss_fn(model_out, noise, reduction='none')
        # diff = model_out - noise
        # # diff is also an angle, so map between [-1, 1]
        # # TODO not sure what this does for gradients??
        # diff[..., 8] = ((diff[..., 8] + 1.0) % 2.0) - 1.0
        # # account for -1 and 1 having zero distance
        # #diff[..., 8] = torch.where(diff[..., 8].abs() > 1.0, 1.0 - diff[..., 8], diff[..., 8]
        # loss = diff ** 2
        # diff for angle
        #

        # apply mask
        if mask is not None:
            # TODO: I wanted to try this as it makes sense, unfortunately leads to nan loss - not exactly sure why
            # mask should increase weight for turning angle
            # mask[:, :, 14:16] *= 1.5
            loss = loss * mask
            loss = loss.reshape(B, -1).sum(dim=1) / mask.reshape(B, -1).sum(dim=1)

        return loss.mean()

    def loss(self, x, context, mask=None):
        # print('x shape', x.shape, self.dx, self.xu_dim)
        b = x.shape[0]
        x = x[:, :, :self.dx]
        x = x.reshape(b, -1, self.xu_dim)
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, context, mask=mask)

    def set_norm_constants(self, mu, std):
        self.mu = mu
        self.std = std

    def resample(self, x, condition, context, timestep):
        B = x.shape[0]
        if context is not None:
            B, num_constraints, dc = context.shape
            context = context.reshape(B, 1, num_constraints, dc)

        ##context = context.reshape(B, 1, num_constraints, dc).repeat(1, N, 1, 1).reshape(B * N, num_constraints, -1)

        # for replanning
        # takes a current data estimate x_0, samples from forward diffusion to noise to x_timestep < T
        # then runs reverse diffusion to get a new updated sample
        # what if we noise it a little less than we were supposed to
        batched_times = torch.full((B,), timestep, device=x.device, dtype=torch.long)

        x_noised = self.q_sample(x, batched_times - 1)
        # x_noised = x
        # return resampled
        return self.p_sample_loop(x.shape, condition=condition, context=context,
                                  return_all_timesteps=False,
                                  start_timestep=timestep,
                                  trajectory=x_noised)

    def approximate_likelihood(self, x, context=None, forward_kl=False):
        B, H, d = x.shape
        device = self.betas.device
        # N = 100
        # we could randomly choose timesteps, or do all of them. For now let's randomly generatre
        # t = torch.randint(1, self.num_timesteps, (N,), device=device).long()
        # t = torch.arange(1, self.num_timesteps, device=device).long()
        t = torch.arange(1, self.num_timesteps, 8, device=device).long()
        # t = torch.arange(5, 30, 2, device=device).long()

        N = t.shape[0]
        t = t[None, :].repeat(B, 1).reshape(B * N)
        x_0 = x[:, None, ...].repeat(1, N, 1, 1).reshape(B * N, H, d)

        noise_x = torch.randn_like(x_0)
        x_t = self.q_sample(x_start=x_0, t=t, noise=noise_x)

        # compute the posterior q(t-1 | t, 0)
        q_next_x = self.q_posterior(x_start=x_0, x_t=x_t, t=t)

        # Compute our diffusing step from t to t-1
        p_next_x = self.p_mean_variance(x=x_t, t=t, context=context)

        if forward_kl:
            kl_x = self._gaussian_kl(p_next_x[0], p_next_x[1], q_next_x[0], q_next_x[1])
        else:
            kl_x = self._gaussian_kl(q_next_x[0], q_next_x[1], p_next_x[0], p_next_x[1])
        overall_kl = kl_x.reshape(B, N).mean(dim=1)
        return -overall_kl

    @staticmethod
    def _gaussian_kl(mu_1, var_1, mu_2, var_2):
        """ computes kl divergence KL(p_1 | p_2) """
        first_term = 0.5 * torch.log(var_2 / var_1)
        second_term = (var_1 + (mu_1 - mu_2) ** 2) / (2 * var_2)
        kl = first_term + second_term - 0.5
        return kl.reshape(kl.shape[0], -1).sum(dim=1)

    def classifier_loss(self, x_start, context, label, mask, noise=None):
        loss_fn = nn.BCELoss(reduction='none')
        if self.classifier is None:
            raise ValueError('Classifier must be set before calling classifier_loss')
        # fist sample t
        b = x_start.shape[0]
        x = x_start.reshape(b, -1, self.xu_dim)
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        # mask defines in-painting, model should receive un-noised copy of masked states
        # note -- only mask states, not controls
        masked_idx = (mask == 0).nonzero()
        x[masked_idx[:, 0], masked_idx[:, 1], masked_idx[:, 2]] = x_start[
            masked_idx[:, 0], masked_idx[:, 1], masked_idx[:, 2]]

        # get weights
        #eps, h = self.model(t, x, context, dropout=False)
        pred_label = self.classifier(t, x, context)

        #print(torch.where(torch.isnan(h)))
        #print(torch.where(torch.isnan(x)))

        loss = loss_fn(pred_label, label)
        loss = torch.mean(extract(torch.sqrt(self.betas), t, loss.shape) * loss)

        # accuracy
        pred = torch.round(pred_label)
        accuracy = torch.mean(torch.where(pred == label, 1.0, 0.0))
        return loss, accuracy

class JointDiffusion(GaussianDiffusion):
    """

        Class for jointly diffusing p(\tau, c) using score function networks e(\tau|c) and e(c|\tau)
        where \tau is the trajectory and c is the context.

    """

    def __init__(
            self,
            horizon,
            dx, du,
            context_dim,
            timesteps=1000,
            sampling_timesteps=20,
            loss_type='l2',
            hidden_dim=32,
            model_type='conv_unet',
            unconditional=True,
            inits_noise=None,
            noise_noise=None,
            guided=False
    ):
        super().__init__(
            horizon,
            dx,
            du,
            context_dim,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            loss_type=loss_type,
            hidden_dim=hidden_dim,
            unconditional=unconditional,
            model_type=model_type
        )
        if model_type not in ['conv_unet', 'transformer']:
            raise ValueError('Invalid model type')
        # model provides score fcn for both trajectory and context
        if model_type == 'conv_unet':
            self.model = TemporalUNetContext(horizon, dx + du, context_dim, dim=hidden_dim)
        else:
            self.model = TransformerContext(dx + du, dx + du, horizon, 1, cond_dim=context_dim, n_emb=hidden_dim)

        self.register_buffer('context_dropout_p', torch.tensor([0.25]))
        self.grad_cost = torch.func.vmap(torch.func.jacrev(self._cost))
        self.cost = torch.func.vmap(self._cost)

        self.inits_noise = inits_noise
        self.noise_noise = noise_noise

        self.guided = guided

    def model_predictions(self, x, t, context=None):
        B, N = x.shape[:2]

        if context is not None:
            context = context.reshape(B * N, -1)

        # print(t.reshape(B * N).shape, x.reshape(B * N, -1, self.xu_dim).shape, context.shape)
        e_x, e_c = self.model(t.reshape(B * N), x.reshape(B * N, -1, self.xu_dim), context)
        e_x = e_x.reshape(B, N, -1, self.xu_dim)
        e_c = e_c.reshape(B, N, -1)

        # combine score for knot points
        # print(N)
        # print('--')
        # print(e_x[0, 0, -1])
        # print(e_x[0, 1, 0])
        for i in range(1, N):
            tmp = e_x[:, i, 0, :self.dx].clone()  # = e_x[:, i]
            e_x[:, i, 0, :self.dx] = (tmp + e_x[:, i - 1, -1, :self.dx]) / 2.0
            e_x[:, i - 1, -1, :self.dx] = e_x[:, i, 0, :self.dx].clone()
            # e_x[:, i, 0, :10] = e_x[:, i-1, -1, :10]
            # e_x[:, i-1, -1, :10] = (tmp + e_x[:, i-1, -1, :10]) / 2
            # e_x[:, i, 0, :10] = e_x[:, i-1, -1, :10]

        if N == 1:
            e_x.squeeze_(1)
            e_c.squeeze_(1)

        x_start = self.predict_start_from_noise(x.reshape(B * N, -1, self.xu_dim),
                                                t.reshape(B * N),
                                                e_x.reshape(B * N, -1, self.xu_dim))  # .reshape(B, N, -1, self.xu_dim)
        c_start = self.predict_start_from_noise(context.reshape(B * N, -1),
                                                t.reshape(B * N),
                                                e_c.reshape(B * N, -1))  # .reshape(B, N, -1)
        return ModelPrediction(e_x, x_start), ModelPrediction(e_c, c_start)

    def p_mean_variance(self, x, t, context):
        x = x.reshape(x.shape[0], -1, *x.shape[-2:])
        t = t[:, None].repeat(1, x.shape[1])
        B, N, h, d = x.shape
        pred_x, pred_c = self.model_predictions(x, t, context)

        x_start = pred_x.pred_x_start
        c_start = pred_c.pred_x_start
        x_mean, x_var, x_log_var = self.q_posterior(x_start=x_start, x_t=x.reshape(B * N, h, d), t=t.reshape(B * N))
        c_mean, c_var, c_log_var = self.q_posterior(x_start=c_start, x_t=context.reshape(B * N, -1), t=t.reshape(B * N))
        return {'x': {'mean': x_mean.reshape(B, N, h, d), 'var': x_var.reshape(B, N, 1, 1),
                      'logvar': x_log_var.reshape(B, N, 1, 1), 'start': x_start.reshape(B, N, h, d)},
                'c': {'mean': c_mean.reshape(B, N, -1), 'var': c_var.reshape(B, N, 1),
                      'logvar': c_log_var.reshape(B, N, 1), 'start': c_start.reshape(B, N, -1)}}

    def _cost(self, x):
        return torch.sum((x[1:] - x[:-1]) ** 2)
        # ctheta = x[:, 8]
        # stheta = x[:, 9]
        # theta = torch.atan2(stheta, ctheta)
        # return torch.sum((theta[0] - theta[1]) ** 2) + torch.sum((theta[-1] - theta[-1]) ** 2)# - 0.01*(theta[-1]-theta[0]) ** 2

    @torch.no_grad()
    def p_sample(self, x, t, context):
        b, *_, device = *x.shape, x.device
        alpha = 1.0
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        mu_var = self.p_mean_variance(x=x, t=batched_times, context=context)
        # model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, context=context)
        if self.noise_noise is not None:
            noise_x = .707**2 * self.noise_noise[t].to(x.device) if t > 0 else 0.
            # noise_x = 1 * self.noise_noise[t].to(x.device) if t > 0 else 0.
            noise_c = 0
        else:
            noise_x = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
            noise_c = torch.randn_like(context) if t > 0 else 0.  # no noise if t == 0
        # make noise the same
        b, N = x.shape[:2]
        if t > 0 and len(x.shape) > 3:
            for i in range(1, N):
                noise_x[:, i, 0, :self.dx] = noise_x[:, i - 1, -1, :self.dx]

        # use same noise vector
        grad = torch.zeros_like(mu_var['x']['mean'])
        # add some guidance using gradient - maximise turn angle, keep upright
        eta = 0.0 if not self.guided else .02
        grad[:, :, :, self.dx - 3:self.dx - 1] = -2 * 0.1 * (mu_var['x']['mean'][:, :, :, self.dx - 3:self.dx - 1] +
                                                             self.mu[self.dx - 3:self.dx - 1].to(device=x.device))
        # grad[:, -1, -1, self.dx - 1] = -eta
        grad[:, :, :, self.dx - 1] = -1
        # print(mu_var['x']['logvar'].exp())
        pred_x = mu_var['x']['mean'] + alpha * (0.5 * mu_var['x']['logvar']).exp() * (noise_x + eta * grad)
        # pred_x = pred_x - 0.01 * self.grad_cost(mu_var['x']['mean'].reshape(b*N, -1, self.xu_dim)).reshape(b, N, -1, self.xu_dim)
        pred_c = mu_var['c']['mean'].squeeze(1) + alpha * (0.5 * mu_var['c']['logvar']).exp().squeeze(1) * noise_c
        # print(self.cost(pred_x).mean())#, self.cost(pred_x).max())
        return pred_x, mu_var['x']['start'], pred_c, mu_var['c']['start']

    @torch.no_grad()
    def p_sample_loop(self, shape, condition, context,
                      return_all_timesteps=False,
                      start_timestep=None,
                      trajectory=None):
        batch, device = shape[0], self.betas.device

        if self.inits_noise is not None:
            x = self.inits_noise.to(device)
        elif trajectory is None:
            x = torch.randn(shape, device=device)
        else:
            x = trajectory
        if context is None:
            c = torch.randn((batch, self.context_dim), device=device)
        else:
            c = context

        if start_timestep is None:
            start_timestep = self.num_timesteps

        x = self._apply_conditioning(x, condition)
        all_x = [x]
        all_c = [c]
        for t in reversed(range(0, start_timestep)):
            x, x_start, c, c_start = self.p_sample(x.unsqueeze(1), t, c)
            x = x.squeeze(1)
            condition = condition
            # apply conditioning
            x = self._apply_conditioning(x, condition)
            if context is not None:
                c = context

            all_x.append(x)
            all_c.append(c)

        ret_x = x if not return_all_timesteps else torch.stack(all_x, dim=1)
        ret_c = c if not return_all_timesteps else torch.stack(all_c, dim=1)

        return ret_x, ret_c

    def p_losses(self, x_start, t, context, noise=None, mask=None):
        B = x_start.shape[0]
        noise_x = default(noise, lambda: torch.randn_like(x_start))
        noise_c = default(noise, lambda: torch.randn_like(context))
        x = self.q_sample(x_start=x_start, t=t, noise=noise_x)
        c = self.q_sample(x_start=context, t=t, noise=noise_c)

        # mask defines in-painting, model should receive un-noised copy of masked states
        # note -- only mask states, not controls
        masked_idx = (mask == 0).nonzero()
        x[masked_idx[:, 0], masked_idx[:, 1], masked_idx[:, 2]] = x_start[
            masked_idx[:, 0], masked_idx[:, 1], masked_idx[:, 2]]

        # dropout for training to condition on context as well as diffuse context
        context_mask_dist = torch.distributions.Bernoulli(probs=1 - self.context_dropout_p)
        context_mask = context_mask_dist.sample((B,))  # .to(device=context.device)

        # do masking
        masked_idx = (context_mask == 0).nonzero()
        c[masked_idx[:, 0], masked_idx[:, 1]] = context[masked_idx[:, 0], masked_idx[:, 1]]

        # predict and take gradient step
        e_x, e_c = self.model(t, x, c)
        loss_x = self.loss_fn(e_x, noise_x, reduction='none')
        loss_c = self.loss_fn(e_c, noise_c, reduction='none')
        # apply mask
        if mask is not None:
            loss_x = loss_x * mask
            loss_x = loss_x.reshape(B, -1).sum(dim=1) / mask.reshape(B, -1).sum(dim=1)

            loss_c = loss_c * context_mask
            loss = loss_x + torch.sum(loss_c.reshape(B, -1), -1)

        # print(loss_x.mean(), loss_c.mean())

        ##print(context)
        # print(loss_x.mean())
        # print(loss_c.mean())
        # loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # loss = loss * extract(self.loss_weight, t, loss.shape)
        # exit(0)
        return loss.mean()

    def loss(self, x, context, mask=None):
        b = x.shape[0]
        x = x.reshape(b, -1, self.xu_dim)
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, context, mask=mask)

    @torch.no_grad()
    def p_multi_sample_loop(self, shape, condition, context, return_all_timesteps=False,
                            start_timestep=None,
                            trajectory=None):
        device = self.betas.device
        B, N, H, xu_dim = shape
        if self.inits_noise is not None:
            x = self.inits_noise.to(device)
        if trajectory is None:
            x = torch.randn(shape, device=device)
        else:
            x = trajectory
        if context is None:
            c = torch.randn((*shape[:-2], self.context_dim), device=device)
        else:
            c = context

        # first state of each subtrajectory should be the same as the final state of the previous
        # make noises the same
        for i in range(1, N):
            x[:, i, 0, :self.dx] = x[:, i - 1, -1, :self.dx]

        if start_timestep is None:
            start_timestep = self.num_timesteps

        x[:, 0] = self._apply_conditioning(x[:, 0], condition)
        all_x = [x]
        all_c = [c]
        for t in reversed(range(0, start_timestep)):
            x, x_start, c, c_start = self.p_sample(x.reshape(B, N, H, -1), t, c.reshape(B, N, -1))
            x = x.reshape(B, N, H, -1)
            c = c.reshape(B, N, -1)
            for i in range(1, N):
                tmp = x[:, i, 0, :self.dx].clone()
                x[:, i, 0, :self.dx] = (tmp + x[:, i - 1, -1, :self.dx]) / 2
                x[:, i - 1, -1, :self.dx] = x[:, i, 0, :self.dx]
            # apply conditioning
            x[:, 0] = self._apply_conditioning(x[:, 0], condition)

            if context is not None:
                c = context

            all_x.append(x)
            all_c.append(c)

        ret_x = x if not return_all_timesteps else torch.stack(all_x, dim=1)
        ret_c = c if not return_all_timesteps else torch.stack(all_c, dim=1)

        return ret_x, ret_c

    def sample(self, N, H=None, condition=None, context=None, return_all_timesteps=False):
        B = 1
        if H is None:
            H = self.horizon

        if context is not None:
            N2, num_constraints, dc = context.shape
            assert N2 == N
            if num_constraints == 1:
                context = context.squeeze(1)
            # context = context.reshape(B, 1, num_constraints, dc).repeat(1, N, 1, 1).reshape(B * N, num_constraints, -1)

        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if H > self.horizon:
            # get closest multiple of horizon
            factor = math.ceil(H / self.horizon)
            # H_total = factor * self.horizon
            sample_x, sample_c = self.p_multi_sample_loop((B * N, factor, self.horizon, self.xu_dim),
                                                          condition=condition, context=context,
                                                          return_all_timesteps=return_all_timesteps)

            likelihood = self.approximate_likelihood(sample_x.reshape(-1, self.horizon, self.xu_dim),
                                                     sample_c.reshape(-1, sample_c.shape[-1])).reshape(B * N, -1)
            likelihood = likelihood.sum(dim=1)

            # combine samples
            combined_samples = [sample_x[:, i, :-1] for i in range(0, factor - 1)]
            combined_samples.append(sample_x[:, -1])
            sample_x = torch.cat([sample_x[:, i] for i in range(0, factor)], dim=1)
            sample_c = torch.stack([sample_c[:, i] for i in range(0, factor)], dim=1)
            return sample_x, sample_c, likelihood

        sample_x, sample_c = sample_fn((B * N, H, self.xu_dim), condition=condition, context=context,
                                       return_all_timesteps=return_all_timesteps)
        return sample_x, sample_c, self.approximate_likelihood(sample_x, sample_c)

    def approximate_likelihood(self, x, context, forward_kl=False):
        B, H, d = x.shape
        device = self.betas.device
        # N = 100
        # we could randomly choose timesteps, or do all of them. For now let's randomly generatre
        # t = torch.randint(0, self.num_timesteps, (N,), device=device).long()
        t = torch.arange(1, self.num_timesteps, 4, device=device).long()
        # t = torch.arange(5, 30, 2, device=device).long()

        N = t.shape[0]
        t = t[None, :].repeat(B, 1).reshape(B * N)
        x_0 = x[:, None, ...].repeat(1, N, 1, 1).reshape(B * N, H, d)
        c_0 = context[:, None].repeat(1, N, 1).reshape(B * N, -1)

        noise_x = torch.randn_like(x_0)
        noise_c = torch.randn_like(c_0)
        x_t = self.q_sample(x_start=x_0, t=t, noise=noise_x)
        c_t = self.q_sample(x_start=c_0, t=t, noise=noise_c)

        # compute the posterior q(t-1 | t, 0)
        q_next_x = self.q_posterior(x_start=x_0, x_t=x_t, t=t)
        q_next_c = self.q_posterior(x_start=c_0, x_t=c_t, t=t)

        # Compute our diffusing step from t to t-1
        p_next = self.p_mean_variance(x=x_t, t=t, context=c_t)

        if forward_kl:
            kl_x = self._gaussian_kl(p_next['x']['mean'].squeeze(1), p_next['x']['var'].squeeze(1), q_next_x[0],
                                     q_next_x[1])
            kl_c = self._gaussian_kl(p_next['c']['mean'], p_next['c']['var'], q_next_c[0], q_next_c[1])
        else:
            kl_x = self._gaussian_kl(q_next_x[0], q_next_x[1], p_next['x']['mean'].squeeze(1),
                                     p_next['x']['var'].squeeze(1))
            kl_c = self._gaussian_kl(q_next_c[0], q_next_c[1], p_next['c']['mean'].squeeze(1),
                                     p_next['c']['var'].squeeze(1))

        overall_kl = (kl_c + kl_x).reshape(B, N).mean(dim=1)

        return -overall_kl


class ConstrainedDiffusion(GaussianDiffusion):
    def __init__(
            self,
            horizon,
            dx,
            du,
            context_dim,
            opt_problem,
            timesteps=1000,
            sampling_timesteps=10,
            loss_type='l2',
            constrain=True,
            hidden_dim=32,
            unconditional=False,
            alpha_J=2.5e-4,
            alpha_C=0.1,
            inits_noise=None,
            noise_noise=None,
            guided=False
    ):
        super().__init__(
            horizon,
            dx,
            du,
            context_dim,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            loss_type=loss_type,
            hidden_dim=hidden_dim,
            unconditional=unconditional,
        )
        self.problem_dict = opt_problem
        self.constrain = constrain
        self.alpha_J = alpha_J
        self.alpha_C = alpha_C

        if constrain:
            self.use_gauss_newton = True

        self.skip_first = False
        self.anneal_factor = .1
        self.noise_factor = 0.707
        self.max_norm = 10

        self.model = TemporalUNetContext(horizon, dx + du, context_dim, dim=hidden_dim)

        self.register_buffer('context_dropout_p', torch.tensor([0.25]))
        self.grad_cost = torch.func.vmap(torch.func.jacrev(self._cost))
        self.cost = torch.func.vmap(self._cost)

        self.inits_noise = inits_noise
        self.noise_noise = noise_noise

        self.guided = guided

    def model_predictions(self, x, t, context=None):
        B, N = x.shape[:2]

        if context is not None:
            context = context.reshape(B * N, -1)
        e_x, e_c = self.model(t.reshape(B * N), x.reshape(B * N, -1, self.xu_dim), context)
        e_x = e_x.reshape(B, N, -1, self.xu_dim)
        e_c = e_c.reshape(B, N, -1)

        # combine score for knot points
        # print(N)
        # print('--')
        # print(e_x[0, 0, -1])
        # print(e_x[0, 1, 0])
        for i in range(1, N):
            tmp = e_x[:, i, 0, :self.dx].clone()  # = e_x[:, i]
            e_x[:, i, 0, :self.dx] = (tmp + e_x[:, i - 1, -1, :self.dx]) / 2.0
            e_x[:, i - 1, -1, :self.dx] = e_x[:, i, 0, :self.dx].clone()
            # e_x[:, i, 0, :10] = e_x[:, i-1, -1, :10]
            # e_x[:, i-1, -1, :10] = (tmp + e_x[:, i-1, -1, :10]) / 2
            # e_x[:, i, 0, :10] = e_x[:, i-1, -1, :10]

        if N == 1:
            e_x.squeeze_(1)
            e_c.squeeze_(1)

        x_start = self.predict_start_from_noise(x.reshape(B * N, -1, self.xu_dim),
                                                t.reshape(B * N),
                                                e_x.reshape(B * N, -1, self.xu_dim))  # .reshape(B, N, -1, self.xu_dim)
        return ModelPrediction(e_x, x_start)
    
    def p_mean_variance(self, x, t, context):
        x = x.reshape(x.shape[0], -1, *x.shape[-2:])
        t = t[:, None].repeat(1, x.shape[1])
        B, N, h, d = x.shape
        pred_x = self.model_predictions(x, t, context)

        x_start = pred_x.pred_x_start
        x_mean, x_var, x_log_var = self.q_posterior(x_start=x_start, x_t=x.reshape(B * N, h, d), t=t.reshape(B * N))
        return x_mean.reshape(B, N, h, d), x_var.reshape(B, N, 1, 1), x_log_var.reshape(B, N, 1, 1), x_start.reshape(B, N, h, d)

    def c_state_mask(self, context, x):
        c_state = tuple(context[0].tolist())
        z_dim = self.problem_dict[c_state].dz
        mask = torch.ones((self.xu_dim + z_dim), device=x.device).bool()
        if c_state == (-1, -1, -1):
            mask[27:36] = False
        elif c_state == (-1 , 1, 1):
            mask[27:30] = False
        elif c_state == (1, -1, -1):
            mask[30:36] = False
        # Concat False to mask to match the size of x
        mask = torch.cat((mask, torch.zeros(self.xu_dim + z_dim -x.shape[-1], device=x.device).bool()))
        mask_no_z = mask.clone()
        if z_dim > 0:
            mask_no_z[-z_dim:] = False
        return c_state, mask, mask_no_z
    
    def p_sample(self, x, t, context, anneal=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        with torch.no_grad():
            model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x[..., :self.xu_dim],
                                                                              t=batched_times,
                                                                              context=context)
        # we have to unnormalize first
        if self.skip_first:
            _x = x[:, 1:].clone()
            _model_mean = model_mean[:, 1:].squeeze()
            if len(x.shape) < 4:
                model_log_variance = model_log_variance.squeeze(-1)
            H = self.horizon - 1
        else:
            _x = x.clone()
            _model_mean = model_mean.squeeze()
            if len(x.shape) < 4:
                model_log_variance = model_log_variance.squeeze(-1)
            H = self.horizon

        x_norm = _x.clone()

        x_norm[:, :, :self.xu_dim] *= self.std
        x_norm[:, :, :self.xu_dim] += self.mu

        pred_x = torch.zeros_like(_x)

        if self.noise_noise is not None:
            sample = self.noise_noise[t].to(device).squeeze()
        else:
            sample = torch.randn_like(_x)
        noise = self.noise_factor * sample[:, :, :self.xu_dim] if t > 0 else 0.  # no noise if t == 0
        if t > 0 and len(x.shape) > 3:
            for i in range(1, x.shape[1]):
                noise[:, i, 0, :self.dx] = noise[:, i - 1, -1, :self.dx]
        update = _model_mean + (0.5 * model_log_variance).exp() * self.noise_factor * noise - _x[:, :, :self.xu_dim]

        for context_ind in (range(context.shape[1])):
            c_state, mask, mask_no_z = self.c_state_mask(context[:, context_ind], x)
            num_dim = mask.long().sum().item()
            problem = self.problem_dict[c_state]


            problem._preprocess(x_norm[:, context_ind, :, mask_no_z], projected_diffusion=True)

            z_dim = problem.dz

            C, dC, _ = problem.combined_constraints(x_norm[:, context_ind, :, mask], compute_hess=False, projected_diffusion=True)
            # C, dC, _ = problem._con_eq(x_norm[:, :, mask_no_z], compute_hess=False, projected_diffusion=True)
            # compute unconstrained update with cost guide

            # make update be unnormalized
            # Fix the indexing here
            if problem.dz > 0:
                unnormalized_update = update[:, context_ind, :, mask_no_z] * self.std[mask_no_z[:-problem.dz]]

            else:
                unnormalized_update = update[:, context_ind, :, mask_no_z] * self.std[mask_no_z]

            if self.guided:
                # _, dJ, _ = problem._objective(x_norm[:, context_ind, :, mask_no_z])
                # dJ = dJ.reshape(b, H, -1)
                # unnormalized_update -= self.alpha_J * dJ
                # grad = torch.zeros_like(mu_var['x']['mean'])
                # # add some guidance using gradient - maximise turn angle, keep upright
                eta = 0.0 if not self.guided else .0005
                unnormalized_update[:, :, self.dx - 3:self.dx - 1] = -2 * 0.1 * (_model_mean[:, context_ind, :, self.dx - 3:self.dx - 1] +
                                                                    self.mu[self.dx - 3:self.dx - 1].to(device=x.device))
                # grad[:, -1, -1, self.dx - 1] = -eta
                unnormalized_update[:, :, self.dx - 1] = -1

            update_this_b_ind = torch.cat((unnormalized_update, torch.zeros(b, H, z_dim, device=device)[:, :, mask[36:]]), dim=2)

            dC = dC.reshape(b, -1, (H) * num_dim)

            # add zero update for z
            # update_this_b_ind = torch.cat((update, torch.zeros(1, H, z_dim, device=device)), dim=2)

            xi_C = None
            if self.constrain:
                with (torch.no_grad()):
                    # convert to float64
                    dtype = torch.float64
                    C = C.to(dtype=dtype)
                    dC = dC.to(dtype=dtype)
                    eye = torch.eye(C.shape[1]).repeat(b, 1, 1).to(device=C.device, dtype=dtype)
                    update_this_b_ind = update_this_b_ind.to(dtype=dtype)
                    # Damping anneals from 1 to 0
                    if anneal:
                        max_damping = 1
                        min_damping = 1e-6
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
                    eye2 = torch.eye(_x.shape[-2] * num_dim, device=x.device, dtype=dtype).unsqueeze(0)
                    update_this_b_ind = (eye2 - projection) @ update_this_b_ind.reshape(b, -1, 1)
                    update_this_b_ind = update_this_b_ind.squeeze(-1)

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
            update_this_b_ind = update_this_b_ind.to(dtype=dtype)
            # update_this_b_ind = update_this_b_ind.reshape(1, -1)

            # total update
            update_this_b_ind -= self.alpha_C * xi_C

            # maximum norm on xi_C
            norm_update = torch.linalg.norm(update_this_b_ind, dim=1, keepdim=True)
            update_this_b_ind = torch.where(norm_update < self.max_norm, update_this_b_ind, update_this_b_ind * self.max_norm / norm_update)

            # normalize update
            update_this_b_ind = update_this_b_ind.reshape(b, H, -1)
            if problem.dz > 0:
                update_this_b_ind[:, :, :-problem.dz] = update_this_b_ind[:, :, :-problem.dz] / self.std[mask[:36]]
            else:
                update_this_b_ind = update_this_b_ind / self.std[mask[:36]]

            update[:, context_ind, :, mask] = update_this_b_ind

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

        if self.inits_noise is not None:
            trajectory = self.inits_noise.to(device)
        elif trajectory is None:
            trajectory = torch.randn(shape, device=device)
        c_state, mask, mask_no_z = self.c_state_mask(context, trajectory)
        problem = self.problem_dict[c_state]

        if self.skip_first:
            z = torch.zeros(batch, T, self.problem.dz, device=device)
            z[:, 1:] = problem.get_initial_z(trajectory[:, 1:])
        else:
            z = problem.get_initial_z(trajectory, projected_diffusion=True)

        if start_timestep is None:
            start_timestep = self.num_timesteps

        if z is not None:
            augmented_trajectory = torch.cat((trajectory, z), dim=-1)
        else:
            augmented_trajectory = trajectory
        augmented_trajectory = self._apply_conditioning(augmented_trajectory, condition)
        trajectories = [augmented_trajectory]

        for t in reversed(range(0, start_timestep)):
            c_state, mask, mask_no_z = self.c_state_mask(context, augmented_trajectory)
            augmented_trajectory, x_start = self.p_sample(augmented_trajectory, t, context, anneal)
            augmented_trajectory = self._apply_conditioning(augmented_trajectory, condition)
            # also clamp between bounds
            self._clamp_in_bounds(augmented_trajectory[:, :, :self.xu_dim], self.problem_dict[c_state], mask[:self.xu_dim])

            trajectories.append(augmented_trajectory)

        ret = augmented_trajectory[:, :, :self.xu_dim] if not return_all_timesteps else torch.stack(trajectories, dim=1)
        return ret, context

    def sample(self, N, H=None, condition=None, context=None, return_all_timesteps=False):
        B = 1
        if H is None:
            H = self.horizon

        if context is not None:
            N2, num_constraints, dc = context.shape
            assert N2 == N
            if num_constraints == 1:
                context = context.squeeze(1)
            # context = context.reshape(B, 1, num_constraints, dc).repeat(1, N, 1, 1).reshape(B * N, num_constraints, -1)

        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if H > self.horizon:
            # get closest multiple of horizon
            factor = math.ceil(H / self.horizon)
            # H_total = factor * self.horizon
            sample_x, sample_c = self.p_multi_sample_loop((B * N, factor, self.horizon, self.xu_dim),
                                                          condition=condition, context=context,
                                                          return_all_timesteps=return_all_timesteps)

            likelihood = self.approximate_likelihood(sample_x.reshape(-1, self.horizon, self.xu_dim),
                                                     sample_c.reshape(-1, sample_c.shape[-1])).reshape(B * N, -1)
            likelihood = likelihood.sum(dim=1)

            # combine samples
            combined_samples = [sample_x[:, i, :-1] for i in range(0, factor - 1)]
            combined_samples.append(sample_x[:, -1])
            sample_x = torch.cat([sample_x[:, i] for i in range(0, factor)], dim=1)
            sample_c = torch.stack([sample_c[:, i] for i in range(0, factor)], dim=1)
            return sample_x, sample_c, likelihood

        sample_x, sample_c = sample_fn((B * N, H, self.xu_dim), condition=condition, context=context,
                                       return_all_timesteps=return_all_timesteps)
        return sample_x, sample_c, self.approximate_likelihood(sample_x, sample_c)
    
    def _clamp_in_bounds(self, xuz, problem, mask):
        min_x = problem.x_min.reshape(1, 1, -1).expand(-1, self.horizon, -1)
        max_x = problem.x_max.reshape(1, 1, -1).expand(-1, self.horizon, -1)
        torch.clamp_(xuz[:, :, mask], min=min_x.to(device=xuz.device) / self.std[mask],
                     max=max_x.to(device=xuz.device) / self.std[mask])

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
    
    def _cost(self, x):
        return torch.sum((x[1:] - x[:-1]) ** 2)
        # ctheta = x[:, 8]
        # stheta = x[:, 9]
        # theta = torch.atan2(stheta, ctheta)
        # return torch.sum((theta[0] - theta[1]) ** 2) + torch.sum((theta[-1] - theta[-1]) ** 2)# - 0.01*(theta[-1]-theta[0]) ** 2

    @torch.no_grad()
    def p_multi_sample_loop(self, shape, condition, context, return_all_timesteps=False,
                            start_timestep=None,
                            trajectory=None):
        device = self.betas.device
        B, N, H, xu_dim = shape
        if self.inits_noise is not None:
            x = self.inits_noise.to(device)
        if trajectory is None:
            x = torch.randn(shape, device=device)
        else:
            x = trajectory
        if context is None:
            c = torch.randn((*shape[:-2], self.context_dim), device=device)
        else:
            c = context

        # first state of each subtrajectory should be the same as the final state of the previous
        # make noises the same
        for i in range(1, N):
            x[:, i, 0, :self.dx] = x[:, i - 1, -1, :self.dx]

        if start_timestep is None:
            start_timestep = self.num_timesteps

        x[:, 0] = self._apply_conditioning(x[:, 0], condition)
        all_x = [x]
        all_c = [c]
        for t in reversed(range(0, start_timestep)):
            x, x_start = self.p_sample(x.reshape(B, N, H, -1), t, c.reshape(B, N, -1))
            x = x.reshape(B, N, H, -1)
            c = c.reshape(B, N, -1)
            
            for i in range(1, N):
                tmp = x[:, i, 0, :self.dx].clone()
                x[:, i, 0, :self.dx] = (tmp + x[:, i - 1, -1, :self.dx]) / 2
                x[:, i - 1, -1, :self.dx] = x[:, i, 0, :self.dx]
            # apply conditioning
            x[:, 0] = self._apply_conditioning(x[:, 0], condition)

            if context is not None:
                c = context

            all_x.append(x)
            all_c.append(c)

        ret_x = x if not return_all_timesteps else torch.stack(all_x, dim=1)
        ret_c = c if not return_all_timesteps else torch.stack(all_c, dim=1)

        return ret_x, ret_c

class LatentDiffusion(GaussianDiffusion):
    def __init__(
            self,
            vae,
            horizon,
            dx,
            du,
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
            unconditional=True,
            model_type='conv_unet'
    ):
        super().__init__(
            horizon,
            dx,
            du,
            context_dim,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            loss_type=loss_type,
            objective=objective,
            schedule_fn_kwargs=schedule_fn_kwargs,
            ddim_sampling_eta=ddim_sampling_eta,
            min_snr_loss_weight=min_snr_loss_weight,
            min_snr_gamma=min_snr_gamma,
            hidden_dim=hidden_dim,
            unconditional=unconditional,
            model_type=model_type
        )
        if vae is None:
            raise ValueError('VAE must be provided')
        self.vae = vae

    def _apply_conditioning(self, x, condition=None, context=None):
        if condition is None:
            return x

        for t, (start_idx, val) in condition.items():
            val = val.reshape(val.shape[0], -1, val.shape[-1])
            n, h, d = val.shape
            x_decode = self.vae.vae_t.decode(x, context)
            x_decode[:, t:t + h, start_idx:start_idx + d] = val.clone()
            _, x, _ = self.vae.vae_t.encode(x_decode, context)
        return x
    
    def sample(self, N, H=None, condition=None, context=None, return_all_timesteps=False):
        B = 1
        if H is None:
            H = self.horizon

        if context is not None:
            N2, num_constraints, dc = context.shape
            assert N2 == N

            # context = context.reshape(B, 1, num_constraints, dc).repeat(1, N, 1, 1).reshape(B * N, num_constraints, -1)

        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if H > self.horizon:
            # get closest multiple of horizon
            factor = math.ceil(H / self.horizon)
            # H_total = factor * self.horizon
            sample = self.p_multi_sample_loop((B * N, factor, self.horizon, self.xu_dim),
                                              condition=condition, context=context,
                                              return_all_timesteps=return_all_timesteps)
            # combine samples
            combined_samples = [sample[:, i, :-1] for i in range(0, factor - 1)]
            combined_samples.append(sample[:, -1])
            sample = torch.cat([sample[:, i] for i in range(0, factor)], dim=1)
            # get combined trajectory
            # sample = torch.cat(combined_samples, dim=1)
            return sample

        latent_samples = sample_fn((B * N, H, self.xu_dim), condition=condition, context=context.squeeze(1),
                         return_all_timesteps=return_all_timesteps)
        
        context_repeated = context.repeat(1, H, 1)
        
        decoded_samples = self.vae.vae_t.decode(latent_samples, context_repeated)

        decoded_samples_x = self.vae.vae_x.decode(decoded_samples[..., :8], context_repeated)
        decoded_samples_u = self.vae.vae_u.decode(decoded_samples[..., 8:24], context_repeated)

        return torch.cat((decoded_samples_x, decoded_samples_u), dim=-1), context, None