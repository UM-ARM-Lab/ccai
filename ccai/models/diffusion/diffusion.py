import torch
from torch import nn
import numpy as np

from ccai.models.temporal import TemporalUnet

from einops import reduce


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


class oldGaussianDiffusion(nn.Module):

    def __init__(self, horizon, xu_dim, context_dim, num_timesteps):
        super().__init__()

        self.horizon = horizon
        self.xu_dim = xu_dim
        self.context_dim = context_dim
        self.net = TemporalUnet(self.horizon, self.xu_dim, cond_dim=context_dim, dim=32)

        self.n_timesteps = num_timesteps

        betas = cosine_beta_schedule(num_timesteps)
        betas = betas.reshape(-1, 1, 1)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, 1, 1), alphas_cumprod], dim=0)

        alphas_cumprod = alphas_cumprod
        alphas_cumprod_prev = alphas_cumprod_prev
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', 1.0 / torch.sqrt(1 - alphas_cumprod))

        self.register_buffer('sqrt_alphas_cumprod_prev', torch.sqrt(alphas_cumprod_prev))
        self.register_buffer('sqrt_one_minus_alphas_cumprod_prev', torch.sqrt(1. - alphas_cumprod_prev))
        self.register_buffer('sqrt_recip_alphas_cumprod_prev', torch.sqrt(1. / alphas_cumprod_prev))

        posterior_variance = betas * (1. - alphas_cumprod_prev[:-1]) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        # self.register_buffer('posterior_mean_coef1',
        #                     betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        # self.register_buffer('posterior_mean_coef2',
        #                     (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.register_buffer('posterior_mean_coef',
                             betas / torch.sqrt(1 - alphas_cumprod))

    def sample_q(self, x_start, t, noise=None):
        """
        sample from the forward diffusion q(x_t | x0)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        return self.sqrt_alphas_cumprod[t] * x_start + self.sqrt_one_minus_alphas_cumprod[t] * noise

    @torch.no_grad()
    def sample_p(self, N, context, x=None):
        """
        sample from the reverse diffusion p(x0 | x_t)
        """
        B = context.shape[0]
        if x is None:
            x = torch.randn(B * N, self.horizon, self.xu_dim, device=self.betas.device)

        context_repeated = context.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
        for t in np.arange(self.n_timesteps)[::-1]:
            if t > 0:
                z = torch.randn_like(x)
            else:
                z = 0

            t_tensor = torch.tensor([t]).long().to(x.device).reshape(1).repeat(B * N)
            epsilon = self.net(x, t_tensor, context_repeated)

            x = 1 / torch.sqrt(1 - self.betas[t]) * (x - epsilon * self.posterior_mean_coef[t]) + \
                torch.sqrt(self.betas[t]) * z
        return x.reshape(B, N, -1)

    @torch.no_grad()
    def sample_ddim(self, N, context, x=None, subsample=10):
        """
        sample from the reverse diffusion using Denoising Diffusion Implicit Models formulation
        """
        B = context.shape[0]
        if x is None:
            x = torch.randn(B * N, self.horizon, self.xu_dim, device=self.betas.device)

        context_repeated = context.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)

        # sigma_t = 0
        subsampled_t = np.arange(self.n_timesteps)[::subsample][::-1]
        seq = np.concatenate((np.array([self.n_timesteps]), subsampled_t[:-1]))
        prev_seq = subsampled_t

        for t, t_prev in zip(seq, prev_seq):
            t_tensor = torch.tensor([t]).long().to(x.device).reshape(1).repeat(B * N)
            epsilon = self.net(x, t_tensor, context_repeated)

            pred_x0 = self.sqrt_alphas_cumprod_prev[t_prev] * (x - self.sqrt_one_minus_alphas_cumprod_prev[t] *
                                                               self.sqrt_recip_alphas_cumprod_prev[t] * epsilon)

            direction_to_xt = torch.sqrt(1 - self.alphas_cumprod_prev[t]) * epsilon

            x = pred_x0 + direction_to_xt

        return x.reshape(B, N, -1)

    def loss(self, x, context, *args, **kwargs):
        B = x.shape[0]
        x = x.reshape(B, self.horizon, self.xu_dim)
        # randomly sample timestep
        t = torch.randint(0, self.n_timesteps, (B,), device=x.device).long()

        # sample epsilon
        epsilon = torch.randn_like(x)

        # predict epsilon
        x_t = self.sample_q(x, t, epsilon)
        epilson_pred = self.net(x_t, t, context)

        loss = nn.functional.mse_loss(epilson_pred, epsilon, reduction='mean')

        return loss


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
            sampling_timesteps=10,
            loss_type='l2',
            objective='pred_noise',
            schedule_fn_kwargs=dict(),
            ddim_sampling_eta=0.,
            auto_normalize=True,
            min_snr_loss_weight=False,  # https://arxiv.org/abs/2303.09556
            min_snr_gamma=5,
            guidance_w=1.2
    ):
        super().__init__()
        self.guidance_w = guidance_w
        self.horizon = horizon
        self.xu_dim = xu_dim
        self.model = TemporalUnet(self.horizon, self.xu_dim, cond_dim=context_dim, dim=128)
        self.objective = objective

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

    def model_predictions(self, x, t, context):
        if context is not None:
            # classifier free guidance
            unconditional = self.model.compiled_unconditional_test(t, x)
            conditional = self.model.compiled_conditional_test(t, x, context)
            model_output = unconditional + self.guidance_w * (conditional - unconditional)
        else:
            model_output = self.model(t, x, context)
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
        return pred_img, x_start

    def _apply_conditioning(self, x, condition):
        for t, val in condition.items():
            x[:, t, :12] = val.clone()
        return x

    @torch.no_grad()
    def p_sample_loop(self, shape, condition, context, return_all_timesteps=False):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        img = self._apply_conditioning(img, condition)
        imgs = [img]

        for t in reversed(range(0, self.num_timesteps)):
            img, x_start = self.p_sample(img, t, context)
            img = self._apply_conditioning(img, condition)
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

    @torch.no_grad()
    def sample(self, N, condition, context, return_all_timesteps=False):
        B = context.shape[0]
        context = context.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((B * N, self.horizon, self.xu_dim), condition=condition, context=context,
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

    def p_losses(self, x_start, t, context, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        # predict and take gradient step
        model_out = self.model.compiled_conditional_train(t, x, context)

        loss = self.loss_fn(model_out, noise, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def loss(self, x, context):
        b = x.shape[0]
        x = x.reshape(b, self.horizon, self.xu_dim)
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, context)

