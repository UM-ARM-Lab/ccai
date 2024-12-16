""" CNF trained using Flow Matching """

import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
from torch.nn.functional import mse_loss, huber_loss
from torch.func import jacrev, vmap
# from torchcubicspline import(natural_cubic_spline_coeffs, 
#                              NaturalCubicSpline)

from ccai.models.temporal import TemporalUnet, TemporalUnetDynamics, TemporalUnetStateAction, StateActionMLP

from ccai.models.cnf.ffjord.layers import CNF, ODEfunc
from torchcfm.conditional_flow_matching import SchrodingerBridgeConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher
from ccai.models.helpers import SinusoidalPosEmb
from torch_cg import cg_batch
# import ot
import numpy as np
from tqdm import tqdm

import os
import argparse
import pickle
import time

fpath = os.path.dirname(os.path.realpath(__file__))


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


from ccai.models.temporal import ResidualTemporalBlock

def set_cnf_options(solver, model):
    def _set(module):
        if isinstance(module, CNF):
            # Set training settings
            module.solver = solver
            module.atol = 1e-5
            module.rtol = 1e-5

            # module.solver_options['first_step'] = 0.2
            # If using fixed-grid adams, restrict order to not be too high.
            if solver in ['fixed_adams', 'explicit_adams']:
                module.solver_options['max_order'] = 4
            if solver == 'rk4':
                module.solver_options['step_size'] = .03

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
            opt_problem=None,
            hidden_dim=32,
            inflation_noise=0.0,
            state_only=False,
            state_control_only=False,
    ):
        super().__init__()

        self.horizon = horizon
        self.dx = x_dim
        self.du = u_dim
        self.xu_dim = x_dim + u_dim
        self.context_dim = context_dim
        self.problem_dict = opt_problem
        # self.loss_type = 'ot'
        # self.loss_type = 'conditional_ot_sb'
        if state_only:
            self.loss_type = 'state_only'
            self.model = TemporalUnetDynamics(self.horizon, self.dx+1,
                                    cond_dim=context_dim,
                                    dim=hidden_dim, dim_mults=(1, 2, 4, 8),
                                    attention=False)
        elif state_control_only:
            self.loss_type = 'state_control_only'
            self.model = TemporalUnetStateAction(self.horizon, self.xu_dim,
                                    cond_dim=context_dim,
                                    dim=hidden_dim, dim_mults=(1, 2, 4, 8),
                                    attention=False,
                                    problem_dict=self.problem_dict)
            # self.model = StateActionMLP(self.horizon, self.xu_dim,
            #                         cond_dim=context_dim,
            #                         dim=hidden_dim, dim_mults=(1, 2, 4, 8),
            #                         attention=False)
            self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=.1)
        else:
            self.loss_type = 'conditional_ot_sb'
            # sigma = .025 ** .5
            sigma = .2
            self.FM = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)

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
            diffeq=self.masked_grad,# if self.loss_type == 'conditional_ot_sb' else self.state_control_only_forward,
            divergence_fn='approximate',
            residual=False,
            rademacher=False,
        )

        solver = 'dopri5'
        # solver = 'bosh3'
        # solver = 'rk4'
        self.flow = CNF(odefunc=odefunc,
                        T=1.0,
                        train_T=False,
                        solver=solver
                        )

        set_cnf_options(solver, self.flow)

        mask = torch.ones(1, self.horizon, self.xu_dim)
        #self.register_buffer('_grad_mask', mask)
        self.register_buffer('start_mask', mask[0])
        self._grad_mask = mask.cuda()

        self.all_contexts = [
            (-1., 1, 1),
            (1, -1, -1),
            (1, 1, 1)
        ]

        self.interp = vmap(self._interp, randomness='same')
        self.dinterp_dt = vmap(jacrev(self._interp, argnums=2), randomness='same')

    def state_control_only_forward(self, t, x, context=None):
        dx, _ = self.model(t, x, context)
        return dx * (self.horizon-1)

    def masked_grad(self, t, x, context=None):
        # print(f'step {self.step_id}')
        self.step_id += 1

        dx, _, _ = self.model.compiled_conditional_test(t, x, context)

        # dx *= (self.horizon - 1)
        
        return dx

    def flow_matching_loss(self, xu, context, mask=None):
        if self.loss_type == 'diffusion':
            return self.flow_matching_loss_diffusion(xu, context, mask)
        elif self.loss_type == 'ot':
            return self.flow_matching_loss_ot(xu, context, mask)
        elif self.loss_type == 'conditional_ot':
            return self.conditional_flow_matching_loss_ot(xu, context, mask)
        elif self.loss_type == 'conditional_sb':
            return self.conditional_flow_matching_loss_sb(xu, context, mask)
        elif self.loss_type == 'conditional_ot_sb':
            return self.conditional_flow_matching_loss_ot_sb(xu, context, mask)
        elif self.loss_type == 'state_only':
            return self.flow_matching_loss_state_only(xu, context, mask)
        elif self.loss_type == 'state_control_only':
            return self.flow_matching_loss_state_control_only(xu, context, mask)
        
    def conditional_flow_matching_loss_ot_sb(self, xu, context, mask=None):
        x0 = torch.randn_like(xu)
        t, xut, truevt = self.FM.sample_location_and_conditional_flow(x0, xu)

        vt, _ = self.model.compiled_conditional_train(t.reshape(-1), xut, context)

        return mse_loss(vt, truevt)
    
    def flow_matching_loss_state_only(self, xu, context, mask=None):
        r = torch.randn(xu.shape[0], 1, device=xu.device)
        t = torch.rand(xu.shape[0], 1, device=xu.device)
        # t *= 0
        # t += .99
        t_ind = (t * (self.horizon-1)).long()
        t_ind.clamp_(0, self.horizon-2)
        t_ind_for_gather = t_ind.reshape(xu.shape[0], 1, 1).repeat(1, 1, self.xu_dim)
        t_ind_for_gather_1 = t_ind_for_gather + 1
        if t_ind_for_gather_1.max() >= self.horizon:
            print('t_ind_for_gather_1', t_ind_for_gather_1.max())
        x0 = torch.gather(xu, 1, t_ind_for_gather).squeeze()
        x1 = torch.gather(xu, 1, t_ind_for_gather_1).squeeze()
        x0 = x0[..., :self.dx]
        x1 = x1[..., :self.dx]
        truevt = (x1 - x0) * (self.horizon - 1)
        # t_ind_for_gather_u = t_ind.reshape(xu.shape[0], 1, 1).repeat(1, 1, self.xu_dim)
        # true_u = torch.gather(xu, 1, t_ind_for_gather).squeeze()
        true_u = x0[..., self.dx:]
        xut = x1 * (t*(self.horizon-1) - t_ind) + x0 * (t_ind + 1 - t*(self.horizon-1))
        xut = torch.cat((xut, r), dim=-1)
        vt, _, u_hat = self.model.compiled_conditional_train(t.reshape(-1), xut, context)
        flow_loss = mse_loss(vt[..., :-1], truevt)
        action_loss = mse_loss(u_hat, true_u)

        return {
            'loss': flow_loss + action_loss,
            'flow_loss': flow_loss,
            'action_loss': action_loss
        }

    def h_poly(self, t):
        tt = t[None, :]**torch.arange(8, device=t.device)[:, None]
        A = torch.tensor([
        #[0, 1,  2,   3,    4,   5,    6,   7],
            [1, 0,  0,   0,  -35,  84,  -70,  20], #p0
            [0, 1,  0,   0,  -20,  45,  -36,  10], #v0
            [0, 0, .5,   0,   -5,  10, -7.5,   2], #a0
            [0, 0,  0, 1/6, -2/3,   1, -2/3, 1/6], #j0
            [0, 0,  0,   0,   35, -84,   70, -20], #p1
            [0, 0,  0,   0,  -15,  39,  -34,  10], #v1
            [0, 0,  0,   0,  5/2,  -7, 13/2,  -2], #a1
            [0, 0,  0,   0, -1/6, 1/2, -1/2, 1/6], #j1
        ], dtype=t.dtype, device=t.device)
        return A @ tt

    def dh_poly(self, t):
        tt = t[None, :]**torch.arange(8, device=t.device)[:, None]
        A = torch.tensor([[   0.0000,    0.0000,    0.0000, -140.0000,  420.0000, -420.0000,
            140.0000,    0.0000],
            [   1.0000,    0.0000,    0.0000,  -80.0000,  225.0000, -216.0000,
            70.0000,    0.0000],
            [   0.0000,    1.0000,    0.0000,  -20.0000,   50.0000,  -45.0000,
            14.0000,    0.0000],
            [   0.0000,    0.0000,    0.5000,   -2.6667,    5.0000,   -4.0000,
                1.1667,    0.0000],
            [   0.0000,    0.0000,    0.0000,  140.0000, -420.0000,  420.0000,
            -140.0000,    0.0000],
            [   0.0000,    0.0000,    0.0000,  -60.0000,  195.0000, -204.0000,
            70.0000,    0.0000],
            [   0.0000,    0.0000,    0.0000,   10.0000,  -35.0000,   39.0000,
            -14.0000,    0.0000],
            [   0.0000,    0.0000,    0.0000,   -0.6667,    2.5000,   -3.0000,
                1.1667,    0.0000]], dtype=t.dtype, device=t.device)
        
        return A @ tt

    def d2h_poly(self, t):
        tt = t[None, :]**torch.arange(8, device=t.device)[:, None]
        A = torch.tensor([[ 0.0000e+00,  0.0000e+00, -4.2000e+02,  1.6800e+03, -2.1000e+03,
            8.4000e+02,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00, -2.4000e+02,  9.0000e+02, -1.0800e+03,
            4.2000e+02,  0.0000e+00,  0.0000e+00],
            [ 1.0000e+00,  0.0000e+00, -6.0000e+01,  2.0000e+02, -2.2500e+02,
            8.4000e+01,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  1.0000e+00, -8.0000e+00,  2.0000e+01, -2.0000e+01,
            7.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  4.2000e+02, -1.6800e+03,  2.1000e+03,
            -8.4000e+02,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00, -1.8000e+02,  7.8000e+02, -1.0200e+03,
            4.2000e+02,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  3.0000e+01, -1.4000e+02,  1.9500e+02,
            -8.4000e+01,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00, -2.0000e+00,  1.0000e+01, -1.5000e+01,
            7.0000e+00,  0.0000e+00,  0.0000e+00]], dtype=t.dtype, device=t.device)
        return A @ tt

    def _interp(self,x, y, xs):
        x_ = x.reshape(-1, 1)
        m = (y[1:] - y[:-1]) / (x_[1:] - x_[:-1])
        m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
        m_prime = (m[1:] - m[:-1]) / (x_[1:] - x_[:-1])
        m_prime = torch.cat([m_prime[[0]], (m_prime[1:] + m_prime[:-1]) / 2, m_prime[[-1]]])

        m_prime_prime = (m_prime[1:] - m_prime[:-1]) / (x_[1:] - x_[:-1])
        m_prime_prime = torch.cat([m_prime_prime[[0]], (m_prime_prime[1:] + m_prime_prime[:-1]) / 2, m_prime_prime[[-1]]])
        
        idxs = torch.searchsorted(x[1:], xs)
        dx = (x[idxs + 1] - x[idxs])
        hh = self.h_poly((xs - x[idxs]) / dx).unsqueeze(-1)
        # hh_deriv = self.dh_poly((xs - x[idxs]) / dx).unsqueeze(-1)#.unsqueeze(-1)        # ret = hh[0] * torch.gather(y, 1, idxs.reshape(-1, 1, 1).expand(-1, -1, y.shape[-1]))
        hh_deriv2 = self.d2h_poly((xs - x[idxs]) / dx).unsqueeze(-1)#.unsqueeze(-1)
        dx = dx.unsqueeze(-1)#.unsqueeze(-1)
        # ret += hh[1] * torch.gather(m, 1, idxs.reshape(-1, 1, 1).expand(-1, -1, y.shape[-1])) * dx
        # ret += hh[2] * torch.gather(y, 1, 1+idxs.reshape(-1, 1, 1).expand(-1, -1, y.shape[-1]))
        # ret += hh[3] * torch.gather(m, 1, 1+idxs.reshape(-1, 1, 1).expand(-1, -1, y.shape[-1])) * dx
        
        ret = hh[0] * y[idxs]
        ret += hh[1] * m[idxs] * dx
        ret += hh[2] * m_prime[idxs] * dx * dx * .5
        ret += hh[3] * m_prime_prime[idxs] * dx * dx * dx * (1/6)

        ret += hh[4] * y[idxs+1]
        ret += hh[5] * m[idxs+1] * dx
        ret += hh[6] * m_prime[idxs+1] * dx * dx * .5
        ret += hh[7] * m_prime_prime[idxs+1] * dx * dx * dx * (1/6)

        ret_d2h = hh_deriv2[0] * y[idxs]
        ret_d2h += hh_deriv2[1] * m[idxs] * dx
        ret_d2h += hh_deriv2[2] * m_prime[idxs] * dx * dx * .5
        ret_d2h += hh_deriv2[3] * m_prime_prime[idxs] * dx * dx * dx * (1/6)

        ret_d2h += hh_deriv2[4] * y[idxs+1]
        ret_d2h += hh_deriv2[5] * m[idxs+1] * dx
        ret_d2h += hh_deriv2[6] * m_prime[idxs+1] * dx * dx * .5
        ret_d2h += hh_deriv2[7] * m_prime_prime[idxs+1] * dx * dx * dx * (1/6)
        return ret.squeeze(), ret_d2h.squeeze()
    
    def standard_normal_kl_div(self, mu, logvar):
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

    def flow_matching_loss_state_control_only(self, xu, context, mask=None):

        # t0 = torch.rand(xu.shape[0], 1, device=xu.device) # initial time (for receding horizon)
        # t = torch.rand(xu.shape[0], 1, device=xu.device) * (1-t0) # t offset (for vector field prediction)

        t = torch.rand(xu.shape[0], 1, device=xu.device) # initial time (for receding horizon)
        t0 = t-t

        # t_ind = ((t-t + t0) * (self.horizon-1)).long() # Get global time index for trajectory
        t_ind = ((t + t0) * (self.horizon-1)).long() # Get global time index for trajectory
        t0_ind = (t0 * (self.horizon-1)).long() # Get local time index for vector field prediction
        t_ind.clamp_(0, self.horizon-2)
        t0_ind.clamp_(0, self.horizon-2)
        t0_ind_for_gather = t0_ind.reshape(xu.shape[0], 1, 1).repeat(1, 1, self.xu_dim)

        x_init = torch.gather(xu, 1, t0_ind_for_gather).squeeze()

        x_arange = torch.linspace(0, 1, xu.shape[1], device=xu.device).expand(xu.shape[0], -1)

        rand_for_u0 = torch.randn_like(xu[:, 0,  self.dx:])

        noise_mean, noise_logvar = self.model.noise_dist(x_init[..., :self.dx])

        pred_noise_u0 = noise_mean + rand_for_u0 * torch.exp(.5 * noise_logvar)

        # goal_u0 = x1[:, self.dx:]

        # arange0 = torch.arange(rand_for_u0.shape[0], device=xu.device)
        # arange1 = torch.arange(goal_u0.shape[0], device=xu.device)

        # rand_for_u0, goal_u0, arange0, arange1 = self.FM.ot_sampler.sample_plan_with_labels(rand_for_u0, goal_u0, y0=arange0, y1=arange1, replace=False)
        # # Rearrange ut, true_uv using arange
        # inds = torch.sort(arange1).indices
        # rand_for_u0 = rand_for_u0[inds]#.contiguous()

        xu[:, 0, self.dx:] = pred_noise_u0
        # xu[:, 0, self.dx:] = rand_for_u0

        xt, truevt = self.interp(x_arange, xu, t)
        # truevt = self.dinterp_dt(x_arange, xu, t).squeeze()
        truevt = truevt.squeeze()
        # xt += torch.randn_like(xt) * .1
        xt += torch.randn_like(xt) * .03

        vt, p_matrix, xi_C = self.model.compiled_conditional_train(t.reshape(-1), xt, context)
        flow_loss = mse_loss(vt, truevt)
        state_loss = mse_loss(vt[:, :self.dx], truevt[:, :self.dx])
        action_loss = mse_loss(vt[:, self.dx:], truevt[:, self.dx:])
        kl_loss = self.standard_normal_kl_div(noise_mean, noise_logvar)
        return {
            'loss': flow_loss + .01 * kl_loss,
            'flow_loss': flow_loss,
            'kl_loss': kl_loss,
            'action_loss': state_loss,
            'state_loss': action_loss
        }

    def _sample(self, context=None, condition=None, mask=None, H=None, noise=None):
        N = context.shape[0]
        if H is None:
            H = self.horizon

        sigma_save = self.FM.sigma
        self.FM.sigma = 0
        num_sub_trajectories = H // self.horizon
        sample_shape = torch.Size([N, num_sub_trajectories])
        assert context.shape[1] == num_sub_trajectories

        # if noise is None:
        #     prior = torch.distributions.Normal(self.prior_mu, self.prior_sigma)
        #     noise = prior.sample(sample_shape=sample_shape).to(dtype=context.dtype, device=context.device)

        # # if longer horizon, set noise of starts / ends to be same value
        # for i in range(num_sub_trajectories - 1):
        #     noise[:, i, -1] = noise[:, i + 1, 0]

        self.noise = condition[0][1].clone()
        self.noise = self.noise.reshape(1, -1).repeat(N, 1)


        #TODO: Is this defined properly?
        self.model.delta_goal = torch.zeros((N, 3), dtype=self.noise.dtype, device=self.noise.device)
        self.model.delta_goal[:, -1] = -torch.pi/2
        self.model.delta_goal -= self.x_mean[12:15]
        self.model.delta_goal /= self.x_std[12:15]

        with torch.no_grad():
            noise_mean, noise_logvar = self.model.noise_dist(self.noise)
        # torch.manual_seed(234)
        pred_noise_u0 = noise_mean + torch.randn_like(noise_mean) * torch.exp(.5 * noise_logvar)
        # pred_noise_u0 = torch.randn_like(noise_mean)
        self.noise = torch.cat((self.noise, pred_noise_u0), dim=-1)
        # manually set the conditions
        if condition is not None:
            self.condition = condition
            # noise[:, 0] = self._apply_conditioning(noise[:, 0], condition)
            # self._grad_mask = mask
            # if num_sub_trajectories > 1:
                # self._grad_mask = torch.cat((self._grad_mask[:, None],
                #                              torch.ones(N, num_sub_trajectories - 1,
                #                                         self.horizon, self.xu_dim, device=mask.device)), dim=1)
        else:
            self.condition = None
            # self._grad_mask = torch.ones_like(noise)
        # self._grad_mask = self._grad_mask.to(device=self.noise.device)
        # self.noise = self._apply_conditioning(self.noise, None, condition=condition)
        log_prob = torch.zeros(N * num_sub_trajectories, device=self.noise.device)

        self.step_id = 0
        with torch.no_grad():
            a = time.perf_counter()
            self.model.reset_z()
            out = self.flow(self.noise,
                            logpx=log_prob,
                            context=context.reshape(-1, context.shape[-1]),
                            reverse=False,
                            integration_times=torch.linspace(0, 1, self.horizon, device=self.noise.device))
            print(f'Elapsed time: {time.perf_counter() - a}')

        trajectories, log_prob = out[:2]
        trajectories = trajectories.permute(1, 0, 2)
        self.FM.sigma = sigma_save

        return trajectories.reshape(N, -1, self.xu_dim), log_prob.reshape(N, -1).sum(dim=1)

    def project(self, H=None, condition=None, context=None):
        N = context.shape[0]
        x = condition[0][1]
        x.requires_grad = True
        optimizer = torch.optim.SGD([x], lr=3e-2, momentum=0.9)
        all_samples = []
        all_losses = []
        all_likelihoods = []
        for proj_t in tqdm(range(10)):
            optimizer.zero_grad()
            # Sample N trajectories
            samples, likelihoods = self._sample(H, condition=condition, context=context)
            all_samples.append(samples.clone().detach())
            all_likelihoods.append(likelihoods.clone().detach())
            if proj_t == 0:
                samples_0 = samples.clone().detach()
            likelihoods_loss = -likelihoods.mean()
            all_losses.append(likelihoods_loss.item())
            print(f'Projection step: {proj_t}, Loss: {likelihoods_loss.item()}')
            likelihoods_loss.backward()
            # x.grad[:, -2:] = 0.0

            optimizer.step()
        samples, likelihoods = self.sample(N, H, condition=condition, context=context, no_grad=False)
        all_samples.append(samples.clone().detach())
        all_likelihoods.append(likelihoods.clone().detach())
        likelihoods_loss = -likelihoods.mean(0)
        all_losses.append(likelihoods_loss.item())
        print(f'Projection step: {proj_t+1}, Loss: {likelihoods_loss.item()}')
        likelihoods_loss.backward()
        best_sample = all_samples[np.argmin(all_losses)]
        best_likelihood = all_likelihoods[np.argmin(all_losses)]
        return (best_sample, best_likelihood), samples_0, (all_losses, all_samples, all_likelihoods)

    def _apply_conditioning(self, x, condition=None):
        if condition is None:
            return x

        for t, (start_idx, val) in condition.items():
            val = val.reshape(val.shape[0], -1, val.shape[-1])
            n, h, d = val.shape
            x[:, t:t + h, start_idx:start_idx + d] = val.clone()
        return x

    def log_prob(self, xu, context, noise=False):
        # inflate with noise
        B, T, _ = xu.shape
        if noise:
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

    def diffusion_grad_wrapper(self, t, xu, context):
        s = self.model(t, xu, context)

        Ts, grad_Ts = self._get_Ts(1 - t)

        alpha_1_minus_t = torch.exp(-Ts / 2)
        sigma_t = torch.sqrt(1 - alpha_1_minus_t ** 2) * torch.ones_like(xu)
        return s / sigma_t


import matplotlib.pyplot as plt

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--noise', type=float, default=0.0)
    argparser.add_argument('--train', action='store_true')
    argparser.add_argument('--project', action='store_true')
    args = argparser.parse_args()
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
    # x = x + torch.randn_like(x) * 0.01
    # x = 0.25 * torch.randn(N, 2, device=device)
    # reshape so horizon is 1
    x = x.reshape(N, 1, 2)

    plt.scatter(np.array(x[:, 0, 0].cpu().tolist()), np.array(x[:, 0, 1].cpu().tolist()))
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.show()

    # let's say no context
    model = TrajectoryCNF(horizon=1, x_dim=2, u_dim=0, context_dim=0, inflation_noise=args.noise).to(device)

    if args.train:
        sigma=.01
        FM = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
        # FM = s_.1OptimalTransportConditionalFlowMatcher()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for i in tqdm(range(2001)):
            # shuffle x
            x = x[torch.randperm(len(x))]
            x_batch = x.reshape(-1, 250, 1, 2)
            total_loss = 0
            model.train()
            for xb in x_batch:
                # loss = model.flow_matching_loss(xb, None)
                # loss, _ = model.log_prob(xb, None)
                # loss = -loss.mean()
                x0 = torch.randn_like(xb)
                t, xut, truet = FM.sample_location_and_conditional_flow(x0, xb + torch.randn_like(xb) * args.noise)

                vt = model.model.compiled_conditional_test(t.reshape(-1), xut, None)[0]
                # vt = model.model(t.reshape(-1), xut, None)[0]

                loss = torch.mean((vt - truet) ** 2)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            model.eval()
            if i % 100 == 0:
                total_ll = 0
                all_z = []
                for xb in x_batch:
                    ll, z = model.log_prob(xb, None, noise=True)
                    ll = ll.mean()
                    all_z.append(z)
                    total_ll += ll.item()
                print(f'iter {i}, loss {total_loss / len(x_batch)}, ll {total_ll / len(x_batch)}')
                all_z = torch.stack(all_z, dim=0).reshape(-1, 2).detach().cpu().numpy()
                # Now let's sample from the model and see how things look
                x_hat, log_prob = model._sample(None)
                x_hat = x_hat.reshape(-1, 2).detach().cpu().numpy()
                x_plot = x.reshape(-1, 2).detach().cpu().numpy()

                lims = [-3, 3]
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))

                axes[0, 0].scatter(x_plot[:, 0], x_plot[:, 1], label='data', alpha=0.25)
                axes[0, 0].scatter(x_hat[:, 0], x_hat[:, 1], label='model', alpha=.75)
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
                Nplot = 125
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
                plt.savefig(f'{fpath}/plots/test_gaussian_inflate_{args.noise}_{i}_s_.01.png')
                plt.close()

                # Save model
                torch.save(model.state_dict(), f'{fpath}/models/test_gaussian_inflate_{args.noise}_{i}_s_.01.pt')

    args.project = True
    if args.project:
        plt.clf()

        # Set dpi
        plt.rcParams['figure.dpi'] = 300
        # Load model
        model.load_state_dict(torch.load(f'{fpath}/models/test_gaussian_inflate_{args.noise}_2000_s_.01.pt'))
            
        # Intentionally generate OOD data

        N_OOD = 100
        x_ood = (torch.randn(N_OOD, 1, 2) * 3).to(device)
        x_ood_hat, log_prob_ood = model._sample(noise=x_ood)

        x_ood_hat_orig = x_ood_hat.clone()

        # Copy the data from x_ood_hat into a new tensor with requires_grad=True
        x_ood_hat = x_ood_hat.detach().requires_grad_(True)
        
        project_iters = 25
        projection_path = [x_ood_hat.clone().detach().cpu().numpy()]
        times = []
        for i in tqdm(range(project_iters)):
            a = time.perf_counter()
            log_prob, _ = model.log_prob(x_ood_hat, None)
            loss = -log_prob.mean()
            loss.backward()
            x_ood_hat.data = x_ood_hat.data - 1 * x_ood_hat.grad
            b = time.perf_counter()
            x_ood_hat.grad.zero_()
            projection_path.append(x_ood_hat.clone().detach().cpu().numpy())
            times.append(b - a)
        print(f'Average time per iteration: {np.mean(times)}')
        
        with open(f'{fpath}/data/test_gaussian_inflate_{args.noise}_OOD_proj_s_.01.pkl', 'wb') as f:
            pickle.dump((projection_path, times), f)
        x_ood_hat = x_ood_hat.detach().cpu().numpy()
        x_ood_hat_orig = x_ood_hat_orig.detach().cpu().numpy()

        # Plot the original data
        plt.scatter(np.array(x[:, 0, 0].cpu().tolist()), np.array(x[:, 0, 1].cpu().tolist()), label='ID data', alpha=1)

        x_ood_hat = x_ood_hat.reshape(-1, 2)
        x_ood_hat_orig = x_ood_hat_orig.reshape(-1, 2)

        for t in range(len(projection_path)):
            this_project_step = projection_path[t]
            this_project_step = this_project_step.reshape(-1, 2)
            if t > 0:
                last_project_step = projection_path[t - 1]
                last_project_step = last_project_step.reshape(-1, 2)
            if t == 0:
                # Plot the orig OOD data
                plt.scatter(this_project_step[:, 0], this_project_step[:, 1], label='OOD gen data', c='r', alpha=.5)
            elif t > 0:
                # Plot a grey line connecting this step to the previous step
                plt.plot([last_project_step[:, 0], this_project_step[:, 0]], [last_project_step[:, 1], this_project_step[:, 1]], c='grey', alpha=0.5)
            if t == len(projection_path) - 1:
                # Plot the OOD data after optimization
                plt.scatter(this_project_step[:, 0], this_project_step[:, 1], label='ID Proj gen data', c='g', alpha=.5)

            # Draw an arrow from the original data to the optimized data
            # for i in range(N_OOD):
            #     plt.arrow(x_ood_hat_orig[i, 0], x_ood_hat_orig[i, 1], x_ood_hat[i, 0] - x_ood_hat_orig[i, 0], x_ood_hat[i, 1] - x_ood_hat_orig[i, 1], head_width=0.05, head_length=0.1, fc='k', ec='k')

        x_lim_min = min(x_ood_hat_orig[:, 0].min(), x_ood_hat[:, 0].min()) - .25
        x_lim_max = max(x_ood_hat_orig[:, 0].max(), x_ood_hat[:, 0].max()) + .25
        y_lim_min = min(x_ood_hat_orig[:, 1].min(), x_ood_hat[:, 1].min()) - .25
        y_lim_max = max(x_ood_hat_orig[:, 1].max(), x_ood_hat[:, 1].max()) + .25

        square_min = min(x_lim_min, y_lim_min)
        square_max = max(x_lim_max, y_lim_max)

        square_bound = max(abs(square_min), abs(square_max))

        plt.xlim([-square_bound, square_bound])
        plt.ylim([-square_bound, square_bound])

        plt.legend()

        plt.savefig(f'{fpath}/plots/test_gaussian_inflate_{args.noise}_OOD_proj_s_.01.png')
        plt.close()