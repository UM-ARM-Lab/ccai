import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ccai.models.cnf.ffjord.layers import diffeq_layers
from ccai.models.cnf.ffjord.layers.squeeze import squeeze, unsqueeze
from torch.func import jacrev, vmap

__all__ = ["ODEnet", "ODEfunc", 'RegularizedODEfunc']


def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


# def divergence_bf(f, y, **unused_kwargs):
#     jac = _get_minibatch_jacobian(f, y)
#     diagonal = jac.view(jac.shape[0], -1)[:, ::jac.shape[1]]
#     return torch.sum(diagonal, 1)


def _get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.

    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(y[:, j], x, torch.ones_like(y[:, j]), retain_graph=True,
                                      create_graph=True)[0].view(x.shape[0], -1)
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac


'''
def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx
'''


def divergence_approx(f, y, e=None):
    samples = []
    sqnorms = []
    for e_ in e:
        e_dzdx = torch.autograd.grad(f, y, e_, create_graph=True)[0]
        n = e_dzdx.reshape(y.size(0), -1).pow(2).mean(dim=1, keepdim=True)
        sqnorms.append(n)
        e_dzdx_e = e_dzdx * e_
        samples.append(e_dzdx_e.reshape(y.shape[0], -1).sum(dim=1, keepdim=True))

    S = torch.cat(samples, dim=1)
    approx_tr_dzdx = S.mean(dim=1)

    N = torch.cat(sqnorms, dim=1).mean(dim=1)

    return approx_tr_dzdx, N


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x ** 2),
    "identity": Lambda(lambda x: x),
}


class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(
            self, hidden_dims, input_dim, context_dim, strides, conv, layer_type="concat", nonlinearity="softplus",
            num_squeeze=0
    ):
        super(ODEnet, self).__init__()
        self.num_squeeze = num_squeeze
        if conv:
            assert len(strides) == len(hidden_dims) + 1
            base_layer = {
                "ignore": diffeq_layers.IgnoreConv2d,
                "hyper": diffeq_layers.HyperConv2d,
                "squash": diffeq_layers.SquashConv2d,
                "concat": diffeq_layers.ConcatConv2d,
                "concat_v2": diffeq_layers.ConcatConv2d_v2,
                "concatsquash": diffeq_layers.ConcatSquashConv2d,
                "blend": diffeq_layers.BlendConv2d,
                "concatcoord": diffeq_layers.ConcatCoordConv2d,
            }[layer_type]
        else:
            strides = [None] * (len(hidden_dims) + 1)
            base_layer = {
                "ignore": diffeq_layers.IgnoreLinear,
                "hyper": diffeq_layers.HyperLinear,
                "squash": diffeq_layers.SquashLinear,
                "concat": diffeq_layers.ConcatLinear,
                "concat_v2": diffeq_layers.ConcatLinear_v2,
                "concatsquash": diffeq_layers.ConcatSquashLinear,
                "blend": diffeq_layers.BlendLinear,
                "concatcoord": diffeq_layers.ConcatLinear,
            }[layer_type]

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = (input_dim,)

        for dim_out, stride in zip(hidden_dims + (input_dim,), strides):
            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {"ksize": 3, "stride": 1, "padding": 1, "transpose": False}
            elif stride == 2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": False}
            elif stride == -2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": True}
            else:
                raise ValueError('Unsupported stride: {}'.format(stride))

            layer = base_layer(hidden_shape[0], dim_out, context_dim, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] // 2, hidden_shape[2] // 2
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] * 2, hidden_shape[2] * 2

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t, y, context=None):
        dx = y
        # squeeze
        for _ in range(self.num_squeeze):
            dx = squeeze(dx, 2)
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx, context)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        # unsqueeze
        for _ in range(self.num_squeeze):
            dx = unsqueeze(dx, 2)
        return dx


batch_trace = vmap(torch.trace)


class ODEfunc(nn.Module):

    def __init__(self, diffeq, divergence_fn="approximate", residual=False, rademacher=False):
        super(ODEfunc, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")

        # self.diffeq = diffeq_layers.wrappers.diffeq_wrapper(diffeq)
        self.diffeq = diffeq

        diffeq_fn = lambda t, y, context: self.diffeq.vmapped_fwd(t, y, context)

        self.residual = residual
        self.rademacher = rademacher

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx
        #self.divergence_fn = vmap(jacrev(diffeq_fn, argnums=1))
        self.register_buffer("_num_evals", torch.tensor(0.))
        self.div_samples = 1

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)
        self._sqjacnorm = None

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        if not torch.is_tensor(t):
            t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]

        if self._e is None:
            if self.rademacher:
                self._e = [sample_rademacher_like(y) for k in range(self.div_samples)]
            else:
                self._e = [sample_gaussian_like(y) for k in range(self.div_samples)]

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            # Hack for 2D data to use brute force divergence computation.
            # if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
            #divergence = self.divergence_fn(t.reshape(1, 1).repeat(y.shape[0], 1), y, states[2])
            #yshape = torch.prod(torch.tensor(y.shape[1:]))
            #divergence = batch_trace(divergence.reshape(batchsize, yshape, yshape))
            divergence = torch.zeros(batchsize, 1).to(y)
            # else:
            #    divergence, sqjacnorm = self.divergence_fn(dy, y, e=self._e)
            #    divergence = divergence.view(batchsize, 1)
            #    self.sqjacnorm = sqjacnorm

        if self.residual:
            dy = dy - y
            divergence -= torch.ones_like(divergence) * torch.tensor(np.prod(y.shape[1:]), dtype=torch.float32
                                                                     ).to(divergence)

        return tuple([dy, -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])


class RegularizedODEfunc(nn.Module):
    def __init__(self, odefunc):
        super(RegularizedODEfunc, self).__init__()
        self.odefunc = odefunc
        self.regularization_fns = [quadratic_cost, jacobian_frobenius_regularization_fn]

    def before_odeint(self, *args, **kwargs):
        self.odefunc.before_odeint(*args, **kwargs)

    def forward(self, t, state):
        if self.training:
            with torch.enable_grad():

                x, logp = state[:2]
                x.requires_grad_(True)
                t.requires_grad_(True)
                logp.requires_grad_(True)

                # check if context involved
                if len(state) > 2 + len(self.regularization_fns):
                    context = state[2]
                    ode_input = (x, logp, context)
                else:
                    ode_input = (x, logp)
                    context = None
                dstate = self.odefunc(t, ode_input)
                if len(state) > 3:
                    if context is None:
                        dx, dlogp = dstate[:2]
                        dstate = tuple([dx, dlogp])
                    else:
                        dx, dlogp, context = dstate[:3]
                        dstate = tuple([dx, dlogp, context])

                    reg_states = tuple(
                        reg_fn(x, t, logp, dx, dlogp, self.odefunc) for reg_fn in self.regularization_fns)
                    s = dstate + reg_states
                    return s
                else:
                    return dstate
        else:
            return self.odefunc(t, state)

    @property
    def _num_evals(self):
        return self.odefunc._num_evals


def quadratic_cost(x, t, logp, dx, dlogp, unused_context):
    del x, logp, dlogp, t, unused_context
    dx = dx.reshape(dx.shape[0], -1)
    return 0.5 * dx.pow(2).mean(dim=-1)


def jacobian_frobenius_regularization_fn(x, t, logp, dx, dlogp, context):
    sh = x.shape
    del logp, dlogp, t, dx, x
    sqjac = context.sqjacnorm

    return context.sqjacnorm
