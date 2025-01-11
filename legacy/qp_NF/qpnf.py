import torch
import torch.nn.functional as F
from torch import nn
from qpth.qp import QPFunction

from odefunc import ODEfunc, RegularizedODEfunc, quadratic_cost, jacobian_frobenius_regularization_fn
from functorch import jacrev, vmap

from torchdiffeq import odeint_adjoint as odeint


class Symmetric(nn.Module):
    def forward(self, X):
        return 0.5 * (X.triu() + X.triu(1).transpose(-1, -2))


class MatrixExponential(nn.Module):
    def forward(self, X):
        print(X.shape)
        return torch.linalg.matrix_exp(X)


class SPD(nn.Module):
    def __init__(self, dx):
        super().__init__()
        self.dx = dx
        self.sym = Symmetric()
        self.matrix_exp = MatrixExponential()

    def forward(self, x):
        x = x.reshape(-1, self.dx, self.dx)
        return x @ x.permute(0, 2, 1)


import time

class QPDiffEq(nn.Module):
    def __init__(self, input_size, num_eq_constraints, num_ineq_constraints):
        super().__init__()
        if num_eq_constraints > 0:
            self.G = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, num_eq_constraints)
            )
        else:
            self.G = None
        if num_ineq_constraints > 0:
            self.H = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, num_ineq_constraints)
            )
        else:
            self.H = None

        self.dF = nn.Sequential(nn.Linear(input_size, 64),
                                nn.ReLU(),
                                nn.Linear(64, 64),
                                nn.ReLU(),
                                nn.Linear(64, input_size))

        self.HF = nn.Sequential(nn.Linear(input_size, 64),
                                nn.ReLU(),
                                nn.Linear(64, 64),
                                nn.ReLU(),
                                nn.Linear(64, input_size * input_size),
                                SPD(input_size))

        self.num_eq_constraints = num_eq_constraints
        self.num_ineq_constraints = num_ineq_constraints

        self.Q = None
        self.dx = input_size
        self.eta = 1

    def forward(self, t, x):
        N, d = x.shape
        #return self.dF(x)
        if self.Q is None:
            self.Q = torch.eye(self.dx).unsqueeze(0).repeat(N, 1, 1).to(x)

        """ solves QP to get descent direction """
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)

            if self.G is not None:
                A = vmap(jacrev(self.G))(x)
                #A = torch.autograd.functional.jacobian(self.G, x, create_graph=True)
                #A = torch.diagonal(A, dim1=-2, dim2=-1)
                b = - self.G(x)
            else:
                A = torch.autograd.Variable(torch.Tensor())
                b = torch.autograd.Variable(torch.Tensor())
            if self.H is not None:
                G = vmap(jacrev(self.H))(x)
                #G = torch.autograd.functional.jacobian(self.G, x, create_graph=True)
                #G = torch.diagonal(G, dim1=-2, dim2=-1)
                h = -self.H(x)
            else:
                G = torch.autograd.Variable(torch.Tensor())
                h = torch.autograd.Variable(torch.Tensor())

        #Q = self.HF(x)
        #Q = self.HF(x)
        #Q = Q# @ Q.permute(0, 2, 1)
        Q = self.Q
        p = self.dF(x)

        dx, lams, nus = QPFunction(verbose=False)(Q.double(), p.double(), G.double(), h.double(), A.double(),
                                                  b.double())
        dx = self.eta * dx.float()
        if False:
            HF = vmap(jacrev(self.dF))(x+dx)
            HG = vmap(jacrev(jacrev(self.G)))(x+dx)
            HH = vmap(jacrev(jacrev(self.H)))(x+dx)

            self.Q = self.Q + HF + (lams.reshape(N, -1, 1, 1) * HH + nus.reshape(N, -1, 1, 1) * HG).reshape(N, d, d)

            # try to make SPD (minimizes fronemius norm difference)
            self.Q = 0.5 * (self.Q + self.Q.permute(0, 2, 1))
            try:
                D, E = torch.linalg.eigh(self.Q)
            except:
                print(self.Q)
            D = torch.diag_embed(torch.maximum(D.real, torch.zeros_like(D.real)))
            self.Q = E @ D @ E.permute(0, 2, 1) + 1e-6 * torch.eye(d).to(x).unsqueeze(0)

        if False:
            lams = lams.float()
            nus = nus.float()

            eps = 1
            next_dF = self.dF(x + eps * dx)
            next_dG = vmap(jacrev(self.G))(x + eps * dx)
            next_dH = vmap(jacrev(self.H))(x + eps * dx)

            dLagrangian = (self.Q @ x.unsqueeze(-1)).squeeze(-1) + p + (lams.unsqueeze(1) @ G).reshape(N, d) + (
                        nus.unsqueeze(1) @ A).reshape(N, d)
            next_dLagrangian = (self.Q @ (x + dx).unsqueeze(-1)).squeeze(-1) + next_dF + (
                        lams.unsqueeze(1) @ next_dH).reshape(N, d) + (nus.unsqueeze(1) @ next_dG).reshape(
                N, d)

            y = next_dLagrangian - dLagrangian
            theta = 0.01
            y = torch.where(bdot(y, dx).unsqueeze(1) < 0,
                            theta * y + (1 - theta) * (self.Q @ dx.unsqueeze(2)).squeeze(2),
                            y
                            )

            term1 = b_outer(y, y) / bdot(y, dx).reshape(N, 1, 1)
            term2_num = self.Q @ b_outer(dx, dx) @ self.Q
            term2_den = dx.unsqueeze(1) @ self.Q @ dx.unsqueeze(2)

            self.Q = self.Q + term1 - term2_num / term2_den

        return dx

class NullSpaceFlowDiffEq(nn.Module):
    def __init__(self, input_size, num_eq_constraints):
        super().__init__()
        hidden_dim = 128
        if num_eq_constraints > 0:
            #self.G = nn.Sequential(
            #    nn.Linear(input_size, hidden_dim),
            #    nn.ReLU(),
            #    nn.Linear(hidden_dim, hidden_dim),
            #    nn.ReLU(),
            #    nn.Linear(hidden_dim, num_eq_constraints)
            #)
            self.G = nn.Linear(input_size, num_eq_constraints, bias=False)
        else:
            self.G = None

        self.dF = nn.Sequential(nn.Linear(input_size, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, input_size))

        self.num_eq_constraints = num_eq_constraints

        self.Q = None
        self.dx = input_size
        self.eta = 1

    def forward(self, t, x):
        N, d = x.shape

        """ solves QP to get descent direction """
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            dG = vmap(jacrev(self.G))(x)

        g = self.G(x)
        dF = self.dF(x).unsqueeze(-1)

        eta_J_null = dG.permute(0, 2, 1) @ torch.linalg.solve(dG @ dG.permute(0, 2, 1), dG)
        eta_J = dF - eta_J_null @ dF
        eta_C = dG.permute(0, 2, 1) @ torch.linalg.solve(dG @ dG.permute(0, 2, 1), g.unsqueeze(-1))
        dx = -eta_J - eta_C
        #print(t, g.abs().max())
        return dx.squeeze(-1)

class QPNF(nn.Module):

    def __init__(self, input_size, num_eq_constraints, num_ineq_constraints, regularization_fns=[quadratic_cost,
                                                                                                 jacobian_frobenius_regularization_fn]):
        super().__init__()
        #self.diffeq = QPDiffEq(input_size, num_eq_constraints, num_ineq_constraints)
        self.diffeq = NullSpaceFlowDiffEq(input_size, num_eq_constraints)
        odefunc = ODEfunc(self.diffeq)
        nreg = 0
        if regularization_fns is not None:
            odefunc = RegularizedODEfunc(odefunc, regularization_fns)
            nreg = len(regularization_fns)

        self.odefunc = odefunc
        self.nreg = nreg

        self.solver = 'dopri5'
        #self.solver = 'rk4'
        self.solver_options = {}
        self.atol = 1e-5
        self.rtol = 1e-5
        self.test_atol = 1e-5
        self.test_rtol = 1e-5
        self.dx = input_size
        self.Q = None
        self.T = nn.Parameter(torch.tensor([1.0]))
        self.T.requires_grad = True

    def forward(self, z, logpz=None, reg_states=tuple(), reverse=False):

        N, _ = z.shape
        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if not len(reg_states) == self.nreg and self.training:
            reg_states = tuple(torch.zeros(z.size(0)).to(z) for i in range(self.nreg))

        steps = 2
        integration_times = torch.cat((torch.zeros(1).to(z), self.T))

        if reverse:
            integration_times = torch.cat((self.T, torch.zeros(1).to(z)))

        self.odefunc.before_odeint()
        self.diffeq.Q = None#torch.eye(self.dx).unsqueeze(0).repeat(N, 1, 1).to(z)

        if self.training:
            if self.training:
                state_t = odeint(
                    self.odefunc,
                    (z, _logpz) + reg_states,
                    integration_times,
                    atol=[self.atol, self.atol] + [1e20] * len(reg_states) if self.solver in ['dopri5',
                                                                                              'bosh3'] else self.atol,
                    rtol=[self.rtol, self.rtol] + [1e20] * len(reg_states) if self.solver in ['dopri5',
                                                                                              'bosh3'] else self.rtol,
                    method=self.solver,
                    options=self.solver_options,
                    adjoint_atol=self.atol,
                    adjoint_rtol=self.rtol
                )
        else:
            integration_times[-1] *= 2
            state_t = odeint(
                self.odefunc,
                (z, _logpz),
                integration_times,
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.solver,
                options=self.solver_options,
            )

        state_t = tuple(s[-1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        reg_states = state_t[2:]

        if logpz is not None:
            return z_t, logpz_t, reg_states

        return z_t


def bdot(x, y):
    return torch.bmm(y.unsqueeze(1), x.unsqueeze(2)).reshape(-1)


def b_outer(x, y):
    N, d = x.shape
    return torch.bmm(y.unsqueeze(2), x.unsqueeze(1)).reshape(N, d, d)


import math


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


if __name__ == '__main__':

    # gaussian data on a manifold
    num_batches = 10
    batch_size = 128
    N = num_batches * batch_size

    xy = torch.randn(N, 2)
    xyz = torch.cat((xy, torch.zeros(N, 1)), dim=1)

    # generate random matrix
    A = torch.randn(3, 3)
    #A = torch.tensor([[1.7297, -0.4128, -0.7133],
    #                  [-0.6333, 1.1659, -0.8268],
    #                  [0.7509, -0.6883, 1.1011]])

    import scipy
    z = A @ torch.tensor([[1.0, 0.0],
                          [0.0, 1.0],
                          [0.0, 0.0]])
    w = scipy.linalg.null_space(z.numpy().T)
    X = (A @ xyz.unsqueeze(-1)).squeeze(-1)


    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    xn = X.numpy()
    device = 'cuda:0'
    nf = QPNF(3, 1, 1, regularization_fns=[]).to(device=device)
    #nf = QPNF(3, 1, 1).to(device=device)
    #print(nf.diffeq.G.weight.data.shape)
    #print(w.shape)
    #nf.diffeq.G.weight.data = torch.from_numpy(w.T).to(device)

    X = X.to(device=device).reshape(num_batches, batch_size, 3)

    optim = torch.optim.Adam(nf.parameters(), lr=1e-3)
    for _ in range(100):
        xy = torch.randn(N, 2)
        xyz = torch.cat((xy, torch.zeros(N, 1)), dim=1)
        X = (A @ xyz.unsqueeze(-1)).squeeze(-1)
        X = X.to(device=device).reshape(num_batches, batch_size, 3)

        ep_loss = 0.0
        ep_g_loss = 0.0
        for x in X:
            g = (nf.diffeq.G(x)**2).mean()
            x = x + 0.01 * torch.randn_like(x)
            z, delta_logprob, reg = nf(x, torch.zeros(batch_size).to(X), reverse=True)
            # now we re-sample - samples should obey manifold
            xhat = nf(z, reverse=False)
            logpz = standard_normal_logprob(z).sum(dim=1)
            logpz = logpz - delta_logprob

            loss = -logpz.mean()
            loss = loss + 100 * g
            if len(reg) > 0:
                loss = loss + 0.01 * reg[0].mean() + 0.01 * reg[1].mean()
            loss.backward()
            #nf.diffeq.G.weight.grad = None
            optim.step()
            torch.nn.utils.clip_grad_norm_(nf.parameters(), 1)
            ep_loss += loss.item()
            ep_g_loss += g.item()
            optim.zero_grad()


        ep_loss /= num_batches
        ep_g_loss /= num_batches
        print(
            'iter', _, 'loss', ep_loss, 'g', ep_g_loss
        )
    print('learned manifold', nf.diffeq.G.weight)
    print('actual manifold', w)

    # now we sample and check
    Z = torch.randn(num_batches, batch_size, 3).to(device=device)
    Xhat = []
    nf.eval()
    for z in Z:
        x = nf(z, reverse=False)
        Xhat.append(x)

    Xhat = torch.cat(Xhat, dim=0).cpu().detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(xn[:, 0], xn[:, 1], xn[:, 2], color='b')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(Xhat[:, 0], Xhat[:, 1], Xhat[:, 2], color='r')
    plt.show()

    import numpy as np
    x = np.linspace(-4.0, 4.0, num=100)
    xx, yy, zz = np.meshgrid(x, x, x)

    X = np.stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)), axis=1)

    with torch.no_grad():
        constraint_val = nf.diffeq.G(torch.from_numpy(X).float().to(device=device))
        constraint_val = constraint_val

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    constraint_val = constraint_val.abs().reshape(-1).cpu().numpy()
    idx = constraint_val < 1e-3
    print(idx.shape)
    ax.scatter(xx.reshape(-1)[idx], yy.reshape(-1)[idx], zz.reshape(-1)[idx])
    plt.show()

