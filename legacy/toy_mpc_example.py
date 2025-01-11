import torch
import numpy as np
import matplotlib.pyplot as plt
from nullspace_optimizer import *
from legacy.nullspace_optimizer_pytorch import nlspace_solve as nlspace_solve_pytorch
import osqp

from functorch import jacrev

torch.manual_seed(1234)

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['shooting', 'collocation', 'shooting_sqp', 'collocation_sqp'],
                        default='shooting')
    parser.add_argument('--iters', type=int, default=100)
    return parser.parse_args()


def rbf_kernel(x, xbar, lengthscale=1.):
    diff = (x.unsqueeze(0) - xbar.unsqueeze(1))
    diff = ((diff / lengthscale) ** 2).sum(dim=-1)
    return torch.exp(-0.5 * diff)


# we have a constraint on the position - must lie on unit circle
def state_constraint(x):
    return x[:, 0] ** 2 + x[:, 1] ** 2 - 1.0


# have constraint on controls -- must be below unit norm
def control_constraint(u):
    return (u ** 2).sum(dim=-1) - 1.0


def dynamics(x, u):
    dt = 0.25
    #u = constrained_controller(x, u)
    # x is [x, y, xdot, ydot]
    A = torch.tensor([
        [1.0, 0.0, dt, 0.0],
        [0., 1.0, 0.0, dt],
        [0.0, 0.0, 0.9, 0.0],
        [0.0, 0.0, 0.0, 0.9]
    ]).to(x)
    B = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [dt, 0.0],
            [0.0, dt]
        ]
    ).to(x)
    new_x = A.unsqueeze(0) @ x.unsqueeze(-1) + B.unsqueeze(0) @ u.unsqueeze(-1)
    # new_x = new_x + torch.randn_like(new_x) * 0.01
    return new_x.squeeze(-1)


def bdot(x, y):
    return torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).reshape(-1)


def constrained_controller(x, u):
    # takes current control u and projects it into half plane defined by circle tangent, i.e. removes components
    # of control that move into the circle

    # x should be N x 4
    # u should be N x 2
    xy = x[:, :2]
    tangent_vector = torch.stack((-xy[:, 1], xy[:, 0]), dim=1)

    # first check if we are on the right side of the halfplane - should be
    coeff = bdot(u, tangent_vector) / bdot(tangent_vector, tangent_vector)
    projected_u = coeff.unsqueeze(1) * tangent_vector
    # only project us that are pointing towards centre
    constrained_u = torch.where(bdot(xy, u).reshape(-1, 1) < 0, projected_u, u)

    # only when x is approximately on the surface
    distance_to_surface = (xy[:, 0] ** 2 + xy[:, 1] ** 2).unsqueeze(1)

    constrained_u = torch.where(distance_to_surface < 1.00, constrained_u, u)

    return constrained_u


def trajectory_cost(u_sequence,
                    start=torch.tensor([[0.0, -1, 0.0, 0.0]]),
                    goal=torch.tensor([[0.0, 1.0, 0.0, 0.0]])
                    ):
    # u is N x T x du
    N, T, du = u_sequence.shape
    cost = 0
    x = start.repeat(N, 1)
    for t in range(T):
        if du == 2:
            u = u_sequence[:, t]
        else:
            u = u_sequence[:, t].reshape(-1, 2, 5)[:, :, :-1] @ x[-1].unsqueeze(-1)
            u = u.squeeze(-1) + u_sequence[:, t].reshape(-1, 2, 5)[:, :, -1]

        x = dynamics(x, u)
        diff = x - goal
        if t < T - 1:
            diff[:, 2:] *= 0.1

        running_cost = (diff ** 2).sum(dim=-1)

        if t == (T - 1):
            running_cost = running_cost * 10
        # running_cost = running_cost + 0.1 * (u ** 2).sum(dim=1)
        cost = cost + 0.1 * running_cost

    return cost


def trajectory_cost_collocation(x, u,
                                start=torch.tensor([[0.0, -1, 0.0, 0.0]]),
                                goal=torch.tensor([[0.0, 1.0, 0.0, 0.0]])
                                ):
    N, T, _ = u.shape
    diff = x - goal.unsqueeze(0).to(x)
    # diff[:, :, 2:] *= 0.1
    # cost = torch.linalg.norm(diff, dim=-1)# + 0.1 * torch.linalg.norm(u, dim=-1)
    weight = torch.ones_like(diff)
    weight[:, :-1, 2:] *= 0.1  # down weight velocity
    weight[:, :-1] *= 0.1  # downweight running

    cost = weight * (diff ** 2)
    cost = cost.sum(dim=-1)  # + 0.01 * (u ** 2).sum(dim=-1)

    return cost.sum(dim=-1)


def dynamics_constraint(x_new, x_old, u):
    error = x_new - dynamics(x_old, u)
    return error.reshape(-1)


class MPCProblem(EuclideanOptimizable):

    def __init__(self, u0, cost, M, T, udim):
        super().__init__(M * T * udim)
        self.nconstraints = M * T
        self.nineqconstraints = M * T
        self._x0 = u0
        self.M = M
        self.T = T
        self.dx = udim
        self.cost = cost

    def x0(self):
        return self._x0.reshape(-1)

    def J(self, x):
        tx = torch.from_numpy(x).reshape(self.M, -1).float()
        u_sequence = tx.reshape(self.M, self.T, -1)
        return self.cost(u_sequence).sum().numpy()

    def dJ(self, x):
        tx = torch.from_numpy(x).reshape(self.M, -1).float()
        lengthscale = torch.median(tx) / self.M
        tx.requires_grad = True
        # lengthscale = 0.1
        u_sequence = tx.reshape(self.M, self.T, -1)
        log_prob = -self.cost(u_sequence)
        K = rbf_kernel(tx, tx.detach(), lengthscale.detach())

        score = torch.autograd.grad(log_prob.sum(), u_sequence)[0].reshape(self.M, -1)
        grad_K = torch.autograd.grad(K.sum(), tx)[0]

        dJ = K @ score + grad_K

        return -dJ.reshape(-1).detach().numpy() / self.M

    def H(self, x):
        tx = torch.from_numpy(x).reshape(self.M * self.T, -1).float()
        return control_constraint(tx)

    def dH(self, x):
        tx = torch.from_numpy(x).reshape(self.M * self.T, -1).float()
        tx.requires_grad = True
        H = control_constraint(tx)
        dH = torch.autograd.grad(H.sum(), tx)[0]

        # dH is M*T 2 - need to convert to TM x (TMx2) - will mostly be zero vectors
        dH = torch.diag_embed(dH.permute(1, 0)).permute(2, 1, 0).reshape(self.M * self.T, self.M * self.T * self.dx)
        return dH.detach().numpy()

    def G(self, x, grad=False):
        tx = torch.from_numpy(x).reshape(self.M, self.T, -1).float()
        if grad is True:
            tx.requires_grad = True
        x = [torch.tensor([[0.0, -1.0, 0.0, 0.0]]).repeat(self.M, 1)]

        x[-1].requires_grad = True
        for t in range(self.T):
            if tx.shape[-1] == 2:
                u = tx[:, t]
            else:
                u = tx[:, t].reshape(-1, 2, 5)[:, :, :-1] @ x[-1].unsqueeze(-1)
                u = u.squeeze(-1) + tx[:, t].reshape(-1, 2, 5)[:, :, -1]

            x.append(dynamics(x[-1], u))
        x = torch.stack(x[1:], dim=1)

        G = state_constraint(x.reshape(-1, 4))
        if grad is True:
            dG = []
            G = G.reshape(self.M, self.T)
            for t in range(self.T):
                dG.append(torch.autograd.grad(G[:, t].sum(), tx, retain_graph=True)[0])
            return torch.stack(dG, dim=1).detach()

        return G.detach().numpy()

    def dG(self, x):
        dG = self.G(x, grad=True)
        # dG is M*T 2 - need to convert to TM x (TMx2) - will mostly be zero vectors
        dG = torch.diag_embed(dG.permute(3, 2, 1, 0)).permute(4, 2, 3, 1, 0).reshape(self.M * self.T,
                                                                                     self.M * self.T * self.dx)
        # print(dG[10])
        return dG.detach().numpy()


class MPCProblemCollacation(EuclideanOptimizable):

    def __init__(self, x0, cost, M, T):
        super().__init__(M * T * 6)
        self.nconstraints = 4 * M * T + M * (T - 1)
        self.nineqconstraints = 0 * M * T
        self._x0 = x0
        self.M = M
        self.T = T
        self.dx = 4
        self.du = 2
        self.cost = cost

        self.fwd = 1

    def x0(self):
        return self._x0.reshape(-1)

    def J(self, xu):
        tx = torch.from_numpy(xu).reshape(self.M, -1).float()
        txu = tx.reshape(self.M, self.T, -1)
        x = txu[:, :, :self.dx]
        u = txu[:, :, self.dx:]
        return self.cost(x, u).sum()

    def dJ(self, xu):
        tx = torch.from_numpy(xu).reshape(self.M, -1).float()
        lengthscale = torch.median(tx) / self.M
        tx.requires_grad = True
        # lengthscale = 0.1
        txu = tx.reshape(self.M, self.T, -1)
        x = txu[:, :, :self.dx]
        u = txu[:, :, self.dx:]

        log_prob = -self.cost(x, u)
        K = rbf_kernel(tx, tx.detach(), lengthscale.detach())

        score = torch.autograd.grad(log_prob.sum(), txu)[0].reshape(self.M, -1)
        grad_K = torch.autograd.grad(K.sum(), tx)[0]

        dJ = K @ score + grad_K

        return -dJ.reshape(-1).detach().numpy() / self.M

    """
    def HJ(self, xu):
        tx = torch.from_numpy(xu).reshape(self.M, -1).float()
        lengthscale = torch.median(tx) / self.M
        tx.requires_grad = True
        txu = tx.reshape(self.M, self.T, -1)

        def log_prob(xu):
            x = xu[:, :, :self.dx]
            u = xu[:, :, self.dx:]
            return -self.cost(x, u).sum()

        K = rbf_kernel(tx, tx.detach(), lengthscale.detach())
        grad_K = torch.autograd.grad(K.sum(), tx)[0]
        Hcost = torch.autograd.functional.hessian(log_prob, txu).reshape(self.M,
                                                                         self.T * (self.dx + self.du),
                                                                         self.M,
                                                                         self.T * (self.dx + self.du))

        print(Hcost[0, :, 0])
        print(Hcost.shape)
        print(K.shape)
        print(grad_K.shape)

        HJ = Hcost

        # score = torch.autograd.grad(log_prob.sum(), txu)[0].reshape(self.M, -1)

        exit(0)
        # get hessian as derivative of grad
        return HJ
    """

    def H(self, x, grad=False):
        txu = torch.from_numpy(x).reshape(self.M, self.T, -1).float()
        if grad:
            txu.requires_grad = True

        def _constraint(txu):
            u = txu[:, :, self.dx:]
            return control_constraint(u.reshape(-1, self.du))

        if grad:
            dH = torch.autograd.functional.jacobian(_constraint, txu)
            return dH.reshape(self.M * self.T, -1).detach().numpy()
        else:
            H = _constraint(txu)
            return H.detach().numpy()

    def dH(self, x):
        return self.H(x, grad=True)

    def G(self, xu, grad=False):
        txu = torch.from_numpy(xu).reshape(self.M, self.T, -1).float()
        if grad is True:
            txu.requires_grad = True

        def _dynamic_constraints(x, u):
            old_x = torch.cat(
                (torch.tensor([0.0, -1.0, -0.0, 0.0]).reshape(1, 1, 4).repeat(self.M, 1, 1),
                 x[:, :-1]), dim=1
            )
            # if self.fwd:
            #    old_x = old_x.detach()
            #    u = u.detach()
            # else:
            #    x = x.detach()
            return dynamics_constraint(x.reshape(-1, 4), old_x.reshape(-1, 4), u.reshape(-1, 2))

        def _state_constraint(x, u):
            # annoying reason cannot constrain the first vector because of stupid gradients
            return state_constraint(x.reshape(-1, 4))

        def _constraint(xu):
            x = xu[:, :, :self.dx]
            u = xu[:, :, self.dx:]
            # return _dynamic_constraints(x, u).reshape(-1)
            return torch.cat((_dynamic_constraints(x, u), _state_constraint(x[:, 1:], u)), dim=0)

        if grad is True:
            dG = torch.autograd.functional.jacobian(_constraint, txu).reshape(-1, self.M * self.T * (self.dx + self.du))
            # import pandas as pd
            # x_df = pd.DataFrame(dG.detach().numpy())
            # x_df.to_csv('tmp.csv')
            # exit(0)
            return dG.detach().numpy()
        else:
            G = _constraint(txu)
            return G.detach().numpy()

    def dG(self, x):
        self.fwd = not self.fwd  # flip
        return self.G(x, grad=True)

def squared_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Computes squared distance matrix between two arrays of row vectors.
    Code originally from:
        https://github.com/pytorch/pytorch/issues/15253#issuecomment-491467128
    """
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(
        x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
    ).add_(x1_norm)
    return res.clamp(min=0)  # avoid negative distances due to num precision

import time

class MPCProblemCollocationPytorch:
    def __init__(self, x0, cost, M, T):
        self._x0 = torch.from_numpy(x0)
        self.M = M
        self.T = T
        self.dx = 4
        self.du = 2
        self.cost = cost

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x):
        self._x0 = x

    def eval(self, tx):

        tx = tx.reshape(self.M, -1)
        pairwise_dist = squared_distance(tx, tx)
        lengthscale = torch.median(pairwise_dist) / np.log(self.M)
        lengthscale = torch.clamp(lengthscale, min=1e-3)

        txu = tx.reshape(self.M, self.T, -1)
        x = txu[:, :, :self.dx]
        u = txu[:, :, self.dx:]

        log_prob = -self.cost(x, u)
        K = rbf_kernel(tx, tx.detach(), lengthscale.detach())

        score = torch.autograd.grad(log_prob.sum(), txu)[0].reshape(self.M, -1)
        grad_K = torch.autograd.grad(K.sum(), tx)[0]

        dJ = K @ score + grad_K
        dJ = -dJ.reshape(-1) / self.M
        J = -log_prob.mean()


        def _dynamic_constraints(x, u):
            old_x = torch.cat(
                (torch.tensor([0.0, -1.0, -0.0, 0.0]).to(x).reshape(1, 1, 4).repeat(self.M, 1, 1),
                 x[:, :-1]), dim=1
            )
            return dynamics_constraint(x.reshape(-1, 4), old_x.reshape(-1, 4), u.reshape(-1, 2))

        def _state_constraint(x, u):
            # annoying reason cannot constrain the first vector because of stupid gradients
            return state_constraint(x.reshape(-1, 4))

        def Gfn(xu):
            x = xu[:, :, :self.dx]
            u = xu[:, :, self.dx:]
            # return _dynamic_constraints(x, u).reshape(-1)
            return torch.cat((_dynamic_constraints(x, u), _state_constraint(x[:, 1:], u)), dim=0)

        #dG = torch.autograd.functional.jacobian(_constraint, txu).reshape(-1, self.M * self.T * (self.dx + self.du))
        dG = jacrev(Gfn)(txu).reshape(-1, self.M*self.T * (self.dx + self.du))
        G = Gfn(txu)

        def Hfn(txu):
            u = txu[:, :, self.dx:]
            return control_constraint(u.reshape(-1, self.du))

        dH = jacrev(Hfn)(txu).reshape(self.M * self.T, -1)
        #dH = torch.autograd.functional.jacobian(_constraint, txu).reshape(self.M * self.T, -1)
        H = Hfn(txu)

        constraint_time = time.time()
        return J, G, H, dJ, dG, dH


from scipy import sparse


class SQPSolver:

    def __init__(self, problem, iters, use_approx_hessian=True):
        self.problem = problem
        self.iters = iters
        # initialise estimate of HJ
        self.B = np.eye(problem.n)

        self.options = {'verbose': False,
                        'eps_rel': 1e-6, 'eps_abs': 1e-6,
                        'eps_prim_inf': 1e-6, 'eps_dual_inf': 1e-6}
        self.tol = 1e-3

        self.approx_H = use_approx_hessian

    def solve(self):
        # first we use x0

        x = problem.x0()
        B = self.B

        dJkp1 = problem.dJ(x)
        dHkp1 = problem.dH(x)
        dGkp1 = problem.dG(x)

        J_hist = [problem.J(x)]

        for i in range(self.iters):
            dJk = dJkp1
            dHk = dHkp1
            dGk = dGkp1

            G = problem.G(x)
            H = problem.H(x)

            if not self.approx_H:
                B = problem.HJ(x)
                print(B.shape)
                print(B)
            # two inequality constraints for equality constraints (because of affine term)
            A = np.concatenate((dGk, dHk), axis=0)
            A = sparse.csc_matrix(A)

            u = np.concatenate((-G, -H), axis=0)
            l = np.concatenate((-G, -np.inf * np.ones(H.shape[0])), axis=0)

            P = sparse.csc_matrix(B)

            prob = osqp.OSQP()
            prob.setup(P=P, q=dJk, A=A, u=u, l=l, **self.options)
            res = prob.solve()
            dx = res.x
            dual_var = res.y
            dx *= 0.1
            x = x + dx
            # if np.linalg.norm(dx) < self.tol:
            #    break
            dJkp1 = problem.dJ(x)
            dHkp1 = problem.dH(x)
            dGkp1 = problem.dG(x)

            dconstraintkp1 = np.concatenate((dGkp1, dHkp1), axis=0)
            dconstraintk = np.concatenate((dGk, dHk), axis=0)

            y = (dJkp1 + dual_var @ dconstraintkp1) - (dJk + dual_var @ dconstraintk)

            if self.approx_H:
                dx = dx.reshape(-1, 1)
                y = y.reshape(-1, 1)
                theta = 0.01
                if (dx.T @ y).item() < 0:
                    y = theta * y + (1 - theta) * B @ dx
                B = B + y @ y.T / (y.T @ dx) - (B @ dx @ dx.T @ B) / (dx.T @ B @ dx)

            J_hist.append(self.problem.J(x))
            print('av cost', J_hist[-1])

        return x, J_hist


if __name__ == "__main__":

    args = parse_args()
    mode = args.mode

    if mode == "shooting":
        M = 4
        T = 20
        particles = torch.randn(M, T, 2)
        problem = MPCProblem(u0=particles.reshape(-1).numpy(),
                             cost=trajectory_cost,
                             T=T,
                             M=M,
                             udim=2)

        dt = 0.1
        params = {'alphaC': 0.5, 'debug': 0, 'alphaJ': 1, 'dt': dt, 'maxtrials': 1, 'qp_solver': 'osqp',
                  'maxit': args.iters}
        retval = nlspace_solve(problem, params)
        np.savez(f'shooting_retval_{dt}_controller.npz', **retval)

        u = torch.from_numpy(retval['x'][-1].reshape(M, T, 2)).float()

        x_sequence = [torch.tensor([[0.0, -1.0, 0.0, 0.0]]).repeat(M, 1)]

        for t in range(T):
            x = dynamics(x_sequence[-1], u[:, t])
            x_sequence.append(x)

        x = torch.stack(x_sequence, dim=1)
        # plt.plot(lx, ly)
        circle = plt.Circle((0, 0), 1, fill=True, alpha=0.2, color='k', linestyle='--')
        goal = plt.Circle((0.0, 1.0), 0.05, color='r')
        start = plt.Circle((0.0, -1.0), 0.05, color='b')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        for xx in x:
            print(xx.shape)
            plt.plot(xx[:, 0], xx[:, 1])
        plt.gca().add_patch(circle)
        plt.gca().add_patch(goal)
        plt.gca().add_patch(start)
        plt.show()

    if mode == "collocation":
        M = 4
        T = 20
        # what if we initialize so that it obeys the constraint?
        # particles = 0.1 * torch.randn(M, T, 6)
        u = torch.randn(M, T, 2)
        #u_norm = torch.linalg.norm(u, dim=-1, keepdim=True)
        #u = torch.where(u_norm < 1, u, u / u_norm)

        x = [torch.tensor([0.0, -1.0, 0.0, 0.0]).reshape(1, 4).repeat(M, 1)]
        for t in range(T):
            x.append(dynamics(x[-1], u[:, t]))

        x = torch.stack(x, dim=1)[:, 1:]
        particles = torch.cat((x, u), dim=-1)
        # problem = MPCProblemCollacation(x0=particles.reshape(-1).numpy(),
        #                                cost=trajectory_cost_collocation,
        #                                T=T,
        #                                M=M)

        problem = MPCProblemCollocationPytorch(x0=particles.reshape(-1).numpy(),
                                               cost=trajectory_cost_collocation,
                                               T=T,
                                               M=M)
        dt = 0.1
        alphas = np.ones(3 * M * T)
        # alphas = np.concatenate((1 * alphas, 0 * alphas, 0 * alphas), axis=0)
        params = {'alphaC': 0.5, 'debug': 0,
                  'alphaJ': 1,
                  'dt': dt,
                  'maxtrials': 1,
                  'qp_solver': 'osqp',
                  'maxit': args.iters,
                  'damping': 0}

        # retval = nlspace_solve(problem, params)
        # xu = torch.from_numpy(retval['x'][-1].reshape(M, T, 6)).float()
        solver = nlspace_solve_pytorch(problem=problem, params=params)
        xu = solver.solve().reshape(M, T, 6)
        x_sequence = torch.cat((torch.tensor([[0.0, -1.0, 0.0, 0.0]]).repeat(M, 1).unsqueeze(1),
                                xu[:, :, :4]), dim=1)

        #np.savez(f'collocation_retval_{dt}.npz', **retval)
        x = x_sequence.detach().cpu().numpy()
        # plt.plot(lx, ly)
        circle = plt.Circle((0, 0), 1, fill=True, alpha=0.2, color='k', linestyle='--')
        goal = plt.Circle((0.0, 1.0), 0.05, color='r')
        start = plt.Circle((0.0, -1.0), 0.05, color='b')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        for xx in x:
            print(xx.shape)
            plt.plot(xx[:, 0], xx[:, 1])
        plt.gca().add_patch(circle)
        plt.gca().add_patch(goal)
        plt.gca().add_patch(start)
        plt.show()

    if mode == 'collocation_sqp':
        M = 4
        T = 20
        # what if we initialize so that it obeys the constraint?
        # particles = 1 * torch.randn(M, T, 6)
        u = torch.randn(M, T, 2)
        x = [torch.tensor([0.0, -1.0, 0.0, 0.0]).reshape(1, 4).repeat(M, 1)]
        for t in range(T):
            x.append(dynamics(x[-1], u[:, t]))

        x = torch.stack(x, dim=1)[:, 1:]
        particles = torch.cat((x, u), dim=-1)
        problem = MPCProblemCollacation(x0=particles.reshape(-1).numpy(),
                                        cost=trajectory_cost_collocation,
                                        T=T,
                                        M=M)

        solver = SQPSolver(problem, iters=args.iters, use_approx_hessian=True)
        xu, J = solver.solve()
        print(problem.G(xu))
        print(problem.H(xu))
        xu = torch.from_numpy(xu.reshape(M, T, 6)).float()
        x_sequence = torch.cat((torch.tensor([[0.0, -1.0, 0.0, 0.0]]).repeat(M, 1).unsqueeze(1),
                                xu[:, :, :4]), dim=1)

        u = xu[:, :, 4:]
        x = x_sequence.cpu().numpy()
        np.savez(f'sqp_collocation.npz', x=x, u=u, J=J)

        # plt.plot(lx, ly)
        circle = plt.Circle((0, 0), 1, fill=True, alpha=0.2, color='k', linestyle='--')
        goal = plt.Circle((0.0, 1.0), 0.05, color='r')
        start = plt.Circle((0.0, -1.0), 0.05, color='b')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        for i, xx in enumerate(x):
            print(xx.shape)
            print(xx[:, :2])
            print(u[[i]])
            plt.plot(xx[:, 0], xx[:, 1])
        plt.gca().add_patch(circle)
        plt.gca().add_patch(goal)
        plt.gca().add_patch(start)
        plt.show()
    if mode == "shooting_sqp":
        M = 4
        T = 20
        particles = torch.randn(M, T, 2)
        problem = MPCProblem(u0=particles.reshape(-1).numpy(),
                             cost=trajectory_cost,
                             T=T,
                             M=M,
                             udim=2)
        solver = SQPSolver(problem, iters=args.iters)
        u, J = solver.solve()
        u = torch.from_numpy(u.reshape(M, T, 2)).float()

        x_sequence = [torch.tensor([[0.0, -1.0, 0.0, 0.0]]).repeat(M, 1)]

        for t in range(T):
            x = dynamics(x_sequence[-1], u[:, t])
            x_sequence.append(x)

        x = torch.stack(x_sequence, dim=1)
        np.savez(f'../data/mpc_toy_problem/shooting_sqp.npz', u=u, J=J, x=x)

        # plt.plot(lx, ly)
        circle = plt.Circle((0, 0), 1, fill=True, alpha=0.2, color='k', linestyle='--')
        goal = plt.Circle((0.0, 1.0), 0.05, color='r')
        start = plt.Circle((0.0, -1.0), 0.05, color='b')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        for xx in x:
            print(xx.shape)
            print(xx[:, :2])

            plt.plot(xx[:, 0], xx[:, 1])
        plt.gca().add_patch(circle)
        plt.gca().add_patch(goal)
        plt.gca().add_patch(start)
        plt.show()
