import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from functorch import vmap, jacrev, hessian

from ccai.kernels import rbf_kernel
from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inequality', action='store_true')
    return parser.parse_args()


def cost(X, goal):
    T, _ = X.shape
    u = X[:, 6:]
    x = X[:, :6]
    Q = torch.eye(6, device=X.device)
    Q[3:, 3:] *= 0.01

    state_cost = (x - goal).reshape(T, 1, 6) @ Q.unsqueeze(0) @ (x - goal).reshape(T, 6, 1)

    terminal_cost = torch.sum((x[-1] - goal) ** 2)

    control_cost = 0.001 * torch.sum(u ** 2)
    return 0 * torch.sum(state_cost) + 0 * terminal_cost + control_cost


def goal_constraint(X, goal):
    x = X[:, :6]
    return torch.sum((x[-1] - goal) ** 2).reshape(-1)


def surface_constraint(X):
    # x is T x 3 trajectory
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return (x ** 2 + y ** 2 + z ** 2 - 1).reshape(-1)


def halfsphere_constraint(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return y - 0.05
    # return 0.1 - x ** 2 - z ** 2


def equality_constraints(X, start, goal):
    return torch.cat((surface_constraint(X), dynamics(X, start, goal)), dim=0)


def dynamics(X, start, goal):
    T = X.shape[0]
    x = X[:, :6]
    u = X[:, 6:]
    x = torch.cat((start.reshape(1, 6), x[:-1], goal.reshape(1, 6)), dim=0)
    dt = 0.05
    A = torch.tensor([
        [1.0, 0.0, 0.0, dt, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, dt, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, dt],
        [0.0, 0.0, 0.0, 0.9, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.9, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.9]
    ], device=X.device
    ).reshape(1, 6, 6)

    B = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [dt, 0.0, 0.0],
        [0.0, dt, 0.0],
        [0.0, 0.0, dt]
    ], device=X.device
    ).reshape(1, 6, 3)

    diff = x[1:].reshape(T, 6, 1) - A @ x[:-1].reshape(T, 6, 1) + B @ u.reshape(T, 3, 1)

    return torch.sum(diff ** 2, dim=1).reshape(-1)


def rollout(x0, u):
    dt = 0.05
    K = 10
    A = torch.tensor([
        [1.0, 0.0, 0.0, dt, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, dt, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, dt],
        [0.0, 0.0, 0.0, 0.9, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.9, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.9]
    ], device=u.device
    ).reshape(1, 6, 6)

    B = K * torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [dt, 0.0, 0.0],
        [0.0, dt, 0.0],
        [0.0, 0.0, dt]
    ], device=u.device
    ).reshape(1, 6, 3)

    states = [x0.unsqueeze(-1)]
    for t in range(u.shape[1]):
        states.append(A @ states[-1] + B @ u[:, t].unsqueeze(-1))

    return torch.stack(states[1:], dim=1).squeeze(-1)


class SphereDoubleIntegratorProblem:

    def __init__(self, start, goal, T, use_inequality=True):
        self.T = T
        self.dx = 6
        self.du = 3
        if use_inequality:
            self.dz = 1
        else:
            self.dz = 0
        self.dh = self.dz * T
        self.dg = 2 * T
        self.J = vmap(partial(cost, goal=goal))
        self.dJ = vmap(jacrev(partial(cost, goal=goal)))
        self.HJ = vmap(hessian(partial(cost, goal=goal)))

        self.start = start
        self.goal = goal
        self.g = vmap(self._combined_contraints)
        self.grad_g = vmap(jacrev(self._combined_contraints))
        self.hess_g = vmap(hessian(self._combined_contraints))

        self.K = rbf_kernel
        self.dK = jacrev(rbf_kernel, argnums=0)
        self.x_max = torch.ones(self.dx + self.du)

        self.use_inequality = use_inequality
        self.x_min = -self.x_max

    def _combined_contraints(self, augmented_trajectory):
        X = augmented_trajectory[:, :self.dx + self.du]
        z = augmented_trajectory[:, self.dx + self.du:]
        g = equality_constraints(X, self.start, self.goal)
        if not self.use_inequality:
            return g
        h = halfsphere_constraint(X).reshape(self.T, self.dz)
        h_g = h + 0.5 * z ** 2
        return torch.cat((g, h_g.reshape(-1)), dim=0)

    def eval(self, augmented_trajectory):
        augmented_trajectory = augmented_trajectory.detach()
        augmented_trajectory.requires_grad = True
        N = augmented_trajectory.shape[0]
        augmented_trajectory = augmented_trajectory.reshape(N, self.T, -1)
        d = augmented_trajectory.shape[2]
        X = augmented_trajectory[:, :, :self.dx + self.du]

        # objective, grad objective, hessian objective
        grad_J = self.dJ(X)
        grad_J = torch.cat((grad_J, torch.zeros(N, self.T, self.dz, device=X.device)), dim=2).reshape(N, -1)

        Xk = X.reshape(N, -1)
        K = self.K(Xk, Xk)
        grad_K = -self.dK(Xk, Xk)
        grad_K = torch.einsum('nmmi->nmi', grad_K)
        grad_K = torch.cat((grad_K.reshape(N, N, self.T, self.dx + self.du),
                            torch.zeros(N, N, self.T, self.dz, device=X.device)), dim=-1)
        grad_K = grad_K.reshape(N, N, -1)

        # now we do constraints
        g = self.g(augmented_trajectory)
        grad_g = self.grad_g(augmented_trajectory).reshape(N, -1, d * self.T)
        hess_g = self.hess_g(augmented_trajectory).reshape(N, -1, d * self.T, d * self.T)

        return grad_J.detach(), K.detach(), grad_K.detach(), g.detach(), grad_g.detach(), hess_g.detach()

    def get_initial_z(self, X):
        N = X.shape[0]
        fh = vmap(halfsphere_constraint)
        h = fh(X)
        z = torch.where(h < 0, torch.sqrt(- 2 * h), 0)
        return z.reshape(N, self.T, -1)


if __name__ == "__main__":
    N = 16
    T = 20
    device = 'cuda:0'
    goal = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device=device)
    start = torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device=device)

    args = get_args()
    problem = SphereDoubleIntegratorProblem(start, goal, T, use_inequality=args.inequality)

    # Initialise random controls and roll them out
    u = torch.randn(N, T, 3, device=device)
    x = rollout(start.repeat(N, 1), u)
    x0 = torch.cat((x, u), dim=2)

    if args.inequality:
        alpha_C = 1
        alpha_J = 0.5
    else:
        alpha_C = 0.5
        alpha_J = 1

    solver = ConstrainedSteinTrajOpt(problem, dt=0.025, alpha_C=alpha_C, alpha_J=alpha_J)
    solver.iters = 1000

    x = solver.solve(x0)
    x = x.detach().cpu().numpy()
    s = torch.cat((start, torch.zeros(3, device=start.device)))
    g = torch.cat((goal, torch.zeros(3, device=start.device)))
    x = np.concatenate((s.reshape(1, 1, 9).repeat(N, 1, 1).cpu().numpy(), x), axis=1)
    x[:, -1] = g.reshape(1, 9).repeat(N, 1).cpu().numpy()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    # plot 3d sphere
    r = 0.99
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    xx = np.cos(u) * np.sin(v)
    yy = np.sin(u) * np.sin(v)
    zz = np.cos(v)
    ax.plot_surface(xx, yy, zz, alpha=0.1)

    # plot trajectories
    start = start.cpu().numpy()
    goal = goal.cpu().numpy()
    for n in range(N):
        ax.plot3D(x[n, :, 0], x[n, :, 1], x[n, :, 2])
    ax.scatter(goal[0], goal[1], goal[2], s=250)
    ax.scatter(start[0], start[1], start[2], s=250)

    plt.show()
