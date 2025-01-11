import torch
from functorch import vmap, jacrev, hessian
import math
from functools import partial
from torch_cg import cg_batch
import time
import numpy as np

DEVICE = 'cuda:0'
START = torch.tensor([1.0, 0.0, 0.0]).to(device=DEVICE)
GOAL = torch.tensor([-1.0, 0.0, 0.0]).to(device=DEVICE)


def cost(X):
    # x is T x 3 trajectory
    T, d = X.shape
    # return torch.sum((X - GOAL.unsqueeze(0)) ** 2) * 10
    X = torch.cat((START.unsqueeze(0), X, GOAL.unsqueeze(0)), dim=0)
    return 100 * torch.sum((X[1:] - X[:-1]) ** 2)


def surface_constraint(X):
    # x is T x 3 trajectory
    x, y, z = torch.chunk(X, dim=-1, chunks=3)
    return (x ** 2 + y ** 2 + z ** 2 - 1).reshape(-1)


def obstacle_constraint(X):
    x, y, z = torch.chunk(X, dim=-1, chunks=3)
    obs1 = 0.1 - x ** 2 - y ** 2
    return obs1.reshape(-1)
    # obs2 = 0.25 - y ** 2 - z ** 2
    return torch.cat((obs1, obs2)).reshape(-1)


def start_and_goal_constraint(X):
    start_constraint = X[0] - START
    goal_constraint = X[-1] - GOAL
    return torch.cat((start_constraint, goal_constraint)).reshape(-1)


def equality_constraints(X):
    return surface_constraint(X)
    return torch.cat(((surface_constraint(X)), start_and_goal_constraint(X)))


def median(tensor):
    """
    torch.median() acts differently from np.median(). We want to simulate numpy implementation.
    """
    tensor = tensor.detach().flatten()
    tensor_max = tensor.max()[None]
    return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.


def rbf_kernel(X, Xbar, M):
    # X is N x d
    n, d = X.shape
    diff = X.unsqueeze(0) - Xbar.unsqueeze(1)
    scaled_diff = diff @ M.unsqueeze(0)
    scaled_diff = (scaled_diff.reshape(-1, 1, d) @ diff.reshape(-1, d, 1)).reshape(n, n)
    return torch.exp(- 0.5 * scaled_diff / d)


# create functorch versions of relevant functions
class Problem:

    def __init__(self):
        self.J = vmap(cost)
        self.dJ = vmap(jacrev(cost))
        self.HJ = vmap(hessian(cost))

        self.g = vmap(equality_constraints)
        self.dg = vmap(jacrev(equality_constraints))
        self.Hg = vmap(hessian(equality_constraints))
        self.h = vmap(obstacle_constraint)
        self.dh = vmap(jacrev(obstacle_constraint))
        self.Hh = vmap(hessian(obstacle_constraint))
        self.K = rbf_kernel
        self.dK = jacrev(rbf_kernel)

    def eval(self, X):
        X = X.detach()
        X.requires_grad = True
        n, T, d = X.shape

        # objective, grad objective, hessian objective
        J = self.J(X)
        score = -self.dJ(X).reshape(n, -1)
        HJ = self.HJ(X).reshape(n, d * T, d * T)  # hessian of negative log likelihood

        Xk = X.reshape(n, -1)
        K = rbf_kernel(Xk, Xk.detach(), torch.mean(HJ, dim=0))
        dK = torch.autograd.grad(K.sum(), Xk)[0]
        df = -(K @ score + dK)
        # kernel_gauss_newton = dK.unsqueeze(2) @ dK.unsqueeze(1)
        # We use gauss newton approx of cost for hessian -> this has links to natural gradient & fisher information
        Hf = (K @ K @ HJ.reshape(n, -1)).reshape(n, d * T, d * T) + dK.unsqueeze(2) @ dK.unsqueeze(1)
        # Hf = fisher_info_mat.reshape(1, d*T, d*T).repeat(n, 1, 1)

        df = df / n
        Hf = Hf / n

        # now we do constraints
        g = self.g(X)
        dg = self.dg(X).reshape(n, -1, d * T)
        Hg = self.Hg(X).reshape(n, -1, d * T, d * T)

        h = self.h(X)
        dh = self.dh(X).reshape(n, -1, d * T)
        Hh = self.Hh(X).reshape(n, -1, d * T, d * T)

        # gradients and hessians for all constraints and cost
        return df.detach(), Hf.detach(), g.detach(), dg.detach(), \
               Hg.detach(), h.detach(), dh.detach(), Hh.detach(), K.detach(), HJ.mean(dim=0)

    def merit(self, X, lam, mu, M, ck):
        X = X.detach()
        n, T, d = X.shape
        HJ = self.HJ(X).reshape(n, d * T, d * T)  # hessian of negative log likelihood
        M = HJ.mean(dim=0)
        log_prob = -self.J(X).reshape(n)

        # first let's compute our overall objective gradient and approx hessian
        Xk = X.reshape(n, -1)
        K = rbf_kernel(Xk, Xk.detach(), M)

        g = self.g(X).reshape(n, -1, 1)
        h = self.h(X).reshape(n, -1, 1)
        merit = -log_prob.mean() + n * K.mean() + \
                torch.sum(lam * g, dim=1).mean() + ck * torch.sum(g ** 2, dim=1).mean() / 2

            # print(log_prob.mean(), torch.sum(lam*g, dim=1).mean(), ck * torch.sum(g**2, dim=1).mean() / 2)
        return merit

    def dLagrangian(self, X, lam, mu, M, ck=None):
        X = X.detach()
        X.requires_grad = True
        n, T, d = X.shape
        HJ = self.HJ(X).reshape(n, d * T, d * T)  # hessian of negative log likelihood
        M = HJ.mean(dim=0)
        score = -self.dJ(X).reshape(n, -1)

        # first let's compute our overall objective gradient and approx hessian
        Xk = X.reshape(n, -1)
        K = rbf_kernel(Xk, Xk.detach(), M)
        dK = torch.autograd.grad(K.sum(), Xk)[0]
        df = -(K @ score + dK) / n

        dg = self.dg(X).reshape(n, -1, T * d)
        dh = self.dh(X).reshape(n, -1, T * d)

        dL = df + torch.sum(lam * dg, dim=1)

        if ck is not None:
            g = self.g(X).reshape(n, 1, -1)
            dL = dL + ck * (g @ dg).reshape(n, T * d)
        return dL


class ConstrainedNewton:
    """ Primal-dual algorithm using Projected newton (for ineq constraints)"""

    def __init__(self, iters=500, atol=1e-2):
        self.iters = iters
        self.atol = atol

        self.prob = Problem()

    def solve(self, x0):

        n, T, d = x0.shape
        x = x0.clone()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        df, Hf, g, dg, Hg, h, dh, Hh, K, M = self.prob.eval(x)

        # number of constraints
        n_eq = g.shape[1]
        n_ineq = h.shape[1]

        # now we are going to initialise the dual variables
        lam = -0.1 * torch.ones_like(g).reshape(n, -1, 1)
        mu = torch.where(h < 0, torch.ones_like(h), torch.zeros_like(h)).reshape(n, -1, 1)
        mode = 'full'
        mode = 'eq'
        # mode = 'ineq'
        # mode = 'none'
        eps_k = 10
        w_k = torch.linalg.norm(self.prob.dLagrangian(x, lam, mu, M)) ** 2
        c_k = 0
        r = 1.1
        gamma = 0.99
        old_merit = self.prob.merit(x, mu, lam, M, c_k).item()

        for iter in range(self.iters):

            if mode == 'eq':
                mu *= 0
            elif mode == 'ineq':
                lam *= 0
            elif mode == 'none':
                mu *= 0
                lam *= 0

            # mu = torch.where(h < 0, 10000 * torch.ones_like(h), torch.zeros_like(h)).reshape(n, -1, 1)
            # compute hessian of lagrangian
            K_inv = torch.linalg.inv(K)
            HL = Hf + torch.sum(lam.reshape(n, -1, 1, 1) * Hg, dim=1) + torch.sum(mu.reshape(n, -1, 1, 1) * Hh,
                                                                                  dim=1)
            HL = (K_inv @ HL.reshape(n, -1)).reshape(n, T*d, T*d)

            # HL = HL + torch.eye(T*d).unsqueeze(0).to(device=DEVICE) * 0.01
            # gradient of the lagrangian
            dL = df + torch.sum(lam * dg, dim=1) + torch.sum(mu * dh, dim=1)
            #dL = self.prob.dLagrangian(x, lam, mu, M, c_k)
            #new_merit = self.prob.merit(x, mu, lam, M, c_k).item()

            #x = x - 1e-3 * dL.reshape(n, T, d)
            #df, Hf, g, dg, Hg, h, dh, Hh, K, M = self.prob.eval(x)
            #old_merit = new_merit
            #continue

            # print(torch.linalg.norm(df, dim=1))
            # print(torch.linalg.norm(torch.sum(mu * dh, dim=1), dim=1))
            # print(torch.linalg.norm(dL))
            # terminate when lagrangian converged
            if torch.all(torch.linalg.norm(dL, dim=1) < self.atol):
                print(f'Converged after {iter} iterations')
                break


            if mode == 'full':
                A1 = torch.cat((HL, dg.transpose(1, 2), dh.transpose(1, 2)), dim=2)
                A2 = torch.cat((dg, -c_k * torch.ones(n, n_eq, n_eq, device=DEVICE),
                                torch.zeros(n, n_eq, n_ineq, device=DEVICE)), dim=2)
                A3 = torch.cat((dh, torch.zeros(n, n_ineq, n_eq, device=DEVICE),
                                -c_k * torch.ones(n, n_ineq, n_ineq, device=DEVICE)), dim=2)
                A = torch.cat((A1, A2, A3), dim=1)

                # then the b matrix
                b = -torch.cat((dL, g, h), dim=1).unsqueeze(-1)
            elif mode == 'eq':
                A1 = torch.cat((HL, dg.transpose(1, 2)), dim=2)
                A2 = torch.cat((dg, -c_k * torch.ones(n, n_eq, n_eq, device=DEVICE)), dim=2)
                A = torch.cat((A1, A2), dim=1)
                b = -torch.cat((dL, g), dim=1).unsqueeze(-1)
            elif mode == 'ineq':
                A1 = torch.cat((HL, dh.transpose(1, 2)), dim=2)
                A2 = torch.cat((dh, -c_k * torch.ones(n, n_ineq, n_ineq, device=DEVICE)), dim=2)
                A = torch.cat((A1, A2), dim=1)
                b = -torch.cat((dL, h), dim=1).unsqueeze(-1)
            elif mode == 'none':
                A = Hf
                b = df.unsqueeze(-1)

            # First try a lagrangian step
            A_mm = lambda x: A @ x

            descent_direction, _ = cg_batch(A_mm, b, verbose=False)
            descent_direction = descent_direction.reshape(n, -1)

            dx = descent_direction[:, :T*d].reshape(n, T, d)
            dlam = descent_direction[:, T*d:].reshape(n, -1, 1)

            max_norm = 1
            dx = torch.where(torch.linalg.norm(dx, dim=1, keepdim=True) > max_norm,
                             max_norm * dx / torch.linalg.norm(dx, dim=1, keepdim=True), dx)
            new_x = x + dx.reshape(n, T, d)
            if torch.all(torch.linalg.norm(dx, dim=1) < self.atol):
                print(f'Converged after {iter} iterations')
                break

            # g = self.prob.g(new_x).unsqueeze(-1)
            # dg = self.prob.dg(new_x).reshape(n, -1, T*d)
            # Hinv = torch.linalg.inv(HL + c_k * dg.transpose(1, 2) @ dg)
            # tmp = torch.linalg.inv(dg @ Hinv @ dg.transpose(2, 2))
            # new_lam = tmp @ (g - dg @ Hinv @ df.unsqueeze(-1)) - c_k * g

            #dlam = self.prob.g(new_x).unsqueeze(-1)
            dlam = torch.where(torch.linalg.norm(dlam, dim=1, keepdim=True) > max_norm,
                             max_norm * dlam / torch.linalg.norm(dlam, dim=1, keepdim=True), dlam)

            new_lam = lam + dlam#elf.prob.g(new_x).unsqueeze(-1)
            new_mu = mu# + self.prob.h(new_x).unsqueeze(-1)

            # print(lam)
            # check - should we accept this?
            new_dL = self.prob.dLagrangian(new_x, new_lam, 0 * new_mu, M)
            #print('norm lagrandian dL', torch.linalg.norm(new_dL) ** 2, w_k, c_k)

            x = new_x
            lam = new_lam
            df, Hf, g, dg, Hg, h, dh, Hh, K, M = self.prob.eval(x)
            continue

            if torch.linalg.norm(new_dL) ** 2 < w_k:
                print('lagrange step')
                x = new_x
                lam = new_lam
                mu = new_mu
                w_k = gamma * torch.linalg.norm(new_dL) ** 2
            else:
                # do not accept, so we do method of multipliers approach
                old_merit = self.prob.merit(x, lam, mu, M, c_k).item()
                target = 0 * 0.1 * (
                        dx.reshape(n, 1, T * d) @ self.prob.dLagrangian(x, lam, mu, M, c_k).reshape(n, T * d,
                                                                                                    1)).mean()
                print('target', target)
                # print(target.shape)
                dx = -self.prob.dLagrangian(x, lam, mu, M, c_k).reshape(n, T, d)
                beta = 0.5
                while True:
                    new_merit = self.prob.merit(x + beta * dx, lam, mu, M, c_k).item()
                    # print(old_merit - new_merit, target.item() * beta)
                    if old_merit - new_merit >= beta * target:
                        x = x + beta * dx
                        new_dL = self.prob.dLagrangian(x, lam, mu, M, c_k)
                        print('norm dL', torch.linalg.norm(new_dL), eps_k)
                        if torch.linalg.norm(new_dL) < eps_k:
                            print('updating dual')
                            lam = lam + c_k * self.prob.g(x).unsqueeze(-1)
                            # g = self.prob.g(x).unsqueeze(-1)
                            # Hinv = torch.linalg.inv(HL + c_k * dg.transpose(1, 2) @ dg)
                            # tmp = torch.linalg.inv(dg @ Hinv @ dg.transpose(1, 2))
                            # lam = tmp @ (g - dg @ Hinv @ df.unsqueeze(-1)) - c_k * g

                            mu = mu + c_k * self.prob.h(x).unsqueeze(-1)
                            new_dL = self.prob.dLagrangian(x, lam, 0 * mu, M)

                            eps_k = gamma * eps_k
                            c_k = r * c_k
                            w_k = gamma * torch.linalg.norm(new_dL)

                        break
                    beta *= 0.5

            df, Hf, g, dg, Hg, h, dh, Hh, K, M = self.prob.eval(x)

        end.record()
        torch.cuda.synchronize('cuda:0')

        print(f'Average time per iteration: {start.elapsed_time(end) / (iter+1)} ms')
        return x


if __name__ == "__main__":
    solver = ConstrainedNewton()
    N = 8
    T = 20

    alpha = torch.linspace(0, 1, steps=T).to(device=DEVICE)
    x0 = alpha.unsqueeze(1) @ START.unsqueeze(0) + (1 - alpha.unsqueeze(1) @ GOAL.unsqueeze(0))
    x0 = x0.unsqueeze(0) + 0.1 * torch.randn(N, T, 3).to(device=DEVICE)
    x = solver.solve(x0)

    x = x.detach().cpu().numpy()
    #print(x.shape)
    x = np.concatenate((START.reshape(1, 1, 3).repeat(N, 1, 1).cpu().numpy(),
                        x, GOAL.reshape(1, 1, 3).repeat(N, 1, 1).cpu().numpy()), axis=1)
    #print(np.mean(x, axis=0))
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    r = 0.99
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    xx = np.cos(u) * np.sin(v)
    yy = np.sin(u) * np.sin(v)
    zz = np.cos(v)
    ax.plot_surface(xx, yy, zz, alpha=0.1)

    for n in range(N):
        ax.plot3D(x[n, :, 0], x[n, :, 1], x[n, :, 2])
    ax.scatter(x[0, 0, 0], x[0, 0, 1], x[0, 0, 2], s=250)
    ax.scatter(x[0, -1, 0], x[0, -1, 1], x[0, -1, 2], s=250)

    plt.show()
