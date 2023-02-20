import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal, Categorical


def rbf_kernel(x, xbar, lengthscale=1.):
    diff = (x.unsqueeze(0) - xbar.unsqueeze(1))
    diff = ((diff / lengthscale) ** 2).sum(dim=-1)
    return torch.exp(-0.5 * diff)


def linear_kernel(x, xbar):
    return x @ xbar.transpose(0, 1) + 1


class MixtureOfGaussians:

    def __init__(self, weights, means, covariances):
        self.mu = means
        self.cov = covariances
        self.w = weights
        self.dists = [MultivariateNormal(mu, cov) for mu, cov in zip(means, covariances)]
        self.mixture_dist = Categorical(weights)
        self.m, self.dx = means.shape

    def log_prob(self, x):
        prob = 0
        for w, dist in zip(self.w, self.dists):
            prob += w * dist.log_prob(x).exp()
        return torch.log(prob)

    def sample(self, N):
        samples = torch.zeros(N, self.dx)
        components = self.mixture_dist.sample(sample_shape=(N,))
        for i, dist in enumerate(self.dists):
            csample = dist.sample(sample_shape=(N,))
            samples = torch.where(components.unsqueeze(-1) == i, csample, samples)

        return samples


if __name__ == "__main__":

    mog = MixtureOfGaussians(
        means=torch.tensor([[0.3, 0.8],
                            [-0.6, 0.2],
                            [0.4, -0.6]]),
        covariances=torch.tensor([
            [[0.3, -0.0],
             [-0.0, 0.08]],
            [[0.08, -0.015],
             [-0.015, 0.06]],
            [[0.24, 0.06],
             [0.06, 0.16]],
        ]),
        weights=torch.tensor([0.333, 0.333, 0.333])
    )

    # mog = MixtureOfGaussians(means=torch.zeros(1, 2),
    #                         covariances=torch.eye(2).unsqueeze(0),
    #                         weights=torch.ones(1))
    N = 1000
    xx, yy = np.meshgrid(np.linspace(-2, 2, num=N), np.linspace(-2, 2, num=N))
    X = np.stack((xx.flatten(), yy.flatten()), axis=1)
    density = mog.log_prob(torch.from_numpy(X))
    density = density.reshape(N, N).numpy()
    samples = mog.sample(N).detach().cpu().numpy()

    # Run SVGD
    # M is number of particles
    M = 500
    iters = 10000
    particles = torch.randn(M, 2)
    particles.requires_grad = True
    epsilon = 1e-3
    # optimiser = optim.Adam([{'params': particles}], lr=1e-3)
    # plt.scatter(particles[:, 0].detach(), particles[:, 1].detach())
    plt.contour(xx, yy, density, levels=40)
    # plt.show()

    # Run SVGD
    for i in range(iters):
        lengthscale = torch.median(particles) ** 2 / np.log(M)
        # lengthscale = 0.2
        # K = rbf_kernel(particles, particles.detach(), lengthscale=lengthscale)
        K = linear_kernel(particles, particles.detach())
        log_prob = mog.log_prob(particles)

        score = torch.autograd.grad(log_prob.sum(), particles)[0]
        grad_K = torch.autograd.grad(K.sum(), particles)[0]
        grad = (K @ score + grad_K) / M

        particles = particles + epsilon * grad.detach()

    print(torch.mean(particles, dim=0))
    print(torch.var(particles, dim=0))

    plt.scatter(particles[:, 0].detach(), particles[:, 1].detach())
    plt.contour(xx, yy, density, levels=40)
    plt.show()


    ### INTRODUCE CONSTRAINTS
    # place a constraint that g(x) <= 0
    # start with linear constraint with line y=ax + b
    def linear_constraint(a, b, x):
        return a * x[:, 0] + b - x[:, 1]


    a = 1.5
    b = 0

    """
    # one option - project to constraint at every iteration
    # other option - project gradient to operate in nullspace of constraint
    # todo this we need to first find which constraints are active (only one constraint here)

    # M is number of particles
    M = 10
    iters = 1000
    particles = 0.5 * torch.randn(M, 2)
    particles.requires_grad = True
    epsilon = 1e-2
    # optimiser = optim.Adam([{'params': particles}], lr=1e-3)
    plt.scatter(particles[:, 0].detach(), particles[:, 1].detach())
    plt.contour(xx, yy, density, levels=40)
    plt.show()

    # Run SVGD
    for i in range(iters):
        print(i)
        lengthscale = torch.median(particles) ** 2 / M
        # lengthscale = 0.2
        K = rbf_kernel(particles, particles.detach(), lengthscale=lengthscale)
        log_prob = mog.log_prob(particles)

        score = torch.autograd.grad(log_prob.sum(), particles)[0]
        grad_K = torch.autograd.grad(K.sum(), particles)[0]
        grad = (K @ score + grad_K) / M

        # get constraint fcn and grad of constrant
        constraint = linear_constraint(a, b, particles)
        grad_constraint = torch.autograd.grad(constraint.sum(), particles)[0]

        # now we need to solve a non-negative least squares - use projected grad descent
        mu = torch.randn(M)
        mu.requires_grad = True
        for j in range(100):
            loss = torch.linalg.norm(grad + grad_constraint * mu.unsqueeze(1), dim=1)
            grad_mu = torch.autograd.grad(loss.sum(), mu)[0]
            mu = mu - 1e-2 * grad_mu.detach()
            mu = torch.where(mu < 0, 0, mu)

        threshold = 1e-5
        grad_constraint = torch.where(mu.unsqueeze(1) > threshold, grad_constraint, 0)
        constraint = torch.where(mu > threshold, constraint, 0)

        print(constraint)
        if constraint.sum().abs() > 0:
            print('constraints not satisfied')
            # project gradient using grad_constraint
            grad_constraint.permute(1, 0) @ grad_constraint
            dc_inv = torch.linalg.inv(grad_constraint.permute(1, 0) @ grad_constraint)

            P = grad_constraint @ dc_inv
            eta_J = (torch.eye(M) - P @ grad_constraint.permute(1, 0)) @ -grad

            eta_C = P * constraint.unsqueeze(1)
            print(eta_C)
            particles = particles - epsilon * eta_J.detach() - epsilon * 10 * eta_C.detach()
        else:
            particles = particles + epsilon * grad_K.detach()
    """

    # use nullspace optimizer
    from nullspace_optimizer import *

    def circ_constraint(x):
        return x[:, 0] ** 2 + x[:, 1] ** 2 - 1.0


    class ConstrainedProblem(EuclideanOptimizable):
        def __init__(self, x0, M):
            super().__init__(M * 2)
            self.nconstraints = M
            self.nineqconstraints = M
            self._x0 = x0
            self.M = M
            self.dx = 2

        def x0(self):
            return self._x0.reshape(-1)

        def J(self, x):
            # tx = torch.from_numpy(x).reshape(self.M, -1).float()
            # lengthscale = torch.median(tx) ** 2 / self.M
            # log_prob = mog.log_prob(tx)
            # K = rbf_kernel(tx, tx, lengthscale)
            # J = K @ log_prob + K
            # print(K.shape, (K @ log_prob).shape, J.shape)
            return 0  # -J.sum().detach().numpy()

        def dJ(self, x):
            tx = torch.from_numpy(x).reshape(self.M, -1).float()
            lengthscale = torch.median(tx) / self.M
            tx.requires_grad = True
            # lengthscale = 0.1
            log_prob = mog.log_prob(tx)
            K = rbf_kernel(tx, tx.detach(), lengthscale.detach())

            score = torch.autograd.grad(log_prob.sum(), tx)[0]
            grad_K = torch.autograd.grad(K.sum(), tx)[0]

            dJ = K @ score + grad_K

            return -dJ.reshape(-1).detach().numpy() / self.M

        def H(self, x):
            tx = torch.from_numpy(x).reshape(self.M, -1).float()
            return linear_constraint(a, b, tx).reshape(self.M).detach().numpy()

        def dH(self, x):
            tx = torch.from_numpy(x).reshape(self.M, -1).float()
            tx.requires_grad = True
            H = linear_constraint(a, b, tx).reshape(self.M)
            dH = torch.autograd.grad(H.sum(), tx)[0]

            # dH is M x 2 - need to convert to M x (Mx2) - will mostly be zero vectors
            dH = torch.diag_embed(dH.permute(1, 0)).permute(2, 1, 0).reshape(self.M, self.M * 2)
            return dH.detach().numpy()

        def G(self, x):
            tx = torch.from_numpy(x).reshape(self.M, -1).float()
            return circ_constraint(tx).reshape(self.M).detach().numpy()

        def dG(self, x):
            tx = torch.from_numpy(x).reshape(self.M, -1).float()
            tx.requires_grad = True
            H = circ_constraint(tx).reshape(self.M)
            dH = torch.autograd.grad(H.sum(), tx)[0]

            # dH is M x 2 - need to convert to M x (Mx2) - will mostly be zero vectors
            dH = torch.diag_embed(dH.permute(1, 0)).permute(2, 1, 0).reshape(self.M, self.M * 2)
            return dH.detach().numpy()


    M = 50
    particles = 0.5 * torch.randn(M, 2)
    problem = ConstrainedProblem(x0=particles.reshape(-1).numpy(), M=M)
    x = np.random.randn(M * 2)
    problem.J(x)
    problem.dJ(x)
    problem.H(x)
    problem.dH(x)

    params = {'alphaC': 1, 'debug': 0, 'alphaJ': 1, 'dt': 0.1, 'maxtrials': 1, 'qp_solver': 'osqp', 'maxit': 100}
    import time
    s = time.time()
    retval = nlspace_solve(problem, params)
    e = time.time()
    print(e - s)
    particles = retval['x'][-1].reshape(M, 2)
    circle = plt.Circle((0, 0), 1, fill=False)
    lx = np.linspace(-1, 2, 100)
    ly = a * lx + b
    plt.plot(lx, ly)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.scatter(particles[:, 0], particles[:, 1])
    plt.contour(xx, yy, density, levels=40)
    plt.gca().add_patch(circle)
    plt.show()


    # Let's try another where we have an equality constraint that the points lie on a circle


    class EqConstrainedProblem(EuclideanOptimizable):
        def __init__(self, x0, M):
            super().__init__(M * 2)
            self.nconstraints = M
            self.nineqconstraints = 0
            self._x0 = x0
            self.M = M
            self.dx = 2

        def x0(self):
            return self._x0.reshape(-1)

        def J(self, x):
            # tx = torch.from_numpy(x).reshape(self.M, -1).float()
            # lengthscale = torch.median(tx) ** 2 / self.M
            # log_prob = mog.log_prob(tx)
            # K = rbf_kernel(tx, tx, lengthscale)
            # J = K @ log_prob + K
            # print(K.shape, (K @ log_prob).shape, J.shape)
            return 0  # -J.sum().detach().numpy()

        def dJ(self, x):
            tx = torch.from_numpy(x).reshape(self.M, -1).float()
            lengthscale = torch.median(tx) / self.M
            tx.requires_grad = True
            # lengthscale = 0.1
            log_prob = mog.log_prob(tx)
            K = rbf_kernel(tx, tx.detach(), lengthscale.detach())

            score = torch.autograd.grad(log_prob.sum(), tx)[0]
            grad_K = torch.autograd.grad(K.sum(), tx)[0]

            dJ = K @ score + grad_K

            return -dJ.reshape(-1).detach().numpy() / self.M

        def G(self, x):
            tx = torch.from_numpy(x).reshape(self.M, -1).float()
            return circ_constraint(tx).reshape(self.M).detach().numpy()

        def dG(self, x):
            tx = torch.from_numpy(x).reshape(self.M, -1).float()
            tx.requires_grad = True
            H = circ_constraint(tx).reshape(self.M)
            dH = torch.autograd.grad(H.sum(), tx)[0]

            # dH is M x 2 - need to convert to M x (Mx2) - will mostly be zero vectors
            dH = torch.diag_embed(dH.permute(1, 0)).permute(2, 1, 0).reshape(self.M, self.M * 2)
            return dH.detach().numpy()


    M = 50
    particles = 0.5 * torch.randn(M, 2)
    problem = EqConstrainedProblem(x0=particles.reshape(-1).numpy(), M=M)
    x = np.random.randn(M * 2)

    params = {'alphaC': 1, 'debug': 0, 'alphaJ': 1, 'dt': 0.05, 'maxtrials': 1, 'qp_solver': 'osqp', 'maxit': 1000}
    retval = nlspace_solve(problem, params)

    particles = retval['x'][-1].reshape(M, 2)
    lx = np.linspace(-1, 2, 100)
    ly = a * lx + b
    # plt.plot(lx, ly)
    circle = plt.Circle((0, 0), 1, fill=False)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.scatter(particles[:, 0], particles[:, 1])
    plt.contour(xx, yy, density, levels=40)
    plt.gca().add_patch(circle)
    plt.show()
