import torch
from torch.distributions import MultivariateNormal, Categorical
from ccai.problem import ConstrainedSVGDProblem
from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
from ccai.kernels import rbf_kernel

from functorch import vmap, jacrev
import numpy as np
import matplotlib.pyplot as plt


class MixtureOfGaussians:

    def __init__(self, weights, means, covariances):
        self.mu = means
        self.cov = covariances
        self.w = weights
        self.dists = [MultivariateNormal(mu, cov, validate_args=False) for mu, cov in zip(means, covariances)]
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


class ToyProblem(ConstrainedSVGDProblem):

    def __init__(self, device):
        super().__init__(start=None, goal=None, T=1, device=device)
        self.dx = 2
        self.du = 0
        self.dz = 1
        self.dg = 1
        self.dh = self.dz
        self.squared_slack = True

        self.mog = MixtureOfGaussians(
            means=torch.tensor([[0.3, 0.8],
                                [-0.6, 0.2],
                                [0.4, -0.6]], device=device),
            covariances=torch.tensor([
                [[0.3, -0.0],
                 [-0.0, 0.08]],
                [[0.08, -0.015],
                 [-0.015, 0.06]],
                [[0.24, 0.06],
                 [0.06, 0.16]],
            ], device=device),
            weights=torch.tensor([0.333, 0.333, 0.333], device=device)
        )

        self.grad_log_prob = vmap(jacrev(self.mog.log_prob))
        self.K = rbf_kernel
        self.grad_kernel = jacrev(self.K)
        self.x_max = 2 * torch.ones(2)
        self.x_min = -self.x_max

    def _preprocess(self, x):
        pass

    def _objective(self, x):
        alpha = 2
        return alpha * -self.mog.log_prob(x), alpha * -self.grad_log_prob(x), None

    def _con_eq(self, x, compute_grads=True, compute_hess=True):
        x = x.reshape(-1, self.dx)
        N = x.shape[0]
        g = x[:, 0] ** 2 + x[:, 1] ** 2 - 1.0
        g = g.reshape(N, 1)
        if compute_grads is False:
            return g, None, None

        grad_g = 2 * x.reshape(N, 1, 2)

        if not compute_hess:
            return g, grad_g, None

        hess_g = 2 * torch.eye(2, device=self.device).repeat(N, 1, 1).reshape(N, 1, 2, 2)

        return g, grad_g, hess_g

    def _con_ineq(self, x, compute_grads=True, compute_hess=True):
        a = 1.5
        b = 0
        x = x.reshape(-1, self.dx)
        N = x.shape[0]
        #a1, a2, a3 = -0.14815, .37037, 1.222
        a1, a2, a3 = -0.43386, 0.7984, 2.5094
        a1, a2, a3 = -0.22424, 0.22727, 2.1697
        a1, a2, a3 = -0.17583, 0.22727, 1.7945

        h = a1* x[:, 0] ** 3 + a2 * x[:, 0] ** 2 + a3 * x[:, 0] - x[:, 1] - 0.18155

        h = h.reshape(N, 1)
        if compute_grads is False:
            return h, None, None

        grad_h = torch.zeros(N, 1, 2, device=self.device)
        grad_h[:, 0, 0] = 3 * a1 * x[:, 0] ** 2 + 2 * a2 * x[:, 0] + a1
        grad_h[:, 0, 1] = -1

        if not compute_hess:
            return h, grad_h, None
        hess_h = torch.zeros(N, 1, 2, 2, device=self.device)
        hess_h[:, 0, 0, 0] = 6 * a1 * x[:, 0] + 2 * a2
        return h, grad_h, hess_h

    def eval(self, augmented_trajectory):
        N = augmented_trajectory.shape[0]
        augmented_trajectory = augmented_trajectory.clone().reshape(N, self.T, -1)
        x = augmented_trajectory[:, :, :self.dx + self.du]

        J, grad_J, _ = self._objective(x)
        # hess_J = None
        grad_J = torch.cat((grad_J.reshape(N, self.T, -1),
                            torch.zeros(N, self.T, self.dz, device=x.device)), dim=2).reshape(N, -1)

        Xk = x.reshape(N, -1)
        K = self.K(Xk, Xk, None)  # hess_J.mean(dim=0))
        grad_K = -self.grad_kernel(Xk, Xk, None)  # @hess_J.mean(dim=0))
        grad_K = grad_K.reshape(N, N, N, self.T * (self.dx + self.du))
        grad_K = torch.einsum('nmmi->nmi', grad_K)
        grad_K = torch.cat((grad_K.reshape(N, N, self.T, self.dx + self.du),
                            torch.zeros(N, N, self.T, self.dz, device=x.device)), dim=-1)
        grad_K = grad_K.reshape(N, N, -1)
        G, dG, hessG = self.combined_constraints(augmented_trajectory)

        return grad_J.detach(), None, K.detach(), grad_K.detach(), G.detach(), dG.detach(), hessG.detach()

    def get_initial_xu(self, N):
        return torch.randn(N, self.dx, device=self.device)

    def update(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    device = 'cuda:0'
    toy_problem = ToyProblem(device=device)

    params = {
        'N': 50,
        'step_size': 0.25,
        'momentum': 0.0,
        'iters': 250,
        'alpha_J': 0.1,
        'alpha_C': 1,
        'resample_sigma': 0.05,
        'resample_temperature': 0.25
    }

    optimizer = ConstrainedSteinTrajOpt(toy_problem, params)
    initial_particles = toy_problem.get_initial_xu(params['N'])
    particle_path1 = optimizer.solve(initial_particles.reshape(-1, 1, 2)).reshape(-1, params['N'], 2)
    particle_path2 = optimizer.solve(particle_path1[-1].reshape(-1, 1, 2), resample=True).reshape(-1, params['N'], 2)
    particle_path = torch.cat((particle_path1, particle_path2), dim=0)
    particle_path = particle_path.cpu().numpy()
    # get density for plotting
    N = 1000
    cpu_toy_problem = ToyProblem(device='cpu')

    xx, yy = np.meshgrid(np.linspace(-2, 2, num=N), np.linspace(-2, 2, num=N))
    X = np.stack((xx.flatten(), yy.flatten()), axis=1)
    density = cpu_toy_problem.mog.log_prob(torch.from_numpy(X))
    density = density.reshape(N, N).cpu().numpy()
    # samples = toy_problem.mog.sample(N).detach().cpu().numpy()

    circle = plt.Circle((0, 0), 1, fill=False)
    lx = np.linspace(-2, 2, 100)
    ly = 1.5 * lx
    ly = lx ** 2 - 0.5
    a1, a2, a3 = -0.14815, .37037, 1.222
    #a1, a2, a3 = -0.4444, 0.4444, 1.3333
    a1, a2, a3 = -0.43386, 0.7984, 2.5094
    a1, a2, a3 = -0.17583, 0.22727, 1.7945
    ly = a1 * lx ** 3 + a2 * lx ** 2 + a3 * lx - 0.18155

    plt.fill_between(lx, -2 * np.ones_like(ly), ly, color='k', alpha=0.3)

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.contour(xx, yy, density, levels=50)
    plt.gca().add_patch(circle)

    sc = plt.scatter(particle_path[0, :, 0], particle_path[0, :, 1])

    plt.axis('off')
    import pathlib

    CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

    fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/toy_problem')
    pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
    plt.savefig(f'{fpath}/plot_000.pdf', bbox_inches='tight')
    for i, particle in enumerate(particle_path[1:]):
        print(particle.shape)
        # if i % 100 == 0:
        sc.set_offsets(particle[:, :2])
        plt.savefig(f'{fpath}/plot_{i + 1:03d}.pdf', bbox_inches='tight')
