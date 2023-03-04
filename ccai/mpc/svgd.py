import torch
from ccai.kernels import rbf_kernel, structured_rbf_kernel

from functorch import vmap, jacrev
import torch.distributions as dist
from torch import optim


def get_gmm(means, weights, scale):
    mix = dist.Categorical(weights)
    comp = dist.Independent(dist.Normal(loc=means.detach(), scale=scale), 1)
    return dist.mixture_same_family.MixtureSameFamily(mix, comp)


def d_log_qU(U_samples, U_mean, U_sigma):
    return (U_samples - U_mean) / U_sigma ** 2


class SVMPC:

    def __init__(self, problem, params):
        """

            SVMPC
            optionally uses flow to represent posterior q(U) --
            samples maintained in U space. q(U) is mixture of Gaussians, in Flow space

        """
        self.problem = problem
        self.dx = problem.dx
        self.du = problem.du
        self.H = self.problem.T
        self.fixed_H = params.get('receding_horizon', True)
        self.N = params.get('N', 64)
        self.M = params.get('M', 4)
        self.lr = params.get('step_size', 0.01)
        self.device = params.get('device', 'cuda:0')
        self.sigma = params.get('sigma', torch.ones(self.du))
        self.lambda_ = params.get('lambda', 1)
        self.warmup_iters = params.get('warmup_iters', 100)
        self.online_iters = params.get('online_iters', 100)
        self.use_true_grad = params.get('use_grad', False)

        self.kernel = structured_rbf_kernel
        self.kernel_grad = jacrev(self.kernel, argnums=0)

        self.grad_cost = jacrev(self._combined_rollout_cost, argnums=1)
        self.warmed_up = False

        # sample initial actions
        self.U = self.sigma * torch.randn(self.M, self.H, self.du, device=self.device)
        self.U.requires_grad = True
        self.w = torch.ones(self.M, device=self.device) / self.M
        self.optim = None
        self.reset()

    def _combined_rollout_cost(self, x, u):
        pred_x = self._rollout_dynamics(x, u)
        return self._cost(pred_x, u, False)

    def _cost(self, x, u, normalize=True):
        xu = torch.cat((x, u), dim=-1)
        J = self.problem.objective(xu)
        if normalize:
            return (J - J.min()) / J.max()
        return J

    def _rollout_dynamics(self, x0, u):
        N, H, du = u.shape
        assert H == self.H
        assert du == self.du

        x = [x0.reshape(1, self.dx).repeat(N, 1)]

        for t in range(self.H):
            x.append(self.problem.dynamics(x[-1], u[:, t]))

        return torch.stack(x[1:], dim=1)

    def step(self, state, **kwargs):
        self.U = self.U.detach()
        self.U.requires_grad = True
        self.optim = torch.optim.Adam(params=[self.U], lr=self.lr)

        if self.fixed_H:
            new_T = None
        else:
            new_T = self.problem.T - 1
            self.H = new_T

        self.problem.update(state, T=new_T, **kwargs)

        if self.warmed_up:
            iterations = self.online_iters
        else:
            iterations = self.warmup_iters
            self.warmed_up = True

        for _ in range(iterations):
            self.update_particles(state)

        with torch.no_grad():
            pred_x = self._rollout_dynamics(state, self.U)
            # compute costs
            costs = self._cost(pred_x, self.U)

            # Update weights
            weights = torch.softmax(-costs / self.lambda_, dim=0)
            U = self.U

        self.U = U[torch.argsort(weights, descending=True)[:self.M]]
        self.weights = U[torch.argsort(weights, descending=True)[:self.M]]

        pred_X = self._rollout_dynamics(state, self.U)
        pred_traj = torch.cat((pred_X, self.U), dim=-1)

        # shift actions & psterior
        self.U = torch.roll(self.U, -1, dims=1)
        self.U[:, -1] = self.sigma * torch.randn(self.M, self.du, device=self.device)

        return pred_traj[0].detach(), pred_traj.detach()

    def update_particles(self, state):

        # get kernel and kernel gradient
        self.U.requires_grad = True
        Kxx = self.kernel(self.U, self.U)
        grad_K = self.kernel_grad(self.U, self.U)
        grad_K = torch.mean(torch.einsum('nmmti->nmti', grad_K), dim=0)

        if self.use_true_grad:
            grad_lik = - self.grad_cost(state, self.U) / self.lambda_
            grad_lik = torch.einsum('nnti->nti', grad_lik)
        else:
            # first N actions from each mixture
            with torch.no_grad():
                noise = torch.randn(self.N, self.M, self.H, self.du, device=self.device)
                U_samples = self.sigma * noise + self.U.unsqueeze(0)

                # rollout trajectories
                pred_x = self._rollout_dynamics(state, U_samples.reshape(-1, self.H, self.du))
                costs = self._cost(pred_x, U_samples.reshape(-1, self.H, self.du)).reshape(self.N, self.M)

                # Evaluate cost of action samples - evaluate each action set N times
                weights = torch.softmax(-costs / self.lambda_, dim=0)

                grad_lik = d_log_qU(U_samples, self.U.unsqueeze(0), self.sigma)
                grad_lik = (weights.reshape(self.N, self.M, 1, 1) * grad_lik).sum(dim=0)

        phi = grad_K + torch.tensordot(Kxx, grad_lik, 1) / self.M
        self.U.grad = -phi
        self.optim.step()
        self.optim.zero_grad()

    def reset(self):
        # sample initial actions
        self.U = self.sigma * torch.randn_like(self.U)
        self.U.requires_grad = True
        self.warmed_up = False
        self.weights = torch.ones(self.M, device=self.device) / self.M
