import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!
import time
from functools import partial
from functorch import vmap, jacrev, hessian

from ccai.kernels import rbf_kernel, structured_rbf_kernel
from ccai.quadrotor_env import QuadrotorEnv
from ccai.quadrotor import Quadrotor12DDynamics
from ccai.gp import GPSurfaceModel

from ccai.mpc.csvgd import Constrained_SVGD_MPC
from ccai.mpc.ipopt import IpoptMPC
from ccai.mpc.mppi import MPPI
from ccai.mpc.svgd import SVMPC
import argparse
import yaml
from ccai.problem import ConstrainedSVGDProblem, IpoptProblem, UnconstrainedPenaltyProblem
from ccai.batched_stein_gradient import compute_constrained_gradient
from ccai.dataset import QuadrotorSingleConstraintTrajectoryDataset
import pathlib
from torch import nn
import tqdm

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

from flow_mpc.flows import ResNetCouplingLayer as CouplingLayer
from flow_mpc.flows import RandomPermutation, BatchNormFlow, ActNorm
from flow_mpc.flows import LULinear
from flow_mpc.flows.sequential import SequentialFlow
from flow_mpc.flows.splits_and_priors import ConditionalSplitFlow, ConditionalPrior
from flow_mpc.flows.OTFlow.ot_flow import OTFlow


def build_nvp_flow(T, dx, du, context_dim, flow_length, split_prior=False):
    flow_chain = []
    flow_dim = T * (dx + du)
    if split_prior:
        flow_chain.append(ConditionalSplitFlow(flow_dim, context_dim=context_dim, z_split_dim=T * du))

    for _ in range(flow_length):
        flow_chain.append(CouplingLayer(flow_dim, context_dim=context_dim))
        # flow_chain.append(BatchNormFlow(flow_dim))
        # flow_chain.append(RandomPermutation(flow_dim))
        flow_chain.append(LULinear(flow_dim, context_dim=context_dim))

    flow_chain.append(CouplingLayer(flow_dim, context_dim=context_dim))

    return SequentialFlow(flow_chain)


def cost(trajectory, goal):
    x = trajectory[:, :12]
    u = trajectory[:, 12:]
    T = x.shape[0]
    Q = torch.eye(12, device=trajectory.device)
    Q[5, 5] = 1e-2
    Q[2, 2] = 1e-3
    Q[3:, 3:] *= 0.5
    P = Q * 100
    R = 1 * torch.eye(4, device=trajectory.device)
    P[5, 5] = 1e-2
    d2goal = x - goal.reshape(-1, 12)

    running_state_cost = d2goal.reshape(-1, 1, 12) @ Q.reshape(1, 12, 12) @ d2goal.reshape(-1, 12, 1)
    running_control_cost = u.reshape(-1, 1, 4) @ R.reshape(1, 4, 4) @ u.reshape(-1, 4, 1)
    terminal_state_cost = d2goal[-1].reshape(1, 12) @ P @ d2goal[-1].reshape(12, 1)

    cost = torch.sum(running_control_cost + running_state_cost, dim=0) + terminal_state_cost

    # Compute cost grad analytically
    grad_running_state_cost = Q.reshape(1, 12, 12) @ d2goal.reshape(-1, 12, 1)
    grad_running_control_cost = R.reshape(1, 4, 4) @ u.reshape(-1, 4, 1)
    grad_terminal_cost = P @ d2goal[-1].reshape(12, 1)  # only on terminal
    grad_terminal_cost = torch.cat((torch.zeros(T - 1, 12, 1, device=trajectory.device),
                                    grad_terminal_cost.unsqueeze(0)), dim=0)

    grad_cost = torch.cat((grad_running_state_cost + grad_terminal_cost, grad_running_control_cost), dim=1)
    grad_cost = grad_cost.reshape(T * 16)

    # compute hessian of cost analytically
    running_state_hess = Q.reshape(1, 12, 12).repeat(T, 1, 1)
    running_control_hess = R.reshape(1, 4, 4).repeat(T, 1, 1)
    terminal_hess = torch.cat((torch.zeros(T - 1, 12, 12, device=trajectory.device), P.unsqueeze(0)), dim=0)

    state_hess = running_state_hess + terminal_hess
    hess_cost = torch.cat((
        torch.cat((state_hess, torch.zeros(T, 4, 12, device=trajectory.device)), dim=1),
        torch.cat((torch.zeros(T, 12, 4, device=trajectory.device), running_control_hess), dim=1)
    ), dim=2)  # will be N x T x 16 x 16

    # now we need to refactor hess to be (N x Td x Td)
    hess_cost = torch.diag_embed(hess_cost.permute(1, 2, 0)).permute(2, 0, 3, 1).reshape(T * 16, T * 16)

    return cost.flatten(), grad_cost, hess_cost


class QuadrotorProblem(ConstrainedSVGDProblem):

    def __init__(self, T, device='cuda:0', alpha=0.1, include_obstacle=False):
        super().__init__(None, None, T, device)
        self.T = T
        self.dx = 12
        self.du = 4
        if include_obstacle:
            self.dz = 1
        else:
            self.dz = 0

        self.dh = self.dz * T
        self.dg = 12 * T + T - 1
        self.alpha = alpha
        self.include_obstacle = include_obstacle
        # self.dg = 2 * T - 1
        data = np.load('surface_data.npz')
        # GP which models surface
        self.surface_gp = GPSurfaceModel(torch.from_numpy(data['xy']).to(dtype=torch.float32, device=device),
                                         torch.from_numpy(data['z']).to(dtype=torch.float32, device=device))

        # gradient and hessian of dynamics
        self._dynamics = Quadrotor12DDynamics(dt=0.1)
        self.dynamics_constraint = vmap(vmap(self._dynamics_constraint))
        self.grad_dynamics = vmap(vmap(jacrev(self._dynamics_constraint)))
        self.hess_dynamics = vmap(vmap(hessian(self._dynamics_constraint)))

        self.height_constraint = vmap(vmap(self._height_constraint))

        self._obj = vmap(vmap(partial(cost)))

        # kernel = structured_rbf_kernel
        kernel = rbf_kernel
        self.K = vmap(kernel)
        self.dK = vmap(jacrev(kernel, argnums=0))
        self.x_max = torch.ones(self.dx + self.du)
        self.x_max[:3] = 6
        self.x_max[3:5] = 0.4 * torch.pi
        self.x_max[5] = 1000  # torch.pi
        self.x_max[6:12] = 1000
        self.x_max[12:] = 1000
        # self.x_max = 1000 * torch.ones(self.dx + self.du)
        self.x_min = -self.x_max

        self.obstacle_centre = torch.tensor([0.0, 0.0], device=self.device)
        self.obstacle_rad = 1.01

    def _objective(self, x, goals):

        J, dJ, HJ = self._obj(x, goals)
        return J * self.alpha, dJ * self.alpha, HJ * self.alpha

    def dynamics(self, x, u):
        return self._dynamics(x, u)

    def _dynamics_constraint(self, trajectory):
        x = trajectory[:, :12]
        u = trajectory[:-1, 12:]
        # current_x = torch.cat((self.start.reshape(1, 12), x[:-1]), dim=0)
        current_x = x[:-1]
        next_x = x[1:]
        pred_next_x = self.dynamics(current_x, u)
        return torch.reshape(pred_next_x - next_x, (-1,))

    def _height_constraint(self, trajectory):
        """

        :param trajectory: T x 16
        :return constr: T
        :return grad_constr: T x 16
        :return hess_constr: T x 16 x 16
        """

        T = trajectory.shape[0]
        xy, z = trajectory[:, :2], trajectory[:, 2]

        # compute z of surface and gradient and hessian
        surface_z, grad_surface_z, hess_surface_z = self.surface_gp.posterior_mean(xy)

        constr = surface_z - z
        grad_constr = torch.cat((grad_surface_z,
                                 -torch.ones(T, 1, device=trajectory.device),
                                 torch.zeros(T, 13, device=trajectory.device)), dim=1)
        hess_constr = torch.cat((
            torch.cat((hess_surface_z, torch.zeros(T, 2, 14, device=trajectory.device)), dim=2),
            torch.zeros(T, 14, 16, device=trajectory.device)), dim=1)

        return constr, grad_constr, hess_constr

    def _con_eq(self, trajectory, starts, compute_grads=True):
        """

        :param trajectory: (B, N, T, dx + du)
        :param starts: (B, dx)
        :param compute_grads: (bool)
        :return g: (B, N, dg)
        :return grad_g: (B, N, dg, T * (dx + du))
        :return hess_g: (B, N, dg, T * (dx + du), T * (dx + du))

        """
        B, N = trajectory.shape[:2]

        # total problem dimension
        prob_dim = self.T * (self.dx + self.du)

        # dynamics constraint
        dynamics_constr = self.dynamics_constraint(trajectory).reshape(B, N, -1)

        # surface constraint
        surf_constr, grad_surf_constr, hess_surf_constr = self.height_constraint(trajectory)

        # start constraint
        start_constraint = (trajectory[:, :, 0, :12] - starts.reshape(B, 1, 12)).reshape(B, N, 12)

        g = torch.cat((dynamics_constr, start_constraint, surf_constr[:, :, 1:]), dim=2)

        if not compute_grads:
            return g, None, None

        grad_dynamics_constr = self.grad_dynamics(trajectory).reshape(B, N, -1, prob_dim)
        hess_dynamics_constr = self.hess_dynamics(trajectory).reshape(B, N, -1, prob_dim, prob_dim)

        # currently only take derivative of height at time t wrt state at time t - need to include other times
        # (all derivatives and hessians will be zero)
        grad_surf_constr = torch.diag_embed(grad_surf_constr.permute(0, 1, 3, 2))  # B x N x 16 x T x T
        grad_surf_constr = grad_surf_constr.permute(0, 1, 3, 4, 2).reshape(B, N, self.T, prob_dim)
        hess_surf_constr = torch.diag_embed(torch.diag_embed(hess_surf_constr.permute(0, 1, 3, 4, 2)))
        hess_surf_constr = hess_surf_constr.permute(0, 1, 4, 5, 2, 6, 3).reshape(B, N, self.T, prob_dim, prob_dim)

        grad_start_constraint = torch.zeros(B, N, 12, self.T, 16, device=trajectory.device)
        grad_start_constraint[:, :, :12, 0, :12] = torch.eye(12, device=trajectory.device)
        grad_start_constraint = grad_start_constraint.reshape(B, N, 12, -1)
        hess_start_constraint = torch.zeros(B, N, 12, self.T * 16, self.T * 16, device=trajectory.device)

        Dg = torch.cat((grad_dynamics_constr, grad_start_constraint, grad_surf_constr[:, :, 1:]), dim=2)
        DDg = torch.cat((hess_dynamics_constr, hess_start_constraint, hess_surf_constr[:, :, 1:]), dim=2)

        return g, Dg, DDg

    def _con_ineq(self, trajectory, compute_grads=True):
        if not self.include_obstacle:
            return None, None, None

        N = trajectory.shape[0]
        xy = trajectory.reshape(N, self.T, -1)[:, :, :2] - self.obstacle_centre.reshape(1, 1, 2)

        h = self.obstacle_rad ** 2 - torch.sum(xy ** 2, dim=-1)  # N x T
        if not compute_grads:
            return h, None, None

        # xy is (N, T, 2)
        grad_h = torch.zeros(N, self.T, self.T, self.dx + self.du, device=self.device)
        # make grad_xy (N, T, T, 2)
        grad_h_xy = -2 * torch.diag_embed(xy.permute(0, 2, 1)).permute(0, 2, 3, 1)  # (N, T, T, 2)
        grad_h[:, :, :, :2] = grad_h_xy
        grad_h = grad_h.reshape(N, self.T, self.T * (self.dx + self.du))

        hess_h_xy = -torch.eye(2, device=self.device).reshape(1, 1, 2, 2).repeat(N, self.T, 1, 1)
        # need to make (N, T, T, 2, T, 2)
        hess_h_xy = torch.diag_embed(torch.diag_embed(hess_h_xy.permute(0, 2, 3, 1)))  # now (N, 2, 2, T, T, T)
        hess_h_xy = hess_h_xy.permute(0, 3, 4, 1, 5, 2)  # now (N, T, T, 2, T, 2)
        hess_h = torch.zeros(N, self.T, self.T, self.dx + self.du, self.T, self.dx + self.du, device=self.device)
        hess_h[:, :, :, :2, :, :2] = hess_h_xy
        hess_h = hess_h.reshape(N, self.T, self.T * (self.dx + self.du),
                                self.T * (self.dx + self.du))
        return h, grad_h, hess_h

    def eval(self, augmented_trajectory, starts, goals):
        B, N = augmented_trajectory.shape[:2]

        trajectory = augmented_trajectory.reshape(B, N, self.T, -1)[:, :, :self.dx + self.du]
        cost, grad_cost, hess_cost = self._objective(trajectory, goals.reshape(B, 1, -1).repeat(1, N, 1))
        M = None

        grad_cost = torch.cat((grad_cost.reshape(B, N, self.T, -1),
                               torch.zeros(B, N, self.T, self.dz, device=trajectory.device)
                               ), dim=3).reshape(B, N, -1)

        # compute kernel and grad kernel
        Xk = trajectory  # .reshape(N, -1)
        K = self.K(Xk, Xk)
        grad_K = -self.dK(Xk, Xk).reshape(B, N, N, N, -1)
        grad_K = torch.einsum('bnmmi->bnmi', grad_K)
        grad_K = torch.cat((grad_K.reshape(B, N, N, self.T, self.dx + self.du),
                            torch.zeros(B, N, N, self.T, self.dz, device=trajectory.device)), dim=-1)
        grad_K = grad_K.reshape(B, N, N, -1)

        # Now we need to compute constraints and their first and second partial derivatives
        g, Dg, DDg = self.combined_constraints(augmented_trajectory, starts)
        # print(cost.mean(), g.abs().mean())

        return cost.detach(), grad_cost.detach(), hess_cost, K.detach(), grad_K.detach(), g.detach(), Dg.detach(), DDg.detach()

    def update(self, start, goal, T):
        pass

    def get_initial_xu(self, N):
        pass


class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, z, context):
        x = torch.cat((z, context), dim=1)
        dz = self.net(x)
        return dz


class TrajectoryFlowModel(nn.Module):

    def __init__(self, T, dx, du, context_dim):
        super().__init__()
        self.T = T
        self.dx = dx
        self.du = du
        split_prior = False
        self.flow = build_nvp_flow(T, dx, du, context_dim, 16, split_prior)

        if split_prior:
            prior_dim = T * du
        else:
            prior_dim = T * (dx + du)

        mu = torch.zeros(prior_dim)
        sigma = torch.ones(prior_dim)

        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)

    def sample(self, start, goal, constraints=None):
        B = start.shape[0]
        context = torch.cat((start, goal), dim=1)
        if constraints is not None:
            context = torch.cat((context, constraints), dim=1)

        prior = torch.distributions.Normal(self.mu, self.sigma)
        z = prior.sample(sample_shape=torch.Size([B, ]))
        log_prob = prior.log_prob(z).sum(dim=-1)
        out = self.flow(z, logpx=log_prob, context=context, reverse=False)
        trajectories, log_prob = out[:2]
        return trajectories.reshape(B, self.T, self.dx + self.du), log_prob.reshape(B)

    def log_prob(self, trajectories, start, goal, constraints=None):
        B = start.shape[0]
        context = torch.cat((start, goal), dim=1)
        if constraints is not None:
            context = torch.cat((context, constraints), dim=1)
        log_prob = torch.zeros(B, device=trajectories.device)
        out = self.flow(trajectories.reshape(B, -1), logpx=log_prob, context=context, reverse=True)
        z, delta_logp = out[:2]
        log_prob = torch.distributions.Normal(self.mu, self.sigma).log_prob(z).sum(dim=-1) + delta_logp
        return log_prob.reshape(B)


def train_model_from_demonstrations():
    model = TrajectoryFlowModel(T=12, dx=12, du=4, context_dim=12 + 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    from torch.utils.data import random_split, DataLoader, RandomSampler

    dataset = QuadrotorSingleConstraintTrajectoryDataset('../data/quadrotor_data_collection_single_constraint')

    train_dataset, val_dataset = random_split(dataset, [int(0.9 * len(dataset)),
                                                        len(dataset) - int(0.9 * len(dataset))])

    train_sampler = RandomSampler(train_dataset)
    val_sampler = RandomSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=512, sampler=val_sampler)

    epochs = 500
    pbar = tqdm.tqdm(range(epochs))

    model.cuda()
    for epoch in pbar:
        train_loss = 0.0
        val_loss = 0.0
        model.train()
        for trajectories, starts, goals in train_loader:
            trajectories = trajectories.to(device='cuda:0')
            B, N, T, dxu = trajectories.shape
            starts = starts.to(device='cuda:0')
            goals = goals.to(device='cuda:0')
            optimizer.zero_grad()
            log_prob = model.log_prob(trajectories.reshape(B * N, T, dxu),
                                      starts.reshape(B * N, -1),
                                      goals.reshape(B * N, -1))
            loss = -log_prob.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            train_loss += loss.item()

        with torch.no_grad():
            model.eval()
            for trajectories, starts, goals in val_loader:
                trajectories = trajectories.to(device='cuda:0')
                B, N, T, dxu = trajectories.shape
                starts = starts.to(device='cuda:0')
                goals = goals.to(device='cuda:0')
                log_prob = model.log_prob(trajectories.reshape(B * N, T, dxu),
                                          starts.reshape(B * N, -1),
                                          goals.reshape(B * N, -1))
                loss = -log_prob.mean()
                val_loss += loss.item()

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        pbar.set_description(
            f'Train loss {train_loss:.3f} Val loss {val_loss:.3f}')
        # generate samples and plot them
        if (epoch + 1) % 100 == 0:
            for _, starts, goals in val_loader:
                starts = starts.to(device='cuda:0')
                goals = goals.to(device='cuda:0')
                N = starts.shape[1]
                starts = starts[:16].reshape(-1, 12)
                goals = goals[:16].reshape(-1, 3)
                with torch.no_grad():
                    trajectories, _ = model.sample(starts, goals)
                trajectories = trajectories.reshape(16, N, 12, 16)
                starts = starts.reshape(16, N, 12)
                goals = goals.reshape(16, N, 3)
                for i, trajs in enumerate(trajectories):
                    env = QuadrotorEnv('surface_data.npz')
                    env.state = starts[i, 0].cpu().numpy()
                    env.goal = goals[i, 0].cpu().numpy()
                    ax = env.render_init()
                    update_plot_with_trajectories(ax, trajs.reshape(N, 12, 16))
                    plt.savefig(
                        f'learning_plots/demonstrations/learned_sampler_quadrotor_single_constraint_{epoch}_{i}.png')
                    plt.close()

                break

    torch.save(model.state_dict(), 'learned_sampler_quadrotor_single_constraint_from_demonstration.pt')


def train_multi_constraint_model_from_demonstrations():
    model = TrajectoryFlowModel(T=12, dx=12, du=4, context_dim=12 + 3 + 100)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    from torch.utils.data import random_split, DataLoader, RandomSampler

    dataset = QuadrotorSingleConstraintTrajectoryDataset('../data/quadrotor_data_collection_multi_constraint')

    train_dataset, val_dataset = random_split(dataset, [int(0.9 * len(dataset)),
                                                        len(dataset) - int(0.9 * len(dataset))])

    train_sampler = RandomSampler(train_dataset)
    val_sampler = RandomSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=512, sampler=val_sampler)

    epochs = 500
    pbar = tqdm.tqdm(range(epochs))

    model.cuda()
    for epoch in pbar:
        train_loss = 0.0
        val_loss = 0.0
        model.train()
        for trajectories, starts, goals, constraints in train_loader:
            trajectories = trajectories.to(device='cuda:0')
            B, N, T, dxu = trajectories.shape
            starts = starts.to(device='cuda:0')
            goals = goals.to(device='cuda:0')
            constraints = constraints.to(device='cuda:0')

            optimizer.zero_grad()
            log_prob = model.log_prob(trajectories.reshape(B * N, T, dxu),
                                      starts.reshape(B * N, -1),
                                      goals.reshape(B * N, -1),
                                      constraints.reshape(B * N, -1))

            loss = -log_prob.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            train_loss += loss.item()

        with torch.no_grad():
            model.eval()
            for trajectories, starts, goals, constraints in val_loader:
                trajectories = trajectories.to(device='cuda:0')
                B, N, T, dxu = trajectories.shape
                starts = starts.to(device='cuda:0')
                goals = goals.to(device='cuda:0')
                constraints = constraints.to(device='cuda:0')
                log_prob = model.log_prob(trajectories.reshape(B * N, T, dxu),
                                          starts.reshape(B * N, -1),
                                          goals.reshape(B * N, -1),
                                          constraints.reshape(B * N, -1)
                                          )
                loss = -log_prob.mean()
                val_loss += loss.item()

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        pbar.set_description(
            f'Train loss {train_loss:.3f} Val loss {val_loss:.3f}')
        # generate samples and plot them
        if (epoch + 1) % 100 == 0:
            for _, starts, goals, constraints in val_loader:
                starts = starts.to(device='cuda:0')
                goals = goals.to(device='cuda:0')
                N = starts.shape[1]
                starts = starts[:16].reshape(-1, 12)
                goals = goals[:16].reshape(-1, 3)
                constraints = constraints[:16].reshape(-1, 100).to(device='cuda:0')
                with torch.no_grad():
                    trajectories, _ = model.sample(starts, goals, constraints)
                trajectories = trajectories.reshape(16, N, 12, 16)
                starts = starts.reshape(16, N, 12)
                goals = goals.reshape(16, N, 3)
                constraints = constraints.reshape(16, N, 100)
                for i, trajs in enumerate(trajectories):
                    env = QuadrotorEnv('surface_data.npz')
                    xy_data = env.surface_model.train_x.cpu().numpy()
                    z_data = constraints.cpu().numpy()
                    np.savez('tmp_surface_data.npz', xy_data=xy_data, z_data=z_data)
                    env = QuadrotorEnv('tmp_surface_data.npz')
                    env.state = starts[i, 0].cpu().numpy()
                    env.goal = goals[i, 0].cpu().numpy()
                    ax = env.render_init()
                    update_plot_with_trajectories(ax, trajs.reshape(N, 12, 16))
                    plt.savefig(
                        f'learning_plots/demonstrations/learned_sampler_quadrotor_single_constraint_{epoch}_{i}.png')
                    plt.close()

                break

    torch.save(model.state_dict(), 'learned_sampler_quadrotor_multi_constraint_from_demonstration.pt')


def fine_tune_model_with_stein(model):
    B, N, T = 64, 8, 12
    batched_problem = QuadrotorProblem(T=12, device='cuda', include_obstacle=False, alpha=0.1)

    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 1000

    pbar = tqdm.tqdm(range(epochs))
    for epoch in pbar:
        # randomly sample starts and goals for the batch
        # For now start at random position but zero velocity and orientation
        starts = torch.zeros(B, 12, device='cuda:0')
        starts[:, 6:] = torch.randn(B, 6, device='cuda:0')
        starts[:, 5] = 2 * torch.pi * torch.rand(B, device='cuda:0') - torch.pi
        starts[:, 3:5] = 0.0 * torch.randn(B, 2, device='cuda:0')
        goals = torch.zeros(B, 12, device='cuda:0')
        starts[:, :3] = 10 * torch.rand(B, 3, device='cuda:0') - 5
        goals[:, :3] = 10 * torch.rand(B, 3, device='cuda:0') - 5
        starts[:, 2] = batched_problem.surface_gp.posterior_mean(starts[:, :2])[0]
        goals[:, 2] = batched_problem.surface_gp.posterior_mean(goals[:, :2])[0]

        # sample trajectories
        s = starts.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
        g = goals.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
        trajectories = model.sample(s, g[:, :3])[0].reshape(B, N, T, 16)

        phi = []
        J = []
        C = []
        for i in range(B):
            # now compute the gradient
            _phi, _J, _C = compute_constrained_gradient(trajectories[i].unsqueeze(0),
                                                        starts[i].unsqueeze(0),
                                                        goals[i].unsqueeze(0), batched_problem, alpha_J=0.1, alpha_C=1)
            phi.append(_phi)
            J.append(_J)
            C.append(_C)

        J = torch.stack(J, dim=0)
        C = torch.stack(C, dim=0)
        phi = torch.stack(phi, dim=0)

        phi = phi.reshape(B, N, batched_problem.T, -1)

        trajectories.backward(phi)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        pbar.set_description(
            f'Average Cost {J.mean().item():.3f} Average Constraint Violation {C.abs().mean().item():.3f}')
        optimizer.step()
        optimizer.zero_grad()

        if (epoch) % 100 == 0:
            for i, trajs in enumerate(trajectories[:16]):
                env = QuadrotorEnv('surface_data.npz')
                env.state = starts[i].cpu().numpy()
                env.goal = goals[i].cpu().numpy()
                ax = env.render_init()
                update_plot_with_trajectories(ax, trajs.reshape(N, T, 16))
                plt.savefig(f'learning_plots/finetuning/learned_sampler_quadrotor_single_constraint_{epoch}_{i}.png')
                plt.close()

    torch.save(model.state_dict(), 'learned_sampler_quadrotor_single_constraint_finetuned.pt')


def update_plot_with_trajectories(ax, trajectory):
    traj_lines = []
    for traj in trajectory:
        traj_np = traj.detach().cpu().numpy()
        traj_lines.extend(ax.plot(traj_np[1:, 0],
                                  traj_np[1:, 1],
                                  traj_np[1:, 2], color='g', alpha=0.5, linestyle='--'))


if __name__ == "__main__":
    # train_model_from_demonstrations()

    model = TrajectoryFlowModel(T=12, dx=12, du=4, context_dim=12 + 3)
    model.load_state_dict(torch.load('learned_sampler_quadrotor_single_constraint_from_demonstration.pt'))
    model.cuda()
    fine_tune_model_with_stein(model)
