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

import pathlib
from torch import nn
import tqdm

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]


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

        #kernel = structured_rbf_kernel
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
        #print(cost.mean(), g.abs().mean())

        return grad_cost.detach(), hess_cost, K.detach(), grad_K.detach(), g.detach(), Dg.detach(), DDg.detach()

    def update(self, start, goal, T):
        pass

    def get_initial_xu(self, N):
        pass


class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, z, context):
        x = torch.cat((z, context), dim=1)
        dz = self.net(x)
        return dz


def update_plot_with_trajectories(ax, trajectory):
    traj_lines = []
    for traj in trajectory:
        traj_np = traj.detach().cpu().numpy()
        traj_lines.extend(ax.plot(traj_np[1:, 0],
                                  traj_np[1:, 1],
                                  traj_np[1:, 2], color='g', alpha=0.5, linestyle='--'))


if __name__ == "__main__":
    B, N, T = 16, 8, 12
    batched_problem = QuadrotorProblem(T=12, device='cuda', include_obstacle=False, alpha=0.1)
    # now we will train the MLP to predict entire trajectories
    learned_sampler = MLP(4 * T + 12 + 12, 4 * T)
    learned_sampler = learned_sampler.cuda()
    #optimizer = torch.optim.SGD(learned_sampler.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    optimizer = torch.optim.Adam(learned_sampler.parameters(), lr=1e-3)
    epochs = 5000

    for epoch in tqdm.tqdm(range(epochs)):
        # randomly sample starts and goals for the batch
        # For now start at random position but zero velocity and orientation
        starts = torch.zeros(B, 12, device='cuda:0')
        starts[:, 6:] = torch.randn(B, 6, device='cuda:0')
        starts[:, 5] = 2 * torch.pi * torch.rand(B, device='cuda:0') - torch.pi
        starts[:, 3:5] = 0.01 * torch.randn(B, 2, device='cuda:0')
        goals = torch.zeros(B, 12, device='cuda:0')
        starts[:, :3] = 10 * torch.rand(B, 3, device='cuda:0') - 5
        goals[:, :3] = 10 * torch.rand(B, 3, device='cuda:0') - 5
        starts[:, 2] = batched_problem.surface_gp.posterior_mean(starts[:, :2])[0]
        goals[:, 2] = batched_problem.surface_gp.posterior_mean(starts[:, :2])[0]

        # randomly sample latent trajectories
        # randomly sampled latent trajectories are sampld by sampling controls from prior then rolling out dynamics
        z = torch.randn(B, N, T, 4, device='cuda:0')

        # context is start and goal
        context = torch.cat((starts.reshape(B, 1, -1).repeat(1, N, 1),
                            goals.reshape(B, 1, -1).repeat(1, N, 1)), dim=2)

        # use NN to generate trajectories
        u = learned_sampler(z.reshape(B*N, -1), context.reshape(B*N, -1)).reshape(B, N, batched_problem.T, -1)
        x = [starts.reshape(B, 1, -1).repeat(1, N, 1).reshape(B*N, -1)]

        for t in range(batched_problem.T - 1):
            x.append(batched_problem.dynamics(x[-1], u[:, :, t].reshape(B*N, -1)))
        x = torch.stack(x, dim=1)
        trajectories = torch.cat((x.reshape(B, N, -1, 12), u.reshape(B, N, -1, 4)), dim=-1).reshape(B, N, batched_problem.T, 16)


        phi = []
        for i in range(B):
            # now compute the gradient
            phi.append(compute_constrained_gradient(trajectories[i].unsqueeze(0),
                                               starts[i].unsqueeze(0),
                                               goals[i].unsqueeze(0), batched_problem, alpha_J=0.1, alpha_C=1))
        phi = torch.stack(phi, dim=0)

        phi_U = phi.reshape(B, N, batched_problem.T, -1)[:, :, :, 12:]

        u.backward(phi_U)
        torch.nn.utils.clip_grad_norm_(learned_sampler.parameters(), 1)


        optimizer.step()
        optimizer.zero_grad()

    # now let's plot some samples from the flow and see if they make sense
    print(trajectories[0].reshape(N, T, -1)[:, :, 6:9].abs().max())
    print(trajectories[0].reshape(N, T, -1)[:, :, 9:12].abs().max())

    for i, trajs in enumerate(trajectories):
        env = QuadrotorEnv('surface_data.npz')
        env.state = starts[i].cpu().numpy()
        env.goal = goals[i].cpu().numpy()
        ax = env.render_init()
        update_plot_with_trajectories(ax, trajs.reshape(N, T, 16))
        plt.savefig('learned_sampler_quadrotor_single_constraint_{}.png'.format(i))
        plt.close()
    torch.save(learned_sampler.state_dict(), 'learned_sampler_quadrotor_single_constraint.pt')
