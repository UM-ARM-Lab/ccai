import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!

from functools import partial

from functorch import vmap, jacrev, hessian
from ccai.kernels import rbf_kernel, structured_rbf_kernel
from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
from ccai.quadrotor_env import QuadrotorEnv
from ccai.quadrotor import Quadrotor12DDynamics
from ccai.gp import GPSurfaceModel

from ccai.mpc.csvgd import Constrained_SVGD_MPC
from ccai.mpc.ipopt import IpoptMPC
from ccai.mpc.mppi import MPPI
from ccai.mpc.svgd import SVMPC
import argparse

from ccai.problem import ConstrainedSVGDProblem, IpoptProblem, UnconstrainedPenaltyProblem


def cost(trajectory, goal):
    x = trajectory[:, :12]
    u = trajectory[:, 12:]
    T = x.shape[0]
    Q = torch.eye(12, device=trajectory.device)
    Q[2:, 2:] *= 0.5
    Q[5, 5] = 1e-2
    P = Q * 100
    R = torch.eye(4, device=trajectory.device)

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

    def __init__(self, start, goal, T, device='cuda:0', alpha=0.005):
        super().__init__(start, goal, T, device)
        self.T = T
        self.dx = 12
        self.du = 4
        self.dz = 0
        self.dh = self.dz * T
        self.dg = 12 * T + T - 1
        self.alpha = alpha
        # self.dg = 2 * T - 1
        data = np.load('surface_data.npz')
        # GP which models surface
        self.surface_gp = GPSurfaceModel(torch.from_numpy(data['xy']).to(dtype=torch.float32, device=device),
                                         torch.from_numpy(data['z']).to(dtype=torch.float32, device=device))

        # gradient and hessian of dynamics
        self.dynamics = Quadrotor12DDynamics(dt=0.1)
        self.dynamics_constraint = vmap(self._dynamics_constraint)
        self.grad_dynamics = vmap(jacrev(self._dynamics_constraint))
        self.hess_dynamics = vmap(hessian(self._dynamics_constraint))

        self.height_constraint = vmap(self._height_constraint)

        self._objective = vmap(partial(cost, goal=goal))

        self.start = start
        self.goal = goal
        # self.g = vmap(self._combined_contraints)
        # self.grad_g = vmap(jacrev(self._combined_contraints))
        # self.hess_g = vmap(hessian(self._combined_contraints))

        # kernel = structured_rbf_kernel
        kernel = rbf_kernel
        self.K = kernel
        self.dK = jacrev(kernel, argnums=0)
        self.x_max = torch.ones(self.dx + self.du)
        self.x_max[:3] = 6
        self.x_max[3:5] = 0.4 * torch.pi
        self.x_max[5] = 1000  # torch.pi
        self.x_max[6:12] = 1000
        self.x_max[12:] = 1000
        # self.x_max = 1000 * torch.ones(self.dx + self.du)
        self.x_min = -self.x_max

        self.obstacle_centre = torch.tensor([0.0, 0.0], device=self.device)
        self.obstacle_rad = 1.0

    def _objective(self, x):
        return cost(x, self.goal)

    def _dynamics_constraint(self, trajectory):
        x = trajectory[:, :12]
        u = trajectory[:-1, 12:]
        # current_x = torch.cat((self.start.reshape(1, 12), x[:-1]), dim=0)
        current_x = x[:-1]
        next_x = x[1:]
        pred_next_x = self.dynamics(current_x, u)
        return torch.reshape(pred_next_x - next_x, (-1,))

    def _height_constraint(self, trajectory):
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

    def _con_eq(self, trajectory, compute_grads=True):
        trajectory = trajectory.reshape(-1, self.T, self.dx + self.du)
        N = trajectory.shape[0]

        # total problem dimension
        prob_dim = self.T * (self.dx + self.du)

        # dynamics constraint
        dynamics_constr = self.dynamics_constraint(trajectory).reshape(N, -1)
        # surface constraint
        surf_constr, grad_surf_constr, hess_surf_constr = self.height_constraint(trajectory)
        # start constraint
        start_constraint = (trajectory[:, 0, :12] - self.start.reshape(1, 12)).reshape(N, 12)
        g = torch.cat((dynamics_constr, start_constraint, surf_constr[:, 1:]), dim=1)

        if not compute_grads:
            return g, None, None

        grad_dynamics_constr = self.grad_dynamics(trajectory).reshape(N, -1, prob_dim)
        hess_dynamics_constr = self.hess_dynamics(trajectory).reshape(N, -1, prob_dim, prob_dim)

        # currently only take derivative of height at time t wrt state at time t - need to include other times
        # (all derivatives and hessians will be zero)
        grad_surf_constr = torch.diag_embed(grad_surf_constr.permute(0, 2, 1))
        grad_surf_constr = grad_surf_constr.permute(0, 2, 3, 1).reshape(N, self.T, prob_dim)
        hess_surf_constr = torch.diag_embed(torch.diag_embed(hess_surf_constr.permute(0, 2, 3, 1)))
        hess_surf_constr = hess_surf_constr.permute(0, 3, 4, 1, 5, 2).reshape(N, self.T, prob_dim, prob_dim)

        grad_start_constraint = torch.zeros(N, 12, self.T, 16, device=trajectory.device)
        grad_start_constraint[:, :12, 0, :12] = torch.eye(12, device=trajectory.device)
        grad_start_constraint = grad_start_constraint.reshape(N, 12, -1)
        hess_start_constraint = torch.zeros(N, 12, self.T * 16, self.T * 16, device=trajectory.device)

        Dg = torch.cat((grad_dynamics_constr, grad_start_constraint, grad_surf_constr[:, 1:]), dim=1)
        DDg = torch.cat((hess_dynamics_constr, hess_start_constraint, hess_surf_constr[:, 1:]), dim=1)
        # g = torch.cat((dynamics_constr, start_constraint, surf_constr[:, 1:]), dim=1)
        # Dg = torch.cat((grad_dynamics_constr,  grad_start_constraint, grad_surf_constr[:, 1:]), dim=1)
        # DDg = torch.cat((hess_dynamics_constr, hess_start_constraint, hess_surf_constr[:, 1:]), dim=1)

        # print(surf_constr[0])
        # return dynamics_constr, grad_dynamics_constr, hess_dynamics_constr
        return g, Dg, DDg

    def _con_ineq(self, trajectory, compute_grads=False):
        N = trajectory.shape[0]
        xy = trajectory.reshape(N, self.T, -1)[:, :, :2] - self.obstacle_centre.reshape(1, 1, 2)

        h = self.obstacle_rad - torch.sum(xy ** 2, dim=-1)  # N x T
        if not compute_grads:
            return h, None, None

        # xy is (N, T, 2)
        grad_h = torch.zeros(N, self.T, self.T, self.dx + self.du, device=self.device)
        # make grad_xy (N, T, T, 2)
        grad_h_xy = 2 * torch.diag_embed(xy.permute(0, 2, 1)).permute(0, 2, 3, 1)  # (N, T, T, 2)
        grad_h[:, :, :, :2] = grad_h_xy
        grad_h = grad_h.reshape(N, self.T, self.T * (self.dx + self.du))

        hess_h_xy = torch.eye(2, device=device).reshape(1, 1, 2, 2).repeat(N, self.T, 1, 1)
        # need to make (N, T, T, 2, T, 2)
        hess_h_xy = torch.diag_embed(torch.diag_embed(hess_h_xy.permute(0, 2, 3, 1)))  # now (N, 2, 2, T, T, T)
        hess_h_xy = hess_h_xy.permute(0, 3, 4, 1, 5, 2)  # now (N, T, T, 2, T, 2)
        hess_h = torch.zeros(N, self.T, self.T, self.dx + self.du, self.dx + self.du, device=self.device)
        hess_h[:, :, :, :2, :, :2] = hess_h_xy
        hess_h = hess_h.reshape(N, self.T, self.T * (self.dx + self.du))

        return h, grad_h, hess_h

    def eval(self, augmented_trajectory):
        N = augmented_trajectory.shape[0]
        trajectory = augmented_trajectory.reshape(N, self.T, -1)

        cost, grad_cost, hess_cost = self._objective(trajectory)
        grad_cost = grad_cost * self.alpha
        cost = cost * self.alpha
        hess_cost = hess_cost.mean(dim=0).detach()
        # hess_cost = None
        # grad_cost = torch.cat((grad_cost.reshape(),
        #                       torch.zeros(N, self.T, self.dz, device=trajectory.device)
        #                       ), dim=2).reshape(N, -1)

        # compute kernel and grad kernel
        # Xk = trajectory.reshape(N, -1)
        K = self.K(trajectory, trajectory, hess_cost)
        grad_K = -self.dK(trajectory, trajectory, hess_cost).reshape(N, N, N, -1)
        grad_K = torch.einsum('nmmi->nmi', grad_K)
        grad_K = torch.cat((grad_K.reshape(N, N, self.T, self.dx + self.du),
                            torch.zeros(N, N, self.T, self.dz, device=trajectory.device)), dim=-1)
        grad_K = grad_K.reshape(N, N, -1)
        # Now we need to compute constraints and their first and second partial derivatives
        g, Dg, DDg = self.combined_constraints(trajectory)

        print(g.abs().max())
        print(cost.reshape(-1))
        # print(g[0])
        return grad_cost.detach(), hess_cost, K.detach(), grad_K.detach(), g.detach(), Dg.detach(), DDg.detach()

    def update(self, start, goal=None, T=None):
        self.start = start
        if goal is not None:
            self.goal = goal
            self._objective = vmap(partial(cost, goal=goal))
        if T is not None:
            self.T = T
            self.dh = self.dz * T
            self.dg = 12 * T + T - 1

    def get_initial_xu(self, N):
        x = [self.start.repeat(N, 1)]
        u = 0.1 * torch.randn(N, self.T, self.du, device=device)

        for t in range(T - 1):
            x.append(self.dynamics(x[-1], u[:, t]))

        return torch.cat((torch.stack(x, dim=1), u), dim=2)


class QuadrotorIpoptProblem(QuadrotorProblem, IpoptProblem):

    def __init__(self, start, goal, T):
        super().__init__(start, goal, T, device='cpu')


class QuadrotorUnconstrainedProblem(QuadrotorProblem, UnconstrainedPenaltyProblem):
    def __init__(self, start, goal, T, device, penalty):
        super().__init__(start, goal, T, device)
        self.penalty = penalty


if __name__ == "__main__":
    control_method = 'svgd'
    N = 8
    T = 12
    device = 'cpu' if control_method == 'ipopt' else 'cuda:0'
    plt.ion()
    env = QuadrotorEnv('surface_data.npz')
    env.reset()
    env.render()
    params = {
        'N': N,
        'alpha_J': 0.75,
        'alpha_C': 1,
        'step_size': 1,
        'momentum': 0.25,
        'device': device,
        'receding_horizon': True,
        'online_iters': 25,
        'warmup_iters': 250
    }
    start = torch.from_numpy(env.state).to(dtype=torch.float32, device=device)
    goal = torch.zeros(12, device=device)
    goal[:2] = 4

    if control_method == 'ipopt':
        problem = QuadrotorIpoptProblem(start, goal, T)
        controller = IpoptMPC(problem, params)
    elif control_method == 'mppi':
        params['lambda'] = 1e-3
        params['sigma'] = 0.25
        params['N'] = 1000
        problem = QuadrotorUnconstrainedProblem(start, goal, T, device=device, penalty=1000)
        controller = MPPI(problem, params)
    elif control_method == 'svgd':
        params['lambda'] = 1000  # 1e-3
        params['sigma'] = 0.1
        params['step_size'] = 0.1
        params['M'] = N
        params['N'] = 128
        params['use_grad'] = True

        problem = QuadrotorUnconstrainedProblem(start, goal, T, device=device, penalty=1000)
        controller = SVMPC(problem, params)
    else:
        problem = QuadrotorProblem(start, goal, T, device=device)
        controller = Constrained_SVGD_MPC(problem, params)

    num_steps = 50
    for step in range(num_steps):
        start = torch.from_numpy(env.state).to(dtype=torch.float32, device=device)
        best, trajectory = controller.step(start)
        u = best[0, -4:].detach().cpu().numpy()
        env.step(u)

        print(env.state)
        print(f'Constraint violation: {env.get_constraint_violation()}')

        if np.linalg.norm(env.state[:2] - np.array([4, 4])) < 0.5:
            print(f'Goal reached after {step} num steps')
            break

        ax = env.render()
        for traj in trajectory:
            traj_np = traj.detach().cpu().numpy()
            ax.plot(traj_np[1:, 0], traj_np[1:, 1], traj_np[1:, 2], color='g')
        traj_np = best.detach().cpu().numpy()
        ax.plot(traj_np[1:, 0], traj_np[1:, 1], traj_np[1:, 2], color='r')
        ax.view_init(60, -50)
        ax.axes.set_xlim3d(left=-5, right=5)
        ax.axes.set_ylim3d(bottom=-5, top=5)
        ax.axes.set_zlim3d(bottom=-5, top=5)

        plt.draw()
        plt.pause(0.1)

    plt.show(block=True)
