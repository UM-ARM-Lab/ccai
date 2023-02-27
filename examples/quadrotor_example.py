import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!

from functools import partial

from functorch import vmap, jacrev, hessian
from ccai.kernels import rbf_kernel, structured_rbf_kernel
from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
from ccai.quadrotor_env import QuadrotorEnv
from ccai.gp import GPSurfaceModel

import argparse


def cost(trajectory, goal):
    x = trajectory[:, :12]
    u = trajectory[:, 12:]
    Q = torch.eye(12, device=trajectory.device)
    Q[3:] *= 0.0
    P = 10 * torch.eye(12, device=trajectory.device)
    R = 0.001*torch.eye(4, device=trajectory.device)

    d2goal = x - goal.reshape(-1, 12)

    running_state_cost = d2goal.reshape(-1, 1, 12) @ Q.reshape(1, 12, 12) @ d2goal.reshape(-1, 12, 1)
    running_control_cost = u.reshape(-1, 1, 4) @ R.reshape(1, 4, 4) @ u.reshape(-1, 4, 1)
    terminal_state_cost = d2goal[-1].reshape(1, 12) @ P @ d2goal[-1].reshape(12, 1)

    cost = torch.sum(running_control_cost + running_state_cost, dim=0) + terminal_state_cost
    return 0.01*cost.reshape(-1)


class Quadrotor12DDynamics(torch.nn.Module):

    def __init__(self, dt):
        super().__init__()
        self.dt = dt

    def forward(self, state, control):
        ''' unroll state '''
        g = -9.81
        m = 1
        Ix, Iy, Iz = 0.5, 0.1, 0.3
        K = 5
        x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r = torch.chunk(state, chunks=12, dim=-1)

        u1, u2, u3, u4 = torch.chunk(control, chunks=4, dim=-1)

        # Trigonometric fcns on all the angles needed for dynamics
        cphi = torch.cos(phi)
        ctheta = torch.cos(theta)
        cpsi = torch.cos(psi)

        sphi = torch.sin(phi)
        stheta = torch.sin(theta)
        spsi = torch.sin(psi)

        ttheta = torch.tan(theta)

        x_ddot = -(sphi * spsi + cpsi * cphi * stheta) * K * u1 / m
        y_ddot = - (cpsi * sphi - cphi * spsi * stheta) * K * u1 / m
        z_ddot = g - (cphi * ctheta) * K * u1 / m

        p_dot = ((Iy - Iz) * q * r + K * u2) / Ix
        q_dot = ((Iz - Ix) * p * r + K * u3) / Iy
        r_dot = ((Ix - Iy) * p * q + K * u4) / Iz

        ''' velocities'''
        psi_dot = q * sphi / ctheta + r * cphi / ctheta
        theta_dot = q * cphi - r * sphi
        phi_dot = p + q * sphi * ttheta + r * cphi * ttheta

        dstate = torch.cat((x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot,
                            x_ddot, y_ddot, z_ddot, p_dot, q_dot, r_dot), dim=-1)

        return state + dstate * self.dt


class QuadrotorProblem:

    def __init__(self, start, goal, T, device='cuda:0'):
        self.T = T
        self.dx = 12
        self.du = 4
        self.dz = 0
        self.dh = self.dz * T
        self.dg = 12 * T + (T-1)
        #self.dg = 2 * T - 1
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

        self.J = vmap(partial(cost, goal=goal))
        self.dJ = vmap(jacrev(partial(cost, goal=goal)))
        self.HJ = vmap(hessian(partial(cost, goal=goal)))

        self.start = start
        self.goal = goal
        # self.g = vmap(self._combined_contraints)
        # self.grad_g = vmap(jacrev(self._combined_contraints))
        # self.hess_g = vmap(hessian(self._combined_contraints))

        kernel = structured_rbf_kernel
        self.K = kernel
        self.dK = jacrev(kernel, argnums=0)
        self.x_max = torch.ones(self.dx + self.du)
        self.x_max[:3] = 5
        self.x_max[3:6] = torch.pi
        self.x_max[6:12] = 2
        self.x_max[12:] = 10
        self.x_max = 1000 * torch.ones(self.dx + self.du)
        self.x_min = -self.x_max

    def _dynamics_constraint(self, trajectory):
        x = trajectory[:, :12]
        u = trajectory[:, 12:]

        current_x = torch.cat((self.start.reshape(1, 12), x[:-1]), dim=0)
        next_x = x

        pred_next_x = self.dynamics(current_x, u)

        diff = pred_next_x - next_x
        return torch.reshape(pred_next_x - next_x, (-1,))
        return torch.reshape(torch.sum((pred_next_x - next_x)**2, dim=1), (-1,))

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

    def combined_constraints(self, trajectory):
        trajectory = trajectory.reshape(-1, self.T, self.dx + self.du)
        N = trajectory.shape[0]

        # total problem dimension
        prob_dim = self.T * (self.dx + self.du)

        # dynamics constraint
        dynamics_constr = self.dynamics_constraint(trajectory).reshape(N, -1)
        grad_dynamics_constr = self.grad_dynamics(trajectory).reshape(N, -1, prob_dim)
        hess_dynamics_constr = self.hess_dynamics(trajectory).reshape(N, -1, prob_dim, prob_dim)

        # surface constraint
        surf_constr, grad_surf_constr, hess_surf_constr = self.height_constraint(trajectory)

        # currently only take derivative of height at time t wrt state at time t - need to include other times
        # (all derivatives and hessians will be zero)
        grad_surf_constr = torch.diag_embed(grad_surf_constr.permute(0, 2, 1))
        grad_surf_constr = grad_surf_constr.permute(0, 2, 3, 1).reshape(N, self.T, prob_dim)
        hess_surf_constr = torch.diag_embed(torch.diag_embed(hess_surf_constr.permute(0, 2, 3, 1)))
        hess_surf_constr = hess_surf_constr.permute(0, 3, 4, 1, 5, 2).reshape(N, self.T, prob_dim, prob_dim)

        g = torch.cat((dynamics_constr, surf_constr[:, 1:]), dim=1)
        Dg = torch.cat((grad_dynamics_constr, grad_surf_constr[:, 1:]), dim=1)
        DDg = torch.cat((hess_dynamics_constr, hess_surf_constr[:, 1:]), dim=1)

        #print(surf_constr.abs().max(), dynamics_constr.abs().max())
        #return dynamics_constr, grad_dynamics_constr, hess_dynamics_constr
        return g, Dg, DDg

    def eval(self, augmented_trajectory):
        N = augmented_trajectory.shape[0]
        trajectory = augmented_trajectory.reshape(N, self.T, -1)

        grad_cost = self.dJ(trajectory).reshape(N, self.T, -1)
        grad_cost = torch.cat((grad_cost,
                               torch.zeros(N, self.T, self.dz, device=trajectory.device)
                               ), dim=2).reshape(N, -1)

        # compute kernel and grad kernel
        #Xk = trajectory.reshape(N, -1)
        K = self.K(trajectory, trajectory)
        grad_K = -self.dK(trajectory, trajectory).reshape(N, N, N, -1)
        grad_K = torch.einsum('nmmi->nmi', grad_K)
        grad_K = torch.cat((grad_K.reshape(N, N, self.T, self.dx + self.du),
                            torch.zeros(N, N, self.T, self.dz, device=trajectory.device)), dim=-1)
        grad_K = grad_K.reshape(N, N, -1)

        # Now we need to compute constraints and their first and second partial derivatives
        g, Dg, DDg = self.combined_constraints(trajectory)

        print(g.abs().max())
        print(self.J(trajectory).reshape(-1))
        return grad_cost.detach(), K.detach(), grad_K.detach(), g.detach(), Dg.detach(), DDg.detach()

    def constraints(self, trajectory):
        return self.combined_constraints(trajectory)[0]

if __name__ == "__main__":
    N = 8
    T = 12
    device = 'cuda:0'
    plt.ion()
    env = QuadrotorEnv('surface_data.npz')
    env.reset()
    env.render()
    u = 0.1*torch.randn(N, T, 4, device=device)
    start = torch.from_numpy(env.state).to(dtype=torch.float32, device=device)
    goal = torch.zeros(12, device=device)
    problem = QuadrotorProblem(start, goal, T)

    # get initial x0
    x = [start.repeat(N, 1)]
    for t in range(T):
        x.append(problem.dynamics(x[-1], u[:, t]))

    trajectory = torch.cat((torch.stack(x[1:], dim=1), u), dim=2)
    #trajectory = torch.randn_like(trajectory)
    alpha_C = 1
    alpha_J = 0.5
    solver_dt = 0.1
    solver = ConstrainedSteinTrajOpt(problem, dt=solver_dt, alpha_J=alpha_J, alpha_C=alpha_C)
    solver.iters = 500
    trajectory = solver.solve(trajectory)
    num_steps = 100
    online_iters = 25
    for _ in range(num_steps):
        start = torch.from_numpy(env.state).to(dtype=torch.float32, device=device)
        problem = QuadrotorProblem(start, goal, T)
        solver = ConstrainedSteinTrajOpt(problem, dt=solver_dt, alpha_J=alpha_J, alpha_C=alpha_C)
        solver.iters = online_iters
        trajectory = solver.solve(trajectory)

        costs = problem.J(trajectory)
        lowest_cost_idx = torch.argmin(costs)

        u = trajectory[lowest_cost_idx, 0, -4:].detach().cpu().numpy()
        env.step(u)
        ax = env.render()
        for traj in trajectory:
            traj_np = traj.detach().cpu().numpy()
            ax.plot(traj_np[:, 0], traj_np[:, 1], traj_np[:, 2])
        ax.view_init(60, -50)
        ax.axes.set_xlim3d(left=-5, right=5)
        ax.axes.set_ylim3d(bottom=-5, top=5)
        ax.axes.set_zlim3d(bottom=-5, top=5)

        plt.draw()
        plt.pause(0.1)
        #print('--')
        #print(start)
        #print(env.state)
        #print(trajectory[lowest_cost_idx, 0, :12])
        #exit(0)
        # display all planned trajectories

        # prepare trajectory for next time
        trajectory = torch.roll(trajectory, shifts=-1, dims=1)
        trajectory[:, -1, -4:] = 0.1*torch.randn(N, 4, device=device)
        trajectory[:, -1, :12] = problem.dynamics(trajectory[:, -2, :12], trajectory[:, -1, -4:])

    plt.show(block=True)