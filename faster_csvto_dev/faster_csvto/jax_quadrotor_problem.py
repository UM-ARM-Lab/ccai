import torch
import numpy as np
from functools import partial
from functorch import vmap, jacrev, hessian

import jax
import jax.numpy as jnp

from ccai.kernels import rbf_kernel, structured_rbf_kernel
from ccai.quadrotor import Quadrotor12DDynamics
from ccai.gp import GPSurfaceModel
from ccai.problem import ConstrainedSVGDProblem, IpoptProblem, UnconstrainedPenaltyProblem, NLOptProblem

"""
NOTE:
We currently only end up using
- problem.eval() - calls
    gives us
        - grad J
        - for other problems hess J (for this problem we won't need it)
        - C
        - grad C
        - hess C
        - K
        - grad K
- problem.get_initial_z()
- problem.get_cost() - wraps _objective(), implemented in this file
- problem.combined_constraints() - wraps _con_eq() and _con_ineq()
"""


def cost(trajectory: jnp.array, goal: jnp.array):
    """ Get the cost value, gradient, and hessian for a trajectory given the trajectory and a goal.

    Args:
        trajectory: jnp.array A single trajectory, shape (T, dx + du).
        goal: jnp.array The goal state, shape (dx + du).
    Returns:
        Tuple
    """
    x: jnp.array = trajectory[:, :12]
    u: jnp.array = trajectory[:, 12:]
    T = x.shape[0]

    # Build Q, P, and R as given in equations (54), (55), and (56).
    Q: jnp.array = jnp.diag(jnp.array([5, 5, 0.5, 2.5, 2.5, 0.025, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]))
    P: jnp.array = 2*Q
    R: jnp.array = jnp.diag(jnp.array([1, 16, 16, 16]))

    d2goal = x - goal.reshape(-1, 12)

    # Compute cost terms according to equation (53).
    running_state_cost = d2goal.reshape(-1, 1, 12) @ Q.reshape(1, 12, 12) @ d2goal.reshape(-1, 12, 1)
    running_control_cost = u.reshape(-1, 1, 4) @ R.reshape(1, 4, 4) @ u.reshape(-1, 4, 1)
    terminal_state_cost = d2goal[-1].reshape(1, 12) @ P @ d2goal[-1].reshape(12, 1)
    cost = jnp.sum(running_control_cost + running_state_cost, axis=0) + terminal_state_cost

    # Compute cost gradient analytically.
    grad_running_state_cost = Q.reshape(1, 12, 12) @ d2goal.reshape(-1, 12, 1)
    grad_running_control_cost = R.reshape(1, 4, 4) @ u.reshape(-1, 4, 1)
    grad_terminal_cost = P @ d2goal[-1].reshape(12, 1)  # only on terminal
    grad_terminal_cost = jnp.concatenate((torch.zeros(T - 1, 12, 1, device=trajectory.device),
                                         grad_terminal_cost.unsqueeze(0)), dim=0)

    grad_cost = torch.cat((grad_running_state_cost + grad_terminal_cost, grad_running_control_cost), dim=1)
    grad_cost = grad_cost.reshape(T * 16)

    # Compute cost hessian analytically.
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

    def __init__(self, start, goal, T, device='cuda:0', alpha=1, include_obstacle=False,
                 gp_sdf_model=None, use_squared_slack=True, compute_hessian=True):
        super().__init__(start, goal, T, device)
        self.T = T
        self.dx = 12
        self.du = 4
        if include_obstacle:
            self.dz = 1
        else:
            self.dz = 0
        self.squared_slack = use_squared_slack
        self.dh = self.dz * T
        self.dg = 12 * T + T - 1
        self.alpha = alpha
        self.alpha = 0.1
        self.include_obstacle = include_obstacle
        # self.dg = 2 * T - 1
        data = np.load('surface_data.npz')
        # GP which models surface
        self.compute_hessian = compute_hessian
        self.surface_gp = GPSurfaceModel(torch.from_numpy(data['xy']).to(dtype=torch.float32, device=device),
                                         torch.from_numpy(data['z']).to(dtype=torch.float32, device=device))

        # GP which models sdf
        if gp_sdf_model is not None:
            gp_sdf_model.train_y = torch.where(gp_sdf_model.train_y > 0, gp_sdf_model.train_y * 2,
                                               gp_sdf_model.train_y)

            self.obs_gp = GPSurfaceModel(gp_sdf_model.train_x.to(device=device),
                                         gp_sdf_model.train_y.to(device=device))

        else:
            self.obs_gp = None

        # gradient and hessian of dynamics
        self._dynamics = Quadrotor12DDynamics(dt=0.1)
        self.dynamics_constraint = vmap(self._dynamics_constraint)
        self.grad_dynamics = vmap(jacrev(self._dynamics_constraint))
        self.hess_dynamics = vmap(hessian(self._dynamics_constraint))

        self.height_constraint = vmap(self._height_constraint)
        self.sdf_constraint = vmap(self._sdf_constraint)
        self._objective = vmap(partial(cost, goal=goal))

        self.start = start
        self.goal = goal

        kernel = structured_rbf_kernel
        self.K = kernel
        self.dK = jacrev(kernel, argnums=0)
        self.x_max = torch.ones(self.dx + self.du)
        self.x_max[:3] = 6
        self.x_max[3:5] = 0.4 * torch.pi
        self.x_max[5] = 1000
        self.x_max[6:12] = 100
        self.x_max[12:] = 100
        self.x_max = self.x_max.to(self.device)
        self.x_min = -self.x_max

        self.obstacle_centre = torch.tensor([0.0, 0.0], device=self.device)
        self.obstacle_rad = 1.05

    def _objective(self, x):
        """
        Args:
        Returns:
        """
        J, dJ, HJ = cost(x, self.goal)
        return J * self.alpha, dJ * self.alpha, HJ * self.alpha

    def dynamics(self, x, u):
        """
        Args:
        Returns:
        """
        return self._dynamics(x, u)

    def _dynamics_constraint(self, trajectory):
        x = trajectory[:, :12]
        u = trajectory[:-1, 12:]
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

    def _con_eq(self, trajectory, compute_grads=True, compute_hess=True):
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
        # g = torch.cat((dynamics_constr, start_constraint), dim=1)

        if torch.any(torch.isinf(surf_constr)):
            print('inf in surface')
        if torch.any(torch.isinf(dynamics_constr)):
            print('inf in dynamics')

        if not compute_grads:
            return g, None, None
        # currently only take derivative of height at time t wrt state at time t - need to include other times
        # (all derivatives and hessians will be zero)

        grad_dynamics_constr = self.grad_dynamics(trajectory).reshape(N, -1, prob_dim)
        grad_surf_constr = torch.diag_embed(grad_surf_constr.permute(0, 2, 1))
        grad_surf_constr = grad_surf_constr.permute(0, 2, 3, 1).reshape(N, self.T, prob_dim)
        grad_start_constraint = torch.zeros(N, 12, self.T, 16, device=trajectory.device)
        grad_start_constraint[:, :12, 0, :12] = torch.eye(12, device=trajectory.device)
        grad_start_constraint = grad_start_constraint.reshape(N, 12, -1)
        Dg = torch.cat((grad_dynamics_constr, grad_start_constraint, grad_surf_constr[:, 1:]), dim=1)

        if not compute_hess:
            return g, Dg, None

        hess_dynamics_constr = self.hess_dynamics(trajectory).reshape(N, -1, prob_dim, prob_dim)
        hess_surf_constr = torch.diag_embed(torch.diag_embed(hess_surf_constr.permute(0, 2, 3, 1)))
        hess_surf_constr = hess_surf_constr.permute(0, 3, 4, 1, 5, 2).reshape(N, self.T, prob_dim, prob_dim)
        hess_start_constraint = torch.zeros(N, 12, self.T * 16, self.T * 16, device=trajectory.device)

        DDg = torch.cat((hess_dynamics_constr, hess_start_constraint, hess_surf_constr[:, 1:]), dim=1)

        if torch.isinf(g).any():
            print('inf in g')
            print(torch.where(torch.isinf(g)))
        if torch.isinf(Dg).any():
            print('inf in Dg')
            print(torch.where(torch.isinf(Dg)))
        if torch.isnan(g).any():
            print('nan in g')
            print(torch.where(torch.isnan(g)))
        if torch.isnan(Dg).any():
            print('nan in Dg')
            print(torch.where(torch.isnan(Dg)))

        return g, Dg, DDg

    def _sdf_constraint(self, trajectory):
        T = trajectory.shape[0]
        xy, z = trajectory[:, :2], trajectory[:, 2]

        # compute z of surface and gradient and hessian
        sdf, grad_sdf, hess_sdf = self.obs_gp.posterior_mean(xy)
        m = 1
        constr = m*sdf + 0.05
        grad_constr = torch.cat((m*grad_sdf,
                                 torch.zeros(T, 14, device=trajectory.device)), dim=1)
        hess_constr = torch.cat((
            torch.cat((m*hess_sdf, torch.zeros(T, 2, 14, device=trajectory.device)), dim=2),
            torch.zeros(T, 14, 16, device=trajectory.device)), dim=1)

        #print(torch.clamp(sdf, min=0).mean(), sdf.max())
        return constr, grad_constr, hess_constr

    def _obs_sdf(self, trajectory, compute_grads=True, compute_hess=True):
        N = trajectory.shape[0]
        prob_dim = self.T * (self.dx + self.du)

        # surface constraint
        sdf_constr, grad_sdf_constr, hess_sdf_constr = self.sdf_constraint(trajectory)

        if not compute_grads:
            return sdf_constr, None, None

        # currently only take derivative of height at time t wrt state at time t - need to include other times
        # (all derivatives and hessians will be zero)
        grad_sdf_constr = torch.diag_embed(grad_sdf_constr.permute(0, 2, 1))
        grad_sdf_constr = grad_sdf_constr.permute(0, 2, 3, 1).reshape(N, self.T, prob_dim)

        if not compute_hess:
            return sdf_constr, grad_sdf_constr, None

        hess_sdf_constr = torch.diag_embed(torch.diag_embed(hess_sdf_constr.permute(0, 2, 3, 1)))
        hess_sdf_constr = hess_sdf_constr.permute(0, 3, 4, 1, 5, 2).reshape(N, self.T, prob_dim, prob_dim)

        return sdf_constr, grad_sdf_constr, hess_sdf_constr

    def _obs_disc(self, trajectory, compute_grads=True, compute_hess=True):
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

        if not compute_hess:
            return h, grad_h, None

        hess_h_xy = -torch.eye(2, device=self.device).reshape(1, 1, 2, 2).repeat(N, self.T, 1, 1)
        # need to make (N, T, T, 2, T, 2)
        hess_h_xy = torch.diag_embed(torch.diag_embed(hess_h_xy.permute(0, 2, 3, 1)))  # now (N, 2, 2, T, T, T)
        hess_h_xy = hess_h_xy.permute(0, 3, 4, 1, 5, 2)  # now (N, T, T, 2, T, 2)
        hess_h = torch.zeros(N, self.T, self.T, self.dx + self.du, self.T, self.dx + self.du, device=self.device)
        hess_h[:, :, :, :2, :, :2] = hess_h_xy
        hess_h = hess_h.reshape(N, self.T, self.T * (self.dx + self.du),
                                self.T * (self.dx + self.du))

        return h, grad_h, hess_h

    def _con_ineq(self, trajectory, compute_grads=True, compute_hess=True):
        if not self.include_obstacle:
            return None, None, None

        # If no gp we assume disc obstacle
        if self.obs_gp is None:
            return self._obs_disc(trajectory, compute_grads=compute_grads, compute_hess=compute_hess)

        # Otherwise we use the GP sdf function
        return self._obs_sdf(trajectory, compute_grads=compute_grads, compute_hess=compute_hess)

    def eval(self, augmented_trajectory):
        N = augmented_trajectory.shape[0]
        trajectory = augmented_trajectory.reshape(N, self.T, -1)[:, :, :self.dx + self.du]

        cost, grad_cost, hess_cost = self._objective(trajectory)

        grad_cost = torch.cat((grad_cost.reshape(N, self.T, -1),
                               torch.zeros(N, self.T, self.dz, device=trajectory.device)
                               ), dim=2).reshape(N, -1)

        # compute kernel and grad kernel
        Xk = trajectory  # .reshape(N, -1)
        K = self.K(Xk, Xk)
        grad_K = -self.dK(Xk, Xk).reshape(N, N, N, -1)
        grad_K = torch.einsum('nmmi->nmi', grad_K)
        grad_K = torch.cat((grad_K.reshape(N, N, self.T, self.dx + self.du),
                            torch.zeros(N, N, self.T, self.dz, device=trajectory.device)), dim=-1)
        grad_K = grad_K.reshape(N, N, -1)

        # Now we need to compute constraints and their first and second partial derivatives
        g, Dg, DDg = self.combined_constraints(augmented_trajectory,
                                               compute_grads=True,
                                               compute_hess=self.compute_hessian)

        hess_cost = None

        if DDg is not None:
            DDg.detach_()

        # print(cost.mean(), g.abs().mean(), g.abs().max())
        return grad_cost.detach(), hess_cost, K.detach(), grad_K.detach(), g.detach(), Dg.detach(), DDg

    def update(self, start, goal=None, T=None, obstacle_pos=None):
        self.start = start
        if goal is not None:
            self.goal = goal
            self._objective = vmap(partial(cost, goal=goal))
        if T is not None:
            self.T = T
            self.dh = self.dz * T
            self.dg = 12 * T + T - 1
        if obstacle_pos is not None:
            self.obstacle_centre = torch.from_numpy(obstacle_pos).to(device=self.device, dtype=torch.float32)

    def get_initial_xu(self, N):
        x = [self.start.repeat(N, 1)]
        u = torch.randn(N, self.T, self.du, device=self.device)
        u[:, :, 1:] *= 0.25
        for t in range(self.T - 1):
            x.append(self.dynamics(x[-1], u[:, t]))

        return torch.cat((torch.stack(x, dim=1), u), dim=2)

    def shift(self, xu):
        """
            Performs shift operation
        """
        N, T, _ = xu.shape
        u = torch.randn(N, self.du, device=self.device)
        u[:, 1:] *= 0.25
        #u[:, 0] *= 0.5
        next_x = self.dynamics(xu[:, -1, :self.dx], u)
        xu = torch.roll(xu, -1, dims=1)
        xu[:, -2, self.dx:] = u
        xu[:, -1, :self.dx] = next_x
        # this is a dummy var that makes no difference
        xu[:, -1, self.dx] = 0
        return xu
