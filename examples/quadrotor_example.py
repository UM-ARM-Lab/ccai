import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!
import time
from functools import partial
from functorch import vmap, jacrev, hessian

# ccai stuff
from ccai.kernels import rbf_kernel, structured_rbf_kernel
from ccai.quadrotor_env import QuadrotorEnv
from ccai.quadrotor import Quadrotor12DDynamics
from ccai.gp import GPSurfaceModel
from ccai.mpc import Constrained_SVGD_MPC, IpoptMPC, MPPI, SVMPC, SQPMPC, iCEM
from ccai.problem import ConstrainedSVGDProblem, IpoptProblem, UnconstrainedPenaltyProblem, NLOptProblem

import yaml

import pathlib

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]


def cost(trajectory, goal):
    x = trajectory[:, :12]
    u = trajectory[:, 12:]
    T = x.shape[0]
    Q = torch.eye(12, device=trajectory.device)
    Q[5, 5] = 1e-2
    Q[2, 2] = 0.1
    Q[3:, 3:] *= 0.5
    #Q[6:9] *= 2.0
    Q *= 5
    P = Q
    R = 16 * torch.eye(4, device=trajectory.device)
    R[0, 0] = 1

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
        # self.g = vmap(self._combined_contraints)
        # self.grad_g = vmap(jacrev(self._combined_contraints))
        # self.hess_g = vmap(hessian(self._combined_contraints))

        kernel = structured_rbf_kernel
        # kernel = rbf_kernel
        self.K = kernel
        self.dK = jacrev(kernel, argnums=0)
        self.x_max = torch.ones(self.dx + self.du)
        self.x_max[:3] = 6
        self.x_max[3:5] = 0.4 * torch.pi
        self.x_max[5] = 1000  # torch.pi
        self.x_max[6:12] = 100
        # self.x_max[6:9] = 100
        self.x_max[12:] = 100
        # self.x_max = 1000 * torch.ones(self.dx + self.du)
        self.x_max = self.x_max.to(self.device)
        self.x_min = -self.x_max

        self.obstacle_centre = torch.tensor([0.0, 0.0], device=self.device)
        self.obstacle_rad = 1.05

    def _objective(self, x):
        J, dJ, HJ = cost(x, self.goal)
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

class QuadrotorIpoptProblem(QuadrotorProblem, IpoptProblem):

    def __init__(self, start, goal, T, include_obstacle=False, gp_sdf_model=None):
        super().__init__(start, goal, T, device='cpu', include_obstacle=include_obstacle, gp_sdf_model=gp_sdf_model)


class QuadrotorSQPProblem(QuadrotorProblem, NLOptProblem):
    def __init__(self, start, goal, T, include_obstacle=False, gp_sdf_model=None):
        super().__init__(start, goal, T, device='cpu', include_obstacle=include_obstacle, gp_sdf_model=gp_sdf_model)


class QuadrotorUnconstrainedProblem(QuadrotorProblem, UnconstrainedPenaltyProblem):
    def __init__(self, start, goal, T, device, penalty, include_obstacle=False, gp_sdf_model=None):
        super().__init__(start, goal, T, device=device, include_obstacle=include_obstacle, gp_sdf_model=gp_sdf_model)
        self.penalty = penalty


def update_plot_with_trajectories(ax, traj_lines, best, trajectory):
    for line in ax.get_lines():
        line.remove()
    if traj_lines is None:
        traj_lines = []
        for traj in trajectory:
            traj_np = traj.detach().cpu().numpy()
            traj_lines.extend(ax.plot(traj_np[1:, 0],
                                      traj_np[1:, 1],
                                      traj_np[1:, 2], color='g', alpha=0.5, linestyle='--'))

        traj_np = best.detach().cpu().numpy()
        traj_lines.extend(ax.plot(traj_np[1:, 0], traj_np[1:, 1], traj_np[1:, 2], color='g'))
    else:
        for traj, traj_line in zip(trajectory, traj_lines[:-1]):
            traj_np = traj.detach().cpu().numpy()
            traj_line.set_xdata(traj_np[1:, 0])
            traj_line.set_ydata(traj_np[1:, 1])
            traj_line.set_3d_properties(traj_np[1:, 2])

        traj_np = best.detach().cpu().numpy()
        traj_lines[-1].set_xdata(traj_np[1:, 0])
        traj_lines[-1].set_ydata(traj_np[1:, 1])
        traj_lines[-1].set_3d_properties(traj_np[1:, 2])


def do_trial(env, params, fpath):
    if params['visualize']:
        env.render_init()

    start = torch.from_numpy(env.state).to(dtype=torch.float32, device=params['device'])
    goal = torch.zeros(12, device=params['device'])
    goal[:2] = 4

    include_obstacle = True if params['obstacle_mode'] is not None else False
    sdf_model = env.obstacle_model if params['obstacle_mode'] == 'gp' else None

    if 'csvgd' in params['controller']:
        problem = QuadrotorProblem(start, goal, params['T'], device=params['device'], include_obstacle=include_obstacle,
                                   gp_sdf_model=sdf_model, use_squared_slack=params['squared_slack'],
                                   compute_hessian=params['use_true_hess'])
        controller = Constrained_SVGD_MPC(problem, params)
    elif 'ipopt' in params['controller']:
        problem = QuadrotorIpoptProblem(start, goal, params['T'], include_obstacle=include_obstacle,
                                        gp_sdf_model=sdf_model)
        controller = IpoptMPC(problem, params)
    elif 'sqp' in params['controller']:
        problem = QuadrotorSQPProblem(start, goal, params['T'], include_obstacle=include_obstacle,
                                      gp_sdf_model=sdf_model)
        controller = SQPMPC(problem, params)
    elif 'mppi' in params['controller']:
        problem = QuadrotorUnconstrainedProblem(start, goal, params['T'],
                                                device=params['device'], penalty=params['penalty'],
                                                include_obstacle=include_obstacle,
                                                gp_sdf_model=sdf_model)
        controller = MPPI(problem, params)
    elif 'svgd' in params['controller']:
        problem = QuadrotorUnconstrainedProblem(start, goal, params['T'],
                                                device=params['device'], penalty=params['penalty'],
                                                include_obstacle=include_obstacle,
                                                gp_sdf_model=sdf_model)
        controller = SVMPC(problem, params)
    elif 'icem' in params['controller']:
        problem = QuadrotorUnconstrainedProblem(start, goal, params['T'],
                                                device=params['device'], penalty=params['penalty'],
                                                include_obstacle=include_obstacle,
                                                gp_sdf_model=sdf_model)
        controller = iCEM(problem, params)
    else:
        raise ValueError('Invalid controller')
    # plt.axis('off')
    ax = env.ax
    traj_lines = None
    collision = False
    actual_traj = [env.state]
    plans = []
    all_violation = []

    duration = 0.0
    for step in range(params['num_steps']):
        if not collision:
            start = torch.from_numpy(env.state).to(dtype=torch.float32, device=params['device'])

            #if step > 0:
            torch.cuda.synchronize()
            start_time = time.time()
            best_traj, trajectories = controller.step(start, obstacle_pos=env.obstacle_pos)

            plans.append(trajectories.detach().cpu().numpy())

            if step > 0:
                torch.cuda.synchronize()
                duration += time.time() - start_time
            else:
                torch.cuda.synchronize()
                print('Time to initial plan', time.time() - start_time)

            u = best_traj[0, -4:].detach().cpu().numpy()

            if torch.any(torch.isnan(best_traj)):
                collision = True
            else:
                _, violation = env.step(u)

            # print(violation)
            if include_obstacle:
                if violation['obstacle'] > 0:
                    collision = True

        actual_traj.append(env.state)
        all_violation.append(violation)

        if params['visualize']:
            env.render_update()
            update_plot_with_trajectories(ax, traj_lines, best_traj, trajectories)
            plt.savefig(f'{fpath.resolve()}/im_{step:02d}.png')
            plt.gcf().canvas.flush_events()

    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')
    actual_traj = np.stack(actual_traj)
    all_violation = np.stack(all_violation)
    plans = np.stack(plans)
    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_traj,
             constr=all_violation,
             plans=plans, goal=goal.cpu().numpy())

    if params['visualize']:
        plt.close()

    return np.min(np.linalg.norm(actual_traj[:, :2] - np.array([[4, 4]]), axis=1))

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='quadrotor_no_obs.yaml')
    parser.add_argument('--load-starts', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/config/{args.config}').read_text())
    from tqdm import tqdm

    if config['obstacle_mode'] == 'none':
        config['obstacle_mode'] = None

    results = {}
    for i in tqdm(range(config['num_trials'])):
        env = QuadrotorEnv(False, 'surface_data.npz', obstacle_mode=config['obstacle_mode'],
                           obstacle_data_fname='obstacle_data_20.npz')
        env.reset()

        if args.load_starts:
            data = dict(np.load(f'../data/experiments/{config["experiment_name"]}/csvgd/trial_{i+1}/trajectory.npz',
                            allow_pickle=True))
            x = data['x']
            env.state = x[0]

        start_state = env.state.copy()

        for controller in config['controllers'].keys():
            env.reset()
            env.state = start_state
            print(env.state[:3])
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)

            # set up params
            params = config.copy()
            params.pop('controllers')
            params.update(config['controllers'][controller])
            params['controller'] = controller
            final_distance_to_goal = do_trial(env, params, fpath)

            if controller not in results.keys():
                results[controller] = [final_distance_to_goal]
            else:
                results[controller].append(final_distance_to_goal)

        print(results)
