import numpy as np
from isaacgym.torch_utils import quat_apply, quat_mul
from isaac_victor_envs.tasks.victor import VictorPuckObstacleEnv, orientation_error

import torch
import time
import yaml
import pathlib
from functools import partial
from functorch import vmap, jacrev, hessian

from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
from ccai.kernels import rbf_kernel

from ccai.problem import ConstrainedSVGDProblem, UnconstrainedPenaltyProblem, IpoptProblem
from ccai.mpc.csvgd import Constrained_SVGD_MPC
from ccai.mpc.mppi import MPPI
from ccai.mpc.svgd import SVMPC
from ccai.mpc.ipopt import IpoptMPC

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]


class VictorTableProblem(ConstrainedSVGDProblem):

    def __init__(self, start, goal, T, device='cuda:0'):
        super().__init__(start, goal, T, device)
        self.dz = 2
        self.dh = self.dz * T
        self.dg = 2 * T + 2
        self.dx = 7
        self.du = 0
        self.dt = 0.1
        self.T = T
        self.start = start
        self.goal = goal
        self.K = rbf_kernel
        self.grad_kernel = jacrev(rbf_kernel, argnums=0)
        self.alpha = 1000

        self.h = vmap(partial(inequality_constraint, start=self.start))
        self.grad_h = vmap(jacrev(partial(inequality_constraint, start=self.start)))
        self.hess_h = vmap(hessian(partial(inequality_constraint, start=self.start)))

        self.g = vmap(partial(equality_constraints, goal=self.goal))
        self.grad_g = vmap(jacrev(partial(equality_constraints, goal=self.goal)))
        self.hess_g = vmap(hessian(partial(equality_constraints, goal=self.goal)))

        self.x_max = torch.ones(self.dx)
        self.x_min = -self.x_max
        self.x_max[:2] = torch.tensor([1.0, 0.775])
        self.x_min[:2] = torch.tensor([0.4, -0.2])
        self.x_max[2] = 0.80
        self.x_min[2] = 0.75

        self.dynamics_constraint = vmap(self._dynamics_constraint)
        self.grad_dynamics_constraint = vmap(jacrev(self._dynamics_constraint))
        self.hess_dynamics_constraint = vmap(hessian(self._dynamics_constraint))

        self.cost = vmap(partial(cost, start=self.start))
        self.grad_cost = vmap(jacrev(partial(cost, start=self.start)))
        self.hess_cost = vmap(hessian(partial(cost, start=self.start)))

    def dynamics(self, x, u):
        N = x.shape[0]
        # dynamics will be delta_pose
        p = x[:, :3]
        q = x[:, 3:]  # quaternion

        v = u[:, :3]  # linear velocity
        omega = u[:, 3:]  # angular velocity

        # quaternion representation of omega
        omega_q = torch.cat((omega, torch.zeros(N, 1, device=self.device)), dim=1)
        _omega = 0.5 * self.dt * omega
        l = torch.linalg.norm(_omega, dim=1, keepdim=True)
        delta_q = torch.where(l > 1e-7,
                              torch.cat((_omega * torch.sin(l) / l, torch.cos(l).reshape(-1, 1)), dim=1),
                              torch.cat((_omega, torch.ones(N, 1, device=x.device)), dim=1)
                              )
        q_new = quat_mul(delta_q, q)
        # Use first order approximation to predict next q
        # q_new = 0.5 * self.dt * quat_mul(omega_q, q) + q
        # q_new = q_new / torch.linalg.norm(q_new, dim=1, keepdim=True)

        # Euler integrate to get next p
        p_new = p + v * self.dt
        return torch.cat((p_new, q_new), dim=1)

    def _dynamics_constraint(self, trajectory):
        x = trajectory[:, :self.dx]
        u = trajectory[:, self.dx:]
        current_x = torch.cat((self.start.reshape(1, self.dx), x[:-1]), dim=0)
        next_x = x
        pred_next_x = self.dynamics(current_x, u)
        return torch.reshape(pred_next_x - next_x, (-1,))

    def _objective(self, x):
        # x = x[:, :, :self.dx]

        # J, grad_J, hess_J = self.cost(x)
        J, grad_J, hess_J = self.cost(x), self.grad_cost(x), self.hess_cost(x)
        N = x.shape[0]
        return (self.alpha * J.reshape(N),
                self.alpha * grad_J.reshape(N, -1),
                self.alpha * hess_J.reshape(N, self.T * (self.dx + self.du), self.T * (self.dx + self.du)))

    def _con_eq(self, x, compute_grads=True):
        # g_dynamics = self.dynamics_constraint(x)
        # g = torch.cat((self.g(x), g_dynamics), dim=1)
        g = self.g(x)
        N = x.shape[0]
        if not compute_grads:
            return g, None, None

        return g, self.grad_g(x), self.hess_g(x)
        # grad_g = torch.cat((self.grad_g(x), self.grad_dynamics_constraint(x)), dim=1)
        # hess_g = torch.cat((self.hess_g(x), self.hess_dynamics_constraint(x)), dim=1)

        # return (g.reshape(N, -1), grad_g.reshape(N, -1, self.T * (self.dx + self.du)),
        #        hess_g.reshape(N, -1, self.T * (self.dx + self.du), self.T * (self.dx + self.du)))
        # return self.g(x), self.grad_g(x), self.hess_g(x)

    def _con_ineq(self, x, compute_grads=True):
        if not compute_grads:
            return self.h(x), None, None
        return self.h(x), self.grad_h(x), self.hess_h(x)

    def eval(self, augmented_trajectory):
        N = augmented_trajectory.shape[0]
        augmented_trajectory = augmented_trajectory.clone().reshape(N, self.T, -1)
        x = augmented_trajectory[:, :, :self.dx + self.du]

        J, grad_J, hess_J = self._objective(x)
        # hess_J = None
        grad_J = torch.cat((grad_J.reshape(N, self.T, -1),
                            torch.zeros(N, self.T, self.dz, device=x.device)), dim=2).reshape(N, -1)

        Xk = x.reshape(N, -1)
        K = self.K(Xk, Xk, hess_J.mean(dim=0))
        grad_K = -self.grad_kernel(Xk, Xk, hess_J.mean(dim=0))
        grad_K = torch.einsum('nmmi->nmi', grad_K)
        grad_K = torch.cat((grad_K.reshape(N, N, self.T, self.dx + self.du),
                            torch.zeros(N, N, self.T, self.dz, device=x.device)), dim=-1)
        grad_K = grad_K.reshape(N, N, -1)
        K = torch.eye(N, N, device=K.device)
        G, dG, hessG = self.combined_constraints(augmented_trajectory)

        if hess_J is not None:
            hess_J_ext = torch.zeros(N, self.T, self.dx + self.du + self.dz, self.T, self.dx + self.du + self.dz,
                                     device=x.device)
            hess_J_ext[:, :, :self.dx + self.du, :, :self.dx + self.du] = hess_J.reshape(N, self.T, self.dx + self.du,
                                                                                         self.T, self.dx + self.du)
            hess_J = hess_J_ext.reshape(N, self.T * (self.dx + self.du + self.dz),
                                        self.T * (self.dx + self.du + self.dz))

        ## augment the objective to make the hessian scale better
        ## no lagrange multiplier,  TODO add g^2(x) to objective
        grad_J_augmented = grad_J + 2 * (G.reshape(N, 1, -1) @ dG).reshape(N, -1)
        hess_J_augmented = hess_J + 2 * (torch.sum(G.reshape(N, -1, 1, 1) * hessG, dim=1) + dG.permute(0, 2, 1) @ dG)
        grad_J = grad_J_augmented
        hess_J = hess_J_augmented.mean(dim=0)
        # print(J, G.abs().max())
        return grad_J.detach(), hess_J.detach(), K.detach(), grad_K.detach(), G.detach(), dG.detach(), hessG.detach()

    def update(self, start, goal=None, T=None):
        self.start = start

        # update functions that require start
        self.cost = vmap(partial(cost, start=self.start))
        self.grad_cost = vmap(jacrev(partial(cost, start=self.start)))
        self.hess_cost = vmap(hessian(partial(cost, start=self.start)))

        self.h = vmap(partial(inequality_constraint, start=self.start))
        self.grad_h = vmap(jacrev(partial(inequality_constraint, start=self.start)))
        self.hess_h = vmap(hessian(partial(inequality_constraint, start=self.start)))

        if goal is not None:
            self.goal = goal
            self.g = vmap(partial(equality_constraints, goal=self.goal))
            self.grad_g = vmap(jacrev(partial(equality_constraints, goal=self.goal)))
            self.hess_g = vmap(hessian(partial(equality_constraints, goal=self.goal)))

        if T is not None:
            self.T = T
            self.dh = self.dz * T
            self.dg = 2 * T + 2

    def get_initial_xu(self, N):

        u = torch.randn(N, self.T, 6, device=self.device)
        x = [self.start.reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            x.append(self.dynamics(x[-1], u[:, t]))

        # particles = torch.cumsum(particles, dim=1) + self.start.reshape(1, 1, self.dx)
        x = torch.stack(x[1:], dim=1)
        xu = torch.cat((x, u), dim=2)
        return x


class VictorTableIpoptProblem(VictorTableProblem, IpoptProblem):

    def __init__(self, start, goal, T):
        super().__init__(start, goal, T, device='cpu')


class VictorTableUnconstrainedProblem(VictorTableProblem, UnconstrainedPenaltyProblem):

    def __init__(self, start, goal, T, device, penalty):
        super().__init__(start, goal, T, device=device)
        self.penalty = penalty
        self.dt = 0.1
        self.du = 6

        self.x_min = torch.cat((self.x_min, -torch.ones(6)))
        self.x_max = torch.cat((self.x_max, torch.ones(6)))


def cost(x, start):
    x = torch.cat((start.reshape(1, 7), x[:, :7]), dim=0)
    T, _ = x.shape
    position = x[:, :3]
    orientation = x[:, 3:]

    p0 = position[:-1]
    q0 = orientation[:-1]
    p1 = position[1:]
    q1 = orientation[1:]
    q_diff = orientation_error(q1.reshape(-1, 4), q0.reshape(-1, 4)).reshape(T - 1, 3)
    p_diff = p1 - p0
    diff = torch.stack((p_diff, 0.1 * q_diff), dim=1)
    return torch.sum(diff ** 2)


def goal_constraint(x, goal):
    return x[-1, :2] - goal


#### EQUALITY CONSTRAINTS
def table_constraint(x, table_height=0.8):
    # will be N * T constraints
    T, _ = x.shape
    z = x[:, 2]
    return (z - table_height).reshape(-1)


def valid_quat_constraint(x):
    # N constraints
    T, _ = x.shape
    q = x[:, 3:7]
    # valid quaternion -- must have unit norm
    return torch.sum(q ** 2, dim=-1).reshape(-1) - 1


def pose_constraint(x):
    # be flat against table (SE2 constraint)
    T, _ = x.shape
    # will be N * T constraints
    q = x[:, 3:7].reshape(-1, 4)
    z = torch.tensor([0.0, 0.0, 1.0], device=x.device).reshape(1, 3).repeat(T, 1)
    # q should transform [0, 0, 1] to [0, 0, -1]
    # dot product should be -1

    return 1 + torch.bmm(quat_apply(q, z).unsqueeze(1), z.unsqueeze(2)).reshape(-1)


def obstacle_constraint(x):
    T, _ = x.shape
    xy = x[:, :2]

    centre1 = torch.tensor([0.6, 0.35], device=x.device).reshape(1, 2)
    centre2 = torch.tensor([0.775, 0.45], device=x.device).reshape(1, 2)

    constr1 = 0.12 ** 2 - torch.sum((xy - centre1) ** 2, dim=1)
    constr2 = 0.12 ** 2 - torch.sum((xy - centre2) ** 2, dim=1)
    # return constr1
    return torch.cat((constr1.reshape(-1), constr2.reshape(-1)), dim=0)


def equality_constraints(x, goal):
    return torch.cat((
        # table_constraint(x, 0.805),
        valid_quat_constraint(x),
        pose_constraint(x),
        goal_constraint(x, goal)
    ), dim=0).reshape(-1)


def inequality_constraint(x, start):
    return torch.cat((
        obstacle_constraint(x),
    ), dim=0).reshape(-1)


def do_trial(env, params, fpath):
    state = env.get_state()
    ee_pos, ee_ori = state['ee_pos'], state['ee_ori']
    start = torch.cat((ee_pos, ee_ori), dim=-1).reshape(7).to(device=params['device'])

    if params['controller'] == 'csvgd':
        problem = VictorTableProblem(start, params['goal'], params['T'], device=params['device'])
        controller = Constrained_SVGD_MPC(problem, params)
    elif params['controller'] == 'ipopt':
        problem = VictorTableIpoptProblem(start, params['goal'], params['T'])
        controller = IpoptMPC(problem, params)
    elif 'svgd' in params['controller']:
        problem = VictorTableUnconstrainedProblem(start, params['goal'], params['T'], device=params['device'],
                                                  penalty=params['penalty'])
        controller = SVMPC(problem, params)
    elif 'mppi' in params['controller']:
        problem = VictorTableUnconstrainedProblem(start, params['goal'], params['T'], device=params['device'],
                                                  penalty=params['penalty'])
        controller = MPPI(problem, params)
    else:
        raise ValueError('Invalid controller')

    actual_trajectory = []
    commanded_trajectory = []
    for k in range(params['num_steps']):
        state = env.get_state()
        ee_pos, ee_ori = state['ee_pos'], state['ee_ori']
        start = torch.cat((ee_pos, ee_ori), dim=-1).reshape(7).to(device=params['device'])
        actual_trajectory.append(start)
        best_traj, trajectories = controller.step(start)

        x = best_traj[0, :7]
        commanded_trajectory.append(x)
        # add goal lines to sim
        line_vertices = np.array([
            [goal[0].item() - 0.025, goal[1].item() - 0.025, 0.803],
            [goal[0].item() + 0.025, goal[1].item() + 0.025, 0.803],
            [goal[0].item() - 0.025, goal[1].item() + 0.025, 0.803],
            [goal[0].item() + 0.025, goal[1].item() - 0.025, 0.803],
        ], dtype=np.float32)

        line_colors = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ], dtype=np.float32)

        for e in env.envs:
            gym.add_lines(viewer, e, 2, line_vertices, line_colors)

        # add trajectory lines to sim
        # trajectory_colors
        # traj_line_colors = np.array([[0.5, 0., 0.5]*M], dtype=np.float32)
        M = len(trajectories)
        if M > 0:
            traj_line_colors = np.random.random((1, M)).astype(np.float32)

            for e in env.envs:
                p = torch.stack((start[:3].reshape(1, 3).repeat(M, 1),
                                 trajectories[:, 0, :3]), dim=1).reshape(2 * M, 3).cpu().numpy()
                p[:, 2] += 0.005
                gym.add_lines(viewer, e, M, p, traj_line_colors)
                T = trajectories.shape[1]
                for t in range(T - 1):
                    p = torch.stack((trajectories[:, t, :3], trajectories[:, t + 1, :3]), dim=1).reshape(2 * M, 3)
                    p = p.cpu().numpy()
                    p[:, 2] += 0.005
                    gym.add_lines(viewer, e, M, p, traj_line_colors)
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, False)
                gym.sync_frame_time(sim)

        env.step(x.reshape(1, 7).to(device=env.device))

        gym.clear_lines(viewer)

    state = env.get_state()
    ee_pos, ee_ori = state['ee_pos'], state['ee_ori']
    obs_move = torch.linalg.norm(state['obs1_pos']) + torch.linalg.norm(state['obs2_pos'])
    state = torch.cat((ee_pos, ee_ori), dim=-1).reshape(7).to(device=params['device'])
    actual_trajectory.append(state)
    commanded_trajectory = torch.stack(commanded_trajectory, dim=0).reshape(-1, 7)

    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 7)
    constraint_val = problem.g(actual_trajectory.unsqueeze(0)).squeeze(0)

    # check if obstacles moves
    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
             constr=constraint_val.cpu().numpy(),
             x_des=commanded_trajectory.cpu().numpy(),
             obs_move=obs_move.cpu().numpy())

    return np.linalg.norm(state[:2].cpu().numpy() - goal.cpu().numpy())


if __name__ == "__main__":
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/victor_table.yaml').read_text())

    from tqdm import tqdm

    # instantiate environment
    env = VictorPuckObstacleEnv(1)
    sim, gym, viewer = env.get_sim()
    """
    state = env.get_state()
    ee_pos, ee_ori = state['ee_pos'], state['ee_ori']
    try:
        while True:
            start = torch.cat((ee_pos, ee_ori), dim=-1).reshape(1, 7)
            env.step(start)
            print('waiting for you to finish camera adjustment, ctrl-c when done')
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    """
    results = {}

    for i in tqdm(range(config['num_trials'])):
        goal = torch.tensor([0.6, 0.6])
        goal = goal + torch.tensor([0.25, 0.1]) * torch.rand(2)
        for controller in config['controllers'].keys():
            env.reset()
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/victor_table/{controller}/trial_{i + 1}')
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
            # set up params
            params = config.copy()
            params.pop('controllers')
            params.update(config['controllers'][controller])
            params['controller'] = controller
            params['goal'] = goal.to(device=params['device'])
            final_distance_to_goal = do_trial(env, params, fpath)

            if controller not in results.keys():
                results[controller] = [final_distance_to_goal]
            else:
                results[controller].append(final_distance_to_goal)
        print(results)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
