import numpy as np
from isaacgym.torch_utils import quat_apply
from isaac_victor_envs.tasks.victor import VictorPuckEnv, orientation_error

import torch
import time
from functools import partial
from functorch import vmap, jacrev, hessian

from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
from ccai.kernels import rbf_kernel

DEVICE = 'cuda:0'
BOX_CENTRE = torch.tensor([0.75, 0.4, 0.803], device=DEVICE)
BOX_HALFWIDTH = 0.04


class VictorTableProblem:

    def __init__(self, start, goal, T):
        self.dz = 1
        self.dh = self.dz * T
        self.dg = 3 * T + 2
        self.dx = 7
        self.du = 0
        self.T = T
        self.start = start
        self.goal = goal

        self.cost = vmap(partial(_cost_steps, start=self.start))
        self.grad_cost = vmap(jacrev(partial(_cost_steps, start=self.start)))
        self.grad_kernel = jacrev(rbf_kernel, argnums=0)

        self.inequality_constraints = partial(inequality_constraint, start=self.start)
        self.equality_constraints = partial(equality_constraints, goal=self.goal)

        self.g = vmap(self.combined_constraint_fn)
        self.grad_g = vmap(jacrev(self.combined_constraint_fn))
        self.hess_g = vmap(hessian(self.combined_constraint_fn))

        self.x_max = None
        self.x_min = None

    def combined_constraint_fn(self, augmented_trajectory):
        x = augmented_trajectory[:, :self.dx]
        z = augmented_trajectory[:, self.dx:]

        g = self.equality_constraints(x)
        gh = self.inequality_constraints(x).reshape(self.T, -1) + 0.5 * z ** 2

        return torch.cat((g, gh.reshape(-1)), dim=0)

    def eval(self, augmented_trajectory):
        N = augmented_trajectory.shape[0]
        augmented_trajectory = augmented_trajectory.clone().reshape(N, self.T, -1)
        x = augmented_trajectory[:, :, :self.dx]

        grad_J = self.grad_cost(x)
        grad_J = torch.cat((grad_J, torch.zeros(N, self.T, self.dz, device=x.device)), dim=2).reshape(N, -1)

        Xk = x.reshape(N, -1)
        K = rbf_kernel(Xk, Xk)
        grad_K = -self.grad_kernel(Xk, Xk)
        grad_K = torch.einsum('nmmi->nmi', grad_K)
        grad_K = torch.cat((grad_K.reshape(N, N, self.T, self.dx),
                            torch.zeros(N, N, self.T, self.dz, device=x.device)), dim=-1)
        grad_K = grad_K.reshape(N, N, -1)

        G = self.g(augmented_trajectory)
        dG = self.grad_g(augmented_trajectory).reshape(N, -1, self.T * (self.dx + self.dz))
        hessG = self.hess_g(augmented_trajectory).reshape(N, -1, self.T * (self.dx + self.dz),
                                                          self.T * (self.dx + self.dz))

        return grad_J, K, grad_K, G, dG, hessG

    def get_initial_z(self, x):
        N = x.shape[0]
        fh = vmap(self.inequality_constraints)
        h = fh(x).reshape(N, self.T, 1)
        z = torch.where(h < 0, torch.sqrt(- 2 * h), 0)
        return z


def _cost_steps(x, start):
    x = torch.cat((start.reshape(1, 7), x), dim=0)
    T, _ = x.shape
    position = x[:, :3]
    orientation = x[:, 3:]

    p0 = position[:-1]
    q0 = orientation[:-1]
    p1 = position[1:]
    q1 = orientation[1:]
    q_diff = orientation_error(q1.reshape(-1, 4), q0.reshape(-1, 4)).reshape(T - 1, 3)
    p_diff = p1 - p0
    diff = torch.stack((p_diff, q_diff), dim=1)
    return 1000 * torch.sum(diff ** 2)


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
    q = x[:, 3:]
    # valid quaternion -- must have unit norm
    return torch.sum(q ** 2, dim=-1).reshape(-1) - 1


def pose_constraint(x):
    # be flat against table (SE2 constraint)
    T, _ = x.shape
    # will be N * T constraints
    q = x[:, 3:].reshape(-1, 4)
    z = torch.tensor([0.0, 0.0, 1.0], device=DEVICE).reshape(1, 3).repeat(T, 1)
    # q should transform [0, 0, 1] to [0, 0, -1]
    # dot product should be -1

    return 1 + torch.bmm(quat_apply(q, z).unsqueeze(1), z.unsqueeze(2)).reshape(-1)


def box_constraint(x):
    T, _ = x.shape
    xy = x[:, :2]

    box_centre = BOX_CENTRE[:2].reshape(1, 2)

    # constraint on infinity norm - let's see
    return 0.07 + BOX_HALFWIDTH - torch.linalg.norm(xy - box_centre, dim=-1).reshape(-1)


def equality_constraints(x, goal):
    x = x.reshape(-1, 7)
    return torch.cat((
        table_constraint(x, 0.8),
        valid_quat_constraint(x),
        pose_constraint(x),
        goal_constraint(x, goal)
    ), dim=0).reshape(-1)


def inequality_constraint(x, start):
    x = x.reshape(-1, 7)
    return torch.cat((
        box_constraint(x),
    ), dim=0).reshape(-1)


if __name__ == "__main__":

    M = 8
    T = 12
    env = VictorPuckEnv(M)
    sim, gym, viewer = env.get_sim()
    state = env.get_state()
    ee_pos, ee_ori = state['ee_pos'], state['ee_ori']
    try:
        while True:
            start = torch.cat((ee_pos, ee_ori), dim=-1).reshape(M, 7)
            env.step(start)
            print('waiting for you to finish camera adjustment, ctrl-c when done')
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    device = DEVICE
    goals = torch.tensor([
        [0.76, 0.55],
        [0.75, 0.25]
    ], device=device)
    particles = 0.1 * torch.randn(M, T, 7, device=device)
    particles = torch.cumsum(particles, dim=1) + start.reshape(M, 1, 7)
    trajectory = particles
    for goal in goals:
        for k in range(T):
            state = env.get_state()
            ee_pos, ee_ori = state['ee_pos'], state['ee_ori']
            ee_ori = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(ee_ori).reshape(1, 4).repeat(M, 1)
            start = torch.cat((ee_pos, ee_ori), dim=-1).reshape(M, 7)

            problem = VictorTableProblem(start[0], goal, T=T - k)
            solver = ConstrainedSteinTrajOpt(problem, dt=0.1, alpha_J=0.25, alpha_C=1)
            if k == 0:
                solver.iters = 500
            else:
                solver.iters = 50
            s = time.time()
            trajectory = solver.solve(trajectory)
            torch.cuda.synchronize()
            e = time.time()
            print(e - s)
            # input('Optimization finished')
            print('')
            print(T - k)

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

            box_centre = BOX_CENTRE.cpu().numpy()
            box_lines = np.array([
                box_centre + np.array([BOX_HALFWIDTH, BOX_HALFWIDTH, 0.0]),
                box_centre + np.array([BOX_HALFWIDTH, -BOX_HALFWIDTH, 0.0]),
                box_centre + np.array([BOX_HALFWIDTH, -BOX_HALFWIDTH, 0.0]),
                box_centre + np.array([-BOX_HALFWIDTH, -BOX_HALFWIDTH, 0.0]),
                box_centre + np.array([-BOX_HALFWIDTH, -BOX_HALFWIDTH, 0.0]),
                box_centre + np.array([-BOX_HALFWIDTH, BOX_HALFWIDTH, 0.0]),
                box_centre + np.array([-BOX_HALFWIDTH, BOX_HALFWIDTH, 0.0]),
                box_centre + np.array([BOX_HALFWIDTH, BOX_HALFWIDTH, 0.0])
            ], dtype=np.float32)

            box_colors = np.array([[0.0, 0.0, 1.0] * 4], dtype=np.float32)
            for e in env.envs:
                gym.add_lines(viewer, e, 2, line_vertices, line_colors)
                gym.add_lines(viewer, e, 4, box_lines, box_colors)
            # trajectory_colors
            # traj_line_colors = np.array([[0.5, 0., 0.5]*M], dtype=np.float32)
            traj_line_colors = np.random.random((1, M)).astype(np.float32)
            for e in env.envs:
                p = torch.stack((start[:, :3], trajectory[:, 0, :3]), dim=1).reshape(2 * M, 3).cpu().numpy()
                p[:, 2] += 0.005
                gym.add_lines(viewer, e, M, p, traj_line_colors)
                for t in range(T - 1 - k):
                    p = torch.stack((trajectory[:, t, :3], trajectory[:, t + 1, :3]), dim=1).reshape(2 * M,
                                                                                                     3).cpu().numpy()
                    p[:, 2] += 0.005
                    gym.add_lines(viewer, e, M, p, traj_line_colors)
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, False)
                gym.sync_frame_time(sim)

            costs = problem.cost(trajectory)
            idx = torch.argmin(costs)
            x = trajectory[idx, 0].reshape(1, -1).repeat(M, 1)
            env.step(x)
            trajectory = trajectory[:, 1:]

            gym.clear_lines(viewer)

    while not gym.query_viewer_has_closed(viewer):
        time.sleep(0.1)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
