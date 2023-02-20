import numpy as np

from nullspace_optimizer import *

from isaac_victor_envs.tasks.victor import VictorEnv, orientation_error
from isaacgym.torch_utils import quat_apply
import torch
import time

BOX_CENTRE = torch.tensor([0.75, 0.4, 0.803])
BOX_HALFWIDTH = 0.04

def rbf_kernel(x, xbar, lengthscale=1.):
    lengthscale = torch.median(x.reshape(M, T, 7), dim=0).values / np.log(M)
    diff = (x.unsqueeze(0) - xbar.unsqueeze(1))
    diff = diff.reshape(M, M, T, 7)
    diff = ((diff / lengthscale) ** 2).sum(dim=-1)

    # STRUCTURED KERNEL
    return torch.exp(-0.5 * diff).mean(dim=-1)


### INEQUALITY CONSTRAINTS
def dynamics_constraint(x):
    # x is N x T x 7
    N, T, _ = x.shape
    position = x[:, :, :3]
    orientation = x[:, :, 3:]

    p0 = position[:, :-1]
    q0 = orientation[:, :-1]
    p1 = position[:, 1:]
    q1 = orientation[:, 1:]
    q_diff = orientation_error(q1.reshape(-1, 4), q0.reshape(-1, 4)).reshape(N, T - 1, 3)
    p_diff = p1 - p0

    # convert back to total number of constraints - N * (T-1)
    diff = torch.stack((p_diff, q_diff), dim=1)

    # two different constraints on ori change and pos change
    # let's just say that the norm has to be small
    return torch.linalg.norm(diff, dim=-1).reshape(-1) - 0.025

    # abs of diff must be less than 0.025
    return diff.abs() - 0.025


#### EQUALITY CONSTRAINTS
def table_constraint(x, table_height=0.8):
    # will be N * T constraints
    N, T, _ = x.shape
    z = x[:, :, 2]
    return (z - table_height).reshape(-1)


def valid_quat_constraint(x):
    # N constraints
    N, T, _ = x.shape
    q = x[:, :, 3:]
    # valid quaternion -- must have unit norm
    return torch.sum(q**2, dim=-1).reshape(-1) - 1
    #return torch.linalg.norm(q, dim=-1).reshape(-1) - 1


def pose_constraint(x):
    # be flat against table (SE2 constraint)
    N, T, _ = x.shape
    # will be N * T constraints
    q = x[:, :, 3:].reshape(-1, 4)
    z = torch.tensor([0.0, 0.0, 1.0]).reshape(1, 3).repeat(N * T, 1)
    # q should transform [0, 0, 1] to [0, 0, -1]
    # dot product should be -1
    return 1 + torch.bmm(quat_apply(q, z).unsqueeze(1), z.unsqueeze(2)).reshape(-1)


def box_constraint(x):
    N, T, _ = x.shape
    xy = x[:, :, :2]

    box_centre = BOX_CENTRE[:2].reshape(1, 1, 2)

    # constraint on infinity norm - let's see
    return 0.07 + BOX_HALFWIDTH - torch.linalg.norm(xy - box_centre, dim=-1).reshape(-1)

def cost(x, goal):
    # cost can just be sq distance to goal
    # goal is only a position in x,y
    N, T, _ = x.shape
    xy = x[:, :, :2]
    return torch.sum((xy - goal.reshape(1, 1, 2)) ** 2, dim=-1).sum(dim=-1)


class VictorPuckProblem(EuclideanOptimizable):

    def __init__(self, x0, start, goal, M, T, xdim=7):
        super().__init__(M * T * xdim)
        self.nconstraints = M * T + M * T + M * T
        self.nineqconstraints = 2 * M * T + M * T
        self._x0 = x0
        self.M = M
        self.T = T
        self.dx = xdim
        self.goal = goal
        self.start = start

    def x0(self):
        return self._x0.reshape(-1)

    def J(self, x):
        tx = torch.from_numpy(x).reshape(self.M, -1).float()
        x_sequence = tx.reshape(self.M, self.T, -1)
        return cost(x_sequence, self.goal).sum().numpy()

    def dJ(self, x):
        tx = torch.from_numpy(x).reshape(self.M, -1).float()
        lengthscale = torch.clamp(torch.median(tx) / self.M, min=1e-8)
        tx.requires_grad = True

        # lengthscale = 0.1
        x_sequence = tx.reshape(self.M, self.T, -1)
        log_prob = -cost(x_sequence, self.goal)
        score = torch.autograd.grad(log_prob.sum(), x_sequence)[0].reshape(self.M, -1)

        if self.M > 1:
            K = rbf_kernel(tx, tx.detach(), lengthscale.detach())
            grad_K = torch.autograd.grad(K.sum(), tx)[0]
            dJ = K @ score + grad_K
        else:
            dJ = score
        return -dJ.reshape(-1).detach().cpu().numpy() / self.M

    def H(self, x, grad=False):
        tx = torch.from_numpy(x).reshape(self.M, self.T, -1).float()
        if grad:
            tx.requires_grad = True

        def dynamics_wrapper(tx):
            x = torch.cat((self.start.reshape(self.M, 1, 7).to(tx), tx), dim=1)
            return torch.cat((dynamics_constraint(x), box_constraint(x[:, 1:])), dim=0)

        if grad:
            dH = torch.autograd.functional.jacobian(dynamics_wrapper, tx).reshape(self.nineqconstraints, -1)
            return dH.detach().cpu().numpy()

        return dynamics_wrapper(tx).detach().cpu().numpy()

    def dH(self, x):
        return self.H(x, grad=True)

    def G(self, x, grad=False):
        tx = torch.from_numpy(x).reshape(self.M, self.T, -1).float()
        if grad:
            tx.requires_grad = True

        def all_equality_constraints(tx):
            return torch.cat((valid_quat_constraint(tx), pose_constraint(tx), table_constraint(tx)))

        if grad:
            dG = torch.autograd.functional.jacobian(all_equality_constraints, tx).reshape(self.nconstraints, -1)
            return dG.detach().cpu().numpy()

        return all_equality_constraints(tx).cpu().detach().numpy()

    def dG(self, x):
        return self.G(x, grad=True)


if __name__ == "__main__":

    M = 1
    T = 10
    env = VictorEnv(M)
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

    goals = torch.tensor([
        [0.76, 0.55],
        [0.75, 0.25]
    ])

    for goal in goals:
        state = env.get_state()
        ee_pos, ee_ori = state['ee_pos'], state['ee_ori']
        ee_ori = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(ee_ori).reshape(1, 4).repeat(M, 1)
        start = torch.cat((ee_pos, ee_ori), dim=-1).reshape(M, 7).cpu()

        particles = 0.1 * torch.randn(M, T, 7)
        particles = torch.cumsum(particles, dim=1) + start.reshape(M, 1, 7)

        problem = VictorPuckProblem(x0=particles.reshape(-1).numpy(),
                                T=T,
                                M=M,
                                xdim=7,
                                start=start,
                                goal=goal)
        dt = 0.05
        params = {'alphaC': 1, 'debug': 0, 'alphaJ': 0.25, 'dt': dt, 'maxtrials': 1, 'qp_solver': 'osqp',
                  'maxit': 200}
        s= time.time()
        retval = nlspace_solve(problem, params)
        trajectory = retval['x'][-1].reshape(M, T, -1)
        print(time.time() - s)
        exit(0)
        # just choose first trajectory for now
        trajectory = trajectory
        #input('Optimization finished')

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

        box_centre = BOX_CENTRE.numpy()
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

        box_colors = np.array([[0.0, 0.0, 1.0]*4], dtype=np.float32)
        for e in env.envs:
            gym.add_lines(viewer, e, 2, line_vertices, line_colors)
            gym.add_lines(viewer, e, 4, box_lines, box_colors)

        for t in range(T):
            x = trajectory[:, t]
            env.step(torch.from_numpy(x).float().to(device='cuda:0'))

        time.sleep(1.0)
        gym.clear_lines(viewer)

    while not gym.query_viewer_has_closed(viewer):
       time.sleep(0.1)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)