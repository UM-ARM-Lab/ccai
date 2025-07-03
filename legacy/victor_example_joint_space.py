import numpy as np

from nullspace_optimizer import *

from isaac_victor_envs.tasks.victor import VictorEnv, orientation_error, quat_change_convention
from isaacgym.torch_utils import quat_apply
import torch
import time
import allegro_optimized_wrapper as pk

BOX_CENTRE = torch.tensor([0.75, 0.4, 0.803])
BOX_HALFWIDTH = 0.04
asset = '/home/tpower/dev/isaac_test/IsaacVictorEnvs/isaac_victor_envs/assets/victor/victor.urdf'
ee_name = 'victor_left_arm_striker_mallet_tip'

chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name)


def rbf_kernel(x, xbar, lengthscale=1.):

    lengthscale = torch.median(x.reshape(M, T, 7), dim=0).values / np.log(M)
    if M == 1:
        lengthscale = torch.ones_like(lengthscale)
    diff = (x.unsqueeze(0) - xbar.unsqueeze(1))
    diff = diff.reshape(M, M, T, 7)
    diff = ((diff / lengthscale) ** 2).sum(dim=-1)
    # STRUCTURED KERNEL
    return torch.exp(-0.5 * diff).mean(dim=-1)


### INEQUALITY CONSTRAINTS
def dynamics_constraint(x):
    # x is N x T x 7
    N, T, _ = x.shape
    diff = x[:, 1:] - x[:, :-1]
    # Diff must be less than this
    return torch.linalg.norm(diff, dim=-1, ord=1).reshape(-1) - 0.1
    #return diff.abs().reshape(-1) - 0.025


#### EQUALITY CONSTRAINTS
def goal_constraint(x, goal):
    N, T, _ = x.shape
    xy = x[:, -1, :2]
    return (xy - goal.reshape(1, 2)).reshape(-1)


def table_constraint(x, table_height=0.79):
    # will be N * T constraints
    N, T, _ = x.shape
    z = x[:, :, 2]
    return (z - table_height).reshape(-1)


def valid_quat_constraint(x):
    # N constraints
    N, T, _ = x.shape
    q = x[:, :, 3:]
    # valid quaternion -- must have unit norm
    return torch.linalg.norm(q, dim=-1).reshape(-1) - 1


def pose_constraint(x):
    # be flat against table (SE2 constraint)
    N, T, _ = x.shape
    # will be N * T constraints
    q = x[:, :, 3:].reshape(-1, 4)
    z = torch.tensor([0.0, 0.0, 1.0]).reshape(1, 3).repeat(N * T, 1)
    # q should transform [0, 0, 1] to [0, 0, -1]
    # dot product should be -1
    return 1 + torch.bmm(quat_apply(q, z).unsqueeze(1), z.unsqueeze(2)).reshape(-1)


def box_constraint(q):
    N, T, _ = q.shape
    transform = chain.forward_kinematics(q.reshape(-1, 7))
    m = transform.get_matrix()
    x = m[:, :3, 3]
    x = x.reshape(N, T, 3)
    xy = x[:, :, :2]
    box_centre = BOX_CENTRE[:2].reshape(1, 1, 2)

    # constraint on infinity norm - let's see
    return 0.07 + BOX_HALFWIDTH - torch.linalg.norm(xy - box_centre, dim=-1).reshape(-1)


def equality_constraint(q, goal=None):
    # all equality constraints on end effector by first performing FK
    N, T, _ = q.shape
    transform = chain.forward_kinematics(q.reshape(-1, 7))
    m = transform.get_matrix()
    p, quat = m[:, :3, 3], pk.matrix_to_quaternion(m[:, :3, :3])
    quat = quat_change_convention(quat, 'wxyz')
    x = torch.cat((p.reshape(N, T, 3), quat.reshape(N, T, 4)), dim=-1)
    if goal is None:
        return torch.cat((table_constraint(x), pose_constraint(x)))

    return torch.cat((goal_constraint(x, goal), table_constraint(x), pose_constraint(x)))


def joint_limits(q):
    joint_lims = torch.tensor([2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05]).reshape(1, 1, 7)
    return (q.abs() - joint_lims).reshape(-1)
    # upper = q - joint_lims
    # lower = - q - joint_lims
    # return torch.cat((upper, lower), dim=-1).reshape(-1)


def goal_cost(q, goal):
    # cost can just be sq distance to goal
    # goal is only a position in x,y
    N, T, _ = q.shape
    transform = chain.forward_kinematics(q.reshape(-1, 7))
    m = transform.get_matrix()
    x = m[:, :3, 3]
    x = x.reshape(N, T, 3)
    xy = x[:, -1, :2]
    diff = xy - goal.reshape(1, 1, 2)
    running = 0.1 * torch.sum(diff[:, :-1]**2, dim=-1).sum(dim=-1)
    terminal = 20*(diff[:, -1]**2).sum(dim=-1)
    return running + terminal


def path_cost(q):
    N, T, _ = q.shape
    diff = q[:, 1:] - q[:, :-1]
    return torch.sum(diff ** 2, dim=-1).sum(dim=-1)
    return torch.linalg.norm(diff, dim=-1).sum(dim=-1)


def force_constraint(q):
    N, T, _ = q.shape
    # only care about constraint at final pose
    J = chain.jacobian(q[:, -1])
    desired_wrench = torch.tensor([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 6, 1).repeat(N, 1, 1)

    # must be within actuation limits (say 30)
    torques = J.transpose(1, 2) @ desired_wrench

    torque_constraint = (torques.abs() - 10).reshape(-1)
    nullspace_P = torch.eye(6).reshape(1, 6, 6) - torch.linalg.pinv(J.transpose(1, 2)) @ J.transpose(1, 2)
    force_null_constraint = (nullspace_P @ desired_wrench).sum(dim=1).reshape(-1)

    return torch.cat((torque_constraint, force_null_constraint - 1e-3))


class VictorPuckProblem(EuclideanOptimizable):

    def __init__(self, x0, start, goal, M, T, xdim=7):
        super().__init__(M * T * xdim)
        self.nconstraints = 2 * (M * T)
        self.nineqconstraints = 1 * xdim * M * T + xdim * M + 2 * M * T + M
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
        x_sequence = torch.cat((self.start.reshape(self.M, 1, 7).to(x_sequence), x_sequence), dim=1)
        return -goal_cost(x_sequence, self.goal).sum().numpy() / self.M
        return path_cost(x_sequence).sum().numpy()

    def dJ(self, x):
        tx = torch.from_numpy(x).reshape(self.M, -1).float()
        lengthscale = torch.median(tx) / self.M
        tx.requires_grad = True

        # lengthscale = 0.1
        def cost_wrapper(x):
            x = x.reshape(self.M, self.T, -1)
            x = torch.cat((self.start.reshape(self.M, 1, 7).to(x), x), dim=1)
            return -goal_cost(x, self.goal).sum() - path_cost(x).sum() * 0.01

        score = torch.autograd.functional.jacobian(cost_wrapper, tx).reshape(self.M, -1)
        K = rbf_kernel(tx, tx.detach(), lengthscale.detach())
        grad_K = torch.autograd.grad(K.sum(), tx)[0]

        dJ = K @ score + grad_K

        return -dJ.reshape(-1).detach().cpu().numpy() / self.M

    def H(self, x, grad=False):
        tx = torch.from_numpy(x).reshape(self.M, self.T, -1).float()
        if grad:
            tx.requires_grad = True

        def ineq_wrapper(tx):
            x = torch.cat((self.start.reshape(self.M, 1, 7).to(tx), tx), dim=1)
            return torch.cat((dynamics_constraint(x), box_constraint(tx),
                              force_constraint(tx), joint_limits(tx)))

        if grad:
            dH = torch.autograd.functional.jacobian(ineq_wrapper, tx).reshape(self.nineqconstraints, -1)
            return dH.detach().cpu().numpy()

        return ineq_wrapper(tx).detach().cpu().numpy()

    def dH(self, x):
        return self.H(x, grad=True)

    def G(self, x, grad=False):
        tx = torch.from_numpy(x).reshape(self.M, self.T, -1).float()
        if grad:
            tx.requires_grad = True

        def eq_wrapper(x):
            return equality_constraint(x)

        if grad:
            dG = torch.autograd.functional.jacobian(eq_wrapper, tx).reshape(self.nconstraints, -1)
            return dG.detach().cpu().numpy()

        return eq_wrapper(tx).cpu().detach().numpy()

    def dG(self, x):
        return self.G(x, grad=True)


if __name__ == "__main__":

    M = 1
    env = VictorEnv(M, control_mode='joint_impedance')
    sim, gym, viewer = env.get_sim()
    q = env.get_state()['q']

    # transform = chain.forward_kinematics(q.reshape(-1, 7))
    # m = transform.get_matrix()
    # p, q = m[:, :3, 3], pk.matrix_to_quaternion(m[:, :3, :3])
    # q = quat_change_convention(q, 'wxyz')

    # x = torch.cat((p, q), dim=1).unsqueeze(0)
    # print(pose_constraint(x))

    try:
        while True:
            env.step(q)
            print('waiting for you to finish camera adjustment, ctrl-c when done')
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    goals = torch.tensor([
        [0.75, 0.55],
        [0.75, 0.25]
    ])

    for goal in goals:
        q = env.get_state()['q']
        start = q.cpu()

        T = 20
        particles_end = torch.randn(M, 7)
        alpha = torch.linspace(0, 1, steps=T).reshape(1, -1, 1).repeat(M, 1, 1)
        particles = (1 - alpha) * start.reshape(M, 1, 7) + alpha * particles_end.reshape(M, 1, 7)

        particles = torch.randn(M, T, 7) * 0.1
        particles = torch.cumsum(particles, dim=1) + start.reshape(M, 1, 7)

        problem = VictorPuckProblem(x0=particles.reshape(-1).numpy(),
                                    T=T,
                                    M=M,
                                    xdim=7,
                                    start=start,
                                    goal=goal)

        dt = 0.05
        params = {'alphaC': 1, 'debug': 0, 'alphaJ': 0.25, 'dt': dt, 'maxtrials': 1, 'qp_solver': 'osqp',
                  'maxit': 150}
        retval = nlspace_solve(problem, params)
        trajectory = retval['x'][-1].reshape(M, T, -1)

        # just choose first trajectory for now
        trajectory = trajectory
        # input('Optimization finished')

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

        box_colors = np.array([[0.0, 0.0, 1.0] * 4], dtype=np.float32)
        for e in env.envs:
            gym.add_lines(viewer, e, 2, line_vertices, line_colors)
            gym.add_lines(viewer, e, 4, box_lines, box_colors)

        transform = chain.forward_kinematics(trajectory.reshape(-1, 7))
        m = transform.get_matrix()
        p, quat = m[:, :3, 3], pk.matrix_to_quaternion(m[:, :3, :3])


        input('Computed trajectory, waiting to execute...')


        for t in range(T):
            x = trajectory[:, t]
            env.step(torch.from_numpy(x).float().to(device='cuda:0'))

        time.sleep(1.0)
        gym.clear_lines(viewer)

    while not gym.query_viewer_has_closed(viewer):
        time.sleep(0.1)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
