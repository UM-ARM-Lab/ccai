from isaac_victor_envs.tasks.victor import VictorEnv, orientation_error
from isaacgym.torch_utils import quat_apply
import torch
import numpy as np
from nullspace_optimizer_pytorch import nlspace_solve

from functorch import vmap, grad_and_value, jacrev

import time
from torch import optim, nn

from torch.utils.data import DataLoader, Dataset, RandomSampler


class PushingDataset(Dataset):
    def __init__(self, max_size=1000000):
        super().__init__()
        self.D = torch.empty(max_size, 28)
        self.N = 0

    def add_chunk(self, trajectories):
        # takes in N x T x 14 trajectories
        N, T, dx = trajectories.shape
        assert dx == 14
        # convert to robot_t, block_t, block_tp1
        robot, block = torch.chunk(trajectories, chunks=2, dim=-1)
        new_data = torch.cat((robot[:, :-1], block[:, :-1], robot[:, 1:], block[:, 1:]), dim=-1).reshape(-1, 28)
        M = new_data.shape[0]
        self.D[self.N:self.N + M] = new_data.cpu()
        self.N = M + self.N

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        return self.D[item]

    def save(self, name):
        np.savez(name, data=self.D.numpy(), N=self.N)

    def load(self, name):
        data = dict(np.load(name))
        self.D = torch.from_numpy(data['data'])
        self.N = data['N']


def rbf_kernel(x, xbar, lengthscale=1.):
    diff = (x.unsqueeze(0) - xbar.unsqueeze(1)) ** 2

    if M > 1:
        diff = torch.sum(diff.reshape(M, M, T, -1), dim=-1)
        unique_differences_idx = torch.triu_indices(M, M, offset=1)
        unique_differences = diff[unique_differences_idx[0], unique_differences_idx[1], :]
        lengthscale = 0.5 * torch.median(unique_differences.detach(), dim=0).values / np.log(M + 1)
    else:
        lengthscale = 1
    # we weight by time, i.e. we don't care about start and goal differences but do care about middle differeces
    w = torch.linspace(0, torch.pi, steps=T).to(diff)
    w = torch.sin(w)
    w = w / torch.sum(w)
    # STRUCTURED KERNEL
    K = torch.exp(-0.5 * diff / lengthscale).permute(2, 0, 1)
    return (w.reshape(-1, 1, 1) * K).sum(dim=0)


### INEQUALITY CONSTRAINTS
def stepsize_constraint(x):
    # x is T x 14
    T, _ = x.shape
    position = x[:, :3]
    orientation = x[:, 3:]
    p0 = position[:-1]
    q0 = orientation[:-1]
    p1 = position[1:]
    q1 = orientation[1:]
    q_diff = orientation_error(q1.reshape(-1, 4), q0.reshape(-1, 4)).reshape(T - 1, 3)
    p_diff = p1 - p0
    # convert back to total number of constraints - N * (T-1)
    diff = torch.stack((p_diff, q_diff), dim=1)
    # return diff.reshape(-1)**2 - 0.05**2
    # two different constraints on ori change and pos change
    # let's just say that the norm has to be small
    # return diff.reshape(-1) - 0.025
    return torch.linalg.norm(diff, dim=-1).reshape(-1) - 0.025


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
    z = torch.tensor([0.0, 0.0, 1.0]).reshape(1, 3).repeat(T, 1).to(x)
    # q should transform [0, 0, 1] to [0, 0, -1]
    # dot product should be -1
    return 1 + torch.bmm(quat_apply(q, z).unsqueeze(1), z.unsqueeze(2)).reshape(-1)


def cost_to_goal(x, goal):
    # cost can just be sq distance to goal
    # goal is only a position in x,y
    T, _ = x.shape
    xy = x[:, :2]
    d2goal = torch.sum((xy - goal.reshape(1, 2)) ** 2, dim=-1)
    # w = 10 * torch.ones_like(d2goal)
    # w[:-1] *= 0.01
    w = torch.ones_like(d2goal)
    w[-1] *= 10
    return torch.sum(w * d2goal)


def cost_to_block(x):
    rob_x, block_x = torch.chunk(x, chunks=2, dim=-1)
    return torch.sum((rob_x[:, :2] - block_x[:, :2]) ** 2)


def dynamics_constraint(dynamics, x):
    robx, blockx = torch.chunk(x, chunks=2, dim=-1)
    dynamics_input = torch.cat((robx[:-1], blockx[:-1], blockx[1:]), dim=-1)
    return dynamics(dynamics_input).reshape(-1)


def dynamics_constraint2(dynamics, x):
    robx, blockx = torch.chunk(x, chunks=2, dim=-1)
    dynamics_input = torch.cat((robx[:-1], blockx[:-1]), dim=-1)
    return (blockx[1:] - dynamics(dynamics_input)).reshape(-1)


class VictorMoveToGoalProblem:
    def __init__(self, x0, start, goal, M, T, xdim=7):
        self.nconstraints = M * T + M * T + M * T
        self.nineqconstraints = 2 * M * T + M * T
        self._x0 = x0
        self.M = M
        self.T = T
        self.dx = xdim
        self.goal = goal
        # assume start is same for every trajectory
        self.start = start
        self.cost = vmap(grad_and_value(self._cost))

        self.G = vmap(self._G)
        self.dG = vmap(jacrev(self._G))
        self.H = vmap(self._H)
        self.dH = vmap(jacrev(self._H))

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x):
        self._x0 = x

    def _cost(self, x):
        return 10 * cost_to_goal(x, self.goal)

    def _G(self, x):
        return valid_quat_constraint(x)
        return torch.cat((valid_quat_constraint(x), pose_constraint(x), table_constraint(x)), dim=0)

    def _H(self, x):
        xtmp = torch.cat((self.start, x), dim=0)
        return stepsize_constraint(xtmp)

    def eval(self, x):
        x_for_kernel = x.reshape(self.M, -1)
        x_sequence = x.reshape(self.M, self.T, -1)
        # print(x_sequence[0, 0, 3:])
        # get cost
        dcost, cost = self.cost(x_sequence)

        if self.M > 1:
            # get kernel
            Kxx = rbf_kernel(x_for_kernel, x_for_kernel.detach())
            grad_K = torch.autograd.grad(-Kxx.sum(), x_for_kernel)[0]
            score = -dcost.reshape(self.M, -1)
            dJ = Kxx @ score + grad_K
        else:
            dJ = -dcost

        # get J and dJ
        J = torch.mean(cost)
        # dJ = score
        dJ = -dJ.reshape(-1) / self.M

        # x_sequence[0, 0, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(x_sequence)
        G = self.G(x_sequence).reshape(-1)
        dG = self.dG(x_sequence)
        H = self.H(x_sequence).reshape(-1)
        dH = self.dH(x_sequence)

        # print(x_sequence[:, :, 3:])
        # print(G)
        # print(dG)
        # need to create block diagonals from per sample H and dH
        traj_vars = self.T * self.dx
        dG = torch.diag_embed(dG.permute(3, 2, 1, 0)).permute(4, 2, 3, 1, 0).reshape(-1, self.M * traj_vars)
        dH = torch.diag_embed(dH.permute(3, 2, 1, 0)).permute(4, 2, 3, 1, 0).reshape(-1, self.M * traj_vars)

        # dH = None
        # H = None
        return J, G, H, dJ, dG, dH


class VictorBlockProblem:

    def __init__(self, x0, start, goal, M, T, dynamics, xdim=14):
        self._x0 = x0
        self.start = start
        self.goal = goal
        self.M = M
        self.T = T
        self.dx = xdim
        self.dynamics = dynamics
        self.cost = vmap(grad_and_value(self._cost))
        self.G = vmap(self._G)
        self.dG = vmap(jacrev(self._G))
        self.H = vmap(self._H)
        self.dH = vmap(jacrev(self._H))

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x):
        self._x0 = x

    def _cost(self, x):
        rob_x, block_x = torch.chunk(x, chunks=2, dim=-1)

        return 10 * cost_to_goal(block_x, self.goal) + cost_to_block(x)

    def _G(self, x):
        xtmp = torch.cat((self.start, x), dim=0)
        robx, blockx = torch.chunk(xtmp, chunks=2, dim=-1)
        return torch.cat((valid_quat_constraint(robx[1:]), pose_constraint(robx[1:]), table_constraint(robx[1:]),
                          dynamics_constraint2(self.dynamics, xtmp)), dim=0)

    def _H(self, x):
        xtmp = torch.cat((self.start, x), dim=0)
        rob_x, block_x = torch.chunk(xtmp, chunks=2, dim=-1)
        return stepsize_constraint(rob_x)

    def eval(self, x):
        x_for_kernel = x.reshape(self.M, -1)
        x_sequence = x.reshape(self.M, self.T, -1)
        # print(x_sequence[0, 0, 3:])
        # get cost -- cost on getting block to goal location
        dcost, cost = self.cost(x_sequence)

        if self.M > 1:
            # get kernel
            Kxx = rbf_kernel(x_for_kernel, x_for_kernel.detach())
            grad_K = torch.autograd.grad(-Kxx.sum(), x_for_kernel)[0]
            # get J and dJ
            score = -dcost.reshape(self.M, -1)
            dJ = Kxx @ score + grad_K
        else:
            dJ = -dcost.reshape(-1)

        J = torch.mean(cost)

        dJ = -dJ.reshape(-1) / self.M

        # x_sequence[0, 0, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(x_sequence)
        G = self.G(x_sequence).reshape(-1)
        dG = self.dG(x_sequence)
        H = self.H(x_sequence).reshape(-1)
        dH = self.dH(x_sequence)

        # print(x_sequence[:, :, 3:])
        # print(G)
        # print(dG)
        # need to create block diagonals from per sample H and dH
        traj_vars = self.T * self.dx
        dG = torch.diag_embed(dG.permute(3, 2, 1, 0)).permute(4, 2, 3, 1, 0).reshape(-1, self.M * traj_vars)
        dH = torch.diag_embed(dH.permute(3, 2, 1, 0)).permute(4, 2, 3, 1, 0).reshape(-1, self.M * traj_vars)

        # dH = None
        # H = None
        return J, G, H, dJ, dG, dH


class PrimalDualPlanner:

    def __init__(self, problem):
        self.primal_eta = 0.001
        self.dual_eta = 0.01
        self.iters = 1000
        self.problem = problem
        self.dx = problem.dx

        self.primal_steps = 1
        self.dual_steps = 1

    def solve(self):
        x = self.problem.x0
        J, G, H, dJ, dG, dH = self.problem.eval(x)
        # initialise dual vars
        lam = torch.ones_like(G).reshape(-1, 1)
        lam = torch.where(G.reshape(-1, 1) > 0, lam, -lam)
        mu = torch.ones_like(H).reshape(-1, 1)

        for iter in range(self.iters):
            # first we do descent on the primal problem
            for primal_iter in range(self.primal_steps):
                _step = dJ + 0*mu.t() @ dH + lam.t() @ dG
                x = x - _step * self.primal_eta
                # Now we re eval and max the dual problem
                J, G, H, dJ, dG, dH = self.problem.eval(x)

            lam = lam + self.dual_eta * G.reshape(-1, 1)
            #mu = mu + self.dual_eta * H.reshape(-1, 1)
            #print(lam[0])
            print(G[0])
            mu = torch.where(mu < 0, 0, mu)
            #print(J, G.abs().max(), H.max())
            # print(G)
            # print(H)

        return x

    def plan(self, start, goal):
        print(start)
        print(goal)
        self.problem.start = start.reshape(-1, self.dx)[0].reshape(1, self.dx)
        self.problem.goal = goal.reshape(1, 2)

        # set up initial particles
        particles = 0.1 * torch.randn(self.problem.M, self.problem.T, self.dx).to(start)
        particles = torch.cumsum(particles, dim=1) + start.reshape(self.problem.M, 1, self.dx)
        # particles[:, :, 3:] /= torch.linalg.norm(particles[:, :, 3:], dim=-1, keepdim=True)
        self.problem.x0 = particles.reshape(-1)

        import time
        s = time.time()
        x_sequence = self.solve()
        e = time.time()
        print('optim time', e - s)
        # x_sequence = self.solver.solve_ode()
        J, G, H, _, _, _ = self.problem.eval(x_sequence)

        x_sequence = x_sequence.reshape(self.problem.M, self.problem.T, -1)
        # print(start[0, :3])
        # returns M trajectories where M is decided in problem
        return torch.cat((start.reshape(self.problem.M, 1, self.dx), x_sequence), dim=1)


class TrajectoryPlanner:

    def __init__(self, problem):
        params = {'alphaC': 1,
                  'alphaJ': 0.25,
                  'dt': 0.1,
                  'maxit': 1000}

        self.solver = nlspace_solve(problem, params)
        self.dx = problem.dx
        self.M = problem.M
        self.T = problem.T

    def plan(self, start, goal):
        self.solver.problem.start = start.reshape(-1, self.dx)[0].reshape(1, self.dx)
        self.solver.problem.goal = goal.reshape(1, 2)

        # set up initial particles
        particles = 0.1 * torch.randn(self.solver.problem.M, self.solver.problem.T, self.dx).to(start)
        particles = torch.cumsum(particles, dim=1) + start.reshape(self.solver.problem.M, 1, self.dx)
        # particles[:, :, 3:] /= torch.linalg.norm(particles[:, :, 3:], dim=-1, keepdim=True)
        self.solver.problem.x0 = particles.reshape(-1)

        x_sequence = self.solver.solve()
        # x_sequence = self.solver.solve_ode()
        J, G, H, _, _, _ = self.solver.problem.eval(x_sequence)

        x_sequence = x_sequence.reshape(self.solver.problem.M, self.solver.problem.T, -1)
        # returns M trajectories where M is decided in problem
        return torch.cat((start.reshape(self.solver.problem.M, 1, self.dx), x_sequence), dim=1)


def generate_initial_dataset(env, trajectory_generator, N=1):
    dataset = []
    # randomly sample goal from table
    for n in range(N):
        # reset env
        env.reset()

        state = env.get_state()
        start = torch.cat((state['ee_pos'], state['ee_ori']), dim=-1)

        trajectory = [torch.cat((state['ee_pos'], state['ee_ori'],
                                 state['block_pos'], state['block_ori']), dim=-1)]

        d2block = state['block_pos'] - state['ee_pos']
        goal = state['ee_pos'][0, :2] + 1.5 * d2block[0, :2]

        # goal[0] = 0.4 * goal[0] + 0.55
        # goal[1] = 0.7 * goal[1] - 0.1

        # plan trajectory
        plan = trajectory_generator.plan(start, goal)
        add_goal_to_sim(env, goal)
        for t in range(plan.shape[1] - 1):
            env.step(plan[:, t + 1])
            state = env.get_state()
            trajectory.append(torch.cat((state['ee_pos'], state['ee_ori'],
                                         state['block_pos'], state['block_ori']), dim=-1))

        trajectory = torch.stack(trajectory, dim=1)
        dataset.append(trajectory)
        gym.clear_lines(viewer)

    dataset = torch.stack(dataset, dim=0)
    return dataset  # should be N x M x T x 14


def generate_dataset(env, trajectory_generator, N=1):
    dataset = []
    # randomly sample goal from table
    for n in range(N):
        # reset env
        env.reset()

        state = env.get_state()
        start = torch.cat((state['ee_pos'], state['ee_ori'], state['block_pos'], state['block_ori']), dim=-1)

        d2block = state['block_pos'] - state['ee_pos']
        goal = state['ee_pos'][0, :2] + 1.5 * d2block[0, :2]

        goal += 0.05 * torch.randn(2).to(start)  # + torch.tensor([0.75, 0.25]).to(start)

        trajectory = [torch.cat((state['ee_pos'], state['ee_ori'],
                                 state['block_pos'], state['block_ori']), dim=-1)]

        # plan trajectory
        plan = trajectory_generator.plan(start, goal)
        add_goal_to_sim(env, goal)

        for t in range(plan.shape[1] - 1):
            env.step(plan[:, t + 1, :7])
            state = env.get_state()
            trajectory.append(torch.cat((state['ee_pos'], state['ee_ori'],
                                         state['block_pos'], state['block_ori']), dim=-1))

        trajectory = torch.stack(trajectory, dim=1)
        dataset.append(trajectory)
        gym.clear_lines(viewer)

    dataset = torch.stack(dataset, dim=0)
    return dataset  # should be N x M x T x 14


def add_goal_to_sim(env, goal):
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


def fit_dynamics_implicit(net, dataset, epochs, batch_size=32):
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)

    optimiser = optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x in loader:
            x = x.to(device='cuda:0')
            # now we regress on a perturbation to the dynamics
            perturbation = 0 * 0.1 * torch.randn(x.shape[0], 7).to(x)
            # we want some to actually be told to have zero residual for the constraint
            perturbation[:x.shape[0] // 2] *= 0
            x[:, -7:] += perturbation

            # renormalize quaternion
            # x[:, -4:] /= torch.linalg.norm(x[:, -4:], dim=1, keepdim=True)
            residual = net(x)
            loss = torch.sum((residual - perturbation) ** 2)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            epoch_loss += loss.item() / len(x)
        print('epoch loss', epoch_loss)
    return net


def fit_dynamics_explicit(net, dataset, epochs, batch_size=32):
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)

    optimiser = optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x in loader:
            x = x.to(device='cuda:0')
            y = x[:, -7:]
            x = x[:, :-7]

            # renormalize quaternion
            # x[:, -4:] /= torch.linalg.norm(x[:, -4:], dim=1, keepdim=True)
            y_hat = net(x)
            ori_error = orientation_error(y[:, 3:], y_hat[:, 3:])
            pos_error = y[:, :3] - y_hat[:, :3]
            loss = torch.sum(ori_error ** 2 + pos_error ** 2)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            epoch_loss += loss.item() / len(x)
        print('epoch loss', epoch_loss)
    return net


if __name__ == "__main__":
    M = 1
    T = 20

    # set up viewer
    env = VictorEnv(M, control_mode='cartesian_impedance', block=True)
    sim, gym, viewer = env.get_sim()
    state = env.get_state()

    try:
        while True:
            start = torch.cat((state['ee_pos'], state['ee_ori']), dim=-1).reshape(M, 7)
            env.step(start)
            print('waiting for you to finish camera adjustment, ctrl-c when done')
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    dataset = PushingDataset()
    dataset.load('victor_block_pushing_dataset_iter_2.npz')

    if True:
        # set up initial planner
        MoveToGoalPlanner = TrajectoryPlanner(problem=VictorMoveToGoalProblem(x0=None,
                                                                              T=T,
                                                                              M=M,
                                                                              xdim=7,
                                                                              start=None,
                                                                              goal=None)
                                              )

    dataset.add_chunk(generate_initial_dataset(env, MoveToGoalPlanner, N=100).reshape(-1, T + 1, 14))

    # dynamics = nn.Sequential(
    #    nn.Linear(21, 64),
    #    nn.ReLU(),
    #    nn.Linear(64, 64),
    #    nn.ReLU(),
    #    nn.Linear(64, 7)
    # )
    dynamics = nn.Sequential(
        nn.Linear(21, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 7)
    )
    dynamics = dynamics.to('cuda:0')

    N = 10
    for iter in range(N):
        print('##################')
        print(iter, len(dataset))
        # save the dataset
        dataset.save(f'victor_block_pushing_dataset_iter_{iter}.npz')
        # train dynamics
        dynamics = fit_dynamics_explicit(dynamics, dataset, epochs=1000, batch_size=256)
        # set up initial planner
        PushBlockPlanner = TrajectoryPlanner(problem=VictorBlockProblem(x0=None,
                                                                        T=T,
                                                                        M=M,
                                                                        xdim=14,
                                                                        start=None,
                                                                        goal=None,
                                                                        dynamics=dynamics))

        dataset.add_chunk(generate_dataset(env, PushBlockPlanner, N=100).reshape(-1, T + 1, 14))

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
