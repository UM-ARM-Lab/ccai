import torch
from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt


class Constrained_SVGD_MPC:

    def __init__(self, problem, params):
        self.fix_T = params.get('receding_horizon', True)
        self.device = params.get('device', 'cuda:0')
        self.online_iters = params.get('online_iters', 10)
        self.warmup_iters = params.get('warmup_iters', 100)
        self.N = params.get('N')
        self.problem = problem
        self.solver = ConstrainedSteinTrajOpt(problem, params)

        # initialize randomly
        self.x = self.problem.get_initial_xu(self.N)
        self.warmed_up = False

    def step(self, state, **kwargs):
        if self.fix_T:
            new_T = None
        else:
            if self.warmed_up:
                new_T = self.problem.T - 1
            else:
                new_T = self.problem.T

        self.problem.update(state, T=new_T, **kwargs)
        # warm starting
        if self.warmed_up:
            self.solver.iters = self.online_iters
        else:
            self.solver.iters = self.warmup_iters
            self.warmed_up = True

        self.x = self.solver.solve(self.x)

        # choose lowest cost trajectory
        J = self.problem.get_cost(self.x)
        best_trajectory = self.x[torch.argmin(J, dim=0)].clone().reshape(self.problem.T, -1)
        all_trajectories = self.x.clone()
        self.shift()
        return best_trajectory, all_trajectories

    def shift(self):
        if self.fix_T:
            self.x = torch.roll(self.x, shifts=-1, dims=1)
            self.x[:, -1] = self.x[:, -2]  # just copy over previous last
        else:
            self.x = self.x[:, 1:]

    def reset(self, start, **kwargs):
        self.problem.update(start, **kwargs)
        self.warmed_up = False
