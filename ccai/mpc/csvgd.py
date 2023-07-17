import torch
from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt


class Constrained_SVGD_MPC:

    def __init__(self, problem, params):
        self.fix_T = params.get('receding_horizon', True)
        self.device = params.get('device', 'cuda:0')
        self.online_iters = params.get('online_iters', 10)
        self.warmup_iters = params.get('warmup_iters', 100)
        self.N = params.get('N')
        self.resample_steps = params.get('resample_steps', 1)
        self.flow_model = params.get('flow_model', None)
        self.problem = problem
        self.solver = ConstrainedSteinTrajOpt(problem, params)

        # initialize randomly
        self.x = self.problem.get_initial_xu(self.N)
        self.warmed_up = False
        self.iter = 0

    def step(self, state, constr_param, **kwargs):
        if self.fix_T:
            new_T = None
        else:
            if self.warmed_up:
                new_T = self.problem.T - 1
            else:
                new_T = self.problem.T

        self.problem.update(state, T=new_T, **kwargs)
        if self.flow_model is not None:
            state_norm = (state - self.flow_model.x_mean[:self.problem.dx]) / self.flow_model.x_std[:self.problem.dx]

            sampled_x = self.flow_model.sample(state_norm.repeat(self.N, 1),
                                               self.problem.goal[:3].repeat(self.N, 1),
                                               constr_param.repeat(self.N, 1, 1))
            sampled_x = sampled_x.detach()
            # want to choose between sampled x and current x
            all_x = torch.cat((sampled_x, self.x), dim=0)

            # get values for z
            all_z = self.problem.get_initial_z(all_x)
            all_x = torch.cat((all_x, all_z), dim=-1)

            J = self.problem.get_cost(all_x.reshape(2*self.N, self.problem.T, -1)[:, :, :self.problem.dx + self.problem.du])
            C, _, _ = self.problem.combined_constraints(all_x.reshape(2*self.N, self.problem.T, -1))
            penalty = J.reshape(2*self.N) + self.solver.penalty * torch.sum(C.reshape(2*self.N, -1).abs(), dim=1)
            idx = torch.argsort(penalty, descending=False)
            self.x = all_x[idx[:self.N], :, :self.problem.dx + self.problem.du]

        # warm starting
        if self.warmed_up:
            self.solver.iters = self.online_iters
            resample = True if (self.iter + 1) % self.resample_steps == 0 else False
        else:
            self.solver.iters = self.warmup_iters
            self.warmed_up = True
            resample = False

        path = self.solver.solve(self.x, resample)
        self.x = path[-1]
        self.iter += 1
        best_trajectory = self.x[0].clone()
        all_trajectories = self.x.clone()
        self.shift()
        return best_trajectory, all_trajectories

    def shift(self):
        if self.fix_T:
            self.x = torch.roll(self.x, shifts=-1, dims=1)
            self.x[:, -1] = self.x[:, -2]  # just copy over previous last
        else:
            self.x = self.x[:, 1:]

    def reset(self, start, initial_x=None, **kwargs):
        self.problem.update(start, **kwargs)
        self.warmed_up = False
        self.iter = 0
        self.x = self.problem.get_initial_xu(self.N)
        if initial_x is not None:
            self.x = initial_x
            self.warmup_iters = 10