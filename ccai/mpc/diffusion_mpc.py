import torch
from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt


class Diffusion_MPC:

    def __init__(self, problem, params):
        self.fix_T = params.get('receding_horizon', True)
        self.device = params.get('device', 'cuda:0')
        self.online_iters = params.get('online_iters', 10)
        self.warmup_iters = params.get('warmup_iters', 100)
        self.N = params.get('N')
        self.resample_steps = params.get('resample_steps', 1)
        self.flow_model = params.get('flow_model', None)
        self.add_start = params.get('add_start', True)
        self.penalty = params.get('penalty', 100)
        self.use_solver = params.get('use_solver', True)
        self.solver_steps = params.get('solver_steps', 10)
        self.solver = ConstrainedSteinTrajOpt(problem, params)
        self.solver.iters = self.solver_steps

        if self.flow_model is None:
            raise ValueError

        self.problem = problem

        # initialize randomly
        self.warmed_up = False
        self.iter = 0
        self.x = None

    def step(self, state, constr_param, **kwargs):
        if self.fix_T:
            new_T = None
        else:
            if self.warmed_up:
                new_T = self.problem.T - 1
            else:
                new_T = self.problem.T

        self.problem.update(state, T=new_T, **kwargs)
        # update problem
        self.flow_model.model.diffusion_model.problem = self.problem

        if not self.warmed_up:
            x = self.flow_model.sample(state.expand(self.N, -1),
                                            self.problem.goal[:3].expand(self.N, -1),
                                            constr_param.expand(self.N, -1, -1))
            self.warmed_up = True
            self.solver.iters = self.warmup_iters
        else:
            self.solver.iters = self.solver_steps
            if self.add_start:
                initial_traj = torch.cat((state.expand(self.N, 1, -1), self.x), dim=1)
            else:
                initial_traj = self.x
            x = self.flow_model.resample(state.expand(self.N, -1),
                                            self.problem.goal[:3].expand(self.N, -1),
                                            constr_param.expand(self.N, -1, -1),
                                            initial_trajectory=initial_traj,
                                            timestep=self.online_iters)

        # choose best for solver
        if self.add_start:
            x = x[:, 1:].detach()
        if self.x is None:
            self.x = x
        else:
            all_x = torch.cat((x, self.x), dim=0)
            all_z = self.problem.get_initial_z(all_x)
            all_x = torch.cat((all_x, all_z), dim=-1)

            J = self.problem.get_cost(
                all_x.reshape(2 * self.N, self.problem.T, -1)[:, :, :self.problem.dx + self.problem.du])
            C, _, _ = self.problem.combined_constraints(all_x.reshape(2 * self.N, self.problem.T, -1))
            penalty = J.reshape(2 * self.N) + self.solver.penalty * torch.sum(C.reshape(2 * self.N, -1).abs(), dim=1)
            idx = torch.argsort(penalty, descending=False)
            self.x = all_x[idx[:self.N], :, :self.problem.dx + self.problem.du]

        # first is just the start
        if self.use_solver:
            path = self.solver.solve(self.x, False)
            self.x = path[-1]
        else:
            # get values for z
            z = self.problem.get_initial_z(x)
            xz = torch.cat((x, z), dim=-1)

            J = self.problem.get_cost(x.reshape(self.N, self.problem.T, -1))

            C, _, _ = self.problem.combined_constraints(xz.reshape(self.N, self.problem.T, -1))
            penalty = J.reshape(self.N) + self.penalty * torch.sum(C.reshape(self.N, -1).abs(), dim=1)
            idx = torch.argsort(penalty, descending=False)
            # order by penalty
            self.x = x[idx, :, :self.problem.dx + self.problem.du]

        best_trajectory = self.x[0].clone()
        all_trajectories = self.x.clone()
        # don't need to shift, diffusion model *should* take care of that for us
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
        self.x = None
        #if initial_x idds not None:
        #    self.x = initial_x
        #    self.warmup_iters = 10