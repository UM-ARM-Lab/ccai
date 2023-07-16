import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from torch.func import vmap, jacrev, hessian
import yaml

from ccai.kernels import rbf_kernel
from ccai.quadrotor_env import QuadrotorEnv
from ccai.quadrotor import Quadrotor12DDynamics
from ccai.gp import BatchGPSurfaceModel

from ccai.problem import ConstrainedSVGDProblem
from ccai.batched_stein_gradient import compute_constrained_gradient
from ccai.dataset import QuadrotorSingleConstraintTrajectoryDataset, QuadrotorMultiConstraintTrajectoryDataset
import pathlib
from torch import nn
import tqdm
import copy

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
from ccai.models.training import EMA

from ccai.models.trajectory_samplers import TrajectorySampler


def cost(trajectory, goal):
    x = trajectory[:, :12]
    u = trajectory[:, 12:]
    T = x.shape[0]
    Q = torch.eye(12, device=trajectory.device)
    Q[5, 5] = 1e-2
    Q[2, 2] = 1e-3
    Q[3:, 3:] *= 0.5
    P = Q * 100
    R = 1 * torch.eye(4, device=trajectory.device)
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

    def __init__(self, T, device='cuda:0', alpha=0.1, include_obstacle=False):
        super().__init__(None, None, T, device)
        self.T = T
        self.dx = 12
        self.du = 4
        if include_obstacle:
            self.dz = 1
        else:
            self.dz = 0

        self.dh = self.dz * T
        self.dg = 12 * T + T - 1
        self.alpha = alpha
        self.include_obstacle = include_obstacle
        # self.dg = 2 * T - 1
        data = np.load('surface_data.npz')
        self.gp_x_points = torch.from_numpy(data['xy']).to(dtype=torch.float32, device=device)

        # GP which models surface
        # self.surface_gp = GPSurfaceModel(torch.from_numpy(data['xy']).to(dtype=torch.float32, device=device),
        #                                 torch.from_numpy(data['z']).to(dtype=torch.float32, device=device))
        self.surface_gp = BatchGPSurfaceModel()

        # gradient and hessian of dynamics
        self._dynamics = Quadrotor12DDynamics(dt=0.1)
        self.dynamics_constraint = vmap(vmap(self._dynamics_constraint))
        self.grad_dynamics = vmap(vmap(jacrev(self._dynamics_constraint)))
        self.hess_dynamics = vmap(vmap(hessian(self._dynamics_constraint)))

        self.height_constraint = vmap(vmap(self._height_constraint))

        self._obj = vmap(vmap(partial(cost)))

        # kernel = structured_rbf_kernel
        kernel = rbf_kernel
        self.K = vmap(kernel)
        self.dK = vmap(jacrev(kernel, argnums=0))
        self.x_max = torch.ones(self.dx + self.du)
        self.x_max[:3] = 6
        self.x_max[3:5] = 0.4 * torch.pi
        self.x_max[5] = 1000  # torch.pi
        self.x_max[6:12] = 1000
        self.x_max[12:] = 1000
        # self.x_max = 1000 * torch.ones(self.dx + self.du)
        self.x_min = -self.x_max

        self.obstacle_centre = torch.tensor([0.0, 0.0], device=self.device)
        self.obstacle_rad = 1.01

    def _objective(self, x, goals):
        J, dJ, HJ = self._obj(x, goals)
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

    def _height_constraint(self, trajectory, surface_z_points):
        """

        :param trajectory: T x 16
        :param surface_z_points: N x 1: points parameterizing surface GP
        :return constr: T
        :return grad_constr: T x 16
        :return hess_constr: T x 16 x 16
        """

        T = trajectory.shape[0]
        xy, z = trajectory[:, :2], trajectory[:, 2]

        # compute z of surface and gradient and hessian
        surface_z, grad_surface_z, hess_surface_z = self.surface_gp.posterior_mean(xy,
                                                                                   self.gp_x_points,
                                                                                   surface_z_points)

        constr = surface_z - z
        grad_constr = torch.cat((grad_surface_z,
                                 -torch.ones(T, 1, device=trajectory.device),
                                 torch.zeros(T, 13, device=trajectory.device)), dim=1)
        hess_constr = torch.cat((
            torch.cat((hess_surface_z, torch.zeros(T, 2, 14, device=trajectory.device)), dim=2),
            torch.zeros(T, 14, 16, device=trajectory.device)), dim=1)

        return constr, grad_constr, hess_constr

    def _con_eq(self, trajectory, starts, constraint_params, compute_grads=True):
        """

        :param trajectory: (B, N, T, dx + du)
        :param starts: (B, dx)
        :param compute_grads: (bool)
        :return g: (B, N, dg)
        :return grad_g: (B, N, dg, T * (dx + du))
        :return hess_g: (B, N, dg, T * (dx + du), T * (dx + du))

        """
        B, N = trajectory.shape[:2]

        # total problem dimension
        prob_dim = self.T * (self.dx + self.du)

        # dynamics constraint
        dynamics_constr = self.dynamics_constraint(trajectory).reshape(B, N, -1)

        # surface constraint
        surf_constr, grad_surf_constr, hess_surf_constr = self.height_constraint(trajectory,
                                                                                 constraint_params.reshape(B, 1, -1,
                                                                                                           1).repeat(1,
                                                                                                                     N,
                                                                                                                     1,
                                                                                                                     1)
                                                                                 )

        # start constraint
        start_constraint = (trajectory[:, :, 0, :12] - starts.reshape(B, 1, 12)).reshape(B, N, 12)

        g = torch.cat((dynamics_constr, start_constraint, surf_constr[:, :, 1:]), dim=2)

        if not compute_grads:
            return g, None, None

        grad_dynamics_constr = self.grad_dynamics(trajectory).reshape(B, N, -1, prob_dim)
        hess_dynamics_constr = self.hess_dynamics(trajectory).reshape(B, N, -1, prob_dim, prob_dim)

        # currently only take derivative of height at time t wrt state at time t - need to include other times
        # (all derivatives and hessians will be zero)
        grad_surf_constr = torch.diag_embed(grad_surf_constr.permute(0, 1, 3, 2))  # B x N x 16 x T x T
        grad_surf_constr = grad_surf_constr.permute(0, 1, 3, 4, 2).reshape(B, N, self.T, prob_dim)
        hess_surf_constr = torch.diag_embed(torch.diag_embed(hess_surf_constr.permute(0, 1, 3, 4, 2)))
        hess_surf_constr = hess_surf_constr.permute(0, 1, 4, 5, 2, 6, 3).reshape(B, N, self.T, prob_dim, prob_dim)

        grad_start_constraint = torch.zeros(B, N, 12, self.T, 16, device=trajectory.device)
        grad_start_constraint[:, :, :12, 0, :12] = torch.eye(12, device=trajectory.device)
        grad_start_constraint = grad_start_constraint.reshape(B, N, 12, -1)
        hess_start_constraint = torch.zeros(B, N, 12, self.T * 16, self.T * 16, device=trajectory.device)

        Dg = torch.cat((grad_dynamics_constr, grad_start_constraint, grad_surf_constr[:, :, 1:]), dim=2)
        DDg = torch.cat((hess_dynamics_constr, hess_start_constraint, hess_surf_constr[:, :, 1:]), dim=2)

        return g, Dg, DDg

    def _con_ineq(self, trajectory, *args, **kwargs):
        if not self.include_obstacle:
            return None, None, None

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

        hess_h_xy = -torch.eye(2, device=self.device).reshape(1, 1, 2, 2).repeat(N, self.T, 1, 1)
        # need to make (N, T, T, 2, T, 2)
        hess_h_xy = torch.diag_embed(torch.diag_embed(hess_h_xy.permute(0, 2, 3, 1)))  # now (N, 2, 2, T, T, T)
        hess_h_xy = hess_h_xy.permute(0, 3, 4, 1, 5, 2)  # now (N, T, T, 2, T, 2)
        hess_h = torch.zeros(N, self.T, self.T, self.dx + self.du, self.T, self.dx + self.du, device=self.device)
        hess_h[:, :, :, :2, :, :2] = hess_h_xy
        hess_h = hess_h.reshape(N, self.T, self.T * (self.dx + self.du),
                                self.T * (self.dx + self.du))
        return h, grad_h, hess_h

    def eval(self, augmented_trajectory, starts, goals, constraint_params):
        B, N = augmented_trajectory.shape[:2]

        trajectory = augmented_trajectory.reshape(B, N, self.T, -1)[:, :, :self.dx + self.du]
        cost, grad_cost, hess_cost = self._objective(trajectory, goals.reshape(B, 1, -1).repeat(1, N, 1))
        M = None

        grad_cost = torch.cat((grad_cost.reshape(B, N, self.T, -1),
                               torch.zeros(B, N, self.T, self.dz, device=trajectory.device)
                               ), dim=3).reshape(B, N, -1)

        # compute kernel and grad kernel
        if N > 1:
            Xk = trajectory  # .reshape(N, -1)
            K = self.K(Xk, Xk)
            grad_K = -self.dK(Xk, Xk).reshape(B, N, N, N, -1)
            grad_K = torch.einsum('bnmmi->bnmi', grad_K)
            grad_K = torch.cat((grad_K.reshape(B, N, N, self.T, self.dx + self.du),
                                torch.zeros(B, N, N, self.T, self.dz, device=trajectory.device)), dim=-1)
            grad_K = grad_K.reshape(B, N, N, -1)
        else:
            K = torch.ones(B, 1, 1, device=trajectory.device)
            grad_K = torch.zeros(B, 1, 1, self.dx + self.du + self.dz, device=trajectory.device)

        # Now we need to compute constraints and their first and second partial derivatives
        g, Dg, DDg = self.batched_combined_constraints(augmented_trajectory, starts, constraint_params)
        # print(cost.mean(), g.abs().mean())

        return cost.detach(), grad_cost.detach(), hess_cost, K.detach(), grad_K.detach(), g.detach(), Dg.detach(), DDg.detach()

    def update(self, start, goal, T):
        pass

    def get_initial_xu(self, N):
        pass


def test_model_multi(model, loader, problem):
    model.eval()
    val_ll = 0.0
    val_sample_loss = 0.0
    val_av_constraint_volation = 0.0

    for i, (trajectories, starts, goals, constraints, constraint_type) in enumerate(loader):
        trajectories = trajectories.to(device='cuda:0')
        B, T, dxu = trajectories.shape
        starts = starts.to(device='cuda:0')
        goals = goals.to(device='cuda:0')
        constraints = constraints.to(device='cuda:0')
        constraint_type = constraint_type.to(device='cuda:0').reshape(B)

        # make one hot
        constraint_type = torch.nn.functional.one_hot(constraint_type, num_classes=2).float()
        # constraint consists of parameters of constraint and constraint type
        c = torch.cat([constraints, constraint_type], dim=-1)

        with torch.no_grad():
            log_prob = model.loss(trajectories.reshape(B, T, dxu),
                                  starts.reshape(B, -1),
                                  goals.reshape(B, -1),
                                  c.reshape(B, -1))

        loss = -log_prob.mean()
        val_ll += loss.item()

        sampled_trajectories = model.sample(starts.reshape(B, -1),
                                            goals.reshape(B, -1),
                                            c.reshape(B, -1)).reshape(B, T, dxu)
        goals = torch.cat((goals, torch.zeros(B, 9).to(device='cuda:0')), dim=-1)
        # J, _, _, _, _, C, _, _ = problem.eval(sampled_trajectories, starts[:, 0], goals[:, 0], constraints[:, 0])
        J, _, _ = problem._objective(sampled_trajectories.unsqueeze(1), goals.unsqueeze(1))
        C, _, _ = problem.batched_combined_constraints(sampled_trajectories.unsqueeze(1), starts, constraints,
                                                       compute_grads=False)

        val_sample_loss += J.mean().item()
        val_av_constraint_volation += C[:, :, -11:].abs().mean().item()

    val_ll /= len(loader)
    val_sample_loss /= len(loader)
    val_av_constraint_volation /= len(loader)
    return val_ll, val_sample_loss, val_av_constraint_volation


def fine_tune_model_with_stein(model, train_loader, val_loader, problem, config):
    fpath = pathlib.Path(f'{CCAI_PATH}/data/training/quadrotor/{config["model_name"]}_{config["model_type"]}')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    epochs = config['epochs']
    model.train()
    pbar = tqdm.tqdm(range(epochs))
    for epoch in pbar:
        N = config['num_samples']
        for batch_no, (trajectories, starts, goals, constraints, _) in enumerate(train_loader):
            trajectories = trajectories.to(device=config['device'])
            B, T, dxu = trajectories.shape
            starts = starts.to(device=config['device'])
            goals = goals.to(device=config['device'])
            constraints = constraints.to(device=config['device'])

            # sample trajectories
            s = starts.reshape(B, 1, 12).repeat(1, N, 1).reshape(B * N, 12)
            goals = torch.cat((goals.reshape(N, 1, 3).repeat(1, N, 1), torch.zeros(B, N, 9).to(device='cuda:0')),
                              dim=-1)
            g = goals.reshape(B * N, 12)
            c = constraints.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
            trajectories = model.sample(s, g[:, :3], c).reshape(B, N, T, 16)
            phi, J, C = compute_constrained_gradient(trajectories,
                                                     starts,
                                                     goals,
                                                     constraints,
                                                     batched_problem,
                                                     alpha_J=1,
                                                     alpha_C=1)
            # Update plot
            phi = phi.reshape(B, N, batched_problem.T, -1)
            trajectories.backward(phi)

            if (batch_no + 1) % config['optim_update_every'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

                pbar.set_description(
                    f'Average Cost {J.mean().item():.3f} Average Constraint Violation {C.abs().mean().item():.3f}')
                optimizer.step()
                optimizer.zero_grad()
                # generate samples and plot them

        if (epoch + 1) % 10 == 0:
            demo_ll, sample_cost, constraint_violation = test_model_multi(model, val_loader, problem)
            print(
                f'Epoch: {epoch + 1}  Demonstration log-likelihood: {demo_ll}   Sample Cost: {sample_cost}   Constraint Violation: {constraint_violation}')
            # PLOT FIRST 16 EXAMPLES
            for i, (_, starts, goals, constraints) in enumerate(val_loader):
                starts = starts.to(device='cuda:0')
                goals = goals.to(device='cuda:0')
                N = starts.shape[1]
                starts = starts[:16].reshape(-1, 12)
                goals = goals[:16].reshape(-1, 3)
                constraints = constraints[:16].reshape(-1, 100).to(device='cuda:0')
                with torch.no_grad():
                    trajectories = model.sample(starts, goals, constraints)
                trajectories = trajectories.reshape(16, N, 12, 16)
                starts = starts.reshape(16, N, 12)
                goals = goals.reshape(16, N, 3)
                constraints = constraints.reshape(16, N, 100)
                for i, trajs in enumerate(trajectories):
                    env = QuadrotorEnv('surface_data.npz')
                    xy_data = env.surface_model.train_x.cpu().numpy()
                    z_data = constraints[i, 0].cpu().numpy()
                    np.savez('tmp_surface_data.npz', xy=xy_data, z=z_data)
                    env = QuadrotorEnv('tmp_surface_data.npz')
                    env.state = starts[i, 0].cpu().numpy()
                    env.goal = goals[i, 0].cpu().numpy()
                    ax = env.render_init()
                    update_plot_with_trajectories(ax, trajs.reshape(N, 12, 16))
                    plt.savefig(
                        f'{fpath}_finetuned_{epoch}_{i}.png')
                    plt.close()

                break

    torch.save(model.state_dict(), f'{fpath}_finetuned.pt')


def update_plot_with_trajectories(ax, trajectory, color='g'):
    traj_lines = []
    for traj in trajectory:
        traj_np = traj.detach().cpu().numpy()
        traj_lines.extend(ax.plot(traj_np[1:, 0],
                                  traj_np[1:, 1],
                                  traj_np[1:, 2], color=color, alpha=0.5, linestyle='--'))


def train_model_from_demonstrations(trajectory_sampler, train_loader, val_loader, problem, config):
    fpath = f'{CCAI_PATH}/data/training/quadrotor/{config["model_name"]}_{config["model_type"]}'

    if config['use_ema']:
        ema = EMA(beta=config['ema_decay'])
        ema_model = copy.deepcopy(trajectory_sampler)

    def reset_parameters():
        ema_model.load_state_dict(trajectory_sampler.state_dict())

    def update_ema(model):
        if step < config['ema_warmup_steps']:
            reset_parameters()
        else:
            ema.update_model_average(ema_model, model)

    optimizer = torch.optim.Adam(trajectory_sampler.parameters(), lr=config['lr'])

    step = 0
    start_epoch = 0

    if config['load_checkpoint']:
        checkpoint = torch.load(f'{fpath}/checkpoint.pth')
        trajectory_sampler.load_state_dict(checkpoint['model'])
        if config['use_ema']:
            ema_model.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        step = checkpoint['step']

    pbar = tqdm.tqdm(range(config['epochs']), initial=start_epoch)

    for epoch in pbar:
        train_loss = 0.0
        trajectory_sampler.train()
        for trajectories, starts, goals, constraints, constraint_type in train_loader:
            trajectories = trajectories.to(device='cuda:0')
            B, T, dxu = trajectories.shape
            starts = starts.to(device='cuda:0')
            goals = goals.to(device='cuda:0')
            constraints = constraints.to(device='cuda:0')
            constraint_type = constraint_type.to(device='cuda:0').reshape(B)

            # make one hot
            constraint_type = torch.nn.functional.one_hot(constraint_type, num_classes=2).float()
            # constraint consists of parameters of constraint and constraint type
            constraints = torch.cat([constraints, constraint_type], dim=-1)
            sampler_loss = trajectory_sampler.loss(trajectories,
                                                   starts,
                                                   goals,
                                                   constraints)
            loss = sampler_loss
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            step += 1
            if config['use_ema']:
                if step % config['ema_update_every'] == 0:
                    update_ema(trajectory_sampler)

        train_loss /= len(train_loader)
        pbar.set_description(
            f'Train loss {train_loss:.3f}')

        # generate samples and plot them
        if (epoch + 1) % config['test_every'] == 0:
            if config['use_ema']:
                test_model = ema_model
            else:
                test_model = trajectory_sampler
            """
            demo_ll, sample_cost, constraint_violation = test_model_multi(test_model,
                                                                          train_loader,
                                                                          problem)
            print('TRAINING')
            print(
                f'Epoch: {epoch + 1}  Demonstration loss: {demo_ll}  Sample Cost: {sample_cost} Constraint Violation: {constraint_violation}')
            demo_ll, sample_cost, constraint_violation = test_model_multi(test_model,
                                                                          val_loader,
                                                                          problem)
            print('VALIDATION')
            print(
                f'Epoch: {epoch + 1}  Demonstration loss: {demo_ll}  Sample Cost: {sample_cost} Constraint Violation: {constraint_violation}')
            """
            for trajectories, starts, goals, constraints, constraint_type in val_loader:
                starts = starts.to(device='cuda:0')
                goals = goals.to(device='cuda:0')
                N = 16  # trajectories.shape[1]
                B = 9
                starts = starts[:B].reshape(-1, 12)
                goals = goals[:B].reshape(-1, 3)
                true_trajectories = trajectories[:B].to(device='cuda:0')
                constraints = constraints[:B].reshape(-1, 100).to(device='cuda:0')
                constraint_type = constraint_type[:B].to(device='cuda:0').reshape(B)

                # make one hot
                constraint_type = torch.nn.functional.one_hot(constraint_type, num_classes=2).float()
                s = starts.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
                g = goals.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
                c = constraints.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
                ctype = constraint_type.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
                c = torch.cat([c, ctype], dim=1)
                with torch.no_grad():
                    trajectories = trajectory_sampler.sample(s, g, c)

                trajectories = trajectories.reshape(B, N, 12, 16)
                starts = s.reshape(B, N, 12)

                # unnormalize starts
                starts = starts * trajectory_sampler.x_std[:12] + trajectory_sampler.x_mean[:12]
                true_trajectories = true_trajectories * trajectory_sampler.x_std + trajectory_sampler.x_mean

                goals = g.reshape(B, N, 3)
                constraints = constraints.reshape(B, 100)

                for i, trajs in enumerate(trajectories):
                    env = QuadrotorEnv('surface_data.npz')
                    xy_data = env.surface_model.train_x.cpu().numpy()
                    z_data = constraints[i].cpu().numpy()
                    np.savez('tmp_surface_data.npz', xy=xy_data, z=z_data)

                    if constraint_type[i, 0] == 1:
                        env = QuadrotorEnv(randomize_GP=False, surface_data_fname='tmp_surface_data.npz')
                    else:
                        env = QuadrotorEnv(randomize_GP=False, obstacle_data_fname='tmp_surface_data.npz',
                                           obstacle_mode='gp')

                    env.state = starts[i, 0].cpu().numpy()
                    env.goal = goals[i, 0].cpu().numpy()
                    ax = env.render_init()
                    update_plot_with_trajectories(ax, trajs.reshape(N, 12, 16))
                    update_plot_with_trajectories(ax, true_trajectories[i].reshape(1, 12, 16), color='red')
                    plt.savefig(
                        f'{fpath}/from_demonstrations_{epoch}_{i}.png')
                plt.close()

                break
        if (epoch + 1) % config['save_every'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model': trajectory_sampler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema_model.state_dict(),
                'step': step,
            }
            torch.save(checkpoint, f'{fpath}/checkpoint.pth')

    if config['use_ema']:
        torch.save(ema_model.state_dict(), f'{fpath}/from_demonstrations_ema.pt')
    else:
        torch.save(model.state_dict(),
                   f'{fpath}/from_demonstrations_.pt')


if __name__ == "__main__":
    # load config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/config/training_configs/quadrotor_diffusion.yaml').read_text())
    torch.set_float32_matmul_precision('high')

    # make path for saving model and plots
    fpath = pathlib.Path(f'{CCAI_PATH}/data/training/quadrotor/{config["model_name"]}_{config["model_type"]}')
    pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)

    batched_problem = QuadrotorProblem(T=12, device=config['device'], include_obstacle=False, alpha=0.01)
    model = TrajectorySampler(T=12, dx=12, du=4,
                              context_dim=12 + 3 + 100 + 2,
                              type=config['model_type'],
                              dynamics=Quadrotor12DDynamics(dt=0.1))

    # Load data
    from torch.utils.data import DataLoader, RandomSampler

    data_path = pathlib.Path(f'{CCAI_PATH}/data/training_data/{config["data_directory"]}')
    train_dataset = QuadrotorMultiConstraintTrajectoryDataset([p for p in data_path.glob('*train_data*')])
    val_dataset = QuadrotorMultiConstraintTrajectoryDataset([p for p in data_path.glob('*test_data*')])

    # Get Normalization Constants
    if config['normalize_data']:
        train_dataset.compute_norm_constants()
        val_dataset.set_norm_constants(*train_dataset.get_norm_constants())
        model.set_norm_constants(*train_dataset.get_norm_constants())

    train_sampler = RandomSampler(train_dataset)
    val_sampler = RandomSampler(val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=256)

    if config['load_model'] != 'none':
        model.load_state_dict(torch.load(config['load_model']))
    model.to(device=config['device'])

    if config['use_csvto_gradient']:
        fine_tune_model_with_stein(model, train_loader, val_loader, batched_problem, config)
    else:
        train_model_from_demonstrations(model, train_loader, val_loader, batched_problem, config)
