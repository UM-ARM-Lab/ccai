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
from quadrotor_example import QuadrotorProblem

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


def test_samples(trajectory_sampler, config):
    fpath = f'{CCAI_PATH}/data/training/quadrotor/{config["model_name"]}_{config["model_type"]}'
    data_path = pathlib.Path(f'{CCAI_PATH}/data/test_data/{config["test_directory"]}')
    print(data_path)
    fpaths = [str(d) for d in data_path.glob('*csvgd')]
    print(fpaths)
    goals = []
    surface = []
    obj = []
    starts = []

    if config['load_checkpoint']:
        checkpoint = torch.load(f'{fpath}/checkpoint.pth')
        trajectory_sampler.load_state_dict(checkpoint['model'])
        if config['use_ema']:
            trajectory_sampler.load_state_dict(checkpoint['ema'])

    trajectory_sampler = trajectory_sampler.to(device=config['device'])
    trajectory_sampler.send_norm_constants_to_submodels()
    object_ids = []

    for fpath in fpaths:
        path = pathlib.Path(fpath)

        for p in path.rglob('*trajectory_data.npz'):
            data = np.load(p, allow_pickle=True)
            goal = data['goal']
            start = data['traj'][:, 0, 0, :12]
            goals.append(goal.copy())
            starts.append(start.copy())
            surface.append(data['surface'])
            obj.append(data['obstacle'])

    # create tensors and snd to gpu
    starts = torch.from_numpy(np.stack(starts, axis=0)).to(device=config['device'], dtype=torch.float32)
    goals = torch.from_numpy(np.stack(goals, axis=0)).to(device=config['device'], dtype=torch.float32)
    surfaces = torch.from_numpy(np.stack(surface, axis=0)).to(device=config['device'], dtype=torch.float32)
    objs = torch.from_numpy(np.stack(obj, axis=0)).to(device=config['device'], dtype=torch.float32)

    B1, B2, _ = starts.shape
    B2 = 10

    starts = starts[:, :B2]

    N = 16

    constraints = torch.stack((surfaces, objs), dim=1)
    constraint_codes = torch.stack((torch.tensor([1.0, 0.0], device=config['device']),
                                    torch.tensor([0.0, 1.0], device=config['device'])), dim=0).expand(B1, -1, -1)
    constraints = torch.cat((constraints, constraint_codes), dim=2)

    from ccai.quadrotor_env import QuadrotorEnv

    if not config['surf']:
        constraints = constraints[:, 1, :].unsqueeze(1)
    elif not config['obj']:
        constraints = constraints[:, 0, :].unsqueeze(1)

    st = starts.reshape(B1, B2, 1, -1).expand(-1, -1, N, -1)
    gl = goals.reshape(B1, 1, 1, -1).expand(-1, B2, N, -1)
    cn = constraints.reshape(B1, 1, 1, -1, 102).expand(-1, B2, N, -1, -1)

    costs = torch.zeros((B1, B2, N))
    av_g = torch.zeros((B1, B2, N))
    av_h = torch.zeros((B1, B2, N))
    M = 10
    xs = torch.linspace(-5, 5, steps=M)
    ys = torch.linspace(-5, 5, steps=M)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    xy = torch.stack((x.flatten(), y.flatten()), dim=1)

    goals_extended = torch.zeros(12, device=config['device'])
    for b1 in range(B1):
        np.savez('tmp_surf.npz', z=surface[b1], xy=xy.numpy())
        np.savez('tmp_obs.npz', z=obj[b1], xy=xy.numpy())

        for b2 in range(B2):
            print(b1, b2)
            # need to sample individually for each problem due to guided diffusion
            env = QuadrotorEnv(surface_data_fname='tmp_surf.npz',
                               obstacle_mode='gp',
                               surface_constraint=True,
                               obstacle_data_fname='tmp_obs.npz')

            goals_extended[:3] = goals[b1]
            problem = QuadrotorProblem(starts[b1, b2],
                                       goals_extended,
                                       12, device=config['device'],
                                       include_obstacle=True,
                                       gp_surface_model=env.surface_model, gp_sdf_model=env.obstacle_model,
                                       alpha=1)

            trajectory_sampler.model.diffusion_model.problem = problem
            samples = trajectory_sampler.sample(st[b1, b2], gl[b1, b2], cn[b1, b2])
            samples = samples.reshape(N, -1, 16)
            # evalaute cost
            J, _, _ = problem._objective(samples)
            g, _, _ = problem._con_ineq(samples, compute_grads=False)
            h, _, _ = problem._con_eq(samples, compute_grads=False)


            # collect together all costs and constraint violations
            costs[b1, b2, :] = J.squeeze(1)
            av_h[b1, b2, :] = torch.mean(torch.abs(h), dim=-1)
            av_g[b1, b2, :] = torch.mean(torch.clamp(g, min=0), dim=-1)

            print(J.mean(), h.abs().mean())

        print(costs[:b1+1].mean(), costs[:b1+1].std())
        print('inequality')
        print(av_g[:b1+1].mean(), av_g[:b1+1].std())
        print('equality')
        print(av_h[:b1+1].mean(), av_h[:b1+1].std())

    # now we can save the results, including object ids for identifying diffferent environments
    np.savez(f'{CCAI_PATH}/data/quadrotor_{config["test_name"]}.npz',
             costs=costs.cpu().numpy(),
             av_h=av_h.cpu().numpy(),
             av_g=av_g.cpu().numpy(),
             object_ids=object_ids)


if __name__ == "__main__":
    # load config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/config/training_configs/quadrotor_diffusion.yaml').read_text())
    torch.set_float32_matmul_precision('high')

    # make path for saving model and plots
    fpath = pathlib.Path(f'{CCAI_PATH}/data/training/quadrotor/{config["model_name"]}_{config["model_type"]}')
    pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)

    if config['constrained'] or config['guided']:
        problem = QuadrotorProblem(T=12, start=torch.zeros(12, device=config['device']),
                                   goal=torch.zeros(12, device=config['device']),
                                   device=config['device'], include_obstacle=True, alpha=0.1)
    else:
        problem = None

    model = TrajectorySampler(T=12, dx=12, du=4,
                              context_dim=12 + 3 + 100 + 2,
                              type=config['model_type'],
                              dynamics=Quadrotor12DDynamics(dt=0.1),
                              hidden_dim=config['hidden_dim'],
                              timesteps=config['timesteps'],
                              problem=problem,
                              constrain=config['constrained'],
                              unconditional=config['unconditional'])

    if config['test_samples']:
        test_samples(model, config)
    else:
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

