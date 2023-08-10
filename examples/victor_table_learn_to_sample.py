import torch
import numpy as np
from torch import optim
import pathlib
import tqdm
import matplotlib.pyplot as plt
from ccai.models.training import EMA

from quadrotor_learn_to_sample import TrajectorySampler
from ccai.dataset import VictorTableMultiConstraintTrajectoryDataset
import pytorch_kinematics as pk
import yaml
import copy

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
asset = '/home/tpower/dev/isaac_test/IsaacVictorEnvs/isaac_victor_envs/assets/victor/victor.urdf'
ee_name = 'victor_left_arm_striker_mallet_tip'
chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name).to(device='cuda:0')


def data_for_cylinder_along_z(center_x, center_y, centre_z, radius, height_z):
    z = np.linspace(centre_z - height_z / 2, centre_z + height_z / 2, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


def train_model(trajectory_sampler, train_loader, val_loader, config):
    fpath = f'{CCAI_PATH}/data/training/victor_table/{config["model_name"]}_{config["model_type"]}'
    pathlib.Path.mkdir(pathlib.Path(fpath), parents=True, exist_ok=True)

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

    epochs = config['epochs']
    pbar = tqdm.tqdm(range(epochs))
    for epoch in pbar:
        train_loss = 0.0
        trajectory_sampler.train()
        for trajectories, starts, goals, constraints, constraint_type in train_loader:
            trajectories = trajectories.to(device=config['device'])
            B, T, dxu = trajectories.shape
            starts = starts.to(device=config['device'])
            goals = goals.to(device=config['device'])
            constraints = constraints.to(device=config['device'])
            constraint_type = constraint_type.to(device='cuda:0').reshape(B)
            constraint_type = torch.nn.functional.one_hot(constraint_type, num_classes=2).float()
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
                if step % 10 == 0:
                    update_ema(trajectory_sampler)

        train_loss /= len(train_loader)
        pbar.set_description(
            f'Train loss {train_loss:.3f}')

        # generate samples and plot them
        if (epoch + 1) % config['test_every'] == 0:
            count = 0
            for trajectories, starts, goals, constraints, constraint_type in val_loader:
                if config['use_ema']:
                    test_model = ema_model
                else:
                    test_model = trajectory_sampler
                count += 1
                starts = starts.to(device='cuda:0')
                goals = goals.to(device='cuda:0')
                N = 16  # trajectories.shape[1]
                B = 9
                starts = starts[:B].reshape(-1, 7)
                goals = goals[:B].reshape(-1, 2)
                true_trajectories = trajectories[:B].to(device='cuda:0')
                constraints = constraints[:B].reshape(-1, 4).to(device='cuda:0')
                constraint_type = constraint_type[:B].to(device='cuda:0').reshape(B)
                constraint_type = torch.nn.functional.one_hot(constraint_type, num_classes=2).float()

                s = starts.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
                g = goals.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
                c = constraints.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
                ctype = constraint_type.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
                c = torch.cat([c, ctype], dim=1).unsqueeze(1)
                with torch.no_grad():
                    trajectories = test_model.sample(s, g, c)
                # unnormalize starts
                starts = s * test_model.x_std + test_model.x_mean
                starts = starts.reshape(B, N, 1, 7)

                # unnormalize true trajectories
                true_trajectories = true_trajectories * test_model.x_std + test_model.x_mean
                true_trajectories = true_trajectories.reshape(B, 12, 7)
                true_trajectories = torch.cat([starts[:, 0], true_trajectories], dim=1)

                # concatenate starts and trajectories
                trajectories = trajectories.reshape(B, N, 12, 7)
                trajectories = torch.cat([starts, trajectories], dim=2)

                mat = chain.forward_kinematics(trajectories.reshape(B * N * 13, -1))
                ee_trajectories = mat[:, :3, 3].reshape(B, N, 13, 3)
                mat = chain.forward_kinematics(true_trajectories.reshape(B * 13, -1))
                true_ee_trajectories = mat[:, :3, 3].reshape(B, 13, 3)

                goals = g.reshape(B, N, 2)
                constraints = constraints.reshape(B, -1)

                fig, axes = plt.subplots(3, 3, figsize=(12, 12), subplot_kw=dict(projection='3d'))
                axes = axes.flatten()
                for i, (s_ee, t_ee, g) in enumerate(zip(ee_trajectories,
                                                              true_ee_trajectories,
                                                              goals)):
                    s_ee = s_ee.cpu().numpy()
                    t_ee = t_ee.cpu().numpy()
                    if constraint_type[i, 0] == 1:
                        h = constraints[i, 0].cpu().numpy()
                        # plot the table
                        xx, yy = np.meshgrid(np.linspace(0.4, 1.0, 10), np.linspace(0.1, 0.8, 10))
                        zz = np.ones_like(xx) * h
                        axes[i].plot_surface(xx, yy, zz, alpha=0.25, color='k')
                    else:
                        c = constraints[i].reshape(2, 2).cpu().numpy()
                        h = t_ee[0, 2]
                        # add 3D cylinder objects to plot
                        cylinder_1 = data_for_cylinder_along_z(c[0, 0], c[0, 1], h, 0.05, 0.2)
                        cylinder_2 = data_for_cylinder_along_z(c[1, 0], c[1, 1], h, 0.05, 0.2)
                        axes[i].plot_surface(*cylinder_1, alpha=0.5, color='r')
                        axes[i].plot_surface(*cylinder_2, alpha=0.5, color='r')

                    g = g.cpu().numpy()
                    # circle1 = plt.Circle((c[0, 0], c[0, 1]), 0.1, color='r')
                    # circle2 = plt.Circle((c[1, 0], c[1, 1]), 0.1, color='r')
                    # axes[i].add_patch(circle1)
                    # axes[i].add_patch(circle2)
                    goal_z = np.ones_like(s_ee[:, 0]) * h
                    axes[i].plot(t_ee[:, 0], t_ee[:, 1], t_ee[:, 2], c='r')
                    axes[i].scatter(g[0, 0], g[0, 1], goal_z, s=50, c='g')
                    axes[i].scatter(t_ee[0, 0], t_ee[0, 1], t_ee[0, 2], s=50, c='k')
                    for j in range(8):
                        axes[i].plot(s_ee[j, :, 0], s_ee[j, :, 1], s_ee[j, :, 2], c='b', alpha=0.25)
                        # axes[i].scatter(s[:, 0], s[:, 1], c=h, cmap='viridis', s=10)
                    axes[i].view_init(azim=-60, elev=-135)
                    axes[i].set_xlabel('x')
                    axes[i].set_ylabel('y')
                    if constraint_type[i, 0] == 1:
                        axes[i].set_zlim(h - 0.15, h + 0.15)
                    axes[i].grid(False)
                plt.tight_layout()
                plt.savefig(
                    f'{fpath}/{config["model_type"]}_{epoch}.png')
                plt.close()
                break

    if config['use_ema']:
        torch.save(ema_model.state_dict(), f'{fpath}/victor_table_{config["model_type"]}.pt')
    else:
        torch.save(model.state_dict(),
                   f'{fpath}/victor_table_{config["model_type"]}.pt')


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    config = yaml.safe_load(
        pathlib.Path(f'{CCAI_PATH}/config/training_configs/victor_table_diffusion.yaml').read_text())
    model = TrajectorySampler(T=12, dx=7, du=0, context_dim=7 + 2 + 4 + 2, type=config['model_type'])
    from torch.utils.data import DataLoader, RandomSampler

    data_path = pathlib.Path(f'{CCAI_PATH}/data/training_data/{config["data_directory"]}')
    train_dataset = VictorTableMultiConstraintTrajectoryDataset([p for p in data_path.glob('*train_data*')])
    val_dataset = VictorTableMultiConstraintTrajectoryDataset([p for p in data_path.glob('*test_data*')])

    if config['normalize_data']:
        # normalize data
        train_dataset.compute_norm_constants()
        val_dataset.set_norm_constants(*train_dataset.get_norm_constants())
        model.set_norm_constants(*train_dataset.get_norm_constants())

    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4)

    model = model.to(device=config['device'])
    train_model(model, train_loader, val_loader, config)
