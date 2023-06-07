import torch
import numpy as np
from torch import optim
import pathlib
import tqdm
import matplotlib.pyplot as plt
from ccai.models.diffusion.training import EMA

from quadrotor_learn_to_sample import TrajectorySampler
from ccai.dataset import VictorTableMultiConstraintTrajectoryDataset
import pytorch_kinematics as pk

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
asset = '/home/tpower/dev/isaac_test/IsaacVictorEnvs/isaac_victor_envs/assets/victor/victor.urdf'
ee_name = 'victor_left_arm_striker_mallet_tip'
chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name).to(device='cuda:0')


def data_for_cylinder_along_z(center_x, center_y, centre_z, radius, height_z):
    z = np.linspace(centre_z - height_z/2, centre_z + height_z/2, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


def train_cnf(trajectory_sampler, train_loader, val_loader):
    ema = EMA(beta=0.995)
    import copy
    ema_model = copy.deepcopy(trajectory_sampler)

    step = 0
    warmup_steps = 1000

    def reset_parameters():
        ema_model.load_state_dict(trajectory_sampler.state_dict())

    def update_ema(model):
        if step < warmup_steps:
            reset_parameters()
        else:
            ema.update_model_average(ema_model, model)

    optimizer = optim.Adam(trajectory_sampler.parameters(), lr=1e-4)

    epochs = 1
    pbar = tqdm.tqdm(range(epochs))
    for epoch in pbar:
        train_loss = 0.0
        trajectory_sampler.train()
        for trajectories, starts, goals, constraints, constraint_idx in train_loader:
            break
            trajectories = trajectories.to(device='cuda:0')
            B, T, dxu = trajectories.shape
            starts = starts.to(device='cuda:0')
            goals = goals.to(device='cuda:0')
            constraints = constraints.to(device='cuda:0')

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
            if step % 10 == 0:
                update_ema(trajectory_sampler)
        train_loss /= len(train_loader)
        pbar.set_description(
            f'Train loss {train_loss:.3f}')

        # generate samples and plot them
        if (epoch + 1) % 1 == 0:
            count = 0
            for trajectories, starts, goals, constraints, constraint_idx in val_loader:
                count += 1
                if count < 4:
                    continue
                starts = starts.to(device='cuda:0')
                goals = goals.to(device='cuda:0')
                N = 16  # trajectories.shape[1]
                B = 9
                starts = starts[:B].reshape(-1, 7)
                goals = goals[:B].reshape(-1, 2)
                true_trajectories = trajectories[:B].to(device='cuda:0')
                constraints = constraints[:B].reshape(-1, 5).to(device='cuda:0')
                constraint_idx = constraint_idx[:B].reshape(-1).to(device='cuda:0')

                s = starts.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
                g = goals.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
                c = constraints.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
                with torch.no_grad():
                    trajectories = ema_model.sample(s, g, c)
                # unnormalize starts
                starts = s * ema_model.x_std + ema_model.x_mean
                starts = starts.reshape(B, N, 1, 7)

                # unnormalize true trajectories
                true_trajectories = true_trajectories * ema_model.x_std + ema_model.x_mean
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
                height = constraints[:, 0]
                height_error = ee_trajectories[:, :, :, 2] - height[:, None, None]
                print(height_error.abs().mean())
                centres = constraints[:, 1:].reshape(B, 2, 2)

                fig, axes = plt.subplots(3, 3, figsize=(12, 12), subplot_kw=dict(projection='3d'))
                axes = axes.flatten()
                for i, (s_ee, t_ee, h, c, g) in enumerate(zip(ee_trajectories,
                                                                true_ee_trajectories,
                                                                height,
                                                                centres, goals)):
                    s_ee = s_ee.cpu().numpy()
                    t_ee = t_ee.cpu().numpy()
                    h = h.cpu().numpy()
                    c = c.cpu().numpy()
                    g = g.cpu().numpy()
                    # add 3D cylinder objects to plot
                    cylinder_1 = data_for_cylinder_along_z(c[0, 0], c[0, 1], h, 0.05, 0.1)
                    cylinder_2 = data_for_cylinder_along_z(c[1, 0], c[1, 1], h, 0.05, 0.1)
                    axes[i].plot_surface(*cylinder_1, alpha=0.5, color='r')
                    axes[i].plot_surface(*cylinder_2, alpha=0.5, color='r')

                    # plot the table
                    xx, yy = np.meshgrid(np.linspace(0.4, 1.0, 10), np.linspace(0.1, 0.8, 10))
                    zz = np.ones_like(xx) * h
                    axes[i].plot_surface(xx, yy, zz, alpha=0.25, color='k')
                    #circle1 = plt.Circle((c[0, 0], c[0, 1]), 0.1, color='r')
                    #circle2 = plt.Circle((c[1, 0], c[1, 1]), 0.1, color='r')
                    #axes[i].add_patch(circle1)
                    #axes[i].add_patch(circle2)
                    goal_z = np.ones_like(s_ee[:, 0]) * h
                    axes[i].plot(t_ee[:, 0], t_ee[:, 1], t_ee[:, 2], c='r')
                    axes[i].scatter(g[0, 0], g[0, 1], goal_z, s=50, c='g')
                    axes[i].scatter(t_ee[0, 0], t_ee[0, 1], t_ee[0, 2], s=50, c='k')
                    for j in range(8):
                        axes[i].plot(s_ee[j, :, 0], s_ee[j, :, 1], s_ee[j, :, 2], c='b', alpha=0.25)
                        #axes[i].scatter(s[:, 0], s[:, 1], c=h, cmap='viridis', s=10)
                    axes[i].view_init(azim=-60, elev=-135)
                    axes[i].set_xlabel('x')
                    axes[i].set_ylabel('y')
                    axes[i].set_zlim(h-0.15, h+0.15)
                    axes[i].grid(False)
                plt.tight_layout()
                plt.savefig(f'learning_plots/victor_table/{ema_model.type}/{epoch}.png')
                plt.close()
                break

        #torch.save(ema_model.state_dict(), f'victor_table_{ema_model.type}.pt')



if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    type='cnf'
    model = TrajectorySampler(T=12, dx=7, du=0, context_dim=7 + 2 + 5, type=type)
    from torch.utils.data import DataLoader, RandomSampler

    train_dataset = VictorTableMultiConstraintTrajectoryDataset(
        [f'../data/experiments/victor_table_jointspace_train_data_new{i + 1}' for i in range(6)])
    val_dataset = VictorTableMultiConstraintTrajectoryDataset(
        ['../data/experiments/victor_table_jointspace_test_data_new'])

    # normalize data
    train_dataset.compute_norm_constants()
    val_dataset.set_norm_constants(*train_dataset.get_norm_constants())
    model.set_norm_constants(*train_dataset.get_norm_constants())

    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=256, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # train_diffusion_from_demonstrations(train_loader, val_loader, batched_problem)
    # from ccai.cgm.cgm import ConstraintGenerativeModel
    model = model.to(device='cuda:0')
    model.load_state_dict(torch.load(f'victor_table_{type}.pt'))
    train_cnf(model, train_loader, val_loader)
