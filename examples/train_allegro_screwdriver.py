# from isaac_victor_envs.tasks.allegro import AllegroValveTurningEnv

import yaml
import copy
import tqdm
import torch
import pathlib
import numpy as np
import open3d as o3d
import pytorch_kinematics as pk
import pytorch_volumetric as pv
import matplotlib.pyplot as plt
from ccai.models.training import EMA
import pytorch_kinematics.transforms as tf
from ccai.dataset import AllegroScrewDriverDataset
from isaac_victor_envs.utils import get_assets_dir
from torch.utils.data import DataLoader, RandomSampler
from ccai.models.trajectory_samplers import TrajectorySampler
from utils.allegro_utils import partial_to_full_state, visualize_trajectory

fingers = ['index', 'middle', 'thumb']


def visualize_trajectories(trajectories, scene, fpath, headless=False):
    for n, trajectory in enumerate(trajectories):

        trajectory = trajectory[:, :16]
        trajectory[:, 15] *= 0
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/kin/img'), parents=True, exist_ok=True)
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/kin/gif'), parents=True, exist_ok=True)
        visualize_trajectory(trajectory, scene, f'{fpath}/trajectory_{n + 1}/kin', headless=headless,
                             fingers=fingers, obj_dof=4)
        # Visualize what happens if we execute the actions in the trajectory in the simulator
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/sim/img'), parents=True, exist_ok=True)
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/sim/gif'), parents=True, exist_ok=True)
        # visualize_trajectory_in_sim(trajectory, config['env'], f'{fpath}/trajectory_{n + 1}/sim')
        # save the trajectory
        np.save(f'{fpath}/trajectory_{n + 1}/traj.npz', trajectory.cpu().numpy())


def train_model(trajectory_sampler, train_loader, config):
    fpath = f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}'
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
        for trajectories, traj_class, masks in train_loader:
            trajectories = trajectories.to(device=config['device'])
            masks = masks.to(device=config['device'])
            B, T, dxu = trajectories.shape
            if config['use_class']:
                traj_class = traj_class.to(device=config['device']).float()
            else:
                traj_class = None
            sampler_loss = trajectory_sampler.loss(trajectories, mask=masks, constraints=traj_class)
            loss = sampler_loss
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # for param in trajectory_sampler.parameters():
            #    print(param.grad)
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
            if config['use_ema']:
                test_model = ema_model
            else:
                test_model = trajectory_sampler

            # we will plot for a variety of different horizons
            N = 8
            min_horizon = 16
            max_horizon = 32

            start = (trajectories[0, 0, :15] * train_dataset.std[:15].to(device=trajectories.device) +
                     train_dataset.mean[:15].to(device=trajectories.device))
            start = start[None, :].repeat(N, 1)

            for H in range(min_horizon, max_horizon + 1, 16):
                plot_fpath = f'{fpath}/epoch_{epoch + 1}/horizon_{H}'
                pathlib.Path.mkdir(pathlib.Path(plot_fpath), parents=True, exist_ok=True)
                sampled_trajectories, sampled_contexts, likelihoods = test_model.sample(N, H=H, start=start)
                visualize_trajectories(sampled_trajectories, config['scene'], plot_fpath, headless=False)
            if config['use_ema']:
                torch.save(ema_model.state_dict(), f'{fpath}/allegro_screwdriver_{config["model_type"]}.pt')
            else:
                torch.save(model.state_dict(),
                           f'{fpath}/allegro_screwdriver_{config["model_type"]}.pt')
    if config['use_ema']:
        torch.save(ema_model.state_dict(), f'{fpath}/allegro_screwdriver_{config["model_type"]}.pt')
    else:
        torch.save(model.state_dict(),
                   f'{fpath}/allegro_screwdriver_{config["model_type"]}.pt')
#
#
# def test_long_horizon(model, loader, config):
#     fpath = f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/long_w_start'
#     pathlib.Path.mkdir(pathlib.Path(fpath), parents=True, exist_ok=True)
#     N = 16
#     if config['sine_cosine']:
#         theta = 2 * np.pi * (torch.rand(1, device=config['device']) - 0.5)
#         start = torch.tensor([0.2, 0.5, 0.7, 0.7, 1.3, 0, 0.1, 1.0, torch.cos(theta), torch.sin(theta)],
#                              device=config['device'])
#     else:
#         start = torch.tensor([0.2, 0.5, 0.7, 0.7, 1.3, 0, 0.1, 1.0, 1.0, 0.0], device=config['device'])
#         theta = 2 * np.pi * (torch.rand(1, device=config['device']) - 0.5)
#         start[-2] = torch.cos(theta)
#         start[-1] = torch.sin(theta)
#     num_turns = 2
#     if config['use_class']:
#         context = torch.tensor([1.0, -1.0], device=config['device']).repeat(num_turns)
#         context = context.reshape(1, -1, 1).repeat(N, 1, 1)
#     else:
#         context = None
#
#     goal = 0.5 * torch.tensor([-np.pi / 2.0], device=config['device'])
#     start = start[None, :].repeat(N, 1)
#     # start = None
#     # goal = (goal[None, :].repeat(N, 1) - model.x_mean[8]) / model.x_std[8]
#     # start = None
#     context = None
#     # context = torch.randn(N, 1, device=config['device'])
#     # context = torch.where(context > 0, torch.ones_like(context), -torch.ones_like(context))
#     # print(context)
#     goal = None
#     # goal = None
#     # print(model.x_mean[:8])
#     # print(model.x_std[:8])
#
#     # for item in loader:
#     #    print(item[0, 0, :8].cuda() * model.x_std[:8] + model.x_mean[:8] - start[0, :8])
#
#     # exit(0)
#     print(context)
#     sampled_trajectories, sampled_contexts, likelihoods = model.sample(N=N, start=start, goal=goal,
#                                                                        H=int(num_turns * 32), constraints=context)
#     print(likelihoods)
#     # exit(0)
#     if config['sine_cosine']:
#         eps = 1e-6
#         cos_theta = sampled_trajectories[:, :, 8].clamp(min=-1 + eps, max=1 - eps)
#         sin_theta = sampled_trajectories[:, :, 9].clamp(min=-1 + eps, max=1 - eps)
#         theta = torch.atan2(sin_theta, cos_theta)
#         sampled_trajectories = torch.cat((
#             sampled_trajectories[:, :, :8], theta[:, :, None], sampled_trajectories[:, :, -8:]
#         ), dim=-1)
#     visualize_trajectories(sampled_trajectories, config['scene'],
#                            fpath, headless=False)
#

def vis_dataset(loader, config, N=100):
    fpath = f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/dataset_vis'
    n = 0
    for trajectories, _, masks in loader:
        trajectories = trajectories * train_dataset.std + train_dataset.mean

        for trajectory, mask in zip(trajectories, masks):
            trajectory = trajectory[mask[:, 0].nonzero().reshape(-1)]
            trajectory[:, 15] *= 0
            # visualize the trajectory kinematically
            pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/kin/img'), parents=True, exist_ok=True)
            pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/kin/gif'), parents=True, exist_ok=True)
            visualize_trajectory(trajectory, config["scene"], f'{fpath}/trajectory_{n + 1}/kin', headless=False,
                                 fingers=fingers, obj_dof=4)

            n += 1
        if n > N:
            break


def visualize_trajectory_in_sim(trajectory, env, fpath):
    # reset environment
    env.frame_fpath = f'{fpath}/img'
    env.frame_id = 0
    x0 = trajectory[0, :9].to(device=env.device)
    env.reset(x0.unsqueeze(0))

    # rollout actions
    u = trajectory[:-1, -8:].to(device=env.device)  # controls
    for i in range(u.shape[0]):
        x = env.get_state()['q'].reshape(1, 9)[:, :8]
        des_x = x + u[i].reshape(1, 8)
        env.step(des_x)

    import subprocess
    output_dir = f'{fpath}/gif/trajectory.gif'
    cmd = f"ffmpeg -y -i {fpath}/img/frame_%6d.png -vf palettegen ~/palette.png"
    subprocess.call(cmd, shell=True)
    cmd = f"ffmpeg -y -framerate 2 -i {fpath}/img/frame_%6d.png -i ~/palette.png " \
          f"-lavfi paletteuse {output_dir}"
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

    torch.set_float32_matmul_precision('high')
    config = yaml.safe_load(
        pathlib.Path(f'{CCAI_PATH}/config/training/allegro_screwdriver_diffusion.yaml').read_text())

    if config['sine_cosine']:
        raise NotImplementedError
    else:
        dx = config['dx']

    if config['use_class']:
        dcontext = 3
    else:
        dcontext = 0

    model = TrajectorySampler(T=config['T'], dx=dx, du=config['du'], context_dim=dcontext, type=config['model_type'],
                              hidden_dim=config['hidden_dim'], timesteps=config['timesteps'],
                              generate_context=config['diffuse_class'])

    data_path = pathlib.Path(f'{CCAI_PATH}/data/training_data/{config["data_directory"]}')
    train_dataset = AllegroScrewDriverDataset([p for p in data_path.glob('*train_data*')],
                                              cosine_sine=config['sine_cosine'],
                                              states_only=config['du'] == 0)

    if config['normalize_data']:
        # normalize data
        train_dataset.compute_norm_constants()
        model.set_norm_constants(*train_dataset.get_norm_constants())

    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                  sampler=train_sampler, num_workers=4, pin_memory=True)

    model = model.to(device=config['device'])

    # set up pytorch volumetric for rendering
    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    index_ee_name = 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link'
    thumb_ee_name = 'allegro_hand_oya_finger_3_aftc_base_link'
    # combined chain
    chain = pk.build_chain_from_urdf(open(asset).read())
    # TODO currently hardcoded relative pose
    p = [0.0, -0.1, 1.33]
    #r = [0.6645, 0.2418, 0.2418, 0.6645]
    r = [0.2418448, 0.2418448, 0.664463, 0.664463]

    object_pos = np.array([0., 0, 1.205])

    world_trans = tf.Transform3d(pos=torch.tensor(p, device=config['device']),
                                 rot=torch.tensor([r[3], r[0], r[1], r[2]], device=config['device']),
                                 device=config['device'])

    asset_object = get_assets_dir() + '/screwdriver/screwdriver.urdf'
    chain_object = pk.build_chain_from_urdf(open(asset_object).read())
    chain = chain.to(device=config['device'])
    chain_object = chain_object.to(device=config['device'])
    object_sdf = pv.RobotSDF(chain_object, path_prefix=get_assets_dir() + '/screwdriver',
                             use_collision_geometry=False)
    robot_sdf = pv.RobotSDF(chain, path_prefix=get_assets_dir() + '/xela_models')
    scene_trans = world_trans.inverse().compose(
        pk.Transform3d(device=config['device']).translate(object_pos[0], object_pos[1], object_pos[2]))
    contact_links = ['allegro_hand_oya_finger_3_aftc_base_link',
                     'allegro_hand_naka_finger_finger_1_aftc_base_link',
                     'allegro_hand_hitosashi_finger_finger_0_aftc_base_link']
    scene = pv.RobotScene(robot_sdf, object_sdf, scene_trans,
                          collision_check_links=contact_links,
                          softmin_temp=100.0)

    i = np.random.randint(low=0, high=len(train_dataset))
    # visualize_trajectory(train_dataset[i] * train_dataset.std + train_dataset.mean,
    #                     scene, scene_fpath=f'{CCAI_PATH}/examples', headless=False)
    config['scene'] = scene
    train_model(model, train_loader, config)
    # vis_dataset(train_loader, config, N=64)

    model.load_state_dict(torch.load(
        f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/allegro_screwdriver_{config["model_type"]}.pt'
    ))
    # test_long_horizon(model, train_loader, config)
