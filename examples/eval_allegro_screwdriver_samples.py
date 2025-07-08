# from isaac_victor_envs.tasks.allegro import AllegroValveTurningEnv
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv

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
from pytorch_kinematics import transforms as tf
from ccai.dataset import AllegroScrewDriverDataset
from isaac_victor_envs.utils import get_assets_dir
from torch.utils.data import DataLoader, RandomSampler
from ccai.models.trajectory_samplers import TrajectorySampler
from utils.allegro_utils import partial_to_full_state, visualize_trajectory
import pickle

fingers = ['index', 'middle', 'thumb']


def visualize_trajectories(trajectories, scene, fpath, headless=False):
    for n, trajectory in enumerate(trajectories):

        state_trajectory = trajectory[:, :16]
        state_trajectory[:, 15] *= 0
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/kin/img'), parents=True, exist_ok=True)
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/kin/gif'), parents=True, exist_ok=True)
        visualize_trajectory(state_trajectory, scene, f'{fpath}/trajectory_{n + 1}/kin', headless=headless,
                             fingers=fingers, obj_dof=4)
        # Visualize what happens if we execute the actions in the trajectory in the simulator
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/sim/img'), parents=True, exist_ok=True)
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/sim/gif'), parents=True, exist_ok=True)
        visualize_trajectory_in_sim(trajectory, config['env'], f'{fpath}/trajectory_{n + 1}/sim')
        # save the trajectory
        np.save(f'{fpath}/trajectory_{n + 1}/traj.npz', trajectory.cpu().numpy())


def test_long_horizon(test_model, loader, config):
    fpath = f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/eta_0'
    pathlib.Path.mkdir(pathlib.Path(fpath), parents=True, exist_ok=True)
    N = 16

    # we will plot for a variety of different horizons
    N = 32
    min_horizon = 32
    max_horizon = 64

    trajectories, traj_class, masks = next(iter(train_loader))

    start = (trajectories[:N, 0, :15] * test_model.x_std[:15].to(device=trajectories.device) +
         test_model.x_mean[:15].to(device=trajectories.device))
    start = start[0].repeat(N, 1)
    start = start.to(device=config['device'])
    #start = start[None, :].repeat(N, 1).to(device=config['device'])

    for H in range(min_horizon, max_horizon + 1, 16):
        plot_fpath = f'{fpath}/final/horizon_{H}'
        pathlib.Path.mkdir(pathlib.Path(plot_fpath), parents=True, exist_ok=True)
        sampled_trajectories, sampled_contexts, likelihoods = test_model.sample(N, H=H, start=start)
        print(sampled_contexts)
        visualize_trajectories(sampled_trajectories, config['scene'], plot_fpath, headless=False)


def rollout_trajectory_in_sim(trajectory, env):
    env.frame_id = 0
    x0 = trajectory[0, :15].to(device=env.device)
    #q = partial_to_full_state(x0[:12], fingers)
    #x0 = torch.cat((q, x0[12:]))
    #print(x0.shape)
    env.reset(x0.unsqueeze(0))
    # rollout actions
    u = trajectory[:-1, 15:15+12].to(device=env.device)  # controls
    actual_trajectory = torch.zeros_like(trajectory)
    for i in range(u.shape[0]):
        x = env.get_state()['q'].reshape(1, -1)[:, :12]
        des_x = x + u[i].reshape(1, 12)
        env.step(des_x)
        actual_trajectory[i, :15] = x
        actual_trajectory[i, 15:15+12] = u[i]
    actual_trajectory[-1, :15] = env.get_state()['q'].reshape(1, -1)[:, :12]
    return actual_trajectory
        

def visualize_trajectory_in_sim(trajectory, env, fpath):
    # reset environment
    env.frame_fpath = f'{fpath}/img'
    env.frame_id = 0
    x0 = trajectory[0, :15].to(device=env.device)
    #q = partial_to_full_state(x0[:12], fingers)
    #x0 = torch.cat((q, x0[12:]))
    #print(x0.shape)
    env.reset(x0.unsqueeze(0))
    print(trajectory.shape)
    # rollout actions
    u = trajectory[:-1, 15:15+12].to(device=env.device)  # controls
    print(u.shape)
    for i in range(u.shape[0]):
        x = env.get_state()['q'].reshape(1, -1)[:, :12]
        des_x = x + u[i].reshape(1, 12)
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
        pathlib.Path(f'{CCAI_PATH}/config/training/allegro_screwdriver_sim_eval.yaml').read_text())

    if config['sine_cosine']:
        raise NotImplementedError
    else:
        dx = config['dx']

    if config['use_class']:
        dcontext = 3
    else:
        dcontext = 0

    env = AllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
                                       use_cartesian_controller=False,
                                       viewer=config['visualize'],
                                       steps_per_action=60,
                                       friction_coefficient=1.05,
                                       # friction_coefficient=1.0,  # DEBUG ONLY, set the friction very high
                                       device=config['device'],
                                       video_save_path=f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}',
                                       joint_stiffness=3,
                                       fingers=['index', 'middle', 'thumb'],
                                       )
    import time
    try:
        while True:
            start = env.get_state()['q'][:, :-1]
            env.step(start)
            print('waiting for you to finish camera adjustment, ctrl-c when done')
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    data_path = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["data_directory"]}/csvgd')


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

    config['scene'] = scene
    config['env'] = env

    for trial in tqdm.tqdm(range(1, 11)):
        for t in range(1, 2):
            # Load traj_data.p from {data_path}/csvgd/trial_{trial} with pickle
            with open(data_path / f'trial_{trial}/traj_data.p', 'rb') as f:
                data = pickle.load(f)
            data[t]['init_sim_rollouts'] = []
            inits = data[t]['inits']
            # inits = torch.stack(inits).to(device=config['device'])
            for i in tqdm.tqdm(range(len(inits))):
                sim_rollout = rollout_trajectory_in_sim(inits[i], env)
                data[t]['init_sim_rollouts'].append(sim_rollout)
            # data[t]['init_sim_rollouts'] = torch.stack(data[t]['init_sim_rollouts'])
        torch.save(data, data_path / f'trial_{trial}/traj_data_with_init_sim_rollouts.p')


    # train_model(model, train_loader, config)
    # vis_dataset(train_loader, config, N=64)


