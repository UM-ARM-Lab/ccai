# from isaac_victor_envs.tasks.allegro import AllegroValveTurningEnv
from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv

import yaml
import copy
import tqdm
import torch
import pathlib
import numpy as np
import argparse
import open3d as o3d
import pytorch_kinematics as pk
import pytorch_volumetric as pv
import matplotlib.pyplot as plt
from ccai.models.training import EMA
import pytorch_kinematics.transforms as tf
from ccai.dataset import AllegroScrewDriverDataset, FakeDataset, RealAndFakeDataset
from isaac_victor_envs.utils import get_assets_dir
from torch.utils.data import DataLoader, RandomSampler
from ccai.models.trajectory_samplers import TrajectorySampler
from ccai.utils.allegro_utils import partial_to_full_state, visualize_trajectory

TORCH_LOGS = "+dynamo"
TORCHDYNAMO_VERBOSE = 1
fingers = ['index', 'middle', 'thumb']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='allegro_screwdriver_diffusion.yaml')
    return parser.parse_args()


def visualize_trajectories(trajectories, scene, fpath, headless=False):
    for n, trajectory in enumerate(trajectories):
        state_trajectory = trajectory[:, :16].clone()
        state_trajectory[:, 15] *= 0
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/kin/img'), parents=True, exist_ok=True)
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/kin/gif'), parents=True, exist_ok=True)
        visualize_trajectory(state_trajectory, scene, f'{fpath}/trajectory_{n + 1}/kin', headless=headless,
                             fingers=fingers, obj_dof=4)
        # Visualize what happens if we execute the actions in the trajectory in the simulator
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/sim/img'), parents=True, exist_ok=True)
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/sim/gif'), parents=True, exist_ok=True)
        #visualize_trajectory_in_sim(trajectory, config['env'], f'{fpath}/trajectory_{n + 1}/sim')
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
        for trajectories, traj_class, masks in tqdm.tqdm(train_loader):
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

        if (epoch + 1) % config['save_every'] == 0:
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


def plot_long_horizon(test_model, loader, config, name):
    fpath = f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/{name}'
    pathlib.Path.mkdir(pathlib.Path(fpath), parents=True, exist_ok=True)
    N = 16

    # we will plot for a variety of different horizons
    N = 16
    min_horizon = 16
    max_horizon = 16 * 5
    trajectories, traj_class, masks = next(iter(train_loader))
    if train_loader.dataset.cosine_sine:
        dx = 16
    else:
        dx = 15
    start = (trajectories[:N, 0, :dx] * test_model.x_std[:dx].to(device=trajectories.device) +
             test_model.x_mean[:dx].to(device=trajectories.device))
    start = start[0].repeat(N, 1)
    start = start.to(device=config['device'])
    # start = start[None, :].repeat(N, 1).to(device=config['device'])

    full_context = torch.tensor([
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0]
    ], device=config['device'])

    for H in range(min_horizon, max_horizon + 1, 16):
        T = H // 16
        context = full_context[:T]
        plot_fpath = f'{fpath}/final/horizon_{H}'
        pathlib.Path.mkdir(pathlib.Path(plot_fpath), parents=True, exist_ok=True)
        print(start.shape, context.shape)
        sampled_trajectories, sampled_contexts, likelihoods = test_model.sample(N, H=H,
                                                                                start=start,
                                                                                constraints=context.repeat(N, 1, 1))
        print(sampled_contexts)
        visualize_trajectories(sampled_trajectories, config['scene'], plot_fpath, headless=False)


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

def rollout_trajectory_in_sim(trajectory, env):
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
    # q = partial_to_full_state(x0[:12], fingers)
    # x0 = torch.cat((q, x0[12:]))
    # print(x0.shape)
    env.reset(x0.unsqueeze(0))
    # rollout actions
    u = trajectory[:-1, 15:15 + 12].to(device=env.device)  # controls
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


def generate_simulated_data(model, loader, config, name=None):
    fpath = f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/simulated_dataset'
    pathlib.Path.mkdir(pathlib.Path(fpath), parents=True, exist_ok=True)
    # assume we want a dataset size as large as the original dataset
    N = len(loader.dataset)
    loader.dataset.update_masks(p1=1.0, p2=1.0)  # no masking

    # generate three subtrajectories
    num_sub_traj = 3

    ACTION_DICT = {
        0: torch.tensor([[-1.0, -1.0, -1.0]]),  # pregrasp
        1: torch.tensor([[1.0, -1.0, -1.0]]),  # regrasp thumb / middle
        2: torch.tensor([[-1.0, 1.0, 1.0]]),  # regrasp index
        3: torch.tensor([[1.0, 1.0, 1.0]]),  # turn
    }
    ACTION_TENSOR = torch.cat([ACTION_DICT[i] for i in ACTION_DICT.keys()], dim=0).to(device=config['device'])

    simulated_trajectories = []
    simulated_class = []
    dataset_size = 0

    for trajectories, traj_class, _ in train_loader:

        if train_loader.dataset.cosine_sine:
            dx = 16
        else:
            dx = 15


        B = trajectories.shape[0]
        trajectories = trajectories.to(device=config['device'])
        traj_class = traj_class.to(device=config['device'])
        p = torch.tensor([0.25, 0.25, 0.25, 0.25], device=config['device'])
        idx = p.multinomial(num_samples=((num_sub_traj - 1) * B), replacement=True)
        _next_class = ACTION_TENSOR[idx].reshape(B, -1, 3)

        traj_class = torch.cat((traj_class.reshape(B, 1, 3), _next_class), dim=1)
        start = (trajectories[:N, 0, :dx] * model.x_std[:dx].to(device=trajectories.device) +
                 model.x_mean[:dx].to(device=trajectories.device))

        new_traj, _, _ = model.sample(N=B, H=num_sub_traj * 16, start=start, constraints=traj_class)

        # visualize_trajectories(new_traj, config['scene'], fpath, headless=False)

        for i in range(num_sub_traj):
            if i == 0:
                idx = 0
            else:
                idx = i * 16 - 1

            simulated_trajectories.append(new_traj[:, idx:idx + 16].detach().cpu())
            simulated_class.append(traj_class[:, i].detach().cpu())

        dataset_size += B * num_sub_traj
        if dataset_size >= N:
            break

    simulated_trajectories = torch.cat(simulated_trajectories, dim=0).numpy()[:N]
    simulated_class = torch.cat(simulated_class, dim=0).numpy()[:N]

    # save the new dataset

    np.savez(f'{fpath}/simulated_trajectories_{name}.npz', trajectories=simulated_trajectories, contact=simulated_class)


def train_classifier(model, train_loader, config):
    fpath = f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}'
    pathlib.Path.mkdir(pathlib.Path(fpath), parents=True, exist_ok=True)

    # add classifier
    model.model.diffusion_model.add_classifier()
    model.model.diffusion_model.classifier = model.model.diffusion_model.classifier.to(device=config['device'])
    optimizer = torch.optim.Adam(model.model.diffusion_model.classifier.parameters(), lr=config['lr'])

    step = 0
    epochs = config['classifier_epochs']
    pbar = tqdm.tqdm(range(epochs))
    for epoch in pbar:
        train_loss = 0.0
        model.train()
        for real_traj, real_class, real_masks, fake_traj, fake_class, fake_masks in train_loader:
            B1, B2 = real_traj.shape[0], fake_traj.shape[0]
            trajectories = torch.cat((real_traj, fake_traj), dim=0).to(device=config['device'])
            context = torch.cat((real_class, fake_class), dim=0).to(device=config['device'])
            masks = torch.cat((real_masks, fake_masks), dim=0).to(device=config['device'])
            labels = torch.cat((torch.ones(B1, 1), torch.zeros(B2, 1)), dim=0).to(device=config['device'])

            loss, acc = model.classifier_loss(trajectories, mask=masks, label=labels, context=context)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # for param in trajectory_sampler.parameters():
            #    print(param.grad)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        pbar.set_description(
            f'Train loss {train_loss:.3f}')

        if (epoch + 1) % config['save_every'] == 0:
            torch.save(model.state_dict(),
                       f'{fpath}/allegro_screwdriver_{config["model_type"]}_w_classifier.pt')

    torch.save(model.state_dict(),
               f'{fpath}/allegro_screwdriver_{config["model_type"]}_w_classifier.pt')


def eval_classifier(model, train_loader, config):
    step = 0
    train_accuracy = 0.0
    model.eval()
    for real_traj, real_class, real_masks, fake_traj, fake_class, fake_masks in train_loader:
        B1, B2 = real_traj.shape[0], fake_traj.shape[0]
        trajectories = torch.cat((real_traj, fake_traj), dim=0).to(device=config['device'])
        context = torch.cat((real_class, fake_class), dim=0).to(device=config['device'])
        masks = torch.cat((real_masks, fake_masks), dim=0).to(device=config['device'])
        labels = torch.cat((torch.ones(B1, 1), torch.zeros(B2, 1)), dim=0).to(device=config['device'])

        _, accuracy = model.classifier_loss(trajectories, mask=masks, label=labels, context=context)
        train_accuracy += accuracy.item()
    train_accuracy /= len(train_loader)
    print(train_accuracy)


if __name__ == "__main__":
    CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
    args = get_args()
    torch.set_float32_matmul_precision('high')
    print(args.config)
    config = yaml.safe_load(
        pathlib.Path(f'{CCAI_PATH}/config/training/{args.config}').read_text())

    if config['sine_cosine']:
        dx = config['dx'] + 1
    else:
        dx = config['dx']

    if config['use_class']:
        dcontext = 3
    else:
        dcontext = 0

    env = None
    # env = AllegroScrewdriverTurningEnv(1, control_mode='joint_impedance',
    #                                    use_cartesian_controller=False,
    #                                    viewer=True,
    #                                    steps_per_action=60,
    #                                    friction_coefficient=1.05,
    #                                    # friction_coefficient=1.0,  # DEBUG ONLY, set the friction very high
    #                                    device=config['device'],
    #                                    video_save_path=f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}',
    #                                    joint_stiffness=3,
    #                                    fingers=['index', 'middle', 'thumb'],
    #                                    )
    # import time
    # try:
    #     while True:
    #         start = env.get_state()['q'][:, :-1]
    #         env.step(start)
    #         print('waiting for you to finish camera adjustment, ctrl-c when done')
    #         time.sleep(0.1)
    # except KeyboardInterrupt:
    #     pass

    model = TrajectorySampler(T=config['T'], dx=dx, du=config['du'], context_dim=dcontext, type=config['model_type'],
                              hidden_dim=config['hidden_dim'], timesteps=config['timesteps'],
                              generate_context=config['diffuse_class'],
                              discriminator_guidance=config['discriminator_guidance'],
                              learn_inverse_dynamics=config['inverse_dynamics'])

    data_path = pathlib.Path(f'{CCAI_PATH}/data/training_data/{config["data_directory"]}')
    train_dataset = AllegroScrewDriverDataset([p for p in data_path.glob('*train_data*')],
                                              config['T']-1,
                                              cosine_sine=config['sine_cosine'],
                                              states_only=config['du'] == 0,
                                              skip_pregrasp=config['skip_pregrasp'])
    train_dataset.update_masks(p1=1, p2=1)
    if config['normalize_data']:
        # normalize data
        train_dataset.compute_norm_constants()
        model.set_norm_constants(*train_dataset.get_norm_constants())

    print('Dset size', len(train_dataset))
    
    print(train_dataset.mean)
    train_sampler = RandomSampler(train_dataset)
    #train_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)

    model = model.to(device=config['device'])

    # set up pytorch volumetric for rendering
    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    index_ee_name = 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link'
    thumb_ee_name = 'allegro_hand_oya_finger_3_aftc_base_link'
    # combined chain
    chain = pk.build_chain_from_urdf(open(asset).read())
    # TODO currently hardcoded relative pose
    p = [0.0, -0.1, 1.33]
    # r = [0.6645, 0.2418, 0.2418, 0.6645]
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
    config['env'] = env

    if config['train_diffusion']:
        train_model(model, train_loader, config)

    if config['vis_dataset']:
        vis_dataset(train_loader, config, N=8)


    if config['load_model']:
        if config['discriminator_guidance']:
            model_name = f'allegro_screwdriver_{config["model_type"]}_w_classifier.pt'
        else:
            model_name = f'allegro_screwdriver_{config["model_type"]}.pt'

        model.load_state_dict(torch.load(
            f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/{model_name}'
        ))

    if config['plot']:
        if config['load_model'] and config['discriminator_guidance']:
            plot_name = 'w_guidance'
        else:
            plot_name = 'no_guidance'
        plot_long_horizon(model, train_loader, config, plot_name)

    if config['train_classifier']:
        model.model.diffusion_model.classifier = None
        # generate dataset from trained diffusion model
        generate_simulated_data(model, train_loader, config, name='')
        train_loader.dataset.update_masks(p1=0.5, p2=0.75)

        fpath = f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/simulated_dataset/simulated_trajectories_.npz'

        fake_dataset = FakeDataset(fpath, config['sine_cosine'])
        fake_dataset.set_norm_constants(train_dataset.mean, train_dataset.std)
        train_loader = DataLoader(fake_dataset, batch_size=config['batch_size'])

        classifier_dataset = RealAndFakeDataset(train_dataset, fake_dataset)
        classifier_sampler = RandomSampler(classifier_dataset)
        # train_sampler = None
        classifier_loader = DataLoader(classifier_dataset, batch_size=config['batch_size'],
                                   sampler=classifier_sampler, num_workers=4, pin_memory=True, drop_last=True)


        train_classifier(model, classifier_loader, config)
        # eval trained classifier on training data
        eval_classifier(model, classifier_loader, config)

    if config['eval_classifier']:
        # generate new test set with classifier guidance and evaluate on that - ideally classifier performance gets worse!
        generate_simulated_data(model, train_loader, config, name='with_guidance')
        train_loader.dataset.update_masks(p1=0.5, p2=0.75)
        fpath = f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/simulated_dataset/simulated_trajectories_with_guidance.npz'
        fake_dataset = FakeDataset(fpath, config['sine_cosine'])
        classifier_dataset = RealAndFakeDataset(train_dataset, fake_dataset)
        classifier_sampler = RandomSampler(classifier_dataset)
        # train_sampler = None
        classifier_loader = DataLoader(classifier_dataset, batch_size=config['batch_size'],
                                       sampler=classifier_sampler, num_workers=4, pin_memory=True, drop_last=True)
        # evaluate the classifier on this new data set
        # By using classifier guidance we should e slightly better at fooling the classifier
        eval_classifier(model, classifier_loader, config)
        plot_long_horizon(model, train_loader, config, 'w_guidance')
