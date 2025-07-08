# from isaac_victor_envs.tasks.allegro import AllegroValveTurningEnv
# from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv

import yaml
import copy
import tqdm
import torch
from torch import nn
from torch.utils.data import Subset
import pathlib
import numpy as np
import argparse
import open3d as o3d
import pytorch_kinematics as pk

import pytorch_volumetric as pv
import matplotlib.pyplot as plt
from ccai.models.training import EMA
from pytorch_kinematics import transforms as tf
from ccai.dataset import AllegroScrewDriverDataset, AllegroScrewDriverStateDataset, FakeDataset, RealAndFakeDataset, PerEpochBalancedSampler
from isaac_victor_envs.utils import get_assets_dir
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from ccai.models.trajectory_samplers_sac import TrajectorySampler
from ccai.utils.allegro_utils import partial_to_full_state, visualize_trajectory
from ccai.allegro_screwdriver_problem_diffusion import AllegroScrewdriverDiff
import pickle
import sys
from torch.utils.data import random_split

from typing import Dict

import scipy

import datetime
import time

import wandb
from sklearn.metrics import confusion_matrix

TORCH_LOGS = "+dynamo"
TORCHDYNAMO_VERBOSE = 1
fingers = ['index', 'middle', 'thumb']


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='allegro_screwdriver_diffusion_eval_train_likelihood.yaml')
    # parser.add_argument('--config', type=str, default='allegro_screwdriver_diffusion_id_ood_states.yaml')
    # parser.add_argument('--config', type=str, default='allegro_screwdriver_diffusion_project_ood_states.yaml')
    # parser.add_argument('--config', type=str, default='allegro_screwdriver_diffusion_id_ood_states.yaml')
    parser.add_argument('--config', type=str, default='allegro_screwdriver_diffusion_5_10_15_7000_training_best_traj_only_diffuse_c_mode_alt_2.yaml')
    # parser.add_argument('--config', type=str, default='allegro_screwdriver_diffusion_recovery_best_traj_only_gen_sim_data.yaml')
    return parser.parse_args()


def train_model(trajectory_sampler, train_loader, config):
    fpath = f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}'
    pathlib.Path.mkdir(pathlib.Path(fpath), parents=True, exist_ok=True)
    run = wandb.init(project='ccai-screwdriver', entity='abhinavk99', config=config)

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
        train_loss_tau = 0.0
        train_loss_c = 0
        trajectory_sampler.train()
        for trajectories, traj_class, masks in (train_loader):
            trajectories = trajectories.to(device=config['device'])
            masks = masks.to(device=config['device'])
            B, T, dxu = trajectories.shape
            if config['use_class']:
                traj_class = traj_class.to(device=config['device']).float()
            else:
                traj_class = None
            sampler_losses = trajectory_sampler.loss(trajectories, mask=masks, constraints=traj_class)
            loss = sampler_losses['loss']
            loss.backward()
            # wandb.log({'train_loss': loss.item()})
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # for param in trajectory_sampler.parameters():
            #    print(param.grad)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            if config['diffuse_class']:
                train_loss_tau += sampler_losses['loss_tau']
                train_loss_c += sampler_losses['loss_c']
            step += 1
            if config['use_ema']:
                if step % 10 == 0:
                    update_ema(trajectory_sampler)

        train_loss /= len(train_loader)
        train_loss_tau /= len(train_loader)
        train_loss_c /= len(train_loader)
        pbar.set_description(
            f'Train loss {train_loss:.3f}')
        try:
            wandb.log({
                'train_loss_epoch': train_loss,
                'train_loss_diffusion_epoch': train_loss_tau,
                'train_loss_c_mode_epoch': train_loss_c,
                'time': time.time()
                })
        except:
            print('Could not log to wandb')
            print({
                'train_loss_epoch': train_loss,
                'train_loss_diffusion_epoch': train_loss_tau,
                'train_loss_c_mode_epoch': train_loss_c,
                'time': time.time()
                })
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
                torch.save(ema_model.state_dict(), f'{fpath}/allegro_screwdriver_{config["model_type"]}_{epoch}.pt')
            else:
                torch.save(model.state_dict(),
                           f'{fpath}/allegro_screwdriver_{config["model_type"]}_{epoch}.pt')
    if config['use_ema']:
        torch.save(ema_model.state_dict(), f'{fpath}/allegro_screwdriver_{config["model_type"]}_{epoch}.pt')
    else:
        torch.save(model.state_dict(),
                   f'{fpath}/allegro_screwdriver_{config["model_type"]}_{epoch}.pt')

def convert_yaw_to_sine_cosine(xu):
    """
    xu is shape (N, T, 36)
    Replace the yaw in xu with sine and cosine and return the new xu
    """
    yaw = xu[..., 14]
    sine = torch.sin(yaw)
    cosine = torch.cos(yaw)
    xu_new = torch.cat([xu[..., :14], cosine.unsqueeze(-1), sine.unsqueeze(-1), xu[..., 15:]], dim=-1)
    return xu_new

def convert_sine_cosine_to_yaw(xu):
    """
    xu is shape (N, T, 37)
    Replace the sine and cosine in xu with yaw and return the new xu
    """
    orig_type = torch.is_tensor(xu)
    if not orig_type:
        xu = torch.tensor(xu)
    sine = xu[..., 14]
    cosine = xu[..., 15]
    yaw = torch.atan2(sine, cosine)
    xu_new = torch.cat([xu[..., :14], yaw.unsqueeze(-1), xu[..., 16:]], dim=-1)
    if not orig_type:
        xu_new = xu_new.numpy()
    return xu_new

def train_q_function()
if __name__ == "__main__":
    CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
    print(CCAI_PATH)
    args = get_args()
    torch.set_float32_matmul_precision('high')
    print(args.config)
    config = yaml.safe_load(
        pathlib.Path(f'{CCAI_PATH}/config/training/{args.config}').read_text())

    if config['sine_cosine']:
        dx = config['dx'] + 1
        config['dx'] = dx
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
    if 'state_only' not in config:
        config['state_only'] = False
    if 'state_control_only' not in config:
        config['state_control_only'] = False

    for key in ['eval_train_likelihood', 'id_ood_states']:
        if key not in config:
            config[key] = False
    if config['train_diffusion'] and config['state_control_only']:
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
        obj_dof = 3
        object_location = torch.tensor([0, 0, 1.205]).to(
                        config['device'])
        config['object_location'] = object_location
        table_pose = torch.tensor([0, 0, 1.205]).to(
                        config['device'])
        p = [0, -0.095, 1.33]
        r = [0.2418448, 0.2418448, 0.664463, 0.664463]
        world_trans = tf.Transform3d(pos=torch.tensor(p, device=config['device']),
                                    rot=torch.tensor(
                                        [r[3], r[0], r[1], r[2]],
                                        device=config['device']), device=config['device'])
        dummy_start = torch.rand(16 if config['sine_cosine'] else 15).to(device=config['device'])
        pregrasp_problem_diff = AllegroScrewdriverDiff(
            start=dummy_start,
            goal=None,
            T=1,
            chain=chain,
            device=config['device'],
            object_asset_pos=table_pose,
            object_location=config['object_location'],
            object_type='screwdriver',
            world_trans=world_trans,
            regrasp_fingers=fingers,
            contact_fingers=[],
            obj_dof=obj_dof,
            obj_joint_dim=1,
            optimize_force=True,
        )
        # finger gate index
        index_regrasp_problem_diff = AllegroScrewdriverDiff(
            start=dummy_start,
            goal=None,
            T=1,
            chain=chain,
            device=config['device'],
            object_asset_pos=table_pose,
            object_location=config['object_location'],
            object_type='screwdriver',
            world_trans=world_trans,
            regrasp_fingers=['index'],
            contact_fingers=['middle', 'thumb'],
            obj_dof=obj_dof,
            obj_joint_dim=1,
            optimize_force=True,
            default_dof_pos=None
        )
        thumb_and_middle_regrasp_problem_diff = AllegroScrewdriverDiff(
            start=dummy_start,
            goal=None,
            T=1,
            chain=chain,
            device=config['device'],
            object_asset_pos=table_pose,
            object_location=config['object_location'],
            object_type='screwdriver',
            world_trans=world_trans,
            contact_fingers=['index'],
            regrasp_fingers=['middle', 'thumb'],
            obj_dof=obj_dof,
            obj_joint_dim=1,
            optimize_force=True,
            default_dof_pos=None
        )
        turn_problem_diff = AllegroScrewdriverDiff(
            start=dummy_start,
            goal=None,
            T=1,
            chain=chain,
            device=config['device'],
            object_asset_pos=table_pose,
            object_location=config['object_location'],
            object_type='screwdriver',
            world_trans=world_trans,
            contact_fingers=['index', 'middle', 'thumb'],
            obj_dof=obj_dof,
            obj_joint_dim=1,
            optimize_force=True,
            default_dof_pos=None
        )

        problem_for_sampler = {
            (-1, -1, -1): pregrasp_problem_diff,
            (-1, 1, 1): index_regrasp_problem_diff,
            (1, -1, -1): thumb_and_middle_regrasp_problem_diff,
            (1, 1, 1): turn_problem_diff
        }

    model = TrajectorySampler(T=config['T'], dx=dx, du=config['du'], context_dim=dcontext, type=config['model_type'],
                              hidden_dim=config['hidden_dim'], timesteps=config['timesteps'],
                              generate_context=config['diffuse_class'],
                              discriminator_guidance=config['discriminator_guidance'],
                              learn_inverse_dynamics=config['inverse_dynamics'],
                              state_only=config['state_only'], state_control_only=config['state_control_only'],
                              problem=problem_for_sampler if config['state_control_only'] else None,
                              dropout_p=config.get('context_dropout_p', .25), trajectory_condition=config.get('trajectory_condition', False),
                              true_s0=config.get('true_s0', False), 
                              )

    data_path = pathlib.Path(f'{CCAI_PATH}/data/training_data/{config["data_directory"]}')
    if config.get('eval_train_likelihood', False) or config.get('id_ood_states', False):
        train_dataset = AllegroScrewDriverStateDataset([p for p in data_path.glob('*train_data*')],
                                                config['T']-1,
                                                cosine_sine=config['sine_cosine'],
                                                states_only=config['du'] == 0,
                                                skip_pregrasp=config['skip_pregrasp'],
                                                type=config['model_type'],)
    elif not config.get('project_ood_states', False):
        train_dataset = AllegroScrewDriverDataset([p for p in data_path.glob('*train_data*')],
                                                config['T']-1,
                                                cosine_sine=config['sine_cosine'],
                                                states_only=config['du'] == 0,
                                                skip_pregrasp=config['skip_pregrasp'],
                                                type=config['model_type'],
                                                exec_only=config.get('train_classifier', False) or config.get('likelihood_ecdf_calc', False),
                                                best_traj_only=config['best_traj_only'])
                                                
    if not config.get('project_ood_states', False):
        if config['normalize_data']:
            # normalize data
            train_dataset.compute_norm_constants()
            model.set_norm_constants(*train_dataset.get_norm_constants())

            print('Dset size', len(train_dataset))
            
            print(train_dataset.mean)
        if 'recovery' in config['data_directory']:
            classes = train_dataset.trajectory_type
            print(torch.unique(classes, dim=0, return_counts=True))
            if config['balance'] == 'weighted_random':
                weights = [1/(classes.sum(1)).tolist().count(classes[i].sum().item()) for i in range(classes.shape[0])]
                train_sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)

            elif config['balance'] == 'once':
                # Convert to numpy for easier manipulation
                classes_np = classes.numpy()
                
                # Get unique class combinations and their counts
                unique_classes = np.unique(classes_np, axis=0)
                class_counts = {tuple(c): np.sum(np.all(classes_np == c, axis=1)) for c in unique_classes}
                
                # Find minimum class count
                min_count = min(class_counts.values())
                
                # Create balanced indices
                balanced_indices = []
                for c in unique_classes:
                    # Find indices for this class
                    class_indices = np.where(np.all(classes_np == c, axis=1))[0]
                    # Randomly sample min_count indices
                    sampled_indices = np.random.choice(class_indices, size=min_count, replace=False)
                    balanced_indices.extend(sampled_indices)
                
                # Convert to tensor and shuffle
                balanced_indices = torch.tensor(balanced_indices)[torch.randperm(len(balanced_indices))]
                
                # Create subset dataset
                train_dataset = Subset(train_dataset, balanced_indices)
                
                print(f"Balanced dataset size: {len(train_dataset)}")
                print(f"Samples per class: {min_count}")
                
                train_sampler = RandomSampler(train_dataset)

            elif config['balance'] == 'per_epoch':
                    train_sampler = PerEpochBalancedSampler(train_dataset)
                    print(f"Samples per class per epoch: {train_sampler.samples_per_class}")
                    print(f"Total samples per epoch: {len(train_sampler)}")
            else:
                train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = RandomSampler(train_dataset)
        # train_sampler = None
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True,
                                )

    model = model.to(device=config['device'])
    # i = np.random.randint(low=0, high=len(train_dataset))
    # visualize_trajectory(train_dataset[i] * train_dataset.std + train_dataset.mean,
    #                     scene, scene_fpath=f'{CCAI_PATH}/examples', headless=False)

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
    config['scene'] = scene
    config['env'] = env


    if config['train_diffusion'] and not (config['state_only'] or config['state_control_only']):
        train_model(model, train_loader, config)
    elif config['train_diffusion'] and (config['state_only'] or config['state_control_only']):
        train_model_state_only(model, train_loader, config)

    if config['vis_dataset']:
        vis_dataset(train_loader, config, N=8)


    if config['load_model']:
        # if config['discriminator_guidance']:
        #     model_name = f'allegro_screwdriver_{config["model_type"]}_w_classifier.pt'
        # else:
        model_name = f'allegro_screwdriver_{config["model_type"]}.pt'
        d = torch.load(
                f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/{model_name}'
            , map_location=config['device'])
        model.model.diffusion_model.classifier = None
        d = {k:v for k, v in d.items() if 'classifier' not in k}
        model.load_state_dict(d, strict=True)
        model.send_norm_constants_to_submodels()


    if config['eval_train_likelihood']:
        eval_train_likelihood(model, train_loader, config)

    if config['id_ood_states']:
        identify_OOD_states(model, train_loader, config)

    if config.get('project_ood_states', False):
        project_OOD_states(model, config)
        visualize_ood_projection(model, config)

    # if config['filter_recovery_trajectories']:
    #     filter_bad_traj(model, train_loader, config)

    if config['plot']:
        if config['load_model'] and config['discriminator_guidance']:
            plot_name = 'w_guidance'
        else:
            plot_name = 'no_guidance'
        plot_long_horizon(model, train_loader, config, plot_name)

    if config.get('likelihood_ecdf_calc', False):
        train_loader.dataset.update_masks(p1=1., p2=1.)
        train_loader.batch_size = 1
        likelihood_ecdf_calc(model, train_loader, config)

    if config['train_classifier']:
        model.model.diffusion_model.classifier = None
        # generate dataset from trained diffusion model
        # generate_simulated_data(model, train_loader, config, name='')
        train_loader.dataset.update_masks(p1=1., p2=1.)

        # fpath = f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/simulated_dataset/simulated_trajectories_.npz'

        # fake_dataset = FakeDataset(fpath, config['sine_cosine'])
        # fake_dataset.set_norm_constants(train_dataset.mean, train_dataset.std)
        # train_loader = DataLoader(fake_dataset, batch_size=config['batch_size'])

        # length = min(len(train_dataset), len(fake_dataset)
        # # Split train, val
        # train_size = int(0.9 * length)
        # val_size = length - train_size

        # inds = torch.randperm(length)
        # train_inds = inds[:train_size]
        # val_inds = inds[train_size:]

        # train_real_dataset = Subset(train_dataset, train_inds)
        # val_real_dataset = Subset(train_dataset, val_inds)

        # train_fake_dataset = Subset(fake_dataset, train_inds)
        # val_fake_dataset = Subset(fake_dataset, val_inds)

        # train_classifier_dataset = RealAndFakeDataset(train_real_dataset, train_fake_dataset)

        # val_classifier_dataset = RealAndFakeDataset(val_real_dataset, val_fake_dataset)

        # # classes = train_dataset.trajectory_type[train_inds]
        # # weights = torch.where(classes[:, 0] == 1, 6636/4668, 1.)

        # # train_classifier_sampler = WeightedRandomSampler(weights, len(train_classifier_dataset), replacement=True)

        # Train val split of train_dataset
        train_classifier_dataset, val_classifier_dataset = random_split(train_dataset, [0.9, .1])

        train_classifier_loader = DataLoader(train_classifier_dataset, batch_size=64,
                                   shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

        val_classifier_loader = DataLoader(val_classifier_dataset, batch_size=64,
                                   shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        train_classifier(model, train_classifier_loader, val_classifier_loader, config)
        # eval trained classifier on training data

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
