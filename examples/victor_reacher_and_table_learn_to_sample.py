import torch
import numpy as np
from torch import optim
import pathlib
import tqdm
import matplotlib.pyplot as plt
from ccai.models.training import EMA

from ccai.models.trajectory_samplers import TrajectorySampler
from ccai.dataset import VictorReachingDataset, VictorTableHeightDataset
from ccai.models.helpers import SinusoidalPosEmb
from ccai.models.constraint_embedding.vae import Conv3DEncoder

import pytorch_kinematics as pk
import yaml
import copy
from isaac_victor_envs.utils import get_assets_dir
from victor_table_surface_jointspace import VictorTableProblem, update_chain

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
asset_dir = get_assets_dir()
asset = asset_dir + '/victor/victor_mallet.urdf'
ee_name = 'victor_left_arm_striker_mallet_tip'
chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name)
chain_cc = pk.build_chain_from_urdf(open(asset).read())


def test(trajectory_sampler, env_encoder, height_encoder, reach_loader, table_loader, config):
    loss = 0.0
    # with torch.no_grad():
    with torch.enable_grad():
        for batch_reach, batch_table in zip(reach_loader, table_loader):
            trajectories_reach, starts_reach, goals_reach, constraints_reach = batch_reach
            trajectories_table, starts_table, goals_table, constraints_table = batch_table

            B1, B2 = trajectories_reach.shape[0], trajectories_table.shape[0]

            # encode constraints
            constraints_reach = env_encoder(constraints_reach.to(device=config['device']))
            constraints_table = height_encoder(constraints_table.to(device=config['device'])).reshape(B2, -1)

            # Combine batches
            trajectories = torch.cat([trajectories_reach, trajectories_table], dim=0)
            starts = torch.cat([starts_reach, starts_table], dim=0)
            goals = torch.cat([goals_reach, goals_table], dim=0)
            constraints = torch.cat([constraints_reach, constraints_table], dim=0)
            constraint_type = torch.cat([torch.zeros(B1, 1), torch.zeros(B2, 1)], dim=0).to(dtype=torch.long)

            B, T, dxu = trajectories.shape
            # send to device
            trajectories = trajectories.to(device=config['device'])
            starts = starts.to(device=config['device'])
            goals = goals.to(device=config['device'])
            constraints = constraints.to(device=config['device'])
            constraint_type = constraint_type.to(device=config['device']).reshape(B)
            constraint_type = torch.nn.functional.one_hot(constraint_type, num_classes=2).float()
            constraints = torch.cat([constraints, constraint_type], dim=-1)

            # Forward pass
            sampler_loss = trajectory_sampler.loss(trajectories,
                                                   starts,
                                                   goals,
                                                   constraints)
            loss += sampler_loss.item()

    return loss / min(len(reach_loader), len(table_loader))


def train_model(trajectory_sampler, env_encoder, height_encoder, train_loader_reach,
                train_loader_table, val_loader_reach, val_loader_table, config):
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

    start_epoch = 0
    if config['load_checkpoint']:
        checkpoint = torch.load(f'{fpath}/checkpoint.pth')
        trajectory_sampler.load_state_dict(checkpoint['model'])
        if config['use_ema']:
            ema_model.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        env_encoder.load_state_dict(checkpoint['env_encoder'])
        height_encoder.load_state_dict(checkpoint['height_encoder'])
        start_epoch = checkpoint['epoch']
        step = checkpoint['step']

    pbar = tqdm.tqdm(range(config['epochs']), initial=start_epoch)

    test_loss = 1000.0
    for epoch in pbar:
        train_loss = 0.0
        trajectory_sampler.train()

        for batch_reach, batch_table in zip(train_loader_reach, train_loader_table):
            trajectories_reach, starts_reach, goals_reach, constraints_reach = batch_reach
            trajectories_table, starts_table, goals_table, constraints_table = batch_table

            B1, B2 = trajectories_reach.shape[0], trajectories_table.shape[0]

            # encode constraints
            constraints_reach = env_encoder(constraints_reach.to(device=config['device']))
            constraints_table = height_encoder(constraints_table.to(device=config['device'])).reshape(B2, -1)
            # Combine batches
            trajectories = torch.cat([trajectories_reach, trajectories_table], dim=0)
            starts = torch.cat([starts_reach, starts_table], dim=0)
            goals = torch.cat([goals_reach, goals_table], dim=0)
            constraints = torch.cat([constraints_reach, constraints_table], dim=0)
            constraint_type = torch.cat([torch.zeros(B1, 1), torch.zeros(B2, 1)], dim=0).to(dtype=torch.long)

            B, T, dxu = trajectories.shape
            # send to device
            trajectories = trajectories.to(device=config['device'])
            starts = starts.to(device=config['device'])
            goals = goals.to(device=config['device'])
            constraints = constraints.to(device=config['device'])
            constraint_type = constraint_type.to(device=config['device']).reshape(B)
            constraint_type = torch.nn.functional.one_hot(constraint_type, num_classes=2).float()
            constraints = torch.cat([constraints, constraint_type], dim=-1)

            # Forward pass
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
            break

        if (epoch + 1) % config['test_every'] == 0:
            if config['use_ema']:
                test_model = ema_model
            else:
                test_model = trajectory_sampler
            # sample trajectories and save for plotting
            # train_loss = test(test_model, env_encoder, height_encoder, train_loader_reach, train_loader_table, config)
            test_loss = test(test_model, env_encoder, height_encoder, val_loader_reach, val_loader_table, config)
            count = 0
            for batch_reach, batch_table in zip(val_loader_reach, val_loader_table):
                trajectories_reach, starts_reach, goals_reach, constraints_reach = batch_reach
                trajectories_table, starts_table, goals_table, constraints_table = batch_table
                B1, B2 = trajectories_reach.shape[0], trajectories_table.shape[0]

                # encode constraints
                constraints_reach = constraints_reach.to(device=config['device'])
                constraints_table = constraints_table.to(device=config['device'])
                constraints_z_reach = env_encoder(constraints_reach)
                constraints_z_table = height_encoder(constraints_table).reshape(B2, -1)

                # Combine batches
                trajectories = torch.cat([trajectories_reach, trajectories_table], dim=0)
                starts = torch.cat([starts_reach, starts_table], dim=0)
                goals = torch.cat([goals_reach, goals_table], dim=0)
                constraints = torch.cat([constraints_z_reach, constraints_z_table], dim=0)
                constraint_type = torch.cat([torch.zeros(B1, 1), torch.ones(B2, 1)], dim=0).to(dtype=torch.long)

                B, T, dxu = trajectories.shape
                # send to device
                trajectories = trajectories.to(device=config['device'])
                starts = starts.to(device=config['device'])
                goals = goals.to(device=config['device'])
                constraints = constraints.to(device=config['device'])
                constraint_type = constraint_type.to(device=config['device']).reshape(B)
                constraint_type = torch.nn.functional.one_hot(constraint_type, num_classes=2).float()
                constraints = torch.cat([constraints, constraint_type], dim=-1)

                count += 1
                starts = starts.to(device=config['device'])
                goals = goals.to(device=config['device'])
                N = 16  # trajectories.shape[1]

                s = starts.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
                g = goals.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, -1)
                c = constraints.reshape(B, 1, -1).repeat(1, N, 1).reshape(B * N, 1, -1)

                with torch.no_grad():
                    sampled_trajectories = test_model.sample(s, g, c)

                # save data for visualization
                sampled_trajectories = sampled_trajectories.reshape(B, N, T, dxu)
                sampled_trajectories_reach = sampled_trajectories[:B1]
                sampled_trajectories_table = sampled_trajectories[B1:]

                # unnormalize true trajectories
                trajectories_table = trajectories_table.to(
                    device=config['device']) * test_model.x_std + test_model.x_mean
                trajectories_reach = trajectories_reach.to(
                    device=config['device']) * test_model.x_std + test_model.x_mean

                # unnormalize starts
                starts_table = starts_table.to(device=config['device']) * test_model.x_std + test_model.x_mean
                starts_reach = starts_reach.to(device=config['device']) * test_model.x_std + test_model.x_mean

                data = {
                    'table': {
                        'trajectories': trajectories_table.cpu().numpy(),
                        'sampled_trajectories': sampled_trajectories_table.cpu().numpy(),
                        'starts': starts_table.cpu().numpy(),
                        'goals': goals_table.cpu().numpy(),
                        'constraints': constraints_table.cpu().numpy()
                    },
                    'reach': {
                        'trajectories': trajectories_reach.cpu().numpy(),
                        'sampled_trajectories': sampled_trajectories_reach.cpu().numpy(),
                        'starts': starts_reach.cpu().numpy(),
                        'goals': goals_reach.cpu().numpy(),
                        'constraints': constraints_reach.cpu().numpy()
                    }
                }
                np.savez(f'{fpath}/data_vis_{epoch}.npz', **data)

        train_loss /= min(len(train_loader_reach), len(train_loader_table))

        pbar.set_description(
            f'Train loss {train_loss:.3f}   Test loss {test_loss:.3f}')

        if (epoch + 1) % config['save_every'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model': trajectory_sampler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema_model.state_dict(),
                'env_encoder': env_encoder.state_dict(),
                'height_encoder': height_encoder.state_dict(),
                'step': step,
                'test_loss': test_loss,
            }
            torch.save(checkpoint, f'{fpath}/checkpoint_{epoch}.pth')


def test_samples(trajectory_sampler, env_encoder, height_encoder, config):
    fpath = f'{CCAI_PATH}/data/training/victor_table/{config["model_name"]}_{config["model_type"]}'
    data_path = pathlib.Path(f'{CCAI_PATH}/data/test_data/{config["test_directory"]}')
    fpaths = [str(d) for d in data_path.glob('*/csvgd')]
    goals = []
    heights = []
    starts = []
    sdfs = []

    if config['load_checkpoint']:
        checkpoint = torch.load(f'{fpath}/checkpoint.pth')
        trajectory_sampler.load_state_dict(checkpoint['model'])
        if config['use_ema']:
            trajectory_sampler.load_state_dict(checkpoint['ema'])
        env_encoder.load_state_dict(checkpoint['env_encoder'])
        height_encoder.load_state_dict(checkpoint['height_encoder'])

    trajectory_sampler = trajectory_sampler.to(device=config['device'])
    trajectory_sampler.send_norm_constants_to_submodels()
    env_encoder = env_encoder.to(device=config['device'])
    height_encoder = height_encoder.to(device=config['device'])

    floating_spheres_1 = np.load(f'{CCAI_PATH}/floating_spheres_1.npz')['sdf_grid']
    floating_spheres_1 = torch.from_numpy(floating_spheres_1).to(device=config['device']
                                                                 ).reshape(1, 1, 64, 64, 64).to(dtype=torch.float32)
    floating_spheres_1 = env_encoder(floating_spheres_1)

    floating_spheres_2 = np.load(f'{CCAI_PATH}/floating_spheres_2.npz')['sdf_grid']
    floating_spheres_2 = torch.from_numpy(floating_spheres_2).to(device=config['device']
                                                                 ).reshape(1, 1, 64, 64, 64).to(dtype=torch.float32)

    floating_spheres_2 = env_encoder(floating_spheres_2)

    tabletop_ycb = np.load(f'{CCAI_PATH}/tabletop_ycb.npz')['sdf_grid']
    tabletop_ycb = torch.from_numpy(tabletop_ycb).to(device=config['device']
                                                     ).reshape(1, 1, 64, 64, 64).to(dtype=torch.float32)
    tabletop_ycb = env_encoder(tabletop_ycb)

    object_ids = []

    for fpath in fpaths:
        path = pathlib.Path(fpath)

        goal = np.zeros(3)
        for p in path.rglob('*trajectory.npz'):
            data = np.load(p, allow_pickle=True)
            goal[:2] = data['goal'][:2]
            goal[2] = data['height']
            start = data['x'][:-1]
            goals.append(goal.copy())
            starts.append(start.copy())
            heights.append(np.array([data['height']]).reshape(-1))

            if 'floating_spheres_1' in fpath:
                sdfs.append(floating_spheres_1)
                object_ids.append(1)
            elif 'floating_spheres_2' in fpath:
                sdfs.append(floating_spheres_2)
                object_ids.append(2)
            elif 'tabletop_ycb' in fpath:
                sdfs.append(tabletop_ycb)
                object_ids.append(3)

    # create tensors and send to gpu
    starts = torch.from_numpy(np.stack(starts, axis=0)).to(device=config['device'], dtype=torch.float32)
    goals = torch.from_numpy(np.stack(goals, axis=0)).to(device=config['device'], dtype=torch.float32)
    heights = torch.from_numpy(np.stack(heights, axis=0)).to(device=config['device'], dtype=torch.float32)
    sdfs = torch.stack(sdfs, dim=0).squeeze(1)
    object_ids = np.array(object_ids)

    print(starts.shape, goals.shape, heights.shape, sdfs.shape)
    B1, B2, _ = starts.shape

    # only consider first 10 of B2
    B2 = 10
    starts = starts[:, :B2]
    N = 16

    heights_enc = height_encoder(heights.reshape(-1)).reshape(B1, -1)
    constraints = torch.stack((sdfs, heights_enc), dim=1)
    constraint_codes = torch.stack((torch.tensor([1.0, 0.0], device=config['device']),
                                    torch.tensor([0.0, 1.0], device=config['device'])), dim=0).expand(B1, -1, -1)
    constraints = torch.cat((constraints, constraint_codes), dim=2)

    if not config['height']:
        constraints = constraints[:, 0, :].unsqueeze(1)
    elif not config['obj']:
        constraints = constraints[:, 1, :].unsqueeze(1)

    st = starts.reshape(B1, B2, 1, -1).expand(-1, -1, N, -1)
    gl = goals.reshape(B1, 1, 1, -1).expand(-1, B2, N, -1)
    cn = constraints.reshape(B1, 1, 1, -1, 66).expand(-1, B2, N, -1, -1)
    poses = [
        {'obs': torch.tensor([[[1.0000, 0.0000, 0.0000, 0.7500],
                               [0.0000, 1.0000, 0.0000, 0.2500],
                               [0.0000, 0.0000, 1.0000, 1.0000],
                               [0.0000, 0.0000, 0.0000, 1.0000]]], device='cuda:0')},

        {'obs': torch.tensor([[[1.0000, 0.0000, 0.0000, 0.7500],
                               [0.0000, 1.0000, 0.0000, 0.2500],
                               [0.0000, 0.0000, 1.0000, 1.0000],
                               [0.0000, 0.0000, 0.0000, 1.0000]]], device='cuda:0')},
        {'mug1': torch.tensor([[[0.0000, -1.0000, 0.0000, 0.6000],
                                [1.0000, 0.0000, 0.0000, 0.2000],
                                [0.0000, 0.0000, 1.0000, 0.8010],
                                [0.0000, 0.0000, 0.0000, 1.0000]]], device='cuda:0'),
         'mug2': torch.tensor([[[0.0000, -1.0000, 0.0000, 0.9500],
                                [1.0000, 0.0000, 0.0000, 0.3250],
                                [0.0000, 0.0000, 1.0000, 0.8010],
                                [0.0000, 0.0000, 0.0000, 1.0000]]], device='cuda:0'),
         'pitcher': torch.tensor([[[0.0000, 1.0000, 0.0000, 0.6750],
                                   [-1.0000, 0.0000, -0.0000, 0.3000],
                                   [-0.0000, 0.0000, 1.0000, 0.8010],
                                   [0.0000, 0.0000, 0.0000, 1.0000]]], device='cuda:0')}
    ]

    # now we want to evaluate the quality of each trajectory
    obj_types = ['floating_spheres_1', 'floating_spheres_2', 'tabletop_ycb']

    costs = torch.zeros((B1, B2, N))
    av_g = torch.zeros((B1, B2, N))
    av_h = torch.zeros((B1, B2, N))

    update_chain(config['device'])
    for b1 in range(B1):
        for b2 in range(B2):
            print(b1, b2)
            # need to sample individually for each problem due to guided diffusion
            problem = VictorTableProblem(starts[b1, b2], goals[b1],
                                         12, device=config['device'],
                                         obstacle_poses=poses[object_ids[b1]],
                                         table_height=heights[b1],
                                         obstacle_type=obj_types[object_ids[b1] - 1],
                                         flow_model=None,
                                         constr_params=None)

            trajectory_sampler.model.diffusion_model.problem = problem
            samples = trajectory_sampler.sample(st[b1, b2], gl[b1, b2], cn[b1, b2])
            samples = samples.reshape(N, -1, 7)[:, 1:, :]

            # evalaute cost
            J, _, _ = problem._objective(samples)
            g, _, _ = problem._con_ineq(samples, compute_grads=False)
            h, _, _ = problem._con_eq(samples, compute_grads=False)
            print(J.mean(), h.abs().mean())
            # collect together all costs and constraint violations
            costs[b1, b2, :] = J.reshape(-1)
            av_h[b1, b2, :] = torch.mean(torch.abs(h), dim=-1)
            av_g[b1, b2, :] = torch.mean(torch.clamp(g, min=0), dim=-1)

        # now we can save the results, including object ids for identifying diffferent environments
        np.savez(f'{CCAI_PATH}/data/victor_table_{config["test_name"]}.npz',
                 costs=costs[:b1 + 1].cpu().numpy(),
                 av_h=av_h[:b1 + 1].cpu().numpy(),
                 av_g=av_g[:b1 + 1].cpu().numpy(),
                 object_ids=object_ids[:b1])

    print(costs.mean(), costs.std())
    print('inequality')
    print(av_g.mean(), av_g.std())
    print('equality')
    print(av_h.mean(), av_h.std())


import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='victor_table_flow_matching.yaml')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    # load config
    config = yaml.safe_load(
        pathlib.Path(f'{CCAI_PATH}/config/training_configs/{args.config}').read_text())

    # overwrite config device with argument, just to make it easier to launch multiple jobs from command line
    if args.device is not None:
        config['device'] = args.device

    if config['constrained'] or config['guided']:
        constrained = config['constrained']
        p = {'obs': torch.tensor([[[1.0000, 0.0000, 0.0000, 0.7500],
                                   [0.0000, 1.0000, 0.0000, 0.2500],
                                   [0.0000, 0.0000, 1.0000, 1.0000],
                                   [0.0000, 0.0000, 0.0000, 1.0000]]], device='cuda:0')},

        problem = VictorTableProblem(torch.zeros(7, device=config['device']),
                                     torch.zeros(3, device=config['device']),
                                     12, device=config['device'],
                                     obstacle_poses=p,
                                     table_height=0.8,
                                     obstacle_type='floating_spheres_1',
                                     flow_model=None,
                                     constr_params=None)
    else:
        problem = None
        constrained = False

    trajector_sampler = TrajectorySampler(T=12 + 1, dx=7, du=0,
                                          context_dim=7 + 3 + config['constraint_embedding_dim'] + 2,
                                          type=config['model_type'], timesteps=config['timesteps'],
                                          hidden_dim=config['hidden_dim'],
                                          problem=problem,
                                          constrain=constrained,
                                          unconditional=config['unconditional'])

    env_encoder = Conv3DEncoder(config['constraint_embedding_dim'])
    from torch import nn

    # height_encoder = nn.Identity()
    height_encoder = SinusoidalPosEmb(config['constraint_embedding_dim'])
    # height_encoder = nn.Linear(1, config['constraint_embedding_dim'])

    if config['test_samples']:
        test_samples(trajectory_sampler=trajector_sampler, env_encoder=env_encoder, height_encoder=height_encoder,
                     config=config)
    else:
        from torch.utils.data import DataLoader, RandomSampler

        data_path = pathlib.Path(f'{CCAI_PATH}/data/training_data/{config["data_directory"]}')

        train_dirs = [str(d) for d in data_path.glob('*train_data*')]
        test_dirs = [str(d) for d in data_path.glob('*test_data*')]
        train_dataset_reach = VictorReachingDataset([d for d in train_dirs if 'reacher' in d])
        train_dataset_table = VictorTableHeightDataset([d for d in train_dirs if 'table' in d])
        test_dataset_reach = VictorReachingDataset([d for d in test_dirs if 'reacher' in d])
        test_dataset_table = VictorTableHeightDataset([d for d in test_dirs if 'table' in d])

        # OK, test only training on the table dataset
        # train_dataset_reach, train_dataset_table = torch.utils.data.random_split(train_dataset_table,
        #                                                                         [len(train_dataset_table) // 2,
        #                                                                          len(train_dataset_table) // 2])
        # test_dataset_reach, test_dataset_table = torch.utils.data.random_split(test_dataset_table#,
        #                                                                       [len(test_da#taset_table) // 2,
        # # #                                                                      len(test_dataset_table) // 2])

        if config['normalize_data']:
            # normalize data
            all_trajectories = np.concatenate((train_dataset_table.trajectories,
                                               train_dataset_reach.trajectories), axis=0)

            mu = np.mean(all_trajectories, axis=(0, 1, 2, 3))
            std = np.std(all_trajectories, axis=(0, 1, 2, 3))

            train_dataset_table.set_norm_constants(mu, std)
            train_dataset_reach.set_norm_constants(mu, std)
            test_dataset_table.set_norm_constants(mu, std)
            test_dataset_reach.set_norm_constants(mu, std)
            trajector_sampler.set_norm_constants(mu, std)


        def get_train_loader(dataset):
            sampler = RandomSampler(dataset)

            loader = DataLoader(dataset, batch_size=config['batch_size'],
                                sampler=sampler, num_workers=4, pin_memory=True)

            return loader


        train_loader_reach = get_train_loader(train_dataset_reach)
        train_loader_table = get_train_loader(train_dataset_table)
        val_loader_reach = DataLoader(test_dataset_reach, batch_size=512, shuffle=False, num_workers=4)
        val_loader_table = DataLoader(test_dataset_table, batch_size=512, shuffle=False, num_workers=4)

        trajectory_sampler = trajector_sampler.to(device=config['device'])
        env_encoder = env_encoder.to(device=config['device'])

        height_encoder = height_encoder.to(device=config['device'])
        train_model(trajectory_sampler, env_encoder, height_encoder,
                    train_loader_reach, train_loader_table,
                    val_loader_reach, val_loader_table, config)
