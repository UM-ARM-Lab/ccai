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

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

def test(trajectory_sampler, env_encoder, height_encoder, reach_loader, table_loader, config):
    loss = 0.0
    #with torch.no_grad():
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
    import time
    epochs = config['epochs']
    pbar = tqdm.tqdm(range(epochs))
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

        if (epoch + 1) % config['test_every'] == 0:
            if config['use_ema']:
                test_model = ema_model
            else:
                test_model = trajectory_sampler
            # sample trajectories and save for plotting
            #train_loss = test(test_model, env_encoder, height_encoder, train_loader_reach, train_loader_table, config)
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
                break

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
            }
            torch.save(checkpoint, f'{fpath}/checkpoint.pth')


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    config = yaml.safe_load(
        pathlib.Path(f'{CCAI_PATH}/config/training_configs/victor_table_diffusion.yaml').read_text())

    trajector_sampler = TrajectorySampler(T=12+1, dx=7, du=0, context_dim=7 + 3 + config['constraint_embedding_dim'] + 2,
                                          type=config['model_type'], timesteps=config['timesteps'],
                                          hidden_dim=config['hidden_dim'])
    env_encoder = Conv3DEncoder(config['constraint_embedding_dim'])
    from torch import nn

    # height_encoder = nn.Identity()
    height_encoder = SinusoidalPosEmb(config['constraint_embedding_dim'])
    #height_encoder = nn.Linear(1, config['constraint_embedding_dim'])

    from torch.utils.data import DataLoader, RandomSampler

    data_path = pathlib.Path(f'{CCAI_PATH}/data/training_data/{config["data_directory"]}')

    train_dirs = [str(d) for d in data_path.glob('*train_data*')]
    test_dirs = [str(d) for d in data_path.glob('*test_data*')]
    train_dataset_reach = VictorReachingDataset([d for d in train_dirs if 'reacher' in d])
    train_dataset_table = VictorTableHeightDataset([d for d in train_dirs if 'table' in d])
    test_dataset_reach = VictorReachingDataset([d for d in test_dirs if 'reacher' in d])
    test_dataset_table = VictorTableHeightDataset([d for d in test_dirs if 'table' in d])

    # OK, test only training on the table dataset
    #train_dataset_reach, train_dataset_table = torch.utils.data.random_split(train_dataset_table,
    #                                                                         [len(train_dataset_table) // 2,
    #                                                                          len(train_dataset_table) // 2])
    #test_dataset_reach, test_dataset_table = torch.utils.data.random_split(test_dataset_table#,
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
