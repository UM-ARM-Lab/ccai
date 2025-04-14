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
import pytorch_kinematics.transforms as tf
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
    parser.add_argument('--config', type=str, default='allegro_valve_diffusion.yaml')
    # parser.add_argument('--config', type=str, default='allegro_screwdriver_diffusion_recovery_best_traj_only_gen_sim_data.yaml')
    return parser.parse_args()


def visualize_trajectories(trajectories, scene, fpath, headless=False):
    for n, trajectory in enumerate(trajectories):
        state_trajectory = trajectory[:, :16].clone()
        if state_trajectory.shape[1] == 16:
            state_trajectory[:, 15] *= 0
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/kin/img'), parents=True, exist_ok=True)
        pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/kin/gif'), parents=True, exist_ok=True)
        visualize_trajectory(state_trajectory, scene, f'{fpath}/trajectory_{n + 1}/kin', headless=headless,
                             fingers=fingers, obj_dof=4)
        # Visualize what happens if we execute the actions in the trajectory in the simulator
        # pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/sim/img'), parents=True, exist_ok=True)
        # pathlib.Path.mkdir(pathlib.Path(f'{fpath}/trajectory_{n + 1}/sim/gif'), parents=True, exist_ok=True)
        #visualize_trajectory_in_sim(trajectory, config['env'], f'{fpath}/trajectory_{n + 1}/sim')
        # save the trajectory
        # np.save(f'{fpath}/trajectory_{n + 1}/traj.npz', trajectory.cpu().numpy())


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

def train_model_state_only(trajectory_sampler, train_loader, config):
    fpath = f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}'
    pathlib.Path.mkdir(pathlib.Path(fpath), parents=True, exist_ok=True)
    dt = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run = wandb.init(project='ccai-screwdriver', entity='abhinavk99', config=config,
                     name=f'{config["model_name"]}_{config["model_type"]}_{dt}')


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
        flow_loss = 0.0
        action_loss = 0.0
        trajectory_sampler.train()
        for trajectories, traj_class, masks in tqdm.tqdm(train_loader):
            trajectories = trajectories.to(device=config['device'])
            masks = masks.to(device=config['device'])

            if config['use_class']:
                traj_class = traj_class.to(device=config['device']).float()
            else:
                traj_class = None
            sampler_loss = trajectory_sampler.loss(trajectories, mask=masks, constraints=traj_class)
            loss = sampler_loss['loss']
            loss.backward()
            # wandb.log({'train_loss': loss.item()})
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # for param in trajectory_sampler.parameters():
            #    print(param.grad)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            flow_loss += sampler_loss['flow_loss'].item()
            action_loss += sampler_loss['action_loss'].item()
            step += 1
            if config['use_ema']:
                if step % 10 == 0:
                    update_ema(trajectory_sampler)

        train_loss /= len(train_loader)
        flow_loss /= len(train_loader)
        action_loss /= len(train_loader)
        pbar.set_description(
            f'Train loss {train_loss:.3f}')
        try:
            wandb.log({
                    'train_loss_epoch': train_loss,
                    'flow_loss_epoch': flow_loss,
                    'action_loss_epoch': action_loss,
                    'time': time.time()
                })
        except:
            print('Could not log to wandb')
            print({
                    'train_loss_epoch': train_loss,
                    'flow_loss_epoch': flow_loss,
                    'action_loss_epoch': action_loss,
                    'time': time.time()
            })

        if (epoch + 1) % config['save_every'] == 0:
            if config['use_ema']:
                torch.save(ema_model.state_dict(), f'{fpath}/allegro_screwdriver_{config["model_type"]}_state_only_{epoch+1}_{train_loss:.4f}.pt')
            else:
                torch.save(model.state_dict(),
                           f'{fpath}/allegro_screwdriver_{config["model_type"]}_state_only_{epoch+1}_{train_loss:.4f}.pt')
    if config['use_ema']:
        torch.save(ema_model.state_dict(), f'{fpath}/allegro_screwdriver_{config["model_type"]}_state_only_{epoch+1}_{train_loss:.4f}.pt')
    else:
        torch.save(model.state_dict(),
                   f'{fpath}/allegro_screwdriver_{config["model_type"]}_state_only_{epoch+1}_{train_loss:.4f}.pt')

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

    # Check if simulated_trajectories_{name}.npz already exists. If it does, return
    if name is not None:
        if pathlib.Path(f'{fpath}/simulated_trajectories_{name}.npz').exists():
            return
    # assume we want a dataset size as large as the original dataset
    N = len(loader.dataset)
    loader.dataset.update_masks(p1=1.0, p2=1.0)  # no masking

    # generate three subtrajectories
    num_sub_traj = 1

    ACTION_DICT = {
        0: torch.tensor([[1.0, -1.0, -1.0]]),  # regrasp thumb / middle
        1: torch.tensor([[-1.0, 1.0, 1.0]]),  # regrasp index
    }
    ACTION_TENSOR = torch.cat([ACTION_DICT[i] for i in ACTION_DICT.keys()], dim=0).to(device=config['device'])

    simulated_trajectories = []
    simulated_class = []
    dataset_size = 0

    for trajectories, traj_class, _ in tqdm.tqdm(train_loader):

        if train_loader.dataset.cosine_sine:
            dx = 16
        else:
            dx = 15

        B = trajectories.shape[0]
        trajectories = trajectories.to(device=config['device'])
        traj_class = traj_class.to(device=config['device'])
        # p = torch.tensor([0.5, 0.5], device=config['device'])
        # idx = p.multinomial(num_samples=((num_sub_traj - 1) * B), replacement=True)
        # _next_class = ACTION_TENSOR[idx].reshape(B, -1, 3)

        # traj_class = torch.cat((traj_class.reshape(B, 1, 3), _next_class), dim=1)
        start = (trajectories[:N, 0, :dx] * model.x_std[:dx].to(device=trajectories.device) +
                 model.x_mean[:dx].to(device=trajectories.device))

        new_traj, _, _ = model.sample(N=B, H=num_sub_traj * config['T'], start=start, constraints=traj_class, skip_likelihood=True)

        if config['sine_cosine']:
            new_traj = convert_sine_cosine_to_yaw(new_traj)

        # visualize_trajectories(new_traj, config['scene'], fpath, headless=False)


        simulated_trajectories.append(new_traj.detach().cpu())
        simulated_class.append(traj_class.detach().cpu())

        dataset_size += B * num_sub_traj
        if dataset_size >= N:
            break

    simulated_trajectories = torch.cat(simulated_trajectories, dim=0).numpy()[:N]
    simulated_class = torch.cat(simulated_class, dim=0).numpy()[:N]

    # save the new dataset

    np.savez(f'{fpath}/simulated_trajectories_{name}.npz', trajectories=simulated_trajectories, contact=simulated_class)

def likelihood_ecdf_calc(model, train_loader, config):
    fpath = f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}'
    pathlib.Path.mkdir(pathlib.Path(fpath), parents=True, exist_ok=True)

    N = 8
    # x = 5336
    # y = 7987-5336
    # model.model.diffusion_model.classifier.class_loss = torch.nn.CrossEntropyLoss(torch.tensor([1, y/x]).to(device=config['device']))

    # Loop through the dataset and evaluate the likelihood of the training data
    model.model.diffusion_model.classifier = None
    model.eval()

    train_likelihoods = {
        'index': [],
        'thumb_middle': [],
    }


    for trajectories, traj_class, _ in tqdm.tqdm(train_loader):

        initial_state = trajectories[:, 0, :16]
        initial_state = initial_state.to(device=config['device'])
        traj_class = traj_class.to(device=config['device'])

        if traj_class.sum().item() == -1:
            mode = 'thumb_middle'
        else:
            mode = 'index'

        _, _, likelihood = model.sample(N*trajectories.shape[0], H=config['T'], start=initial_state.repeat_interleave(N, 0),
                                        constraints=traj_class.repeat_interleave(N, 0))
        likelihood = likelihood.reshape(-1, N).mean(1)
        train_likelihoods[mode] += likelihood.tolist()

        for mode in train_likelihoods.keys():
            likelihood_mean = np.mean(train_likelihoods[mode])
            likelihood_std = np.std(train_likelihoods[mode])
            print(f'Mode {mode}: train likelihood mean: {likelihood_mean:.3f} std: {likelihood_std:.3f}')

        # Get label for traj_class by summing across dim 1 and then looking up in mode_dict
    with open(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/gen_train_likelihoods_by_mode.pkl', 'wb') as f:
        pickle.dump(train_likelihoods, f)

    for mode in train_likelihoods.keys():
        likelihoods = -np.array(train_likelihoods[mode])
        plt.hist(likelihoods, bins=75)
        plt.xlabel('likelihood')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of training likelihoods for mode {mode}')
        plt.savefig(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/train_likelihood_hist_{mode}.png')
        plt.clf()
    # torch.save(model.state_dict(),
    #            f'{fpath}/allegro_screwdriver_{config["model_type"]}_w_classifier.pt')


def train_classifier(model: TrajectorySampler, train_loader: DataLoader, val_loader: DataLoader, config: Dict) -> None:
    """
    Train an MLP classifier that predicts contact modes from states.
    
    Args:
        model: The trajectory sampler model whose classifier will be trained
        train_loader: DataLoader providing training data
        config: Configuration parameters
    """
    fpath = f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}'
    pathlib.Path.mkdir(pathlib.Path(fpath), parents=True, exist_ok=True)
    
    # Initialize a classifier MLP with appropriate dimensions
    model.model.diffusion_model.classifier = nn.Sequential(
        nn.Linear(config['dx'], 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 3),  # 3 binary outputs for contact mode
    ).to(config['device'])
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.model.diffusion_model.classifier.parameters(), lr=1e-4)
    
    # Use binary cross entropy loss for each dimension
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Training loop
    model.model.diffusion_model.classifier.train()
    
    num_epochs = config.get('classifier_epochs', config['epochs'])
    pbar = tqdm.tqdm(range(num_epochs))
    
    # Training metrics
    best_loss = float('inf')
    best_accuracy = 0.0
    
    # x_mean = train_loader.dataset.trajectories[:, 0, :config['dx']].mean(dim=0)
    # x_std = train_loader.dataset.trajectories[:, 0, :config['dx']].std(dim=0)
    
    for epoch in pbar:
        total_loss = 0.0
        total_accuracy = 0.0
        total_batches = 0
        
        for trajectories, traj_class, _ in train_loader:
            # Extract the initial state from each trajectory
            initial_states = trajectories[:, 0, :config['dx']].to(device=config['device'])

            # Normalize initial states
            # initial_states = (initial_states - model.x_mean[:config['dx']].to(device=config['device'])) / model.x_std[:config['dx']].to(device=config['device'])
            traj_class = (traj_class + 1) / 2
            # Process target contact modes
            targets = traj_class.to(device=config['device']).float()
            
            # Forward pass
            logits = model.model.diffusion_model.classifier(initial_states)
            
            # Calculate loss
            loss = loss_fn(logits, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy (correct predictions across all dimensions)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predictions == targets).all(dim=1).float().mean()
            
            # Update metrics
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_batches += 1
        
        # Calculate epoch metrics
        epoch_loss = total_loss / total_batches
        epoch_accuracy = total_accuracy / total_batches
        
        # Update progress bar
        pbar.set_description(f'Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}')
        
        
        # Evaluate on a validation set
        model.model.diffusion_model.classifier.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for trajectories, traj_class, _ in val_loader:
                initial_states = trajectories[:, 0, :config['dx']].to(device=config['device'])
                traj_class = (traj_class + 1) / 2
                targets = traj_class.to(device=config['device']).float()
                
                logits = model.model.diffusion_model.classifier(initial_states)
                loss = loss_fn(logits, targets)
                predictions = (torch.sigmoid(logits) > 0.5).float()
                accuracy = (predictions == targets).all(dim=1).float().mean()
                
                val_loss += loss.item()
                val_accuracy += accuracy.item()
                val_batches += 1
        
        val_loss /= val_batches
        val_accuracy /= val_batches
        
        print(f'Validation loss: {val_loss:.4f} | Validation accuracy: {val_accuracy:.4f}')
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_loss = epoch_loss
            torch.save(model.state_dict(), f'{fpath}/allegro_screwdriver_{config["model_type"]}_contact_classifier.pt')
            print(f'Saved best model with accuracy: {best_accuracy:.4f} and loss: {best_loss:.4f}')

    # Evaluate confusion matrix on validation data
    model.model.diffusion_model.classifier.eval()
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for trajectories, traj_class, _ in val_loader:
            initial_states = trajectories[:, 0, :config['dx']].to(device=config['device'])
            targets = traj_class.to(device=config['device'])
            
            logits = model.model.diffusion_model.classifier(initial_states)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
    
    all_targets = (np.concatenate(all_targets, axis=0) + 1) / 2
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Print per-dimension accuracy
    for dim in range(3):
        dim_accuracy = np.mean(all_targets[:, dim] == all_predictions[:, dim])
        print(f'Dimension {dim} accuracy: {dim_accuracy:.4f}')
        cm = confusion_matrix(all_targets[:, dim], all_predictions[:, dim])
        print(f'Confusion matrix for dimension {dim}:')
        print(cm)
    
    print(f'Final model saved with accuracy: {best_accuracy:.4f}')

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

def eval_train_likelihood(model, train_loader, config, mode=None):
    # with open(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/gen_train_likelihoods_all_states.pkl', 'rb') as f:
    #     train_likelihoods = pickle.load(f)

    # train_likelihoods = -np.array(train_likelihoods)
    # # plt.hist(train_likelihoods, bins=75)
    # # plt.xlabel('likelihood')
    # # plt.ylabel('Frequency')
    # # plt.title('Histogram of training likelihoods')
    # # plt.savefig(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/train_likelihood_hist.png')
    # # Fit the data to the lognormal distribution
    # alpha, loc, beta = scipy.stats.gamma.fit((train_likelihoods))
    # lognorm = scipy.stats.gamma(alpha, loc=loc, scale=beta)
    # print(f'Fitted alpha: {alpha}, loc: {loc}, beta: {beta}')
    # # Plot the histogram
    # plt.hist(train_likelihoods, bins=75, density=True, alpha=0.6, color='g')
    # # Plot the PDF.
    # x_min = min(train_likelihoods)
    # x_max = max(train_likelihoods)
    # x = np.linspace(x_min, x_max, 1000)
    # p = lognorm.pdf(x)
    # c = lognorm.cdf(x)

    # plt.ylim((0, .2))
    # # p = p[::-1]
    # plt.plot(x, p, 'k', linewidth=2, label='PDF')
    # # plt.plot(x, c, 'r', linewidth=2, label='CDF')
    # title = "Gamma fit results: alpha = %.2f, loc = %.2f, beta = %.2f" % (alpha, loc, beta)
    # plt.title(title)
    # plt.legend()
    # plt.savefig(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/gen_train_likelihood_hist_fit_all_states.png')

    N = 16
    contact = torch.ones(N, 3).to(device=config['device'])
    if mode == 'index':
        contact[:, 0] = -1.0
    elif mode == 'thumb_middle':
        contact[:, 1] = -1.0
        contact[:, 2] = -1.0

    # Loop through the dataset and evaluate the likelihood of the training data
    model.model.diffusion_model.classifier = None
    model.eval()
    print(f'Evaluating likelhood on training dset (all states) of model {config["model_name"]}_{config["model_type"]}\n')
    train_likelihoods = []
    for (trajectories, ind) in (tqdm.tqdm(train_loader)):
        trajectories = trajectories.to(device=config['device'])
        trajectories = trajectories.flatten(0, 1)
        traj_class = None
        with torch.no_grad():
            # likelihood = model.model.diffusion_model.approximate_likelihood(trajectories, context=traj_class)
            # print(likelihood)
            # start = trajectories[:, 0, :15]
            start_sine_cosine = trajectories[:, :16]
            _, _, likelihood = model.sample(N*trajectories.shape[0], H=config['T'], start=start_sine_cosine.repeat_interleave(N, 0),
                                            constraints=contact.repeat_interleave(trajectories.shape[0], 0))
            likelihood = likelihood.reshape(-1, N).mean(1)
            print(likelihood)

        train_likelihoods += likelihood.tolist()

        likelihood_mean = np.mean(train_likelihoods)
        likelihood_std = np.std(train_likelihoods)
        print(f'Train likelihood mean: {likelihood_mean:.3f} std: {likelihood_std:.3f}')
    with open(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/gen_train_likelihoods_all_states.pkl', 'wb') as f:
        pickle.dump(train_likelihoods, f)
    likelihood_mean = np.mean(train_likelihoods)
    likelihood_std = np.std(train_likelihoods)
    print(f'Train likelihood mean: {likelihood_mean:.3f} std: {likelihood_std:.3f}')

    train_likelihoods = -np.array(train_likelihoods)
    # plt.hist(train_likelihoods, bins=75)
    # plt.xlabel('likelihood')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of training likelihoods')
    # plt.savefig(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/train_likelihood_hist.png')
    # Fit the data to the lognormal distribution
    # alpha, loc, beta = scipy.stats.gamma.fit((train_likelihoods))
    # lognorm = scipy.stats.gamma(alpha, loc=loc, scale=beta)
    # print(f'Fitted alpha: {alpha}, loc: {loc}, beta: {beta}')
    # Plot the histogram
    plt.hist(train_likelihoods, bins=75, density=True, alpha=0.6, color='g')
    # Plot the PDF.
    x_min = min(train_likelihoods)
    x_max = max(train_likelihoods)
    x = np.linspace(x_min, x_max, 1000)
    # p = lognorm.pdf(x)
    # c = lognorm.cdf(x)
    # p = p[::-1]
    # plt.plot(x, p, 'k', linewidth=2, label='PDF')
    # plt.plot(x, c, 'r', linewidth=2, label='CDF')
    # title = "Gamma fit results: alpha = %.2f, loc = %.2f, beta = %.2f" % (alpha, loc, beta)
    # plt.title(title)
    plt.legend()
    plt.savefig(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/gen_train_likelihood_hist_fit_all_states.png')
    #     import sys
    #     sys.exit()

def identify_OOD_states(model, train_loader, config):
    N = 16
    # alpha = .5491095797801617
    # loc = .5124469995498656
    # beta = 4.292522581313451
    # gamma = scipy.stats.gamma(alpha, loc=loc, scale=beta)
    
    # Loop through the dataset and evaluate the likelihood of the training data
    model.model.diffusion_model.classifier = None
    print(f'Identifying OOD states of model {config["model_name"]}_{config["model_type"]}\n')
    all_states = []
    all_trajectories = []
    all_likelihoods = []
    all_inds = []
    for i, (trajectories, ind, contact) in enumerate(tqdm.tqdm(train_loader)):
        trajectories = trajectories.to(device=config['device'])
        contact = contact.to(device=config['device'])
        B, T, dxu = trajectories.shape
        trajectories = trajectories.flatten(0, 1)
        # if config['use_class']:
        #     traj_class = traj_class.to(device=config['device']).float()
        #     traj_class = traj_class.repeat_interleave(T, 0)
        # else:
        #     traj_class = None
        with torch.no_grad():
            # likelihood = model.model.diffusion_model.approximate_likelihood(trajectories, context=traj_class)
            start_sine_cosine = trajectories[:, :16]
            start_yaw = convert_sine_cosine_to_yaw(start_sine_cosine)
            samples, _, likelihood = model.sample(N*trajectories.shape[0], H=config['T'], start=start_sine_cosine.repeat_interleave(N, 0),
                                                  constraints=contact.repeat_interleave(N, 0))
            likelihood = likelihood.reshape(-1, N).mean(1)
            samples = samples.cpu().numpy()
            all_trajectories.append(samples)
            all_likelihoods.append(likelihood.cpu().numpy())
            all_inds.append(ind)
            print(likelihood)

            # likelihood_mean = likelihood.reshape(-1, N).mean(1)
            # likelihood_mean = likelihood_mean.cpu().numpy()
            # cdf = gamma.cdf(-likelihood_mean)
            # idx = np.where(cdf > 0.99)[0]
            # print(f'Found {len(idx)} OOD states out of {len(likelihood_mean)}: {len(idx)/len(likelihood_mean)}')
            # print(likelihood_mean[idx])
            # torch_idx = torch.tensor(idx).to(device=config['device'])
            # ood_states = trajectories[torch_idx]
            all_states.append(start_yaw.cpu().numpy())

        if i % 25 == 0:
            all_trajectories_save = np.concatenate(all_trajectories, axis=0)
            all_likelihoods_save = np.concatenate(all_likelihoods, axis=0)
            all_states_save = np.concatenate(all_states, axis=0)
            all_inds_save = np.concatenate(all_inds, axis=0)

            print('All ood states:', all_states_save.shape)
            np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/ood_states.npy', all_states_save)
            np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/candidate_ood_likelihoods.npy', all_likelihoods_save)
            np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/candidate_ood_trajectories.npy', all_trajectories_save)
            np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/candidate_ood_inds.npy', all_inds_save)
    all_trajectories_save = np.concatenate(all_trajectories, axis=0)
    all_likelihoods_save = np.concatenate(all_likelihoods, axis=0)
    all_states_save = np.concatenate(all_states, axis=0)
    all_inds_save = np.concatenate(all_inds, axis=0)

    print('All ood states:', all_states_save.shape)
    np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/ood_states.npy', all_states_save)
    np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/candidate_ood_likelihoods.npy', all_likelihoods_save)
    np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/candidate_ood_trajectories.npy', all_trajectories_save)
    np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/candidate_ood_inds.npy', all_inds_save)

    import sys
    sys.exit()

def project_OOD_states(model, config):
    # ood_states = np.load(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/ood_states.npy')
    N = 16
    print(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/gen_train_likelihoods_all_states.pkl')
    train_likelihoods = np.load(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/gen_train_likelihoods_all_states.pkl', allow_pickle=True)
    
    # plt.hist(train_likelihoods, bins=750)
    # plt.xlabel('likelihood')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of training likelihoods (No Outliers)')
    # plt.xlim([-25, 0])
    # plt.savefig(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/train_likelihood_hist_no_outliers.png')

    ood_likelihoods = np.load(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/candidate_ood_likelihoods.npy')
    ood_states = np.load(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/ood_states.npy')

    quantile = np.quantile(train_likelihoods, config['quantile'])
    print(quantile)

    model.model.diffusion_model.cutoff = quantile
    ood_states = ood_states[(ood_likelihoods < quantile) & (ood_likelihoods > -250)]
    # ood_states = ood_states[(ood_likelihoods < quantile) & (ood_likelihoods > -75) & (ood_likelihoods < -45)]

    # project the ood states to the manifold
    model.model.diffusion_model.classifier = None
    model.eval()
    print(f'Projecting {ood_states.shape[0]} OOD states of model {config["model_name"]}_{config["model_type"]}\n')
    all_projected_samples = []
    all_all_losses = []
    all_all_samples = []
    all_all_likelihoods = []
    bs = config['batch_size']

    for i in tqdm.tqdm(range(0, len(ood_states), bs)):

        ood_states_batch = torch.tensor(ood_states[i:i+bs]).to(device=config['device'])
        ood_states_batch = convert_yaw_to_sine_cosine(ood_states_batch)
        # ood_likelihoods_batch = torch.tensor(ood_likelihoods[i:i+bs]).to(device=config['device'])
        # ood_trajectories_batch = torch.tensor(ood_trajectories[i:i+bs]).to(device=config['device'])
        # with torch.no_grad():
        projected_samples, _, _, _, (all_losses, all_samples, all_likelihoods) = model.sample(N*bs, H=config['T'], start=ood_states_batch, 
                constraints=torch.ones(N*bs, 3).to(device=config['device']), 
                project=True)
        all_projected_samples.append(projected_samples.cpu().numpy())
        all_all_losses.append(all_losses)
        all_all_samples.append([i.numpy() for i in all_samples])
        all_all_likelihoods.append([i.numpy() for i in all_likelihoods])

        if i % 20 == 0 or i == len(ood_states) - bs:
            all_projected_states_save = np.concatenate(all_projected_samples, axis=0)
            # all_all_losses_save = np.concatenate(all_all_losses, axis=0)
            # all_all_samples_save = np.concatenate(all_all_samples, axis=0)
            # all_all_likelihoods_save = np.concatenate(all_all_likelihoods, axis=0)

            np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/projected_samples.npy', all_projected_states_save)
            # np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/all_losses.npy', all_all_losses_save)
            # np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/all_samples.npy', all_all_samples_save)
            # np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/all_likelihoods.npy', all_all_likelihoods_save)
            pickle.dump(all_all_losses, open(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/all_losses.pkl', 'wb'))
            pickle.dump(all_all_samples, open(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/all_samples.pkl', 'wb'))
            pickle.dump(all_all_likelihoods, open(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/all_likelihoods.pkl', 'wb'))

def visualize_ood_projection(model, config):
    N = 16
    fpath = f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/ood_projection'
    projected_samples = np.load(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/projected_samples.npy')
    all_losses = pickle.load(open(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/all_losses.pkl', 'rb'))
    all_samples = pickle.load(open(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/all_samples.pkl', 'rb'))
    all_likelihoods = pickle.load(open(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/all_likelihoods.pkl', 'rb'))

    # train_likelihoods = np.load(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/gen_train_likelihoods_all_states.pkl', allow_pickle=True)
    
    # plt.hist(train_likelihoods, bins=125)
    # plt.xlabel('likelihood')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of training likelihoods')
    # # plt.xlim([-40, 0])
    # # plt.savefig(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/train_likelihood_hist_no_outliers.png')
    # plt.savefig(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/train_likelihood_hist_all_states_no_gamma_possible_outliers.png')
    # sys.exit()
    # plot the losses
    # plt.hist(all_losses, bins=100)
    # plt.xlabel('loss')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of losses')
    # plt.savefig(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/loss_hist.png')

    # # plot the likelihoods
    # plt.hist(all_likelihoods, bins=100)
    # plt.xlabel('likelihood')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of likelihoods')
    # plt.savefig(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/likelihood_hist.png')

    # # plot the samples
    # for i in range(10):
    #     visualize_trajectories(all_samples[i], config['scene'], f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/ood_samples_{i}', headless=False)

    # # plot the projected samples
    # for i in range(10):
    #     visualize_trajectories(projected_samples[i], config['scene'], f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/5_10_15/ood_projected_samples_{i}', headless=False)
    if config['model_name'] == 'allegro_high_force_high_eps_pi_6':
        min_likelihood = -22.618363952636717
        # min_likelihood = -30
    else:
        raise ValueError('Model name not recognized')

    print(min_likelihood)
    bs = config['batch_size']

    proj_data = []
    proj_data_succ = []
    proj_data_fail = []
    all_states = 0
    all_ood = 0
    all_succ_proj = 0
    all_fail_proj = 0

    for idx, (sample_list, likelihood_list) in (enumerate(zip(all_samples, all_likelihoods))):
        # if likelihood_list[-1]
        initial_likelihoods = likelihood_list[0].reshape(-1, N).mean(1)
        all_states += len(initial_likelihoods)
        idx_0 = np.where(initial_likelihoods < min_likelihood)[0]
        all_ood += len(np.where(initial_likelihoods < min_likelihood)[0])
        final_likelihoods = likelihood_list[-1].reshape(-1, N).mean(1)
        idx_final = np.where(final_likelihoods > min_likelihood)[0]
        # idx is intersection of both
        idx_full = np.intersect1d(idx_0, idx_final)
        all_succ_proj += len(idx_full)
        all_fail_proj += len(idx_0) - len(idx_full)
        
        for i in range(initial_likelihoods.shape[0]):
            this_idx_data = {
                'initial_likelihood': initial_likelihoods[i],
                'final_likelihood': final_likelihoods[i],
            }
            proj_path = []

            succ_str = 'Success' if i in idx_full else 'Fail'
            for proj_ind in range(len(sample_list)):

                initial_traj = sample_list[proj_ind]
                initial_traj = convert_sine_cosine_to_yaw(initial_traj)
                initial_traj = initial_traj.reshape(-1, N, 13, 36)[:, 0]
                initial_state_traj = initial_traj[..., :15]

                traj_for_viz = torch.from_numpy(initial_state_traj[i])
                traj_for_viz[:, :14] = traj_for_viz[:, :14] * model.x_std[:14].to(traj_for_viz.device) + model.x_mean[:14].to(traj_for_viz.device)
                
                if proj_ind == 0:
                    this_idx_data['initial_state'] = traj_for_viz[0]
                elif proj_ind == len(sample_list) - 1:
                    this_idx_data['final_state'] = traj_for_viz[0]
                
                traj_for_viz = torch.cat((
                    traj_for_viz,
                    torch.zeros((traj_for_viz.shape[0], 1)).to(traj_for_viz.device),
                ), dim=1)
                if config['vis_dataset']:
                    vid_path = f'{fpath}/{succ_str}/state_{bs*idx + i}_{initial_likelihoods[i]:.1f}_{final_likelihoods[i]:.1f}/proj_step_{proj_ind}'
                    # Pathlib with vid_path
                    pathlib.Path.mkdir(pathlib.Path(vid_path + '/img'), parents=True, exist_ok=True)
                    pathlib.Path.mkdir(pathlib.Path(vid_path + '/gif'), parents=True, exist_ok=True)
                    visualize_trajectory(traj_for_viz, config["scene"], vid_path, headless=False,
                                            fingers=fingers, obj_dof=4)
                
                proj_path.append(traj_for_viz[0])

            proj_path = torch.stack(proj_path, dim=0)
            if config['vis_dataset']:
                vid_path = f'{fpath}/{succ_str}/state_{bs*idx + i}_{initial_likelihoods[i]:.1f}_{final_likelihoods[i]:.1f}/proj_path'
                pathlib.Path.mkdir(pathlib.Path(vid_path + '/img'), parents=True, exist_ok=True)
                pathlib.Path.mkdir(pathlib.Path(vid_path + '/gif'), parents=True, exist_ok=True)
                visualize_trajectory(proj_path, config["scene"], vid_path, headless=False,
                                        fingers=fingers, obj_dof=4)
            this_idx_data['proj_path'] = proj_path
            if i in idx_full:
                proj_data_succ.append(this_idx_data)
            else:
                proj_data_fail.append(this_idx_data)
            proj_data.append(this_idx_data)
    print(f'All states: {all_states}, All OOD: {all_ood}, All Successful Projection: {all_succ_proj}, All Failed Projection: {all_fail_proj}')
    pathlib.Path.mkdir(pathlib.Path(fpath), parents=True, exist_ok=True)
    with open(f'{fpath}/proj_data_succ.pkl', 'wb') as f:
        pickle.dump(proj_data_succ, f)
    with open(f'{fpath}/proj_data_fail.pkl', 'wb') as f:
        pickle.dump(proj_data_fail, f)
    with open(f'{fpath}/proj_data.pkl', 'wb') as f:
        pickle.dump(proj_data, f)

def filter_bad_traj(model, train_loader, config):
    N = 8
    model.model.diffusion_model.classifier = None
    all_likelihoods = []
    for i, (trajectories, _, _) in enumerate(tqdm.tqdm(train_loader)):
        trajectories = trajectories.to(device=config['device'])
        B, T, dxu = trajectories.shape
        # trajectories = trajectories.flatten(0, 1)
        # if config['use_class']:
        #     traj_class = traj_class.to(device=config['device']).float()
        #     traj_class = traj_class.repeat_interleave(T, 0)
        # else:
        #     traj_class = None
        with torch.no_grad():
            # likelihood = model.model.diffusion_model.approximate_likelihood(trajectories, context=traj_class)
            end = trajectories[:, -1, :15]
            start_sine_cosine = convert_yaw_to_sine_cosine(end)
            _, _, likelihood = model.sample(N*trajectories.shape[0], H=config['T'], start=start_sine_cosine.repeat_interleave(N, 0))
            likelihood = likelihood.reshape(-1, N).mean(1)
            all_likelihoods.append(likelihood.cpu().numpy())


        if i % 25 == 0:
            all_likelihoods_save = np.concatenate(all_likelihoods, axis=0)

            np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/ood_likelihoods.npy', all_likelihoods_save)
    all_likelihoods_save = np.concatenate(all_likelihoods, axis=0)

    np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/ood_likelihoods.npy', all_likelihoods_save)
    # Hist plot of all likelihoods
    plt.hist(all_likelihoods_save, bins=100)
    plt.xlabel('likelihood')
    plt.ylabel('Frequency')
    plt.title('Histogram of likelihoods')
    plt.savefig(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/ood_likelihood_hist.png')
    import sys
    sys.exit()

if __name__ == "__main__":
    CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
    print(CCAI_PATH)
    args = get_args()
    torch.set_float32_matmul_precision('high')
    print(args.config)
    config = yaml.safe_load(
        pathlib.Path(f'{CCAI_PATH}/config/training/{args.config}').read_text())
    dx_original = config['dx']
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
                                                dx_original,
                                                cosine_sine=config['sine_cosine'],
                                                states_only=config['du'] == 0,
                                                skip_pregrasp=config['skip_pregrasp'],
                                                type_=config['model_type'],
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
