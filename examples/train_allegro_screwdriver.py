# from isaac_victor_envs.tasks.allegro import AllegroValveTurningEnv
# from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv

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
from ccai.dataset import AllegroScrewDriverDataset, AllegroScrewDriverStateDataset, FakeDataset, RealAndFakeDataset
from isaac_victor_envs.utils import get_assets_dir
from torch.utils.data import DataLoader, RandomSampler
from ccai.models.trajectory_samplers import TrajectorySampler
from ccai.utils.allegro_utils import partial_to_full_state, visualize_trajectory
from ccai.allegro_screwdriver_problem_diffusion import AllegroScrewdriverDiff
import pickle
import sys

import scipy

import datetime
import time

import wandb

TORCH_LOGS = "+dynamo"
TORCHDYNAMO_VERBOSE = 1
fingers = ['index', 'middle', 'thumb']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='allegro_screwdriver_diffusion_project_ood_states.yaml')
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
            # wandb.log({'train_loss': loss.item()})
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
        try:
            wandb.log({'train_loss_epoch': train_loss})
        except:
            print('Could not log to wandb')
            print({'train_loss_epoch': train_loss})
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
                torch.save(ema_model.state_dict(), f'{fpath}/allegro_screwdriver_{config["model_type"]}_{epoch+1}_{train_loss:.4f}.pt')
            else:
                torch.save(model.state_dict(),
                           f'{fpath}/allegro_screwdriver_{config["model_type"]}_{epoch+1}_{train_loss:.4f}.pt')
    if config['use_ema']:
        torch.save(ema_model.state_dict(), f'{fpath}/allegro_screwdriver_{config["model_type"]}_{epoch+1}_{train_loss:.4f}.pt')
    else:
        torch.save(model.state_dict(),
                   f'{fpath}/allegro_screwdriver_{config["model_type"]}_{epoch+1}_{train_loss:.4f}.pt')

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

def eval_train_likelihood(model, train_loader, config):
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
    # Loop through the dataset and evaluate the likelihood of the training data
    model.model.diffusion_model.classifier = None
    model.eval()
    print(f'Evaluating likelhood on training dset (all states) of model {config["model_name"]}_{config["model_type"]}\n')
    train_likelihoods = []
    for (trajectories) in (tqdm.tqdm(train_loader)):
        trajectories = trajectories.to(device=config['device'])
        traj_class = None
        with torch.no_grad():
            # likelihood = model.model.diffusion_model.approximate_likelihood(trajectories, context=traj_class)
            # print(likelihood)
            start = trajectories.flatten(0, 1)[:, :15]
            # start = trajectories[:, 0, :15]
            start_sine_cosine = convert_yaw_to_sine_cosine(start)
            _, _, likelihood = model.sample(N*trajectories.shape[0], H=config['T'], start=start_sine_cosine.repeat_interleave(N, 0))
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
    alpha, loc, beta = scipy.stats.gamma.fit((train_likelihoods))
    lognorm = scipy.stats.gamma(alpha, loc=loc, scale=beta)
    print(f'Fitted alpha: {alpha}, loc: {loc}, beta: {beta}')
    # Plot the histogram
    plt.hist(train_likelihoods, bins=75, density=True, alpha=0.6, color='g')
    # Plot the PDF.
    x_min = min(train_likelihoods)
    x_max = max(train_likelihoods)
    x = np.linspace(x_min, x_max, 1000)
    p = lognorm.pdf(x)
    c = lognorm.cdf(x)
    # p = p[::-1]
    plt.plot(x, p, 'k', linewidth=2, label='PDF')
    plt.plot(x, c, 'r', linewidth=2, label='CDF')
    title = "Gamma fit results: alpha = %.2f, loc = %.2f, beta = %.2f" % (alpha, loc, beta)
    plt.title(title)
    plt.legend()
    plt.savefig(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/gen_train_likelihood_hist_fit_all_states.png')
    #     import sys
    #     sys.exit()

def identify_OOD_states(model, train_loader, config):
    N = 8
    # alpha = .5491095797801617
    # loc = .5124469995498656
    # beta = 4.292522581313451
    # gamma = scipy.stats.gamma(alpha, loc=loc, scale=beta)
    
    # Loop through the dataset and evaluate the likelihood of the training data
    model.model.diffusion_model.classifier = None
    model.eval()
    print(f'Identifying OOD states of model {config["model_name"]}_{config["model_type"]}\n')
    all_states = []
    all_trajectories = []
    all_likelihoods = []
    all_inds = []
    for i, (trajectories, ind) in enumerate(tqdm.tqdm(train_loader)):
        trajectories = trajectories.to(device=config['device'])
        B, T, dxu = trajectories.shape
        trajectories = trajectories.flatten(0, 1)
        # if config['use_class']:
        #     traj_class = traj_class.to(device=config['device']).float()
        #     traj_class = traj_class.repeat_interleave(T, 0)
        # else:
        #     traj_class = None
        with torch.no_grad():
            # likelihood = model.model.diffusion_model.approximate_likelihood(trajectories, context=traj_class)
            start = trajectories[:, :15]
            start_sine_cosine = convert_yaw_to_sine_cosine(start)
            samples, _, likelihood = model.sample(N*trajectories.shape[0], H=config['T'], start=start_sine_cosine.repeat_interleave(N, 0))
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
            all_states.append(start.cpu().numpy())

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
    N = 8
    print(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/gen_train_likelihoods_all_states.pkl')
    train_likelihoods = np.load(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/gen_train_likelihoods_all_states.pkl', allow_pickle=True)
    
    # plt.hist(train_likelihoods, bins=750)
    # plt.xlabel('likelihood')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of training likelihoods (No Outliers)')
    # plt.xlim([-25, 0])
    # plt.savefig(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/train_likelihood_hist_no_outliers.png')

    ood_likelihoods = np.load(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/candidate_ood_likelihoods.npy')
    ood_states = np.load(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/ood_states.npy')

    quantile = np.quantile(train_likelihoods, .75)
    print(quantile)
    ood_states = ood_states[(ood_likelihoods < quantile) & (ood_likelihoods > -500)]
    # ood_states = ood_states[(ood_likelihoods < quantile) & (ood_likelihoods > -75) & (ood_likelihoods < -50)]
    ood_states = ood_states[:, :16]
    
    # project the ood states to the manifold
    model.model.diffusion_model.classifier = None
    model.eval()
    print(f'Projecting OOD states of model {config["model_name"]}_{config["model_type"]}\n')
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
        projected_samples, _, _, _, (all_losses, all_samples, all_likelihoods) = model.sample(N*bs, H=config['T'], start=ood_states_batch, project=True)
        all_projected_samples.append(projected_samples.cpu().numpy())
        all_all_losses.append(all_losses)
        all_all_samples.append([i.numpy() for i in all_samples])
        all_all_likelihoods.append([i.numpy() for i in all_likelihoods])

        if i % 20 == 0 or i == len(ood_states) - bs:
            all_projected_states_save = np.concatenate(all_projected_samples, axis=0)
            # all_all_losses_save = np.concatenate(all_all_losses, axis=0)
            # all_all_samples_save = np.concatenate(all_all_samples, axis=0)
            # all_all_likelihoods_save = np.concatenate(all_all_likelihoods, axis=0)

            np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/projected_samples.npy', all_projected_states_save)
            # np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/all_losses.npy', all_all_losses_save)
            # np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/all_samples.npy', all_all_samples_save)
            # np.save(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/all_likelihoods.npy', all_all_likelihoods_save)
            pickle.dump(all_all_losses, open(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/all_losses.pkl', 'wb'))
            pickle.dump(all_all_samples, open(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/all_samples.pkl', 'wb'))
            pickle.dump(all_all_likelihoods, open(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/all_likelihoods.pkl', 'wb'))

def visualize_ood_projection(model, config):
    N = 8
    fpath = f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/ood_projection'
    projected_samples = np.load(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/projected_samples.npy')
    all_losses = pickle.load(open(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/all_losses.pkl', 'rb'))
    all_samples = pickle.load(open(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/all_samples.pkl', 'rb'))
    all_likelihoods = pickle.load(open(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/all_likelihoods.pkl', 'rb'))

    # train_likelihoods = np.load(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/gen_train_likelihoods_all_states.pkl', allow_pickle=True)
    
    # plt.hist(train_likelihoods, bins=125)
    # plt.xlabel('likelihood')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of training likelihoods')
    # # plt.xlim([-40, 0])
    # # plt.savefig(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/train_likelihood_hist_no_outliers.png')
    # plt.savefig(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/train_likelihood_hist_all_states_no_gamma_possible_outliers.png')
    # sys.exit()
    # plot the losses
    # plt.hist(all_losses, bins=100)
    # plt.xlabel('loss')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of losses')
    # plt.savefig(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/loss_hist.png')

    # # plot the likelihoods
    # plt.hist(all_likelihoods, bins=100)
    # plt.xlabel('likelihood')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of likelihoods')
    # plt.savefig(f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/likelihood_hist.png')

    # # plot the samples
    # for i in range(10):
    #     visualize_trajectories(all_samples[i], config['scene'], f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/ood_samples_{i}', headless=False)

    # # plot the projected samples
    # for i in range(10):
    #     visualize_trajectories(projected_samples[i], config['scene'], f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/ood_projected_samples_{i}', headless=False)
    if config['model_name'] == 'allegro_high_force_high_eps_pi_6':
        # min_likelihood = -14.2147057056427
        min_likelihood = -30
    else:
        min_likelihood = -10.685428142547607

    print(min_likelihood)
    bs = config['batch_size']

    proj_data = []
    all_states = 0
    all_ood = 0
    all_succ_proj = 0
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

        
        for i in idx_full:
            # if bs*idx + i != 1776:
            #     continue
            # for i in range(initial_state_traj.shape[0]):
                # initial_likelihood = initial_likelihoods[i]
                # if initial_likelihood > -40 and initial_likelihood < -30 and (bs*idx+1) > 347:
                #     print('State', bs*idx + i)
                # else:
                #     continue
            this_idx_data = {
                'initial_likelihood': initial_likelihoods[i],
                'final_likelihood': final_likelihoods[i],
            }
            proj_path = []
            for proj_ind in range(len(sample_list)):

                initial_traj = sample_list[proj_ind]
                initial_traj = convert_sine_cosine_to_yaw(initial_traj)
                initial_traj = initial_traj.reshape(-1, N, 13, 36)[:, 0]
                initial_state_traj = initial_traj[..., :15]

                # if bs*idx + i == 1776:
                #     print(proj_ind)
                #     traj_for_viz = torch.from_numpy(initial_state_traj[i])
                #     traj_for_viz[:, :14] = traj_for_viz[:, :14] * model.x_std[:14].to(traj_for_viz.device) + model.x_mean[:14].to(traj_for_viz.device)
                #     print(traj_for_viz[0])
                #     print()
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
                # vid_path = f'{fpath}/state_{bs*idx + i}/proj_step_{proj_ind}'
                # # Pathlib with vid_path
                # pathlib.Path.mkdir(pathlib.Path(vid_path + '/img'), parents=True, exist_ok=True)
                # pathlib.Path.mkdir(pathlib.Path(vid_path + '/gif'), parents=True, exist_ok=True)
                # visualize_trajectory(traj_for_viz, config["scene"], vid_path, headless=False,
                #                         fingers=fingers, obj_dof=4)
                
                proj_path.append(traj_for_viz[0])

            proj_path = torch.stack(proj_path, dim=0)
            # vid_path = f'{fpath}/state_{bs*idx + i}/proj_path'
            # pathlib.Path.mkdir(pathlib.Path(vid_path + '/img'), parents=True, exist_ok=True)
            # pathlib.Path.mkdir(pathlib.Path(vid_path + '/gif'), parents=True, exist_ok=True)
            # visualize_trajectory(proj_path, config["scene"], vid_path, headless=False,
            #                         fingers=fingers, obj_dof=4)
            this_idx_data['proj_path'] = proj_path
            proj_data.append(this_idx_data)
    print(f'All states: {all_states}, All OOD: {all_ood}, All Successful Projection: {all_succ_proj}')
    with open(f'{fpath}/proj_data.pkl', 'wb') as f:
        pickle.dump(proj_data, f)

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
                              problem=problem_for_sampler if config['state_control_only'] else None,)

    data_path = pathlib.Path(f'{CCAI_PATH}/data/training_data/{config["data_directory"]}')
    if config['eval_train_likelihood'] or config['id_ood_states']:
        train_dataset = AllegroScrewDriverStateDataset([p for p in data_path.glob('*train_data*')],
                                                config['T']-1,
                                                cosine_sine=config['sine_cosine'],
                                                states_only=config['du'] == 0,
                                                skip_pregrasp=config['skip_pregrasp'],
                                                type=config['model_type'],)
    else:
        train_dataset = AllegroScrewDriverDataset([p for p in data_path.glob('*train_data*')],
                                                config['T']-1,
                                                cosine_sine=config['sine_cosine'],
                                                states_only=config['du'] == 0,
                                                skip_pregrasp=config['skip_pregrasp'],
                                                type=config['model_type'],)
    train_dataset.update_masks(p1=1, p2=1)
    if config['normalize_data']:
        # normalize data
        train_dataset.compute_norm_constants()
        model.set_norm_constants(*train_dataset.get_norm_constants())

    print('Dset size', len(train_dataset))
    
    print(train_dataset.mean)
    # train_sampler = RandomSampler(train_dataset)
    train_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True,
                              shuffle=False)

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
        if config['discriminator_guidance']:
            model_name = f'allegro_screwdriver_{config["model_type"]}_w_classifier.pt'
        else:
            model_name = f'allegro_screwdriver_{config["model_type"]}.pt'

        model.load_state_dict(torch.load(
            f'{CCAI_PATH}/data/training/allegro_screwdriver/{config["model_name"]}_{config["model_type"]}/{model_name}'
        ))
        model.send_norm_constants_to_submodels()


    if config['eval_train_likelihood']:
        eval_train_likelihood(model, train_loader, config)

    if config['id_ood_states']:
        identify_OOD_states(model, train_loader, config)

    if config['project_ood_states']:
        # project_OOD_states(model, config)
        visualize_ood_projection(model, config)

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
