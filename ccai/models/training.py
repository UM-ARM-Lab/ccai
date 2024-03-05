import torch
import numpy as np
import copy
import time
import os

from ccai.quadrotor_env import QuadrotorEnv
import matplotlib.pyplot as plt


class Timer:

    def __init__(self):
        self._start = time.time()

    def __call__(self, reset=True):
        now = time.time()
        diff = now - self._start
        if reset:
            self._start = now
        return diff


def cycle(dl):
    while True:
        for data in dl:
            yield data


def to_device(x, device='cuda:0'):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        raise RuntimeError(f'Unrecognized type in `to_device`: {type(x)}')


class EMA():
    '''
        empirical moving average
    '''

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_plot_with_trajectories(ax, trajectory):
    traj_lines = []
    for traj in trajectory:
        traj_np = traj.detach().cpu().numpy()
        traj_lines.extend(ax.plot(traj_np[1:, 0],
                                  traj_np[1:, 1],
                                  traj_np[1:, 2], color='g', alpha=0.5, linestyle='--'))


class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            train_loader,
            val_loader,
            ema_decay=0.995,
            train_batch_size=32,
            train_lr=1e-4,
            gradient_accumulate_every=2,
            step_start_ema=2000,
            update_ema_every=10,
            log_freq=100,
            sample_freq=10000,
            save_freq=1000,
            label_freq=100000,
            save_parallel=False,
            results_folder='./results',
            n_reference=8,
            bucket=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.model.mu.data = torch.from_numpy(train_loader.dataset.dataset.mu).to(device=self.model.mu.device)
        self.model.std.data = torch.from_numpy(train_loader.dataset.dataset.std).to(device=self.model.mu.device)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_loader = cycle(train_loader)
        self.val_loader = val_loader
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.train_loader)
                trajectories = batch[0]
                trajectories = to_device(trajectories)
                B, N = trajectories.shape[:2]
                trajectories = trajectories.reshape(B * N, -1)
                loss = self.model.loss(trajectories)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if (self.step + 1) % self.log_freq == 0:
                print(f'{self.step}: {loss:8.4f} | t: {timer():8.4f}', flush=True)

            if (self.step + 1) % self.sample_freq == 0:
                J, C = self.val_samples()
                print(f'{self.step}: {loss:8.4f} | J: {J:8.4f} | C: {C:8.4f} | t: {timer():8.4f}', flush=True)

            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def val_samples(self):

        val_constr_violation = 0
        val_sample_loss = 0

        for batch_no, batch in enumerate(self.val_loader):
            trajectories = batch[0]
            trajectories = to_device(trajectories)
            B = trajectories.shape[0]
            N = 4
            Nt = trajectories.shape[1]
            starts = batch[1].to(device='cuda:0')
            goals = batch[2].to(device='cuda:0')
            constr = batch[3].to(device='cuda:0')
            goals = torch.cat((goals, torch.zeros(B, Nt, 9).to(device='cuda:0')), dim=-1)

            sampled_trajectories = self.model.sample_constrained(N, starts[:, 0], goals[:, 0], constr[:, 0])

            # now we want to evaluate those trajectories
            J, _, _ = self.model.problem._objective(sampled_trajectories, goals[:, :N])
            C, _, _ = self.model.problem.batched_combined_constraints(sampled_trajectories, starts[:, 0], constr[:, 0],
                                                                      compute_grads=False)

            val_sample_loss += J.mean().item()
            val_constr_violation += C[:, :, -11:].abs().mean().item()

            for i, trajs in enumerate(sampled_trajectories):
                env = QuadrotorEnv('surface_data.npz')
                xy_data = env.surface_model.train_x.cpu().numpy()
                z_data = constr[i, 0].cpu().numpy()
                np.savez('tmp_surface_data.npz', xy=xy_data, z=z_data)
                env = QuadrotorEnv('tmp_surface_data.npz')
                env.state = starts[i, 0].cpu().numpy()
                env.goal = goals[i, 0].cpu().numpy()
                ax = env.render_init()
                update_plot_with_trajectories(ax, trajs.reshape(N, 12, 16))
                plt.savefig(
                    f'{self.logdir}/learned_sampler_quadrotor_{self.step}_{i}.png')
                plt.close()
            break

        val_constr_violation /= (batch_no + 1)
        val_sample_loss /= (batch_no + 1)
        return val_sample_loss, val_constr_violation
