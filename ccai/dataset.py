from torch.utils.data import Dataset
import torch
import pathlib
import numpy as np


class QuadrotorSingleConstraintTrajectoryDataset(Dataset):

    def __init__(self, fpath):
        goals = []
        trajectories = []
        path = pathlib.Path(fpath)
        for p in path.rglob('*planned_trajector_data.npz'):
            data = np.load(p)
            goals.append(data['goal'])
            trajectories.append(data['traj'])
        self.goals = np.stack(goals, axis=0)
        self.trajectories = np.stack(trajectories, axis=0)

        num_trials, num_steps, num_particles, horizon, xu_dim = self.trajectories.shape
        self.goals = self.goals.reshape(num_trials, 1, 1, 3)
        self.goals = np.tile(self.goals, (1, num_steps, num_particles, 1))

        self.trajectories = self.trajectories.reshape(-1, num_particles, horizon, xu_dim)
        self.goals = self.goals.reshape(-1, num_particles, 3)
        print(num_trials, num_steps, num_particles, horizon, xu_dim)

    def __len__(self):
        return self.trajectories.shape[0]

    def __getitem__(self, idx):
        # trajectory, goal, start
        return (torch.from_numpy(self.trajectories[idx]).to(dtype=torch.float32),
                torch.from_numpy(self.trajectories[idx, :, 0, :12]).to(dtype=torch.float32),
                torch.from_numpy(self.goals[idx]).to(dtype=torch.float32)
                )


class QuadrotorMultiConstraintTrajectoryDataset(Dataset):

    def __init__(self, fpaths):
        goals = []
        trajectories = []
        constraints = []
        for fpath in fpaths:
            if not isinstance(fpath, pathlib.PosixPath):
                path = pathlib.Path(fpath)
            else:
                path = fpath
            for p in path.rglob('*planned_trajector_data.npz'):
                data = np.load(p)
                goals.append(data['goal'])
                trajectories.append(data['traj'])
                constraints.append(data['surface'])

        self.goals = np.stack(goals, axis=0)
        self.trajectories = np.stack(trajectories, axis=0)
        self.constraints = np.stack(constraints, axis=0)

        self.num_trials, self.num_steps, self.num_particles, horizon, xu_dim = self.trajectories.shape

        self.xu_mean = None
        self.xu_std = None

    def compute_norm_constants(self):
        self.xu_mean = np.mean(self.trajectories, axis=(0, 1, 2, 3))
        self.xu_std = np.std(self.trajectories, axis=(0, 1, 2, 3))
        self.trajectories = (self.trajectories - self.xu_mean) / self.xu_std

    def set_norm_constants(self, xu_mean, xu_std):
        self.xu_mean = xu_mean
        self.xu_std = xu_std
        self.trajectories = (self.trajectories - self.xu_mean) / self.xu_std

    def get_norm_constants(self):
        return self.xu_mean, self.xu_std

    def __len__(self):
        return self.num_trials * self.num_steps * self.num_particles

    def __getitem__(self, idx):
        trial_idx = idx // (self.num_steps * self.num_particles)
        step_particle_idx = idx % (self.num_steps * self.num_particles)
        step_idx = step_particle_idx // self.num_particles
        particle_idx = step_particle_idx % self.num_particles

        # trajectory, goal, start
        return (torch.from_numpy(self.trajectories[trial_idx, step_idx, particle_idx]).to(dtype=torch.float32),
                torch.from_numpy(self.trajectories[trial_idx, step_idx, particle_idx, 0, :12]).to(dtype=torch.float32),
                torch.from_numpy(self.goals[trial_idx]).to(dtype=torch.float32),
                torch.from_numpy(self.constraints[trial_idx]).to(dtype=torch.float32),
                torch.tensor([trial_idx])
                )


class VictorTableMultiConstraintTrajectoryDataset(Dataset):

    def __init__(self, fpaths):
        goals = []
        trajectories = []
        object_centres = []
        height = []

        for fpath in fpaths:
            path = pathlib.Path(fpath)
            for p in path.rglob('*trajectory.npz'):
                data = np.load(p)
                goals.append(data['goal'])
                object_centres.append(data['obs'].reshape(-1))
                height.append(data['height'])
                trajectories.append(data['traj'])
        self.goals = np.stack(goals, axis=0)
        self.trajectories = np.stack(trajectories, axis=0)
        object_centres = np.stack(object_centres, axis=0)
        height = np.stack(height, axis=0)
        # height and object centre define constraints
        self.constraints = np.concatenate((height.reshape(-1, 1), object_centres), axis=1)
        self.num_trials, self.num_steps, self.num_particles, horizon, xu_dim = self.trajectories.shape

        self.xu_mean = None
        self.xu_std = None
    def compute_norm_constants(self):
        self.xu_mean = np.mean(self.trajectories, axis=(0, 1, 2, 3))
        self.xu_std = np.std(self.trajectories, axis=(0, 1, 2, 3))
        self.trajectories = (self.trajectories - self.xu_mean) / self.xu_std

    def set_norm_constants(self, xu_mean, xu_std):
        self.xu_mean = xu_mean
        self.xu_std = xu_std
        self.trajectories = (self.trajectories - self.xu_mean) / self.xu_std

    def get_norm_constants(self):
        return self.xu_mean, self.xu_std

    def __len__(self):
        return self.num_trials * self.num_steps * self.num_particles

    def __getitem__(self, idx):
        trial_idx = idx // (self.num_steps * self.num_particles)
        step_particle_idx = idx % (self.num_steps * self.num_particles)
        step_idx = step_particle_idx // self.num_particles
        particle_idx = step_particle_idx % self.num_particles
        # trajectory, goal, start
        return (torch.from_numpy(self.trajectories[trial_idx, step_idx, particle_idx]).to(dtype=torch.float32),
                torch.from_numpy(self.trajectories[trial_idx, step_idx, particle_idx, 0]).to(dtype=torch.float32),
                torch.from_numpy(self.goals[trial_idx]).to(dtype=torch.float32),
                torch.from_numpy(self.constraints[trial_idx]).to(dtype=torch.float32),
                torch.tensor([trial_idx])
                )


if __name__ == "__main__":
    dataset = QuadrotorSingleConstraintTrajectoryDataset('data/quadrotor_data_collection_single_constraint')
