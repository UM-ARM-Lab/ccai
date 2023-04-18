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

    def __len__(self):
        return self.trajectories.shape[0]

    def __getitem__(self, idx):
        # trajectory, goal, start
        return (torch.from_numpy(self.trajectories[idx]).to(dtype=torch.float32),
                torch.from_numpy(self.trajectories[idx, :, 0, :12]).to(dtype=torch.float32),
                torch.from_numpy(self.goals[idx]).to(dtype=torch.float32)
                )


class QuadrotorMultiConstraintTrajectoryDataset(Dataset):

    def __init__(self, fpath):
        goals = []
        trajectories = []
        constraints = []
        path = pathlib.Path(fpath)
        for p in path.rglob('*planned_trajector_data.npz'):
            data = np.load(p)
            goals.append(data['goal'])
            trajectories.append(data['traj'])
            constraints.append(data['surface'])

        self.goals = np.stack(goals, axis=0)
        self.trajectories = np.stack(trajectories, axis=0)
        self.constraints = np.stack(constraints, axis=0)

        num_trials, num_steps, num_particles, horizon, xu_dim = self.trajectories.shape
        self.constraints = self.constraints.reshape(num_trials, 1, 1, 100)
        self.goals = self.goals.reshape(num_trials, 1, 1, 3)
        self.goals = np.tile(self.goals, (1, num_steps, num_particles, 1))
        self.constraints = np.tile(self.constraints, (1, num_steps, num_particles, 1))

        self.trajectories = self.trajectories.reshape(-1, num_particles, horizon, xu_dim)
        self.goals = self.goals.reshape(-1, num_particles, 3)
        self.constraints = self.constraints.reshape(-1, num_particles, 100)

    def __len__(self):
        return self.trajectories.shape[0]

    def __getitem__(self, idx):
        # trajectory, goal, start
        return (torch.from_numpy(self.trajectories[idx]).to(dtype=torch.float32),
                torch.from_numpy(self.trajectories[idx, :, 0, :12]).to(dtype=torch.float32),
                torch.from_numpy(self.goals[idx]).to(dtype=torch.float32),
                torch.from_numpy(self.constraints[idx]).to(dtype=torch.float32)
                )


if __name__ == "__main__":
    dataset = QuadrotorSingleConstraintTrajectoryDataset('data/quadrotor_data_collection_single_constraint')
