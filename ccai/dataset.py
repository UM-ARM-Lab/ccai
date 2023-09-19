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
        constraint_type = []
        for fpath in fpaths:
            if not isinstance(fpath, pathlib.PosixPath):
                path = pathlib.Path(fpath)
            else:
                path = fpath
            for p in path.rglob('*planned_trajector_data.npz'):
                data = np.load(p)
                goals.append(data['goal'])
                trajectories.append(data['traj'])

                if 'obstacle' in data.keys():
                    if data['obstacle'] is not None:
                        constraints.append(data['obstacle'])
                        constraint_type.append(np.array([1]))
                    else:
                        constraints.append(data['surface'])
                        constraint_type.append(np.array([0]))
                else:
                    constraints.append(data['surface'])
                    constraint_type.append(np.array([0]))

        self.goals = np.stack(goals, axis=0)
        self.trajectories = np.stack(trajectories, axis=0)
        self.constraints = np.stack(constraints, axis=0)
        self.constraint_type = np.stack(constraint_type, axis=0)

        self.num_trials, self.num_steps, self.num_particles, horizon, xu_dim = self.trajectories.shape
        print(self.num_trials)
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
                torch.from_numpy(self.constraint_type[trial_idx]).to(dtype=torch.long)
                )


class VictorTableMultiConstraintTrajectoryDataset(Dataset):

    def __init__(self, fpaths):
        goals = []
        trajectories = []
        constraints = []
        constraint_type = []

        for fpath in fpaths:
            path = pathlib.Path(fpath)
            for p in path.rglob('*trajectory.npz'):
                data = np.load(p, allow_pickle=True)

                if data['obs'].reshape(-1)[0] is not None:
                    constraints.append(data['obs'].reshape(-1))
                    constraint_type.append(np.array([1]))
                else:
                    constraints.append(np.array([data['height'], 0.0, 0.0, 0.0]).reshape(-1))
                    constraint_type.append(np.array([0]))

                goals.append(data['goal'])
                trajectories.append(data['traj'])
        self.goals = np.stack(goals, axis=0)
        self.trajectories = np.stack(trajectories, axis=0)
        self.constraints = np.stack(constraints, axis=0)
        self.constraint_type = np.stack(constraint_type, axis=0)
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
                torch.from_numpy(self.constraint_type[trial_idx]).to(dtype=torch.long)
                )


class VictorConstraintDataset(Dataset):

    def __init__(self):
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
        return (self.trajectories[trial_idx, step_idx, particle_idx],
                self.trajectories[trial_idx, step_idx, particle_idx, 0],
                self.goals[trial_idx],
                self.constraints[trial_idx]
                )

        return (torch.from_numpy(self.trajectories[trial_idx, step_idx, particle_idx]).to(dtype=torch.float32),
                torch.from_numpy(self.trajectories[trial_idx, step_idx, particle_idx, 0]).to(dtype=torch.float32),
                torch.from_numpy(self.goals[trial_idx]).to(dtype=torch.float32),
                torch.from_numpy(self.constraints[trial_idx]).to(dtype=torch.float32),
                )


class VictorTableHeightDataset(VictorConstraintDataset):

    def __init__(self, fpaths):
        super().__init__()
        goals = []
        trajectories = []
        constraints = []
        constraint_type = []

        for fpath in fpaths:
            path = pathlib.Path(fpath)
            goal = np.zeros(3)
            for p in path.rglob('*trajectory.npz'):
                data = np.load(p, allow_pickle=True)
                constraints.append(np.array([data['height']]).reshape(-1))
                goal[:2] = data['goal'][:2]
                goal[2] = data['height']
                goals.append(goal.copy())
                traj = data['traj']
                n = traj.shape[1]
                # add start state to trajectories
                traj = np.concatenate([traj, data['x'][:-1, None, None].repeat(n, axis=1)], axis=2)
                trajectories.append(traj)

        self.goals = np.stack(goals, axis=0)
        self.trajectories = np.stack(trajectories, axis=0)
        self.constraints = np.stack(constraints, axis=0)
        self.num_trials, self.num_steps, self.num_particles, horizon, xu_dim = self.trajectories.shape
        print(self.ddtrajectories.shape)

        self.goals = torch.from_numpy(self.goals).to(dtype=torch.float32)
        self.trajectories = torch.from_numpy(self.trajectories).to(dtype=torch.float32)
        self.constraints = torch.from_numpy(self.constraints).to(dtype=torch.float32)


class VictorReachingDataset(VictorConstraintDataset):

    def __init__(self, fpaths):
        super().__init__()
        goals = []
        trajectories = []
        constraints = []
        constraint_type = []
        starts = []
        print('running init')
        for fpath in fpaths:
            path = pathlib.Path(fpath)
            for p in path.rglob('*trajectory.npz'):
                data = np.load(p, allow_pickle=True)
                constraints.append(np.float32(np.array([data['sdf_grid']]).reshape((1, 64, 64, 64))))
                goals.append(data['goal'])
                traj = data['traj']
                n = traj.shape[1]
                # add start state to trajectories
                traj = np.concatenate([traj, data['x'][:-1, None, None].repeat(n, axis=1)], axis=2)
                trajectories.append(traj)

        self.goals = np.stack(goals, axis=0)
        self.trajectories = np.stack(trajectories, axis=0)
        self.constraints = np.stack(constraints, axis=0)
        self.num_trials, self.num_steps, self.num_particles, horizon, xu_dim = self.trajectories.shape
        self.goals = torch.from_numpy(self.goals).to(dtype=torch.float32)
        self.trajectories = torch.from_numpy(self.trajectories).to(dtype=torch.float32)
        self.constraints = torch.from_numpy(self.constraints).to(dtype=torch.float32)


if __name__ == "__main__":
    dataset = QuadrotorSingleConstraintTrajectoryDataset('data/quadrotor_data_collection_single_constraint')
