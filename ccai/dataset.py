import torch
import pickle
import pathlib
import numpy as np
from torch.utils.data import Dataset


class AllegroValveDataset(Dataset):

    def __init__(self, folders, turn=True, regrasp=True, cosine_sine=True, states_only=False):
        super().__init__()
        assert (turn or regrasp)
        self.cosine_sine = cosine_sine
        # TODO: only using trajectories for now, also includes closest points and their sdf values
        turn_trajectories = []
        turn_starts = []
        regrasp_trajectories = []
        regrasp_starts = []
        regrasp_masks = []
        turn_masks = []
        min_t = 1
        max_T = 15

        use_actual_traj = True
        for fpath in folders:
            path = pathlib.Path(fpath)

            plans = []
            for p in path.rglob('*turn_data.p'):

                with open(p, 'rb') as f:
                    data = pickle.load(f)
                    actual_traj = []
                    for t in range(max_T, min_t - 1, -1):
                        actual_traj.append(data[t]['starts'][:, :, None, :])
                        traj = data[t]['plans']
                        # combine traj and starts
                        if use_actual_traj:
                            traj = np.concatenate(actual_traj + [traj], axis=2)
                            turn_masks.append(np.ones((traj.shape[0], traj.shape[1], traj.shape[2])))
                        else:
                            zeros = [np.zeros_like(actual_traj[0])] * (len(actual_traj) - 1)
                            traj = np.concatenate([actual_traj[-1]] + [traj] + zeros, axis=2)
                            mask = np.zeros((traj.shape[0], traj.shape[1], traj.shape[2]))
                            mask[:, :, :t + 1] = 1
                            turn_masks.append(mask)

                        # duplicated first control, rearrange so that it is (x_0, u_0, x_1, u_1, ..., x_{T-1}, u_{T-1}, x_T, 0)
                        traj[:, :, :-1, -8:] = traj[:, :, 1:, -8:]
                        traj[:, :, -1, -8:] = 0
                        turn_trajectories.append(traj)

            for p in path.rglob('*regrasp_data.p'):
                with open(p, 'rb') as f:
                    data = pickle.load(f)
                    actual_traj = []
                    for t in range(max_T, min_t - 1, -1):
                        actual_traj.append(data[t]['starts'][:, :, None, :16])
                        traj = data[t]['plans']
                        # combine traj and starts
                        if use_actual_traj:
                            traj = np.concatenate(actual_traj + [traj], axis=2)
                            regrasp_masks.append(np.ones((traj.shape[0], traj.shape[1], traj.shape[2])))
                        else:
                            zeros = [np.zeros_like(actual_traj[0])] * (len(actual_traj) - 1)
                            traj = np.concatenate([actual_traj[-1]] + [traj] + zeros, axis=2)
                            mask = np.zeros((traj.shape[0], traj.shape[1], traj.shape[2]))
                            mask[:, :, :t + 1] = 1
                            regrasp_masks.append(mask)

                        # duplicated first control, rearrange so that it is (x_0, u_0, x_1, u_1, ..., x_{T-1}, u_{T-1}, x_T, 0)
                        traj[:, :, :-1, -8:] = traj[:, :, 1:, -8:]
                        traj[:, :, -1, -8:] = 0
                        # just say the valve is at a random angle
                        theta = (np.random.rand(traj.shape[0], traj.shape[1], 1, 1) - 0.5) * 2 * np.pi
                        theta = theta.repeat(repeats=max_T + 1, axis=2)
                        traj = np.concatenate((traj[:, :, :, :8], theta, traj[:, :, :, -8:]), axis=3)

                        if not use_actual_traj:
                            traj[:, :, t + 1:, 8] = 0
                        regrasp_trajectories.append(traj)
        print([turn_traj.shape for turn_traj in turn_trajectories])

        self.turn_trajectories = np.concatenate(turn_trajectories, axis=0)
        self.regrasp_trajectories = np.concatenate(regrasp_trajectories, axis=0)
        turn_masks = np.concatenate(turn_masks, axis=0)
        regrasp_masks = np.concatenate(regrasp_masks, axis=0)
        if turn and regrasp:
            self.trajectories = np.concatenate((self.turn_trajectories, self.regrasp_trajectories), axis=0)
            self.masks = np.concatenate((turn_masks, regrasp_masks), axis=0)
        elif turn:
            self.trajectories = self.turn_trajectories
            self.masks = turn_masks
        elif regrasp:
            self.trajectories = self.regrasp_trajectories
            self.masks = regrasp_masks

        self.trajectory_type = torch.zeros(*self.trajectories.shape[:-2], 1, dtype=torch.long)
        self.trajectory_type[self.turn_trajectories.shape[0]:] = 1
        self.trajectory_type[:self.turn_trajectories.shape[0]] = -1

        self.trajectories = self.trajectories.reshape(-1, self.trajectories.shape[-2], self.trajectories.shape[-1])
        self.trajectory_type = self.trajectory_type.reshape(-1)
        self.masks = self.masks.reshape(-1, self.masks.shape[-2])
        self.trajectories = torch.from_numpy(self.trajectories).float()
        self.masks = torch.from_numpy(self.masks).float()
        if states_only:
            self.trajectories = self.trajectories[:, :, :9]
        if self.cosine_sine:
            dx = self.trajectories.shape[-1] + 1
        else:
            dx = self.trajectories.shapes[-1]
        self.masks = self.masks[:, :, None].repeat(1, 1, dx)  # for states
        # self.trajectories[:, :, 8] += np.pi # add pi to the valve angle
        ## some wrap around issues here:
        # self.trajectories[:, :, 8] = torch.where(self.trajectories[:, :, 8] > np.pi,
        #                                         self.trajectories[:, :, 8] - 2 * np.pi,
        #                                         self.trajectories[:, :, 8])
        # self.trajectories[:, :, 8] = torch.where(self.trajectories[:, :, 8] < -np.pi,
        #                                         self.trajectories[:, :, 8] + 2 * np.pi,
        #                                         self.trajectories[:, :, 8])

        print(self.trajectories.shape)
        # self.trajectories = self.trajectories[:, :, :9]
        # self.masks = torch.from_numpy(self.masks).float()
        # self.masks = torch.ones(self.trajectories.shape[0], self.trajectories.shape[1])
        self.mean = 0
        self.std = 1

        self.mask_dist = torch.distributions.bernoulli.Bernoulli(probs=0.75)
        self.initial_state_mask_dist = torch.distributions.bernoulli.Bernoulli(probs=0.75)
        self.states_only = states_only

    def __len__(self):
        return self.trajectories.shape[0]

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        # randomly perturb the valve angles
        traj[:, 8] += 2 * np.pi * (np.random.rand() - 0.5)
        # modulo to make sure between [-pi and pi]
        # add pi to make [0, 2pi]
        traj[:, 8] += np.pi
        traj[:, 8] = traj[:, 8] % (2.0 * np.pi)
        traj[:, 8] = traj[:, 8] - np.pi  # subtract to make [-pi, pi]
        dx = 9
        if self.cosine_sine:
            traj_q = traj[:, :8]
            traj_theta = traj[:, 8][:, None]
            traj_u = traj[:, 9:]
            dx = 10
            traj = torch.cat((traj_q, torch.cos(traj_theta), torch.sin(traj_theta), traj_u), dim=1)

        # randomly mask the initial state and the final state
        # need to find the final state - last part of mask that is 1
        final_idx = self.masks[idx, :, 0].nonzero().max().item()
        mask = self.masks[idx].clone()

        # sample mask for initial state
        mask[0, :dx] = self.initial_state_mask_dist.sample((1,)).to(device=traj.device)

        # sample mask for final state
        mask[final_idx, :dx] = self.initial_state_mask_dist.sample((1,)).to(device=traj.device)

        # also mask out the rest with small probability
        mask = mask * self.mask_dist.sample((mask.shape[0],)).to(device=traj.device).reshape(-1, 1)
        ## we can't mask out everything
        if mask.sum() == 0:
            # randomly choose an index to un-mask
            mask[np.random.randint(0, final_idx)] = 1
        print(traj)
        exit(0)
        return self.masks[idx] * (traj - self.mean) / self.std, self.trajectory_type[idx, None], mask

    def compute_norm_constants(self):
        # compute norm constants not including the zero padding
        x = self.trajectories.clone()
        # x[:, :, 8] += 2 * np.pi * (torch.rand(x.shape[0], 1) - 0.5)
        x = x.reshape(-1, x.shape[-1])

        mask = self.masks[:, :, 0].reshape(-1)
        mean = x.sum(dim=0) / mask.sum()
        std = np.sqrt(np.average((x - mean) ** 2, weights=mask, axis=0))

        if self.states_only:
            dim = 9
        else:
            dim = 17
        # for angle we force to be between [-1, 1]
        if self.cosine_sine:
            self.mean = torch.zeros(dim + 1)
            self.std = torch.ones(dim + 1)
            self.mean[:8] = mean[:8]
            self.std[:8] = torch.from_numpy(std[:8]).float()
            self.mean[10:] = mean[9:]
            self.std[10:] = torch.from_numpy(std[9:]).float()
        else:
            mean[8] = 0
            std[8] = np.pi
            self.mean = mean
            self.std = torch.from_numpy(std).float()

    def set_norm_constants(self, mean, std):
        self.mean = mean
        self.std = std

    def get_norm_constants(self):
        return self.mean, self.std


class AllegroScrewDriverDataset(Dataset):

    def __init__(self, folders, max_T, cosine_sine=False, states_only=False, skip_pregrasp=False):
        super().__init__()
        self.cosine_sine = cosine_sine
        self.skip_pregrasp = skip_pregrasp
        # TODO: only using trajectories for now, also includes closest points and their sdf values
        starts = []
        trajectories = []
        classes = []
        masks = []

        min_t = 1

        use_actual_traj = True
        for fpath in folders:
            path = pathlib.Path(fpath)
            plans = []
            for p in path.rglob('*traj_data.p'):
                with open(p, 'rb') as f:
                    data = pickle.load(f)
                    actual_traj = []
                    for t in range(max_T, min_t - 1, -1):
                        actual_traj.append(data[t]['starts'][:, :, None, :])
                        traj = data[t]['plans']
                        classes.append(data[t]['contact_state'][:, None, :].repeat(traj.shape[1], axis=1))

                        # combine traj and starts
                        if use_actual_traj:
                            traj = np.concatenate(actual_traj + [traj], axis=2)
                            masks.append(np.ones((traj.shape[0], traj.shape[1], traj.shape[2])))
                        else:
                            zeros = [np.zeros_like(actual_traj[0])] * (len(actual_traj) - 1)
                            traj = np.concatenate([actual_traj[-1]] + [traj] + zeros, axis=2)
                            mask = np.zeros((traj.shape[0], traj.shape[1], traj.shape[2]))
                            mask[:, :, :t + 1] = 1
                            masks.append(mask)

                        # duplicated first control, rearrange so that it is (x_0, u_0, x_1, u_1, ..., x_{T-1}, u_{T-1}, x_T, 0)
                        traj[:, :, :-1, 15:] = traj[:, :, 1:, 15:]
                        traj[:, :, -1, 15:] = 0
                        trajectories.append(traj)

        self.trajectories = np.concatenate(trajectories, axis=0)
        self.masks = np.concatenate(masks, axis=0)
        self.trajectory_type = np.concatenate(classes, axis=0).reshape(-1, 3)

        self.trajectories = self.trajectories.reshape(-1, self.trajectories.shape[-2], self.trajectories.shape[-1])
        self.masks = self.masks.reshape(-1, self.masks.shape[-1])
        self.trajectories = torch.from_numpy(self.trajectories).float()
        self.masks = torch.from_numpy(self.masks).float()
        if states_only:
            self.trajectories = self.trajectories[:, :, :15]
        self.trajectory_type = torch.from_numpy(self.trajectory_type)
        self.trajectory_type = 2 * (self.trajectory_type - 0.5)  # scale to be [-1, 1]

        print(self.trajectories.shape)
        # TODO consider alternative SO3 representation that is better for learning
        if self.cosine_sine:
            dx = self.trajectories.shape[-1] + 1
        else:
            dx = self.trajectories.shape[-1]

        self.masks = self.masks[:, :, None].repeat(1, 1, dx)
        self.mean = 0
        self.std = 1

        self.mask_dist = torch.distributions.bernoulli.Bernoulli(probs=0.75)
        self.initial_state_mask_dist = torch.distributions.bernoulli.Bernoulli(probs=0.5)
        self.states_only = states_only

    def update_masks(self, p1, p2):
        self.mask_dist = torch.distributions.bernoulli.Bernoulli(probs=p2)
        self.initial_state_mask_dist = torch.distributions.bernoulli.Bernoulli(probs=p1)

    def __len__(self):
        return self.trajectories.shape[0]

    def __getitem__(self, idx):
        traj = self.trajectories[idx]

        # TODO: figure out how to do data augmentation on screwdriver angle
        # a little more complex due to rotation representation
        dx = 15

        ## randomly perturb angle of screwdriver
        traj[:, dx-1] += 2 * np.pi * (np.random.rand() - 0.5)

        if self.cosine_sine:
                traj_q = traj[:, :14]
                traj_theta = traj[:, 14][:, None]
                traj_u = traj[:, 15:]
                dx = dx + 1
                traj = torch.cat((traj_q, torch.cos(traj_theta), torch.sin(traj_theta), traj_u), dim=1)

        # randomly mask the initial state and the final state
        # need to find the final state - last part of mask that is 1
        final_idx = self.masks[idx, :, 0].nonzero().max().item()
        mask = self.masks[idx].clone()

        # sample mask for initial state
        mask[0, :dx] = self.initial_state_mask_dist.sample((1,)).to(device=traj.device)

        # sample mask for final state
        mask[final_idx, :dx] = self.initial_state_mask_dist.sample((1,)).to(device=traj.device)

        # also mask out the rest with small probability
        mask = mask * self.mask_dist.sample((mask.shape[0],)).to(device=traj.device).reshape(-1, 1)

        ## we can't mask out everything
        if mask.sum() == 0:
            # randomly choose an index to un-mask
            mask[np.random.randint(0, final_idx)] = 1
        # print(mask)
        return self.masks[idx] * (traj - self.mean) / self.std, self.trajectory_type[idx], mask

    def compute_norm_constants(self):
        # compute norm constants not including the zero padding
        x = self.trajectories.clone()
        # x[:, :, 8] += 2 * np.pi * (torch.rand(x.shape[0], 1) - 0.5)
        x = x.reshape(-1, x.shape[-1])

        mask = self.masks[:, :, 0].reshape(-1)
        mean = x.sum(dim=0) / mask.sum()
        std = np.sqrt(np.average((x - mean) ** 2, weights=mask, axis=0))

        if self.states_only:
            dim = 15
        else:
            dim = 15 + 12 + 9

        # for angle we force to be between [-1, 1]
        if self.cosine_sine:
            self.mean = torch.zeros(dim + 1)
            self.std = torch.ones(dim + 1)
            self.mean[:14] = mean[:14]
            self.std[:14] = torch.from_numpy(std[:14]).float()
            self.mean[16:] = mean[15:]
            self.std[16:] = torch.from_numpy(std[15:]).float()
        else:
            # mean[12:15] = 0
            # std[12:15] = np.pi
            # mean[14] = 0
            # std[14] = np.pi
            self.mean = mean
            self.std = torch.from_numpy(std).float()

    def set_norm_constants(self, mean, std):
        self.mean = mean
        self.std = std

    def get_norm_constants(self):
        return self.mean, self.std

class AllegroScrewDriverTransitionDataset(AllegroScrewDriverDataset):

    def __init__(self, folders, cosine_sine=False, states_only=False):
        super().__init__(folders, cosine_sine, states_only)

    def __len__(self):
        return self.trajectories.shape[0] * 15

    def __getitem__(self, idx):
        traj_idx = idx // 15
        traj = self.trajectories[traj_idx]
        idx = idx % 15
        
        # print(traj.shape)
        x_t = traj[idx, :15]
        u_t = traj[idx, 15:]
        x_t_1 = traj[idx+1, :15]

        return x_t, u_t, x_t_1, self.trajectory_type[traj_idx]

    def compute_norm_constants(self):
        # compute norm constants not including the zero padding
        x = self.trajectories.clone()
        #x[:, :, 8] += 2 * np.pi * (torch.rand(x.shape[0], 1) - 0.5)
        x = x.reshape(-1, x.shape[-1])

        mask = self.masks[:, :, 0].reshape(-1)
        mean = x.sum(dim=0) / mask.sum()
        std = np.sqrt(np.average((x - mean) ** 2, weights=mask, axis=0))

        if self.states_only:
            dim = 15
        else:
            dim = 15 + 12 + 9

        # for angle we force to be between [-1, 1]
        if self.cosine_sine:
            self.mean = torch.zeros(dim+1)
            self.std = torch.ones(dim+1)
            self.mean[:8] = mean[:8]
            self.std[:8] = torch.from_numpy(std[:8]).float()
            self.mean[10:] = mean[9:]
            self.std[10:] = torch.from_numpy(std[9:]).float()
        else:
            #mean[12:15] = 0
            #std[12:15] = np.pi
            mean[14] = 0
            std[14] = np.pi
            self.mean = mean
            self.std = torch.from_numpy(std).float()

    def set_norm_constants(self, mean, std):
        self.mean = mean
        self.std = std

    def get_norm_constants(self):
        return self.mean, self.std

class FakeDataset(Dataset):

    def __init__(self, fpath, cosine_sine=False):
        data = dict(np.load(fpath))
        self.trajectories = torch.from_numpy(data['trajectories'])
        self.contact = torch.from_numpy(data['contact'])

        self.N = len(self.trajectories)
        self.mean = 0
        self.std = 1
        self.cosine_sine = cosine_sine
        self.mask_dist = torch.distributions.bernoulli.Bernoulli(probs=0.75)
        self.initial_state_mask_dist = torch.distributions.bernoulli.Bernoulli(probs=0.5)

    def update_masks(self, p1, p2):
        self.mask_dist = torch.distributions.bernoulli.Bernoulli(probs=p2)
        self.initial_state_mask_dist = torch.distributions.bernoulli.Bernoulli(probs=p1)

    def set_norm_constants(self, mean, std):
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        traj = self.trajectories[idx]

        # TODO: figure out how to do data augmentation on screwdriver angle
        # a little more complex due to rotation representation
        # a little more complex due to rotation representation
        dx = 15

        if self.cosine_sine:
            traj_q = traj[:, :14]
            traj_theta = traj[:, 14][:, None]
            traj_u = traj[:, 15:]
            dx = dx + 1
            traj = torch.cat((traj_q, torch.cos(traj_theta), torch.sin(traj_theta), traj_u), dim=1)

        mask = torch.ones_like(traj)
        # sample mask for initial state
        mask[0, :dx] = self.initial_state_mask_dist.sample((1,)).to(device=traj.device)

        # sample mask for final state
        mask[-1, :dx] = self.initial_state_mask_dist.sample((1,)).to(device=traj.device)

        # also mask out the rest with small probability
        mask = mask * self.mask_dist.sample((mask.shape[0],)).to(device=traj.device).reshape(-1, 1)

        ## we can't mask out everything
        if mask.sum() == 0:
            # randomly choose an index to un-mask
            mask[np.random.randint(0, mask.shape[0])] = 1

        # print(mask)
        return (traj - self.mean) / self.std, self.contact[idx], mask


class RealAndFakeDataset(Dataset):

    def __init__(self, real_dataset, fake_dataset):
        self.real_dataset = real_dataset
        self.fake_dataset = fake_dataset

        self.fake_dataset.set_norm_constants(self.real_dataset.mean, self.real_dataset.std)

    def __len__(self):
        return len(self.fake_dataset)

    def __getitem__(self, idx):
        real_traj, real_contact, real_mask = self.real_dataset[idx]
        fake_traj, fake_contact, fake_mask = self.fake_dataset[idx]

        return real_traj, real_contact, real_mask, fake_traj, fake_contact, fake_mask


if __name__ == "__main__":
    # d = AllegroValveDataset('../data/experiments/allegro_turning_data_collection')

    d = AllegroScrewDriverDataset(['../data/experiments/allegro_screwdriver_data_collection_2_fixed'])

    traj, contact, mask = d[0]

    print(traj.shape)
    print(contact.shape)
    print(mask.shape)
