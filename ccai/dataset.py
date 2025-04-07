from typing import Optional
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
        # return self.masks[idx] * (traj), self.trajectory_type[idx, None], mask

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

    def __init__(self, folders, max_T, cosine_sine=False, states_only=False, 
                 skip_pregrasp=False, type='diffusion', exec_only=False,
                 best_traj_only=False):
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
            traj_data = list(path.rglob('*traj_data.p'))
            trajectory = list(path.rglob('*trajectory.pkl'))
            for p, traj_p in zip(traj_data, trajectory):
                with open(p, 'rb') as f, open(traj_p, 'rb') as f_traj:
                    try:
                        data = pickle.load(f)
                        traj_data = pickle.load(f_traj)
                    except:
                        print('Fail')
                        continue
                    actual_traj = []

                    need_to_continue = False
                    
                    dropped_recovery = data['dropped_recovery']
                    # post_recovery_likelihoods = [i for i in data['final_likelihoods'] if len(i) == 1]
                    # post_recovery_likelihood_bool = []
                    # for fl in post_recovery_likelihoods:
                    #     if fl[0] < -1250:
                    #         post_recovery_likelihood_bool.append(False)
                    #     else:
                    #         post_recovery_likelihood_bool.append(True)
                    for t in range(max_T, min_t - 1, -1):

                        # Filter out any trajectories with contact_state [1, 1, 1]
                        new_starts = []
                        for s in data[t]['starts']:
                            if (s.sum(0) == 0).any():
                            # if s.shape[-1] != 36:
                                new_starts.append(s)
                        new_plans = []
                        for p in data[t]['plans']:
                            # if p.shape[-1] != 36:
                            #     new_plans.append(p)
                            if (p.sum(0).sum(0) == 0).any():
                                new_plans.append(p)

                        new_cs = []
                        for c in data[t]['contact_state']:
                            # new_cs.append(c.sum() != 3)
                            if c.sum() != 3:
                                # if isinstance(c, np.ndarray):
                                #     c = torch.from_numpy(c).float()
                                if torch.is_tensor(c):
                                    c = c.cpu().numpy()
                                new_cs.append(c)
                        if len(new_starts) == 0:
                            print('No starts')
                            need_to_continue = True
                            break
                        data[t]['starts'] = np.stack(new_starts, axis=0)
                        data[t]['plans'] = np.stack(new_plans, axis=0)
                        data[t]['contact_state'] = np.stack(new_cs, axis=0)

                        if dropped_recovery:
                            data[t]['starts'] = data[t]['starts'][:-1]
                            data[t]['plans'] = data[t]['plans'][:-1]
                            data[t]['contact_state'] = data[t]['contact_state'][:-1]
                        if len(data[t]['starts']) == 0:
                            print('No starts')
                            need_to_continue = True
                            break
                        # for key in ['starts', 'plans']:
                        #     new_data = []
                        #     new_cs = []
                        #     for d, c in zip(data[t][key], data[t]['contact_state']):

                        #         # If d is np array, convert to tensor
                        #         if isinstance(d, np.ndarray):
                        #             d = torch.from_numpy(d).float()
                        #         if c.sum() == 1:
                        #             new_data.append(
                        #                 torch.cat((d, torch.torch.zeros(*d.shape[:-1], 6).to(device=d.device)), dim=-1).cpu()
                        #             )
                        #         elif c.sum() == 2:
                        #             new_data.append(
                        #                 torch.cat(
                        #                     (d[..., :-6],
                        #             torch.zeros(*d.shape[:-1], 3).to(device=d.device),
                        #             d[..., -6:]), dim=-1
                        #                 ).cpu()
                        #             )
                        #     data[t][key] = torch.stack(new_data, dim=0)
                            
                        # data[t]['contact_state'] = torch.stack(data[t]['contact_state'], dim=0)                   
                        
                        # for key in ['starts', 'plans', 'contact_state']:
                        #     data[t][key] = data[t][key].cpu().numpy()

                        # if len(data['final_likelihoods']) == 1 and data['final_likelihoods'][0] < -45:
                        #     print('No Data')
                        #     need_to_continue = True
                        #     break
                    if need_to_continue:
                        continue
                    for t in range(max_T, min_t - 1, -1):
                        actual_traj.append(data[t]['starts'][:, :, None, :])
                        traj = data[t]['plans']
                        end_states = []
                        for i in range(1, len(traj_data)):
                            if i < len(traj_data) - 1:
                                end_states.append(traj_data[i][-1])
                            else:
                                zero_pad = np.zeros(21)
                                end_states.append(np.concatenate((traj_data[i], zero_pad)))
                        # print(traj.shape)
                        if not exec_only or (exec_only and t == min_t):
                            to_append = data[t]['contact_state'][:, None, :]
                            if not best_traj_only:
                                to_append = to_append.repeat(traj.shape[1], axis=1)
                            classes.append(to_append)
                            # combine traj and starts
                            if use_actual_traj:
                                traj = np.concatenate(actual_traj + [traj], axis=2)
                                if best_traj_only:
                                    traj = traj[..., :1 , :,:]
                                masks.append(np.ones((traj.shape[0], traj.shape[1], traj.shape[2])))
                            else:
                                zeros = [np.zeros_like(actual_traj[0])] * (len(actual_traj) - 1)
                                traj = np.concatenate([actual_traj[-1]] + [traj] + zeros, axis=2)
                                mask = np.zeros((traj.shape[0], traj.shape[1], traj.shape[2]))
                                mask[:, :, :t + 1] = 1
                                masks.append(mask)

                            if type == 'diffusion':
                                # duplicated first control, rearrange so that it is (x_0, u_0, x_1, u_1, ..., x_{T-1}, u_{T-1}, x_T, 0)
                                traj[:, :, :-1, 15:] = traj[:, :, 1:, 15:]
                                traj[:, :, -1, 15:] = 0
                            elif type == 'cnf':
                                traj[:, :, 0, 15:] = 0
                            trajectories.append(traj)

                    # if exec_only:
                    #     classes.append(data[t]['contact_state'][:, None, :])#.repeat(traj.shape[1], axis=1)
                    #     end_states = np.stack(end_states, axis=0)
                    #     end_states = end_states.reshape(end_states.shape[0], 1, 1, -1)
                    #     end_states = end_states.repeat(traj.shape[1], axis=1)
                    #     end_states = end_states[:traj.shape[0]]
                    #     traj = np.concatenate([traj, end_states], axis=2)
                    #     # combine traj and starts
                    #     if use_actual_traj:
                    #         traj = np.concatenate(actual_traj + [traj], axis=2)[..., :1 , :,:]
                    #         masks.append(np.ones((traj.shape[0], traj.shape[1], traj.shape[2]))[..., :1, :])
                    #     else:
                    #         zeros = [np.zeros_like(actual_traj[0])] * (len(actual_traj) - 1)
                    #         traj = np.concatenate([actual_traj[-1]] + [traj] + zeros, axis=2)
                    #         mask = np.zeros((traj.shape[0], traj.shape[1], traj.shape[2]))
                    #         mask[:, :, :t + 1] = 1
                    #         masks.append(mask)

                    #     if type == 'diffusion':
                    #         # duplicated first control, rearrange so that it is (x_0, u_0, x_1, u_1, ..., x_{T-1}, u_{T-1}, x_T, 0)
                    #         traj[:, :, :-1, 15:] = traj[:, :, 1:, 15:]
                    #         traj[:, :, -1, 15:] = 0
                    #     elif type == 'cnf':
                    #         traj[:, :, 0, 15:] = 0
                    #     trajectories.append(traj)

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
        # return self.masks[idx] * (traj), self.trajectory_type[idx], mask

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
    
class PerEpochBalancedSampler(torch.utils.data.Sampler):
    """Samples equal number of samples from each class, resampling at the start of each epoch.
    
    Args:
        dataset: Dataset containing trajectory_type attribute
        samples_per_class: Optional fixed number of samples per class. If None, uses min class count
    """
    def __init__(self, dataset, samples_per_class: Optional[int] = None):
        self.dataset = dataset
        self.classes_np = dataset.trajectory_type.numpy()
        self.unique_classes = np.unique(self.classes_np, axis=0)
        
        # Get counts per class
        self.class_counts = {
            tuple(c): np.sum(np.all(self.classes_np == c, axis=1)) 
            for c in self.unique_classes
        }
        
        # Set samples per class
        self.samples_per_class = samples_per_class or min(self.class_counts.values())
        
    def __iter__(self):
        # Reset indices each epoch
        balanced_indices = []
        
        # Sample from each class
        for c in self.unique_classes:
            # Get all indices for this class
            class_indices = np.where(np.all(self.classes_np == c, axis=1))[0]
            # Sample with replacement if needed
            sampled_indices = np.random.choice(
                class_indices, 
                size=self.samples_per_class,
                replace=len(class_indices) < self.samples_per_class
            )
            balanced_indices.extend(sampled_indices)
            
        # Shuffle indices
        return iter(torch.tensor(balanced_indices)[torch.randperm(len(balanced_indices))])

    def __len__(self):
        return len(self.unique_classes) * self.samples_per_class

class AllegroScrewDriverStateDataset(Dataset):

    def __init__(self, folders, max_T, cosine_sine=False, states_only=False, 
                 skip_pregrasp=False, type='diffusion'):
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
                    try:
                        data = pickle.load(f)
                        data[max_T]['starts'][:, :, None, :]
                    except:
                        print('Fail')
                        continue
                    actual_traj = []
                    for t in range(max_T, min_t - 1, -1):
                        actual_traj.append(data[t]['starts'][:, :, None, :])
                        traj = data[t]['plans']
                        # print(traj.shape)
                        classes.append(data[t]['contact_state'][:, None, :])
                    classes.append(data[t]['contact_state'][:, None, :])

                    # combine traj and starts
                    if use_actual_traj:
                        traj = np.concatenate(actual_traj + [traj], axis=2)[..., :1 , :,:]
                        masks.append(np.ones((traj.shape[0], traj.shape[1], traj.shape[2]))[..., :1, :])
                    else:
                        zeros = [np.zeros_like(actual_traj[0])] * (len(actual_traj) - 1)
                        traj = np.concatenate([actual_traj[-1]] + [traj] + zeros, axis=2)
                        mask = np.zeros((traj.shape[0], traj.shape[1], traj.shape[2]))
                        mask[:, :, :t + 1] = 1
                        masks.append(mask)

                    if type == 'diffusion':
                        # duplicated first control, rearrange so that it is (x_0, u_0, x_1, u_1, ..., x_{T-1}, u_{T-1}, x_T, 0)
                        traj[:, :, :-1, 15:] = traj[:, :, 1:, 15:]
                        traj[:, :, -1, 15:] = 0
                    elif type == 'cnf':
                        traj[:, :, 0, 15:] = 0
                    trajectories.append(traj)

        self.trajectories = np.concatenate(trajectories, axis=0)
        self.trajectory_type = np.concatenate(classes, axis=0).reshape(-1, 3)

        turn_bool = self.trajectory_type.mean(axis=1) == 1

        self.trajectories = self.trajectories.reshape(-1, 1, self.trajectories.shape[-1])
        self.trajectories = self.trajectories[turn_bool]

        self.traj_indices = np.arange(self.trajectories.shape[0]).reshape(-1, 1).repeat(self.trajectories.shape[2], axis=1).reshape(-1, 1)

        self.trajectories, self.unique_indices = np.unique(self.trajectories, return_index=True, axis=0)

        self.trajectories = torch.from_numpy(self.trajectories).float()
        self.unique_traj_indices = torch.from_numpy(self.traj_indices[self.unique_indices]).long()
        
        randperm = torch.randperm(self.unique_traj_indices.shape[0])
        self.unique_traj_indices = self.unique_traj_indices[randperm]
        self.trajectories = self.trajectories[randperm]
        # self.unique_indices = torch.from_numpy(self.unique_indices).long()
        
        
        if states_only:
            self.trajectories = self.trajectories[:, :, :15]
        self.trajectory_type = torch.from_numpy(self.trajectory_type)
        self.trajectory_type = 2 * (self.trajectory_type - 0.5)  # scale to be [-1, 1]

        self.trajectory_type = self.trajectory_type[self.unique_traj_indices.squeeze()]


        print(self.trajectories.shape)
        # TODO consider alternative SO3 representation that is better for learning
        if self.cosine_sine:
            dx = self.trajectories.shape[-1] + 1
        else:
            dx = self.trajectories.shape[-1]

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
        ind = self.unique_traj_indices[idx]
        contact = self.trajectory_type[ind]

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

        return traj, ind, contact

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

    def compute_norm_constants(self):
        # compute norm constants not including the zero padding
        x = self.trajectories.clone()
        # x[:, :, 8] += 2 * np.pi * (torch.rand(x.shape[0], 1) - 0.5)
        x = x.reshape(-1, x.shape[-1])

        mean = x.mean(dim=0)
        # std = np.sqrt(np.average((x - mean) ** 2, weights=mask, axis=0))
        std = x.std(dim=0)

        if self.states_only:
            dim = 15
        else:
            dim = 15 + 12 + 9

        # for angle we force to be between [-1, 1]
        if self.cosine_sine:
            self.mean = torch.zeros(dim + 1)
            self.std = torch.ones(dim + 1)
            self.mean[:14] = mean[:14]
            self.std[:14] = (std[:14]).float()
            self.mean[16:] = mean[15:]
            self.std[16:] = (std[15:]).float()
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
