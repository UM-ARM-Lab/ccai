from typing import Optional
import torch
import pickle
import pathlib
import numpy as np
from torch.utils.data import Dataset
import io
import matplotlib.pyplot as plt
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

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

    def __init__(self, folders, max_T, dx, cosine_sine=False, states_only=False, 
                 skip_pregrasp=False, type_='diffusion', exec_only=False,
                 best_traj_only=False, q_learning_constraint_violation=False):
        super().__init__()
        self.cosine_sine = cosine_sine
        self.skip_pregrasp = skip_pregrasp
        # TODO: only using trajectories for now, also includes closest points and their sdf values
        starts = []
        trajectories = []
        classes = []
        masks = []

        min_t = 1
        # if self.cosine_sine:
        #     dx += 1
        self.dx = dx
        use_actual_traj = True
        self.screwdriver = 'screwdriver' in str(folders[0])
        print(f'Screwdriver: {self.screwdriver}')
        
        total_num_removed = 0
        for fpath in folders:
            path = pathlib.Path(fpath)
            plans = []
            traj_data_all = list(path.rglob('*traj_data.p'))
            trajectory_all = list(path.rglob('*trajectory.pkl'))
            for p, traj_p in zip(traj_data_all, trajectory_all):
                with open(p, 'rb') as f, open(traj_p, 'rb') as f_traj:
                    # data = CPU_Unpickler(f).load()
                    try:
                        data = CPU_Unpickler(f).load()
                        traj_data = CPU_Unpickler(f_traj).load()
                    except:
                        print('Fail')
                        continue
                    actual_traj = []

                    need_to_continue = False
                    
                    # dropped_recovery = data['dropped_recovery']
                    yaw_change = traj_data[-1][12]-traj_data[0][0, 12]
                    success_cutoff_degrees = 5
                    success_cutoff_rad = np.deg2rad(success_cutoff_degrees)
                    success = yaw_change < (-np.pi/3 + success_cutoff_rad)
                    
                    fl = data['final_likelihoods']
                    pre_action_likelihoods = data['pre_action_likelihoods']
                    fl = [l for l in fl if None not in l and len(l)>0]
                    for k in range(len(fl)):
                        if len(fl[k]) == 0 and len(pre_action_likelihoods[k]) > 0:
                            fl[k].append(pre_action_likelihoods[k][-1])
                    
                    fl = np.array(fl).flatten()
                    fl_delta = []
                    executed_recovery_modes = [m for m in data['executed_contacts'] if m != 'turn']
                    for i in range(len(executed_recovery_modes)):
                        pre_mode_likelihood = fl[i-1]
                        post_mode_likelihood = fl[i]
                        likelihood_delta = post_mode_likelihood - pre_mode_likelihood
                        fl_delta.append(likelihood_delta)
                    fl_improvement_bool = np.array(fl_delta) > 0
                    if len(executed_recovery_modes) > 0 and executed_recovery_modes[-1] != 'turn':
                        fl_improvement_bool[-1] = False
                        print('Overrode last recovery mode')
                    num_removed = np.sum(~fl_improvement_bool)
                    if num_removed > 0:
                        total_num_removed += num_removed
                        print(f'Num removed: {num_removed}, Total removed: {total_num_removed}')
                    
                    # dropped_recovery = False
                    # post_recovery_likelihoods = [i for i in data['final_likelihoods'] if len(i) == 1]
                    # post_recovery_likelihood_bool = []
                    # for fl in post_recovery_likelihoods:
                    #     if fl[0] < -1250:
                    #         post_recovery_likelihood_bool.append(False)
                    #     else:
                    #         post_recovery_likelihood_bool.append(True)
                    for t in range(max_T, min_t - 1, -1):
                        new_cs = []
                        cs_bool = []
                        for c in data[t]['contact_state']:
                            # new_cs.append(c.sum() != 3)
                            if c.sum() != 3:
                                # if isinstance(c, np.ndarray):
                                #     c = torch.from_numpy(c).float()
                                if torch.is_tensor(c):
                                    c = c.cpu().numpy()
                                new_cs.append(c)
                                cs_bool.append(True)
                            else:
                                cs_bool.append(False)
                        # Filter out any trajectories with contact_state [1, 1, 1]
                        new_starts = []
                        for idx, s in enumerate(data[t]['starts']):
                            # if cs_bool[idx]:
                            # if s.shape[-1] != 36:
                            if (s.sum(0) == 0).any():
                                new_starts.append(s)
                        new_plans = []
                        for idx, p in enumerate(data[t]['plans']):
                            # if p.shape[-1] != 36:
                            #     new_plans.append(p)
                            # if cs_bool[idx]:
                            if (p.sum(0) == 0).any():
                                new_plans.append(p)


                        if len(new_starts) == 0:
                            print('No starts')
                            need_to_continue = True
                            break
                        if type(new_starts) == list:
                            if torch.is_tensor(new_starts[0]):
                                new_starts = [s.cpu().numpy() for s in new_starts]
                            if torch.is_tensor(new_plans[0]):
                                new_plans = [p.cpu().numpy() for p in new_plans]
                            if torch.is_tensor(new_cs[0]):
                                new_cs = [c.cpu().numpy() for c in new_cs]
                            data[t]['starts'] = np.stack(new_starts, axis=0)
                            data[t]['plans'] = np.stack(new_plans, axis=0)
                            data[t]['contact_state'] = np.stack(new_cs, axis=0)
                        elif torch.is_tensor(new_starts):
                            data[t]['starts'] = new_starts.cpu().numpy()
                            data[t]['plans'] = new_plans.cpu().numpy()
                            data[t]['contact_state'] = new_cs.cpu().numpy()

                        try:
                            data[t]['contact_state'] = data[t]['contact_state'][fl_improvement_bool]

                        except:
                            data[t]['contact_state'] = data[t]['contact_state'][fl_improvement_bool[:-1]]
                            
                        try:
                            data[t]['starts'] = data[t]['starts'][fl_improvement_bool]
                        except:
                            data[t]['starts'] = data[t]['starts'][fl_improvement_bool[:-1]]
                            
                        try:
                            data[t]['plans'] = data[t]['plans'][fl_improvement_bool]
                        except:
                            data[t]['plans'] = data[t]['plans'][fl_improvement_bool[:-1]]
                        # if dropped_recovery:
                        #     data[t]['starts'] = data[t]['starts'][:-1]
                        #     data[t]['plans'] = data[t]['plans'][:-1]
                        #     data[t]['contact_state'] = data[t]['contact_state'][:-1]
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

                            if type_ == 'diffusion':
                                # duplicated first control, rearrange so that it is (x_0, u_0, x_1, u_1, ..., x_{T-1}, u_{T-1}, x_T, 0)
                                traj[:, :, :-1, self.dx:] = traj[:, :, 1:, self.dx:]
                                traj[:, :, -1, self.dx:] = 0
                            elif type_ == 'cnf':
                                traj[:, :, 0, self.dx:] = 0
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

                    #     if type_ == 'diffusion':
                    #         # duplicated first control, rearrange so that it is (x_0, u_0, x_1, u_1, ..., x_{T-1}, u_{T-1}, x_T, 0)
                    #         traj[:, :, :-1, 15:] = traj[:, :, 1:, 15:]
                    #         traj[:, :, -1, 15:] = 0
                    #     elif type_ == 'cnf':
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
            self.trajectories = self.trajectories[:, :, :self.dx]
        self.trajectory_type = torch.from_numpy(self.trajectory_type)
        self.trajectory_type = 2 * (self.trajectory_type - 0.5)  # scale to be [-1, 1]

        print(self.trajectories.shape)
        # TODO consider alternative SO3 representation that is better for learning


        self.masks = self.masks[:, :, None].repeat(1, 1, self.trajectories.shape[-1] + int(self.cosine_sine))  # for states
        self.mean = 0
        self.std = 1

        if self.screwdriver:
            pre_shape = self.trajectories.shape
            final_roll = self.trajectories[:, -1, 12].abs()
            final_pitch = self.trajectories[:, -1, 13].abs()
            dropped = (final_roll > .25) | (final_pitch > .25)
            self.trajectories = self.trajectories[~dropped]
            self.masks = self.masks[~dropped]
            self.trajectory_type = self.trajectory_type[~dropped]

            post_shape = self.trajectories.shape

            print(f'# Trajectories: {pre_shape[0]} -> {post_shape[0]}')

            final_yaw = self.trajectories[:, -1, -1]
            initial_yaw = self.trajectories[:, 0, -1]
            yaw_change = final_yaw - initial_yaw
            print(yaw_change.mean())

            bad_turn = yaw_change > -.5
            self.trajectories = self.trajectories[~bad_turn]
            self.masks = self.masks[~bad_turn]
            self.trajectory_type = self.trajectory_type[~bad_turn]
            post_bad_turn_shape = self.trajectories.shape

            print(f'# Trajectories: {post_shape[0]} -> {post_bad_turn_shape[0]}')
            final_yaw = self.trajectories[:, -1, -1]
            initial_yaw = self.trajectories[:, 0, -1]
            yaw_change = final_yaw - initial_yaw
            print(yaw_change.mean())
            
        elif 'recovery' not in str(folders[0]):
            pre_shape = self.trajectories.shape
            final_yaw = self.trajectories[:, -1, 12]
            initial_yaw = self.trajectories[:, 0, 12]
            yaw_change = final_yaw - initial_yaw
            
            bad_turn = yaw_change > -np.pi/8
            self.trajectories = self.trajectories[~bad_turn]
            self.masks = self.masks[~bad_turn]
            self.trajectory_type = self.trajectory_type[~bad_turn]
            post_shape = self.trajectories.shape
            print(f'# Trajectories: {pre_shape[0]} -> {post_shape[0]}')
            final_yaw = self.trajectories[:, -1, 12]
            initial_yaw = self.trajectories[:, 0, 12]
            yaw_change = final_yaw - initial_yaw
            print('Average yaw change:', yaw_change.mean())     
            yaw_for_plot = initial_yaw.cpu().numpy()
            plt.hist(yaw_for_plot, bins=100)
            plt.title('Yaw Change')
            plt.xlabel('Yaw')
            plt.ylabel('Count')
            plt.show()
        
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
        dx = self.dx
        if self.screwdriver:
            ## randomly perturb angle of screwdriver
            traj[:, dx-1] += 2 * np.pi * (np.random.rand() - 0.5)
        # else:
        #     # Randomly add a multiple of pi/2 to the yaw angle to account for the valve symmetry
        #     traj[:, 12] += (np.random.randint(-3, 4) * np.pi / 2)
            
        #     # modulo to make sure between [-pi and pi]. Original range was arbitrary
        #     traj[:, 12] = (traj[:, 12] + np.pi) % (2.0 * np.pi) - np.pi  # subtract to make [-pi, pi]

        if self.cosine_sine:
                traj_q = traj[:, :(dx-1)]
                traj_theta = traj[:, (dx-1)][:, None]
                traj_u = traj[:, dx:]
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
            dim = self.dx
        else:
            dim =self.dx + 12 + 9

        dxm1 = self.dx-1
        dxp1 = self.dx+1

        # for angle we force to be between [-1, 1]
        if self.cosine_sine:
            self.mean = torch.zeros(dim + 1)
            self.std = torch.ones(dim + 1)
            self.mean[:dxm1] = mean[:dxm1]
            self.std[:dxm1] = torch.from_numpy(std[:dxm1]).float()
            self.mean[dxp1:] = mean[self.dx:]
            self.std[dxp1:] = torch.from_numpy(std[self.dx:]).float()
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
                 skip_pregrasp=False, type_='diffusion'):
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

                    if type_ == 'diffusion':
                        # duplicated first control, rearrange so that it is (x_0, u_0, x_1, u_1, ..., x_{T-1}, u_{T-1}, x_T, 0)
                        traj[:, :, :-1, 15:] = traj[:, :, 1:, 15:]
                        traj[:, :, -1, 15:] = 0
                    elif type_ == 'cnf':
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

class AllegroScrewDriverPlanTransitionDataset:
    """
    Dataset that represents transitions from the AllegroScrewDriver dataset.
    Also computes constraint violations for each transition.
    
    Args:
        folders: List of folders containing trajectory data
        cosine_sine: Whether to convert yaw to sine/cosine representation
        states_only: Whether to only use state variables
        problems_dict: Dictionary mapping trajectory types to optimization problems
        cache_dir: Directory to store precomputed constraint violations
        batch_size: Number of transitions to process in parallel
    """

    def __init__(self, folders, max_T, cosine_sine=False, states_only=False, problems_dict=None, 
                 cache_dir="/home/abhinav/Documents/ccai/data/constraint_cache",
                 batch_size=1):
        super().__init__()
        
        self.batch_size = batch_size
        self.cosine_sine = cosine_sine

        self.has_constraint_info = problems_dict is not None

        classes = {

        }
        all_x_t = []
        all_x_t1 = []
        all_u_t = []

        min_t = 1

        use_actual_traj = True
        for fpath in folders:
            path = pathlib.Path(fpath)
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
                    for t in range(max_T, min_t - 1, -1):
                        if len(data[t]['starts']) == 0 and t == max_T:
                            print('No starts')
                            need_to_continue = True
                            break
                        if type(data[t]['starts']) == list:
                            data[t]['starts'] = torch.stack(data[t]['starts'], axis=0)
                            data[t]['plans'] = torch.stack(data[t]['plans'], axis=0)
                            data[t]['contact_state'] = torch.stack(data[t]['contact_state'], axis=0)
                        elif type(data[t]['starts']) == np.ndarray:
                            data[t]['starts'] = torch.from_numpy(data[t]['starts']).float()
                            data[t]['plans'] = torch.from_numpy(data[t]['plans']).float()
                            data[t]['contact_state'] = torch.from_numpy(data[t]['contact_state']).float()
                        


                        if len(data[t]['starts']) == 0:
                            print('No starts')
                            need_to_continue = True
                            break

                    if need_to_continue:
                        continue
                    dropped_recovery = data['dropped_recovery']

                    if need_to_continue:
                        continue
                    for t in range(max_T, min_t - 1, -1):
                        x_0 = data[t]['starts'][:, :, None, :][:, :, -1:]
                        traj = data[t]['plans']

                        x_all = torch.cat((x_0, traj), dim=2)
                        for c_ind in range(data[t]['contact_state'].shape[0]):
                            this_c_mode = tuple(data[t]['contact_state'][c_ind].tolist())
                            if this_c_mode not in classes:
                                classes[this_c_mode] = {
                                    'x_t': [],
                                    'x_t1': [],
                                    'u_t': []
                                }
                            for t_idx in range(x_all.shape[2]-1):
                                classes[this_c_mode]['x_t'].append(x_all[c_ind, :, t_idx, :].cpu())
                                classes[this_c_mode]['x_t1'].append(x_all[c_ind, :, t_idx + 1, :].cpu())
                                classes[this_c_mode]['u_t'].append(traj[c_ind, :, t_idx, :].cpu())



        for c_mode in classes:
            classes[c_mode]['x_t'] = torch.stack(classes[c_mode]['x_t'], dim=0)
            classes[c_mode]['x_t1'] = torch.stack(classes[c_mode]['x_t1'], dim=0)
            classes[c_mode]['u_t'] = torch.stack(classes[c_mode]['u_t'], dim=0)

            all_x_t.append(classes[c_mode]['x_t'])
            all_x_t1.append(classes[c_mode]['x_t1'])
            all_u_t.append(classes[c_mode]['u_t'])
        self.all_x_t = torch.cat(all_x_t, dim=0)
        self.all_x_t1 = torch.cat(all_x_t1, dim=0)
        self.all_u_t = torch.cat(all_u_t, dim=0)
        self.classes = classes

        class_index = []
        transition_index = []
        for c_mode in classes:
            class_index += [c_mode] * classes[c_mode]['x_t'].shape[0]
            transition_index += list(range(classes[c_mode]['x_t'].shape[0]))
        self.class_index = class_index
        self.transition_index = transition_index

        # Pre-compute constraint violations for all transitions if needed
        if self.has_constraint_info:
            # Create cache directory if it doesn't exist
            cache_dir = pathlib.Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a unique filename based on the dataset characteristics
            folders_hash = hash(tuple(sorted([str(p) for p in folders])))
            problems_hash = hash(tuple(sorted(self.problems_dict.keys())))
            cache_file = cache_dir / f"constraints_{folders_hash}_{problems_hash}.pt"
            stats_file = cache_dir / f"constraint_stats_{folders_hash}_{problems_hash}.pt"
            
            # Try to load from cache first
            if cache_file.exists() and stats_file.exists():
                print(f"Loading pre-computed constraint violations from {cache_file}")
                try:
                    self.violations = torch.load(cache_file)
                    self.violation_stats = torch.load(stats_file)
                    print(f"Loaded {len(self.violations)} pre-computed constraint violations")
                    print(f"Violation statistics for {len(self.violation_stats)} trajectory types")
                    return
                except Exception as e:
                    print(f"Error loading cache files: {e}")
                    print("Will recompute constraint violations")
            
            # Compute constraints if needed
            print("Pre-computing constraint violations for all transitions...")
            self.violations = self.compute_batched_constraint_violations()
            
            # Calculate statistics for each trajectory type
            self.compute_violation_statistics()
            
            # Save to cache files
            if self.violations is not None:
                print(f"Saving constraint violations to {cache_file}")
                torch.save(self.violations, cache_file)
                torch.save(self.violation_stats, stats_file)

    def compute_batched_constraint_violations(self):
        """Compute constraint violations in batches for efficiency."""
        import tqdm
        
        # Group transitions by trajectory type for proper batching
        transitions_by_type = classes
        
        # Process each trajectory type separately
        for traj_type_key, transitions in tqdm.tqdm(transitions_by_type.items(), 
                                                desc="Processing trajectory types"):
            problem = self.problems_dict[traj_type_key]
            
            # Process in batches
            num_batches = (len(transitions['x_t']) + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min((batch_idx + 1) * self.batch_size, len(transitions))

                batch_x_t = transitions['x_t'][batch_start:batch_end]
                batch_u_t = transitions['u_t'][batch_start:batch_end]
                batch_x_t_1 = transitions['x_t1'][batch_start:batch_end]
                
                # Create combined batch input for true parallel processing
                batch_xu = torch.cat([batch_x_t, batch_u_t], dim=-1)
                
                # Compute violations for the entire batch at once
                batch_violations = self.compute_constraint_violation_batch(
                    batch_x_t, batch_u_t, batch_x_t_1, problem, traj_type_key
                )
                
                # Store results
                for i, idx in enumerate(batch_indices):
                    all_violations[idx] = batch_violations[i]
                    
        # Filter out None values and create tensor
        valid_violations = [v for v in all_violations if v is not None]
        
        if valid_violations:
            return torch.stack(valid_violations)
        else:
            return None

    def compute_constraint_violation_batch(self, batch_x_t, batch_u_t, batch_x_t_1, problem, traj_type_key):
        """
        Compute constraint violations for an entire batch at once.
        
        Args:
            batch_x_t: Batch of current states (batch_size x 15)
            batch_u_t: Batch of actions (batch_size x action_dim)
            batch_x_t_1: Batch of next states (batch_size x 15)
            problem: Optimization problem instance
            traj_type_key: Trajectory type key for statistics
            
        Returns:
            torch.Tensor: Batch of constraint violations
        """
        batch_size = batch_x_t.shape[0]
        
        # Create a batch of inputs
        xu_batch = torch.cat([batch_x_t, batch_u_t], dim=-1).unsqueeze(1)  # Shape: (batch_size, 1, dx+du)
        
        # Update problem with the first state (as a reference)
        # In practice, for constraint evaluation we should evaluate each transition individually
        # but many problems only need the general configuration
        problem.update(start=batch_x_t[0].clone())
        
        # Process the batch through problem preprocessing
        problem._preprocess(xu_batch)
        
        # Compute constraint violations for the batch
        eq_violations, _, _, _ = problem._con_eq(xu_batch, compute_grads=False, 
                                           compute_hess=False, verbose=False)
        
        ineq_violations, _, _, _ = problem._con_ineq(xu_batch, compute_grads=False, 
                                               compute_hess=False, verbose=False)
        
        # Combine violations by summing absolute equality violations and ReLU of inequality violations
        batch_violations = []
        
        for i in range(batch_size):
            # Calculate individual violation for statistical tracking
            if eq_violations is not None and ineq_violations is not None:
                violation = torch.abs(eq_violations[i]).sum() + torch.relu(ineq_violations[i]).sum()
            elif eq_violations is not None:
                violation = torch.abs(eq_violations[i]).sum()
            elif ineq_violations is not None:
                violation = torch.relu(ineq_violations[i]).sum()
            else:
                violation = torch.tensor([0.0], device=batch_x_t.device)
                
            # Track individual violations for this trajectory type
            if traj_type_key not in self.violation_stats:
                self.violation_stats[traj_type_key] = []
            
            self.violation_stats[traj_type_key].append(violation.item())
            batch_violations.append(violation.reshape(1))
            
        return batch_violations

    def compute_violation_statistics(self):
        """
        Compute mean and variance statistics for constraint violations by trajectory type.
        """
        stats = {}
        
        for traj_type_key, violations in self.violation_stats.items():
            if not violations:
                continue
                
            # Convert to tensor for easier computation
            violations_tensor = torch.tensor(violations)
            
            # Compute statistics
            mean = violations_tensor.mean().item()
            var = violations_tensor.var().item()
            median = violations_tensor.median().item()
            min_val = violations_tensor.min().item()
            max_val = violations_tensor.max().item()
            count = len(violations)
            
            # Store in statistics dictionary
            stats[traj_type_key] = {
                'mean': mean,
                'variance': var,
                'median': median,
                'min': min_val,
                'max': max_val,
                'count': count
            }
            
        # Replace raw violations with statistics
        self.violation_stats = stats
        
        # Print summary
        print("\nConstraint Violation Statistics by Trajectory Type:")
        for traj_type, stat in self.violation_stats.items():
            print(f"Type {traj_type}: mean={stat['mean']:.4f}, var={stat['variance']:.4f}, "
                  f"count={stat['count']}, range=[{stat['min']:.4f}, {stat['max']:.4f}]")

    def get_violation_statistics(self, traj_type=None):
        """
        Get constraint violation statistics.
        
        Args:
            traj_type: Optional tuple specifying trajectory type. If None, returns all statistics.
            
        Returns:
            dict: Statistics for the specified trajectory type or all types
        """
        if traj_type is not None:
            if isinstance(traj_type, torch.Tensor):
                traj_type = tuple(traj_type.cpu().numpy().tolist())
            return self.violation_stats.get(traj_type, {})
        return self.violation_stats

    def compute_constraint_violation(self, x_t, u_t, x_t_1, traj_type):
        """
        Compute constraint violations for a single transition.
        
        Args:
            x_t: Current state (dim 15)
            u_t: Action (dim varies)
            x_t_1: Next state (dim 15)
            traj_type: Trajectory type/contact mode (dim 3)
            
        Returns:
            torch.Tensor: Combined constraint violation value or None if no problem found
        """
        if not self.has_constraint_info:
            return None
            
        # Convert trajectory type to tuple for dict lookup
        traj_type_key = tuple(traj_type.cpu().numpy().tolist())
        
        # Get the corresponding problem
        if traj_type_key not in self.problems_dict:
            return None
            
        problem = self.problems_dict[traj_type_key]
        
        # Create a mini-trajectory for the problem to evaluate
        xu = torch.cat([x_t, u_t], dim=-1).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, dx+du)
        
        # Update the problem's start state to match our current state
        problem.update(start=x_t.clone())
        
        # Process the transition through problem preprocessing
        problem._preprocess(xu)
        
        # Compute constraint violations
        equality_violations, _, _, _ = problem._con_eq(xu, compute_grads=False, 
                                                    compute_hess=False, verbose=False)
        
        inequality_violations, _, _, _ = problem._con_ineq(xu, compute_grads=False, 
                                                        compute_hess=False, verbose=False)
        
        # Combine violations by taking their sum
        if equality_violations is not None and inequality_violations is not None:
            combined_violation = torch.abs(equality_violations).sum() + torch.relu(inequality_violations).sum()
            return combined_violation.reshape(1)
        elif equality_violations is not None:
            return torch.abs(equality_violations).sum().reshape(1)
        elif inequality_violations is not None:
            return torch.relu(inequality_violations).sum().reshape(1)
        else:
            return torch.tensor([0.0], device=x_t.device)

    def __len__(self):
        return self.all_x_t.shape[0]

    def __getitem__(self, idx):
        """
        Get a transition tuple with pre-computed constraint violations.
        
        Args:
            idx: Index of transition to retrieve
            
        Returns:
            tuple: (x_t, u_t, x_t_1, traj_type, violation)
        """


        x_t = self.all_x_t[idx]
        u_t = self.all_u_t[idx]
        x_t_1 = self.all_x_t1[idx]

        # Return pre-computed violation
        violation = None
        if self.has_constraint_info and self.violations is not None and idx < len(self.violations):
            violation = self.violations[idx]
        else:
            roll = x_t_1[-3].abs()
            pitch = x_t_1[-2].abs()
            drop = (roll > 0.25) | (pitch > 0.25)
            violation = drop.float().reshape(1)
        
        return x_t, u_t, violation, x_t_1

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


class AllegroTrajectoryTransitionDataset(Dataset):
    """
    Dataset that loads trajectory.pkl files and provides state-action-next_state transitions.
    
    Args:
        folders: List of folders containing trajectory.pkl files
        cosine_sine: Whether to convert yaw to sine/cosine representation
        states_only: Whether to only process state variables
        action_dim: Dimension of action space (default: None, inferred from data)
        state_dim: Dimension of state space (default: 15)
        transform_fn: Optional function to transform states before computing actions
    """
    
    def __init__(self, 
                 folders: list, 
                 cosine_sine: bool = False, 
                 states_only: bool = False,
                 action_dim: Optional[int] = None,
                 state_dim: int = 15,
                 transform_fn = None,
                 num_fingers=3):
        super().__init__()
        self.cosine_sine = cosine_sine
        self.states_only = states_only
        self.state_dim = state_dim
        self.transform_fn = transform_fn
        
        # Storage for transitions
        states = []
        actions = []
        next_states = []
        
        # Load all trajectory.pkl files
        for folder in folders:
            path = pathlib.Path(folder)
            trajectory_files = list(path.rglob('*trajectory.pkl'))
            
            for traj_file in trajectory_files:
                try:
                    with open(traj_file, 'rb') as f:
                        trajectory = pickle.load(f)
                        
                        # Process trajectory to extract transitions
                        if len(trajectory) > 1:  # Need at least 2 states for a transition
                            traj_states, traj_actions, traj_next_states = self._process_trajectory(trajectory)
                            states.extend(traj_states)
                            actions.extend(traj_actions)
                            next_states.extend(traj_next_states)
                except Exception as e:
                    print(f"Error loading {traj_file}: {e}")
        
        if not states:
            raise ValueError("No valid transitions found in the provided folders")
        
        # Convert to tensors
        self.states = torch.stack(states)
        self.actions = torch.stack(actions)[:, :num_fingers*4]
        self.next_states = torch.stack(next_states)
        
        # Infer action dimension if not provided
        if action_dim is None:
            self.action_dim = self.actions.shape[1]
        else:
            self.action_dim = action_dim
            
        # Apply cosine/sine transformation if needed
        self.roll = self.next_states[:, -3].clone()
        self.pitch = self.next_states[:, -2].clone()
        self.dropped = (self.roll.abs() > 0.25) | (self.pitch.abs() > 0.25)
        self.dropped = self.dropped.float().reshape(-1)
        if self.cosine_sine:
            self._apply_cosine_sine_transform()
            
        print(f"Loaded {len(self.states)} transitions")
        print(f"State shape: {self.states.shape}, Action shape: {self.actions.shape}")
    
    def return_as_numpy(self):
        """
        Return as list of tuples (s_t, a_t, dropped, s_t+1)

        s_t, a_t, s_t+1 are np arrays 
        """
        return [(self.states[i].cpu().numpy(), self.actions[i].cpu().numpy(), self.dropped[i].cpu().item(), self.next_states[i].cpu().numpy()) for i in range(len(self))]

    def _process_trajectory(self, trajectory):
        """
        Process a trajectory to extract (state, action, next_state) tuples.
        
        Args:
            trajectory: List of states from trajectory.pkl
            
        Returns:
            tuple: (states, actions, next_states)
        """
        states = []
        actions = []
        next_states = []
        
        # Convert trajectory to tensor if it's not already
        if not isinstance(trajectory, torch.Tensor):
            trajectory = torch.tensor(trajectory, dtype=torch.float32)
        
        # Apply any state transformations if needed
        if self.transform_fn is not None:
            trajectory = self.transform_fn(trajectory)
        
        # Extract transitions
        for i in range(len(trajectory) - 1):
            state = trajectory[i][:self.state_dim].clone()
            next_state = trajectory[i+1][:self.state_dim].clone()
            
            # Extract action - this depends on the structure of your data
            # Option 1: Action is the difference between consecutive states
            # action = next_state - state
            
            # Option 2: Action is stored in the trajectory data
            # This assumes the trajectory data includes actions after the state variables
            if trajectory.shape[1] > self.state_dim:
                action = trajectory[i][self.state_dim:].clone()
            else:
                # Fallback to computing action as the difference between states
                action = next_state - state
            
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
        
        return states, actions, next_states
    
    def _apply_cosine_sine_transform(self):
        """Apply cosine/sine transformation to yaw angle (typically at position 14)"""
        yaw_idx = 14  # Typical position for yaw angle
        
        # Transform states
        yaw = self.states[:, yaw_idx:yaw_idx+1]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        
        # Remove yaw and add cos/sin representation
        states_without_yaw = torch.cat([self.states[:, :yaw_idx], self.states[:, yaw_idx+1:]], dim=1)
        self.states = torch.cat([states_without_yaw, cos_yaw, sin_yaw], dim=1)
        
        # Transform next_states
        yaw = self.next_states[:, yaw_idx:yaw_idx+1]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        
        # Remove yaw and add cos/sin representation
        next_states_without_yaw = torch.cat([self.next_states[:, :yaw_idx], self.next_states[:, yaw_idx+1:]], dim=1)
        self.next_states = torch.cat([next_states_without_yaw, cos_yaw, sin_yaw], dim=1)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        """
        Get a transition tuple.
        
        Args:
            idx: Index of transition to retrieve
            
        Returns:
            tuple: (state, action, next_state)
        """
        return self.states[idx], self.actions[idx], self.next_states[idx]
    
    def get_state_dim(self):
        """Get the dimension of the state space"""
        return self.states.shape[1]
    
    def get_action_dim(self):
        """Get the dimension of the action space"""
        return self.actions.shape[1]
    
    def get_batch(self, batch_size):
        """
        Get a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to include in batch
            
        Returns:
            tuple: (states, actions, next_states)
        """
        indices = torch.randint(0, len(self), (batch_size,))
        return self.states[indices], self.actions[indices], self.next_states[indices]


if __name__ == "__main__":
    # d = AllegroValveDataset('../data/experiments/allegro_turning_data_collection')

    # index_regrasp_problem = AllegroScrewdriver(
    #     start=state[:4 * num_fingers + obj_dof],
    #     goal=goal,
    #     T=params['T'],
    #     chain=params['chain'],
    #     device=params['device'],
    #     object_asset_pos=env.table_pose,
    #     object_location=params['object_location'],
    #     object_type=params['object_type'],
    #     world_trans=env.world_trans,
    #     regrasp_fingers=['index'],
    #     contact_fingers=['middle', 'thumb'],
    #     obj_dof=3,
    #     obj_joint_dim=1,
    #     optimize_force=params['optimize_force'],
    #     default_dof_pos=env.default_dof_pos[:, :16],
    #     obj_gravity=params.get('obj_gravity', False),
    #     min_force_dict=min_force_dict,
    #     full_dof_goal=True,
    #     proj_path=None,
    #     project=True,
    # )

    # thumb_and_middle_regrasp_problem = AllegroScrewdriver(
    #     start=state[:4 * num_fingers + obj_dof],
    #     goal=goal,
    #     T=params['T'],
    #     chain=params['chain'],
    #     device=params['device'],
    #     object_asset_pos=env.table_pose,
    #     object_location=params['object_location'],
    #     object_type=params['object_type'],
    #     world_trans=env.world_trans,
    #     contact_fingers=['index'],
    #     regrasp_fingers=['middle', 'thumb'],
    #     obj_dof=3,
    #     obj_joint_dim=1,
    #     optimize_force=params['optimize_force'],
    #     default_dof_pos=env.default_dof_pos[:, :16],
    #     obj_gravity=params.get('obj_gravity', False),
    #     min_force_dict=min_force_dict,
    #     full_dof_goal=True,
    #     proj_path=None,
    #     project=True,
    # )


    # tp = AllegroScrewdriver(
    #     start=state[:4 * num_fingers + obj_dof],
    #     goal=goal,
    #     T=params['T_orig'] if max_timesteps is None else max_timesteps,
    #     chain=params['chain'],
    #     device=params['device'],
    #     object_asset_pos=env.table_pose,
    #     object_location=params['object_location'],
    #     object_type=params['object_type'],
    #     world_trans=env.world_trans,
    #     contact_fingers=['index', 'middle', 'thumb'],
    #     obj_dof=3,
    #     obj_joint_dim=1,
    #     optimize_force=params['optimize_force'],
    #     default_dof_pos=env.default_dof_pos[:, :16],
    #     turn=True,
    #     obj_gravity=params.get('obj_gravity', False),
    #     min_force_dict=min_force_dict,
    #     full_dof_goal=False,
    #     proj_path=proj_path,
    #     project=False,
    # )

    d = AllegroScrewDriverPlanTransitionDataset(['./data/experiments/allegro_screwdriver_q_learning_data_pi_6'], 12)

    traj, contact, mask = d[0]

    print(traj.shape)
    print(contact.shape)
    print(mask.shape)
