import torch
import numpy as np
from abc import abstractmethod
from better_abc import ABCMeta, abstract_attribute

from astar import AStar

from ccai.models.markov_chain import MarkovChain

class ContactSampler(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample(T, c0):
        pass


class MCContactSampler(ContactSampler):
    def __init__(self, transition_matrix, contact_vec_to_label, *args, **kwargs):
        super().__init__(transition_matrix, contact_vec_to_label, *args, **kwargs)

        self.markov_chain = MarkovChain(transition_matrix)
        self.contact_vec_to_label = contact_vec_to_label

    def contact_vec_list_to_label_list(self, contact_vec_list):
        return [self.contact_vec_to_label[contact_vec] for contact_vec in contact_vec_list]

    def sample(self, T, c0):
        contact_vecs = self.markov_chain.sample(T, c0)
        return self.contact_vec_list_to_label_list(contact_vecs)
    

class Node:
    def __init__(self, trajectory, cost, contact_sequence, likelihoods=None):
        self.trajectory = trajectory
        self.cost = cost
        self.contact_sequence = contact_sequence
        # self.all_trajectory_samples = all_trajectory_samples
        self.likelihoods = likelihoods

    def __hash__(self):
        return hash(self.contact_sequence)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.contact_sequence == other.contact_sequence
        return NotImplemented

class GraphSearch(ContactSampler, AStar):
    def __init__(self, start, model, T, problem_dict, max_depth, heuristic_weight, goal, device,
                 num_samples=16,
                  *args, initial_run=False, multi_particle=False, prior=0, sine_cosine=False, **kwargs):
        ContactSampler.__init__(self, *args, **kwargs)

        self.T = T + 1
        self.dx = 16 if sine_cosine else 15
        self.start = start
        self.start_yaw = start[14].item()
        self.sine_cosine = sine_cosine

        if sine_cosine:
            self.start = self.convert_yaw_to_sine_cosine(self.start)

        self.problem_dict = problem_dict
        self.device = device
        self.model = model

        self.num_samples = num_samples
        self.max_depth = max_depth
        self.heuristic_weight = heuristic_weight

        self.initial_run = initial_run

        self.multi_particle = multi_particle

        self.num_samples_multi = 1
        if self.multi_particle:
            self.num_samples_multi = 1
        else:
            self.num_samples_multi = 16
            self.num_samples = 1

        if prior == 0:
            prior_tensor = torch.tensor([
                    [0, .2, .2, .6],
                    [0, .1, .3, .6],
                    [0, .3, .1, .6],
                    [0, .45, .45, .1]
                ])
        elif prior == 1:
            # prior_tensor = torch.tensor([
            #     [0, .2, .2, .6],
            #     [0, .1, .4, .5],
            #     [0, .4, .1, .5],
            #     [0, .45, .45, .1]
            # ])
            prior_tensor = torch.tensor([
                [0, .1, .1, .8],
                [0, .1, .45, .45],
                [0, .45, .1, .45],
                [0, .45, .45, .1]
            ])

        self.markov_chain = MarkovChain(
            prior_tensor
            # torch.tensor([
            #     [0, .1, .1, .8],
            #     [0, .1, .45, .45],
            #     [0, .45, .1, .45],
            #     [0, .45, .45, .1]
            # ])
        )

        self.goal = float(goal)

        self.neighbors_c_states = torch.tensor([
            [-1., -1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, 1, 1]
        ]).to(self.device)
        
        self.num_c_states = self.neighbors_c_states.shape[0]
        self.neighbors_c_states_orig = self.neighbors_c_states.clone()
        self.neighbors_c_states = self.neighbors_c_states.repeat_interleave(self.num_samples*self.num_samples_multi, 0)

        # Get the index of num_samples that each neighbors_c_states corresponds to
        self.neighbors_c_states_indices = torch.arange(self.num_samples).repeat(self.num_samples_multi)
        self.iter = 0

        self.discount = .9


    def normalize_likelihood(self, likelihood):
        # return torch.nn.functional.softmax(likelihood, dim=0)
        return likelihood / likelihood.sum()

    def get_expected_yaw(self, node):
        if node.trajectory.shape[1] == 0:
            return self.start_yaw
        if self.sine_cosine:
            yaw = self.get_yaw_from_sine_cosine(node.trajectory[:, -1, :16])
        else:
            yaw = node.trajectory[:, -1, 14]
        # likelihood_for_average = torch.nn.functional.softmax(node.likelihoods, dim=0)
        likelihood_for_average = self.normalize_likelihood(node.likelihoods)
        yaw = (yaw * likelihood_for_average.to(yaw.device)).sum().item()
        return yaw

    def get_yaw_from_sine_cosine(self, state):
        # return state[:, 14]
        sine = state[:, 15]
        cosine = state[:, 14]
        return torch.atan2(sine, cosine)
    
    def _goal_reached(self, current):
        yaw = self.get_expected_yaw(current)
        return yaw, yaw <= self.goal

    def is_goal_reached(self, current, goal):
        self.iter += 1
        # if current.trajectory.shape[0] > 0:
        # yaw = current.trajectory[-1, 14].item()
        yaw, success = self._goal_reached(current)
        print(self.iter, current.contact_sequence, yaw)
        return success
        
    def convert_sine_cosine_to_yaw(self, xu):
        """
        xu is shape (N, T, 37)
        Replace the sine and cosine in xu with yaw and return the new xu
        """
        sine = xu[..., 15]
        cosine = xu[..., 14]
        yaw = torch.atan2(sine, cosine)
        xu_new = torch.cat([xu[..., :14], yaw.unsqueeze(-1), xu[..., 16:]], dim=-1)
        return xu_new
    
    def convert_yaw_to_sine_cosine(self, xu):
        """
        xu is shape (N, T, 36)
        Replace the yaw in xu with sine and cosine and return the new xu
        """
        yaw = xu[14]
        sine = torch.sin(yaw)
        cosine = torch.cos(yaw)
        xu_new = torch.cat([xu[:14], cosine.unsqueeze(-1), sine.unsqueeze(-1), xu[15:]], dim=-1)
        return xu_new

    @torch.no_grad()
    def neighbors(self, node):
        neighbors = []
        cur_seq = list(node.contact_sequence)
        max_depth = self.max_depth
        if not self.initial_run:
            max_depth += 1
        if len(cur_seq) == max_depth:
            return neighbors

        if node.trajectory.shape[1] == 0:
            last_state = self.start
            last_state = last_state.reshape(1, -1).repeat(self.num_samples, 1)
        else:
            # last_state = node.trajectory[:, -1, :15]
            last_state = node.trajectory[:, -1, :self.dx]
        last_state = last_state.to(self.device)
        samples, _, likelihood = self.model.sample(N=3*self.num_samples * self.num_samples_multi, start=last_state.reshape(self.num_samples, -1).repeat(3*self.num_samples_multi, 1),
                                    H=self.T, constraints=self.neighbors_c_states[self.num_samples*self.num_samples_multi:])
        

        if likelihood is not None:
            likelihood = likelihood.flatten()
        samples_orig = samples.clone()
        if self.sine_cosine:
            samples = self.convert_sine_cosine_to_yaw(samples)

        for i in range(self.num_c_states - 1):
            sample_range = torch.arange(i*(self.num_samples * self.num_samples_multi), (i+1)*(self.num_samples * self.num_samples_multi))
            c_state = tuple(self.neighbors_c_states_orig[i+1].cpu().tolist())
            mask, mask_no_z, mask_without_z = self.c_state_mask(c_state)
            problem = self.problem_dict[c_state]
            sample_range_mask = samples[sample_range][: , :, mask_without_z]
            problem._preprocess(sample_range_mask, projected_diffusion=True)

            J, _, _ = problem._objective(sample_range_mask)

            likelihood_this_c = likelihood[sample_range] * self.discount ** (len(cur_seq) - 1 * self.initial_run)
            print('likelihood', c_state, likelihood[sample_range])
            # Add node's likelihood to likelihood_this_c
            if node.likelihoods is not None:
                likelihood_this_c += node.likelihoods.repeat(self.num_samples_multi).to(likelihood_this_c.device)
            # likelihood_this_c = constraint_val
            # top_likelihoods = torch.topk(likelihood_this_c, k=self.num_samples, largest=True)
            if self.multi_particle:
                probs = self.normalize_likelihood(likelihood_this_c)
                # print(probs)
                # Sample from the probs
                top_likelihoods = torch.multinomial(probs, self.num_samples, replacement=True)
                # top_likelihoods = torch.topk(constraint_val, k=self.num_samples, largest=True)
                top_samples = samples[sample_range][top_likelihoods]
                top_samples_orig = samples_orig[sample_range][top_likelihoods]

                # print(likelihood[sample_range].max().item())
                # min_violation_ind = top_likelihoods.indices[0]
                # min_violation_ind = torch.argmin(constraint_val)
                sample_likelihoods = likelihood_this_c[top_likelihoods].cpu()
                likelihood_for_average = self.normalize_likelihood(likelihood_this_c[top_likelihoods])
                cost = (J[top_likelihoods] * likelihood_for_average).sum().item()
                traj_this_step = top_samples_orig

                prior_traj_indices = self.neighbors_c_states_indices[top_likelihoods.cpu()]
                # full_traj = torch.cat([node.trajectory, traj_this_step], dim=0)
                old_traj = node.trajectory[prior_traj_indices]
            else:
                # Select argmax likelihood_this_c
                min_violation_ind = torch.argmax(likelihood_this_c)
                cost = J[min_violation_ind].item()
                traj_this_step = samples_orig[sample_range][min_violation_ind]
                sample_likelihoods = likelihood_this_c[min_violation_ind].cpu()
                # Ensure that the sample_likelihood is a vector
                if len(sample_likelihoods.shape) == 0:
                    sample_likelihoods = sample_likelihoods.unsqueeze(0)
                
                old_traj = node.trajectory
                # traj_this_step needs to be a 3D tensor
                traj_this_step = traj_this_step.unsqueeze(0)

            full_traj = torch.cat([old_traj, traj_this_step.cpu()], dim=1)
            neighbors.append(Node(trajectory=full_traj, cost=cost, contact_sequence=tuple(cur_seq + [c_state]), likelihoods=sample_likelihoods))
        return neighbors

    def distance_between(self, n1, n2):
        return n2.cost - n1.cost

    def heuristic_cost_estimate(self, current, goal):
        if current.trajectory.shape[0] == 0:
            return self.heuristic_weight * max(0, self.start_yaw - self.goal)
        c_seq = [((np.array(i) + 1) /2).sum().astype(int) for i in current.contact_sequence]
        if self.initial_run:
            c_seq = [0] + c_seq
        nll = -self.markov_chain.eval_log_prob(c_seq)
        yaw = self.get_expected_yaw(current)
        # yaw = current.trajectory[-1, 14].item()
        # return self.heuristic_weight * max(0, yaw - self.goal)
        # return self.heuristic_weight * (nll * max(0, yaw - self.goal))
        return self.heuristic_weight * (nll + 10* max(0, yaw - self.goal))
        # return self.heuristic_weight * (nll)

        # return self.heuristic_weight * max(0, yaw - self.goal)

    def c_state_mask(self, c_state):
        z_dim = self.problem_dict[c_state].dz
        mask = torch.ones((36 + z_dim), device=self.device).bool()
        if c_state == (-1, -1, -1):
            mask[27:36] = False
        elif c_state == (-1 , 1, 1):
            mask[27:30] = False
        elif c_state == (1, -1, -1):
            mask[30:36] = False
        # Concat False to mask to match the size of x
        mask_no_z = mask.clone()
        if z_dim > 0:
            mask_no_z[-z_dim:] = False
        mask_without_z = mask[:36]
        return mask, mask_no_z, mask_without_z

    def reset(self, c0=None, xu0=None):
        self.contact_sequence = []
        self.xu_sequence = []
        self.costs = []

        if c0 is not None:
            self.contact_sequence.append(c0)
            self.xu_sequence.append(xu0)
            mask, mask_no_z, mask_without_z = self.c_state_mask(c0)
            problem = self.problem_dict[c0]
            problem._preprocess(xu0[..., mask], projected_diffusion=True)
            J, _, _ = problem._objective(xu0[..., mask_no_z], compute_grad=False, compute_hess=False)
            self.costs.append(J.item())

    def sample(self, T, c0, xu0, budget):
        for iter in range(budget):
            pass


class GraphSearchCard(ContactSampler, AStar):
    def __init__(self, start, dx, model, T, problem_dict, max_depth, heuristic_weight, goal, device,
                 num_samples=16,
                  *args, initial_run=False, multi_particle=False, prior=0, sine_cosine=False, **kwargs):
        ContactSampler.__init__(self, *args, **kwargs)

        self.T = T + 1
        self.dx = dx
        self.start = start
        self.start_y = start[9].item()
        self.sine_cosine = sine_cosine

        if sine_cosine:
            self.start = self.convert_yaw_to_sine_cosine(self.start)

        self.problem_dict = problem_dict
        self.device = device
        self.model = model

        self.num_samples = num_samples
        # print('num_samples', num_samples)
        # print('multi particle', multi_particle)
        self.max_depth = max_depth
        self.heuristic_weight = heuristic_weight

        self.initial_run = initial_run

        self.multi_particle = multi_particle

        self.num_samples_multi = 1
        if self.multi_particle:
            self.num_samples_multi = 1
        else:
            self.num_samples_multi = num_samples
            self.num_samples = 1

        if prior == 0:
            prior_tensor = torch.tensor([
                    [0, .2, .2, .6],
                    [0, .1, .3, .6],
                    [0, .3, .1, .6],
                    [0, .45, .45, .1]
                ])
        elif prior == 1:
            # prior_tensor = torch.tensor([
            #     [0, .2, .2, .6],
            #     [0, .1, .4, .5],
            #     [0, .4, .1, .5],
            #     [0, .45, .45, .1]
            # ])
            prior_tensor = torch.tensor([
                [.25, .25, .25, .25],
                [.25, .25, .25, .25],
                [.25, .25, .25, .25],
                [.25, .25, .25, .25],
            ])

        self.markov_chain = MarkovChain(
            prior_tensor
            # torch.tensor([
            #     [0, .1, .1, .8],
            #     [0, .1, .45, .45],
            #     [0, .45, .1, .45],
            #     [0, .45, .45, .1]
            # ])
        )

        self.goal = float(goal)

        self.neighbors_c_states = torch.tensor([
            [-1., -1],
            [-1, 1],
            [1, -1],
            [1, 1]
        ]).to(self.device)
        
        self.num_c_states = self.neighbors_c_states.shape[0]
        self.neighbors_c_states_orig = self.neighbors_c_states.clone()
        self.neighbors_c_states = self.neighbors_c_states.repeat_interleave(self.num_samples*self.num_samples_multi, 0)

        # Get the index of num_samples that each neighbors_c_states corresponds to
        self.neighbors_c_states_indices = torch.arange(self.num_samples).repeat(self.num_samples_multi)
        self.iter = 0

        self.discount = .9


    def normalize_likelihood(self, likelihood):
        # return torch.nn.functional.softmax(likelihood, dim=0)
        return likelihood / likelihood.sum()


    def get_expected_yaw(self, node):
        if node.trajectory.shape[1] == 0:
            return self.start_yaw
        if self.sine_cosine:
            yaw = self.get_yaw_from_sine_cosine(node.trajectory[:, -1, :self.dx])
        else:
            yaw = node.trajectory[:, -1, self.dx-1]
        # likelihood_for_average = torch.nn.functional.softmax(node.likelihoods, dim=0)
        likelihood_for_average = self.normalize_likelihood(node.likelihoods)
        yaw = (yaw * likelihood_for_average.to(yaw.device)).sum().item()
        return yaw
    
    
    def get_expected_yaw(self, node):
        if node.trajectory.shape[1] == 0:
            return self.start_yaw
        if self.sine_cosine:
            yaw = self.get_yaw_from_sine_cosine(node.trajectory[:, -1, :self.dx])
        else:
            yaw = node.trajectory[:, -1, self.dx-1]
        # likelihood_for_average = torch.nn.functional.softmax(node.likelihoods, dim=0)
        likelihood_for_average = self.normalize_likelihood(node.likelihoods)
        yaw = (yaw * likelihood_for_average.to(yaw.device)).sum().item()
        return yaw
    
    def get_expected_y(self, node):
        if node.trajectory.shape[1] == 0:
            return self.start_y
        if self.sine_cosine:
            y = node.trajectory[:, -1, self.dx-3]
        else:
            y = node.trajectory[:, -1, self.dx-2]
        # likelihood_for_average = torch.nn.functional.softmax(node.likelihoods, dim=0)
        likelihood_for_average = self.normalize_likelihood(node.likelihoods)
        y = (y * likelihood_for_average.to(y.device)).sum().item()
        return y

    def get_yaw_from_sine_cosine(self, state):
        # return state[:, 14]
        sine = state[:, self.dx-1]
        cosine = state[:, self.dx-2]
        return torch.atan2(sine, cosine)
    
    def _goal_reached(self, current):
        y = self.get_expected_y(current)
        return y, y <= self.goal

    def is_goal_reached(self, current, goal):
        self.iter += 1
        # if current.trajectory.shape[0] > 0:
        # yaw = current.trajectory[-1, 14].item()
        yaw, success = self._goal_reached(current)
        print(self.iter, current.contact_sequence, yaw)
        return success
        
    def convert_sine_cosine_to_yaw(self, xu):
        """
        xu is shape (N, T, 37)
        Replace the sine and cosine in xu with yaw and return the new xu
        """
        sine = xu[..., self.dx-1]
        cosine = xu[..., self.dx-2]
        yaw = torch.atan2(sine, cosine)
        xu_new = torch.cat([xu[..., :self.dx-2], yaw.unsqueeze(-1), xu[..., self.dx:]], dim=-1)
        return xu_new
    
    def convert_yaw_to_sine_cosine(self, xu):
        """
        xu is shape (N, T, 36)
        Replace the yaw in xu with sine and cosine and return the new xu
        """
        yaw = xu[self.dx-2]
        sine = torch.sin(yaw)
        cosine = torch.cos(yaw)
        xu_new = torch.cat([xu[:self.dx-2], cosine.unsqueeze(-1), sine.unsqueeze(-1), xu[self.dx-1:]], dim=-1)
        return xu_new

    @torch.no_grad()
    def neighbors(self, node):
        neighbors = []
        cur_seq = list(node.contact_sequence)
        max_depth = self.max_depth
        if not self.initial_run:
            max_depth += 1
        if len(cur_seq) == max_depth:
            return neighbors

        if node.trajectory.shape[1] == 0:
            last_state = self.start
            last_state = last_state.reshape(1, -1).repeat(self.num_samples, 1)
        else:
            # last_state = node.trajectory[:, -1, :15]
            last_state = node.trajectory[:, -1, :self.dx]
        last_state = last_state.to(self.device)
        samples, _, likelihood = self.model.sample(N=self.num_c_states*self.num_samples * self.num_samples_multi, start=last_state.reshape(self.num_samples, -1).repeat(self.num_c_states*self.num_samples_multi, 1),
                                    H=self.T, constraints=self.neighbors_c_states)
        

        if likelihood is not None:
            likelihood = likelihood.flatten()
        samples_orig = samples.clone()
        if self.sine_cosine:
            samples = self.convert_sine_cosine_to_yaw(samples)

        for i in range(self.num_c_states):
            sample_range = torch.arange(i*(self.num_samples * self.num_samples_multi), (i+1)*(self.num_samples * self.num_samples_multi))
            c_state = tuple(self.neighbors_c_states_orig[i].cpu().tolist())
            mask, mask_no_z, mask_without_z = self.c_state_mask(c_state)
            problem = self.problem_dict[c_state]
            sample_range_mask = samples[sample_range][: , :, mask_without_z]
            problem._preprocess(sample_range_mask, projected_diffusion=True)

            J, _, _ = problem._objective(sample_range_mask)

            likelihood_this_c = likelihood[sample_range] * self.discount ** (len(cur_seq) - 1 * self.initial_run)
            print(likelihood[sample_range])
            # Add node's likelihood to likelihood_this_c
            if node.likelihoods is not None:
                likelihood_this_c += node.likelihoods.repeat(self.num_samples_multi).to(likelihood_this_c.device)
            # likelihood_this_c = constraint_val
            # top_likelihoods = torch.topk(likelihood_this_c, k=self.num_samples, largest=True)
            if self.multi_particle:
                probs = self.normalize_likelihood(likelihood_this_c)
                # print(probs)
                # Sample from the probs
                top_likelihoods = torch.multinomial(probs, self.num_samples, replacement=True)
                # top_likelihoods = torch.topk(constraint_val, k=self.num_samples, largest=True)
                top_samples = samples[sample_range][top_likelihoods]
                top_samples_orig = samples_orig[sample_range][top_likelihoods]

                # print(likelihood[sample_range].max().item())
                # min_violation_ind = top_likelihoods.indices[0]
                # min_violation_ind = torch.argmin(constraint_val)
                sample_likelihoods = likelihood_this_c[top_likelihoods].cpu()
                likelihood_for_average = self.normalize_likelihood(likelihood_this_c[top_likelihoods])
                cost = (J[top_likelihoods] * likelihood_for_average).sum().item()
                traj_this_step = top_samples_orig

                prior_traj_indices = self.neighbors_c_states_indices[top_likelihoods.cpu()]
                # full_traj = torch.cat([node.trajectory, traj_this_step], dim=0)
                old_traj = node.trajectory[prior_traj_indices]
            else:
                # Select argmax likelihood_this_c
                min_violation_ind = torch.argmax(likelihood_this_c)
                cost = J[min_violation_ind].item()
                traj_this_step = samples_orig[sample_range][min_violation_ind]
                sample_likelihoods = likelihood_this_c[min_violation_ind].cpu()
                # Ensure that the sample_likelihood is a vector
                if len(sample_likelihoods.shape) == 0:
                    sample_likelihoods = sample_likelihoods.unsqueeze(0)
                
                old_traj = node.trajectory
                # traj_this_step needs to be a 3D tensor
                traj_this_step = traj_this_step.unsqueeze(0)

            full_traj = torch.cat([old_traj, traj_this_step.cpu()], dim=1)
            neighbors.append(Node(trajectory=full_traj, cost=cost, contact_sequence=tuple(cur_seq + [c_state]), likelihoods=sample_likelihoods))
        return neighbors

    def distance_between(self, n1, n2):
        return n2.cost - n1.cost

    def heuristic_cost_estimate(self, current, goal):
        if current.trajectory.shape[1] == 0:
            return self.heuristic_weight * (self.start_y - self.goal) ** 2
        # c_seq = [((np.array(i) + 1) /2).sum().astype(int) for i in current.contact_sequence]
        # if self.initial_run:
        #     c_seq = [0] + c_seq
        # nll = -self.markov_chain.eval_log_prob(c_seq)
        y = self.get_expected_y(current)
        # yaw = current.trajectory[-1, 14].item()
        # return self.heuristic_weight * max(0, yaw - self.goal)
        # return self.heuristic_weight * (nll * max(0, yaw - self.goal))
        return self.heuristic_weight * (y - self.goal) ** 2
        # return self.heuristic_weight * (nll)

        # return self.heuristic_weight * max(0, yaw - self.goal)

    def c_state_mask(self, c_state):
        z_dim = self.problem_dict[c_state].dz
        mask = torch.ones((28 + z_dim), device=self.device).bool()
        if c_state == (-1, -1):
            mask[19:25] = False
        elif c_state == (-1 , 1):
            mask[19:22] = False
        elif c_state == (1, -1):
            mask[22:25] = False
        # Concat False to mask to match the size of x
        mask_no_z = mask.clone()
        if z_dim > 0:
            mask_no_z[-z_dim:] = False
        mask_without_z = mask[:28]
        return mask, mask_no_z, mask_without_z

    def reset(self, c0=None, xu0=None):
        self.contact_sequence = []
        self.xu_sequence = []
        self.costs = []

        if c0 is not None:
            self.contact_sequence.append(c0)
            self.xu_sequence.append(xu0)
            mask, mask_no_z, mask_without_z = self.c_state_mask(c0)
            problem = self.problem_dict[c0]
            problem._preprocess(xu0[..., mask], projected_diffusion=True)
            J, _, _ = problem._objective(xu0[..., mask_no_z], compute_grad=False, compute_hess=False)
            self.costs.append(J.item())

    def sample(self, T, c0, xu0, budget):
        for iter in range(budget):
            pass