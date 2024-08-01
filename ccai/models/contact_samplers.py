import torch
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
    def __init__(self, trajectory, cost, contact_sequence, all_trajectory_samples, likelihood_indices=None):
        self.trajectory = trajectory
        self.cost = cost
        self.contact_sequence = contact_sequence
        self.all_trajectory_samples = all_trajectory_samples
        self.likelihood_indices = likelihood_indices

    def __hash__(self):
        return hash(self.contact_sequence)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.contact_sequence == other.contact_sequence
        return NotImplemented

class GraphSearch(ContactSampler, AStar):
    def __init__(self, start, model, problem_dict, max_depth, heuristic_weight, goal, device, *args, **kwargs):
        ContactSampler.__init__(self, *args, **kwargs)

        self.start = start
        self.problem_dict = problem_dict
        self.device = device
        self.model = model

        self.num_samples = 32
        self.max_depth = max_depth
        self.heuristic_weight = heuristic_weight

        self.goal = goal

        self.neighbors_c_states = torch.tensor([
            [-1., -1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, 1, 1]
        ]).cuda()
        
        self.num_c_states = self.neighbors_c_states.shape[0]
        self.neighbors_c_states_orig = self.neighbors_c_states.clone()
        self.neighbors_c_states = self.neighbors_c_states.repeat_interleave(self.num_samples, 0)
        self.iter = 0

    def get_yaw_from_sine_cosine(self, state):
        sine = state[15]
        cosine = state[14]
        return torch.atan2(sine, cosine)
    
    def is_goal_reached(self, current, goal):
        self.iter += 1
        if current.trajectory.shape[0] > 0:
            yaw = self.get_yaw_from_sine_cosine(current.trajectory[-1, :16]).item()
            print(self.iter, current.contact_sequence, yaw)
            return yaw <= self.goal
        else:
            return False
        
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

    def neighbors(self, node):
        neighbors = []
        cur_seq = list(node.contact_sequence)
        if len(cur_seq) == self.max_depth:
            return neighbors
        if node.trajectory.shape[0] == 0:
            last_state = self.start
            samples, _, likelihood = self.model.sample(N=self.num_samples, start=last_state.reshape(1, -1),
                                    H=16, constraints=self.neighbors_c_states[:self.num_samples])
            samples_orig = samples.clone()
            samples = self.convert_sine_cosine_to_yaw(samples)
            for i in range(1):
                sample_range = torch.arange(i*self.num_samples, (i+1)*self.num_samples)
                # top_likelihoods = torch.topk(likelihood[sample_range], k=self.num_samples//2, largest=True)
                c_state = tuple(self.neighbors_c_states_orig[i].cpu().tolist())
                mask, mask_no_z, mask_without_z = self.c_state_mask(c_state)
                problem = self.problem_dict[c_state]
                # sample_range_mask = samples[sample_range][top_likelihoods.indices][: , :, mask_without_z]
                sample_range_mask = samples[sample_range][: , :, mask_without_z]
                problem._preprocess(sample_range_mask, projected_diffusion=True)
                J, _, _ = problem._objective(sample_range_mask)
                g, _, _ = problem._con_eq(sample_range_mask, compute_grads=False, compute_hess=False, verbose=False, projected_diffusion=True)
                h, _, _ = problem._con_ineq(sample_range_mask, compute_grads=False, compute_hess=False, verbose=False, projected_diffusion=True)
                g = torch.abs(g)
                constraint_val = g.sum(1)
                if h is not None:
                    h = torch.relu(h)
                    constraint_val += h.sum(1)
                min_violation_ind = torch.argmin(constraint_val)
                cost = J[min_violation_ind].item()
                # traj_this_step = samples[sample_range][top_likelihoods.indices][min_violation_ind]
                traj_this_step = samples_orig[sample_range][min_violation_ind]
                full_traj = torch.cat([node.trajectory, traj_this_step], dim=0)
                neighbors.append(Node(trajectory=full_traj, cost=cost, contact_sequence=tuple(cur_seq + [c_state]), all_trajectory_samples=samples[sample_range].cpu()))
            return neighbors
        else:
            last_state = node.trajectory[-1, :16]
        samples, _, likelihood = self.model.sample(N=3*self.num_samples, start=last_state.reshape(1, -1),
                                    H=16, constraints=self.neighbors_c_states[self.num_samples:])
        samples_orig = samples.clone()
        samples = self.convert_sine_cosine_to_yaw(samples)

        for i in range(self.num_c_states - 1):
            sample_range = torch.arange(i*self.num_samples, (i+1)*self.num_samples)
            # top_likelihoods = torch.topk(likelihood[sample_range], k=self.num_samples//2, largest=True)
            c_state = tuple(self.neighbors_c_states_orig[i+1].cpu().tolist())
            mask, mask_no_z, mask_without_z = self.c_state_mask(c_state)
            problem = self.problem_dict[c_state]
            sample_range_mask = samples[sample_range][: , :, mask_without_z]
            problem._preprocess(sample_range_mask, projected_diffusion=True)
            J, _, _ = problem._objective(sample_range_mask)
            g, _, _ = problem._con_eq(sample_range_mask, compute_grads=False, compute_hess=False, verbose=False, projected_diffusion=True)
            h, _, _ = problem._con_ineq(sample_range_mask, compute_grads=False, compute_hess=False, verbose=False, projected_diffusion=True)
            g = torch.abs(g)
            constraint_val = g.sum(1)
            if h is not None:
                h = torch.relu(h)
                constraint_val += h.sum(1)
            min_violation_ind = torch.argmin(constraint_val)
            cost = J[min_violation_ind].item()
            traj_this_step = samples_orig[sample_range][min_violation_ind]
            full_traj = torch.cat([node.trajectory, traj_this_step], dim=0)
            neighbors.append(Node(trajectory=full_traj, cost=cost, contact_sequence=tuple(cur_seq + [c_state]), all_trajectory_samples=samples[sample_range].cpu()))
        return neighbors

    def distance_between(self, n1, n2):
        return n2.cost - n1.cost

    def heuristic_cost_estimate(self, current, goal):
        if current.trajectory.shape[0] == 0:
            return self.heuristic_weight * .5 * torch.pi
        yaw = self.get_yaw_from_sine_cosine(current.trajectory[-1, :16]).item()
        return self.heuristic_weight * max(0, yaw - self.goal)

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