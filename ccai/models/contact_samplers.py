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
    
class GraphSearchContactSampler(ContactSampler, AStar):
    def __init__(self, model, problem_dict, device, *args, **kwargs):
        ContactSampler.__init__(*args, **kwargs)

        self.problem_dict = problem_dict
        self.device = device
        self.model = model
    
    def c_state_mask(self, c_state):
        z_dim = self.problem_dict[c_state].dz
        mask = torch.ones((self.xu_dim + z_dim), device=self.device).bool()
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
        return mask, mask_no_z

    def reset(self, c0=None, xu0=None):
        self.contact_sequence = []
        self.xu_sequence = []
        self.costs = []

        if c0 is not None:
            self.contact_sequence.append(c0)
            self.xu_sequence.append(xu0)
            mask, mask_no_z = self.c_state_mask(c0)
            problem = self.problem_dict[c0]
            problem._preprocess(xu0[:, :, mask], projected_diffusion=True)
            J, _, _ = problem._objective(xu0[:, :, mask_no_z], compute_grad=False, compute_hess=False)
            self.costs.append(J.item())

    def sample(self, T, c0, xu0, budget):
        for iter in range(budget):
            pass