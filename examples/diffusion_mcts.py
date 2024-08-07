import torch
import random
import math
from collections import defaultdict
import numpy as np
ACTION_DICT = {
    0: torch.tensor([[-1.0, -1.0, -1.0]]),  # pregrasp
    1: torch.tensor([[1.0, -1.0, -1.0]]),  # regrasp thumb / middle
    2: torch.tensor([[-1.0, 1.0, 1.0]]),  # regrasp index
    3: torch.tensor([[1.0, 1.0, 1.0]]),  # turn
}

ACTION_TENSOR = torch.cat([ACTION_DICT[i] for i in ACTION_DICT.keys()], dim=0)

CHILDREN = {
    0: [1, 2, 3],
    1: [2, 3],
    2: [1, 3],
    3: [1, 2]
}

class Node:
    def __init__(self, state, sampler, max_depth=10, num_evals=10, prev_action=None, goal=None, prior_enum=0):
        # state is action history
        self.state = state
        self.max_depth = max_depth
        self.sampler = sampler
        self.N = num_evals
        self.children = None
        self.prev_action = prev_action
        self.goal = goal
        self.prior_enum = prior_enum

    def is_terminal(self):
        # terminates once max depth is reached
        if len(self.state) == self.max_depth:
            return True
        ## terminates if there is a turn action -- may remove
        # if '3' in self.state:
        #    return True

    def evaluate(self, initial_x):
        # turn state into contact sequence
        if len(self.state) > 0:
            contact_sequence = []
            for action in self.state:
                contact_sequence.append(ACTION_DICT[action])
            contact_sequence = torch.stack(contact_sequence, dim=1).reshape(1, -1, 3).to(initial_x.device).repeat(self.N,
                                                                                                                  1,
                                                                                                                  1)
        if self.prior_enum == 0:
            transition_d = torch.tensor([
                [0.0, 0.2, 0.2, 0.6],
                [0.0, 0.1, 0.4, 0.5],
                [0.0, 0.4, 0.1, 0.5],
                [0.0, 0.45, 0.45, 0.1]
            ])
        elif self.prior_enum == 1:
            transition_d = torch.tensor([
                [0.0, 0.2, 0.2, 0.6],
                [0.0, 0.1, 0.3, 0.6],
                [0.0, 0.3, 0.1, 0.6],
                [0.0, 0.45, 0.45, 0.1]
            ])
        elif self.prior_enum == 2:
            transition_d = torch.tensor([
                [0.0, 0.1, 0.1, 0.8],
                [0.0, 0.1, 0.45, 0.45],
                [0.0, 0.45, 0.1, 0.45],
                [0.0, 0.45, 0.45, 0.1]
            ])
        else:
            raise ValueError("Invalid prior enum")
        prior_ll = torch.zeros(1, device=initial_x.device)
        discount_factor = 0.9
        k = 0
        #transition_d = transition_d.to(device=initial_x.device)
        for i in range(len(self.state)):
            if i == 0:
                if self.prev_action is None:
                    # only pregrasp action, prob unaffected
                    p = torch.ones(1, device=initial_x.device)
                else:
                    p_row = transition_d[self.prev_action].reshape(4)
                    p = p_row[self.state[i]].reshape(-1)
            else:
                p_row = transition_d[self.state[i-1]].reshape(4)
                p = p_row[self.state[i]]
            p = p.to(device=initial_x.device)
            prior_ll += torch.log(p) * discount_factor ** k
            k += 1
        prior_ll = prior_ll.repeat(self.N)
        if not self.is_terminal():
            # randomly choose rest of sequences - each evaluation will have a different end (marginalizing)
            H = self.max_depth - len(self.state)
            idx_sequence = []
            for i in range(H):
                if i == 0:
                    if len(self.state) == 0:
                        if (self.prev_action is None):
                            p = torch.tensor([1.0, 0.0, 0.0, 0.0])
                        else:
                            p = transition_d[self.prev_action].reshape(-1, 4)
                    else:
                        p = transition_d[self.state[-1]].reshape(-1, 4)
                    p = p.repeat(self.N, 1)
                else:
                    p = transition_d[idx].reshape(-1, 4)

                idx = p.multinomial(num_samples=1, replacement=True).reshape(-1)
                ll = torch.log(p[torch.arange(self.N), idx].reshape(-1)).to(device=prior_ll.device)
                prior_ll += ll * discount_factor ** k
                k += 1
                idx_sequence.append(idx)
            if k > 0:
                prior_ll /= k
            _next_contacts = torch.cat([ACTION_TENSOR[idx].reshape(self.N, 1, 3).to(initial_x.device) for idx in idx_sequence], dim=1)
            if len(self.state) > 0:
                contact_sequence = torch.cat((contact_sequence, _next_contacts), dim=1)
            else:
                contact_sequence = _next_contacts

        _start = initial_x.reshape(1, -1)
        # print(contact_sequence)
        traj, _, likelihood = self.sampler.sample(N=self.N, start=_start, constraints=contact_sequence,
                                                  H=self.max_depth * 16,
                                                  goal=self.goal.reshape(-1, 1))

        # print(traj[0, :, 14])

        # only select some of the most likely

        likelihood, indices = torch.topk(likelihood, k=self.N//2, dim=0)
        traj = traj[indices]

        if torch.is_tensor(prior_ll):
            prior_ll = prior_ll.to(device=likelihood.device)
            if len(prior_ll) == self.N:
                prior_ll = prior_ll[indices]

        # score on no angle change when not turning
        turn_no_contact_score = 0
        turn_in_contact_score = 0
        num_turns = 0
        for i, contact in enumerate(contact_sequence[0]):
            if i > 0:
                idx = i * 16 - 1
            else:
                idx = 0
            x = traj[:, idx:idx + 15, 14]
            if not torch.allclose(contact, torch.ones_like(contact)):
                turn_no_contact_score += torch.mean(torch.linalg.norm(x[:, 1:] - x[:, :-1], dim=-1, ord=2)).item()
            else:
                num_turns += 1
                turn_in_contact_score += torch.mean(torch.linalg.norm(x[:, 1:] - x[:, :-1], dim=-1, ord=2)).item()
        turn_in_contact_score /= max(1, num_turns)
        turn_no_contact_score /= max(1, (len(contact_sequence) - num_turns))

        #overall_score = 0 * torch.mean(- traj[:, -1, 14], dim=0) + 0.1 * torch.mean(likelihood)
        # overall_score -= 0.1 * smooth_score.item()# + 5 * turn_no_contact_score
        #overall_score -= 1.0 * turn_no_contact_score
        #overall_score += 0.25 * turn_in_contact_score

        # print(0.05 * torch.mean(likelihood), 10 * torch.mean(-traj[:, -1, 14]), 50 * upright_score.item(), 100 * smooth_score.item(), 5 * turn_no_contact_score, 5 * turn_in_contact_score)
        #return 0.1 * torch.mean(likelihood).item() #0.1 * overall_score.item()
        #print(torch.mean(likelihood).item(), turn_no_contact_score, turn_in_contact_score, overall_score.item())
        # we are going to give a little bonus to turn actions -- let's say it is a prior
        ##if len(self.state) > 0:
        #    if self.state[0] == 3:
        #        overall_score += 0.05
        # alternatively
        #overal_score = overall_score + 0.1 * prior_ll
        overall_score = torch.mean(likelihood + 0.5 * prior_ll)
        return 10 * overall_score.item() / len(contact_sequence[0]) # average over num stages

    def get_children(self):
        # return all possible children
        if self.is_terminal():
            return set()

        # List of all possible children
        if len(self.state) == 0:
            viable_actions = CHILDREN[self.prev_action] if self.prev_action is not None else [0]
        else:
            viable_actions = CHILDREN[self.state[-1]]

        return {Node(self.state + [i], self.sampler, self.max_depth, self.N, goal=self.goal, prev_action=self.prev_action, prior_enum=self.prior_enum) for i in viable_actions}

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(self.__str__())

    def __str__(self):
        return ''.join(str(x) for x in self.state)


class DiffusionMCTS:
    """
        Monte Carlo Tree Search with Diffusion Model as transition model
    """

    def __init__(self, initial_x, sampler, prev_action=None, exploration_weight=1.0, max_depth=5, num_evals=100, prior_enum=0):

        self.Q = defaultdict(int)  # total action value
        self.N = defaultdict(int)
        self.children = dict()
        self.sampler = sampler
        self.initial_x = initial_x
        self.max_depth = max_depth
        self.exploration_weight = exploration_weight
        self.num_evals = num_evals
        self.prev_action = prev_action
        self.prior_enum = prior_enum

    def do_rollout(self, node):
        """ Improve the tree with one iteration of MCTS """
        path = self._select(node)
        leaf = path[-1]
        self.expand(leaf)
        reward = self.simulate(leaf)
        print(leaf.state, reward)

        self._backpropagate(path, reward)

    def choose(self, node):
        """
            Choose the best action based on the current state
        """
        if node not in self.children or not self.children[node]:
            raise ValueError(f"Node {node} not in the tree")

        def score(n):
            if self.N[n] == 0:
                return float("-inf")
            return self.Q[n] / self.N[n]

        return max(self.children[node], key=score)

    def expand(self, node):
        if node not in self.children:
            self.children[node] = node.get_children()

    def _select(self, node):
        # Find unexplored descendent of Node
        import copy
        path = []
        _node = copy.deepcopy(node)
        while True:
            path.append(_node)
            if _node not in self.children or not self.children[_node]:
                # node is either unexplored or terminal
                return path

            unexplored = self.children[_node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            _node = self._uct_select(_node)

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def simulate(self, node):
        """
            Simulate the environment from the current state
        """
        # choose a random sequence of actions and evaluate
        return node.evaluate(self.initial_x)
        # import copy
        # # print("Non terminal node, depth is ", len(node.state))
        # # otherwise we randomly sample the rest of the sequence
        # contact_sequence = copy.deepcopy(node.state)
        # for _ in range(self.max_depth - len(node.state)):
        #     contact_sequence.append(random.choice(list(ACTION_DICT.keys())))
        # final_node = Node(state=contact_sequence, sampler=self.sampler, max_depth=self.max_depth,
        #                   num_evals=self.num_evals)
        # return final_node.evaluate(self.initial_x)

    def plan(self, n_rollouts, goal):
        # starting node
        node = Node([], self.sampler, max_depth=self.max_depth,
                    num_evals=self.num_evals, prev_action=self.prev_action,
                    goal=goal, prior_enum=prior_enum)
        for i in range(n_rollouts):
            self.do_rollout(node)
            if (i + 1) % 10 == 0:
                _plan = self.choose(node)
                print(i, _plan.state, self.Q[_plan] / self.N[_plan], len(self.Q))
        return self.choose(node).state[0]


if __name__ == "__main__":
    from ccai.models.trajectory_samplers import TrajectorySampler
    import pathlib

    CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

    model_path = 'data/training/allegro_screwdriver/allegro_upweight_angle_augment_diffusion/allegro_screwdriver_diffusion_w_classifier.pt'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    trajectory_sampler = TrajectorySampler(T=16, dx=16, du=21, type='diffusion',
                                           timesteps=256, hidden_dim=128,
                                           context_dim=3, generate_context=False,
                                           discriminator_guidance=True)
    trajectory_sampler.load_state_dict(torch.load(f'{CCAI_PATH}/{model_path}'))
    trajectory_sampler.to(device=device)

    initial_x = torch.tensor([
        [0.0, 0.5, 0.65, 0.65, -0.2, 0.5, 0.7, 0.7, 1.3, 0.3, 0.2, 1.1, 0.0, 0.0, 0.0]
    ], device=device)


    # let's check the best plan has a good evaluation
    def eval_plan(plan):
        best_node = Node(plan, trajectory_sampler, max_depth=5, num_evals=100, prior_enum=prior_enum)
        return best_node.evaluate(initial_x)


    import numpy as np

    best_plan = [0, 3, 1, 2, 3]
    reward_for_best = eval_plan(best_plan)
    partial_plan = [0, 3]
    partial_plan2 = [0, 2]
    partial_plan3 = [0, 1]
    for _ in range(10):
        print('--')
        print(eval_plan(partial_plan))
        print(eval_plan(partial_plan2))
        print(eval_plan(partial_plan3))

    print(reward_for_best)
    print(eval_plan(partial_plan))
    print(eval_plan([3, 3, 1, 3, 3]))
    # for i in range(100):
    #     plan = np.random.randint(0, 4, 5)
    #     print(plan, eval_plan(plan), reward_for_best)
    # exit(0)
    mcts = DiffusionMCTS(initial_x, trajectory_sampler, max_depth=5, num_evals=250, exploration_weight=1.0)

    contact_sequence = mcts.plan(n_rollouts=1000)

    contact_sequence = [ACTION_DICT[contact] for contact in contact_sequence]
    print(contact_sequence)
