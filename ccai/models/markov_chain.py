import torch

class MarkovChain:
    def __init__(self, transition_matrix):
        self.transition_matrix = transition_matrix

    def sample(self, T, c0):
        c = [c0]
        for t in range(1, T):
            c.append(torch.multinomial(self.transition_matrix[c[t-1]], 1).squeeze().item())
        return c

    def eval_log_prob(self, c):
        log_prob = 0
        for t in range(1, len(c)):
            log_prob += torch.log(self.transition_matrix[c[t-1]][c[t]])
        return log_prob