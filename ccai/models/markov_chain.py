import torch

class MarkovChain:
    def __init__(self, transition_matrix):
        self.transition_matrix = transition_matrix

    def sample(self, T, c0):
        c = [c0]
        for t in range(1, T):
            c.append(torch.multinomial(self.transition_matrix[c[t-1]], 1).squeeze().item())
        return c