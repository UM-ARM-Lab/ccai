import torch

def median(tensor):
    """
    torch.median() acts differently from np.median(). We want to simulate numpy implementation.
    """
    tensor = tensor.detach().flatten()
    tensor_max = tensor.max()[None]
    return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.


def rbf_kernel(X, Xbar, M=None):
    # X is N x d
    n, d = X.shape
    diff = X.unsqueeze(0) - Xbar.unsqueeze(1)
    if M is not None:
        scaled_diff = diff @ M.unsqueeze(0)
    else:
        scaled_diff = diff

    scaled_diff = (scaled_diff.reshape(-1, 1, d) @ diff.reshape(-1, d, 1)).reshape(n, n)
    h = median(scaled_diff)
    h = torch.sqrt(h / 2) + 1e-3
    # h = 0.1
    return torch.exp(-0.5 * scaled_diff / h)
