from typing import Optional

import jax
import jax.numpy as jnp

def structured_rbf_kernel(X: jnp.array, X_bar: jnp.array, Q: Optional[jnp.array] = None) -> jnp.array:
    """Evaluate the structured rbf kernel on two trajectories.

    Args:
        X: A distribution of trajectories, shape (N, T, dx + du).
        X_bar: A distribution of trajectories, shape (N, T, dx + du).
        Q:

    Returns:
         The value of the kernel function applied to each (X, X_bar) pair of trajectories, shape (N, N).
    """
    N, T, d = X.shape
    if Q is not None:
        X_bar = (Q.reshape(1, T * d, T * d) @ X_bar.reshape(N, T * d, 1)).reshape(N, T, d)

    # when only one window is used, switch to normal RBF kernel
    if T < 4:
        return rbf_kernel(X.reshape(N, T * d), X_bar.reshape(N, T * d), Q)

    X, X_bar = get_window_splits(X), get_window_splits(X_bar)

    M = X.shape[0]
    X = X.reshape(M, N, -1)
    X_bar = X_bar.reshape(M, N, -1)
    d = X.shape[2]
    diff = X.unsqueeze(1) - X_bar.unsqueeze(2)

    sq_diff = (diff.reshape(-1, 1, d) @ diff.reshape(-1, d, 1)).reshape(M, N, N)
    h = median(torch.sqrt(sq_diff)) ** 2
    h = h / np.log(N) + EPS
    return torch.exp(-sq_diff / h.reshape(M, 1, 1)).mean(dim=0)


def get_window_splits(X, window_size=3):
    splits = [X[:, i:i + window_size] for i in range(0, X.shape[1] - window_size + 1, 1)]
    return torch.stack(splits, dim=0)


def median(tensor):
    """
    torch.median() acts differently from np.median(). We want to simulate numpy implementation.
    """
    if len(tensor.shape) > 2:
        n = tensor.shape[1]
    else:
        n = tensor.shape[0]
        tensor = tensor.unsqueeze(0)
    m = tensor.shape[0]
    idx = torch.triu_indices(n, n, 1)
    tensor = tensor[:, idx[0], idx[1]]
    tensor = tensor.detach().reshape(m, -1)
    tensor_max = tensor.max(dim=1).values.reshape(-1, 1)
    return (torch.cat((tensor, tensor_max), dim=1).median(dim=1).values + tensor.median(dim=1).values) / 2.

def get_chunk(X, num_chunks, expected_chunk_size):
    x = torch.chunk(X, chunks=num_chunks, dim=1)
    if x[-1].shape[1] < expected_chunk_size:
        x = x[:-1]
    return x

def rbf_kernel(X, Xbar, Q=None):
    # X is N x d
    n = X.shape[0]
    X = X.reshape(n, -1)
    Xbar = Xbar.reshape(n, -1)
    n, d = X.shape
    diff = X.unsqueeze(0) - Xbar.unsqueeze(1)
    # Q must be PD - add diagonal to make it so
    if Q is not None:
        scaled_diff = diff @ Q.unsqueeze(0)
    else:
        scaled_diff = diff

    scaled_diff = (scaled_diff.reshape(-1, 1, d) @ diff.reshape(-1, d, 1)).reshape(n, n)
    h = median(torch.sqrt(scaled_diff)) ** 2
    h = h / np.log(n) + EPS
    # h = 0.1
    return torch.exp(-scaled_diff / h)
