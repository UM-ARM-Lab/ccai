def structured_rbf_kernel(X, X_bar, Q=None):
    """Evaluate the structured rbf kernel on two trajectories.

    Args:
        X:
        X_bar:
        Q:

    Returns:
         The value of the kernel function.
    """
    # X is N x T x d
    n, T, d = X.shape
    if Q is not None:
        X_bar = (Q.reshape(1, T * d, T * d) @ X_bar.reshape(n, T * d, 1)).reshape(n, T, d)

    # when only one window is used, switch to normal RBF kernel
    if T < 4:
        return rbf_kernel(X.reshape(n, T * d), X_bar.reshape(n, T * d), Q)

    X, X_bar = get_window_splits(X), get_window_splits(X_bar)

    M = X.shape[0]
    X = X.reshape(M, n, -1)
    X_bar = X_bar.reshape(M, n, -1)
    d = X.shape[2]
    diff = X.unsqueeze(1) - X_bar.unsqueeze(2)

    sq_diff = (diff.reshape(-1, 1, d) @ diff.reshape(-1, d, 1)).reshape(M, n, n)
    h = median(torch.sqrt(sq_diff)) ** 2
    h = h / np.log(n) + EPS
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
