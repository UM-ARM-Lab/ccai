from typing import Tuple

import jax
import jax.numpy as jnp

EPS = 1e-9


def get_window_splits(X: jnp.array, window_size: int=3) -> jnp.array:
    """Break a set of trajectories into splits of size `window_size`, where windows are slid incrementally along each
    point in each trajectory spline (along axis 1) to generate individual splits.

    Args:
        X: A set of trajectories, shape (N, T, d).
        window_size: The size of the sliding window.

    Returns:
        The window splits for the set of trajectories, shape (N - window_size + 1, N, window_size, d).
    """
    splits = jnp.stack([X[:, i:i + window_size] for i in range(0, X.shape[1] - window_size + 1)], axis=0)
    return splits


# def get_window_splits(X: jnp.array, window_size: int=4) -> jnp.array:
#     """Break a trajectory into splits of size `window_size`, where windows are slid incrementally along each
#     point in each trajectory spline (along axis 0) to generate individual splits.
#
#     Args:
#         X: A trajectory, shape (T, d).
#         window_size: The size of the sliding window.
#
#     Returns:
#         The window splits for the trajectory, shape (T - window_size + 1, window_size, d).
#     """
#     splits = jnp.stack([X[i:i + window_size, :] for i in range(0, X.shape[0] - window_size + 1)], axis=0)
#     return splits


def rbf_kernel(X: jnp.array, X_bar: jnp.array) -> jnp.array:
    """Compute the RBF kernel between each pair in two distributions of trajectories, using the kernel bandwidth from
    equation (48).

    Args:
        X: A set of trajectories, shape (N, l).
        X_bar: A set of trajectories, shape (N, l).

    Returns:
        The output of the RBF kernel for each pair of trajectories, shape (N, N).
    """
    N, l = X.shape

    # Compute element wise difference for all trajectory pairs, shape (N, N, l).
    elementwise_difference = jnp.expand_dims(X, axis=0) - jnp.expand_dims(X_bar, axis=1)

    # Compute the squared norm of each trajectory pair difference.
    difference_norm_squared = (elementwise_difference.reshape(-1, 1, l) @
                               elementwise_difference.reshape(-1, l, 1)).reshape(N, N)

    # Compute the kernel bandwidth via equation (48).
    if N == 1:
        h = jax.lax.stop_gradient(jnp.median(jnp.sqrt(difference_norm_squared)) ** 2 / jnp.log(2))
    else:
        h = jax.lax.stop_gradient(jnp.median(jnp.sqrt(difference_norm_squared)) ** 2 / jnp.log(N))

    # Compute the RBF kernel with an extra weight in the denominator for numerical stability.
    batched_rbf_kernel = jnp.exp(-difference_norm_squared / (h + EPS))
    return batched_rbf_kernel


def structured_rbf_kernel(X: jnp.array, X_bar: jnp.array) -> jnp.array:
    """Evaluate the structured rbf kernel on two trajectories.

    Args:
        X: A set of trajectories, shape (N, T, d).
        X_bar: A set of trajectories, shape (N, T, d).

    Returns:
         The value of the kernel function applied to each (X, X_bar) pair of trajectories, shape (N, N).
    """
    # X = X.reshape(8, 12, 16)
    # X_bar = X_bar.reshape(8, 12, 16)
    N, T, d = X.shape

    # When only one window is used, switch to normal RBF kernel.
    if T < 4:
        return rbf_kernel(X.reshape(N, T * d), X_bar.reshape(N, T * d))

    # Get the window splits for applying a batched RBF kernel along.
    X, X_bar = get_window_splits(X), get_window_splits(X_bar)
    M = X.shape[0]
    X = X.reshape(M, N, -1)
    X_bar = X_bar.reshape(M, N, -1)
    l = X.shape[2]

    # Compute the batched element wise difference for all trajectory pairs, shape (M, N, N, l).
    elementwise_difference = jnp.expand_dims(X, axis=1) - jnp.expand_dims(X_bar, axis=2)

    # Compute the batched squared norm of each trajectory pair difference.
    difference_norm_squared = (elementwise_difference.reshape(-1, 1, l) @
                               elementwise_difference.reshape(-1, l, 1)).reshape(M, N, N)

    # Compute the kernel bandwidth via equation (48).
    if N == 1:
        h = jax.lax.stop_gradient(jnp.median(jnp.sqrt(difference_norm_squared)) ** 2 / jnp.log(2))
    else:
        h = jax.lax.stop_gradient(jnp.median(jnp.sqrt(difference_norm_squared)) ** 2 / jnp.log(N))

    # Compute the kernel according to equation (47) with an extra weight in the denominator for numerical stability.
    structured_rbf_kernel_output = jnp.exp(-difference_norm_squared / (h + EPS)).mean(axis=0)
    return structured_rbf_kernel_output


class RBFKernel:
    """
    """

    def __init__(self, length_scale: jnp.float32, output_scale: jnp.float32) -> None:
        """

        Args:
            length_scale:
            output_scale:
        """
        self.l = length_scale
        self.sigma_sq = output_scale

    def __call__(self, X, X_bar) -> jnp.array:
        """

        Args:
            X:
            X_bar:

        Returns:
            The value of the kernel function applied to each (X, X_bar) pair, shape (N, N).
        """
        n, d = X.shape
        m = X_bar.shape[0]
        diff = jnp.expand_dims(X, axis=1) - jnp.expand_dims(X_bar, axis=0)  # diff should be n x m x d
        sq_diff = (diff.reshape(n, m, 1, d) @ diff.reshape(n, m, d, 1)).reshape(n, m)
        h = self.l ** 2
        sigma_sq = self.sigma_sq

        K = sigma_sq * jnp.exp(-0.5 * sq_diff / h)
        return K
