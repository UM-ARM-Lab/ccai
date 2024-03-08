import jax.numpy as jnp
import pytest

def get_window_splits(X: jnp.array, window_size: int=3) -> jnp.array:
    """Break a set of trajectories into splits of size `window_size`, where windows are slid incrementally along each
    point in each trajectory spline (along axis 1) to generate individual splits.

    Args:
        X: A set of trajectories, shape (N, T, d).
        window_size: The size of the sliding window.

    Returns:
        The window splits for the set of trajectories, shape (N - window_size + 1, N, window_size, d).
    """
    raise NotImplementedError


def test_rbf_kernel(X: jnp.array, X_bar: jnp.array) -> jnp.array:
    """Compute the RBF kernel between each pair in two distributions of trajectories, using the kernel bandwidth from
    equation (48).

    Args:
        X: A set of trajectories, shape (N, l).
        X_bar: A set of trajectories, shape (N, l).

    Returns:
        The output of the RBF kernel for each pair of trajectories, shape (N, N).
    """
    raise NotImplementedError


def test_structured_rbf_kernel(X: jnp.array, X_bar: jnp.array) -> jnp.array:
    """Evaluate the structured rbf kernel on two trajectories.

    Args:
        X: A set of trajectories, shape (N, T, d).
        X_bar: A set of trajectories, shape (N, T, d).

    Returns:
         The value of the kernel function applied to each (X, X_bar) pair of trajectories, shape (N, N).
    """
    raise NotImplementedError
