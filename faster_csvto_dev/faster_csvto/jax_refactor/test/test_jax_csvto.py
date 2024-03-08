from typing import Optional

import jax.numpy as jnp
import pytest


def test_compile(self) -> None:
    """Compile the solve() routine ahead of time to maximize performance for iterative solve() calls on the same
    problem, as will be done in the typical MPC use case.

    NOTE: This will involve computation and subsequent compilation of the gradient and hessian routines for the
    functions specified in the Problem which require derivatives. JIT allows for calling this at runtime,
    potentially enabling behaviors that initialize different optimization problems online as different scenarios
    are encountered.
    """
    raise NotImplementedError


def test_solve(self, x0: jnp.array) -> jnp.array:
    """Execute the CSVTO algorithm to solve for an optimal distribution of trajectories, given an initial guess.

    Args:
        x0: The initialization for the trajectory distribution, shape (N, (dx + du) * T).

    Returns:
        The optimal trajectory from the optimized distribution, shape ((dx + du) * T,).
    """
    raise NotImplementedError


def test_solve_iteration(self, iteration: int, xuz: jnp.array, initial_state: jnp.array) -> jnp.array:
    """Compute a single retraction step for the optimizer and return the updated state.

    Args:
        iteration: The iteration number, in range [0, K).
        xuz: All N trajectories including slack variables stacked as rows, shape (N, (dx + du) * T + dg).
        initial_state: The state of the initial condition for the trajectories, shape (dx,).

    Returns:
        The value of xuz after taking the retraction step.
    """
    raise NotImplementedError


def test_constraint_step(self, constraint: jnp.array, constraint_grad: jnp.array) -> jnp.array:
    """Compute the constraint step according to equation (39).

    Args:
        constraint: The combined constraint evaluated at the current state, TODO: shape ...
        constraint_grad: The gradient of the combined constraint evaluated at the current state, TODO: shape ...

    Returns:
        The constraint step, shape (N, (dx + du) * T + dg).
    """
    raise NotImplementedError


def test_tangent_step(self,
                  c_grad: jnp.array,
                  constraint_grad: jnp.array,
                  constraint_hess: jnp.array,
                  k: jnp.array,
                  k_grad: jnp.array,
                  gamma: Optional[jnp.float32]) -> jnp.array:
    """Compute the tangent step according to either equation (38) or equation (46) based on the value of gamma.

    Args:
        c_grad: The gradient of the cost evaluated at the current state, TODO: shape ...
        constraint_grad: The gradient of the constraint evaluated at the current state, TODO: shape ...
        constraint_hess: The hessian of the constraint evaluated at the current state, TODO: shape ...
        k: The kernel function evaluated at the current state, shape (N, N).
        k_grad: The gradient of the kernel function evaluated at the current state, shape (N, N, (dx + du) * T).

    Returns:
        The tangent step for the current state, shape (N, (dx + du) * T + dg).
    """
    raise NotImplementedError


def test_combined_constraint(self, xuz: jnp.array, initial_state: jnp.array) -> jnp.array:
    """Compute the combined equality constraint function. This should be formed from composition of the specified
    equality constraints, specified inequality constraints paired with slack variables, dynamics constraints, and an
    additional constraint for keeping the initial state in place.

    Args:
        xuz: All N trajectories including slack variables stacked as rows, shape (N, (dx + du) * T + dg).
        initial_state: The state of the initial condition for the trajectories, shape (dx,).

    Returns:
        The evaluated combined constraint, shape (N, dh + dg + dx * T).
    """
    raise NotImplementedError


def test_slack_constraint(self, xuz: jnp.array) -> jnp.array:
    """Compute the slack constraint by evaluating the inequality constraint function and adding in the slack
    variables according to equation (33). This function's output will be constrained to zero in the optimizer.

    Args:
        xuz: All N trajectories including slack variables stacked as rows, shape (N, (dx + du) * T + dg).

    Returns:
        The evaluated slack constraint, shape (N, dg).
    """
    raise NotImplementedError


def test_dynamics_constraint(self, xuz: jnp.array) -> jnp.array:
    """Compute the dynamics constraint which enforces that the points in each trajectory spline are in fact related
    by the supplied dynamics function. This function's output will be constrained to zero in the optimizer.

    Args:
        xuz: All N trajectories including slack variables stacked as rows, shape (N, (dx + du) * T + dg).

    Returns:
        The dynamics error on every point except the first one in every trajectory, shape (N, dx * (T - 1)).
    """
    raise NotImplementedError


def test_start_constraint(self, xuz: jnp.array, initial_state: jnp.array) -> jnp.array:
    """Compute the start constraint function which enforces that the starting state in the trajectories is the same
    as the supplied initial state. This function's output will be constrained to zero in the optimizer.

    Args:
        xuz: All N trajectories including slack variables stacked as rows, shape (N, (dx + du) * T + dg).
        initial_state: The state of the initial condition for the trajectories, shape (dx,).

    Returns:
        The start error on every trajectory, shape (N, dx).
    """
    raise NotImplementedError


def test_bound_projection(self, xuz: jnp.array) -> jnp.array:
    """Clip the trajectories inside the (min, max) bounds on state and control inputs specified by self.x_bounds and
    self.u_bounds.

    Args:
        xuz: All N trajectories including slack variables stacked as rows, shape (N, (dx + du) * T + dg).

    Returns:
        The input array with state and control values clamped inside the range, shape (N, (dx + du) * T + dg).
    """
    raise NotImplementedError
