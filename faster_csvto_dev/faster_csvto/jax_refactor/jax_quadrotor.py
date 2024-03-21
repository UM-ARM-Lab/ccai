import time
from typing import Optional, Callable

import numpy as np
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from ccai.quadrotor_env import QuadrotorEnv
from jax_csvto import JaxCSVTOProblem, JaxCSVTOParams, JaxCSVTOpt
from jax_gp import GPSurfaceModel
from jax_rbf_kernels import structured_rbf_kernel

# from jax import config
# config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)

# import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class QuadrotorCost:
    """Cost function for the 12 DOF quadrotor example problem.
    """

    def __init__(self, goal_state: jnp.array) -> None:
        """Initialize a QuadrotorCost object with a set goal state.

        Args:
            goal_state: The goal state of the quadrotor, shape (dx,).
        """
        self.goal_state = goal_state
        self.dx = 12
        self.du = 4
        self.T = 12

    def __call__(self, trajectory: jnp.array) -> jnp.array:
        """Compute the cost of a trajectory according to equation (53).

        Args:
            trajectory: A trajectories, shape ((dx + du) * T).

        Returns:
            The cost of the trajectory, shape (1,).
        """
        # Set Q, P, and R according to equations (54), (55), and (56), respectively.
        # Q = jnp.eye(self.dx) * 2.5
        # Q = Q.at[:2, :2].set(jnp.eye(2) * 5)
        # Q = Q.at[2, 2].set(0.5)
        # Q = Q.at[5, 5].set(0.025)
        # P = 2 * Q
        # R = jnp.eye(self.du) * 16
        # R = R.at[0, 0].set(1)

        # Trying the setting currently used in quadrotor_example.py
        Q = jnp.eye(self.dx) * 2.5
        Q = Q.at[0, 0].set(5.0)
        Q = Q.at[1, 1].set(5.0)
        Q = Q.at[2, 2].set(5.0)
        Q = Q.at[5, 5].set(0.01)
        P = Q
        R = jnp.eye(self.du) * 16
        R = R.at[0, 0].set(1)

        # Index out intermediate terms with shape (T, -1).
        points = trajectory.reshape(self.T, self.dx + self.du)
        x = points[:, :self.dx]
        u = points[:, -self.du:]
        goal_error = x - self.goal_state.reshape(1, self.dx)

        # Calculate cost according to equation (53).
        terminal_state_cost = (goal_error[-1, :] @ P @ goal_error[-1, :].reshape(self.dx, 1)).squeeze()
        running_state_cost = (jnp.expand_dims(goal_error, axis=1) @ Q.reshape(1, self.dx, self.dx) @
                              jnp.expand_dims(goal_error, axis=2)).squeeze().sum()
        running_control_cost = (jnp.expand_dims(u, axis=1) @ R.reshape(1, self.du, self.du) @
                                jnp.expand_dims(u, axis=2)).squeeze().sum()
        cost = terminal_state_cost + running_state_cost + running_control_cost
        return cost


class QuadrotorDynamics:
    """Dynamics function for the 12 DOF quadrotor example problem.
    """

    def __init__(self, dt: jnp.float32 = 1e-4) -> None:
        """Initialize a QuadrotorDynamics object with a set time step.

        Args:
            dt: The time step to use for the dynamics function.
        """
        self.dt = dt

    def __call__(self, states: jnp.array, control_inputs: jnp.array) -> jnp.array:
        """Compute the batched dynamics function for the quadrotor according to equation (52).

        Args:
            states: Batched states of the quadrotor, shape (M, dx).
            control_inputs: Batched control inputs of the quadrotor (M, du).

        Returns:
            Batched states of the quadrotor forward one time step, shape (M, dx).
        """
        g = -9.81
        m = 1
        Ix, Iy, Iz = 0.5, 0.1, 0.3
        K = 1

        # Index out batched variables for applying the dynamics via equation (52).
        x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r = jnp.split(states, 12, axis=-1)
        u1, u2, u3, u4 = jnp.split(control_inputs, 4, axis=-1)

        # Trigonometric functions on all the angles needed for dynamics.
        cphi = jnp.cos(phi)
        ctheta = jnp.cos(theta)
        cpsi = jnp.cos(psi)
        sphi = jnp.sin(phi)
        stheta = jnp.sin(theta)
        spsi = jnp.sin(psi)
        ttheta = jnp.tan(theta)

        # TODO: Comment the rest of this clearly.
        x_ddot = -(sphi * spsi + cpsi * cphi * stheta) * K * u1 / m
        y_ddot = - (cpsi * sphi - cphi * spsi * stheta) * K * u1 / m
        z_ddot = g - (cphi * ctheta) * K * u1 / m

        p_dot = ((Iy - Iz) * q * r + K * u2) / Ix
        q_dot = ((Iz - Ix) * p * r + K * u3) / Iy
        r_dot = ((Ix - Iy) * p * q + K * u4) / Iz

        new_xdot = x_dot + x_ddot * self.dt
        new_ydot = y_dot + y_ddot * self.dt
        new_zdot = z_dot + z_ddot * self.dt
        new_p = p + p_dot * self.dt
        new_q = q + q_dot * self.dt
        new_r = r + r_dot * self.dt

        psi_dot = new_q * sphi / ctheta + new_r * cphi / ctheta
        theta_dot = new_q * cphi - new_r * sphi
        phi_dot = new_p + new_q * sphi * ttheta + new_r * cphi * ttheta

        new_phi = phi + phi_dot * self.dt
        new_theta = theta + theta_dot * self.dt
        new_psi = psi + psi_dot * self.dt
        new_x = x + new_xdot * self.dt
        new_y = y + new_ydot + self.dt
        new_z = z + new_zdot + self.dt

        return jnp.concatenate((new_x, new_y, new_z, new_phi, new_theta, new_psi, new_xdot, new_ydot, new_zdot, new_p,
                                new_q, new_r), axis=-1)


def height_constraint(trajectory: jnp.array, surface_gp: GPSurfaceModel) -> jnp.array:
    """

    Args:
        trajectory: Single trajectory, shape ((dx + du) * T).
        surface_gp:
    Returns:

    """
    trajectory = trajectory.reshape(12, 16)
    xy, z = trajectory[:, :2], trajectory[:, 2]

    # compute z of surface and gradient and hessian
    surface_z = surface_gp.posterior_mean(xy)
    constraint = surface_z - z
    return constraint


class QuadrotorGPHeightConstraint:
    """Equality constraint function to enforce that the vertical position of each point in each trajectory spline is on
    the surface of a gaussian process. The gaussian process is read in from a .npz file. Used for the 12 DOF quadrotor
    example problem.
    """

    def __init__(self, surface_file_path: str) -> None:
        """

        Args:
        """
        data = np.load(surface_file_path)
        self.surface_gp = GPSurfaceModel(jnp.array(data['xy'], dtype=jnp.float32),
                                         jnp.array(data['z'], dtype=jnp.float32))

    def __call__(self, trajectory):
        """
        """
        return height_constraint(trajectory, self.surface_gp)


def update_plot_with_trajectories(ax, point_dim: int, best_trajectory: np.array,
                                  other_trajectories: Optional[np.array] = None) -> None:
    """
    """
    # Clear any of the previous lines.
    for line in ax.get_lines():
        line.remove()

    if other_trajectories is not None:
        # Plot the other trajectories.
        for trajectory in other_trajectories:
            ax.plot(trajectory[0::point_dim], trajectory[1::point_dim], trajectory[2::point_dim],
                    color='g', alpha=0.5, linestyle='--')

    # Plot the best trajectory.
    ax.plot(best_trajectory[0::point_dim], best_trajectory[1::point_dim], best_trajectory[2::point_dim], color='g')


def plot_trajectory(env: QuadrotorEnv, ax, state_dim, best_trajectory, other_trajectories, number):
    """
    """
    env.render_update()
    update_plot_with_trajectories(ax, state_dim, best_trajectory, other_trajectories)
    plt.savefig(f'output/mpc{number:02d}.png')
    plt.gcf().canvas.flush_events()


def get_initial_guess_from_start_state(start_state: np.array, N: int, T: int, dx: int, du: int,
                                       dynamics_function: Callable) -> np.array:
    """Get an initial guess by rolling out trajectories from a given start using randomly sampled inputs.

    NOTE: New version for trajectories that start with the second state and have the previous control input.
    """
    # Sample and scale random inputs.
    control_inputs = np.random.randn(N, T, du)
    control_inputs[:, :, 1:] *= 0.25

    # Get the first state for all the trajectories.
    tall_start_state = np.tile(start_state, (N, 1))
    trajectories = dynamics_function(tall_start_state, control_inputs[:, 0, :])
    trajectories = np.concatenate((trajectories, control_inputs[:, 0, :]), axis=1)

    # Use the random inputs to roll out T - 1 more new states.
    for t in range(T-1):
        # Get the new spline point x using the new spline point u and the last added x.
        trajectories = np.concatenate((trajectories,
                                       dynamics_function(trajectories[:, -(dx+du): -du], control_inputs[:, t+1, :]),
                                       control_inputs[:, t+1, :]), axis=1)
    return trajectories


def step_trajectories(trajectories, state_dim):
    """Drop the first point in each trajectory spline and add another copy of the last point to the back to maintain the
    spline length.

    TODO: It might be a good idea to copy over the last u, then roll out the last state for less constraint violation.
    """
    trajectories = trajectories[:, state_dim:]
    trajectories = jnp.concatenate((trajectories, trajectories[:, -state_dim:]), axis=1)
    return trajectories


def roll_out_trajectories(start_state, trajectories, dynamics_function, N, T, dx, du):
    """
    """
    state_dim = dx + du

    trajectories = trajectories[:, state_dim:]
    trajectories = jnp.concatenate((trajectories, trajectories[:, -state_dim:]), axis=1)

    points = trajectories.reshape(N, T, state_dim)
    control_inputs = points[:, :, -du:]

    tall_start_state = np.tile(start_state, (N, 1))
    trajectories = dynamics_function(tall_start_state, control_inputs[:, 0, :])
    trajectories = np.concatenate((trajectories, control_inputs[:, 0, :]), axis=1)

    # Use the random inputs to roll out T - 1 more new states.
    for t in range(T-1):
        # Get the new spline point x using the new spline point u and the last added x.
        trajectories = np.concatenate((trajectories,
                                       dynamics_function(trajectories[:, -(dx+du): -du], control_inputs[:, t+1, :]),
                                       control_inputs[:, t+1, :]), axis=1)
    return trajectories


def main() -> None:
    """Set up, compile, and solve a JaxCSVTOpt for the quadrotor problem with no obstacles.
    """
    # Set up the problem.
    goal_state = jnp.zeros(12, dtype=jnp.float32)
    goal_state = goal_state.at[0:2].set(4)
    cost_function = QuadrotorCost(goal_state)
    equality_constraint_function = QuadrotorGPHeightConstraint('../surface_data.npz')
    inequality_constraint_function = None
    dynamics_function = QuadrotorDynamics()
    u_bounds = (jnp.array((-100.0000, -100.0000, -100.0000, -100.0000)),
                jnp.array((100.0000, 100.0000, 100.0000, 100.0000)))
    x_bounds = (jnp.array((-6.0000, -6.0000, -6.0000, -1.2566, -1.2566, -1000.0000, -100.0000, -100.0000, -100.0000,
                           -100.0000, -100.0000, -100.0000)),
                jnp.array((6.0000, 6.0000, 6.0000, 1.2566, 1.2566, 1000.0000, 100.0000,
                          100.0000, 100.0000, 100.0000, 100.0000, 100.0000)))
    dx = 12
    du = 4
    dh = 12
    dg = 0
    quadrotor_problem = JaxCSVTOProblem(cost_function, equality_constraint_function, inequality_constraint_function,
                                        dynamics_function, u_bounds, x_bounds, dx, du, dh, dg)

    # Set up the parameters.
    k = structured_rbf_kernel
    N = 8
    T = 12
    K = 1
    anneal = True
    alpha_J = 1
    alpha_C = 1
    step_scale = 1e-2
    penalty_weight = 1e2
    quadrotor_parameters = JaxCSVTOParams(k, N, T, K, anneal, alpha_J, alpha_C, step_scale, penalty_weight)

    # Set up the optimization and compile its solve() routine.
    quadrotor_opt = JaxCSVTOpt(quadrotor_problem, quadrotor_parameters)
    initial_state = jnp.zeros(dx)
    initial_guess = jnp.array(np.random.random([N, 16 * 12]), dtype=jnp.float32)
    solve_compiled = jax.jit(quadrotor_opt.solve).lower(initial_state, initial_guess).compile()

    # Initialize the environment.
    env = QuadrotorEnv(False, '../surface_data.npz')
    env.reset()
    env.render_init()
    ax = env.ax

    # Get the start from the environment.
    start = env.state
    trajectories = get_initial_guess_from_start_state(start, N, T, dx, du, dynamics_function)
    best_trajectory = None

    # Loop over MPC updates.
    for mpc_step in range(100):
        # Optimize the trajectory distribution.
        if mpc_step == 0:
            # Solve for the trajectory distribution to warm up.
            for _ in range(100):
                best_trajectory, trajectories = solve_compiled(start, trajectories)
        else:
            # Solve for the trajectory distribution online.
            for _ in range(10):
                best_trajectory, trajectories = solve_compiled(start, trajectories)

        # Run the first input from the best trajectory on the simulated quadrotor.
        print(best_trajectory[dx:dx + du])
        start, _ = env.step(best_trajectory[dx:dx + du])

        # Step the trajectories so that they can be used as the initial guess in the next iteration.
        # trajectories = step_trajectories(trajectories, dx + du)
        trajectories = roll_out_trajectories(start, trajectories, dynamics_function, N, T, dx, du)
        best_trajectory = roll_out_trajectories(start, jnp.expand_dims(best_trajectory, axis=0), dynamics_function,
                                                1, T, dx, du)

        # Plot the best trajectories.
        print(f'Ran MPC step {mpc_step}')
        plot_trajectory(env, ax, dx + du, best_trajectory.squeeze(), trajectories, mpc_step)


if __name__ == '__main__':
    main()
