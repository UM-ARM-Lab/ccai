import numpy as np
import jax
import jax.numpy as jnp

from jax_csvto import JaxCSVTOProblem, JaxCSVTOParams, JaxCSVTOpt
from jax_gp import GPSurfaceModel
from jax_rbf_kernels import structured_rbf_kernel


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

    def __call__(self, trajectories: jnp.array) -> jnp.array:
        """Compute the cost of a set of trajectories according to equation (53).

        Args:
            trajectories: Set of trajectories, shape (N, (dx + du) * T).

        Returns:
            The cost of each trajectory, shape (N,).
        """
        assert len(trajectories.shape) == 2
        assert trajectories.shape[1] == (self.dx + self.du) * self.T

        # Set Q, P, and R according to equations (54), (55), and (56), respectively.
        Q = jnp.eye(self.dx) * 2.5
        Q = Q.at[:2, :2].set(jnp.eye(2) * 5)
        Q = Q.at[2, 2].set(0.5)
        Q = Q.at[5, 5].set(0.025)
        P = 2 * Q
        R = jnp.eye(self.du) * 16
        R = R.at[0, 0].set(1)

        # Index out intermediate terms with shape (N, T, -1).
        points = trajectories.reshape(-1, self.T, self.dx + self.du)
        x = points[:, :, self.dx]
        u = points[:, :, -self.du:]
        goal_error = x - self.goal_state.reshape(1, 1, self.dx)

        # Calculate cost according to equation (53).
        terminal_state_cost = (goal_error[:, -1, :] @ P.reshape(1, self.dx, self.dx) @
                               goal_error[:, -1, :].reshape(-1, self.dx, 1)).squeeze()
        running_state_cost = (jnp.expand_dims(goal_error, axis=2) @ Q.reshape(1, 1, self.dx, self.dx) @
                              jnp.expand_dims(goal_error, axis=-1)).sum(axis=1).squeeze()
        running_control_cost = (jnp.expand_dims(u, axis=2) @ R.reshape(1, 1, self.du, self.du) @
                                jnp.expand_dims(u, axis=-1)).sum(axis=1).squeeze()
        cost = terminal_state_cost + running_state_cost + running_control_cost
        return cost


class QuadrotorDynamics:
    """Dynamics function for the 12 DOF quadrotor example problem.
    """

    def __init__(self, dt: jnp.float32 = 0.1) -> None:
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
        trajectory: Single trajectory, shape (N, dx + du).
        surface_gp:
    Returns:
    """
    trajectory = trajectory.reshape(12, 16)
    T = trajectory.shape[0]
    xy, z = trajectory[:, :2], trajectory[:, 2]

    # compute z of surface and gradient and hessian
    surface_z, grad_surface_z, hess_surface_z = surface_gp.posterior_mean(xy)

    constr = surface_z - z
    # grad_constr = jnp.concatenate((grad_surface_z,
    #                                -jnp.ones((T, 1)),
    #                                jnp.zeros((T, 13))), axis=1)
    # hess_constr = jnp.concatenate((
    #     jnp.concatenate((hess_surface_z, jnp.zeros((T, 2, 14))), axis=2),
    #     jnp.zeros((T, 14, 16))), axis=1)
    # return constr, grad_constr, hess_constr
    return constr


class QuadrotorGPHeightConstraint:
    """Equality constraint function to enforce that the vertical position of each point in each trajectory spline is on
    the surface of a gaussian process. The gaussian process is read in from a .npz file. Used for the 12 DOF quadrotor
    example problem.
    """

    def __init__(self, surface_file_path: str) -> None:
        """

        Args:
        """
        data = np.load('surface_data.npz')
        self.surface_gp = GPSurfaceModel(jnp.array(data['xy'], dtype=jnp.float32),
                                         jnp.array(data['z'], dtype=jnp.float32))

    def __call__(self, trajectory):
        """
        """
        return jax.vmap(height_constraint, (0, None), 0)(trajectory, self.surface_gp)


def main() -> None:
    """Set up, compile, and solve a JaxCSVTOpt for the quadrotor problem with no obstacles.
    """
    # Set up the problem.
    goal_state = jnp.zeros(12, dtype=jnp.float32)
    cost_function = QuadrotorCost(goal_state)
    equality_constraint_function = QuadrotorGPHeightConstraint('surface_data.npz')
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
    K = 100
    anneal = True
    alpha_J = 1
    alpha_C = 1
    step_scale = 1e-2
    penalty_weight = 1e2
    quadrotor_parameters = JaxCSVTOParams(k, N, T, K, anneal, alpha_J, alpha_C, step_scale, penalty_weight)

    # Set up and solve the optimization.
    quadrotor_opt = JaxCSVTOpt(quadrotor_problem, quadrotor_parameters)
    initial_guess = jnp.array(np.random.random([8, 16 * 12]), dtype=jnp.float32)

    solution = quadrotor_opt.solve(initial_guess)
    print(solution)

    # # Lower and compile JaxCSVTOpt.solve() ahead of time, marking the class (and all its attributes) as static.
    # solve_method = jax.jit(quadrotor_opt.solve).lower(initial_guess).compile()
    #
    # # Run the compiled solve() method.
    # solution = solve_method(initial_guess)
    # print(solution.shape)


if __name__ == '__main__':
    main()
