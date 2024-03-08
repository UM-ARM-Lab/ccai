import jax
import jax.numpy as jnp


def cost(trajectories: jnp.array, goal: jnp.array) -> jnp.array:
    """

    Args:
        trajectories: Set of trajectories
        goal:

    Returns:

    """
    x = trajectories[:, :12]
    u = trajectories[:, 12:]
    T = x.shape[0]
    Q = torch.eye(12, device=trajectories.device)
    Q[5, 5] = 1e-2
    Q[2, 2] = 0.1
    Q[3:, 3:] *= 0.5
    Q *= 5
    P = Q
    R = 16 * torch.eye(4, device=trajectories.device)
    R[0, 0] = 1

    P[5, 5] = 1e-2
    d2goal = x - goal.reshape(-1, 12)

    running_state_cost = d2goal.reshape(-1, 1, 12) @ Q.reshape(1, 12, 12) @ d2goal.reshape(-1, 12, 1)
    running_control_cost = u.reshape(-1, 1, 4) @ R.reshape(1, 4, 4) @ u.reshape(-1, 4, 1)
    terminal_state_cost = d2goal[-1].reshape(1, 12) @ P @ d2goal[-1].reshape(12, 1)

    cost = torch.sum(running_control_cost + running_state_cost, dim=0) + terminal_state_cost

    # Compute cost grad analytically
    grad_running_state_cost = Q.reshape(1, 12, 12) @ d2goal.reshape(-1, 12, 1)
    grad_running_control_cost = R.reshape(1, 4, 4) @ u.reshape(-1, 4, 1)
    grad_terminal_cost = P @ d2goal[-1].reshape(12, 1)  # only on terminal
    grad_terminal_cost = torch.cat((torch.zeros(T - 1, 12, 1, device=trajectories.device),
                                    grad_terminal_cost.unsqueeze(0)), dim=0)

    grad_cost = torch.cat((grad_running_state_cost + grad_terminal_cost, grad_running_control_cost), dim=1)
    grad_cost = grad_cost.reshape(T * 16)

    # compute hessian of cost analytically
    running_state_hess = Q.reshape(1, 12, 12).repeat(T, 1, 1)
    running_control_hess = R.reshape(1, 4, 4).repeat(T, 1, 1)
    terminal_hess = torch.cat((torch.zeros(T - 1, 12, 12, device=trajectories.device), P.unsqueeze(0)), dim=0)

    state_hess = running_state_hess + terminal_hess
    hess_cost = torch.cat((
        torch.cat((state_hess, torch.zeros(T, 4, 12, device=trajectories.device)), dim=1),
        torch.cat((torch.zeros(T, 12, 4, device=trajectories.device), running_control_hess), dim=1)
    ), dim=2)  # will be N x T x 16 x 16

    # now we need to refactor hess to be (N x Td x Td)
    hess_cost = torch.diag_embed(hess_cost.permute(1, 2, 0)).permute(2, 0, 3, 1).reshape(T * 16, T * 16)

    return cost.flatten(), grad_cost, hess_cost


def quadrotor_dynamics(states: jnp.array, control_inputs: jnp.array) -> jnp.array:
    """Compute the batched dynamics function for the quadrotor according to equation (52).

    Args:
        states: Batched states of the quadrotor, shape (M, dx).
        control_inputs: Batched control inputs of the quadrotor (M, du).

    Returns:
        Batched states of the quadrotor forward one timestep, shape (M, dx).
    """
    # TODO: Remove hard-coding of dt.
    dt = 0.1

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

    new_xdot = x_dot + x_ddot * dt
    new_ydot = y_dot + y_ddot * dt
    new_zdot = z_dot + z_ddot * dt
    new_p = p + p_dot * dt
    new_q = q + q_dot * dt
    new_r = r + r_dot * dt

    psi_dot = new_q * sphi / ctheta + new_r * cphi / ctheta
    theta_dot = new_q * cphi - new_r * sphi
    phi_dot = new_p + new_q * sphi * ttheta + new_r * cphi * ttheta

    new_phi = phi + phi_dot * dt
    new_theta = theta + theta_dot * dt
    new_psi = psi + psi_dot * dt
    new_x = x + new_xdot * dt
    new_y = y + new_ydot + dt
    new_z = z + new_zdot + dt

    return jnp.concatenate((new_x, new_y, new_z, new_phi, new_theta, new_psi, new_xdot, new_ydot, new_zdot, new_p,
                            new_q, new_r), axis=-1)


class GPHeightConstraint:
    """
    """
    def __init__(self):
        """

        Args:
        Returns:
        """
        pass

    def __call__(self):
        """

        Args:
        Returns:
        """
        pass


def main() -> None:
    """Set up, compile, and solve a JaxCSVTOpt for the quadrotor problem with no obstacles.
    """


if __name__ == '__main__':
    main()
