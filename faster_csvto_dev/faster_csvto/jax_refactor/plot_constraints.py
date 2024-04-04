import numpy as np
from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp

def plot_constraint_metrics(constraints, dh, dg, dx, number):
    """
    constraints is shape (K = iterations, N = particles, dh + dg + T * dx).
    """
    average_equality_violation = jnp.mean(constraints[:, :, :dh], axis=-1)
    average_slack_violation = jnp.mean(constraints[:, :, dh:dh+dg], axis=-1)
    average_dynamics_violation = jnp.mean(constraints[:, :, dh+dg:-dx], axis=-1)
    average_start_violation = jnp.mean(constraints[:, :, -dx:], axis=-1)

    K, N, _ = constraints.shape
    iteration_number = np.arange(0, K)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))
    ax1.plot(iteration_number, average_equality_violation[:, :], linewidth=2.0)
    ax1.set_xlabel('average equality violation')

    ax2.plot(iteration_number, average_slack_violation[:, :], linewidth=2.0)
    ax2.set_xlabel('average inequality + slack violation')

    ax3.plot(iteration_number, average_dynamics_violation[:, :], linewidth=2.0)
    ax3.set_xlabel('average dynamics violation')

    ax4.plot(iteration_number, average_start_violation[:, :], linewidth=2.0)
    ax4.set_xlabel('average start violation')

    plt.savefig(f'output/constraint_{number}.png')


if __name__ == "__main__":
    # with open('output/warmup_constraints.npy', 'rb') as f:
    #     warmup_constraints = np.load(f)
    # plot_constraint_metrics(warmup_constraints, 11, 11, 12, 1001)

    with open('output/all_online_constraints.npy', 'rb') as f:
        all_online_constraints = np.load(f)
    for i in range(all_online_constraints.shape[0]):
        plot_constraint_metrics(all_online_constraints[i], 11, 11, 12, i)
