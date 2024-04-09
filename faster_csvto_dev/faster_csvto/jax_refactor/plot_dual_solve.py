import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    with open('output/dual_solve_jax_objective_history.npy', 'rb') as f:
        jax_objective_history = np.load(f)
    print(jax_objective_history.shape)

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(range(jax_objective_history.shape[0]), jax_objective_history)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(range(jax_objective_history.shape[0]), jax_objective_history)
    ax1.set_ylabel('Jax cost')

    with open('output/dual_solve_torch_objective_history.npy', 'rb') as f:
        torch_objective_history = np.load(f)
    print(torch_objective_history.shape)

    ax2.plot(range(torch_objective_history.shape[0]), torch_objective_history)
    ax2.set_ylabel('Torch cost')
    ax2.set_xlabel('iterations')
    plt.show()
