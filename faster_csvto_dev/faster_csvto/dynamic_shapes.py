import torch

"""
The situation:

user supplies function
g: dq -> dg

which gets evaluated in a function that selects parts of its output according to a mask of dimension dg
g_mask: dq x dg -> 0 ... dg

which gets composed into a larger function
hhat: dq -> dh + dx * T + 0 ... dg

which needs to be evaluated and differentiated
hhat: dq -> dh + dx * T + 0 ... dg
dhhat: dq -> dh + dx * T + 0 ... dg, dq (This will already be evaluated in order to set up solving for the mask)
ddhhat: dq -> dh + dx * T + 0 ... dg, dq, dq

and all of these operations need to be batched over a set of particles
"""


def h(trajectory) -> torch.Tensor:
    ones_constraint = torch.ones(trajectory.shape)
    difference_from_ones_constraint = trajectory - ones_constraint
    return difference_from_ones_constraint


def f(trajectory) -> torch.Tensor:
    dynamics_constraint = trajectory * 2
    difference_from_dynamics_constraint = trajectory - dynamics_constraint
    return difference_from_dynamics_constraint


def g(trajectory) -> torch.Tensor:
    zeros_constraint = torch.zeros(trajectory.shape)
    difference_from_zeros_constraint = trajectory - zeros_constraint
    return difference_from_zeros_constraint


def g_mask(trajectory, active_mask) -> torch.Tensor:
    return trajectory[torch.where(trajectory * active_mask != 0)]


def hhat(trajectory, active_mask) -> torch.Tensor:
    combined_constraint = torch.concat((g_mask(trajectory, active_mask),
                                        h(trajectory),
                                        g(trajectory)))
    return combined_constraint


if __name__ == "__main__":
    # Evaluate the gradient of the equality constraint and unmasked inequality constraint and cost.

    # Use these gradients to solve the sub problem and get the mask.

    # Evaluate the combined constraint.

    # Get the gradient of the combined constraint from the initial gradients.

    # Evaluate the hessian of the combined constraint.

    input = torch.tensor([1, 4, 3, 5, 6, 7])
    mask = torch.tensor([0, 1, 0, 1, 1, 0])
    restricted_input = g_mask(input, mask)
    print(restricted_input)
