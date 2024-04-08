from functools import partial
import time

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import torch


@torch.no_grad()
def solve_dual_batched(dJ, dG, dH, iters=100, eps=1e-3, tol=1e-4):
    """
        Solve dual min || dJ + \mu^T dH + \lambda^T dG || s.t. \mu >= 0

        For batched inputs

        Uses projected gradient descent with line search

    """
    B = dJ.shape[0]
    dC = torch.cat((dG, dH), dim=1)
    _mu = torch.zeros(B, dH.shape[1], 1).to(dH)
    _lambda = torch.zeros(B, dG.shape[1], 1).to(dG)
    Ab = (2 * dC @ dJ)
    AtA = dC @ dC.transpose(1, 2)

    def objective(dJ, dG, dH, _lambda, _mu):
        return torch.linalg.norm(dJ + dG.transpose(1, 2) @ _lambda + dH.transpose(1, 2) @ _mu, dim=1)

    obj = objective(dJ, dG, dH, _lambda, _mu)
    x = torch.cat((_lambda, _mu), dim=1)

    for iter in range(iters):
        # Do update
        _step = Ab + AtA @ x
        new_x = x - eps * _step
        torch.clamp_(new_x[:, _lambda.shape[1]:], min=0)

        # Backtracking line search
        new_obj = objective(dJ, dG, dH, new_x[:, :_lambda.shape[1]], new_x[:, _lambda.shape[1]:])
        while torch.any(new_obj > obj):
            print(iter)
            _step = torch.where(new_obj.unsqueeze(-1) > obj.unsqueeze(-1), _step * 0.1, torch.zeros_like(_step))
            new_x = x - eps * _step
            torch.clamp_(new_x[:, _lambda.shape[1]:], min=0)
            new_obj = objective(dJ, dG, dH, new_x[:, :_lambda.shape[1]], new_x[:, _lambda.shape[1]:])

        # check convergence
        diff = torch.linalg.norm(new_obj - obj, dim=1)
        if torch.all(diff < tol):
            break

        # for next time
        obj = new_obj
        x = new_x
    _lambda, _mu = x[:, _lambda.shape[1]], x[:, _lambda.shape[1]:]
    return _lambda, _mu


class JaxDualSolve:
    """
    """

    def __init__(self, cost_grad: jnp.array, equality_grad: jnp.array, inequality_grad: jnp.array,
                 max_iterations: int = 100, step_scale: jnp.float32 = 1e-3, tolerance: jnp.float32 = 1e-4):
        """
        """
        self._cost_grad = cost_grad
        self._equality_grad = equality_grad
        self._inequality_grad = inequality_grad
        self._max_iterations = max_iterations
        self._step_scale = step_scale
        self._tolerance = tolerance

        min_equality_multiplier = -jnp.inf * jnp.ones(self._equality_grad.shape[0])
        min_inequality_multiplier = jnp.zeros(self._inequality_grad.shape[0])
        max_equality_multiplier = jnp.inf * jnp.ones(self._equality_grad.shape[0])
        max_inequality_multiplier = jnp.inf * jnp.ones(self._inequality_grad.shape[0])
        self._min_decision_variable = jnp.concatenate((min_equality_multiplier, min_inequality_multiplier))
        self._max_decision_variable = jnp.concatenate((max_equality_multiplier, max_inequality_multiplier))

    def __hash__(self):
        """
        """
        # return hash((self._cost_grad, self._equality_grad, self._inequality_grad, self._max_iterations,
        #              self._step_scale, self._tolerance))
        return hash((self._max_iterations, self._step_scale, self._tolerance))

    def __eq__(self, other):
        """
        """
        return (self._cost_grad == other._cost_grad and
                self._equality_grad == other._equality_grad and
                self._inequality_grad == other._inequality_grad and
                self._max_iterations == other._max_iterations and
                self._step_scale == other._step_scale and
                self._tolerance == other._tolerance)

    @partial(jax.jit, static_argnums=0)
    def solve(self) -> jnp.array:
        """

        Returns:
            The optimal value of the decision variable.
        """
        objective_grad = jax.grad(self.objective)

        # Initialize decision variable with zero vector of appropriate dimension.
        equality_multiplier = jnp.zeros(self._equality_grad.shape[0])
        inequality_multiplier = jnp.zeros(self._inequality_grad.shape[0])
        decision_variable = jnp.concatenate((equality_multiplier, inequality_multiplier))

        # Set carry variables
        iteration = 0
        old_decision_variable = decision_variable
        _, decision_variable, _ = jax.lax.while_loop(self.not_within_tolerance_or_at_iteration_limit, self.solver_step,
                                                     (iteration, decision_variable, old_decision_variable))
        return decision_variable

    def not_within_tolerance_or_at_iteration_limit(self, variables) -> bool:
        """
        """
        iteration, decision_variable, old_decision_variable = variables

        distance = jnp.linalg.norm(self.objective(decision_variable) - self.objective(old_decision_variable))
        return (iteration < self._max_iterations) & (self._tolerance < distance)

    def solver_step(self, variables) -> jnp.array:
        """
        """
        iteration, decision_variable, old_decision_variable = variables

        # Update via projected gradient descent.
        old_decision_variable = decision_variable
        step = self._step_scale * self.objective_grad(decision_variable)
        decision_variable = jnp.clip(decision_variable - step, self._min_decision_variable, self._max_decision_variable)

        # Perform backtracking to avoid overstepping.
        backtracking_variables = (self._step_scale, decision_variable, old_decision_variable)
        step_scale, decision_variable, old_decision_variable = jax.lax.while_loop(self.new_worse_than_old,
                                                                                  self.backtracking_step,
                                                                                  backtracking_variables)

        # Update iteration and return carry variables.
        iteration += 1
        return iteration, decision_variable, old_decision_variable

    def new_worse_than_old(self, backtracking_variables) -> bool:
        """
        """
        step_scale, decision_variable, old_decision_variable = backtracking_variables

        return self.objective(decision_variable) > self.objective(old_decision_variable)

    def backtracking_step(self, backtracking_variables):
        """
        """
        step_scale, decision_variable, old_decision_variable = backtracking_variables

        decision_variable = old_decision_variable - step_scale * 0.1 * self.objective_grad(old_decision_variable)
        return step_scale, decision_variable, old_decision_variable

    def objective(self, decision_variable: jnp.array) -> jnp.float32:
        """
        """
        return jnp.linalg.norm(self._cost_grad +
                               self._equality_grad.T @ decision_variable[:self._equality_grad.shape[0]] +
                               self._inequality_grad.T @ decision_variable[self._equality_grad.shape[0]:])

    def objective_grad(self, decision_variable: jnp.array) -> jnp.array:
        """
        """
        return jax.grad(self.objective)(decision_variable)


def loop_body(a: jnp.float32) -> jnp.float32:
    b = a/2
    return b


def loop_condition(a: jnp.float32) -> bool:
    return a > 1


def loop_test() -> jnp.float32:
    a: jnp.float32 = 64
    a = jax.lax.while_loop(loop_condition, loop_body, a)
    return a


if __name__ == "__main__":
    device = 'cuda:0'
    dtype = torch.float32

    # Test the batched version
    eps = 1e-3
    tol = 1e-6
    max_iter = 1000
    xdim = 180
    nineq = 400
    neq = 200
    B = 16
    dG = torch.randn(B, neq, xdim, device=device, dtype=dtype)
    dH = torch.randn(B, nineq, xdim, device=device, dtype=dtype)
    dJ = torch.rand(B, xdim, 1, device=device, dtype=dtype)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        mu, lambda_ = solve_dual_batched(dJ, dG, dH, iters=max_iter, eps=eps, tol=tol)
    torch.cuda.synchronize()
    end = time.time()
    print((end - start) / 10)

    dG = jnp.array(np.random.randn(neq, xdim))
    dH = jnp.array(np.random.randn(nineq, xdim))
    dJ = jnp.array(np.random.randn(xdim))

    # jax_solver = JaxDualSolve(dJ, dG, dH, max_iter, eps, tol)
    # lambda_mu = jax_solver.solve()
    # print(lambda_mu)

    # a: jnp.float32 = 64
    # b: jnp.float32 = 10
    # print(loop_test(a))
    a = loop_test()
    b = loop_test()
    print(b)
