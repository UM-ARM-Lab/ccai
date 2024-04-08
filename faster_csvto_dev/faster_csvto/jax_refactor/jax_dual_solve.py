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
            _step = torch.where(new_obj.unsqueeze(-1) > obj.unsqueeze(-1), _step * 0.1, torch.zeros_like(_step))
            new_x = x - eps * _step
            torch.clamp_(new_x[:, _lambda.shape[1]:], min=0)
            new_obj = objective(dJ, dG, dH, new_x[:, :_lambda.shape[1]], new_x[:, _lambda.shape[1]:])

        # check convergence
        diff = torch.abs(new_obj - obj)
        if torch.max(diff) < tol:
            break

        # for next time
        obj = new_obj
        x = new_x
        print(iter)
    _lambda, _mu = x[:, :_lambda.shape[1]], x[:, _lambda.shape[1]:]
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

        # dC = jnp.concatenate((equality_grad, inequality_grad))

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
        objectives = jnp.zeros(self._max_iterations)

        # Initialize decision variable with zero vector of appropriate dimension.
        equality_multiplier = jnp.zeros(self._equality_grad.shape[0])
        inequality_multiplier = jnp.zeros(self._inequality_grad.shape[0])
        decision_variable = jnp.concatenate((equality_multiplier, inequality_multiplier))

        # Set carry variables
        iteration = 0
        old_decision_variable = decision_variable
        step = self._step_scale * self.objective_grad(decision_variable)
        decision_variable = jnp.clip(decision_variable - step, self._min_decision_variable, self._max_decision_variable)
        iteration, decision_variable, _ = jax.lax.while_loop(self.not_within_tolerance_or_at_iteration_limit,
                                                             self.solver_step, (iteration, decision_variable,
                                                                                old_decision_variable))
        return decision_variable, iteration

    def not_within_tolerance_or_at_iteration_limit(self, variables) -> bool:
        """
        """
        iteration, decision_variable, old_decision_variable = variables

        distance = jnp.abs(self.objective(decision_variable) - self.objective(old_decision_variable))
        return (self._tolerance < distance) & (iteration < self._max_iterations)

    def solver_step(self, variables) -> jnp.array:
        """
        """
        iteration, decision_variable, old_decision_variable = variables

        # Update via projected gradient descent.
        old_decision_variable = decision_variable
        step = self._step_scale * self.objective_grad(decision_variable)
        decision_variable = jnp.clip(decision_variable - step, self._min_decision_variable, self._max_decision_variable)

        # Perform backtracking to avoid overstepping.
        back_iteration = 0
        backtracking_variables = (back_iteration, self._step_scale, decision_variable, old_decision_variable)
        back_iteration, step_scale, decision_variable, old_decision_variable = jax.lax.while_loop(self.new_worse_than_old,
                                                                                  self.backtracking_step,
                                                                                  backtracking_variables)

        # Update iteration and return carry variables.
        iteration += 1
        return iteration, decision_variable, old_decision_variable

    def new_worse_than_old(self, backtracking_variables) -> bool:
        """
        """
        back_iteration, step_scale, decision_variable, old_decision_variable = backtracking_variables

        return (self.objective(decision_variable) > self.objective(old_decision_variable)) & (back_iteration < 5)

    def backtracking_step(self, backtracking_variables):
        """
        """
        back_iteration, step_scale, decision_variable, old_decision_variable = backtracking_variables
        back_iteration += 1

        decision_variable = old_decision_variable - step_scale * 0.1 * self.objective_grad(old_decision_variable)
        decision_variable = jnp.clip(decision_variable, self._min_decision_variable, self._max_decision_variable)
        return back_iteration, step_scale, decision_variable, old_decision_variable

    def objective(self, decision_variable: jnp.array) -> jnp.float32:
        """
        """
        return jnp.linalg.norm(self._cost_grad +
                               self._equality_grad.T @ decision_variable[:self._equality_grad.shape[0]] +
                               self._inequality_grad.T @ decision_variable[self._equality_grad.shape[0]:])

    def objective_grad(self, decision_variable: jnp.array) -> jnp.array:
        """
        """
        gradient = jax.grad(self.objective)(decision_variable)
        return gradient


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

    num_trials = 10

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_trials):
        lambda_, mu = solve_dual_batched(dJ, dG, dH, iters=max_iter, eps=eps, tol=tol)
    torch.cuda.synchronize()
    end = time.time()
    print('time', (end - start) / num_trials)
    print('objective', torch.linalg.norm(dJ + dG.transpose(1, 2) @ lambda_ +
                                         dH.transpose(1, 2) @ mu, dim=1))
    print(torch.min(mu))

    dG_jnp = jnp.array(dG[0].cpu().numpy())
    dH_jnp = jnp.array(dH[0].cpu().numpy())
    dJ_jnp = jnp.array(dJ[0].cpu().numpy())

    dGs = [dG_jnp for _ in range(B)]
    dHs = [dH_jnp for _ in range(B)]
    dJs = [dJ_jnp for _ in range(B)]

    jax_solver = JaxDualSolve(dG_jnp, dH_jnp, dJ_jnp, max_iter, eps, tol)
    lambda_mu, iteration = jax_solver.solve()

    start = time.time()
    for _ in range(num_trials):
        lambda_mu, iteration = jax_solver.solve()
    end = time.time()
    print('time', (end - start) / num_trials)
    print('solving iterations', iteration)
    print('objective', jax_solver.objective(lambda_mu))
    print(jnp.min(lambda_mu[neq:]))
    print(jnp.max(lambda_mu[neq:]))
    print(jnp.min(lambda_mu))
    print(jnp.max(lambda_mu))
    # print(lambda_mu)
    print(torch.max(dJ))
