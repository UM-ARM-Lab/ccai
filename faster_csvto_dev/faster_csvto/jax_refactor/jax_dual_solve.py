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

    history = torch.zeros(iters)

    iteration = 0
    for iter in range(iters):
        iteration = iter

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

        history[iter] = obj[0]
    _lambda, _mu = x[:, :_lambda.shape[1]], x[:, _lambda.shape[1]:]
    return _lambda, _mu, iteration, history


class JaxDualSolve:
    """
    """

    def __init__(self, max_iterations: int = 100, step_scale: jnp.float32 = 1e-3, tolerance: jnp.float32 = 1e-4):
        """
        """
        self._max_iterations = max_iterations
        self._step_scale = step_scale
        self._tolerance = tolerance

    def __hash__(self):
        """
        """
        return hash((self._max_iterations, self._step_scale, self._tolerance))

    def __eq__(self, other):
        """
        """
        return (self._max_iterations == other._max_iterations and
                self._step_scale == other._step_scale and
                self._tolerance == other._tolerance)

    @partial(jax.jit, static_argnums=0)
    def solve(self, initial_guess: jnp.array, cost_grad: jnp.array, equality_grad: jnp.array,
              inequality_grad: jnp.array) -> jnp.array:
        """

        Returns:
            The optimal value of the decision variable.
        """
        # Initialize decision variable.
        decision_variable = initial_guess

        # Get variables for clamping the decision variable.
        min_equality_multiplier = -jnp.inf * jnp.ones(equality_grad.shape[0])
        min_inequality_multiplier = jnp.zeros(inequality_grad.shape[0])
        max_equality_multiplier = jnp.inf * jnp.ones(equality_grad.shape[0])
        max_inequality_multiplier = jnp.inf * jnp.ones(inequality_grad.shape[0])
        min_decision_variable = jnp.concatenate((min_equality_multiplier, min_inequality_multiplier))
        max_decision_variable = jnp.concatenate((max_equality_multiplier, max_inequality_multiplier))

        # Set carry variables.
        iteration = 0
        old_decision_variable = decision_variable
        step = self._step_scale * self.objective_grad(decision_variable, cost_grad, equality_grad, inequality_grad)
        decision_variable = jnp.clip(decision_variable - step, min_decision_variable, max_decision_variable)

        # Update history for analysis.
        history = jnp.zeros((self._max_iterations, decision_variable.shape[0]))
        history = history.at[0, :].set(decision_variable)

        iteration, decision_variable, _, history, _, _, _ = jax.lax.while_loop(self.not_within_tolerance_or_at_iteration_limit,
                                                                               self.solver_step,
                                                                               (iteration,
                                                                                decision_variable,
                                                                                old_decision_variable,
                                                                                history,
                                                                                cost_grad,
                                                                                equality_grad,
                                                                                inequality_grad))
        equality_multiplier = decision_variable[:equality_grad.shape[0]]
        inequality_multiplier = decision_variable[equality_grad.shape[0]:]
        return equality_multiplier, inequality_multiplier, iteration, history

    def not_within_tolerance_or_at_iteration_limit(self, variables) -> bool:
        """
        """
        iteration, decision_variable, old_decision_variable, history, cost_grad, equality_grad, inequality_grad = variables

        distance = jnp.abs(self.objective(decision_variable, cost_grad, equality_grad, inequality_grad) -
                           self.objective(old_decision_variable, cost_grad, equality_grad, inequality_grad))
        return (self._tolerance < distance) & (iteration < self._max_iterations)

    def solver_step(self, variables) -> jnp.array:
        """
        """
        iteration, decision_variable, old_decision_variable, history, cost_grad, equality_grad, inequality_grad = variables

        # Get variables for clamping the decision variable.
        min_equality_multiplier = -jnp.inf * jnp.ones(equality_grad.shape[0])
        min_inequality_multiplier = jnp.zeros(inequality_grad.shape[0])
        max_equality_multiplier = jnp.inf * jnp.ones(equality_grad.shape[0])
        max_inequality_multiplier = jnp.inf * jnp.ones(inequality_grad.shape[0])
        min_decision_variable = jnp.concatenate((min_equality_multiplier, min_inequality_multiplier))
        max_decision_variable = jnp.concatenate((max_equality_multiplier, max_inequality_multiplier))

        # Update via projected gradient descent.
        old_decision_variable = decision_variable
        step = self._step_scale * self.objective_grad(decision_variable, cost_grad, equality_grad, inequality_grad)
        decision_variable = jnp.clip(decision_variable - step, min_decision_variable, max_decision_variable)

        # Perform backtracking to avoid overstepping.
        backtracking_variables = (self._step_scale, decision_variable, old_decision_variable, cost_grad, equality_grad,
                                  inequality_grad)
        step_scale, decision_variable, old_decision_variable, _, _, _ = jax.lax.while_loop(self.new_worse_than_old,
                                                                                  self.backtracking_step,
                                                                                  backtracking_variables)

        # Update history for analysis.
        history = history.at[iteration, :].set(decision_variable)

        # Update iteration and return carry variables.
        iteration += 1
        return iteration, decision_variable, old_decision_variable, history, cost_grad, equality_grad, inequality_grad

    def new_worse_than_old(self, backtracking_variables) -> bool:
        """
        """
        step_scale, decision_variable, old_decision_variable, cost_grad, equality_grad, inequality_grad = backtracking_variables

        return (self.objective(decision_variable, cost_grad, equality_grad, inequality_grad) >
                self.objective(old_decision_variable, cost_grad, equality_grad, inequality_grad))

    def backtracking_step(self, backtracking_variables):
        """
        """
        step_scale, decision_variable, old_decision_variable, cost_grad, equality_grad, inequality_grad = backtracking_variables

        # Get variables for clamping the decision variable.
        min_equality_multiplier = -jnp.inf * jnp.ones(equality_grad.shape[0])
        min_inequality_multiplier = jnp.zeros(inequality_grad.shape[0])
        max_equality_multiplier = jnp.inf * jnp.ones(equality_grad.shape[0])
        max_inequality_multiplier = jnp.inf * jnp.ones(inequality_grad.shape[0])
        min_decision_variable = jnp.concatenate((min_equality_multiplier, min_inequality_multiplier))
        max_decision_variable = jnp.concatenate((max_equality_multiplier, max_inequality_multiplier))

        decision_variable = old_decision_variable - step_scale * 0.1 * self.objective_grad(old_decision_variable,
                                                                                           cost_grad,
                                                                                           equality_grad,
                                                                                           inequality_grad)
        decision_variable = jnp.clip(decision_variable, min_decision_variable, max_decision_variable)
        return step_scale, decision_variable, old_decision_variable, cost_grad, equality_grad, inequality_grad

    @staticmethod
    def objective(decision_variable: jnp.array, cost_grad: jnp.array, equality_grad: jnp.array,
                  inequality_grad: jnp.array) -> jnp.float32:
        """
        """
        return jnp.linalg.norm(cost_grad +
                               equality_grad.T @ decision_variable[:equality_grad.shape[0]] +
                               inequality_grad.T @ decision_variable[equality_grad.shape[0]:])

    def objective_grad(self, decision_variable: jnp.array, cost_grad: jnp.array, equality_grad: jnp.array,
                       inequality_grad: jnp.array) -> jnp.array:
        """
        """
        gradient = jax.grad(self.objective)(decision_variable, cost_grad, equality_grad, inequality_grad)
        return gradient


if __name__ == "__main__":
    device = 'cuda:0'
    dtype = torch.float32

    # Test the batched PyTorch implementation (Tom).
    eps = 1e-3
    tol = 1e-6
    max_iter = 100
    xdim = 180
    nineq = 400
    neq = 200
    B = 16
    dG = torch.randn(B, neq, xdim, device=device, dtype=dtype)
    dH = torch.randn(B, nineq, xdim, device=device, dtype=dtype)
    dJ = torch.rand(B, xdim, 1, device=device, dtype=dtype)

    num_trials = 10

    iteration = 0
    history = None
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_trials):
        lambda_, mu, iteration, history = solve_dual_batched(dJ, dG, dH, iters=max_iter, eps=eps, tol=tol)
    torch.cuda.synchronize()
    end = time.time()
    print('time', (end - start) / num_trials)
    print('solving iterations', iteration)
    print('objective (first item)', torch.linalg.norm(dJ + dG.transpose(1, 2) @ lambda_ +
                                         dH.transpose(1, 2) @ mu, dim=1)[0])

    history = history[:iteration].cpu().numpy()
    with open('output/dual_solve_torch_objective_history.npy', 'wb') as f:
        np.save(f, history)

    # Test the Jax implementation.
    dG_jnp = jnp.array(dG[0].cpu().numpy())
    dH_jnp = jnp.array(dH[0].cpu().numpy())
    dJ_jnp = jnp.array(dJ[0].cpu().numpy())
    dGs = [dG_jnp for _ in range(B)]
    dHs = [dH_jnp for _ in range(B)]
    dJs = [dJ_jnp for _ in range(B)]

    # Set the initial guess.
    initial_guess = jnp.zeros((nineq + neq,))

    jax_solver = JaxDualSolve(max_iter, eps, tol)
    equality_multiplier, inequality_multiplier, iteration, history = jax_solver.solve(initial_guess, dJ_jnp, dG_jnp, dH_jnp)
    start = time.time()
    for _ in range(num_trials):
        equality_multiplier, inequality_multiplier, iteration, history = jax_solver.solve(initial_guess, dJ_jnp, dG_jnp, dH_jnp)
    end = time.time()

    lambda_mu = jnp.concatenate((equality_multiplier, inequality_multiplier))

    print('time', (end - start) / num_trials)
    print('solving iterations', iteration)
    print('objective', jax_solver.objective(lambda_mu, dJ_jnp, dG_jnp, dH_jnp))

    history = history[:iteration, :]
    batched_objective = jax.vmap(jax_solver.objective, (0, None, None, None), 0)
    objective_history = batched_objective(history, dJ_jnp, dG_jnp, dH_jnp)

    history_np = np.array(history)
    objective_history_np = np.array(objective_history)
    with open('output/dual_solve_jax_objective_history.npy', 'wb') as f:
        np.save(f, objective_history_np)

    # Determining which constraints were active.
    mu = mu[0, :].cpu().numpy()
    number_active = np.where(mu > 1e-8, 1, 0).sum()
    print('number active torch', number_active)

    mu_jax = np.array(lambda_mu[neq:])
    number_active = np.where(mu_jax > 1e-8, 1, 0).sum()
    print('number active jax', number_active)
