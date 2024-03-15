from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp


# Using this file to sketch out the minimal interface for getting the quad rotor problem to work in Jax. I'm not
# worrying about making this an extensible framework for other optimizers right now. I'm also trying to make the
# notation more consistent with the paper for clarity.


class JaxCSVTOProblem:
    """Class storing the optimization problem formulation to be used by the JaxCSVTOpt optimizer.
    """

    # TODO: It would be ideal to either deduce the dimensionality automatically, for now jax.experimental.checkify() is
    #   used for runtime assertion in the jit compiled solve() function.
    def __init__(self,
                 c: Callable[[jnp.array], jnp.array],
                 h: Callable[[jnp.array], jnp.array],
                 g: Optional[Callable[[jnp.array], jnp.array]],
                 f: Callable[[jnp.array, jnp.array], jnp.array],
                 u_bounds: Tuple[jnp.array, jnp.array], x_bounds: Tuple[jnp.array, jnp.array],
                 dx: int, du: int, dh: int, dg: int) -> None:
        """Set the functions that define the optimization problem along with parameters that determine the size of the
                involved arrays.

        NOTE: In the shape descriptions below, N is the number of particles and T is the time horizon, both of which are
        determined by the optimization parameters when this problem is used to initialize an optimization.

        Args:
            c: The batched cost function, mapping shapes (N, (dx + du) * T) -> (N, 1).
            h: The batched equality constraint function, mapping shapes (N, (dx + du) * T) -> (N, dh).
            g: The batched inequality constraint function, mapping shapes (N, (dx + du) * T) -> (N, dg).
            f: The batched dynamics function, mapping shapes (N, dx), (N, du) -> (N, dx).
            u_bounds: The (min, max) bounds for control inputs u.
            x_bounds: The (min, max) bounds for states x.
            dx: The dimension of the state.
            du: The dimension of the control input.
            dh: The dimension mapped to by the equality constraint. (The number of equality constraints.)
            dg: The dimension mapped to by the inequality constraint. (The number of inequality constraints.)
        """
        self.c = c
        self.h = h
        self.g = g
        self.f = f
        self.u_bounds = u_bounds
        self.x_bounds = x_bounds
        self.dx = dx
        self.du = du
        self.dh = dh
        self.dg = dg


class JaxCSVTOParams:
    """Class storing the algorithmic parameters for JaxCSVTOpt.
    """

    def __init__(self, k: Callable[[jnp.array, jnp.array], jnp.array], N: int, T: int, K: int, anneal: bool,
                 alpha_J: jnp.float32, alpha_C: jnp.float32, step_scale: jnp.float32,
                 penalty_weight: jnp.float32) -> None:
        """Set the parameters for the optimization problem.

        Args:
            k: The kernel function, mapping shapes (N, T, dx + du) and (N, T, dx + du) -> (N, N).
            N: The number of trajectories to represent the distribution with.
            T: The time horizon, which is the number of discrete states in each trajectory.
            K: The number of gradient steps to take before returning from the function.
            anneal: True if the tangent step should use the annealing formula, false for the default formula.
            alpha_J: The factor to scale the tangent step.
            alpha_C: The factor to scale the constraint step.
            step_scale: The factor to scale the sum of the tangent and constraint steps together.
            penalty_weight: The weight placed on the 1-norm of constraint violation within the penalty function.
        """
        self.k = k
        self.N = N
        self.T = T
        self.K = K
        self.anneal = anneal
        self.alpha_J = alpha_J
        self.alpha_C = alpha_C
        self.step_scale = step_scale
        self.penalty_weight = penalty_weight


class JaxCSVTOConfig:
    """Class storing configuration for the JaxCSVTOpt related to visualization, profiling, or debugging.
    """
    def __init__(self) -> None:
        """

        Args:
        """
        pass


class JaxCSVTOpt:
    """
    """

    def __init__(self, problem: JaxCSVTOProblem, params: JaxCSVTOParams, config: JaxCSVTOConfig = None) -> None:
        """Store the optimization problem and the configuration of the solver statically so that the solve() routine is
        a pure function and can be compiled ahead of time by compile().

        Args:
            problem: The optimization problem to solve.
            params: The parameters to use for the CSVTO algorithm during the execution of solve().
            config: An optional struct for storing visualization, profiling, and debugging output options.
        """
        # Store problem functions and bounds.
        self.c: Callable[[jnp.array], jnp.array] = problem.c
        self.h: Callable[[jnp.array], jnp.array] = problem.h
        self.g: Optional[Callable[[jnp.array], jnp.array]] = problem.g
        self.f: Callable[[jnp.array, jnp.array], jnp.array] = problem.f
        self.k: Callable[[jnp.array, jnp.array], jnp.array] = params.k
        self.u_bounds: Tuple[jnp.array, jnp.array] = problem.u_bounds
        self.x_bounds: Tuple[jnp.array, jnp.array] = problem.x_bounds

        # Store dimensionality constants.
        self.dx: int = problem.dx
        self.du: int = problem.du
        self.N: int = params.N
        self.T: int = params.T
        self.dh: int = problem.dh
        self.dg: int = problem.dg

        # Store optimizer parameters.
        self.K: int = params.K
        self.anneal: bool = params.anneal
        self.alpha_J: jnp.float32 = params.alpha_J
        self.alpha_C: jnp.float32 = params.alpha_C
        self.step_scale: jnp.float32 = params.step_scale
        self.penalty_weight: jnp.float32 = params.penalty_weight

        # Store configuration.
        self.config: JaxCSVTOConfig = config if config is not None else None

    def compile(self) -> None:
        """Compile the solve() routine ahead of time to maximize performance for iterative solve() calls on the same
        problem, as will be done in the typical MPC use case.

        NOTE: This will involve computation and subsequent compilation of the gradient and hessian routines for the
        functions specified in the Problem which require derivatives. JIT allows for calling this at runtime,
        potentially enabling behaviors that initialize different optimization problems online as different scenarios
        are encountered.
        """
        raise NotImplementedError

    def solve(self, x0: jnp.array) -> jnp.array:
        """Execute the CSVTO algorithm to solve for an optimal distribution of trajectories, given an initial guess.

        Args:
            x0: The initialization for the trajectory distribution, shape (N, (dx + du) * T).

        Returns:
            The optimal trajectory from the optimized distribution, shape ((dx + du) * T,).
        """
        assert x0.shape == (self.N, self._trajectory_dim())
        # TODO: Assert that the initial state for each trajectory is the same, or add initial state as a second
        #  parameter.

        # Calculate slack variable values via equation (33) and append to the trajectory according to equation (35).
        if self.g is not None:
            xuz: jnp.array = jnp.concatenate((x0, jnp.sqrt(-2*self.g(x0))), axis=1)
        else:
            xuz = x0

        # Loop over the gradient step iterations.
        xuz = jax.lax.fori_loop(1, self.K+1, self._solve_iteration, (xuz, xuz[0, :self.dx]), unroll=True)

        # Find the lowest cost trajectory in the optimized distribution, and return it without slack variables.
        min_cost_trajectory = xuz[:, :self._trajectory_dim()][self.c(xuz[:, :self._trajectory_dim()]).argmin()]
        assert min_cost_trajectory.shape == (self._trajectory_dim(),)
        return min_cost_trajectory

    def _solve_iteration(self, iteration: int, xuz_initial_state: Tuple[jnp.array, jnp.array]) -> jnp.array:
        """Compute a single retraction step for the optimizer and return the updated state.
        TODO: Fix Args.
        Args:
            iteration: The iteration number, in range [0, K).
            xuz: All N trajectories including slack variables stacked as rows, shape (N, (dx + du) * T + dg).
            initial_state: The state of the initial condition for the trajectories, shape (dx,).

        Returns:
            The value of xuz after taking the retraction step.
        """
        xuz, initial_state = xuz_initial_state
        assert xuz.shape == (self.N, self._slack_trajectory_dim())
        assert initial_state.shape == (self.dx,)

        # Set the annealing weight gamma if annealing is on.
        gamma: Optional[jnp.float32] = iteration / self.K if self.anneal else None

        # Evaluate the gradient of the cost function at the current state (without slack variables).
        c_grad = jax.jacfwd(self.c)(xuz[:, :self._trajectory_dim()]).mean(axis=1)

        # Evaluate the combined constraint function, its gradient, and its hessian at the current augmented state.
        constraint = self._combined_constraint(xuz, initial_state)
        constraint_grad = jax.jacrev(self._combined_constraint)(xuz, initial_state).mean(axis=2)
        constraint_hess = jax.jacfwd(jax.jacrev(self._combined_constraint))(xuz, initial_state).mean(axis=2).mean(axis=3)
        # print('constraint shape', constraint.shape)
        # print('constraint_grad shape', constraint_grad.shape)
        # print('constraint_hess shape', constraint_hess.shape)

        # Evaluate the kernel function and its gradient at the current state (without slack variables).
        k = self.k(xuz[:, :self._trajectory_dim()].reshape(self.N, self.T, self.dx + self.du),
                   xuz[:, :self._trajectory_dim()].reshape(self.N, self.T, self.dx + self.du))
        k_grad = jax.jacrev(self.k)(xuz[:, :self._trajectory_dim()].reshape(self.N, self.T, self.dx + self.du),
                                    xuz[:, :self._trajectory_dim()].reshape(self.N, self.T, self.dx + self.du)).mean(
            axis=2).reshape(self.N, self.N, self._trajectory_dim())

        # Compute the retraction steps.
        constraint_step = self._constraint_step(constraint, constraint_grad)
        tangent_step = self._tangent_step(c_grad, constraint_grad, constraint_hess, k, k_grad, gamma)

        # Combine the steps according to equation (26) and use them to update the current augmented state.
        xuz -= self.step_scale * (self.alpha_J * tangent_step + self.alpha_C * constraint_step)

        # Clamp the state in bounds and return.
        xuz = self._bound_projection(xuz)
        assert xuz.shape == (self.N, self._slack_trajectory_dim())
        return xuz, initial_state

    def _constraint_step(self, constraint: jnp.array, constraint_grad: jnp.array) -> jnp.array:
        """Compute the constraint step according to equation (39).

        Args:
            constraint: The combined constraint evaluated at the current state, shape (N, dh + dg + dx * T).
            constraint_grad: The gradient of the combined constraint evaluated at the current state, shape (N, dh + dg +
             dx * T).

        Returns:
            The constraint step, shape (N, (dx + du) * T + dg).
        """
        assert constraint.shape == (self.N, self.dh + self.dg + self.dx * self.T)
        assert constraint_grad.shape == (self.N, self.dh + self.dg + self.dx * self.T, self._slack_trajectory_dim())

        # Get inv(dC * dCT) term to be used below, shape (N, dg + dh, dg + dh).
        dCdCT_inv = jnp.linalg.inv(constraint_grad @ constraint_grad.transpose((0, 2, 1)))

        # Get constraint step from equation (39), shape (N, d * T).
        constraint_step = (constraint_grad.transpose((0, 2, 1)) @ dCdCT_inv @
                           jnp.expand_dims(constraint, axis=-1)).squeeze(-1)
        assert constraint_step.shape == (self.N, self._slack_trajectory_dim())
        return constraint_step

    def _tangent_step(self,
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
        assert c_grad.shape == (self.N, self._trajectory_dim())
        assert constraint_grad.shape == (self.N, self.dh + self.dg + self.dx * self.T, self._slack_trajectory_dim())
        assert constraint_hess.shape == (self.N, self.dh + self.dg + self.dx * self.T, self._slack_trajectory_dim(),
                                         self._slack_trajectory_dim())
        assert k.shape == (self.N, self.N)
        assert k_grad.shape == (self.N, self.N, self._slack_trajectory_dim())

        # Get inv(dC * dCT), shape (N, dg + dh, dg + dh).
        dCdCT_inv = jnp.linalg.inv(constraint_grad @ constraint_grad.transpose((0, 2, 1)))

        # Get projection tensor via equation (36), shape (N, d * T, d * T).
        print('first term', jnp.expand_dims(jnp.eye((self.dx + self.du) * self.T + self.dg), axis=0).shape)
        print('second term', constraint_grad.transpose((0, 2, 1)).shape)
        print('third term', dCdCT_inv.shape)
        print('fourth term', constraint_grad.shape)
        projection = ((jnp.expand_dims(jnp.eye((self.dx + self.du) * self.T + self.dg), axis=0) -
                      constraint_grad.transpose((0, 2, 1))) @ dCdCT_inv @ constraint_grad)

        # Get the tangent space kernel from equation (29), shape (N, N, d * T, d * T).
        k_tangent = (k.reshape(self.N, self.N, 1, 1) * jnp.expand_dims(projection, axis=0) @
                     jnp.expand_dims(projection, axis=1))

        # Reshape inputs for computing the gradient of the projection tensor.
        dCdCT_inv = jnp.expand_dims(dCdCT_inv, axis=1)
        dCT = jnp.expand_dims(constraint_grad.transpose(0, 2, 1), axis=1)
        constraint_grad = jnp.expand_dims(constraint_grad, axis=1)
        constraint_hess = constraint_hess.transpose(0, 3, 1, 2)
        hess_CT = constraint_hess.transpose(0, 1, 3, 2)

        # Get the A term in equation (57) using equation (58) for all trajectories.
        first_term = hess_CT @ dCdCT_inv @ constraint_grad
        third_term = dCT @ dCdCT_inv @ constraint_hess

        # Get the B term in equation (57) using equation (59) and equation (60).
        second_term = (dCT @ dCdCT_inv @ (constraint_hess @ dCT + constraint_grad @ hess_CT) @ dCdCT_inv @
                       constraint_grad)

        # Get the gradient of the projection tensor from equation (57).
        grad_projection = jnp.einsum('mijj->mij',
                                     (first_term + third_term - second_term).transpose(0, 2, 3, 1))

        # Get the first term for the gradient of the projection tensor from equation (32).
        PP = jnp.expand_dims(projection, axis=0) @ jnp.expand_dims(projection, axis=1)
        first_term = jnp.einsum('nmj, nmij->nmi', k_grad, PP)

        # Get the second term for the gradient of the projection tensor from equation (32).
        second_term = (jnp.expand_dims(jnp.expand_dims(k, axis=-1), axis=-1) *
                       jnp.expand_dims(projection, axis=0) @ jnp.expand_dims(grad_projection, axis=1))
        second_term = jnp.sum(second_term, axis=3)

        # Get the gradient of the projection tensor from equation (32).
        grad_K_tangent = jnp.sum(first_term + second_term, axis=0)

        # Get the first term for computing the tangent space step.
        kernelized_score = jnp.sum(k_tangent @ -c_grad.reshape(self.N, 1, -1, 1), axis=0)
        if gamma is not None:
            # Get tangent space step from equation (46).
            tangent_step = -(gamma * kernelized_score.squeeze(-1) / self.N + grad_K_tangent / self.N)
            assert tangent_step.shape == (self.N, self._slack_trajectory_dim())
            return tangent_step
        else:
            # Get tangent space step from equation (38).
            tangent_step = -(kernelized_score.squeeze(-1) / self.N + grad_K_tangent / self.N)
            assert tangent_step.shape == (self.N, self._slack_trajectory_dim())
            return tangent_step

    def _combined_constraint(self, xuz: jnp.array, initial_state: jnp.array) -> jnp.array:
        """Compute the combined equality constraint function. This should be formed from composition of the specified
        equality constraints, specified inequality constraints paired with slack variables, dynamics constraints, and an
        additional constraint for keeping the initial state in place.

        Args:
            xuz: All N trajectories including slack variables stacked as rows, shape (N, (dx + du) * T + dg).
            initial_state: The state of the initial condition for the trajectories, shape (dx,).

        Returns:
            The evaluated combined constraint, shape (N, dh + dg + dx * T).
        """
        assert xuz.shape == (self.N, self._slack_trajectory_dim())
        assert initial_state.shape == (self.dx,)

        if self.g is not None:
            combined_constraint = jnp.concatenate((self.h(xuz[:, :self._trajectory_dim()]),
                                                   self._slack_constraint(xuz),
                                                   self._dynamics_constraint(xuz),
                                                   self._start_constraint(xuz, initial_state)), axis=1)
        else:
            combined_constraint = jnp.concatenate((self.h(xuz[:, :self._trajectory_dim()]),
                                                   self._dynamics_constraint(xuz),
                                                   self._start_constraint(xuz, initial_state)), axis=1)
        assert combined_constraint.shape == (self.N, self.dh + self.dg + self.dx * self.T)
        return combined_constraint

    def _slack_constraint(self, xuz: jnp.array) -> jnp.array:
        """Compute the slack constraint by evaluating the inequality constraint function and adding in the slack
        variables according to equation (33). This function's output will be constrained to zero in the optimizer.

        Args:
            xuz: All N trajectories including slack variables stacked as rows, shape (N, (dx + du) * T + dg).

        Returns:
            The evaluated slack constraint, shape (N, dg).
        """
        assert xuz.shape == (self.N, self._slack_trajectory_dim())

        inequality_with_slack = self.g(xuz[:, :self._trajectory_dim()]) + 0.5 * xuz[:, self._trajectory_dim()+1:] ** 2
        assert inequality_with_slack.shape == (self.N, self.dg)
        return inequality_with_slack

    def _dynamics_constraint(self, xuz: jnp.array) -> jnp.array:
        """Compute the dynamics constraint which enforces that the points in each trajectory spline are in fact related
        by the supplied dynamics function. This function's output will be constrained to zero in the optimizer.

        Args:
            xuz: All N trajectories including slack variables stacked as rows, shape (N, (dx + du) * T + dg).

        Returns:
            The dynamics error on every point except the first one in every trajectory, shape (N, dx * (T - 1)).
        """
        assert xuz.shape == (self.N, self._slack_trajectory_dim())

        # Index out the state-space points for each trajectory spline, to get inputs for batched dynamics function.
        points = xuz[:, :self._trajectory_dim()].reshape(self.N * self.T, self.dx + self.du)
        states = points[:, :self.dx]
        control_inputs = points[:, -self.du:]

        # Compute the error of the dynamics function in predicting the next state along each spline.
        states_forward_one_without_last = self.f(states, control_inputs).reshape(self.N, self.T * self.dx)[:, :-self.dx]
        states_without_first = states.reshape(self.N, self.T * self.dx)[:, self.dx:]
        dynamics_error = states_forward_one_without_last - states_without_first
        assert dynamics_error.shape == (self.N, self.dx * (self.T - 1))
        return dynamics_error

    def _start_constraint(self, xuz: jnp.array, initial_state: jnp.array) -> jnp.array:
        """Compute the start constraint function which enforces that the starting state in the trajectories is the same
        as the supplied initial state. This function's output will be constrained to zero in the optimizer.

        Args:
            xuz: All N trajectories including slack variables stacked as rows, shape (N, (dx + du) * T + dg).
            initial_state: The state of the initial condition for the trajectories, shape (dx,).

        Returns:
            The start error on every trajectory, shape (N, dx).
        """
        assert xuz.shape == (self.N, self._slack_trajectory_dim())

        start_error = xuz[:, :self.dx] - initial_state
        assert start_error.shape == (self.N, self.dx)
        return start_error

    def _bound_projection(self, xuz: jnp.array) -> jnp.array:
        """Clip the trajectories inside the (min, max) bounds on state and control inputs specified by self.x_bounds and
        self.u_bounds.

        Args:
            xuz: All N trajectories including slack variables stacked as rows, shape (N, (dx + du) * T + dg).

        Returns:
            The input array with state and control values clamped inside the range, shape (N, (dx + du) * T + dg).
        """
        assert xuz.shape == (self.N, self._slack_trajectory_dim())

        # Get bounding arrays for projection.
        point_min = jnp.concatenate((jnp.ones((self.N, self.dx)) * self.x_bounds[0],
                                     jnp.ones((self.N, self.du) * self.u_bounds[0])), axis=1)
        point_max = jnp.concatenate((jnp.ones((self.N, self.dx)) * self.x_bounds[1],
                                     jnp.ones((self.N, self.du) * self.u_bounds[1])), axis=1)
        trajectory_min = jnp.tile(point_min, self.T)
        trajectory_max = jnp.tile(point_max, self.T)

        # Project trajectories inside bounds and then append slack variables.
        x_in_bounds = jnp.clip(xuz[:, :self._trajectory_dim()], trajectory_min, trajectory_max)
        xuz_in_bounds = jnp.concatenate((x_in_bounds, xuz[:, self._trajectory_dim()+1:]))
        assert xuz_in_bounds.shape == (self.N, self._slack_trajectory_dim())
        return xuz_in_bounds

    def _trajectory_dim(self) -> int:
        """Get the length of a single trajectory.
        """
        return (self.dx + self.du) * self.T

    def _slack_trajectory_dim(self) -> int:
        """Get the length of a single trajectory with slack variables.
        """
        return (self.dx + self.du) * self.T + self.dg
