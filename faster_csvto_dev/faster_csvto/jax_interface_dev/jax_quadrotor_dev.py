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
    #  used for runtime assertion in the jit compiled solve() function.
    def __init__(self,
                 c: Callable[[jnp.array], jnp.array],
                 h: Callable[[jnp.array], jnp.array],
                 g: Callable[[jnp.array], jnp.array],
                 f: Callable[[jnp.array], jnp.array],
                 u_bounds: Tuple[int], x_bounds: Tuple[int],
                 dx: int, du: int, dh: int, dg: int) -> None:
        """Set the functions that define the optimization problem along with parameters that determine the size of the
                involved arrays.

        NOTE: In the shape descriptions below, N is the number of particles and T is the time horizon, both of which are
        determined by the optimization parameters when this problem is used to initialize an optimization.

        Args:
            c: The batched cost function, mapping shapes (N, (dx + du) * T) -> (N, 1).
            h: The batched equality constraint function, mapping shapes (N, (dx + du) * T) -> (N, dh).
            g: The batched inequality constraint function, mapping shapes (N, (dx + du) * T) -> (N, dg).
            f: The batched dynamics function, mapping shapes (N, dx) -> (N, dx).
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
        self.g: Callable[[jnp.array], jnp.array] = problem.g
        self.f: Callable[[jnp.array], jnp.array] = problem.f
        self.k: Callable[[jnp.array, jnp.array], jnp.array] = params.k
        self.u_bounds: Tuple[int] = problem.u_bounds
        self.x_bounds: Tuple[int] = problem.x_bounds

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
            x0:

        Returns:
        """
        # Calculate slack variable values via equation (33) and append to the trajectory according to equation (35).
        xuz: jnp.array = jnp.concatenate((x0, jnp.sqrt(-2*self.g(x0))), axis=1)

        # Loop over the gradient step iterations.
        xuz = jax.lax.fori_loop(1, self.K+1, self._solve_iteration, xuz, unroll=True)

        # Find the lowest cost trajectory in the optimized distribution.

        # Return the trajectory without slack variables.
        raise NotImplementedError

    def _solve_iteration(self, iteration: int, xuz: jnp.array) -> jnp.array:
        """

        Args:
        Returns:
        """
        # Set the annealing weight gamma if annealing is on.
        gamma: Optional[jnp.float32] = iteration / self.K if self.anneal else None

        # Evaluate the gradient of the cost function at the current state (without slack variables).
        c_grad = jax.jacfwd(self.c)(xuz[:, :-self.dg])

        # Evaluate the combined constraint function, its gradient, and its hessian at the current augmented state.
        constraint = self._hgf_combined(xuz)
        constraint_grad = jax.jacfwd(self._hgf_combined)(xuz)
        constraint_hess = jax.jacfwd(jax.jacrev(self._hgf_combined))(xuz)

        # Evaluate the kernel function and its gradient at the current state (without slack variables).
        k = self.k(xuz[:, :-self.dg], xuz[:, :-self.dg])
        k_grad = jax.jacrev(self.k)(xuz[:, :-self.dg])  # Differentiate w.r.t. first argument by default.

        # Compute the retraction steps.
        constraint_step = self._compute_constraint_step(constraint, constraint_grad)
        tangent_step = self._compute_tangent_step(c_grad, constraint, constraint_grad, constraint_hess, k, k_grad)

        # Combine the steps according to equation (26) and use them to update the current augmented state.
        xuz -= self.step_scale * (self.alpha_J * tangent_step + self.alpha_C * constraint_step)

        # Clamp the state in bounds.

    def _compute_constraint_step(self, constraint: jnp.array, constraint_grad: jnp.array) -> jnp.array:
        """

        Args:
        Returns:
        """
        # # Get inv(dC * dCT), shape (N, dg + dh, dg + dh).
        # dCdCT_inv = jnp.linalg.inv(dC @ dC.transpose((0, 2, 1)))
        #
        # # Get projection tensor via equation (36), shape (N, d * T, d * T).
        # projection = jnp.expand_dims(jnp.eye(d * T + dh), axis=0) - dC.transpose((0, 2, 1)) @ dCdCT_inv @ dC
        #
        # # Get constraint step from equation (39), shape (N, d * T).
        # phi_C = (dC.transpose((0, 2, 1)) @ dCdCT_inv @ jnp.expand_dims(C, axis=-1)).squeeze(-1)
        #
        # # Get the tangent space kernel from equation (29), shape (N, N, d * T, d * T).
        # K_tangent = K.reshape(N, N, 1, 1) * jnp.expand_dims(projection, axis=0) @ jnp.expand_dims(projection, axis=1)
        #
        # # Reshape inputs for computing the gradient of the projection tensor.
        # dCdCT_inv = jnp.expand_dims(dCdCT_inv, axis=1)
        # dCT = jnp.expand_dims(dC.transpose(0, 2, 1), axis=1)
        # dC = jnp.expand_dims(dC, axis=1)
        # hess_C = hess_C.transpose(0, 3, 1, 2)
        # hess_CT = hess_C.transpose(0, 1, 3, 2)
        #
        # # Get the A term in equation (57) using equation (58) for all trajectories.
        # first_term = hess_CT @ dCdCT_inv @ dC
        # third_term = dCT @ dCdCT_inv @ hess_C
        #
        # # Get the B term in equation (57) using equation (59) and equation (60).
        # second_term = dCT @ dCdCT_inv @ (hess_C @ dCT + dC @ hess_CT) @ dCdCT_inv @ dC
        #
        # # Get the gradient of the projection tensor from equation (57).
        # grad_projection = jnp.einsum('mijj->mij',
        #                              (first_term + third_term - second_term).transpose(0, 2, 3, 1))
        #
        # # Get the first term for the gradient of the projection tensor from equation (32).
        # PP = jnp.expand_dims(projection, axis=0) @ jnp.expand_dims(projection, axis=1)
        # first_term = jnp.einsum('nmj, nmij->nmi', grad_K, PP)
        #
        # # Get the second term for the gradient of the projection tensor from equation (32).
        # second_term = (jnp.expand_dims(jnp.expand_dims(K, axis=-1), axis=-1) *
        #                jnp.expand_dims(projection, axis=0) @ jnp.expand_dims(grad_projection, axis=1))
        # second_term = jnp.sum(second_term, axis=3)
        #
        # # Get the gradient of the projection tensor from equation (32).
        # grad_K_tangent = jnp.sum(first_term + second_term, axis=0)
        #
        # # Get tangent space step from equation (46).
        # kernelized_score = jnp.sum(K_tangent @ -grad_J.reshape(N, 1, -1, 1), axis=0)
        # phi_tangent = -(gamma * kernelized_score.squeeze(-1) / N + grad_K_tangent / N)
        #
        # # Return the update terms from equation (26).
        # return alpha_J * phi_tangent + alpha_C * phi_C
        raise NotImplementedError

    def _compute_tangent_step(self,
                              c_grad: jnp.array,
                              constraint: jnp.array,
                              constraint_grad: jnp.array,
                              constraint_hess: jnp.array,
                              k: jnp.array,
                              k_grad: jnp.array) -> jnp.array:
        """

        Args:
        Returns:
        """
        raise NotImplementedError

    def _project_in_bounds(self, xuz: jnp.array) -> jnp.array:
        """

        Args:
        Returns:
        """
        raise NotImplementedError

    def _hgf_combined(self) -> jnp.array:
        """

        Returns:
        """
