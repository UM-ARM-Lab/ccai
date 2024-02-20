import torch
from torch.profiler import profile

import numpy as np
import jax
import jax.numpy as jnp

from torch_cg import cg_batch


def compute_step_jax(grad_J: jnp.array,  # nabla log(p(Tau^j | o)) stacked as rows in a matrix (N, d * T)
                     K: jnp.array,  # K(Tau^i, Tau^j) as (i, j) entries in a matrix (N, N)
                     grad_K: jnp.array,
                     # Add third dimension with the length of the trajectory to hold derivatives (N, N, d * T)
                     C: jnp.array,
                     # Evaluation of constraint functions for each trajectory as rows in a matrix (N, dg + dh)
                     dC: jnp.array,
                     # Add third dimension with the length of the trajectory to hold derivatives (N, dg + dh, d * T)
                     hess_C: jnp.array
                     # Add fourth dimension with the length of the trajectory to hold derivatives (N, dg + dh, d * T, d * T)
                     ) -> jnp.array:
    """
    Args:
    Returns:
    """
    N = 8
    T = 12
    d = 16
    dh = 0
    gamma = 1
    alpha_C = 1
    alpha_J = 0.1

    # Get inv(dC * dCT), shape (N, dg + dh, dg + dh).
    dCdCT_inv = jnp.linalg.inv(dC @ dC.transpose((0, 2, 1)))

    # Get projection tensor via equation (36), shape (N, d * T, d * T).
    projection = jnp.expand_dims(jnp.eye(d * T + dh), axis=0) - dC.transpose((0, 2, 1)) @ dCdCT_inv @ dC

    # Get constraint step from equation (39), shape (N, d * T).
    phi_C = (dC.transpose((0, 2, 1)) @ dCdCT_inv @ jnp.expand_dims(C, axis=-1)).squeeze(-1)

    # Get the tangent space kernel from equation (29), shape (N, N, d * T, d * T).
    K_tangent = K.reshape(N, N, 1, 1) * jnp.expand_dims(projection, axis=0) @ jnp.expand_dims(projection, axis=1)

    # Reshape inputs for computing the gradient of the projection tensor.
    dCdCT_inv = jnp.expand_dims(dCdCT_inv, axis=1)
    dCT = jnp.expand_dims(dC.transpose(0, 2, 1), axis=1)
    dC = jnp.expand_dims(dC, axis=1)
    hess_C = hess_C.transpose(0, 3, 1, 2)
    hess_CT = hess_C.transpose(0, 1, 3, 2)

    # Get the A term in equation (57) using equation (58) for all trajectories.
    first_term = hess_CT @ dCdCT_inv @ dC
    third_term = dCT @ dCdCT_inv @ hess_C

    # Get the B term in equation (57) using equation (59) and equation (60).
    second_term = dCT @ dCdCT_inv @ (hess_C @ dCT + dC @ hess_CT) @ dCdCT_inv @ dC

    # Get the gradient of the projection tensor from equation (57).
    grad_projection = jnp.einsum('mijj->mij',
                                 (first_term + third_term - second_term).transpose(0, 2, 3, 1))

    # Get the first term for the gradient of the projection tensor from equation (32).
    PP = jnp.expand_dims(projection, axis=0) @ jnp.expand_dims(projection, axis=1)
    first_term = jnp.einsum('nmj, nmij->nmi', grad_K, PP)

    # Get the second term for the gradient of the projection tensor from equation (32).
    second_term = (jnp.expand_dims(jnp.expand_dims(K, axis=-1), axis=-1) *
                   jnp.expand_dims(projection, axis=0) @ jnp.expand_dims(grad_projection, axis=1))
    second_term = jnp.sum(second_term, axis=3)

    # Get the gradient of the projection tensor from equation (32).
    grad_K_tangent = jnp.sum(first_term + second_term, axis=0)

    # Get tangent space step from equation (46).
    kernelized_score = jnp.sum(K_tangent @ -grad_J.reshape(N, 1, -1, 1), axis=0)
    phi_tangent = -(gamma * kernelized_score.squeeze(-1) / N + grad_K_tangent / N)

    # Return the update terms from equation (26).
    return alpha_J * phi_tangent + alpha_C * phi_C


jit_compute_step_jax = jax.jit(compute_step_jax)


class FasterConstrainedSteinTrajOpt:
    """Faster implementation of CSVTO
    Attributes:
    Methods:
    """

    def __init__(self, problem, params):
        self.dt = params.get('step_size', 1e-2)  # Full gradient step scale
        self.alpha_J = params.get('alpha_J', 1)  # Tanget step scale
        self.alpha_C = params.get('alpha_C', 1)  # Constraint step scale
        self.iters = params.get('iters', 100)  # Number of gradient steps to take in sequence
        self.penalty = params.get('penalty', 1e2)  # Weight on 1 norm of equality constraint violation

        self.problem = problem  # Problem object
        self.dx: int = problem.dx  # Dimension of state
        self.du: int = problem.du  # Dimension of control inputs
        self.dg: int = problem.dg  # Dimension of inequality constraint
        self.dh: int = problem.dh  # Dimension of equality constraint
        self.dz: int = problem.dz  # Dimension of slack variables
        self.T: int = self.problem.T  # Time horizon

        # TODO: Refactor everything below here eventually
        self.profiler = profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True,
                                profile_memory=True,
                                with_stack=True)  # PyTorch profile for performance evaluation
        self.dtype = torch.float32  # Data type used to maintain all data
        self.gamma = 1
        self.max_gamma = 1

        self.xuz = None
        self.grad_J = None
        self.K = None
        self.grad_K = None
        self.C = None
        self.dC = None
        self.hess_C = None

        self.grad_J_jax = None
        self.K_jax = None
        self.grad_K_jax = None
        self.C_jax = None
        self.dC_jax = None
        self.hess_C_jax = None
        self.compiled_step_jax = None

    def compute_update(self, xuz):
        """Compute the combined tangent and constraint gradient for a given augmented state.

        Args:
            xuz (torch.tensor): Augmented state tensor of shape
                (#particles, time_horizon * (state_dimension + control_dimension + slack_variable_dimension)).

        Returns:
            Gradient tensor parallel to xuz of shape
            (#particles, time_horizon * (state_dimension + control_dimension + slack_variable_dimension)).
        """
        N = xuz.shape[0]
        d = self.dx + self.du
        xuz = xuz.detach()
        xuz.requires_grad = True

        xuz = xuz.to(dtype=torch.float32)
        grad_J, hess_J, K, grad_K, C, dC, hess_C = self.problem.eval(xuz.to(dtype=torch.float32))

        # Use approximation for hessian.
        if hess_C is None:
            hess_C = torch.zeros(N, self.dh + self.dg,
                                 self.T * (self.dx + self.du + self.dz),
                                 self.T * (self.dx + self.du + self.dz), device=xuz.device)

        # Store values for run_compute_update.
        self.N = N
        self.d = d

        self.xuz = xuz
        self.grad_J = grad_J
        self.K = K
        self.grad_K = grad_K
        self.C = C
        self.dC = dC
        self.hess_C = hess_C

        # Store values for run_compute_update_jax.
        xuz = jnp.asarray(xuz.detach().cpu().numpy())
        grad_J = jnp.asarray(grad_J.detach().cpu().numpy())
        K = jnp.asarray(K.detach().cpu().numpy())
        grad_K = jnp.asarray(grad_K.detach().cpu().numpy())
        C = jnp.asarray(C.detach().cpu().numpy())
        dC = jnp.asarray(dC.detach().cpu().numpy())
        hess_C = jnp.asarray(hess_C.detach().cpu().numpy())

        self.grad_J_jax = grad_J
        self.K_jax = K
        self.grad_K_jax = grad_K
        self.C_jax = C
        self.dC_jax = dC
        self.hess_C_jax = hess_C

        self.compiled_step = jit_compute_step_jax.lower(grad_J, K, grad_K, C, dC, hess_C).compile()

    def run_compute_update(self):
        return self._compute_step(self.xuz, self.grad_J, self.K, self.grad_K, self.C, self.dC, self.hess_C, self.N, self.T, self.d, self.dg, self.dh, self.gamma,
                                  self.dtype, self.alpha_C, self.alpha_J)
    def run_compute_update_jax(self):
        jax_return: jnp.array = self.compiled_step(self.grad_J_jax, self.K_jax, self.grad_K_jax, self.C_jax, self.dC_jax, self.hess_C_jax)
        return torch.from_numpy(np.asarray(jax_return)).cuda()

    def _compute_step(self,
                      xuz: torch.Tensor,
                      grad_J: torch.Tensor,
                      K: torch.Tensor,
                      grad_K: torch.Tensor,
                      C: torch.Tensor,
                      dC: torch.Tensor,
                      hess_C: torch.Tensor,
                      N: int, T: int, d: int, dg: int, dh: int, gamma: int,
                      dtype,
                      alpha_C: float, alpha_J: float) -> torch.Tensor:
        """
        Args:
        Returns:
        """
        with torch.no_grad():
            # Solve Ax = I to get A^-1, where A = dC * dCT. dCdCT_inv is of shape (N, dg + dh, dg + dh).
            eye = torch.eye(dg + dh).repeat(N, 1, 1).to(device=C.device, dtype=dtype)
            dCdCT_inv = torch.linalg.solve(dC @ dC.permute(0, 2, 1), eye)

            # get projection operator
            projection = dCdCT_inv @ dC
            eye = torch.eye(d * T + dh, device=xuz.device, dtype=xuz.dtype).unsqueeze(0)
            projection = eye - dC.permute(0, 2, 1) @ projection

            # compute term for repelling towards constraint
            xi_C = dCdCT_inv @ C.unsqueeze(-1)
            xi_C = (dC.permute(0, 2, 1) @ xi_C).squeeze(-1)

            # compute gradient for projection
            # now the second index (1) is the
            # x with which we are differentiating
            dCdCT_inv = dCdCT_inv.unsqueeze(1)

            dCT = dC.permute(0, 2, 1).unsqueeze(1)
            dC = dC.unsqueeze(1)

            hess_C = hess_C.permute(0, 3, 1, 2)
            hess_CT = hess_C.permute(0, 1, 3, 2)

            # compute first term
            first_term = hess_CT @ (dCdCT_inv @ dC)
            second_term = dCT @ dCdCT_inv @ (hess_C @ dCT + dC @ hess_CT) @ dCdCT_inv @ dC
            third_term = dCT @ dCdCT_inv @ hess_C

            # add terms and permute so last dimension is the x which we are differentiating w.r.t
            grad_projection = (first_term - second_term + third_term).permute(0, 2, 3, 1)

            # compute total gradient of kernel
            # first term is grad of scalar
            grad_proj = torch.einsum('mijj->mij', grad_projection)

            # now we need to combine all the different projections together
            PP = projection.unsqueeze(0) @ projection.unsqueeze(1)  # should now be N x N x D x D
            PQ = projection
            first_term = torch.einsum('nmj, nmij->nmi', grad_K, PP)
            matrix_K = K.reshape(N, N, 1, 1) * projection.unsqueeze(0) @ projection.unsqueeze(1)

            grad_matrix_K = first_term
            second_term = K.reshape(N, N, 1, 1) * PQ.unsqueeze(0) @ grad_proj.unsqueeze(1)
            second_term = torch.sum(second_term, dim=3)
            grad_matrix_K = grad_matrix_K + second_term

            grad_matrix_K = torch.sum(grad_matrix_K, dim=0)

            # compute kernelized score
            kernelized_score = torch.sum(matrix_K @ -grad_J.reshape(N, 1, -1, 1), dim=0)
            phi = gamma * kernelized_score.squeeze(-1) / N + grad_matrix_K / N  # maximize phi

            xi_J = -phi
        return (alpha_J * xi_J + alpha_C * xi_C).detach()

    def _clamp_in_bounds(self, xuz):
        """
        Args:
        Returns:
        """
        N = xuz.shape[0]
        min_x = self.problem.x_min.reshape(1, 1, -1).repeat(1, self.problem.T, 1)
        max_x = self.problem.x_max.reshape(1, 1, -1).repeat(1, self.problem.T, 1)
        if self.problem.dz > 0:
            if not self.problem.squared_slack:
                min_val = 0
            else:
                min_val = -1e3  # TODO: Why do we need to adjust when using squared slack?
            min_x = torch.cat((min_x,
                               min_val * torch.ones(1, self.problem.T, self.problem.dz, device=min_x.device)), dim=-1)
            max_x = torch.cat((max_x,
                               1e3 * torch.ones(1, self.problem.T, self.problem.dz, device=max_x.device)), dim=-1)

        torch.clamp_(xuz, min=min_x.to(device=xuz.device).reshape(1, -1),
                     max=max_x.to(device=xuz.device).reshape(1, -1))

    def solve(self, x0):
        """
        Args:
            x0 (torch.tensor):
        Returns:
        """
        # Update from problem in case T has changed.
        self.dg = self.problem.dg
        self.dh = self.problem.dh
        self.T = self.problem.T

        # Get the initial slack variable values and build the augmented state xuz.
        N = x0.shape[0]
        if self.problem.dz > 0:
            z_init = self.problem.get_initial_z(x0)
            xuz = torch.cat((x0.clone(), z_init), dim=2).reshape(N, -1)
        else:
            xuz = x0.clone().reshape(N, -1)

        # Driving force function for helping exploration. TODO: Refactor into separate member function.
        resample_period = 100
        T = self.iters
        C = T // resample_period
        p = 1
        self.max_gamma = 1

        def driving_force(t):
            return ((t % (T / C)) / (T / C)) ** p

        path = [xuz.data.clone()]
        self.gamma = self.max_gamma

        # Loop over sequential optimization iterations to step trajectories in parallel.
        for iter in range(self.iters):
            # Update gamma with driving force function.
            if T > 50:
                self.gamma = self.max_gamma * driving_force(iter)

            # Compute the gradient update step.
            grad = self.compute_update(xuz)

            # Normalize each gradient.
            grad_norm = torch.linalg.norm(grad, dim=1, keepdim=True)
            max_norm = 10
            grad = torch.where(grad > max_norm, max_norm * grad / grad_norm, grad)

            # Apply the gradient step to the augmented state and cache the step delta.
            xuz.data = xuz.data - self.dt * grad
            self.delta_x = self.dt * grad

            # Clamp augmented state between state bounds.
            self._clamp_in_bounds(xuz)

            # Append to a list of all trajectories at each stage in the optimization.
            path.append(xuz.data.clone())

        # Calculate penalties for each particle.
        xuz = xuz.to(dtype=torch.float32)
        J = self.problem.get_cost(xuz.reshape(N, self.T, -1)[:, :, :self.dx + self.du])
        C, _, _ = self.problem.combined_constraints(xuz.reshape(N, self.T, -1),
                                                    compute_grads=False,
                                                    compute_hess=False)
        penalty = J.reshape(N) + self.penalty * torch.sum(C.reshape(N, -1).abs(), dim=1)
        idx = torch.argsort(penalty, descending=False)

        # Rearrange the path to store trajectories ordered by penalty along dim 1 and return.
        path = torch.stack(path, dim=0).reshape(len(path), N, self.T, -1)[:, :, :, :self.dx + self.du]
        path = path[:, idx]
        return path
