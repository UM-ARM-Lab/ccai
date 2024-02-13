import torch
from torch.profiler import profile
import csvgd

from torch_cg import cg_batch


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
        self.resample_sigma = params.get('resample_sigma', 1e-2)  # Weight on diagonal covariance for resampling
        self.resample_temperature = params.get('resample_temperature', 0.1)  # Temperature parameter for resampling

        self.problem = problem  # Problem object
        self.dx: int = problem.dx  # Dimension of state
        self.du: int = problem.du  # Dimension of control inputs
        self.dg: int = problem.dg  # Dimension of inequality constraint
        self.dh: int = problem.dh  # Dimension of equality constraint
        self.dz: int = problem.dz  # Dimension of slack variables
        self.T: int = self.problem.T  # Time horizon

        # TODO: Consider refactoring into params.
        self.profiler = profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True,
                                profile_memory=True,
                                with_stack=True)  # PyTorch profile for performance evaluation
        self.dtype = torch.float32  # Data type used to maintain all data
        self.gamma = 1
        self.max_gamma = 1

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
        return self._compute_step(xuz, grad_J, K, grad_K, C, dC, hess_C, N, self.T, d, self.dg, self.dh, self.gamma, self.dtype, self.alpha_C, self.alpha_J)

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

    def resample(self, xuz):
        """
        Args:
            xuz (torch.tensor): Augmented state tensor of shape
                (#particles, time_horizon * (state_dimension + control_dimension) + slack_variable_dimension).
        Returns:
        """
        N = xuz.shape[0]
        xuz = xuz.to(dtype=torch.float32)
        J = self.problem.get_cost(xuz.reshape(N, self.T, -1)[:, :, :self.dx + self.du])
        C, dC, _ = self.problem.combined_constraints(xuz.reshape(N, self.T, -1),
                                                     compute_grads=True,
                                                     compute_hess=False)

        penalty = J.reshape(N) + self.penalty * torch.sum(C.reshape(N, -1).abs(), dim=1)
        # normalize penalty
        penalty = (penalty - penalty.min()) / (penalty.max() - penalty.min())
        # now we sort particles
        idx = torch.argsort(penalty, descending=False)
        xuz = xuz[idx]
        penalty = penalty[idx]
        weights = torch.softmax(-penalty / self.resample_temperature, dim=0, )
        weights = torch.cumsum(weights, dim=0)
        # now we resample
        p = torch.rand(N, device=xuz.device)
        idx = torch.searchsorted(weights, p)

        # compute projection for noise
        dCdCT = dC @ dC.permute(0, 2, 1)
        A_bmm = lambda x: dCdCT @ x
        eye = torch.eye(self.dg + self.dh).repeat(N, 1, 1).to(device=C.device)
        try:
            dCdCT_inv = torch.linalg.solve(dC @ dC.permute(0, 2, 1), eye)
            if torch.any(torch.isnan(dCdCT_inv)):
                raise ValueError('nan in inverse')
        except Exception as e:
            print(e)
            # dCdCT_inv = torch.linalg.lstsq(dC @ dC.permute(0, 2, 1), eye).solution
            # dCdCT_inv = torch.linalg.pinv(dC @ dC.permute(0, 2, 1))
            dCdCT_inv, _ = cg_batch(A_bmm, eye, verbose=False)
        projection = dCdCT_inv @ dC
        eye = torch.eye((self.dx + self.du) * self.T + self.dh, device=xuz.device, dtype=xuz.dtype).unsqueeze(0)
        projection = eye - dC.permute(0, 2, 1) @ projection

        noise = self.resample_sigma * torch.randn_like(xuz)
        eps = projection[idx] @ noise.unsqueeze(-1)
        # print(penalty, weights)
        # we need to add a very small amount of noise just so that the particles are distinct - otherwise
        # they will never separate
        xuz = xuz[idx] + eps.squeeze(-1)

        return xuz

        # weights = weights / torch.sum(weights)

    def solve(self, x0, resample=False):
        """
        Args:
            x0 (torch.tensor):
            resample (bool):
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

        # Optionally resample the augmented state.
        if resample:
            xuz.data = self.resample(xuz.data)

        path = [xuz.data.clone()]
        self.gamma = self.max_gamma

        # Loop over sequential optimization iterations to step trajectories in parallel.
        for iter in range(self.iters):
            # Update gamma with driving force function.
            if T > 50:
                self.gamma = self.max_gamma * driving_force(iter)

            # Resample if the resample period has been hit.
            if (iter + 1) % resample_period == 0 and (iter < T - 1):
                xuz.data = self.resample(xuz.data)

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
