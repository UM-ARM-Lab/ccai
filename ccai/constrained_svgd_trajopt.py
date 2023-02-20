import torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity


class ConstrainedSteinTrajOpt:

    def __init__(self, problem, dt, alpha_C, alpha_J):
        self.dt = dt

        self.problem = problem
        self.dx = problem.dx
        self.du = problem.du
        self.dg = problem.dg
        self.dh = problem.dh
        self.T = self.problem.T
        self.profile = profile

        self.matrix_kernel = False
        self.use_fisher = False
        self.use_hessian = False  # True
        self.use_constraint_hessian = True

        self.alpha_C = alpha_C
        self.alpha_J = alpha_J
        self.iters = 500

        self.normxiJ = None

    def compute_update(self, xuz):
        N = xuz.shape[0]
        d = self.dx + self.du
        xuz = xuz.detach()
        xuz.requires_grad = True
        grad_J, K, grad_K, C, dC, hess_C = self.problem.eval(xuz)

        with torch.no_grad():
            # we try and invert the dC dCT, if it is singular then we use the psuedo-inverse

            eye = torch.eye(self.dg + self.dh).repeat(N, 1, 1).to(device=C.device)
            try:
                dCdCT_inv = torch.linalg.solve(dC @ dC.permute(0, 2, 1), eye)
                if torch.any(torch.isnan(dCdCT_inv)):
                    raise ValueError('nan in inverse')
            except Exception as e:
                print(e)
                # dCdCT_inv = torch.linalg.lstsq(dC @ dC.permute(0, 2, 1), eye).solution
                dCdCT_inv = torch.linalg.pinv(dC @ dC.permute(0, 2, 1))

            # get projection operator
            projection = dCdCT_inv @ dC
            eye = torch.eye(d * self.T + self.dh, device=xuz.device, dtype=xuz.dtype).unsqueeze(0)
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

            if self.use_constraint_hessian:
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

            if self.matrix_kernel:
                grad_K = torch.cat((grad_K, torch.zeros(N, N, self.T * d, self.dh, device=xuz.device)), dim=-1)
                grad_K = torch.cat((grad_K, torch.zeros(N, N, self.dh, self.T * d + self.dh, device=xuz.device)),
                                   dim=-2)

                first_term = projection.unsqueeze(0) @ grad_K @ projection.unsqueeze(1)
                K_extended = torch.eye(self.T * d + self.dh, self.T * d + self.dh, device=xuz.device).repeat(N, N, 1, 1)
                K_extended[:, :, :self.T * d, :self.T * d] = K
                first_term = torch.sum(first_term, dim=3)
                matrix_K = projection.unsqueeze(0) @ K_extended @ projection.unsqueeze(1)

            else:
                if self.use_fisher or self.use_hessian:
                    Q_inv = 0.1 * torch.eye(d * self.T + self.dh, device=xuz.device)
                    # Q_inv[:self.T*d, :self.T*d] = torch.linalg.pinv(Q)

                    PQ = projection @ Q_inv.unsqueeze(0)
                    PQP = PQ.unsqueeze(0) @ projection.unsqueeze(1)
                    first_term = torch.einsum('nmj, nmij->nmi', grad_K, PQP)
                    matrix_K = K.reshape(N, N, 1, 1) * PQ.unsqueeze(0) @ projection.unsqueeze(1)

                else:
                    PQ = projection
                    first_term = torch.einsum('nmj, nmij->nmi', grad_K, PP)
                    matrix_K = K.reshape(N, N, 1, 1) * projection.unsqueeze(0) @ projection.unsqueeze(1)

            grad_matrix_K = first_term
            if self.use_constraint_hessian:
                second_term = K.reshape(N, N, 1, 1) * PQ.unsqueeze(0) @ grad_proj.unsqueeze(1)
                second_term = torch.sum(second_term, dim=3)
                grad_matrix_K = grad_matrix_K + second_term

            grad_matrix_K = torch.sum(grad_matrix_K, dim=0)

            # compute kernelized score
            kernelized_score = torch.sum(matrix_K @ -grad_J.reshape(N, 1, -1, 1), dim=0)
            phi = self.gamma * kernelized_score.squeeze(-1) / N + grad_matrix_K / N
            xi_J = -phi

            # Normalize gradient
            normxiC = torch.clamp(torch.linalg.norm(xi_C, dim=1, keepdim=True, ord=np.inf), min=1e-6)
            normxiC = torch.min(0.9 * torch.ones_like(normxiC) / self.dt, normxiC)
            xi_C = self.alpha_C * xi_C / normxiC

            if self.normxiJ is None:
                self.normxiJ = torch.clamp(torch.linalg.norm(xi_J, dim=1, keepdim=True, ord=np.inf), min=1e-6)
                xi_J = self.alpha_J * xi_J / self.normxiJ
            else:
                normxiJ = torch.clamp(torch.linalg.norm(xi_J, dim=1, keepdim=True, ord=np.inf), min=1e-6)
                normxiJ = torch.max(normxiJ, self.normxiJ)
                xi_J = self.alpha_J * xi_J / normxiJ

        if torch.any(torch.isnan(xi_J)) or torch.any(torch.isnan(xi_C)):
            # for debugging purposes
            print(torch.any(torch.isnan(dCdCT_inv)))
            print(torch.any(torch.isnan(dC)))
            print(dC.shape)
            print(torch.any(torch.isinf(dC)))
            for d in dC:
                print(d)
                print(d.shape)
                print(torch.linalg.matrix_rank(d))
                print(torch.linalg.svdvals(d))
            exit(0)
        return (self.alpha_J * -xi_J - self.alpha_C * xi_C).detach()

    def _clamp_in_bounds(self, xuz):
        N = xuz.shape[0]
        xuz = xuz.reshape(N, self.T, -1)
        xuz[:, :, :self.dx + self.du] = torch.clamp(xuz[:, :, :self.dx + self.du],
                                                    min=self.problem.x_min.reshape(1, 1, -1).to(device=xuz.device),
                                                    max=self.problem.x_max.reshape(1, 1, -1).to(device=xuz.device))
        return xuz.reshape(N, -1)

    def solve(self, x0):
        self.normxiJ = None

        # Get initial slack variable values
        N = x0.shape[0]
        if self.problem.dz > 0:
            z_init = self.problem.get_initial_z(x0)
            xuz = torch.cat((x0.clone(), z_init), dim=2).reshape(N, -1)
        else:
            xuz = x0.clone().reshape(N, -1)

        # driving force useful for helping exploration -- currently unused
        T = self.iters
        C = 1
        p = 1
        def driving_force(t):
            return ((t % (T / C)) / (T / C)) ** p

        for iter in range(T):
            self.gamma = driving_force(iter+1)
            #self.gamma = 1
            grad = self.compute_update(xuz)
            xuz = xuz + self.dt * grad

            # clamp within bounds for simple bounds
            if self.problem.x_max is not None:
                xuz = self._clamp_in_bounds(xuz)

            if torch.all(torch.linalg.norm(grad, dim=-1) < 1e-3):
                print(f'converged after {iter} iterations')
                break

        return xuz.reshape(N, self.T, -1)[:, :, :self.dx + self.du]
