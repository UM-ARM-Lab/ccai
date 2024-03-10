import torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

from torch_cg import cg_batch


@torch.no_grad()
def solve_dual_batched_no_G(dJ, dH, iters=100, eps=1e-3, tol=1e-4):
    """
           Solve dual NNLS problem min || dJ + \mu^T dH|| s.t. \mu >= 0

           For batched inputs

           Uses projected gradient descent with line search

       """

    B = dJ.shape[0]
    _mu = torch.zeros(B, dH.shape[1], 1).to(dH)
    Ab = (2 * dH @ dJ.unsqueeze(-1))
    AtA = dH @ dH.transpose(1, 2)

    def objective(dJ, dH, _mu):
        return torch.linalg.norm(dJ.unsqueeze(-1) + dH.transpose(1, 2) @ _mu, dim=1)

    obj = objective(dJ, dH, _mu)

    for iter in range(iters):
        # Do update
        _step = Ab + AtA @ _mu
        new_mu = _mu - eps * _step
        torch.clamp_(new_mu, min=0)

        # Backtracking line search
        new_obj = objective(dJ, dH, _mu)
        while torch.any(new_obj > obj):
            _step = torch.where(new_obj.unsqueeze(-1) > obj.unsqueeze(-1), _step * 0.1, torch.zeros_like(_step))
            new_mu = _mu - eps * _step
            torch.clamp_(new_mu, min=0)
            obj = objective(dJ, dH, _mu)

        # check convergence
        diff = torch.linalg.norm(new_obj - obj, dim=1)
        if torch.all(diff < tol):
            break
        # for next time
        obj = new_obj
        _mu = new_mu
    return _mu


@torch.no_grad()
def solve_dual_batched(dJ, dG, dH, iters=100, eps=1e-3, tol=1e-4):
    """
        Solve dual min || dJ + \mu^T dH + \lambda^T dG || s.t. \mu >= 0

        For batched inputs

        Uses projected gradient descent with line search

    """
    B = dJ.shape[0]

    if dG is None:
        return solve_dual_batched_no_G(dJ, dH, iters, eps, tol)

    dC = torch.cat((dG, dH), dim=1)
    _mu = torch.zeros(B, dH.shape[1], 1).to(dH)
    _lambda = torch.zeros(B, dG.shape[1], 1).to(dG)
    Ab = (2 * dC @ dJ.unsqueeze(-1))
    AtA = dC @ dC.transpose(1, 2)

    def objective(dJ, dG, dH, _lambda, _mu):
        return torch.linalg.norm(dJ.unsqueeze(-1) + dG.transpose(1, 2) @ _lambda + dH.transpose(1, 2) @ _mu, dim=1)

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
        diff = torch.linalg.norm(new_obj - obj, dim=1)
        if torch.all(diff < tol):
            break

        # for next time
        obj = new_obj
        x = new_x
    _lambda, _mu = x[:, _lambda.shape[1]], x[:, _lambda.shape[1]:]
    # actually don't care about lambda
    return _mu


def compute_inverse_with_masks(dC, mask):
    B, n, m = dC.shape
    sorting_mask = torch.arange(n, 0, -1).to(device=dC.device)[None, :] * mask
    standard_idx = torch.arange(n)[None, :].repeat(B, 1).to(device=dC.device)
    unsorted_idx = standard_idx.clone()

    sorted_idx = torch.argsort(sorting_mask, descending=True, dim=1)
    B_range = torch.arange(B).to(device=dC.device)
    unsorted_idx[B_range.unsqueeze(-1), sorted_idx] = standard_idx

    # mask out dC
    dC_masked = dC * mask.unsqueeze(-1)

    # sort by the masking
    dC_masked = dC_masked[B_range.unsqueeze(-1), sorted_idx]
    A = dC_masked @ dC_masked.transpose(-1, -2)
    # add diagonal to allow inverse
    eye = torch.eye(n).to(device=dC.device)
    A += eye * (torch.logical_not(mask)[B_range.unsqueeze(-1), sorted_idx].unsqueeze(-1))
    damping = 1e-6
    # now we invert
    A = A + damping * eye

    # we convert to double precision to avoid numerical issues
    A = A.to(torch.float64)
    A_inv = torch.linalg.inv(A + damping * eye)
    A_inv = A_inv.to(torch.float32)

    # now we unsort
    A_inv = A_inv[B_range.unsqueeze(-1), unsorted_idx][B_range.unsqueeze(-1), :, unsorted_idx]
    return A_inv


def eliminate_constraints(dJ, H, dH, G, dG):
    # H is B x dh
    # G is B x dg
    # dH is B x dh x dx
    # dH is B x dg x dx

    # first we need to find violated inequality constraints
    # compute tolerance as ||H||_2 h
    eps = torch.linalg.norm(dH, dim=2)
    tol = 1e-6

    # now we need to find the violated constraints
    I_violated = torch.where(H > 0, torch.ones_like(H), torch.zeros_like(H))

    # constraints within eps of violation (almost violating)
    I_violated_eps = torch.where(H > -eps, torch.ones_like(H), torch.zeros_like(H))

    # now we find active constraints by solving dual problem
    # where we have zeroed out the non-violateed constraints
    _mu = solve_dual_batched(dJ, dG, dH * I_violated_eps.unsqueeze(-1), iters=100000, eps=1e-3, tol=1e-8)

    # make sure that _mu zero for inactive constraints
    _mu = _mu.squeeze(-1) + I_violated

    # used for the tangent-space projection
    I_active = torch.where(_mu > tol, torch.ones_like(_mu), torch.zeros_like(_mu))

    # used for the Gauss-Newton step
    I_active_or_violated = torch.logical_or(I_violated, I_active)

    if G is None:
        C = H
        dC = dH
    else:
        # TODO need to modify indices - maybe make H come before G?
        C = torch.cat((G, H), dim=1)
        dC = torch.cat((dG, dH), dim=1)

    # now let's do the relevant computations
    # first we need to inverse dC dCT for both I_active and I_active_or_violated - need to do some tricks to make it
    # batched when we have different numbers of active constraints
    # compute the two for active/inactive
    dCdCT_inv_active = compute_inverse_with_masks(dC, I_active)
    dCdCT_inv_active_or_violated = compute_inverse_with_masks(dC, I_active_or_violated)

    dC_active = dC * I_active.unsqueeze(-1)
    dC_active_or_violated = dC * I_active_or_violated.unsqueeze(-1)
    C = C * I_active_or_violated

    return C, dC_active, dC_active_or_violated, dCdCT_inv_active, dCdCT_inv_active_or_violated


class ConstrainedSteinTrajOpt:

    def __init__(self, problem, params):
        self.dt = params.get('step_size', 1e-2)
        self.alpha_J = params.get('alpha_J', 1)
        self.alpha_C = params.get('alpha_C', 1)
        self.momentum = params.get('momentum', 0)
        self.iters = params.get('iters', 100)
        self.penalty = params.get('penalty', 1e2)
        self.resample_sigma = params.get('resample_sigma', 1e-2)
        self.resample_temperature = params.get('resample_temperature', 0.1)

        self.problem = problem
        self.dx = problem.dx
        self.du = problem.du
        self.dg = problem.dg
        self.dh = problem.dh
        self.dz = problem.dz
        self.T = self.problem.T
        self.profile = profile

        self.matrix_kernel = False
        self.use_fisher = False
        self.use_hessian = False  # True
        self.use_constraint_hessian = False
        self.active_constraint_method = True
        self.normxiJ = None
        self.dtype = torch.float32
        self.delta_x = None
        self.Bk = None

        if not self.problem.slack_variable:
            self.active_constraint_method = True
            assert self.dz == 0

    def _bfgs_update(self, dC):
        if self.Bk is None:
            # initialize with Gauss-Newton approx
            eye = torch.eye(dC.shape[2]).repeat(dC.shape[0], self.dh + self.dg, 1, 1).to(device=dC.device)
            self.Bk = 1e-3 * eye
        else:
            dx = self.delta_x.unsqueeze(1).repeat(1, self.dh + self.dg, 1).unsqueeze(-1)
            y = (dC - self.old_grad_C).unsqueeze(-1)  # N x (dg+dh) x (dx+du+dz)

            # Use damped BFGS update from Nocedal & Wright Chap 18
            yx = y.transpose(-1, -2) @ dx
            xBx = dx.transpose(-1, 2) @ self.Bk @ dx
            theta = 0.8 * (xBx) / (xBx - yx)
            theta = torch.where(yx < 0.2 * xBx, theta, torch.ones_like(theta))
            r = y * theta + (1 - theta) * self.Bk @ dx

            first_term = r @ r.transpose(-1, -2)
            first_term = first_term / (r.transpose(-1, -2) @ dx)
            second_term = self.Bk @ dx @ dx.transpose(-1, 2) @ self.Bk.transpose(-1, -2)
            second_term = second_term / (dx.transpose(-1, 2) @ self.Bk @ dx)
            self.Bk = self.Bk + first_term - second_term

            # restrict norm of Bk
            Bk_norm = torch.linalg.matrix_norm(self.Bk, dim=(-1, -2), keepdim=True, ord=np.inf)
            max_norm = 100
            self.Bk = torch.where(Bk_norm > max_norm, max_norm * self.Bk / Bk_norm, self.Bk)
        self.old_grad_C = dC

    def compute_update(self, xuz):
        N = xuz.shape[0]
        d = self.dx + self.du
        xuz = xuz.detach()
        xuz.requires_grad = True

        xuz = xuz.to(dtype=torch.float32)
        grad_J, hess_J, K, grad_K, C, dC, hess_C = self.problem.eval(xuz.to(dtype=torch.float32))

        if self.active_constraint_method and self.dh > 0:
            if self.dg == 0:
                G, dG = None, None
                H, dH = C, dC
            else:
                G = C[:, :self.problem.dg]
                H = C[:, self.problem.dg:]
                dG = dC[:, :self.problem.dg]
                dH = dC[:, self.problem.dg:]

            # eliminate inactive constraints
            C, dC_active, dC_active_or_violated, dCdCT_inv_active, dCdCT_inv_active_or_violated = \
                eliminate_constraints(grad_J, H, dH, G, dG)

        else:
            dC_active = dC
            dC_active_or_violated = dC
            dtype = torch.float64

            # Do inversion
            eye = torch.eye(self.dg + self.dh).repeat(N, 1, 1).to(device=C.device, dtype=dtype)
            dCdCT = dC @ dC.permute(0, 2, 1)
            dCdCT = dCdCT.to(dtype=dtype)
            damping_factor = 1e-6
            dCdCT_inv = torch.linalg.inv(dCdCT + damping_factor * eye)
            dCdCT_inv = dCdCT_inv.to(dtype=torch.float32)

            dCdCT_inv_active = dCdCT_inv
            dCdCT_inv_active_or_violated = dCdCT_inv

        # use approximation for hessian
        if hess_C is None:
            # self._bfgs_update(dC)
            # hess_C = self.Bk
            hess_C = torch.zeros(N, self.dh + self.dg,
                                 self.T * (self.dx + self.du + self.dz),
                                 self.T * (self.dx + self.du + self.dz), device=xuz.device)
        with torch.no_grad():

            # get projection operator
            projection = dCdCT_inv_active @ dC_active
            eye = torch.eye((d + self.dz) * self.T, device=xuz.device, dtype=xuz.dtype).unsqueeze(0)
            projection = eye - dC_active.permute(0, 2, 1) @ projection
            # compute term for repelling towards constraint
            xi_C = dCdCT_inv_active_or_violated @ C.unsqueeze(-1)
            xi_C = (dC_active_or_violated.permute(0, 2, 1) @ xi_C).squeeze(-1)

            # compute gradient for projection
            # now the second index (1) is the
            # x with which we are differentiating
            dCdCT_inv_active = dCdCT_inv_active.unsqueeze(1)

            dCT_active = dC_active.permute(0, 2, 1).unsqueeze(1)
            dC_active = dC_active.unsqueeze(1)

            if self.use_constraint_hessian:
                hess_C = hess_C.permute(0, 3, 1, 2)
                hess_CT = hess_C.permute(0, 1, 3, 2)
                # compute first term

                first_term = hess_CT @ (dCdCT_inv_active @ dC_active)
                second_term = dCT_active @ dCdCT_inv_active @ (
                        hess_C @ dCdCT_inv_active + dC @ hess_CT) @ dCdCT_inv_active @ dC_active
                third_term = dCT_active @ dCdCT_inv_active @ hess_C

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
                if hess_J is not None:
                    Q_inv = torch.eye(d * self.T + self.dh, device=xuz.device)
                    Q_inv = torch.linalg.inv(hess_J.unsqueeze(0))

                    PQ = projection @ Q_inv
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
            phi = self.gamma * kernelized_score.squeeze(-1) / N + grad_matrix_K / N  # maximize phi

            xi_J = -phi
            if False:
                # Normalize gradient
                normxiC = torch.clamp(torch.linalg.norm(xi_C, dim=1, keepdim=True, ord=np.inf), min=1e-9)
                # normxiC = torch.min(0.9 * torch.ones_like(normxiC) / self.dt, normxiC)
                xi_C = self.alpha_C * xi_C / normxiC

                # if self.normxiJ is None:
                #    self.normxiJ = torch.clamp(torch.linalg.norm(xi_J, dim=1, keepdim=True, ord=np.inf), min=1e-6)
                #    xi_J = self.alpha_J * xi_J / self.normxiJ
                # else:
                normxiJ = torch.clamp(torch.linalg.norm(xi_J, dim=1, keepdim=True, ord=np.inf), min=1e-6)
                xi_J = self.alpha_J * xi_J / normxiJ

            # make both xi_C and xi_J have a maximum norm of 10
            # norm_xi_C = torch.linalg.norm(xi_C, dim=1, keepdim=True)
            # norm_xi_J = torch.linalg.norm(xi_J, dim=1, keepdim=True)

            # xi_C = torch.where(norm_xi_C > 10, xi_C / norm_xi_C * 10, xi_C)
            # xi_J = torch.where(norm_xi_J > 10, xi_J / norm_xi_J * 10, xi_J)

        return (self.alpha_J * xi_J + self.alpha_C * xi_C).detach()

    def _clamp_in_bounds(self, xuz):
        N = xuz.shape[0]
        min_x = self.problem.x_min.reshape(1, 1, -1).repeat(1, self.problem.T, 1)
        max_x = self.problem.x_max.reshape(1, 1, -1).repeat(1, self.problem.T, 1)
        if self.problem.dz > 0:
            if not self.problem.squared_slack:
                min_val = 0
            else:
                min_val = -1e3
            min_x = torch.cat((min_x,
                               min_val * torch.ones(1, self.problem.T, self.problem.dz, device=min_x.device)), dim=-1)
            max_x = torch.cat((max_x,
                               1e3 * torch.ones(1, self.problem.T, self.problem.dz, device=max_x.device)), dim=-1)

        torch.clamp_(xuz, min=min_x.to(device=xuz.device).reshape(1, -1),
                     max=max_x.to(device=xuz.device).reshape(1, -1))

    def resample(self, xuz):
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
        self.normxiJ = None
        self.Bk = None
        # update from problem incase T has changed
        self.dg = self.problem.dg
        self.dh = self.problem.dh
        self.T = self.problem.T

        # Get initial slack variable values
        N = x0.shape[0]
        if self.problem.dz > 0:
            z_init = self.problem.get_initial_z(x0)
            xuz = torch.cat((x0.clone(), z_init), dim=2).reshape(N, -1)
        else:
            xuz = x0.clone().reshape(N, -1)

        import torch.optim.lr_scheduler as lr_scheduler
        # optim = torch.optim.SGD(params=[xuz], lr=self.dt, momentum=self.momentum,
        #                        nesterov=True if self.momentum > 0 else False)
        # scheduler = lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.1, total_iters=self.iters)

        # optim = torch.optim.RMSprop(params=[xuz], lr=self.dt)

        # driving force useful for helping exploration -- currently unused
        resample_period = 100
        T = self.iters
        C = T // resample_period
        p = 1
        self.max_gamma = 1
        import time
        def driving_force(t):
            return ((t % (T / C)) / (T / C)) ** p

        if resample:
            xuz.data = self.resample(xuz.data)

        path = [xuz.data.clone()]
        self.gamma = self.max_gamma
        for iter in range(T):
            # reset slack variables
            # if self.problem.dz > 0:
            #    xu = xuz.reshape(N, self.T, -1)[:, :, :self.dx + self.du]
            #    z_init = self.problem.get_initial_z(xu)
            #    xuz = torch.cat((xu, z_init), dim=2).reshape(N, -1)

            s = time.time()
            if T > 50:
                self.gamma = self.max_gamma * driving_force(iter)
            #    self.sigma = 0.1 * (1.0 - iter / T) + 1e-2

            # if (iter + 1) % resample_period == 0 and (iter < T - 1):
            #    xuz.data = self.resample(xuz.data)

            # old C
            # oldC, _, _ = self.problem.combined_constraints(xuz.reshape(N, self.T, -1), compute_grads=False)
            #
            # optim.zero_grad()
            grad = self.compute_update(xuz)
            # print(torch.linalg.norm(grad, dim=-1))
            # for k in range(10):
            #    dt = self.dt / (2 ** k)
            #    new_xuz = xuz.data - dt * grad
            #    self._clamp_in_bounds(new_xuz)
            #    # now we compute the new constraint val
            #    newC, _, _ = self.problem.combined_constraints(new_xuz.reshape(N, self.T, -1), compute_grads=False)
            #    if torch.sum(newC ** 2) < torch.sum(oldC ** 2):
            #        xuz.data = new_xuz
            #        break

            # xuz.data = #new_xuz
            grad_norm = torch.linalg.norm(grad, dim=1, keepdim=True)
            max_norm = 10
            grad = torch.where(grad > max_norm, max_norm * grad / grad_norm, grad)
            xuz.data = xuz.data - self.dt * grad
            self.delta_x = self.dt * grad
            # xuz.grad = grad
            # print(torch.linalg.norm(grad))
            # torch.nn.utils.clip_grad_norm_(xuz, 10)
            # new_xuz = xuz + self.dt * grad
            # optim.step()
            # scheduler.step()
            # clamp within bounds for simple bounds
            # if self.problem.x_max is not None:
            self._clamp_in_bounds(xuz)

            path.append(xuz.data.clone())

        # sort particles by penalty
        xuz = xuz.to(dtype=torch.float32)
        J = self.problem.get_cost(xuz.reshape(N, self.T, -1)[:, :, :self.dx + self.du])
        C, _, _ = self.problem.combined_constraints(xuz.reshape(N, self.T, -1),
                                                    compute_grads=False,
                                                    compute_hess=False)
        penalty = J.reshape(N) + self.penalty * torch.sum(C.reshape(N, -1).abs(), dim=1)
        idx = torch.argsort(penalty, descending=False)
        path = torch.stack(path, dim=0).reshape(len(path), N, self.T, -1)[:, :, :, :self.dx + self.du]
        path = path[:, idx]
        return path
