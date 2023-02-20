import torch
import torchdiffeq
from torchdiffeq import odeint  # odeint_adjoint as odeint
import numpy as np


def solve_dual_G_only(dJ, dG):
    # this is just least squares - no non-negative constraitn
    # minimize ||dJ + dG.transpose(0, 1) @_lambda||
    return torch.linalg.lstsq(dG.transpose(0, 1), -dJ)


def solve_dual_H_only(dJ, dH, mu, iters=100, eps=1e-3, tol=1e-4):
    Ab = (2 * dH @ dJ).unsqueeze(-1)
    hessian = dH @ dH.transpose(0, 1)

    # inverse_hessian = torch.linalg.inv(hessian)

    def objective(dJ, dG, dH, _lambda, _mu):
        return torch.linalg.norm(dJ + dG.transpose(0, 1) @ _lambda + dH.transpose(0, 1) @ _mu, dim=0)

    zeros = torch.zeros_like(mu)
    for iter in range(iters):
        grad = Ab + hessian @ mu
        # mu_step = torch.linalg.solve(hessian, eps * grad)
        mu_step = torch.linalg.lstsq(hessian, eps * grad).solution
        new_mu = mu - mu_step  # eps * inverse_hessian @ grad
        new_mu = torch.where(new_mu < 0, zeros, new_mu)

        print(torch.linalg.norm(new_mu - mu))
        if torch.linalg.norm(new_mu - mu) < tol:
            break

        mu = new_mu

    return new_mu


def solve_dual(dJ, dG, dH, iters=100, eps=1e-3, tol=1e-4):
    # Solves dual for _lambda and _mu using projected gradient descent
    # TODO could potentially warm start these but only for G
    if dH is None:
        return None, None  # solve_dual_G_only(dJ, dG), None
    if dG is None:
        _mu = torch.zeros(dH.shape[0], 1).to(dH)
        return None, solve_dual_H_only(dJ, dH, _mu, iters, eps, tol)

    dC = torch.cat((dG, dH), dim=0)
    _mu = torch.zeros(dH.shape[0], 1).to(dH)
    zeros = torch.zeros_like(_mu)
    _lambda = torch.zeros(dG.shape[0], 1).to(dG)
    Ab = (2 * dC @ dJ).reshape(-1, 1)
    hessian = 2 * dC @ dC.transpose(0, 1)

    # inverse_hessian = torch.linalg.inv(hessian)
    def objective(dJ, dG, dH, _lambda, _mu):
        return torch.linalg.norm(dJ.reshape(-1, 1) + dG.transpose(0, 1) @ _lambda + dH.transpose(0, 1) @ _mu, dim=0)

    obj = objective(dJ, dG, dH, _lambda, _mu)

    x = torch.cat((_lambda, _mu), dim=0)

    for iter in range(iters):
        grad = Ab + hessian @ x
        try:
            _step = torch.linalg.solve(hessian, grad)
        except:
            _step = torch.linalg.pinv(hessian) @ grad
        # _step = torch.linalg.lstsq(hessian, grad).solution
        # clip the norm of step to be < 10
        # _step = torch.where(torch.linalg.norm(_step, dim=0, keepdim=True) > 1,
        #                    1 * _step / torch.linalg.norm(_step, dim=0, keepdim=True),
        #                    _step)
        new_lambda = _lambda - eps * _step[:_lambda.shape[0]]
        new_mu = _mu - eps * _step[_lambda.shape[0]:]

        # project mu non negative
        new_mu = torch.where(new_mu < 0, zeros, new_mu)

        new_obj = objective(dJ, dG, dH, new_lambda, new_mu)

        while new_obj > obj:
            _step = _step * 0.1
            new_lambda = _lambda - eps * _step[:_lambda.shape[0]]
            new_mu = _mu - eps * _step[_lambda.shape[0]:]
            new_mu = torch.where(new_mu < 0, zeros, new_mu)
            new_obj = objective(dJ, dG, dH, new_lambda, new_mu)

        _mu = new_mu
        _lambda = new_lambda
        new_x = torch.cat((new_lambda, new_mu), dim=0)
        diff = torch.linalg.norm(x - new_x)
        if diff != diff:
            print(dJ)
            print(dH)
            print(dG)
            print(grad)
            print(_step)
            print(hessian)
            exit(0)

        if torch.linalg.norm(x - new_x) < tol:
            break

        obj = new_obj

        x = new_x

    return _lambda, _mu


def _eliminate(A):
    # eliminates non-independent columns of A
    lam = torch.linalg.eigvals(A @ A.transpose(0, 1))
    return (lam.abs() > 1e-5).nonzero().reshape(-1)


def _eliminate_numpy(A):
    d = A.device
    A = A.detach().cpu().numpy()
    if not 0 in A.shape:
        _, R = np.linalg.qr(A.dot(A.T))
        indices = np.diag(np.abs(R)) > 1e-9
    else:
        indices = []

    return torch.tensor(indices).nonzero().to(d).reshape(-1)


class nlspace_solve:

    def __init__(self, problem, params=None):
        if params is None:
            params = dict()

        self.alpha_J = params.get('alphaJ', 1)
        self.alpha_C = params.get('alphaC', 1)
        self.iters = params.get('maxit', 1000)
        self.dt = params.get('dt', 0.1)
        self.tol = params.get('tol', 1e-4)
        self.problem = problem

    def _grad(self, x):

        #import time
        #start = time.time()
        J, G, H, dJ, dG, dH = self.problem.eval(x)
        #print('eval time', time.time() - start)

        with torch.no_grad():
            # Gidx = self._eliminate(dG)
            # G = G[Gidx]
            # dG = dG[Gidx]

            # get approx active inequality constraints
            _mu = None
            Itilde = []

            if H is not None:
                #ind = _eliminate(torch.cat((dG, dH), dim=0))

                ind = _eliminate_numpy(torch.cat((dG, dH), dim=0))
                #print('elim time', time.time() - start)
                #ind = torch.arange(0, dH.shape[0] + dG.shape[0]).to(device=x.device, dtype=int)
                Gind = ind[(ind < dG.shape[0]).nonzero()].reshape(-1)
                Hind = ind[(ind >= dG.shape[0]).nonzero()].reshape(-1) - dG.shape[0]
                norm1 = (torch.sum(abs(dH[Hind]), 1))
                eps = self.dt * 0.1 * norm1#1 * torch.linalg.norm(dH[Hind], dim=1)

                ItildeEps = (H[Hind] >= -eps).nonzero().reshape(-1)
                Itilde = (H[Hind] >= 0).nonzero().reshape(-1)

                # do a check that any constraints are active
                if len(ItildeEps) == 0 and G is None:
                    raise NotImplementedError()

                _lambda, _mu = solve_dual(dJ, dG[Gind], dH[Hind[ItildeEps]], eps=1)
            else:
                Gind = _eliminate(dG)

            #print('dual time', time.time() - start)
            C_star = torch.tensor([]).to(x)
            dC_star = torch.tensor([]).to(x)
            C_hat = torch.tensor([]).to(x)
            dC_hat = torch.tensor([]).to(x)

            if _mu is not None:
                Ihat = (_mu.reshape(-1) > 1e-8).nonzero().reshape(-1)
                Ihat_eps = Hind[ItildeEps[Ihat]]
                C_hat = H[Ihat_eps]
                dC_hat = dH[Ihat_eps]

                I_star_eps = torch.cat((Ihat_eps, Hind[Itilde]), dim=0).unique()
                C_star = H[I_star_eps]
                dC_star = dH[I_star_eps]

            elif len(Itilde) > 0:
                C_star = H[Hind[Itilde]]
                dC_star = dH[Hind[Itilde]]

            if G is not None:
                C_star = torch.cat((G[Gind], C_star), dim=0)
                dC_star = torch.cat((dG[Gind], dC_star), dim=0)
                C_hat = torch.cat((G[Gind], C_hat), dim=0)
                dC_hat = torch.cat((dG[Gind], dC_hat), dim=0)

            try:
                # cc = dC_star @ dC_star.transpose(0, 1)
                # np.linalg.inv(cc.detach().cpu().numpy())
                dCdCTinv_C_star = torch.linalg.solve(dC_star @ dC_star.transpose(0, 1), C_star)
            except Exception as e:
                #print(e)
                #print('star constraints not qualified, using psuedo-inverse')
                dCdCTinv_C_star = torch.linalg.pinv(dC_star @ dC_star.transpose(0, 1)) @ C_star

            try:
                # cc = dC_hat @ dC_hat.transpose(0, 1)
                # np.linalg.inv(cc.detach().cpu().numpy())
                dCdCTinv_dC_hat = torch.linalg.solve(dC_hat @ dC_hat.transpose(0, 1), dC_hat)
            except Exception as e:
                #print(e)
                #print('hat constraints not qualified, using psuedo-inverse')
                dCdCTinv_dC_hat = torch.linalg.pinv(dC_hat @ dC_hat.transpose(0, 1)) @ dC_hat

            xiJ = dJ - dC_hat.transpose(0, 1) @ dCdCTinv_dC_hat @ dJ
            xiC = dC_star.transpose(0, 1) @ dCdCTinv_C_star

            AC = min(0.9 / self.dt, self.alpha_C / max(torch.linalg.norm(xiC), 1e-9))
            if self.normxiJ is None:
                self.normxiJ = torch.linalg.norm(xiJ)
                AJ = self.alpha_J / max(torch.linalg.norm(xiJ), 1e-9)
            else:
                AJ = self.alpha_J / max(1e-9 + torch.linalg.norm(xiJ), self.normxiJ)

            #print('step time', time.time() - start)
            return AJ * xiJ + AC * xiC

    def solve(self):
        # initialize
        x = self.problem.x0
        x.requires_grad = True
        self.normxiJ = None
        import time

        dual_opt_time = 0.0
        step_time = 0.0
        eval_time = 0.0
        for iter in range(self.iters):
            # Get problem values & gradients
            start = time.time()
            step = self.dt * self._grad(x)
            x = x - step
            end = time.time()
            step_time += end - start
            if torch.linalg.norm(step) < self.tol:
                print(iter)
                break

        print('step time', step_time)
        print('dual optim time', dual_opt_time)
        print('eval time', eval_time)
        return x

    def solve_ode(self):
        import time
        start = time.time()
        self.normxiJ = None

        def ode(t, x):
            return -self._grad(x)

        self.solver = 'dopri5'
        self.solver_options = {}
        self.atol = 1e-4
        self.rtol = 1e-4

        x0 = self.problem.x0
        x0.requires_grad = True
        x = odeint(ode, x0, torch.tensor([0.0, 10]).to(x0),
                   method=self.solver,
                   atol=self.atol,
                   rtol=self.rtol,
                   options=self.solver_options
                   )
        end = time.time()
        print('total time', end - start)
        return x[-1]


class test_problem:

    def __init__(self,
                 x0=torch.tensor([1.5, 2.25]),
                 device='cuda:0'
                 ):
        self.x0 = x0.to(device)
        self.dJ = torch.tensor([0.3, 1.0]).to(device)

    def eval(self, x):
        J = x[1] + 0.3 * x[0]
        H = torch.tensor([-x[1] + 1.0 / x[0], -(3 - x[0] - x[1])]).to(x)
        dH = torch.tensor([[-1.0 / x[0] ** 2, -1], [1, 1]]).to(x)
        G = None
        dG = None
        return J, G, H, self.dJ, dG, dH

    @property
    def x0(self):
        return self._x0.clone()

    @x0.setter
    def x0(self, x):
        self._x0 = x


if __name__ == "__main__":
    import numpy as np

    torch.manual_seed(1234)
    np.random.seed(1234)
    xdim = 1800
    nineq = 1000
    neq = 400
    dG = torch.randn(neq, xdim)
    dH = torch.randn(nineq, xdim)
    dJ = torch.rand(xdim, 1)
    _lambda = torch.zeros(neq, 1)
    _mu = torch.zeros(nineq, 1)

    mu, lambda_ = solve_dual(dJ, dG, dH, iters=100, eps=1)

    problem = test_problem(device='cpu')
    solver = nlspace_solve(problem)
    import time

    start = time.time()
    solver.solve()
    end = time.time()
    print('solver time', end - start)

    """
    import osqp
    import numpy as np
    import scipy as sp
    from scipy import sparse

    x = torch.cat((_lambda, _mu), dim=0).reshape(-1).numpy()
    print(x.shape)
    Ad = torch.cat((dG, dH), dim=0).transpose(0, 1)
    b = -dJ.reshape(-1).numpy()

    m, n = Ad.shape

    print(m, n)
    # OSQP data
    P = sparse.block_diag([sparse.csc_matrix((n, n)), sparse.eye(m)], format='csc')

    q = np.zeros(n + m)
    A = sparse.vstack([
        sparse.hstack([Ad, -sparse.eye(m)]),
        sparse.hstack([sparse.eye(n), sparse.csc_matrix((n, m))])], format='csc')
    l = np.hstack([b, -np.inf * np.ones(n)])
    u = np.hstack([b, np.inf * np.ones(neq), np.zeros(nineq)])

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P, q, A, l, u, eps_abs=1e-20, eps_rel=1e-20, eps_prim_inf=1e-6, eps_dual_inf=1e-6)

    # Solve problem
    res = prob.solve()

    dx = nineq + neq
    error = res.x[dx:]
    x = res.x[:dx]

    print(x)
    #print(error)
    #print(Ad @ x - b)
    """
