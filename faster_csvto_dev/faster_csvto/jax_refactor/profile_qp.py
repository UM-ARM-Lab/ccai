import time

import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP
import torch

import numpy as np

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
        diff = torch.linalg.norm(new_obj - obj, dim=1)
        if torch.all(diff < tol):
            break

        # for next time
        obj = new_obj
        x = new_x
    _lambda, _mu = x[:, _lambda.shape[1]], x[:, _lambda.shape[1]:]
    return _lambda, _mu


@jax.jit
def solve(Q, c, A, l, u):
    qp = BoxOSQP(maxiter=1000, tol=1e-6)
    return qp.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params


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
    B = 1
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

    from jaxopt import BoxOSQP

    # dG: jnp.array = dG.cpu().numpy()
    # dH: jnp.array = dH.cpu().numpy()
    # dJ: jnp.array = dJ.cpu().numpy()

    dG = jnp.array(np.random.randn(xdim, neq))
    dH = jnp.array(np.random.randn(xdim, nineq))
    dJ = jnp.array(np.random.randn(xdim, 1))

    half_Q = jnp.concatenate((dJ, dG, dH), axis=1)
    Q = half_Q.T @ half_Q
    c = jnp.zeros((1 + neq + nineq))
    A = jnp.eye(1 + neq + nineq)
    bound = jnp.ones(1 + neq + nineq)
    l = bound.at[1:].set(-jnp.inf)
    u = bound.at[1:].set(jnp.inf)

    # Q = {'p1': Q,
    #      'p2': Q,
    #      'p3': Q,
    #      'p4': Q,
    #      'p5': Q,
    #      'p6': Q,
    #      'p7': Q,
    #      'p8': Q}
    # c = {'p1': c,
    #      'p2': c,
    #      'p3': c,
    #      'p4': c,
    #      'p5': c,
    #      'p6': c,
    #      'p7': c,
    #      'p8': c}
    # A = {'p1': A,
    #      'p2': A,
    #      'p3': A,
    #      'p4': A,
    #      'p5': A,
    #      'p6': A,
    #      'p7': A,
    #      'p8': A}
    # l = {'p1': l,
    #      'p2': l,
    #      'p3': l,
    #      'p4': l,
    #      'p5': l,
    #      'p6': l,
    #      'p7': l,
    #      'p8': l}
    # u = {'p1': u,
    #      'p2': u,
    #      'p3': u,
    #      'p4': u,
    #      'p5': u,
    #      'p6': u,
    #      'p7': u,
    #      'p8': u}

    qp = BoxOSQP(jit=True)

    compile_start = time.time()
    # sol = qp.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params
    sol = solve(Q, c, A, l, u)
    compile_end = time.time()

    solve_start = time.time()
    # sol = qp.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params
    sol = solve(Q, c, A, l, u)
    solve_end = time.time()

    solve_start2 = time.time()
    # sol2 = qp.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params
    sol = solve(Q, c, A, l, u)
    solve_end2 = time.time()

    # print('primal solution', sol.primal)
    # print(sol.dual_eq)
    # print(sol.dual_ineq)

    # print(sol.primal)
    print(compile_end - compile_start)
    print(solve_end - solve_start)
    print(solve_end2 - solve_start2)
