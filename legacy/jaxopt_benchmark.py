import jax.lax
from jaxopt import OSQP
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import timeit, functools
import time

Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
c = jnp.array([1.0, 1.0])
A = jnp.array([[1.0, 1.0]])
b = jnp.array([1.0])
G = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
h = jnp.array([0.0, 0.0])

qp = OSQP()
N = 10
start = time.time()
for _ in range(N):
    qp.run(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h))
end = time.time()
print((end - start) / N)

qp_run_jit = jit(qp.run)
qp_run_jit(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h))

start = time.time()
for _ in range(N):
    qp_run_jit(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h))
end = time.time()
print((end - start) / N)
print(Q.shape, c.shape, A.shape, b.shape, G.shape, h.shape)

N = 100
# try with jit and vmap
Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]]).reshape(1, 2, 2).repeat(repeats=N, axis=0)
c = jnp.array([1.0, 1.0]).reshape(1, 2).repeat(repeats=N, axis=0)
A = jnp.array([[1.0, 1.0]]).reshape(1, 1, 2).repeat(repeats=N, axis=0)
b = jnp.array([1.0]).reshape(1, 1).repeat(repeats=N, axis=0)
G = jnp.array([[-1.0, 0.0], [0.0, -1.0]]).reshape(1, 2, 2).repeat(repeats=N, axis=0)
h = jnp.array([0.0, 0.0]).reshape(1, 2).repeat(repeats=N, axis=0)
qp = OSQP()
qp_vmap = vmap(qp.run)
qp_vmap_jit = jit(qp_vmap)

print(Q.shape, c.shape, A.shape, b.shape, G.shape, h.shape)
qp_vmap_jit(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h))

N = 10
start = time.time()
for _ in range(N):
    qp_vmap_jit(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h))
end = time.time()
print((end - start) / N)

# try with much larger solver
# let's say 50 particles
# trajectory length of 40, state dimension of 7
# equality constraints = 280, number of inequality constraints = 80
print(jax.devices()[0])
seed = 1701
key = random.PRNGKey(seed)
T = 20
n = 9 * T
N = 50

U = random.normal(key, (N, n, n))
Q = jax.lax.batch_matmul(U, jnp.transpose(U, axes=(0, 2, 1))) + jnp.eye(n).reshape(1, n, n).repeat(N, axis=0) * 1e-3
#Q = jnp.eye(n).reshape(1, n, n).repeat(N, axis=0)
c = random.normal(key, (N, n))
A = random.normal(key, (N, 10, n))
b = random.normal(key, (N, 10))
G = -jnp.eye(n).reshape(1, n, n).repeat(N, axis=0)
h = jnp.concatenate((jnp.inf * jnp.ones((N, 7*T)), jnp.zeros((N, 2*T))), axis=1)

sol = qp_vmap_jit(params_obj=(Q, c), params_ineq=(G, h)).params
print(Q.shape, c.shape, A.shape, b.shape, G.shape, h.shape)
N = 10
start = time.time()
for _ in range(N):
    qp_vmap_jit(params_obj=(Q, c), params_ineq=(G, h))
end = time.time()
print((end - start) / N)
