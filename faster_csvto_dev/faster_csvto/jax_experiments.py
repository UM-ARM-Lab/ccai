import jax
import jax.numpy as jnp


def f(x: jnp.array):
    q = jnp.array([[1, 2], [3, 4]])
    return (x.transpose() @ q @ x).squeeze(0).squeeze(0)


if __name__ == '__main__':
    x = jnp.array([[1], [2]], dtype=jnp.float32)
    grad_f_at_x = jax.jacfwd(f)(x)
    hess_f_at_x = jax.jacfwd(jax.jacfwd(f))(x)
    print(grad_f_at_x.shape)
    print(hess_f_at_x.transpose(1, 0, 2, 3).squeeze(-1).shape)