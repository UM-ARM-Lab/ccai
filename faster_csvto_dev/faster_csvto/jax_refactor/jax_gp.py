import jax
import jax.numpy as jnp

from jax_rbf_kernels import RBFKernel


class GPSurfaceModel:
    """
    """

    def __init__(self, train_x: jnp.array, train_y: jnp.array, length_scale=1.5, output_scale=0.5):
        """

        Args:

        Returns:
        """
        self.train_x = train_x
        self.train_y = train_y.flatten()
        self.kernel = RBFKernel(length_scale=length_scale, output_scale=output_scale)
        self.sigma_sq_noise = 0.01

    def posterior_mean(self, x):
        """Compute the posterior mean and its gradient and hessian.

        Args:
            x:

        Returns:
            Mean function evaluated at x.
            Derivative of mean function evaluated at x.
            Hessian of mean function evaluated at x.
        """
        train_K = self.kernel(self.train_x, self.train_x)
        K = self.kernel(x, self.train_x)

        # TODO: Fix these operations. Avoiding right now to avoid diag_embed, the mean() is not appropriate. Without,
        #  output shapes are (12, 100, 12, 2) and (12, 100, 12, 2, 12, 2). Should be (12, 100, 2) and (12, 100, 2, 2).
        #  May need to reimplement a non-batched version then vmap(grad()) to get the autodiff dimensionality right.
        # grad_K = jax.jacrev(self.kernel)(x, self.train_x).mean(axis=2)
        # hess_K = jax.jacfwd(jax.jacrev(self.kernel))(x, self.train_x).mean(axis=2).mean(axis=3)
        eye = jnp.eye(len(self.train_x))

        # compute [K + sigma_sq I]^-1 y
        tmp = jnp.linalg.solve(train_K + self.sigma_sq_noise * eye, self.train_y.reshape(-1, 1))
        y = K @ tmp
        # grad_y = grad_K.transpose(0, 2, 1) @ tmp
        # hess_y = hess_K.transpose(0, 2, 3, 1) @ tmp
        return y.squeeze(-1)#, grad_y.squeeze(-1), hess_y.squeeze(-1)
