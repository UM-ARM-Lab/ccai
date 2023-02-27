import torch
from ccai.kernels import RBFKernel
from torch.distributions import MultivariateNormal


class GPSurfaceModel:

    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y.flatten()
        self.kernel = RBFKernel(lengthscale=1.5, outputscale=0.25)
        self.sigma_sq_noise = 0.01
        self.training = True

    def prior(self):
        K, grad_K, hess_K = self.kernel(self.train_x, self.train_x)
        return MultivariateNormal(torch.zeros_like(self.train_y), covariance_matrix=K)

    def posterior_mean(self, x):
        """
            Return posterior mean with gradient and hessian
        :param x: Input tensor
        :return:
            y: output tensor - mean function evaluated at x
            grad_y: derivative of mean function wrt x
            hess_y: hessian of mean function wrt x
        """

        train_K, _, _ = self.kernel(self.train_x, self.train_x)
        K, grad_K, hess_K = self.kernel(x, self.train_x)
        eye = torch.eye(len(self.train_x), device=x.device)

        # compute [K + sigma_sq I]^-1 y
        tmp = torch.linalg.solve(train_K + self.sigma_sq_noise * eye, self.train_y.reshape(-1, 1))

        y = K @ tmp
        grad_y = grad_K.permute(0, 2, 1) @ tmp
        hess_y = hess_K.permute(0, 2, 3, 1) @ tmp

        return y.squeeze(-1), grad_y.squeeze(-1), hess_y.squeeze(-1)
