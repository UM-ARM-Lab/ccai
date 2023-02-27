import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!
from ccai.gp import GPSurfaceModel

if __name__ == '__main__':
    lengthscale = 1.5
    outputscale = 1

    N = 10
    xs = torch.linspace(-5, 5, steps=N)
    ys = torch.linspace(-5, 5, steps=N)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    train_x = torch.stack((x.flatten(), y.flatten()), dim=1)
    train_y = torch.randn(len(train_x), 1)
    initial_gp_model = GPSurfaceModel(train_x, train_y)

    # sample from GP prior to get our data
    samples = initial_gp_model.prior().sample()
    # now we will save the data so that we can use it for our quadrotor experiment
    data = {
        'xy': train_x.numpy(),
        'z': samples.numpy()
    }
    np.savez('surface_data.npz', **data)

    x = train_x[:, 0].reshape(N, N)
    y = train_x[:, 1].reshape(N, N)
    z = samples.detach().reshape(N, N)
    ax = plt.axes(projection='3d')
    ax.plot_surface(x.numpy(), y.numpy(), z.numpy())
    ax.axes.set_zlim3d(bottom=-5, top=5)
    plt.show()

    # Now we are going to use this data to fit a GP (we will just use the same kernel hyperparams rather than train them)
    surface_gp = GPSurfaceModel(train_x, samples)
    N = 100
    xs = torch.linspace(-5, 5, steps=N)
    ys = torch.linspace(-5, 5, steps=N)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    test_x = torch.stack((x.flatten(), y.flatten()), dim=1)
    samples, grad_samples, hess_samples = surface_gp.posterior_mean(test_x)

    x = test_x[:, 0].reshape(N, N)
    y = test_x[:, 1].reshape(N, N)
    z = samples.detach().reshape(N, N)
    ax = plt.axes(projection='3d')
    ax.plot_surface(x.numpy(), y.numpy(), z.numpy())
    ax.axes.set_zlim3d(bottom=-5, top=5)
    plt.show()
