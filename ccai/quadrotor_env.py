import torch
import numpy as np
from ccai.gp import GPSurfaceModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!


class QuadrotorEnv:

    def __init__(self, surface_data_fname):
        self.env_dims = [-10, 10]
        self.state_dim = 12
        self.dt = 0.1
        self.state = None
        data = np.load(surface_data_fname)
        self.surface_model = GPSurfaceModel(torch.from_numpy(data['xy']).to(dtype=torch.float32),
                                            torch.from_numpy(data['z']).to(dtype=torch.float32))
        # For rendering height
        N = 100
        xs = torch.linspace(-5, 5, steps=N)
        ys = torch.linspace(-5, 5, steps=N)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        test_x = torch.stack((x.flatten(), y.flatten()), dim=1)
        z, _, _, = self.surface_model.posterior_mean(test_x)

        x = test_x[:, 0].reshape(N, N).numpy()
        y = test_x[:, 1].reshape(N, N).numpy()
        z = z.detach().reshape(N, N).numpy()

        self.surface_plotting_vars = [x, y, z]
        self.reset()

    def _get_surface_h(self, state):
        xy = torch.from_numpy(state[:2]).reshape(1, 2).to(dtype=torch.float32)
        with torch.no_grad():
            z, _, _ = self.surface_model.posterior_mean(xy)
        return z.numpy().reshape(1).astype(np.float64)

    def reset(self):
        start = np.zeros(self.state_dim)

        # positions
        start[:2] = np.array([-4.5, -4.5]) + 0.1 * np.random.randn(2)
        start[2] = self._get_surface_h(start)

        # Angles in rad
        # Heading angle can be between 0 and pi/2
        start[5] = np.pi / 2 * np.random.rand()

        # Other two must be restriced by a lot
        start[3:5] = 0.05*(-np.pi + 2 * np.pi * np.random.rand(2))

        # Initial velocity zero for now - may change in future
        # start[6:] = np.random.randn(6)
        # start[9:] *= 5
        self.state = start

    def step(self, control):
        # Unroll state
        g = -9.81
        m = 1
        Ix, Iy, Iz = 0.5, 0.1, 0.3
        K = 5
        x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r = self.state

        u1, u2, u3, u4 = control

        # Trigonometric fcns on all the angles needed for dynamics

        cphi = np.cos(phi)
        ctheta = np.cos(theta)
        cpsi = np.cos(psi)
        sphi = np.sin(phi)
        stheta = np.sin(theta)
        spsi = np.sin(psi)
        ttheta = np.tan(theta)

        ''' accelerations first '''
        x_ddot = -(sphi * spsi + cpsi * cphi * stheta) * K * u1 / m
        y_ddot = - (cpsi * sphi - cphi * spsi * stheta) * K * u1 / m
        z_ddot = g - (cphi * ctheta) * K * u1 / m

        p_dot = ((Iy - Iz) * q * r + K * u2) / Ix
        q_dot = ((Iz - Ix) * p * r + K * u3) / Iy
        r_dot = ((Ix - Iy) * p * q + K * u4) / Iz

        ''' velocities'''
        psi_dot = q * sphi / ctheta + r * cphi / ctheta
        theta_dot = q * cphi - r * sphi
        phi_dot = p + q * sphi * ttheta + r * cphi * ttheta

        dstate = np.stack((x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot,
                           x_ddot, y_ddot, z_ddot, p_dot, q_dot, r_dot), axis=-1)

        self.state = self.state + dstate * self.dt
        # self.state[3:6] = normalize_angles(self.state[3:6])

        return self.state

    def render(self):
        ax = plt.axes(projection='3d')
        ax.plot_surface(self.surface_plotting_vars[0],
                        self.surface_plotting_vars[1],
                        self.surface_plotting_vars[2] - 1, alpha=0.5)
        ax.axes.set_zlim3d(bottom=-5, top=5)
        ax.scatter(self.state[0], self.state[1], self.state[2], s=100, c='g')
        ax.scatter(0, 0, 0, s=100, c='k')

        return ax