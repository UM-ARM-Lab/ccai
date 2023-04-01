import torch
import numpy as np
from ccai.gp import GPSurfaceModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!


class QuadrotorEnv:

    def __init__(self, surface_data_fname, obstacle_mode=None):
        assert obstacle_mode in [None, 'static', 'dynamic']
        self.env_dims = [-10, 10]
        self.state_dim = 12
        self.dt = 0.1
        self.state = None
        data = np.load(surface_data_fname)
        self.surface_model = GPSurfaceModel(torch.from_numpy(data['xy']).to(dtype=torch.float32),
                                            torch.from_numpy(data['z']).to(dtype=torch.float32))
        # For rendering height
        N = 100
        xs = torch.linspace(-6, 6, steps=N)
        ys = torch.linspace(-6, 6, steps=N)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        test_x = torch.stack((x.flatten(), y.flatten()), dim=1)
        z, _, _, = self.surface_model.posterior_mean(test_x)

        x = test_x[:, 0].reshape(N, N).numpy()
        y = test_x[:, 1].reshape(N, N).numpy()
        z = z.detach().reshape(N, N).numpy()

        self.surface_plotting_vars = [x, y, z]
        self.obstacle_pos = np.array([0.0, 0.0])
        self.obstacle_r = 1
        self.obstacle_mode = obstacle_mode
        self.alpha = 0.5
        self.reset()
        self.ax = None
        self._surf = None
        self._render_pos = None

    def _get_surface_h(self, state):
        xy = torch.from_numpy(state[:2]).reshape(1, 2).to(dtype=torch.float32)
        with torch.no_grad():
            z, _, _ = self.surface_model.posterior_mean(xy)
        return z.numpy().reshape(1).astype(np.float32)

    def reset(self):
        start = np.zeros(self.state_dim)

        # positions
        start[:2] = np.array([-2.5, -2.5]) - 3. * np.random.rand(2)
        start[2] = self._get_surface_h(start)


        # Angles in rad
        start[3:6] = 0.05 * (-np.pi + 2 * np.pi * np.random.rand(3))

        # Initial velocity
        start[6:] = 0.05 * np.random.randn(6)

        # start[9:] *= 5
        self.state = start
        if self.obstacle_mode == 'dynamic':
            self.obstacle_pos = np.array([-1.0, 2.5])
        else:
            self.obstacle_pos = np.array([0.0, 0.0])

    def get_constraint_violation(self):
        surface_h = self._get_surface_h(self.state)
        surface_violation = self.state[2] - surface_h
        if self.obstacle_mode is None:
            return surface_violation

        obstacle_violation = self.obstacle_r ** 2 - np.sum((self.state[:2] - self.obstacle_pos) ** 2)
        obstacle_violation = np.clip(obstacle_violation, a_min=0, a_max=None)
        return np.array([surface_violation.item(), obstacle_violation.item()])

    def step(self, control):

        # Unroll state
        g = -9.81
        m = 1
        Ix, Iy, Iz = 0.5, 0.1, 0.3
        K = 1
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

        # symplectic euler -- update velocities first
        new_xdot = x_dot + x_ddot * self.dt
        new_ydot = y_dot + y_ddot * self.dt
        new_zdot = z_dot + z_ddot * self.dt
        new_p = p + p_dot * self.dt
        new_q = q + q_dot * self.dt
        new_r = r + r_dot * self.dt

        ''' velocities'''
        psi_dot = new_q * sphi / ctheta + new_r * cphi / ctheta
        theta_dot = new_q * cphi - new_r * sphi
        phi_dot = new_p + new_q * sphi * ttheta + new_r * cphi * ttheta

        new_phi = phi + phi_dot * self.dt
        new_theta = theta + theta_dot * self.dt
        new_psi = psi + psi_dot * self.dt
        new_x = x + new_xdot * self.dt
        new_y = y + new_ydot + self.dt
        new_z = z + new_zdot + self.dt

        # dstate = np.stack((x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot,
        #                   x_ddot, y_ddot, z_ddot, p_dot, q_dot, r_dot), axis=-1)

        # self.state = self.state + dstate * self.dt
        self.state = np.stack(
            (new_x, new_y, new_z, new_phi, new_theta, new_psi, new_xdot, new_ydot, new_zdot, new_p, new_q, new_r)
        )
        # self.state[3:6] = normalize_angles(self.state[3:6])

        if self.obstacle_mode == 'dynamic':
            self.obstacle_pos += np.array([0.25, -0.1])

        return self.state, self.get_constraint_violation()

    def _get_surface_colours(self):
        if self.obstacle_mode is None:
            return np.array([1.0, 0.0, 0.0, self.alpha])[None, None, :]

        x, y, z = self.surface_plotting_vars
        in_obs = np.where((x - self.obstacle_pos[0]) ** 2 + (y - self.obstacle_pos[1]) ** 2 < self.obstacle_r ** 2,
                          1, 0)
        c = np.where(in_obs[:, :, None],
                     np.array([1.0, 0.0, 0.0, self.alpha])[None, None, :],
                     np.array([0, 0.0, .7, self.alpha])[None, None, :]
                     )
        return c

    def render_init(self):
        self.ax = plt.axes(projection='3d')

        self._surf = self.ax.plot_surface(self.surface_plotting_vars[0],
                                          self.surface_plotting_vars[1],
                                          self.surface_plotting_vars[2] - 0.1, alpha=self.alpha,
                                          shade=True,
                                          antialiased=True, linewidth=0,
                                          facecolors=self._get_surface_colours(),
                                          rstride=1,
                                          cstride=1)
        self._render_pos = self.ax.scatter(self.state[0], self.state[1], self.state[2], s=100, c='g')
        self.ax.scatter(4, 4, self._get_surface_h(np.array([4, 4])), s=100, c='k')
        self.ax.view_init(60, -50)
        self.ax.axes.set_xlim3d(left=-6, right=6)
        self.ax.axes.set_ylim3d(bottom=-6, top=6)
        self.ax.axes.set_zlim3d(bottom=-3, top=3)
        self.ax.set_xlabel('$x$', fontsize=20)
        self.ax.set_ylabel('$y$', fontsize=20)
        self.ax.set_zlabel('$z$', fontsize=20)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        return self.ax

    def render_update(self):
        self._render_pos._offsets3d = ([self.state[0]], [self.state[1]], [self.state[2]])
        if self.obstacle_mode == 'dynamic':
            self._surf.set_facecolors(self._get_surface_colours()[:-1, :-1].reshape(-1, 4))
