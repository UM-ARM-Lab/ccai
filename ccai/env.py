import torch
import numpy as np
import open3d as o3d
import pytorch_volumetric as pv
from skimage.measure import marching_cubes_lewiner


def compute_hess_sdf(sdf, pts, h=1e-3):
    """

    :param sdf: sdf fun Bx3 -> B
    :param pts: B x 3 query points to evaluate gradient at
    :param h: stepsize for numerical gradient
    :return: B x 3 x 3 hessian
    """
    B = pts.shape[0]
    sdf_hess = np.zeros((B, 3, 3), dtype=np.float32)
    zeros = np.zeros((B, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            if i < j:
                continue

            if i == j:
                # fill diagonal
                d = zeros.copy()
                d[:, i] = h
                sdf_hess[:, i, i] = (sdf(pts + d) - 2 * sdf(pts) + sdf(pts - d)) / (h ** 2)

            else:
                # fill off-diagonal
                d = zeros.copy()
                d[:, i] = h
                d[:, j] = h
                dmix = zeros.copy()
                dmix[:, i] = h
                dmix[:, j] = -h
                sdf_hess[:, i, j] = (sdf(pts + d) - sdf(pts + dmix) - sdf(pts - dmix) + sdf(pts - d)) / (4 * h ** 2)
                sdf_hess[:, j, i] = sdf_hess[:, i, j]
    return sdf_hess


def compute_hess_sdf2(sdf, pts, h=1e-3):
    # compute hessian by numerically differentiating the gradient
    B = pts.shape[0]
    sdf_hess = np.zeros((B, 3, 3), dtype=float)
    zeros = np.zeros((B, 3), dtype=float)

    for i in range(3):
        d = zeros.copy()
        d[:, i] = h
        sdf_hess[:, i, :] = (compute_grad_sdf(sdf, pts + d) - compute_grad_sdf(sdf, pts - d)) / (2 * h)
    return sdf_hess


def compute_grad_sdf(sdf, pts, h=1e-3):
    """

    :param sdf: sdf fun Bx3 -> B
    :param pts: B x 3 query points to evaluate gradient at
    :param h: stepsize for numerical gradient
    :return: B x 3 gradient
    """
    B = pts.shape[0]
    zeros = np.zeros((B, 3))
    grad_sdf = np.zeros((B, 3))
    for i in range(3):
        d = zeros.copy()
        d[:, i] = h
        grad_sdf[:, i] = (sdf(pts + d) - sdf(pts - d)) / (2 * h)

    return grad_sdf


def generate_random_sphere_sdf(max_spheres, min_spheres, max_rad, min_rad, range_per_dim, sdf_size, plot=False):
    """

    :param max_spheres:
    :param min_spheres:
    :param max_rad:
    :param min_rad:
    :param range_per_dim: (3 x 2) range per dim (x, y, z)
    :param sdf_size: N for N x N x N sdf grid size
    :return:
    """
    num_obstacles = np.random.randint(low=min_spheres, high=max_spheres)
    obstacle_positions = np.random.uniform(low=range_per_dim[:, 0], high=range_per_dim[:, 1], size=(num_obstacles, 3))
    obstacle_radii = np.random.uniform(low=min_rad, high=max_rad, size=(num_obstacles))

    xx, yy, zz = np.meshgrid(np.linspace(range_per_dim[0, 0], range_per_dim[0, 1], sdf_size),
                             np.linspace(range_per_dim[1, 0], range_per_dim[1, 1], sdf_size),
                             np.linspace(range_per_dim[2, 0], range_per_dim[2, 1], sdf_size), indexing='ij')

    pts = np.stack((xx, yy, zz), axis=-1)

    def sdf(x):
        """

        :param x: B x 3 query points
        :return: B sdf values
        """
        distance_to_centres = x[None] - obstacle_positions[:, None, :]
        sdf_vals = np.linalg.norm(distance_to_centres, axis=-1) - obstacle_radii[:, None]
        return np.min(sdf_vals, axis=0)

    sdf_val = sdf(pts.reshape((-1, 3))).reshape((sdf_size, sdf_size, sdf_size))
    grad_sdf = compute_grad_sdf(sdf, pts.reshape((-1, 3))).reshape((sdf_size, sdf_size, sdf_size, 3))
    hess_sdf = compute_hess_sdf(sdf, pts.reshape((-1, 3))).reshape((sdf_size, sdf_size, sdf_size, 3, 3))

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib
        norm = matplotlib.colors.Normalize(vmin=np.min(sdf_val) - 0.1, vmax=np.max(sdf_val))
        cmap = 'Greys_r'
        ax = plt.gca()
        x = xx[:, :, 0]
        y = yy[:, :, 0]
        v = sdf_val[:, :, 0]
        cset1 = ax.contourf(x, y, v, norm=norm, cmap=cmap)
        cset2 = ax.contour(x, y, v, colors='k', levels=[0], linestyles='dashed')
        sdf_grad_uv = grad_sdf[:, :, 0]
        ax.quiver(x, y, sdf_grad_uv[:, :, 0],
                  sdf_grad_uv[:, :, 1], color='g')
        plt.show()
    return sdf_val, grad_sdf, hess_sdf


def convert_sdf_to_mesh(sdf_grid, range_per_dim, device='cpu'):
    sdf_size = sdf_grid.shape[0]
    occupancy = np.where(sdf_grid > 0, 1, 0)
    # occupancy = np.zeros_like(occupancy)
    spacing = (range_per_dim[:, 1] - range_per_dim[:, 0]) / sdf_size
    # add padding to the occupancy grid
    pad_width = 1
    occupancy = np.pad(occupancy, pad_width, mode='constant', constant_values=1)
    verts, faces, norms, vals = marching_cubes_lewiner(occupancy, spacing=spacing)
    origin = range_per_dim[:, 0]
    # scale verts
    scale = 1.0 / (1.0 - spacing)
    verts = np.multiply(verts, scale)
    verts = np.subtract(np.add(verts, origin), spacing * pad_width)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh('tmp.obj', mesh)

    # scale = 1
    sdf_from_mesh = pv.MeshSDF(pv.MeshObjectFactory('tmp.obj', scale=scale))
    mesh_range_per_dim = range_per_dim.copy()
    mesh_range_per_dim[:, 0] -= 0.5
    mesh_range_per_dim[:, 1] += 0.5
    cache_sdf_from_mesh = pv.CachedSDF('sphere_world', resolution=spacing[0],
                                       range_per_dim=mesh_range_per_dim, gt_sdf=sdf_from_mesh,
                                       cache_sdf_hessian=True,
                                       clean_cache=True,
                                       device=device)
    return cache_sdf_from_mesh


def generate_random_sphere_world(max_spheres, min_spheres, max_rad, min_rad, range_per_dim, sdf_size, device='cpu',
                                 plot=False):
    sdf_grid, _, _ = generate_random_sphere_sdf(max_spheres, min_spheres, max_rad, min_rad, range_per_dim, sdf_size,
                                                plot)

    cache_sdf_from_mesh = convert_sdf_to_mesh(sdf_grid, range_per_dim, device=device)
    return sdf_grid, cache_sdf_from_mesh


if __name__ == "__main__":
    np.random.seed(123)
    sdf_size = 64
    range_per_dim = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])
    sdf_grid, sdf_grad, sdf_hess = generate_random_sphere_sdf(2,
                                                              1,
                                                              0.6,
                                                              0.4,
                                                              range_per_dim,
                                                              sdf_size)
    sdf_mesh = convert_sdf_to_mesh(sdf_grid, range_per_dim)

    # reconstruct sdf_grid from mesh
    xx, yy, zz = np.meshgrid(np.linspace(range_per_dim[0, 0], range_per_dim[0, 1], sdf_size),
                             np.linspace(range_per_dim[1, 0], range_per_dim[1, 1], sdf_size),
                             np.linspace(range_per_dim[2, 0], range_per_dim[2, 1], sdf_size), indexing='ij')

    pts = torch.from_numpy(np.stack((xx, yy, zz), axis=-1)).float()
