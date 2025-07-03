import numpy as np

np.float = np.float64

from isaac_victor_envs.utils import get_assets_dir
import torch
import time
import yaml
import pathlib
from functools import partial
from torch.func import vmap, jacrev, hessian, jacfwd

from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
from ccai.kernels import rbf_kernel, structured_rbf_kernel

from ccai.problem import ConstrainedSVGDProblem, UnconstrainedPenaltyProblem, IpoptProblem
from ccai.mpc import Constrained_SVGD_MPC, MPPI, SVMPC, IpoptMPC

import time
import allegro_optimized_wrapper as pk

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
asset_dir = get_assets_dir()
asset = asset_dir + '/victor/victor_mallet.urdf'
ee_name = 'victor_left_arm_striker_mallet_tip'
chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name)
chain_cc = pk.build_chain_from_urdf(open(asset).read())
collision_check_links = [
    'victor_left_arm_link_2',
    'victor_left_arm_link_3',
    'victor_left_arm_link_4',
    'victor_left_arm_link_5',
    'victor_left_arm_link_6',
    'victor_left_arm_link_7',
    'victor_left_arm_striker_base',
    'victor_left_arm_striker_mallet'
]


def update_chain(device):
    chain.to(device=device)
    chain_cc.to(device=device)


def create_grid_from_sdf(sdf, device):
    range_per_dim = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])
    sdf_size = 64
    xx, yy, zz = np.meshgrid(np.linspace(range_per_dim[0, 0], range_per_dim[0, 1], sdf_size),
                             np.linspace(range_per_dim[1, 0], range_per_dim[1, 1], sdf_size),
                             np.linspace(range_per_dim[2, 0], range_per_dim[2, 1], sdf_size), indexing='ij')

    pts = np.stack((xx, yy, zz), axis=-1)
    # add origin to pts
    pts = pts + np.array([0.75, 0.25, 1.0]).reshape((1, 1, 1, 3))
    print(pts.shape)
    pts = torch.from_numpy(pts).to(device=device).float()

    vals, _ = sdf(pts)
    vals = vals.reshape(sdf_size, sdf_size, sdf_size)
    return vals


class VictorTableProblem(ConstrainedSVGDProblem):

    def __init__(self, start, goal, T, obstacle_poses, table_height, obstacle_type, device='cuda:0'):
        super().__init__(start, goal, T, device)
        if obstacle_poses is None:
            self.dz = 0
        else:
            self.dz = 1
        self.squared_slack = False
        self.dh = self.dz * T
        if table_height is None:
            self.dg = 0
        else:
            self.dg = 2 * T  # + 2

        self.dx = 7
        self.du = 0
        self.dt = 0.1
        self.T = T
        self.start = start
        self.goal = goal
        # self.K = rbf_kernel
        self.K = structured_rbf_kernel

        self.grad_kernel = jacrev(rbf_kernel, argnums=0)
        self.alpha = 10

        if table_height is not None:
            self._equality_constraints = EndEffectorConstraint(
                chain, partial(ee_equality_constraint, height=table_height)
            )
        else:
            self._equality_constraints = None

        # if obstacle_centres is not None:
        #    self._inequality_constraints = EndEffectorConstraint(
        #        chain, partial(ee_inequality_constraint, centres=obstacle_centres, rads=obstacle_rads)
        #    )
        # else:
        #    self._inequality_constraints = None

        if obstacle_type != 'none':
            import pytorch_volumetric as pv
            sdf = pv.RobotSDF(chain_cc, path_prefix=asset_dir + '/victor')
        if obstacle_type == 'tabletop_ycb':
            # Load SDFs
            mug1_sdf = pv.MeshSDF(pv.MeshObjectFactory(f"{asset_dir}/obstacles/mug/mug.obj"))
            mug2_sdf = pv.MeshSDF(pv.MeshObjectFactory(f"{asset_dir}/obstacles/mug/mug.obj"))
            pitcher_sdf = pv.MeshSDF(pv.MeshObjectFactory(f"{asset_dir}/obstacles/pitcher/pitcher.obj"))
            # Get transforms
            obstacle_transforms = torch.stack([
                obstacle_poses['mug1'],
                obstacle_poses['mug2'],
                obstacle_poses['pitcher']
            ]).reshape(-1, 4, 4).inverse().to(device=self.device)
            # Compose SDFs
            scene_sdf = pv.ComposedSDF([mug1_sdf, mug2_sdf, pitcher_sdf],
                                       pk.Transform3d(matrix=obstacle_transforms))

            scene_sdf_cached = pv.CachedSDF('scene', resolution=0.01, gt_sdf=scene_sdf, range_per_dim=np.array([
                [0.2, 1.2],
                [-0.1, 1.2],
                [0.6, 1.5]
            ]), device=self.device, cache_sdf_hessian=True, clean_cache=False)

            self.robot_scene = pv.RobotScene(sdf, scene_sdf_cached,
                                             pk.Transform3d(matrix=torch.eye(4,
                                                                             device=self.device
                                                                             )
                                                            ),
                                             collision_check_links=collision_check_links
                                             )
        elif obstacle_type == 'tabletop_ycb_2':
            # Load SDFs
            mustard_bottle_sdf = pv.MeshSDF(pv.MeshObjectFactory(f"{asset_dir}/obstacles/mug/mustard_bottle.obj"))
            cracker_box_sdf = pv.MeshSDF(pv.MeshObjectFactory(f"{asset_dir}/obstacles/mug/cracker_box.obj"))
            pitcher_sdf = pv.MeshSDF(pv.MeshObjectFactory(f"{asset_dir}/obstacles/pitcher/pitcher.obj"))
            # Get transforms
            obstacle_transforms = torch.stack([
                obstacle_poses['mustard_bottle'],
                obstacle_poses['cracker_box'],
                obstacle_poses['pitcher']
            ]).reshape(-1, 4, 4).inverse()
            # Compose SDFs
            scene_sdf = pv.ComposedSDF([mustard_bottle_sdf, cracker_box_sdf, pitcher_sdf],
                                       pk.Transform3d(matrix=obstacle_transforms))

            # convert into sdf grid
            grid = create_grid_from_sdf(scene_sdf, self.device)
            np.savez('../tabletop_ycb_2.npz', sdf_grid=grid.cpu().numpy())
            scene_sdf_cached = pv.CachedSDF('scene', resolution=0.01, gt_sdf=scene_sdf, range_per_dim=np.array([
                [0.4, 1.0],
                [-0.3, 1.0],
                [0.6, 1.5]
            ]), device=self.device, cache_sdf_hessian=False, clean_cache=False)

            self.robot_scene = pv.RobotScene(sdf, scene_sdf_cached,
                                             pk.Transform3d(matrix=torch.eye(4,
                                                                             device=self.device
                                                                             )
                                                            ),
                                             collision_check_links=collision_check_links
                                             )
        elif 'floating_spheres' in obstacle_type:
            T = pk.Transform3d(matrix=torch.tensor([[[1., 0., 0., 0.75],
                                                     [0., 1., 0., 0.25],
                                                     [0., 0., 1., 1.],
                                                     [0., 0., 0., 1.]]], device=device))
            # Get transforms
            # Compose SDFs
            range_per_dim = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])
            spacing = (range_per_dim[:, 1] - range_per_dim[:, 0]) / 64
            mesh_range_per_dim = range_per_dim.copy()
            mesh_range_per_dim[:, 0] -= 0.5
            mesh_range_per_dim[:, 1] += 0.5
            scale = 1.0 / (1.0 - spacing)
            scene_sdf = pv.MeshSDF(pv.MeshObjectFactory(f"{asset_dir}/obstacles/{obstacle_type}/{obstacle_type}.obj",
                                                        scale=scale))

            scene_sdf_cached = pv.CachedSDF('sphere_world', resolution=spacing[0],
                                            range_per_dim=mesh_range_per_dim, gt_sdf=scene_sdf,
                                            cache_sdf_hessian=True,
                                            clean_cache=True,
                                            device=self.device)

            self.robot_scene = pv.RobotScene(sdf, scene_sdf_cached, T, collision_check_links=collision_check_links)
        else:
            self._inequality_constraints = None

        self._terminal_constraints = EndEffectorConstraint(
            chain, partial(ee_terminal_constraint, goal=self.goal)
        )

        self.x_max = torch.tensor([2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05])
        self.x_min = -self.x_max

        self.dynamics_constraint = vmap(self._dynamics_constraint)
        self.grad_dynamics_constraint = vmap(jacrev(self._dynamics_constraint))
        self.hess_dynamics_constraint = vmap(hessian(self._dynamics_constraint))

        self.cost = vmap(partial(cost, start=self.start))
        self.grad_cost = vmap(jacrev(partial(cost, start=self.start)))
        self.hess_cost = vmap(hessian(partial(cost, start=self.start)))
        self.right_arm = torch.tensor([1.144, -1.189, 0.590, 0.292, 0.296, 0.265, -0.809], device=self.device)

        # self.robot_scene.visualize_robot(torch.zeros(1, 14, device=self.device))

    def _inequality_constraints(self, x, compute_grads=True):
        N = x.shape[0]

        q = torch.cat((x, self.right_arm.repeat(N, 1)), dim=1)

        ret_scene = self.robot_scene.scene_collision_check(q,
                                                           compute_gradient=compute_grads,
                                                           compute_hessian=compute_grads)

        h = -ret_scene.get('sdf') + 0.01
        grad_h = ret_scene.get('grad_sdf', None)
        hess_h = ret_scene.get('hess_sdf', None)

        if grad_h is not None:
            grad_h = -grad_h[:, :7].unsqueeze(1)
            if hess_h is not None:
                hess_h = -hess_h[:, :7, :7].unsqueeze(1)
            else:
                hess_h = torch.zeros(x.shape[0], self.dz, self.dx, self.dx, device=self.device)

        return h, grad_h, hess_h

    def dynamics(self, x, u):
        N = x.shape[0]
        return x + self.dt * u

    def _dynamics_constraint(self, trajectory):
        x = trajectory[:, :self.dx]
        u = trajectory[:, self.dx:]
        current_x = torch.cat((self.start.reshape(1, self.dx), x[:-1]), dim=0)
        next_x = x
        pred_next_x = self.dynamics(current_x, u)
        return torch.reshape(pred_next_x - next_x, (-1,))

    def _objective(self, x):
        x = x[:, :, :self.dx]
        N = x.shape[0]
        term_g, term_grad_g, term_hess_g = self._terminal_constraints.eval(x.reshape(-1, self.dx))
        # term_grad_g_extended = torch.zeros(N, self.T, self.dx, device=self.device)
        # term_hess_g_extended = torch.zeros(N, self.T, self.dx, self.T, self.dx, device=self.device)
        # term_grad_g_extended[:, -1, :] = term_grad_g.reshape(N, -1)
        # term_hess_g_extended[:, -1, :, -1, :] = term_hess_g.reshape(N, self.dx, self.dx)
        # J, grad_J, hess_J = self.cost(x)
        J, grad_J, hess_J = self.cost(x), self.grad_cost(x), self.hess_cost(x)

        if term_grad_g is not None:
            term_grad_g_extended = term_grad_g.reshape(N, self.T, self.dx)
            term_hess_g_extended = term_hess_g.reshape(N, self.T, self.dx, self.dx).permute(0, 2, 3, 1)
            term_grad_g_extended[:, -1] *= 10
            term_hess_g_extended = torch.diag_embed(term_hess_g_extended).permute(0, 3, 1, 4, 2)
            term_hess_g_extended[:, -1, :, -1] *= 10

            J = J.reshape(-1) + term_g.reshape(N, self.T).sum(dim=1)
            grad_J = grad_J.reshape(N, self.T, -1) + term_grad_g_extended
            hess_J = hess_J.reshape(N, self.T, self.dx, self.T, self.dx) + term_hess_g_extended

        # add auxiliary grad from diffusion model
        if False:  # self.flow_model is not None:
            cparams = self.constr_params.expand(N, -1, -1)
            start_normalized = (self.start - self.flow_model.x_mean[:self.dx]) / self.flow_model.x_std[:self.dx]
            x_normalized = (x - self.flow_model.x_mean[:self.dx]) / self.flow_model.x_std[:self.dx]
            start_normalized = start_normalized.expand(N, -1)
            t = torch.tensor([0], device=self.device, dtype=torch.int64).expand(N)
            aux_grad = self.flow_model.grad(x_normalized.reshape(N, self.T, self.dx), t=t,
                                            start=start_normalized,
                                            goal=self.goal.expand(N, -1),
                                            constraints=cparams)
            aux_grad = aux_grad.reshape(N, self.T, self.dx)
            # unnormalize
            aux_grad = aux_grad * self.flow_model.x_std[:self.dx]
            grad_J = grad_J + aux_grad

        N = x.shape[0]
        return (self.alpha * J.reshape(N),
                self.alpha * grad_J.reshape(N, -1),
                self.alpha * hess_J.reshape(N, self.T * self.dx, self.T * self.dx))

    def _con_eq(self, x, compute_grads=True):
        x = x[:, :, :self.dx]
        N = x.shape[0]
        if self._equality_constraints is None:
            return None, None, None

        g, grad_g, hess_g = self._equality_constraints.eval(x.reshape(-1, self.dx), compute_grads)
        # term_g, term_grad_g, term_hess_g = self._terminal_constraints.eval(x[:, -1])

        g = g.reshape(N, -1)
        # combine terminal constraint with running constraints
        # g = torch.cat((g, term_g), dim=1)

        N = x.shape[0]
        if not compute_grads:
            return g, None, None
            # Expand gradient to include time dimensions

        grad_g = grad_g.reshape(N, self.T, -1, self.dx).permute(0, 2, 3, 1)
        grad_g = torch.diag_embed(grad_g)  # (N, n_constraints, dx + du, T, T)
        grad_g = grad_g.permute(0, 3, 1, 4, 2).reshape(N, -1, self.T * (self.dx))

        # Now do hessian
        hess_g = hess_g.reshape(N, self.T, -1, self.dx, self.dx).permute(0, 2, 3, 4, 1)
        hess_g = torch.diag_embed(torch.diag_embed(hess_g))  # (N, n_constraints, dx + du, dx + du, T, T, T)
        hess_g = hess_g.permute(0, 4, 1, 5, 2, 6, 3).reshape(N, -1,
                                                             self.T * (self.dx),
                                                             self.T * (self.dx))

        return g, grad_g, hess_g

        # now need to get gradients and hessian for terminal constraint
        term_grad_g_extended = torch.zeros(N, term_g.shape[1], self.T, self.dx + self.du,
                                           device=self.device)
        term_grad_g_extended[:, :, -1, :] = term_grad_g
        term_grad_g_extended = term_grad_g_extended.reshape(N, -1, self.T * (self.dx + self.du))
        term_hess_g_extended = torch.zeros(N, term_g.shape[1], self.T, self.dx + self.du,
                                           self.T, self.dx + self.du, device=self.device)
        term_hess_g_extended[:, :, -1, :, -1, :] = term_hess_g
        term_hess_g_extended = term_hess_g_extended.reshape(N, -1, self.T * (self.dx + self.du),
                                                            self.T * (self.dx + self.du))

        # Combine gradients and hessians
        grad_g = torch.cat((grad_g, term_grad_g_extended), dim=1)
        hess_g = torch.cat((hess_g, term_hess_g_extended), dim=1)

        return g, grad_g, hess_g

    def _con_ineq(self, x, compute_grads=True):
        x = x[:, :, :self.dx]

        # return None, None, None
        N = x.shape[0]
        if self._inequality_constraints is None:
            return None, None, None

        h, grad_h, hess_h = self._inequality_constraints(x.reshape(-1, self.dx), compute_grads)

        # Consider time as another batch, need to reshape
        h = h.reshape(N, self.T, -1).reshape(N, -1)

        if not compute_grads:
            return h, None, None

        grad_h = grad_h.reshape(N, self.T, -1, self.dx).permute(0, 2, 3, 1)
        grad_h = torch.diag_embed(grad_h)  # (N, n_constraints, dx + du, T, T)
        grad_h = grad_h.permute(0, 3, 1, 4, 2).reshape(N, -1, self.T * (self.dx))

        # Now do hessian
        hess_h = hess_h.reshape(N, self.T, -1, self.dx, self.dx).permute(0, 2, 3, 4, 1)
        hess_h = torch.diag_embed(torch.diag_embed(hess_h))  # (N, n_constraints, dx + du, dx + du, T, T, T)
        hess_h = hess_h.permute(0, 4, 1, 5, 2, 6, 3).reshape(N, self.dh,
                                                             self.T * (self.dx),
                                                             self.T * (self.dx))

        return h, grad_h, hess_h

    def eval(self, augmented_trajectory):
        N = augmented_trajectory.shape[0]
        augmented_trajectory = augmented_trajectory.clone().reshape(N, self.T, -1)
        x = augmented_trajectory[:, :, :self.dx + self.du]

        J, grad_J, hess_J = self._objective(x)
        hess_J = hess_J + 0.1 * torch.eye(self.T * (self.dx + self.du), device=self.device).unsqueeze(0)
        hess_J = None
        grad_J = torch.cat((grad_J.reshape(N, self.T, -1),
                            torch.zeros(N, self.T, self.dz, device=x.device)), dim=2).reshape(N, -1)

        Xk = x.reshape(N, self.T, -1)
        K = self.K(Xk, Xk, None)  # hess_J.mean(dim=0))
        grad_K = -self.grad_kernel(Xk, Xk, None)  # @hess_J.mean(dim=0))
        grad_K = grad_K.reshape(N, N, N, self.T * (self.dx + self.du))
        grad_K = torch.einsum('nmmi->nmi', grad_K)
        grad_K = torch.cat((grad_K.reshape(N, N, self.T, self.dx + self.du),
                            torch.zeros(N, N, self.T, self.dz, device=x.device)), dim=-1)
        grad_K = grad_K.reshape(N, N, -1)
        G, dG, hessG = self.combined_constraints(augmented_trajectory)

        if hess_J is not None:
            hess_J_ext = torch.zeros(N, self.T, self.dx + self.du + self.dz, self.T, self.dx + self.du + self.dz,
                                     device=x.device)
            hess_J_ext[:, :, :self.dx + self.du, :, :self.dx + self.du] = hess_J.reshape(N, self.T, self.dx + self.du,
                                                                                         self.T, self.dx + self.du)
            hess_J = hess_J_ext.reshape(N, self.T * (self.dx + self.du + self.dz),
                                        self.T * (self.dx + self.du + self.dz))

        ## augment the objective to make the hessian scale better
        ## no lagrange multiplier,  TODO add g^2(x) to objective
        # grad_J_augmented = grad_J + 2 * (G.reshape(N, 1, -1) @ dG).reshape(N, -1)
        # hess_J_augmented = hess_J + 2 * (torch.sum(G.reshape(N, -1, 1, 1) * hessG, dim=1) + dG.permute(0, 2, 1) @ dG)
        # grad_J = grad_J_augmented
        # hess_J = hess_J_augmented.mean(dim=0)

        print(G.abs().max(), G.abs().mean(), J.mean())
        return grad_J.detach(), hess_J, K.detach(), grad_K.detach(), G.detach(), dG.detach(), hessG.detach()

    def update(self, start, goal=None, T=None):
        self.start = start

        # update functions that require start
        self.cost = vmap(partial(cost, start=self.start))
        self.grad_cost = vmap(jacrev(partial(cost, start=self.start)))
        self.hess_cost = vmap(hessian(partial(cost, start=self.start)))

        if goal is not None:
            self.goal = goal

            self._terminal_constraints = EndEffectorConstraint(
                chain, partial(ee_terminal_constraint, goal=self.goal)
            )

        if T is not None:
            self.T = T
            self.dh = self.dz * T
            self.dg = 2 * T

    def get_initial_xu(self, N):

        u = torch.randn(N, self.T, 7, device=self.device)
        x = [self.start.reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            x.append(self.dynamics(x[-1], u[:, t]))

        # particles = torch.cumsum(particles, dim=1) + self.start.reshape(1, 1, self.dx)
        x = torch.stack(x[1:], dim=1)
        xu = torch.cat((x, u), dim=2)
        return x


class VictorTableIpoptProblem(VictorTableProblem, IpoptProblem):

    def __init__(self, start, goal, T, **kwargs):
        super().__init__(start, goal, T, device='cpu', **kwargs)


class VictorTableUnconstrainedProblem(VictorTableProblem, UnconstrainedPenaltyProblem):

    def __init__(self, start, goal, T, device, penalty, **kwargs):
        super().__init__(start, goal, T, device=device, **kwargs)
        self.penalty = penalty
        self.dt = 0.1
        self.du = 7
        self.x_min = torch.cat((self.x_min, -torch.ones(7)))
        self.x_max = torch.cat((self.x_max, torch.ones(7)))


def cost(x, start):
    x = torch.cat((start.reshape(1, 7), x[:, :7]), dim=0)
    weight = torch.tensor([
        0.2, 0.25, 0.4, 0.4, 0.6, 0.75, 1.0], device=x.device, dtype=torch.float32)
    weight = 1.0 / weight
    diff = x[1:] - x[:-1]
    weighted_diff = diff.reshape(-1, 1, 7) @ torch.diag(weight).unsqueeze(0) @ diff.reshape(-1, 7, 1)
    return 10 * torch.sum(weighted_diff)


def obstacle_constraint(x, centres, rads):
    xy = x[:2]
    constr1 = rads[0] ** 2 - torch.sum((xy - centres[0]) ** 2)
    # return constr1.reshape(-1)
    constr2 = rads[1] ** 2 - torch.sum((xy - centres[1]) ** 2)

    # if we were to do an SDF style thing, we would look at which was closest
    # return torch.max(constr1, constr2).reshape(-1)
    return torch.stack((constr1, constr2), dim=0)


class EndEffectorConstraint:

    def __init__(self, chain, ee_constraint_function):
        self.chain = chain
        self._fn = partial(ee_constraint_function)
        self.ee_constraint_fn = vmap(ee_constraint_function)

        self._grad_fn = jacrev(ee_constraint_function, argnums=(0, 1))

        self.grad_constraint = vmap(self._grad_g)
        self.hess_constraint = vmap(jacfwd(self._grad_g, argnums=(0, 1)))

        self._J, self._H, self._dH = None, None, None

        # need to get the end-effector indices
        self.ee_idx = self.chain.frame_to_idx[ee_name]

    def _grad_g(self, p, mat):
        dp, dmat = self._grad_fn(p, mat)
        dmat = dmat @ mat.reshape(1, 3, 3).permute(0, 2, 1)

        # project dmat to be skew-symmetric
        dmat = 0.5 * (dmat - dmat.permute(0, 2, 1))

        omega1 = torch.stack((-dmat[:, 1, 2], dmat[:, 0, 2], -dmat[:, 0, 1]), dim=-1)
        omega2 = torch.stack((dmat[:, 2, 1], -dmat[:, 2, 0], dmat[:, 1, 0]), dim=-1)
        omega = 0.5*(omega1 + omega2)

        return dp, omega

    def eval(self, q, compute_grads=True):
        """

        :param q: torch.Tensor of shape (N, 7) containing set of robot joint config
        :return g: constraint values
        :return Dg: constraint gradient
        :return DDg: constraint hessian
        """

        T = q.shape[0]

        # robot joint configuration
        joint_config = q[:, :7]

        # Get end effector pose
        m = self.chain.forward_kinematics(joint_config).get_matrix()
        p, mat = m[:, :3, 3], m[:, :3, :3]

        # Compute constraints
        constraints = self.ee_constraint_fn(p, mat)

        if not compute_grads:
            return constraints, None, None
        # Compute first and second derivatives of constraints wrt end effector pose
        n_constraints = constraints.shape[1]

        # This is quite complex, but the constraint function takes as input a rotation matrix
        # this means that the gradient and hessian we get from autograd are wrt to parameters of a rotation matrix
        # We need to transform this into something akin to an angular velocity in order to use the robot jacobian
        # to compute derivative and hessian wrt joint config
        # Note: we could use autograd for the whole pipeline but computing the manipulator Jacobian and Hessian
        # manually is much faster than using autograd
        dp, omega = self.grad_constraint(p, mat)
        ddp, domega = self.hess_constraint(p, mat)

        ddp, dp_dmat = ddp
        domega_dp, domega = domega
        dp_omega = domega_dp

        tmp = domega @ mat.reshape(-1, 1, 1, 3, 3).permute(0, 1, 2, 4, 3)
        domega1 = torch.stack((-tmp[:, :, :, 1, 2], tmp[:, :, :, 0, 2], -tmp[:, :, :, 0, 1]), dim=-1)
        domega2 = torch.stack((tmp[:, :, :, 2, 1], -tmp[:, :, :, 2, 0], tmp[:, :, :, 1, 0]), dim=-1)
        domega = (domega1 + domega2)

        # Finally have computed derivative of constraint wrt pose as a (N, num_constraints, 6) tensor
        dpose = torch.cat((dp, omega), dim=-1)

        # cache computation for later
        # self._J, self._H, self._dH = self.chain.jacobian_and_hessian_and_dhessian(joint_config)
        link_indices = torch.ones(T, device=q.device, dtype=torch.long) * self.ee_idx
        self._J, self._H = self.chain.jacobian_and_hessian(joint_config, link_indices=link_indices)
        # self._J = self.chain.jacobian(joint_config)

        # Use Jacobian to get derivative wrt joint configuration
        Dg = (dpose.unsqueeze(-2) @ self._J.unsqueeze(1)).squeeze(-2)
        # now to compute hessian
        hessian_pose_r1 = torch.cat((ddp, dp_omega.permute(0, 1, 3, 2)), dim=-1)
        hessian_pose_r2 = torch.cat((dp_omega, domega), dim=-1)
        hessian_pose = torch.cat((hessian_pose_r1, hessian_pose_r2), dim=-2)

        # Use kinematic hessian and jacobian to get 2nd derivative
        DDg = self._J.unsqueeze(1).permute(0, 1, 3, 2) @ hessian_pose @ self._J.unsqueeze(1)
        DDg_part_2 = torch.sum(self._H.reshape(T, 1, 6, 7, 7) * dpose.reshape(T, n_constraints, 6, 1, 1),
                               dim=2).reshape(
            T,
            n_constraints,
            7, 7)
        DDg = DDg + DDg_part_2.permute(0, 1, 3, 2)
        # DDg = torch.zeros(T, n_constraints, 7, 7, device=q.device)
        return constraints, Dg, DDg

    def reset(self):
        self._J, self._h, self._dH = None, None, None


def ee_terminal_constraint(p, mat, goal):
    """

    :param p:
    :param mat:
    :return:
    """

    return 1 * torch.sum((p[:2] - goal.reshape(-1)[:2]) ** 2).reshape(-1)


def ee_equality_constraint(p, mat, height):
    """

    :param p: torch.Tensor (N, 3) end effector position
    :param mat: torch.Tensor (N, 3, 3) end effector rotation matrix

    :return constraints: torch.Tensor(N, 1) contsraints as specified above

    """
    z_axis = mat[:, 2]  # (tensor of 2 - is 3rd column of rotation matrix
    # orientation constraint - dot product should be 1
    constraints = z_axis.reshape(1, 3) @ torch.tensor([0., 0., 1.], device=p.device).reshape(3, 1)

    constraints_z = p[2] - height
    # return constraints_z.reshape(-1)
    return torch.cat((
        constraints_z.reshape(-1), constraints.reshape(1) + 1), dim=0
    )


def ee_inequality_constraint(p, mat, centres, rads):
    """

     :param p: torch.Tensor (N, 3) end effector position
     :param mat: torch.Tensor (N, 3, 3) end effector rotation matrix

     :return constraints: torch.Tensor(N, 1) contsraints as specified above

     """
    # constraint on z position of end effector
    # constraints_z = torch.stack((
    #    p[2] - 0.8,
    #    0.75 - p[2]), dim=0
    # )
    # constraints_z = p[2]
    # constraint on obstacles
    obs_constraints = obstacle_constraint(p, centres, rads)
    return obs_constraints


def do_trial(env, params, fpath):
    state = env.get_state()
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None
    sim, gym, viewer = env.get_sim()
    # ee_pos, ee_ori = state['ee_pos'], state['ee_ori']
    # start = torch.cat((ee_pos, ee_ori), dim=-1).reshape(7).to(device=params['device'])
    start = state['q'].reshape(7).to(device=params['device'])
    chain.to(device=params['device'])
    chain_cc.to(device=params['device'])
    if params['include_obstacles']:
        obstacle_poses = env.get_object_poses()
    else:
        obstacle_poses = None

    if params['include_table']:
        table_height = env.table_height
    else:
        table_height = None
    if 'csvgd' in params['controller'] or 'diffmpc' in params['controller']:
        problem = VictorTableProblem(start, params['goal'], params['T'], device=params['device'],
                                     obstacle_poses=obstacle_poses,
                                     table_height=table_height,
                                     obstacle_type=params['obstacle_type'])

        controller = Constrained_SVGD_MPC(problem, params)

    elif 'ipopt' in params['controller']:
        problem = VictorTableIpoptProblem(start, params['goal'], params['T'],
                                          obstacle_poses=obstacle_poses,
                                          table_height=table_height,
                                          obstacle_type=params['obstacle_type'])

        controller = IpoptMPC(problem, params)
    elif 'svgd' in params['controller']:
        problem = VictorTableUnconstrainedProblem(start, params['goal'], params['T'], device=params['device'],
                                                  obstacle_poses=obstacle_poses,
                                                  table_height=table_height,
                                                  obstacle_type=params['obstacle_type'],
                                                  penalty=params['penalty'])
        controller = SVMPC(problem, params)
    elif 'mppi' in params['controller']:
        problem = VictorTableUnconstrainedProblem(start, params['goal'], params['T'], device=params['device'],
                                                  obstacle_poses=obstacle_poses,
                                                  table_height=table_height,
                                                  obstacle_type=params['obstacle_type'],
                                                  penalty=params['penalty'])
        controller = MPPI(problem, params)
    else:
        raise ValueError('Invalid controller')

    actual_trajectory = []
    planned_trajectories = []
    duration = 0

    for k in range(params['num_steps']):
        if params['simulate'] or k == 0:
            state = env.get_state()
            start = state['q'].reshape(7).to(device=params['device'])
        else:
            # don't bother simulating, assume we followed the plan noisily
            start = x + torch.randn_like(x) * 0.01

        actual_trajectory.append(start.clone())
        if k > 0:
            torch.cuda.synchronize()
            start_time = time.time()
        best_traj, trajectories = controller.step(start)
        planned_trajectories.append(trajectories)
        if k > 0:
            torch.cuda.synchronize()
            duration += time.time() - start_time

        x = best_traj[0, :7]

        if params['visualize']:
            # add params['goal'] lines to sim
            line_vertices = np.array([
                [params['goal'][0].item() - 0.025, params['goal'][1].item() - 0.025, env.table_height + 0.005],
                [params['goal'][0].item() + 0.025, params['goal'][1].item() + 0.025, env.table_height + 0.005],
                [params['goal'][0].item() - 0.025, params['goal'][1].item() + 0.025, env.table_height + 0.005],
                [params['goal'][0].item() + 0.025, params['goal'][1].item() - 0.025, env.table_height + 0.005],
            ], dtype=np.float32)

            line_colors = np.array([
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0]
            ], dtype=np.float32)

            for e in env.envs:
                gym.add_lines(viewer, e, 2, line_vertices, line_colors)

            # add trajectory lines to sim
            # trajectory_colors
            # traj_line_colors = np.array([[0.5, 0., 0.5]*M], dtype=np.float32)

            M = len(trajectories)
            if M > 0:
                trajectories = chain.forward_kinematics(trajectories[:, :, :7].reshape(-1, 7)).get_matrix().reshape(M,
                                                                                                                    -1,
                                                                                                                    4,
                                                                                                                    4)
                trajectories = trajectories[:, :, :3, 3]

                traj_line_colors = np.random.random((1, M)).astype(np.float32)

                for e in env.envs:
                    s = env.get_state()['ee_pos'].reshape(1, 3).to(device=params['device'])
                    p = torch.stack((s[:3].reshape(1, 3).repeat(M, 1),
                                     trajectories[:, 0, :3]), dim=1).reshape(2 * M, 3).cpu().numpy()
                    p[:, 2] += 0.005
                    gym.add_lines(viewer, e, M, p, traj_line_colors)
                    T = trajectories.shape[1]
                    for t in range(T - 1):
                        p = torch.stack((trajectories[:, t, :3], trajectories[:, t + 1, :3]), dim=1).reshape(2 * M, 3)
                        p = p.cpu().numpy()
                        p[:, 2] += 0.01
                        gym.add_lines(viewer, e, M, p, traj_line_colors)
                    gym.step_graphics(sim)
                    gym.draw_viewer(viewer, sim, False)
                    gym.sync_frame_time(sim)

        if params['simulate']:
            env.step(x.reshape(1, 7).to(device=env.device))
        if params['visualize']:
            gym.clear_lines(viewer)

    state = env.get_state()
    # obs1_pos = state['obs1_pos'][0, :2]
    # obs2_pos = state['obs2_pos'][0, :2]
    state = state['q'].reshape(7).to(device=params['device'])

    # obs = torch.stack((obs1_pos, obs2_pos), dim=0).cpu().numpy()
    # if not params['include_obstacles']:
    #    obs = None
    obs = None
    actual_trajectory.append(state.clone())

    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 7)
    planned_trajectories = torch.stack(planned_trajectories, dim=0)

    problem.T = actual_trajectory.shape[0]
    if params['include_table']:
        constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0).cpu().numpy()
    else:
        constraint_val = None

    if params['include_obstacles']:
        obs_constraint_val = problem._con_ineq(actual_trajectory.unsqueeze(0), False)[0].squeeze(0).cpu().numpy()
    else:
        obs_constraint_val = None

    final_distance_to_goal = torch.linalg.norm(
        chain.forward_kinematics(actual_trajectory[:, :7].reshape(-1, 7)).get_matrix().reshape(-1, 4, 4)[:, :3, 3] -
        params[
            'goal'].unsqueeze(0),
        dim=1
    )
    # print(f'Controller: {params["controller"]} Final distance to goal: {torch.min(final_distance_to_goal)}')
    # print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')
    if params['visualize']:
        env.gym.write_viewer_image_to_file(env.viewer, f'{env.frame_fpath}/frame_{env.frame_id + 1:06d}.png')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
             constr=constraint_val,
             d2goal=final_distance_to_goal.cpu().numpy(),
             traj=planned_trajectories.cpu().numpy(),
             obs=obs,
             height=table_height,
             goal=params['goal'].cpu().numpy(),
             obs_constr=obs_constraint_val,
             )
    return torch.min(final_distance_to_goal).cpu().numpy()