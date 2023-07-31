import numpy as np
from isaacgym.torch_utils import quat_apply
from isaac_victor_envs.tasks.victor import VictorPuckObstacleEnv, orientation_error, quat_change_convention

import torch
import time
import yaml
import pathlib
from functools import partial
from torch.func import vmap, jacrev, hessian, jacfwd

from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
from ccai.kernels import rbf_kernel, structured_rbf_kernel

from ccai.problem import ConstrainedSVGDProblem, UnconstrainedPenaltyProblem, IpoptProblem
from ccai.mpc.csvgd import Constrained_SVGD_MPC
from ccai.mpc.mppi import MPPI
from ccai.mpc.svgd import SVMPC
from ccai.mpc.ipopt import IpoptMPC
import time
import pytorch_kinematics as pk

from quadrotor_learn_to_sample import TrajectorySampler

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
asset = '/home/tpower/dev/isaac_test/IsaacVictorEnvs/isaac_victor_envs/assets/victor/victor.urdf'
ee_name = 'victor_left_arm_striker_mallet_tip'
chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name)


class VictorTableProblem(ConstrainedSVGDProblem):

    def __init__(self, start, goal, T, obstacle_centres, obstacle_rads, table_height, device='cuda:0'):
        super().__init__(start, goal, T, device)
        if obstacle_centres is None:
            self.dz = 0
        else:
            self.dz = 2
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
        self.K = rbf_kernel
        # self.K = structured_rbf_kernel

        self.grad_kernel = jacrev(rbf_kernel, argnums=0)
        self.alpha = 10

        if table_height is not None:
            self._equality_constraints = EndEffectorConstraint(
                chain, partial(ee_equality_constraint, height=table_height)
            )
        else:
            self._equality_constraints = None

        if obstacle_centres is not None:
            self._inequality_constraints = EndEffectorConstraint(
                chain, partial(ee_inequality_constraint, centres=obstacle_centres, rads=obstacle_rads)
            )
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

        h, grad_h, hess_h = self._inequality_constraints.eval(x.reshape(-1, self.dx), compute_grads)

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

        # print(G.abs().max(), G.abs().mean(), J)
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

    def _grad_g(self, p, mat):
        dp, dmat = self._grad_fn(p, mat)

        dmat = dmat @ mat.reshape(1, 3, 3).permute(0, 2, 1)

        omega1 = torch.stack((-dmat[:, 1, 2], dmat[:, 0, 2], -dmat[:, 0, 1]), dim=-1)
        omega2 = torch.stack((dmat[:, 2, 1], -dmat[:, 2, 0], dmat[:, 1, 0]), dim=-1)
        omega = (omega1 + omega2)  # this doesn't seem correct? Surely I should be halfing it
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
        m = self.chain.forward_kinematics(joint_config)
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
        self._J, self._H, self._dH = self.chain.jacobian_and_hessian_and_dhessian(joint_config)

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

        return constraints, Dg, DDg

    def reset(self):
        self._J, self._h, self._dH = None, None, None


def ee_terminal_constraint(p, mat, goal):
    """

    :param p:
    :param mat:
    :return:
    """

    return 10 * torch.sum((p[:2] - goal.reshape(2)) ** 2).reshape(-1)


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
    return 1 + constraints.reshape(1)


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
    return torch.cat((constraints_z, obs_constraints), dim=0)


def do_trial(env, params, fpath):
    state = env.get_state()
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None

    # ee_pos, ee_ori = state['ee_pos'], state['ee_ori']
    # start = torch.cat((ee_pos, ee_ori), dim=-1).reshape(7).to(device=params['device'])
    start = state['q'].reshape(7).to(device=params['device'])
    chain.to(device=params['device'])

    if params['include_obstacles']:
        obs1_pos = state['obs1_pos'][0, :2]
        obs2_pos = state['obs2_pos'][0, :2]
        centres = [obs1_pos, obs2_pos]
    else:
        centres = None

    if params['include_table']:
        table_height = env.table_height
    else:
        table_height = None

    rads = [0.12, 0.12]
    if 'csvgd' in params['controller']:
        if params['flow_model'] != 'none':
            if 'diffusion' in params['flow_model']:
                flow_type = 'diffusion'
            elif 'cnf' in params['flow_model']:
                flow_type = 'cnf'
            else:
                raise ValueError('Invalid flow model type')
            flow_model = TrajectorySampler(T=params['T'], dx=7, du=0, context_dim=7 + 2 + 5, type=flow_type)
            flow_model.load_state_dict(torch.load(f'{CCAI_PATH}/{params["flow_model"]}'))
            flow_model.to(device=params['device'])
        else:
            flow_model = None
        params['flow_model'] = flow_model
        problem = VictorTableProblem(start, params['goal'], params['T'], device=params['device'],
                                     obstacle_rads=rads,
                                     obstacle_centres=centres,
                                     table_height=table_height)
        controller = Constrained_SVGD_MPC(problem, params)
    elif params['controller'] == 'ipopt':
        problem = VictorTableIpoptProblem(start, params['goal'], params['T'], obstacle_centres=centres,
                                          obstacle_rads=rads, table_height=env.table_height)
        controller = IpoptMPC(problem, params)
    elif 'svgd' in params['controller']:
        problem = VictorTableUnconstrainedProblem(start, params['goal'], params['T'], device=params['device'],
                                                  penalty=params['penalty'], obstacle_centres=centres,
                                                  obstacle_rads=rads, table_height=env.table_height)
        controller = SVMPC(problem, params)
    elif 'mppi' in params['controller']:
        problem = VictorTableUnconstrainedProblem(start, params['goal'], params['T'], device=params['device'],
                                                  penalty=params['penalty'], obstace_centres=centres,
                                                  obstcale_rads=rads, table_height=env.table_height)
        controller = MPPI(problem, params)
    else:
        raise ValueError('Invalid controller')

    actual_trajectory = []
    planned_trajectories = []
    duration = 0

    constr_params = []
    if params['include_table']:
        constr_params.append(torch.tensor([env.table_height, 0.0, 0.0, 0.0]).to(device=params['device']))
    if params['include_obstacles']:
        constr_params.append(torch.stack(centres, dim=0).reshape(-1))

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
        best_traj, trajectories = controller.step(start, constr_params)
        planned_trajectories.append(trajectories)
        if k > 0:
            torch.cuda.synchronize()
            duration += time.time() - start_time

        x = best_traj[0, :7]

        if params['visualize']:
            # add goal lines to sim
            line_vertices = np.array([
                [goal[0].item() - 0.025, goal[1].item() - 0.025, env.table_height + 0.005],
                [goal[0].item() + 0.025, goal[1].item() + 0.025, env.table_height + 0.005],
                [goal[0].item() - 0.025, goal[1].item() + 0.025, env.table_height + 0.005],
                [goal[0].item() + 0.025, goal[1].item() - 0.025, env.table_height + 0.005],
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
                trajectories = chain.forward_kinematics(trajectories[:, :, :7].reshape(-1, 7)).reshape(M, -1, 4, 4)
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
    obs1_pos = state['obs1_pos'][0, :2]
    obs2_pos = state['obs2_pos'][0, :2]
    state = state['q'].reshape(7).to(device=params['device'])

    obs = torch.stack((obs1_pos, obs2_pos), dim=0).cpu().numpy()
    if not params['include_obstacles']:
        obs = None

    actual_trajectory.append(state.clone())

    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 7)
    planned_trajectories = torch.stack(planned_trajectories, dim=0)

    problem.T = actual_trajectory.shape[0]
    if params['include_table']:
        constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0).cpu().numpy()
    else:
        constraint_val = None
    final_distance_to_goal = torch.linalg.norm(
        chain.forward_kinematics(actual_trajectory[:, :7].reshape(-1, 7)).reshape(-1, 4, 4)[:, :2, 3] - params[
            'goal'].unsqueeze(0),
        dim=1
    )

    # print(f'Controller: {params["controller"]} Final distance to goal: {torch.min(final_distance_to_goal)}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
             constr=constraint_val,
             d2goal=final_distance_to_goal.cpu().numpy(),
             traj=planned_trajectories.cpu().numpy(),
             obs=obs,
             height=table_height,
             goal=params['goal'].cpu().numpy(),
             )
    return torch.min(final_distance_to_goal).cpu().numpy()


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    # get config
    config = yaml.safe_load(
        pathlib.Path(f'{CCAI_PATH}/config/planning_configs/victor_table_jointspace.yaml').read_text())
    from tqdm import tqdm

    # instantiate environment
    env = VictorPuckObstacleEnv(1, control_mode='joint_impedance',
                                randomize_obstacles=config['random_env'],
                                randomize_start=config['random_env'],
                                viewer=config['visualize'])
    sim, gym, viewer = env.get_sim()

    """
    state = env.get_state()
    ee_pos, ee_ori = state['ee_pos'], state['ee_ori']
    try:
        while True:
            start = torch.cat((ee_pos, ee_ori), dim=-1).reshape(1, 7)
            env.step(start)
            print('waiting for you to finish camera adjustment, ctrl-c when done')
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    """
    results = {}

    for i in tqdm(range(config['num_trials'])):
        table_height = None if config['include_table'] else 0.1
        obstacles_1 = None if config['include_obstacles'] else [3, 3]
        obstacles_2 = None if config['include_obstacles'] else [-3, -3]
        env.reset(table_height, obstacles_1, obstacles_2, start_on_table=config['include_table'])

        if config['random_env']:
            goal = torch.tensor([0.45, 0.5]) * torch.rand(2) + torch.tensor([0.4, 0.2])
            state = env.get_state()
            obs1, obs2 = state['obs1_pos'][0, :2].cpu(), state['obs2_pos'][0, :2].cpu()
            while torch.linalg.norm(goal - obs1) < 0.1 or torch.linalg.norm(goal - obs2) < 0.1:
                goal = torch.tensor([0.4, 0.5]) * torch.rand(2) + torch.tensor([0.5, 0.2])
        else:
            goal = torch.tensor([0.8, 0.15])
            goal = goal + 0.025 * torch.randn(2)  # torch.tensor([0.25, 0.1]) * torch.rand(2)

        for controller in config['controllers'].keys():
            # env.reset()
            env.reset_arm_only()
            fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
            pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
            # set up params
            params = config.copy()
            params.pop('controllers')
            params.update(config['controllers'][controller])
            params['controller'] = controller
            params['goal'] = goal.to(device=params['device'])
            final_distance_to_goal = do_trial(env, params, fpath)

            if controller not in results.keys():
                results[controller] = [final_distance_to_goal]
            else:
                results[controller].append(final_distance_to_goal)
        print(results)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
