import numpy as np
from isaacgym.torch_utils import quat_apply
# from isaac_victor_envs.tasks.victor import VictorPuckObstacleEnv
from isaac_victor_envs.tasks.victor_allegro import VictorAllegroBaseEnv

import torch
import time
import yaml
import pathlib
from functools import partial
from functorch import vmap, jacrev, hessian, jacfwd

from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
from ccai.kernels import rbf_kernel, structured_rbf_kernel

from ccai.problem import ConstrainedSVGDProblem, UnconstrainedPenaltyProblem, IpoptProblem
from ccai.mpc.csvgd import Constrained_SVGD_MPC
from ccai.mpc.mppi import MPPI
from ccai.mpc.svgd import SVMPC
from ccai.mpc.ipopt import IpoptMPC
import time
import pytorch_kinematics as pk
from pytorch_kinematics import SerialChain

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
asset = '/home/fanyang/github/isaacgym-arm-envs/isaac_victor_envs/assets/victor/victor_allegro.urdf'
ee_name = 'victor_left_arm_striker_mallet_tip'
ee_names = ['index_link_3', 'middle_link_3', 'ring_link_3', 'thumb_link_3']

# chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name)
chain = pk.build_chain_from_urdf(open(asset).read(), ee_names)
# serial_chain = SerialChain._generate_serial_chain_recurse(chain._root, ee_names[1]+'_frame')
# jac, H = chain.jacobian_and_hessian(torch.zeros((1,30)))
jac, H, dH = chain.jacobian_and_hessian_dhessian(torch.zeros((1,30)))

HEIGHT = 0.9

class VictorTableProblem(ConstrainedSVGDProblem):

    def __init__(self, start, goal, T, device='cuda:0'):
        super().__init__(start, goal, T, device)
        # self.dz = 2
        # self.dh = self.dz * T
        # self.dg = 2 * T# + 2
        # self.dx = 7
        # self.du = 0
        # self.dt = 0.1
        self.dz = 1
        self.dh = self.dz * T
        self.dg = 1 * T# + 2
        self.dx = 30
        self.du = 0
        self.dt = 0.1
        self.T = T
        self.start = start
        self.goal = goal
        self.K = rbf_kernel
        #self.K = structured_rbf_kernel

        self.grad_kernel = jacrev(rbf_kernel, argnums=0)
        self.alpha = 10

        # self._equality_constraints = EndEffectorConstraint(
        #     chain, ee_equality_constraint
        # )
        # self._inequality_constraints = EndEffectorConstraint(
        #     chain, ee_inequality_constraint
        # )

        # self._terminal_constraints = EndEffectorConstraint(
        #     chain, partial(ee_terminal_constraint, goal=self.goal)
        # )

        self._equality_constraints = EndEffectorConstraint(
            chain, partial(ee_equality_constraint_joint_space, chain=chain)
        )
        self._inequality_constraints = EndEffectorConstraint(
            chain, partial(ee_inequality_constraint_joint_space, chain=chain)
        )

        self._terminal_constraints = EndEffectorConstraint(
            chain, partial(ee_terminal_constraint_joint_space, goal=self.goal, chain=chain)
        )


        self.x_max = torch.tensor([2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05, 2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05])
        self.x_max = torch.cat((self.x_max, 10 * torch.ones(16).to(self.x_max.device)))
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
        with torch.no_grad():
            term_g, term_grad_g, term_hess_g = self._terminal_constraints.eval(x.reshape(-1, self.dx))
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
        g, grad_g, hess_g = self._equality_constraints.eval(x.reshape(-1, self.dx), compute_grads)
        #term_g, term_grad_g, term_hess_g = self._terminal_constraints.eval(x[:, -1])

        g = g.reshape(N, -1)
        # combine terminal constraint with running constraints
        #g = torch.cat((g, term_g), dim=1)

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

    def _con_ineq(self, x, compute_grads=True):
        x = x[:, :, :self.dx]
        #return None, None, None
        N = x.shape[0]
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
        K = self.K(Xk, Xk, None)#hess_J.mean(dim=0))
        grad_K = -self.grad_kernel(Xk, Xk, None)#@hess_J.mean(dim=0))
        grad_K = grad_K.reshape(N, N, N, self.T*(self.dx + self.du))
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

        # u = torch.randn(N, self.T, 7, device=self.device)
        u = torch.randn(N, self.T, 30, device=self.device)
        x = [self.start.reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            x.append(self.dynamics(x[-1], u[:, t]))

        # particles = torch.cumsum(particles, dim=1) + self.start.reshape(1, 1, self.dx)
        x = torch.stack(x[1:], dim=1)
        xu = torch.cat((x, u), dim=2)
        return x


# class EndEffectorConstraint:

#     def __init__(self, chain, ee_constraint_function):
#         self.chain = chain
#         self._fn = partial(ee_constraint_function)
#         self.ee_constraint_fn = vmap(ee_constraint_function)

#         self._grad_fn = jacrev(ee_constraint_function, argnums=(0, 1))

#         self.grad_constraint = vmap(self._grad_g)
#         self.hess_constraint = vmap(jacfwd(self._grad_g, argnums=(0, 1)))

#         self._J, self._H, self._dH = None, None, None

#     def _grad_g(self, p, mat):
#         dp, dmat = self._grad_fn(p, mat)
#         dmat = dmat @ mat.reshape(1, 4, 3, 3).permute(0, 1, 3, 2)
#         omega1 = torch.stack((-dmat[:, :, 1, 2], dmat[:, :, 0, 2], -dmat[:, :, 0, 1]), dim=-1)
#         omega2 = torch.stack((dmat[:, :, 2, 1], -dmat[:, :, 2, 0], dmat[:, :, 1, 0]), dim=-1)
#         omega = (omega1 + omega2)  # this doesn't seem correct? Surely I should be halfing it
#         return dp, omega

#     def eval(self, q, compute_grads=True):
#         """

#         :param q: torch.Tensor of shape (N, 7) containing set of robot joint config
#         :return g: constraint values
#         :return Dg: constraint gradient
#         :return DDg: constraint hessian
#         """

#         T = q.shape[0]

#         # robot joint configuration
#         # joint_config = q[:, :23]
#         joint_config = q

#         # Get end effector pose
#         m = self.chain.forward_kinematics_end(joint_config)
#         #[batch, end effectors, transformation matrix]
#         p, mat = m[:, :, :3, 3], m[:, :, :3, :3]

#         # Compute constraints
#         constraints = self.ee_constraint_fn(p, mat)

#         if not compute_grads:
#             return constraints, None, None
#         # Compute first and second derivatives of constraints wrt end effector pose
#         n_constraints = constraints.shape[1]

#         # This is quite complex, but the constraint function takes as input a rotation matrix
#         # this means that the gradient and hessian we get from autograd are wrt to parameters of a rotation matrix
#         # We need to transform this into something akin to an angular velocity in order to use the robot jacobian
#         # to compute derivative and hessian wrt joint config
#         # Note: we could use autograd for the whole pipeline but computing the manipulator Jacobian and Hessian
#         # manually is much faster than using autograd
#         dp, omega = self.grad_constraint(p, mat)
#         ddp, domega = self.hess_constraint(p, mat)

#         ddp, dp_dmat = ddp
#         domega_dp, domega = domega
#         dp_omega = domega_dp
#         breakpoint()
#         # omega [96 1 3]
#         # dp [96 1 3]
#         # domega = torch.Size([96, 1, 3, 3, 3])
#         # mat = ([96, 3, 3])
#         tmp = domega @ mat.reshape(-1, 1, 1, 3, 3).permute(0, 1, 2, 4, 3)
#         domega1 = torch.stack((-tmp[:, :, :, 1, 2], tmp[:, :, :, 0, 2], -tmp[:, :, :, 0, 1]), dim=-1)
#         domega2 = torch.stack((tmp[:, :, :, 2, 1], -tmp[:, :, :, 2, 0], tmp[:, :, :, 1, 0]), dim=-1)
#         domega = (domega1 + domega2)

#         # Finally have computed derivative of constraint wrt pose as a (N, num_constraints, 6) tensor
#         dpose = torch.cat((dp, omega), dim=-1)

#         # cache computation for later
#         self._J, self._H, self._dH = self.chain.jacobian_and_hessian_and_dhessian(joint_config)

#         # Use Jacobian to get derivative wrt joint configuration
#         Dg = (dpose.unsqueeze(-2) @ self._J.unsqueeze(1)).squeeze(-2)

#         # now to compute hessian
#         hessian_pose_r1 = torch.cat((ddp, dp_omega.permute(0, 1, 3, 2)), dim=-1)
#         hessian_pose_r2 = torch.cat((dp_omega, domega), dim=-1)
#         hessian_pose = torch.cat((hessian_pose_r1, hessian_pose_r2), dim=-2)

#         # Use kinematic hessian and jacobian to get 2nd derivative
#         DDg = self._J.unsqueeze(1).permute(0, 1, 3, 2) @ hessian_pose @ self._J.unsqueeze(1)
#         breakpoint()
#         DDg_part_2 = torch.sum(self._H.reshape(T, 1, 6, 7, 7) * dpose.reshape(T, n_constraints, 6, 1, 1),
#                                dim=2).reshape(
#             T,
#             n_constraints,
#             7, 7)
#         DDg = DDg + DDg_part_2.permute(0, 1, 3, 2)

#         return constraints, Dg, DDg

#     def reset(self):
#         self._J, self._h, self._dH = None, None, None

class EndEffectorConstraint:

    def __init__(self, chain, ee_constraint_function):
        self.chain = chain
        self._fn = partial(ee_constraint_function, chain)
        self.ee_constraint_fn = vmap(ee_constraint_function)

        self._grad_fn = jacrev(ee_constraint_function, argnums=0)

        self.grad_constraint = vmap(self._grad_g)
        self.hess_constraint = vmap(jacfwd(self._grad_g, argnums=0))

        self._J, self._H, self._dH = None, None, None

    def _grad_g(self, q):
        dq = self._grad_fn(q)
        return dq

    def eval(self, q, compute_grads=True):
        """

        :param q: torch.Tensor of shape (N, 7) containing set of robot joint config
        :return g: constraint values
        :return Dg: constraint gradient
        :return DDg: constraint hessian
        """
        T = q.shape[0]

        # robot joint configuration
        # joint_config = q[:, :23]
        joint_config = q

        constraints = self.ee_constraint_fn(joint_config)

        dq = self.grad_constraint(q)
        ddq = self.hess_constraint(q)

        return constraints, dq, ddq

        # # Get end effector pose
        # m = self.chain.forward_kinematics_end(joint_config, self.chain)
        # #[batch, end effectors, transformation matrix]
        # p, mat = m[:, :, :3, 3], m[:, :, :3, :3]

        # # Compute constraints
        # constraints = self.ee_constraint_fn(p, mat)

        # if not compute_grads:
        #     return constraints, None, None
        # # Compute first and second derivatives of constraints wrt end effector pose
        # n_constraints = constraints.shape[1]

        # # This is quite complex, but the constraint function takes as input a rotation matrix
        # # this means that the gradient and hessian we get from autograd are wrt to parameters of a rotation matrix
        # # We need to transform this into something akin to an angular velocity in order to use the robot jacobian
        # # to compute derivative and hessian wrt joint config
        # # Note: we could use autograd for the whole pipeline but computing the manipulator Jacobian and Hessian
        # # manually is much faster than using autograd
        # dp, omega = self.grad_constraint(p, mat)
        # ddp, domega = self.hess_constraint(p, mat)

        # ddp, dp_dmat = ddp
        # domega_dp, domega = domega
        # dp_omega = domega_dp
        # breakpoint()
        # # omega [96 1 3]
        # # dp [96 1 3]
        # # domega = torch.Size([96, 1, 3, 3, 3])
        # # mat = ([96, 3, 3])
        # tmp = domega @ mat.reshape(-1, 1, 1, 3, 3).permute(0, 1, 2, 4, 3)
        # domega1 = torch.stack((-tmp[:, :, :, 1, 2], tmp[:, :, :, 0, 2], -tmp[:, :, :, 0, 1]), dim=-1)
        # domega2 = torch.stack((tmp[:, :, :, 2, 1], -tmp[:, :, :, 2, 0], tmp[:, :, :, 1, 0]), dim=-1)
        # domega = (domega1 + domega2)

        # # Finally have computed derivative of constraint wrt pose as a (N, num_constraints, 6) tensor
        # dpose = torch.cat((dp, omega), dim=-1)

        # # cache computation for later
        # self._J, self._H, self._dH = self.chain.jacobian_and_hessian_and_dhessian(joint_config)

        # # Use Jacobian to get derivative wrt joint configuration
        # Dg = (dpose.unsqueeze(-2) @ self._J.unsqueeze(1)).squeeze(-2)

        # # now to compute hessian
        # hessian_pose_r1 = torch.cat((ddp, dp_omega.permute(0, 1, 3, 2)), dim=-1)
        # hessian_pose_r2 = torch.cat((dp_omega, domega), dim=-1)
        # hessian_pose = torch.cat((hessian_pose_r1, hessian_pose_r2), dim=-2)

        # # Use kinematic hessian and jacobian to get 2nd derivative
        # DDg = self._J.unsqueeze(1).permute(0, 1, 3, 2) @ hessian_pose @ self._J.unsqueeze(1)
        # breakpoint()
        # DDg_part_2 = torch.sum(self._H.reshape(T, 1, 6, 7, 7) * dpose.reshape(T, n_constraints, 6, 1, 1),
        #                        dim=2).reshape(
        #     T,
        #     n_constraints,
        #     7, 7)
        # DDg = DDg + DDg_part_2.permute(0, 1, 3, 2)

        # return constraints, Dg, DDg

    def reset(self):
        self._J, self._h, self._dH = None, None, None


def ee_equality_constraint(p, mat):
    """

    :param p: torch.Tensor (N, 3) end effector position
    :param mat: torch.Tensor (N, 3, 3) end effector rotation matrix

    :return constraints: torch.Tensor(N, 1) contsraints as specified above

    """
    constraints = p[:, 0, 2] - HEIGHT
    return constraints

def ee_terminal_constraint(p, mat, goal):
    """

    :param p:
    :param mat:
    :return:
    """

    return 10 * torch.sum((p[:3] - goal)**2).reshape(-1)

def ee_inequality_constraint(p, mat):
    """

     :param p: torch.Tensor (N, 3) end effector position
     :param mat: torch.Tensor (N, 3, 3) end effector rotation matrix

     :return constraints: torch.Tensor(N, 1) contsraints as specified above

    """
    results = -p[0, 2]
    return results


def ee_equality_constraint_joint_space(q, chain):
    """

    :param p: torch.Tensor (N, 3) end effector position
    :param mat: torch.Tensor (N, 3, 3) end effector rotation matrix

    :return constraints: torch.Tensor(N, 1) contsraints as specified above

    """
    m = chain.forward_kinematics_end(q)
    #[batch, end effectors, transformation matrix]
    p, mat = m[:, :, :3, 3], m[:, :, :3, :3]
    constraints = p[:, 0, 2] - HEIGHT
    return constraints

def ee_terminal_constraint_joint_space(q, goal, chain):
    """

    :param p:
    :param mat:
    :return:
    """
    m = chain.forward_kinematics_end(q)
    #[batch, end effectors, transformation matrix]
    p, mat = m[:, :, :3, 3], m[:, :, :3, :3]

    return 10 * torch.sum((p[:,0] - goal)**2).reshape(-1)

def ee_inequality_constraint_joint_space(q, chain):
    """

     :param p: torch.Tensor (N, 3) end effector position
     :param mat: torch.Tensor (N, 3, 3) end effector rotation matrix

     :return constraints: torch.Tensor(N, 1) contsraints as specified above

    """
    m = chain.forward_kinematics_end(q)
    #[batch, end effectors, transformation matrix]
    p, mat = m[:, :, :3, 3], m[:, :, :3, :3]

    results = -p[:, 0, 2]
    return results
def cost(x, start):
    # x = torch.cat((start.reshape(1, 23), x[:, :7]), dim=0)
    # weight = torch.tensor([
        # 0.2, 0.25, 0.4, 0.4, 0.6, 0.75, 1.0], device=x.device, dtype=torch.float32)
    # weight = 1.0 / weight
    diff = x[1:] - x[:-1]
    # weighted_diff = diff.reshape(-1, 1, 7) @ torch.diag(weight).unsqueeze(0) @ diff.reshape(-1, 7, 1)
    # return 10 * torch.sum(weighted_diff)
    return 10 * torch.sum(diff ** 2)
    
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
    start = state['q'].reshape(30).to(device=params['device'])
    # start = torch.cat((torch.tensor([2.9501, -0.2883, -2.5463, -1.5888, -1.6216, -1.1591]),
    #                 torch.tensor([-0.7822, 1.144, -1.189, 0.590, 0.292, 0.296, 0.265, -0.809])),
    #                 dim=0).to(device=params['device']).unsqueeze(0)
    # start = torch.cat((start, torch.zeros((1,16)).float().to(device=params['device'])), dim=1).to(params['device'])

    chain.to(device=params['device'])

    if params['controller'] == 'csvgd':
        problem = VictorTableProblem(start, params['goal'], params['T'], device=params['device'])
        controller = Constrained_SVGD_MPC(problem, params)

    actual_trajectory = []
    duration = 0
    for k in range(params['num_steps']):
        state = env.get_state()
        # start = state['q'].reshape(7).to(device=params['device'])
        start = state['q'].reshape(30).to(device=params['device'])

        actual_trajectory.append(start.clone())
        if k > 0:
            torch.cuda.synchronize()
            start_time = time.time()
        best_traj, trajectories = controller.step(start)
        if k > 0:
            torch.cuda.synchronize()
            duration += time.time() - start_time

        x = best_traj[0, :30]
        # add goal lines to sim
        line_vertices = np.array([
            [goal[0].item() - 0.025, goal[1].item() - 0.025, 0.808],
            [goal[0].item() + 0.025, goal[1].item() + 0.025, 0.808],
            [goal[0].item() - 0.025, goal[1].item() + 0.025, 0.808],
            [goal[0].item() + 0.025, goal[1].item() - 0.025, 0.808],
        ], dtype=np.float32)

        line_colors = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ], dtype=np.float32)

        # for e in env.envs:
        #     gym.add_lines(viewer, e, 2, line_vertices, line_colors)

        # add trajectory lines to sim
        # trajectory_colors
        # traj_line_colors = np.array([[0.5, 0., 0.5]*M], dtype=np.float32)
        M = len(trajectories)
        # if M > 0:
        #     breakpoint()
        #     trajectories = chain.forward_kinematics_end(trajectories[:, :, :30].reshape(-1, 30)).reshape(M, -1, 4, 4, 4)
        #     trajectories = trajectories[:, :, :, :3, 3]

        #     traj_line_colors = np.random.random((1, M)).astype(np.float32)

        #     for e in env.envs:
        #         s = env.get_state()['ee_pos'].reshape(1, 3).to(device=params['device'])
        #         p = torch.stack((s[:3].reshape(1, 3).repeat(M, 1),
        #                          trajectories[:, 0, :3]), dim=1).reshape(2 * M, 3).cpu().numpy()
        #         p[:, 2] += 0.005
        #         gym.add_lines(viewer, e, M, p, traj_line_colors)
        #         T = trajectories.shape[1]
        #         for t in range(T - 1):
        #             p = torch.stack((trajectories[:, t, :3], trajectories[:, t + 1, :3]), dim=1).reshape(2 * M, 3)
        #             p = p.cpu().numpy()
        #             p[:, 2] += 0.01
        #             gym.add_lines(viewer, e, M, p, traj_line_colors)
        #         gym.step_graphics(sim)
        #         gym.draw_viewer(viewer, sim, False)
        #         gym.sync_frame_time(sim)
        env.step(x.reshape(1, 30).to(device=env.device))
        print(params['goal'])
        print(chain.forward_kinematics_end(state['q'][:, :30].reshape(-1, 30)).reshape(-1, 4, 4, 4)[:, 0, :3, 3] - params['goal'].unsqueeze(0))

        # gym.clear_lines(viewer)

    state = env.get_state()
    state = state['q'].reshape(30).to(device=params['device'])
    actual_trajectory.append(state.clone())

    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 30)
    problem.T = actual_trajectory.shape[0]
    constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    
    final_distance_to_goal = torch.linalg.norm(
        chain.forward_kinematics_end(actual_trajectory[:, :30].reshape(-1, 30)).reshape(-1, 4, 4, 4)[:, 0, :3, 3] - params['goal'].unsqueeze(0),
        dim=2
    )

    print(f'Controller: {params["controller"]} Final distance to goal: {torch.min(final_distance_to_goal)}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"]- 1)}')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
             constr=constraint_val.cpu().numpy(),
             d2goal=final_distance_to_goal.cpu().numpy())
    return torch.min(final_distance_to_goal).cpu().numpy()


if __name__ == "__main__":
    env = VictorAllegroBaseEnv(1, control_mode='joint_torque', viewer=True, steps_per_action=10)
    sim, gym, viewer = env.get_sim()
    env.reset()
    done = False
    while not done:
        # action = torch.cat((torch.tensor([2.9501, -0.2883, -2.5463, -1.5888, -1.6216, -1.1591]),
        #                                   torch.tensor([-0.7822, 1.144, -1.189, 0.590, 0.292, 0.296, 0.265, -0.809])),
        #                                  dim=0).cuda().unsqueeze(0)
        # action = torch.cat((action, torch.zeros((1,16)).float().cuda()), dim=1).cuda()
        action = torch.randn((1,30)).cuda()
        # action = 0.2 * torch.zeros((1, 30)).cuda()
        # action = 0.3 * torch.cat((torch.ones(1, 7),  torch.ones(1, 7), torch.zeros(1, 16)), dim=-1).cuda()
        # action = torch.cat((torch.ones(1, 7), torch.zeros(1, 16), torch.zeros(1, 7)), dim=-1).cuda()
        # action = torch.ones(30).cuda()
        next_state = env.step(action)
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    # config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/victor_table_jointspace.yaml').read_text())
    # del config['controllers']['ipopt']
    # del config['controllers']['mppi_1000']
    # del config['controllers']['mppi_100']
    # del config['controllers']['svgd_100']
    # del config['controllers']['svgd_grad_100']
    # del config['controllers']['svgd_grad_1000']
    # from tqdm import tqdm

    # # instantiate environment
    # env = VictorAllegroBaseEnv(1, control_mode='joint_torque', viewer=True, use_cartesian_controller=False)
    # sim, gym, viewer = env.get_sim()
    # results = {}

    # for i in tqdm(range(config['num_trials'])):
    #     goal = torch.tensor([0.8, 0.15])
    #     goal = goal + 0.025 * torch.randn(2)#torch.tensor([0.25, 0.1]) * torch.rand(2)
    #     goal = torch.cat((goal, torch.tensor([HEIGHT])))
    #     for controller in config['controllers'].keys():
    #         env.reset()
    #         fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/trial_{i + 1}')
    #         pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
    #         # set up params
    #         params = config.copy()
    #         params.pop('controllers')
    #         params.update(config['controllers'][controller])
    #         params['controller'] = controller
    #         params['goal'] = goal.to(device=params['device'])
    #         final_distance_to_goal = do_trial(env, params, fpath)

    #         if controller not in results.keys():
    #             results[controller] = [final_distance_to_goal]
    #         else:
    #             results[controller].append(final_distance_to_goal)
    #     print(results)

    # gym.destroy_viewer(viewer)
    # gym.destroy_sim(sim)