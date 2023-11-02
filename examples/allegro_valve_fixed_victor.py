import numpy as np
from isaacgym.torch_utils import quat_apply
from isaac_victor_envs.tasks.victor_allegro import VictorAllegroValveTurningEnv, orientation_error, quat_change_convention

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
import pytorch_kinematics.transforms as tf


CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
asset = '/home/fanyang/github/isaacgym-arm-envs/isaac_victor_envs/assets/victor/victor_allegro.urdf'
index_name = 'index_biotac_tip'
thumb_name = 'thumb_biotac_tip'

victor_chain = pk.build_serial_chain_from_urdf(open(asset).read(), 'victor_left_arm_link_ee')
index_chain = pk.build_serial_chain_from_urdf(open(asset).read(), 'index_biotac_tip', 'victor_left_arm_link_ee')
index_full_chain = pk.build_serial_chain_from_urdf(open(asset).read(), 'index_biotac_tip')
thumb_chain = pk.build_serial_chain_from_urdf(open(asset).read(), 'thumb_biotac_tip', 'victor_left_arm_link_ee')
thumb_full_chain = pk.build_serial_chain_from_urdf(open(asset).read(), 'thumb_biotac_tip')
valve_location = torch.tensor([0.85, 0.70 ,1.405]).to('cuda:0')

class AllegroValve(ConstrainedSVGDProblem):

    def __init__(self, start, goal, T, chain, valve_location, victor_base_q, device='cuda:0'):
        """
        valve location: the root location of the valve
        """
        super().__init__(start, goal, T, device)
        self.dz = 0
        self.dh = self.dz * T
        self.dg = 3 * T# + 2
        self.dx = 5
        self.du = 0
        self.dt = 0.1
        self.T = T
        self.start = start
        self.goal = goal
        self.K = rbf_kernel
        #self.K = structured_rbf_kernel

        self.valve_location = valve_location

        self.victor_base_q = victor_base_q.to(self.device)
        self.victor_ee_m = victor_chain.forward_kinematics(self.victor_base_q)

        self.chain = chain
        self.grad_kernel = jacrev(rbf_kernel, argnums=0)
        self.alpha = 10

        # self._equality_constraints = EndEffectorConstraint(
        #     self.chain, ee_equality_constraint, self.victor_ee_m
        # )
        self._equality_constraints = JointConstraint(
            self.chain, partial(joint_equality_constraint, chain=self.chain, victor_ee_m=self.victor_ee_m), self.victor_ee_m
        )

        # self._inequality_constraints = EndEffectorConstraint(
        #     self.chain, ee_inequality_constraint
        # )

        self._terminal_constraints = JointConstraint(
            self.chain, partial(joint_terminal_constraint, goal=self.goal)
        )

        self.x_max = torch.tensor([1.0, 3.0, 3.0, 5.0, 1.5*np.pi])
        self.x_min = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.5*np.pi])
        # self.x_min = -self.x_max

        self.dynamics_constraint = vmap(self._dynamics_constraint)
        self.grad_dynamics_constraint = vmap(jacrev(self._dynamics_constraint))
        self.hess_dynamics_constraint = vmap(hessian(self._dynamics_constraint))

        self.cost = vmap(partial(cost, start=self.start))
        self.grad_cost = vmap(jacrev(partial(cost, start=self.start)))
        self.hess_cost = vmap(hessian(partial(cost, start=self.start)))

    def dynamics(self, x, u):
        N = x.shape[0]
        x_finger_prime = x[:, :-1] + self.dt * u
        # m = self.chain.forward_kinematics(x[:-1])
        # p, mat = m[:, :3, 3], m[:, :3, :3]
        m_prime = self.victor_ee_m @ self.chain.forward_kinematics(x_finger_prime) 
        p_prime, mat_prime = m_prime[:, :3, 3], m_prime[:, :3, :3]
        # p_vec = p - self.valve_location
        p_prime_vec = p_prime - self.valve_location
        theta = torch.atan2(p_prime_vec[:, 0], p_prime_vec[:, 2])
        # theta = torch.zeros(theta.shape).to(p_prime.device)
        # print("!!!!!!!!!!!!calling the dynamics")
        # TODO: Update the dynamics 
        return torch.cat((x_finger_prime, theta.unsqueeze(-1)), dim=-1)
        
        # return x + self.dt * u

    def _dynamics_constraint(self, trajectory):
        x = trajectory[:, :self.dx]
        u = trajectory[:, self.dx:]
        breakpoint()
        current_x = torch.cat((self.start.reshape(1, self.dx), x[:-1]), dim=0)
        next_x = x
        pred_next_x = self.dynamics(current_x, u)
        return torch.reshape(pred_next_x - next_x, (-1,))

    def _objective(self, x):
        x = x[:, :, :self.dx]
        N = x.shape[0]
        term_g, term_grad_g, term_hess_g = self._terminal_constraints.eval(x.reshape(-1, self.dx))
        #term_grad_g_extended = torch.zeros(N, self.T, self.dx, device=self.device)
        #term_hess_g_extended = torch.zeros(N, self.T, self.dx, self.T, self.dx, device=self.device)
        #term_grad_g_extended[:, -1, :] = term_grad_g.reshape(N, -1)
        #term_hess_g_extended[:, -1, :, -1, :] = term_hess_g.reshape(N, self.dx, self.dx)
        # J, grad_J, hess_J = self.cost(x)
        J, grad_J, hess_J = self.cost(x), self.grad_cost(x), self.hess_cost(x)

        if term_grad_g is not None:
            with torch.no_grad():
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
        theta = x[:, :, -1]
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


    # def _con_ineq(self, x, compute_grads=True):
    #     x = x[:, :, :self.dx]
    #     #return None, None, None
    #     N = x.shape[0]
    #     h, grad_h, hess_h = self._inequality_constraints.eval(x.reshape(-1, self.dx), compute_grads)

    #     # Consider time as another batch, need to reshape
    #     h = h.reshape(N, self.T, -1).reshape(N, -1)

    #     if not compute_grads:
    #         return h, None, None

    #     grad_h = grad_h.reshape(N, self.T, -1, self.dx).permute(0, 2, 3, 1)
    #     grad_h = torch.diag_embed(grad_h)  # (N, n_constraints, dx + du, T, T)
    #     grad_h = grad_h.permute(0, 3, 1, 4, 2).reshape(N, -1, self.T * (self.dx))

    #     # Now do hessian
    #     hess_h = hess_h.reshape(N, self.T, -1, self.dx, self.dx).permute(0, 2, 3, 4, 1)
    #     hess_h = torch.diag_embed(torch.diag_embed(hess_h))  # (N, n_constraints, dx + du, dx + du, T, T, T)
    #     hess_h = hess_h.permute(0, 4, 1, 5, 2, 6, 3).reshape(N, self.dh,
    #                                                          self.T * (self.dx),
    #                                                          self.T * (self.dx))

    #     return h, grad_h, hess_h
    def _con_ineq(self, x, compute_grads=True):

        return None, None, None

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

            self._terminal_constraints = JointConstraint(
                self.chain, partial(joint_terminal_constraint, goal=self.goal)
            )

        if T is not None:
            self.T = T
            self.dh = self.dz * T
            self.dg = 2 * T


    def get_initial_xu(self, N):

        u = torch.randn(N, self.T, 4, device=self.device)
        x = [self.start.reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            x.append(self.dynamics(x[-1], u[:, t]))

        # particles = torch.cumsum(particles, dim=1) + self.start.reshape(1, 1, self.dx)
        x = torch.stack(x[1:], dim=1)
        xu = torch.cat((x, u), dim=2)
        return x



def cost(x, start):
    x = torch.cat((start.reshape(1, 5), x[:, :5]), dim=0)
    x = x[:, :4]
    weight = torch.tensor([
        0.2, 0.25, 0.2, 0.2], device=x.device, dtype=torch.float32)
    weight = 1.0 / weight
    diff = x[1:] - x[:-1]
    weighted_diff = diff.reshape(-1, 1, 4) @ torch.diag(weight).unsqueeze(0) @ diff.reshape(-1, 4, 1)
    return 10 * torch.sum(weighted_diff)


# class EndEffectorConstraint:

#     def __init__(self, chain, ee_constraint_function, victor_ee_m):
#         self.chain = chain
#         self._fn = partial(ee_constraint_function)
#         self.ee_constraint_fn = vmap(ee_constraint_function)

#         self._grad_fn = jacrev(ee_constraint_function, argnums=(0, 1, 2))

#         self.grad_constraint = vmap(self._grad_g)
#         self.hess_constraint = vmap(jacfwd(self._grad_g, argnums=(0, 1, 2)))

#         self._J, self._H, self._dH = None, None, None

#         self.victor_ee_m = victor_ee_m

#     def _grad_g(self, p, mat, valve_theta):
#         dp, dmat, dtheta = self._grad_fn(p, mat, valve_theta)

#         dmat = dmat @ mat.reshape(1, 3, 3).permute(0, 2, 1)

#         omega1 = torch.stack((-dmat[:, 1, 2], dmat[:, 0, 2], -dmat[:, 0, 1]), dim=-1)
#         omega2 = torch.stack((dmat[:, 2, 1], -dmat[:, 2, 0], dmat[:, 1, 0]), dim=-1)
#         omega = (omega1 + omega2)  # this doesn't seem correct? Surely I should be halfing it
#         return dp, omega, dtheta

#     def eval(self, q, compute_grads=True):
#         """

#         :param q: torch.Tensor of shape (N, 7) containing set of robot joint config
#         :return g: constraint values
#         :return Dg: constraint gradient
#         :return DDg: constraint hessian
#         """

#         T = q.shape[0]

#         # robot joint configuration
#         joint_config = q[:, :4]
#         valve_theta = q[:, 4].unsqueeze(-1)

#         # Get end effector pose
#         m = self.chain.forward_kinematics(joint_config)
#         m = self.victor_ee_m @ m
#         p, mat = m[:, :3, 3], m[:, :3, :3]

#         # Compute constraints
#         # TODO: check if the theta constraints is always 0
#         constraints = self.ee_constraint_fn(p, mat, valve_theta)

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
#         dp, omega, dtheta = self.grad_constraint(p, mat, valve_theta)
#         ddp, domega, ddtheta = self.hess_constraint(p, mat, valve_theta)

#         ddp, dp_dmat, dp_dtheta = ddp
#         domega_dp, domega, omega_dtheta = domega
#         dtheta_dp, dtheta_omega, ddtheta = ddtheta

#         dp_omega = domega_dp
#         dtheta_omega = omega_dtheta

#         tmp = domega @ mat.reshape(-1, 1, 1, 3, 3).permute(0, 1, 2, 4, 3)
#         domega1 = torch.stack((-tmp[:, :, :, 1, 2], tmp[:, :, :, 0, 2], -tmp[:, :, :, 0, 1]), dim=-1)
#         domega2 = torch.stack((tmp[:, :, :, 2, 1], -tmp[:, :, :, 2, 0], tmp[:, :, :, 1, 0]), dim=-1)
#         domega = (domega1 + domega2)

#         # Finally have computed derivative of constraint wrt pose as a (N, num_constraints, 6) tensor
#         dpose = torch.cat((dp, omega), dim=-1)

#         # cache computation for later
#         self._J, self._H, self._dH = self.chain.jacobian_and_hessian_and_dhessian(joint_config)

#         # Use Jacobian to get derivative wrt joint configuration
#         djoint_config = (dpose.unsqueeze(-2) @ self._J.unsqueeze(1)).squeeze(-2)
#         Dg = torch.cat((djoint_config, dtheta), dim=-1)

#         # now to compute hessian
#         hessian_pose_r1 = torch.cat((ddp, dp_omega.permute(0, 1, 3, 2)), dim=-1)
#         hessian_pose_r2 = torch.cat((dp_omega, domega), dim=-1)
#         hessian_pose = torch.cat((hessian_pose_r1, hessian_pose_r2), dim=-2)

#         # Use kinematic hessian and jacobian to get 2nd derivative
#         DDg = self._J.unsqueeze(1).permute(0, 1, 3, 2) @ hessian_pose @ self._J.unsqueeze(1)
#         DDg_part_2 = torch.sum(self._H.reshape(T, 1, 6, 4, 4) * dpose.reshape(T, n_constraints, 6, 1, 1),
#                                dim=2).reshape(
#             T,
#             n_constraints,
#             4, 4)
#         DDg = DDg + DDg_part_2.permute(0, 1, 3, 2)
#         dtheta_dpose = torch.cat((dtheta_dp, dtheta_omega.permute(0, 1, 3, 2)), dim=-1)

#         dtheta_dq = (dtheta_dpose.unsqueeze(2) @ self._J.unsqueeze(1).unsqueeze(1)).squeeze(2)

#         DDg = torch.cat((
#             torch.cat((DDg, dtheta_dq.permute(0, 1, 3, 2)), dim=-1),
#             torch.cat((dtheta_dq, ddtheta), dim=-1)),
#             dim=-2
#         )

#         return constraints, Dg, DDg

#     def reset(self):
#         self._J, self._h, self._dH = None, None, None

class JointConstraint:

    def __init__(self, chain, joint_constraint_function, victor_ee_m=None):
        self.chain = chain
        self._fn = partial(joint_constraint_function, chain)
        self.joint_constraint_fn = vmap(joint_constraint_function)

        self._grad_fn = jacrev(joint_constraint_function, argnums=0)

        self.grad_constraint = vmap(self._grad_g)
        self.hess_constraint = vmap(jacfwd(self._grad_g, argnums=0))

        self._J, self._H, self._dH = None, None, None

        self.victor_ee_m = victor_ee_m

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
        constraints = self.joint_constraint_fn(q)
        if not compute_grads:
            return constraints, None, None

        dq = self.grad_constraint(q)
        ddq = self.hess_constraint(q)

        return constraints, dq, ddq

    def reset(self):
        self._J, self._h, self._dH = None, None, None

class JointThetaConstraint:

    def __init__(self, chain, joint_constraint_function, victor_ee_m=None):
        self.chain = chain
        self._fn = partial(joint_constraint_function, chain)
        self.joint_constraint_fn = vmap(joint_constraint_function)

        self._grad_fn = jacrev(joint_constraint_function, argnums=0)

        self.grad_constraint = vmap(self._grad_g)
        self.hess_constraint = vmap(jacfwd(self._grad_g, argnums=0))

        self._J, self._H, self._dH = None, None, None

        self.victor_ee_m = victor_ee_m

    def _grad_g(self, q, theta=None):
        dq = self._grad_fn(q, theta)
        return dq

    def eval(self, q, theta=None, compute_grads=True):
        """
        :param q: torch.Tensor of shape (N, 7) containing set of robot joint config
        :return g: constraint values
        :return Dg: constraint gradient
        :return DDg: constraint hessian
        """
        T = q.shape[0]
        constraints = self.joint_constraint_fn(q, theta)
        if not compute_grads:
            return constraints, None, None

        dq = self.grad_constraint(q, theta)
        ddq = self.hess_constraint(q, theta)

        return constraints, dq, ddq

    def reset(self):
        self._J, self._h, self._dH = None, None, None

def joint_terminal_constraint(q, goal, theta=None):
    """

    :param p:
    :param mat:
    :return:
    """
    return 1000 * torch.sum((q[-1] - goal.reshape(1))**2).reshape(-1)


def ee_equality_constraint(p, mat, theta):
    """

    :param p: torch.Tensor (N, 3) end effector position
    :param mat: torch.Tensor (N, 3, 3) end effector rotation matrix
    :theta: the angel of the valve (N)

    :return constraints: torch.Tensor(N, 1) contsraints as specified above

    """
    constraint_dist = torch.sqrt((p[2] - valve_location[2]) ** 2 + (p[0] - valve_location[0])**2) - 0.025
    p_vec = p - valve_location
    finger_theta = torch.atan2(p_vec[0], p_vec[2])
    constraint_theta = finger_theta - theta
    return torch.cat((constraint_dist.reshape(-1), constraint_theta.reshape(-1)), dim=0)

def joint_equality_constraint(q, chain, victor_ee_m):
    """

    :param p: torch.Tensor (N, 3) end effector position
    :param mat: torch.Tensor (N, 3, 3) end effector rotation matrix
    :theta: the angel of the valve (N)

    :return constraints: torch.Tensor(N, 1) contsraints as specified above

    """
    theta = q[-1]
    finger_m = victor_ee_m @ chain.forward_kinematics(q) 
    finger_m = finger_m[0]
    p, mat = finger_m[:3, 3], finger_m[:3, :3]
    # finger tip should be on the surface of the object
    constraint_dist = torch.sqrt((p[2] - valve_location[2]) ** 2 + (p[0] - valve_location[0])**2) - 0.025
    # the y coordinate of the finger tip should be fixed
    constraint_y = valve_location[1] - p[1] - 0.02
    # The finger should align with the valve angel
    p_vec = p - valve_location

    # if it is not close enough to the valve, we don't chaneg the valve angel
    finger_theta = torch.atan2(p_vec[0], p_vec[2])
    # finger_theta = (constraint_dist < 0.02)*torch.atan2(p_vec[0], p_vec[2]) + (constraint_dist >= 0.02)* theta
    constraint_theta = finger_theta - theta
    return torch.cat((constraint_dist.reshape(-1), constraint_theta.reshape(-1), constraint_y.reshape(-1)), dim=0)



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
    index_start = torch.cat((state['index_q'], state['valve']), axis=-1).reshape(5).to(device=params['device'])
    thumb_start = torch.cat((state['thumb_q'], state['valve']), axis=-1).reshape(5).to(device=params['device'])
    index_chain.to(device=params['device'])
    thumb_chain.to(device=params['device'])
    index_full_chain.to(device=params['device'])
    victor_chain.to(device=params['device'])
    victor_ee_m = victor_chain.forward_kinematics(env.victor_left_q)

    if params['controller'] == 'csvgd':
        index_problem = AllegroValve(index_start, params['goal'], params['T'], device=params['device'], chain=index_chain, valve_location=valve_location, victor_base_q=victor_base_q)
        index_controller = Constrained_SVGD_MPC(index_problem, params)
    # elif params['controller'] == 'ipopt':
    #     problem = VictorTableIpoptProblem(start, params['goal'], params['T'])
    #     controller = IpoptMPC(problem, params)
    else:
        raise ValueError('Invalid controller')

    actual_trajectory = []
    duration = 0
    for k in range(params['num_steps']):
        state = env.get_state()
        index_start = state['index_q_valve'].reshape(5).to(device=params['device'])
        index_world_ee_m =  victor_ee_m
        victor_ee_m = victor_chain.forward_kinematics(state['victor_index_q'][:, :7])
        gt = index_full_chain.forward_kinematics(state['victor_index_q'])
        index_ee_m = index_chain.forward_kinematics(state['index_q'], world=tf.Transform3d(matrix=victor_ee_m))

        actual_trajectory.append(state['q'].reshape(9).clone())
        if k > 0:
            torch.cuda.synchronize()
            start_time = time.time()
        best_traj, trajectories = index_controller.step(index_start)
        if k > 0:
            torch.cuda.synchronize()
            duration += time.time() - start_time

        x = best_traj[0, :5]
        # add goal lines to sim
        # line_vertices = np.array([
        #     [goal[0].item() - 0.025, goal[1].item() - 0.025, 0.808],
        #     [goal[0].item() + 0.025, goal[1].item() + 0.025, 0.808],
        #     [goal[0].item() - 0.025, goal[1].item() + 0.025, 0.808],
        #     [goal[0].item() + 0.025, goal[1].item() - 0.025, 0.808],
        # ], dtype=np.float32)

        # line_colors = np.array([
        #     [1.0, 0.0, 0.0],
        #     [1.0, 0.0, 0.0]
        # ], dtype=np.float32)

        # for e in env.envs:
        #     gym.add_lines(viewer, e, 2, line_vertices, line_colors)

        # add trajectory lines to sim
        # trajectory_colors
        # traj_line_colors = np.array([[0.5, 0., 0.5]*M], dtype=np.float32)
        M = len(trajectories)
        if M > 0:
            trajectories = index_chain.forward_kinematics(trajectories[:, :, :4].reshape(-1, 4)).reshape(M, -1, 4, 4)
            trajectories = victor_ee_m.unsqueeze(0) @ trajectories
            trajectories = trajectories[:, :, :3, 3]

            traj_line_colors = np.random.random((1, M)).astype(np.float32)

            for e in env.envs:
                s = env.get_state()['index_pos'].reshape(1, 3).to(device=params['device'])
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

        index_action = x.reshape(1,5)[:,:4].to(device=env.device)
        thumb_action = torch.zeros((1,4)).to(device=env.device)
        action = torch.cat((index_action, thumb_action), dim=-1)
        env.step(action)

        gym.clear_lines(viewer)

    state = env.get_state()
    state = state['q'].reshape(9).to(device=params['device'])
    actual_trajectory.append(state.clone())
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 9)
    index_problem.T = actual_trajectory.shape[0]
    constraint_val = index_problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    final_distance_to_goal = actual_trajectory[:, -1] - params['goal']
    # final_distance_to_goal = torch.linalg.norm(
    #     chain.forward_kinematics(actual_trajectory[:, :7].reshape(-1, 7)).reshape(-1, 4, 4)[:, :2, 3] - params['goal'].unsqueeze(0),
    #     dim=1
    # )

    print(f'Controller: {params["controller"]} Final distance to goal: {torch.min(final_distance_to_goal)}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"]- 1)}')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
             constr=constraint_val.cpu().numpy(),
             d2goal=final_distance_to_goal.cpu().numpy())
    return torch.min(final_distance_to_goal).cpu().numpy()


if __name__ == "__main__":
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/victor_table_jointspace.yaml').read_text())
    del config['controllers']['ipopt']
    del config['controllers']['mppi_100']
    del config['controllers']['mppi_1000']
    del config['controllers']['svgd_100']
    del config['controllers']['svgd_grad_100']
    del config['controllers']['svgd_grad_1000']
    from tqdm import tqdm

    # instantiate environment
    env = VictorAllegroValveTurningEnv(1, control_mode='joint_impedance', use_cartesian_controller=False, 
                                       viewer=True, steps_per_action=30)
    sim, gym, viewer = env.get_sim()
    victor_base_q = env.victor_left_q

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
        goal = torch.tensor([-np.pi])
        goal = goal + 0.025 * torch.randn(1) + 0.2
        for controller in config['controllers'].keys():
            env.reset()
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
