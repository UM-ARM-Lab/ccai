import numpy as np
from isaacgym.torch_utils import quat_apply
from isaac_victor_envs.tasks.allegro import AllegroValveTurningEnv, orientation_error, quat_change_convention
from isaac_victor_envs.utils import get_assets_dir

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
from ccai.valve import ValveDynamics
from ccai.utils import rotate_jac
# from ccai.mpc.mppi import MPPI
# from ccai.mpc.svgd import SVMPC
# from ccai.mpc.ipopt import IpoptMPC
import time
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
# import pytorch3d.transforms as tf

from pytorch_volumetric import RobotSDF, RobotScene, MeshSDF

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

# asset = f'{get_assets_dir()}/victor/allegro.urdf'
# index_ee_name = 'index_ee'
# thumb_ee_name = 'thumb_ee'
asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
# thumb_ee_name = 'allegro_hand_oya_finger_link_15'
# index_ee_name = 'allegro_hand_hitosashi_finger_finger_link_3'
index_ee_name = 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link'
thumb_ee_name = 'allegro_hand_oya_finger_3_aftc_base_link'

# combined chain
chain = pk.build_chain_from_urdf(open(asset).read())
index_ee_link = chain.frame_to_idx[index_ee_name]
thumb_ee_link = chain.frame_to_idx[thumb_ee_name]
frame_indices = torch.tensor([index_ee_link, thumb_ee_link])

# for joint in chain.get_joints():
#    print(joint.name)
# exit(0)
valve_location = torch.tensor([0.85, 0.70, 1.405]).to('cuda:0') # the root of the valve
# instantiate environment
# env = AllegroValveTurningEnv(1, control_mode='joint_torque_position',
#                              viewer=True, steps_per_action=60)
env = AllegroValveTurningEnv(1, control_mode='joint_impedance', use_cartesian_controller=False,
                             viewer=True, steps_per_action=60, valve_velocity_in_state=False)
world_trans = env.world_trans
# NOTE: DEBUG ONLY, set the friction coefficient extremely large
friction_coefficient = 10000


def partial_to_full_state(partial):
    """
    :params partial: B x 8 joint configurations for index and thumb
    :return full: B x 16 joint configuration for full hand

    # assume that default is zeros, but could change
    """
    index, thumb = torch.chunk(partial, chunks=2, dim=-1)
    full = torch.cat((
        index,
        torch.zeros_like(index),
        torch.zeros_like(index),
        thumb
    ), dim=-1)
    return full

def full_to_partial_state(full):
    """
    :params partial: B x 8 joint configurations for index and thumb
    :return full: B x 16 joint configuration for full hand

    # assume that default is zeros, but could change
    """
    index, mid, ring, thumb = torch.chunk(full, chunks=4, dim=-1)
    partial = torch.cat((
        index,
        thumb
    ), dim=-1)
    return partial


class AllegroValveProblem(ConstrainedSVGDProblem):

    def __init__(self, start, goal, T, chain, valve_location, finger_name, initial_valve_angle=0, device='cuda:0'):
        """
        valve location: the root location of the valve
        initial_valve_angle: it is designed for continuously turning the valve. For each turn, 
        the valve might not be exactly at 0 degree, we need to subtract that out. 
        If we only care about one turn, we can leave it to 0
        """
        super().__init__(start, goal, T, device)
        self.dz = 3 * 2
        # NOTE: DEBUG ONLY, remove friciton cone constraints
        # self.dz = 2 * 2
        self.dh = self.dz * T # inequality
        self.dg = 5 * 2 * T #equality  # though dynamics function constraints only have 2*(T-1), we make it 2 * T to be compatible with the code style
        self.dx = 9 # position of finger joints and theta and theta dot.
        self.du = 8 # finger joints delta action. 
        # NOTE: the decision variable x means x_1 to x_T, but for the action u, it means u_0 to u_{T-1}. 
        # NOTE: The timestep of x and u doesn't match
        self.dt = 0.1
        self.T = T
        self.start = start
        self.goal = goal
        self.K = rbf_kernel

        self.finger_name = finger_name
        self.valve_location = valve_location
        self.initial_valve_angle = initial_valve_angle

        self.chain = chain
        self.grad_kernel = jacrev(rbf_kernel, argnums=0)
        self.alpha = 10

        self._equality_constraints = JointConstraint(
            partial(get_joint_equality_constraint, chain=self.chain, start_q=self.start, 
                    initial_valve_angle=initial_valve_angle)
        )

        self._inequality_constraints = JointConstraint(
            partial(get_joint_inequality_constraint, chain=self.chain, start_q=self.start)
        )
        # for honda hand
        index_x_max = torch.tensor([0.47, 1.6099999999, 1.7089999, 1.61799999])
        index_x_min = torch.tensor([-0.47, -0.195999999999, -0.174000000, -0.227])
        thumb_x_max = torch.tensor([1.396, 1.1629999999999, 1.644, 1.71899999])
        thumb_x_min = torch.tensor([0.26, -0.1049999999, -0.1889999999, -0.162])

        valve_x_max = torch.tensor([10.0 * np.pi])
        valve_x_min = torch.tensor([-10.0 * np.pi])
        self.x_max = torch.cat((index_x_max, thumb_x_max, valve_x_max))
        self.x_min = torch.cat((index_x_min, thumb_x_min, valve_x_min))

        # TODO: this is not perfect, need a better way to define the limit of u
        self.u_max = torch.tensor([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]) / 5
        self.u_min = torch.tensor([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]) / 5

        self.x_max = torch.cat((self.x_max, self.u_max))
        self.x_min = torch.cat((self.x_min, self.u_min))

        self.cost = vmap(partial(cost, start=self.start, goal=self.goal))
        self.grad_cost = vmap(jacrev(partial(cost, start=self.start, goal=self.goal)))
        self.hess_cost = vmap(hessian(partial(cost, start=self.start, goal=self.goal)))

    def _objective(self, x):
        x = x[:, :, :self.dx+self.du]
        N = x.shape[0]
        # goal_grad_g_extended = torch.zeros(N, self.T, self.dx, device=self.device)
        # goal_hess_g_extended = torch.zeros(N, self.T, self.dx, self.T, self.dx, device=self.device)
        # goal_grad_g_extended[:, -1, :] = goal_grad_g.reshape(N, -1)
        # goal_hess_g_extended[:, -1, :, -1, :] = goal_hess_g.reshape(N, self.dx, self.dx)
        # J, grad_J, hess_J = self.cost(x)
        J, grad_J, hess_J = self.cost(x), self.grad_cost(x), self.hess_cost(x)

        N = x.shape[0]
        return (self.alpha * J.reshape(N),
                self.alpha * grad_J.reshape(N, -1),
                self.alpha * hess_J.reshape(N, self.T * (self.dx+self.du), self.T * (self.dx+self.du)))
    
    
    def _con_eq(self, xu, compute_grads=True):
        N = xu.shape[0]
        T = xu.shape[1]
        g, grad_g, hess_g = self._equality_constraints.eval(xu.reshape(N, T, self.dx+self.du), compute_grads)
        # g_dynamics, grad_g_dynamics, hess_g_dynamics = self._dynamics_constraints.eval(xu, compute_grads)
        # term_g, term_grad_g, term_hess_g = self._terminal_constraints.eval(x[:, -1])

        g = g.reshape(N, -1)
        # combine terminal constraint with running constraints
        # g = torch.cat((g, term_g), dim=1)

        if not compute_grads:
            return g, None, None
            # Expand gradient to include time dimensions
        grad_g = grad_g.reshape(N, g.shape[1], self.T * (self.dx + self.du))
        hess_g = hess_g.reshape(N, g.shape[1], self.T * (self.dx + self.du), self.T * (self.dx + self.du))
        # grad_g = grad_g.reshape(N, self.T, -1, self.dx+self.du).permute(0, 2, 3, 1)
        # grad_g = torch.diag_embed(grad_g)  # (N, n_constraints, dx + du, T, T)
        # grad_g = grad_g.permute(0, 3, 1, 4, 2).reshape(N, -1, self.T * (self.dx+self.du))

        # # Now do hessian
        # hess_g = hess_g.reshape(N, self.T, -1, self.dx+self.du, self.dx+self.du).permute(0, 2, 3, 4, 1)
        # hess_g = torch.diag_embed(torch.diag_embed(hess_g))  # (N, n_constraints, dx + du, dx + du, T, T, T)
        # hess_g = hess_g.permute(0, 4, 1, 5, 2, 6, 3).reshape(N, -1,
        #                                                      self.T * (self.dx+self.du),
        #                                                      self.T * (self.dx+self.du))

        # reshape the 
        # grad_g_dynamics = grad_g_dynamics.reshape((N, 2 * (self.T-1), self.T * (self.dx + self.du)))
        # hess_g_dynamics = hess_g_dynamics.reshape((N, 2 * (self.T-1), self.T * (self.dx + self.du), self.T * (self.dx + self.du)))
        # # combine the gradients from equality costraint and dynamics constraints
        # g = torch.cat((g, g_dynamics), axis=1)
        # grad_g = torch.cat((grad_g, grad_g_dynamics), axis=1)
        # hess_g = torch.cat((hess_g, hess_g_dynamics), axis=1)
        return g, grad_g, hess_g

    def _con_ineq(self, xu, compute_grads=True):
        N = xu.shape[0]
        T = xu.shape[1]
        h, grad_h, hess_h = self._inequality_constraints.eval(xu.reshape(-1, T, self.dx+self.du), compute_grads)

        h = h.reshape(N, -1)
        N = xu.shape[0]
        if not compute_grads:
            return h, None, None
            # Expand gradient to include time dimensions
        grad_h = grad_h.reshape(N, h.shape[1], self.T * (self.dx + self.du))
        hess_h = hess_h.reshape(N, h.shape[1], self.T * (self.dx + self.du), self.T * (self.dx + self.du))


        # grad_h = grad_h.reshape(N, self.T, -1, self.dx+self.du).permute(0, 2, 3, 1)
        # grad_h = torch.diag_embed(grad_h)  # (N, n_constraints, dx + du, T, T)
        # grad_h = grad_h.permute(0, 3, 1, 4, 2).reshape(N, -1, self.T * (self.dx+self.du))

        # # Now do hessian
        # hess_h = hess_h.reshape(N, self.T, -1, self.dx+self.du, self.dx+self.du).permute(0, 2, 3, 4, 1)
        # hess_h = torch.diag_embed(torch.diag_embed(hess_h))  # (N, n_constraints, dx + du, dx + du, T, T, T)
        # hess_h = hess_h.permute(0, 4, 1, 5, 2, 6, 3).reshape(N, -1,
        #                                                      self.T * (self.dx+self.du),
        #                                                      self.T * (self.dx+self.du))

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

        # print(G.abs().max(), G.abs().mean(), J.mean())
        return grad_J.detach(), hess_J, K.detach(), grad_K.detach(), G.detach(), dG.detach(), hessG.detach()

    def update(self, start, goal=None, T=None):
        self.start = start
        if goal is not None:
            self.goal = goal

        # update functions that require start
        self.cost = vmap(partial(cost, start=self.start, goal=self.goal))
        self.grad_cost = vmap(jacrev(partial(cost, start=self.start, goal=self.goal)))
        self.hess_cost = vmap(hessian(partial(cost, start=self.start, goal=self.goal)))

        # want the offset between the current angle and the angle according to end effectors
        # we will use position of index fingertip
        fk_dict = chain.forward_kinematics(partial_to_full_state(self.start[:8]),
                                           frame_indices=torch.tensor([index_ee_link]))

        points_finger_frame = torch.tensor([0.005, 0.005, 0.0], device=self.start.device).unsqueeze(0)
        p = world_trans.compose(fk_dict[index_ee_name]).transform_points(points_finger_frame).reshape(
            -1) - valve_location
        if torch.sqrt(p[0] ** 2 + p[2] ** 2) < 0.05:
            # we are probably close enough to valve to have begun rask
            geometric_theta = torch.atan2(p[0], p[2])
            actual_theta = self.start[-1]
            theta_offset = geometric_theta - actual_theta

        else:
            theta_offset = 0
        self._equality_constraints = JointConstraint(
            partial(get_joint_equality_constraint, chain=self.chain, start_q=self.start)
        )

        self._inequality_constraints = JointConstraint(
            partial(get_joint_inequality_constraint, chain=self.chain, start_q=self.start)
        )
        # # TODO: update the dynamics constraint

        if goal is not None:
            self.goal = goal

        if T is not None:
            self.T = T
            self.dh = self.dz * T
            self.dg = 2 * T

    def get_initial_xu(self, N):
        """
        use delta joint movement to get the initial trajectory
        the action (force at the finger tip) is not used. it is randomly intiailized
        the actual dynamics model is not used
        """
        u = torch.randn(N, self.T, self.du, device=self.device)
        x = [self.start.reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            next_q = x[-1][:, :8] + u[:, t]
            x.append(next_q)
        theta = torch.linspace(self.start[-1], self.goal.item(), self.T)
        theta = theta.repeat((N, 1)).unsqueeze(-1).to(self.start.device)
        x = torch.stack(x[1:], dim=1)
        x = torch.cat((x, theta), dim=-1)
        xu = torch.cat((x, u), dim=2)
        return xu


def cost(xu, start, goal):
    state = xu[:, :9] # state dim = 10
    state = torch.cat((start.reshape(1, 9), state), dim=0) # combine the first time step into it

    action = xu[:, 9: 9+6] # action dim = 6
    action_cost = torch.sum(action ** 2)

    smoothness_cost = torch.sum((state[1:] - state[:-1])**2)

    goal_cost = (20 * (state[-1, -1] - goal)**2)[0]

    # diff = state[0] - start
    # weight = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).to(diff.device)
    # initial_cost = 100 * torch.sum(diff * weight * diff)
    # this goal cost is not very informative and takes a long time for the gradients to back propagate to each state. 
    # thus, a better initialization is necessary for faster convergence
    # return torch.sum(weighted_diff) + goal_cost
    # return smoothness_cost + goal_cost
    return smoothness_cost + action_cost + goal_cost


class JointConstraint:

    def __init__(self, joint_constraint_function):
        self._fn = joint_constraint_function
        self.joint_constraint_fn = joint_constraint_function

        self._grad_fn = jacrev(joint_constraint_function, argnums=(0))

        self.grad_constraint = self._grad_g
        self.hess_constraint = jacfwd(self._grad_g, argnums=0)

        self._J, self._H, self._dH = None, None, None

    def _grad_g(self, q):
        dq = self._grad_fn(q)
        return dq

    def eval(self, qu, compute_grads=True):
        """
        :param qu: torch.Tensor of shape (T, 9 + 8) containing set of state + acton
        :return g: constraint values
        :return Dg: constraint gradient
        :return DDg: constraint hessian
        """
        N = qu.shape[0]
        T = qu.shape[1]
        constraints = self.joint_constraint_fn(qu)
        if not compute_grads:
            return constraints, None, None
        dq = self.grad_constraint(qu)
        dq = torch.stack([dq[i, :, i] for i in range(N)], dim=0) # data from different batches should not affect each other
        # dumb_ddq = torch.eye(qu.shape[-1]).repeat((constraints.shape[0], constraints.shape[1], 1, 1))
        ddq = torch.zeros(dq.shape + dq.shape[2:])
        # ddq = self.hess_constraint(qu)
        # assert ddq.shape == dumb_ddq.shape

        return constraints, dq, ddq

    def reset(self):
        self._J, self._h, self._dH = None, None, None

def get_joint_equality_constraint(qu, chain, start_q, initial_valve_angle=0):
    """

    :param qu: torch.Tensor (T, DoF) joint positions
    :param chain: pytorch kinematics chain

    :return constraints: torch.Tensor(N, 1) contsraints as specified above

    """
    valve_offset_for_index = -np.pi / 2.0
    valve_offset_for_thumb = np.pi / 2.0  # * 0.99
    N = qu.shape[0]
    T = qu.shape[1]
    valve_offset_for_index = -np.pi / 2.0
    valve_offset_for_thumb = np.pi / 2.0  # * 0.99
    theta = qu[:, :, 8]
    index_q = qu[:, :, :4]
    thumb_q = qu[:, :, 4:8]
    action = qu[:, :, 9:17]
    index_action = action[:, :, :4]
    thumb_action = action[:, :, 4:]
    finger_qs = [index_q, thumb_q]
    finger_names = [index_ee_name, thumb_ee_name]
    action_list = [index_action, thumb_action]
    radii = [0.040, 0.038]
    constraint_list = []
    q = torch.cat((start_q.repeat((N, 1, 1)), qu[:, :, :9]), dim=1) # add the current time step in
    fk_dict = chain.forward_kinematics(partial_to_full_state(q[:, :, :8].reshape(-1, 8)), frame_indices=frame_indices) # pytorch_kinematics only supprts one additional dim
    fk_desired_dict = chain.forward_kinematics(partial_to_full_state((q[:, :-1, :8] + action).reshape(-1, 8)), frame_indices=frame_indices)

    for finger_q, finger_name, r, ee_index, finger_action in zip(finger_qs, finger_names, radii, frame_indices, action_list):
        m = world_trans.compose(fk_dict[finger_name])
        points_finger_frame = torch.tensor([0.00, 0.03, 0.00], device=qu.device).unsqueeze(0)
        normals_finger_tip_frame = torch.tensor([0.0, 0.0, 1.0], device=qu.device).unsqueeze(0)
        # it is just the thrid column of the rotation matrix
        normals_finger_tip_frame = normals_finger_tip_frame / torch.norm(normals_finger_tip_frame, dim=1, keepdim=True)
        ee_p = m.transform_points(points_finger_frame).reshape((N, (T+1), 1, 3)) # TODO: check if the dimension match
        ee_p = ee_p.squeeze(-2)
        # normals_world_frame = m.transform_normals(normals_finger_tip_frame).reshape(3)
        ee_mat = m.get_matrix().reshape(N, (T+1), 4, 4)[:, :, :3,:3]
        # jac = chain.jacobian(partial_to_full_state(q[:, :, :8].reshape(-1, 8)), link_indices=ee_index) # pytorch_kinematics only supprts one additional dim
        # jac = jac.reshape((N, (T+1), jac.shape[-2], jac.shape[-1]))
        # jac = full_to_partial_state(jac)
        # jac = rotate_jac(jac, env.world_trans.get_matrix()[0, :3, :3]) # rotate jac into world frame 

        # 1 st equality constraints, the finger tip movement in the tangential direction should match with the commanded movement
        # jac = jac[:, :T] # do not consider the last state, since no action matches the last state
        # translation_jac = jac[:, :, :3, :]
        # delta_p_action = torch.matmul(translation_jac, action.unsqueeze(-1)).squeeze(-1) # FK(q_{i,t) + delta q_{i,t}} - FK(q_{i,t})
        # NOTE: we could use jacobian to simulate the delta action to speed up, but we use ground truth now
        m_desired = world_trans.compose(fk_desired_dict[finger_name])
        ee_p_desired = m_desired.transform_points(points_finger_frame).reshape((N, T, 1, 3)).squeeze(-2)
        delta_p_action = ee_p_desired - ee_p[:, :-1]
        
        delta_p = ee_p[:, 1:] - ee_p[:, :-1]
        # get the surface normal
        radial_vec = ee_p[:, :-1] - valve_location.unsqueeze(0).unsqueeze(0) # do not consider action at the final timestep
        radial_vec[:,:,1] = 0 # do not consider y axis
        # in our simple task, might need to ignore one dimension
        surface_normal = - radial_vec / torch.linalg.norm(radial_vec, dim=-1).unsqueeze(-1) # the normal goes inwards for friction cone computation
        tan_delta_p_action = delta_p_action - (delta_p_action.unsqueeze(-2)@surface_normal.unsqueeze(-1)).squeeze(-1) * surface_normal
        tan_delta_p = delta_p - (delta_p.unsqueeze(-2) @ surface_normal.unsqueeze(-1)).squeeze(-1) * surface_normal
        # constraint_list.append((tan_delta_p_action - tan_delta_p).reshape(N, -1))
        # NOTE DEBUG ONLY: consider all components of delta_p and delta_p_action
        constraint_list.append((delta_p_action - delta_p).reshape(N, -1))

        #2nd constraint, finger tip has to be on the surface
        if finger_name == finger_names[0]:
            # for thumb, it is tracking a differnet theta angles, which has an offset.
            # NOTE: make leads to bug if we want to use the original theta
            nominal_theta = theta - initial_valve_angle - valve_offset_for_index
        elif finger_name == finger_names[1]:
            nominal_theta = theta - initial_valve_angle - valve_offset_for_thumb

        nominal_x = torch.sin(nominal_theta) * r + valve_location[0]
        nominal_z = torch.cos(nominal_theta) * r + valve_location[2]
        constraint_list.append((ee_p[:, 1:, 0] - nominal_x).reshape(N, -1)) # do not consider the current time step
        constraint_list.append((ee_p[:, 1:, 2] - nominal_z).reshape(N, -1))
    
    return torch.cat(constraint_list, dim=-1)

def get_joint_inequality_constraint(qu, chain, start_q):
    """

    :param qu: torch.Tensor (Batch, T, DoF) joint positions
    :param chain: pytorch kinematics chain

    :return constraints: torch.Tensor(N, 1) contsraints as specified above

    """
    N = qu.shape[0]
    T = qu.shape[1]
    valve_offset_for_index = -np.pi / 2.0
    valve_offset_for_thumb = np.pi / 2.0  # * 0.99
    theta = qu[:, :, 8]
    index_q = qu[:, :, :4]
    thumb_q = qu[:, :, 4:8]
    action = qu[:, :, 9:17]
    index_action = action[:, :, :4]
    thumb_action = action[:, :, 4:]
    finger_qs = [index_q, thumb_q]
    finger_names = [index_ee_name, thumb_ee_name]
    action_list = [index_action, thumb_action]
    radii = [0.040, 0.038]
    constraint_list = []
    q = torch.cat((start_q.repeat((N, 1, 1)), qu[:, :, :9]), dim=1) # add the current time step in
    fk_dict = chain.forward_kinematics(partial_to_full_state(q[:, :, :8].reshape(-1, 8)), frame_indices=frame_indices) # pytorch_kinematics only supprts one additional dim

    for finger_q, finger_name, r, ee_index, finger_action in zip(finger_qs, finger_names, radii, frame_indices, action_list):
        m = world_trans.compose(fk_dict[finger_name])
        points_finger_frame = torch.tensor([0.00, 0.03, 0.00], device=qu.device).unsqueeze(0)
        normals_finger_tip_frame = torch.tensor([0.0, 0.0, 1.0], device=qu.device).unsqueeze(0)
        # it is just the thrid column of the rotation matrix
        normals_finger_tip_frame = normals_finger_tip_frame / torch.norm(normals_finger_tip_frame, dim=1, keepdim=True)
        ee_p = m.transform_points(points_finger_frame).reshape((N, (T+1), 1, 3))
        ee_p = ee_p.squeeze(-2)
        # normals_world_frame = m.transform_normals(normals_finger_tip_frame).reshape(3)
        ee_mat = m.get_matrix().reshape(N, (T+1), 4, 4)[:, :, :3,:3]
        jac = chain.jacobian(partial_to_full_state(q[:, :, :8].reshape(-1, 8)), link_indices=ee_index) # pytorch_kinematics only supprts one additional dim
        jac = jac.reshape((N, (T+1), jac.shape[-2], jac.shape[-1]))
        jac = full_to_partial_state(jac)
        jac = rotate_jac(jac, env.world_trans.get_matrix()[0, :3, :3]) # rotate jac into world frame 
        
        # friction cone constraints
        jac = jac[:, :T] # do not consider the last state, since no action matches the last state
        translation_jac = jac[:, :, :3, :]
        translation_jac_T = translation_jac.permute(0, 1, 3, 2)
        force = (torch.linalg.inv(torch.matmul(translation_jac, translation_jac_T)) @ translation_jac) @ (env.joint_stiffness * torch.eye(8).to(jac.device)) @ action.unsqueeze(-1)
        radial_vec = ee_p[:, :-1] - valve_location.unsqueeze(0).unsqueeze(0) # do not consider action at the final timestep
        radial_vec[:,:,1] = 0 # do not consider y axis
        # in our simple task, might need to ignore one dimension
        surface_normal = - radial_vec / torch.linalg.norm(radial_vec, dim=-1).unsqueeze(-1) # the normal goes inwards for friction cone computation
        f_normal = (surface_normal.unsqueeze(-2) @ force).squeeze(-1) * surface_normal
        f_tan = force.squeeze(-1) - f_normal
        # TODO: f_tan @ f_normal is not 0, but a relatively small value.  might need to double check
        # NOTE: DEBUG ONLY, remove friction cone constraints
        constraint_list.append(torch.linalg.norm(f_tan, dim=-1) - friction_coefficient * torch.linalg.norm(f_normal, dim=-1))
        
        # y range of the finger tip should be within a range
        constraint_y_1 = -(valve_location[1] - ee_p[:, 1:, 1]) + 0.02 # do not consider the current time step
        constraint_y_2 = (valve_location[1] - ee_p[:, 1:, 1]) - 0.065
        constraint_list.append(torch.cat((constraint_y_1, constraint_y_2), dim=-1))


    
    return torch.cat(constraint_list, dim=-1)


def do_trial(env, params, fpath):
    "only turn the valve once"
    state = env.get_state()
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None

    start = state['q'].reshape(9).to(device=params['device'])
    # start = torch.cat((state['q'].reshape(10), torch.zeros(1).to(state['q'].device))).to(device=params['device'])
    chain.to(device=params['device'])

    if params['controller'] == 'csvgd':
        problem = AllegroValveProblem(start,
                                      params['goal'],
                                      params['T'],
                                      device=params['device'],
                                      chain=chain,
                                      finger_name='index',
                                      valve_location=valve_location)
        controller = Constrained_SVGD_MPC(problem, params)
    else:
        raise ValueError('Invalid controller')

    actual_trajectory = []
    duration = 0
    for k in range(params['num_steps']):
        state = env.get_state()
        start = state['q'].reshape(9).to(device=params['device'])
        # start = torch.cat((state['q'].reshape(9), torch.zeros(1).to(state['q'].device))).to(device=params['device'])

        actual_trajectory.append(state['q'].reshape(9).clone())
        # if k > 0:
        #     torch.cuda.synchronize()
        #     start_time = time.time()
        # if k == 0:
        start_time = time.time()
        best_traj, trajectories = controller.step(start)
        print(f"solve time: {time.time() - start_time}")

        # add trajectory lines to sim
        add_trajectories(trajectories, best_traj, chain)

        # process the action
        ## end effector force to torque
        x = best_traj[0, :problem.dx+problem.du]
        x = x.reshape(1, problem.dx+problem.du)
        action = x[:, problem.dx:problem.dx+problem.du].to(device=env.device)
        action = action + start.unsqueeze(0)[:, :8] # NOTE: this is required since we define action as delta action
        env.step(action)
        equality_eval, _, _ = problem._con_eq(best_traj.unsqueeze(0))
        print(f"max equality constraint violation: {equality_eval.max()}")
        inequality_eval, _, _ = problem._con_ineq(best_traj.unsqueeze(0))
        # index_jac = rotate_jac(chain.jacobian(partial_to_full_state(start[:8]), link_indices=index_ee_link)[0][:3, :4]) # only cares about the translation
        # thumb_jac = rotate_jac(chain.jacobian(partial_to_full_state(start[:8]), link_indices=thumb_ee_link)[0][:3, 12:16])
        # index_torque = index_jac.T @ action[0, :3]
        # thumb_torque = thumb_jac.T @ action[0, 3:6]
        # action = torch.cat((index_torque, thumb_torque)).unsqueeze(0)

        # zero_joint_q = torch.zeros((1, 8)).float().to(env.device)
        # delta_q = env.kp_inv @ torch.cat((action[:, :4], zero_joint_q, action[:, 4:8]), dim=-1).unsqueeze(-1)
        # delta_q = delta_q.squeeze(-1)
        # desired_q = x[:8].unsqueeze(0)
        # if k <=100:
        #     "position control in the first step to approach the valve"
        #     print("[INFO]: position + force control")
        #     env.control_mode = 'joint_impedance'
        #     action = desired_q + torch.cat((delta_q[:, :4], delta_q[:, 12:16]), dim=-1)
        #     env.step(action)
        # else:
        #     print("[INFO]: using force control")
        #     # env.control_mode = 'joint_torque_position'
        #     env.control_mode = 'joint_torque'
        #     env.step(action)
            # action = state['q'].reshape(9)[:8] + torch.cat((delta_q[:, :4], delta_q[:, 12:16]), dim=-1)


        # distance2surface = torch.sqrt((best_traj_ee[:, 2] - valve_location[2].unsqueeze(0)) ** 2 + (best_traj_ee[:, 0] - valve_location[0].unsqueeze(0))**2)
        distance2goal = (env.get_state()['q'][:, -1] - params['goal']).detach().cpu().item()
        print(distance2goal)

        gym.clear_lines(viewer)

    state = env.get_state()
    state = state['q'].reshape(9).to(device=params['device'])

    # now weee want to turn it again!

    # actual_trajectory.append(state.clone())
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 9)
    problem.T = actual_trajectory.shape[0]
    constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    final_distance_to_goal = actual_trajectory[:, -1] - params['goal']
    # final_distance_to_goal = torch.linalg.norm(
    #     chain.forward_kinematics(actual_trajectory[:, :7].reshape(-1, 7)).reshape(-1, 4, 4)[:, :2, 3] - params['goal'].unsqueeze(0),
    #     dim=1
    # )

    print(f'Controller: {params["controller"]} Final distance to goal: {torch.min(final_distance_to_goal)}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')

    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
             constr=constraint_val.cpu().numpy(),
             d2goal=final_distance_to_goal.cpu().numpy())
    return torch.min(final_distance_to_goal).cpu().numpy()


# def turn(env, params, fpath):
#     "turn the valve multiple times"
#     state = env.get_state()
#     if params['visualize']:
#         env.frame_fpath = fpath
#         env.frame_id = 0
#     else:
#         env.frame_fpath = None
#         env.frame_id = None

#     # ee_pos, ee_ori = state['ee_pos'], state['ee_ori']
#     # start = torch.cat((ee_pos, ee_ori), dim=-1).reshape(7).to(device=params['device'])
#     start = torch.cat((state['q'].reshape(9), torch.zeros(1))).to(device=params['device'])
#     chain.to(device=params['device'])
#     world_trans.to(device=params['device'])

#     # first things first, we grasp the valve by setting a goal of zero
#     problem = AllegroValveProblem(start,
#                                   0 * params['goal'],
#                                   4,
#                                   device=params['device'],
#                                   chain=chain,
#                                   finger_name='index',
#                                   valve_location=valve_location,
#                                   initial_valve_angle=0)
#     controller = Constrained_SVGD_MPC(problem, params)

#     default_start_pos = env.get_state()['q'][:, :8].clone()

#     # get start and initial
#     state = env.get_state()
#     start = torch.cat((state['q'].reshape(9), torch.zeros(1))).to(device=params['device'])
#     best_traj, _ = controller.step(start)

#     # we will just execute this open loop
#     for x in best_traj[:, :8]:
#         env.step(x.reshape(-1, 8).to(device=env.device))

#     state = env.get_state()
#     start = torch.cat((state['q'].reshape(9), torch.zeros(1))).to(device=params['device'])
#     # reset controller
#     problem = AllegroValveProblem(start,
#                                   params['goal'],
#                                   params['T'],
#                                   device=params['device'],
#                                   chain=chain,
#                                   finger_name='index',
#                                   valve_location=valve_location,
#                                   initial_valve_angle=0)
#     controller = Constrained_SVGD_MPC(problem, params)

#     # now ready to do the turn
#     actual_trajectory = []
#     num_turns = 1
#     for k in range(params['num_steps']):
#         state = env.get_state()
#         start = torch.cat((state['q'].reshape(9), torch.zeros(1))).to(device=params['device'])

#         actual_trajectory.append(state['q'].reshape(9).clone())
#         best_traj, trajectories = controller.step(start)
#         x = best_traj[0, :problem.dx]

#         # add trajectory lines to sim
#         plotting_traj = torch.cat((start.unsqueeze(0), best_traj), dim=0)
#         add_trajectories(trajectories, plotting_traj, chain)

#         # step sim
#         action = x.reshape(1, 9)[:, :8].to(device=env.device)
#         env.step(action)

#         # check if task complete
#         distance2goal = torch.linalg.norm(env.get_state()['q'][:, -1] - params[
#             'goal']).detach().cpu().item()
#         gym.clear_lines(viewer)
#         if distance2goal < 0.1:
#             break


#     # Now we will do the turn again
#     # first we need to release the valve in a way that does not disturb the valve
#     # use Jacobian IK to back the thumb up
#     q = env.get_state()['q'][:, :8]
#     eps = 5e-3
#     eye = torch.eye(3).reshape(1, 3, 3).to(device=env.device)
#     # let's do some ik
#     for _ in range(10):
#         J = chain.jacobian(partial_to_full_state(q), link_indices=torch.tensor([thumb_ee_link],
#                                                           device=params['device']))[:, :3, -4:]

#         # get update in robot frame
#         dx = world_trans.inverse().transform_normals(torch.tensor([[-1.0, 0.0, 0.0]],
#                                                                   device=params['device']).reshape(1, 3)).reshape(1, 3, 1)
#         # joint update
#         dq = J.permute(0, 2, 1) @ torch.linalg.inv(J @ J.permute(0, 2, 1) + 1e-5 * eye) @ dx
#         q[:, 4:] += eps * dq.reshape(1, 4)
#     env.step(q)

#     # Now go to default start position
#     env.step(default_start_pos)

#     # First plan to contact
#     problem = AllegroValveProblem(start,
#                                   0 * params['goal'],
#                                   4,
#                                   device=params['device'],
#                                   chain=chain,
#                                   finger_name='index',
#                                   valve_location=valve_location,
#                                   initial_valve_angle=0)
#     controller = Constrained_SVGD_MPC(problem, params)

#     # get start and initial
#     state = env.get_state()
#     start = torch.cat((state['q'].reshape(9), torch.zeros(1))).to(device=params['device'])

#     # we will offset by 90 degrees
#     start[-1] = +np.pi / 2
#     best_traj, _ = controller.step(start)

#     # we will just execute this open loop
#     for x in best_traj[:, :8]:
#         env.step(x.reshape(-1, 8).to(device=env.device))

#     # now ready to do the second turn
#     start = state['q'].reshape(9).to(device=params['device'])
#     # reset controller
#     problem = AllegroValveProblem(start,
#                                   params['goal'],
#                                   params['T'],
#                                   device=params['device'],
#                                   chain=chain,
#                                   finger_name='index',
#                                   valve_location=valve_location,
#                                   initial_valve_angle=0)
#     controller = Constrained_SVGD_MPC(problem, params)

#     actual_trajectory = []
#     num_turns += 1
#     for k in range(params['num_steps']):
#         state = env.get_state()
#         start = torch.cat((state['q'].reshape(9), torch.zeros(1))).to(device=params['device'])
#         start[-1] = +np.pi / 2
#         start.append(0) # add an velocity
#         actual_trajectory.append(state['q'].reshape(10).clone())
#         best_traj, trajectories = controller.step(start)
#         x = best_traj[0, :problem.dx]

#         # add trajectory lines to sim
#         plotting_traj = torch.cat((start.unsqueeze(0), best_traj), dim=0)
#         add_trajectories(trajectories, plotting_traj, chain)

#         # step sim
#         action = x.reshape(1, 9)[:, :8].to(device=env.device)
#         env.step(action)

#         gym.clear_lines(viewer)
#     # check how close we are to the goal
#     distance2goal = torch.linalg.norm(env.get_state()['q'][:, -1] - num_turns * params['goal']).detach().cpu().item()
#     print(f'Final goal distance: {distance2goal}')

#     state = env.get_state()
#     state = state['q'].reshape(9).to(device=params['device'])
#     actual_trajectory.append(state.clone())
#     actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 9)
#     problem.T = actual_trajectory.shape[0]
#     constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
#     return 0


def add_trajectories(trajectories, best_traj, chain):
    M = len(trajectories)
    if M > 0:
        # print(partial_to_full_state(best_traj[:, :8]).shape)
        # frame_indices = torch.tensor([index_ee_link, thumb_ee_link])
        best_traj_ee_fk_dict = chain.forward_kinematics(partial_to_full_state(best_traj[:, :8]),
                                                        frame_indices=frame_indices)

        index_best_traj_ee = world_trans.compose(best_traj_ee_fk_dict[index_ee_name])  # .get_matrix()
        thumb_best_traj_ee = world_trans.compose(best_traj_ee_fk_dict[thumb_ee_name])  # .get_matrix()

        # index_best_traj_ee = best_traj_ee_fk_dict[index_ee_name].compose(world_trans).get_matrix()
        # thumb_best_traj_ee = best_traj_ee_fk_dict[thumb_ee_name].compose(world_trans).get_matrix()
        points_finger_frame = torch.tensor([0.00, 0.02, 0.02], device=best_traj.device).unsqueeze(0)
        index_best_traj_ee = index_best_traj_ee.transform_points(points_finger_frame).squeeze(1)
        thumb_best_traj_ee = thumb_best_traj_ee.transform_points(points_finger_frame).squeeze(1)
        # index_best_traj_ee = index_best_traj_ee[:, :3, 3]
        # thumb_best_traj_ee = thumb_best_traj_ee[:, :3, 3]

        traj_line_colors = np.random.random((3, M)).astype(np.float32)
        thumb_colors = np.array([0, 1, 0]).astype(np.float32)
        index_colors = np.array([0, 0, 1]).astype(np.float32)
        force_colors = np.array([0, 1, 1]).astype(np.float32)
        for e in env.envs:
            index_p = env.get_state()['index_pos'].reshape(1, 3).to(device=params['device'])
            thumb_p = env.get_state()['thumb_pos'].reshape(1, 3).to(device=params['device'])
            # p = torch.stack((s[:3].reshape(1, 3).repeat(M, 1),
            #                  trajectories[:, 0, :3]), dim=1).reshape(2 * M, 3).cpu().numpy()
            # p_best = torch.stack((s[:3].reshape(1, 3).repeat(1, 1), best_traj_ee[0, :3].unsqueeze(0)), dim=1).reshape(2, 3).cpu().numpy()
            # p[:, 2] += 0.005
            # gym.add_lines(viewer, e, 1, p_best, best_traj_line_colors)
            # gym.add_lines(viewer, e, M, p, traj_line_colors)
            # index_best_traj_ee[:, 0] -= 0.05
            # thumb_best_traj_ee[:, 0] -= 0.05
            # index_best_force = best_traj[0, 10:13] * 2
            # index_best_force = torch.cat((index_p.repeat(1, 1), (index_p+index_best_force.unsqueeze(0))), dim=0).cpu().numpy()
            # thumb_best_force = best_traj[0, 13:16] * 2
            # thumb_best_force = torch.cat((thumb_p.repeat(1, 1), (thumb_p + thumb_best_force.unsqueeze(0))), dim=0).cpu().numpy()
            # gym.add_lines(viewer, e, 1, index_best_force, force_colors)
            # gym.add_lines(viewer, e, 1, thumb_best_force, force_colors)

            T = best_traj.shape[0]
            for t in range(T - 1):
                # p = torch.stack((trajectories[:, t, :3], trajectories[:, t + 1, :3]), dim=1).reshape(2 * M, 3)
                # p = p.cpu().numpy()
                # p[:, 2] += 0.01
                index_p_best = torch.stack((index_best_traj_ee[t, :3], index_best_traj_ee[t + 1, :3]), dim=0).reshape(2,
                                                                                                                      3).cpu().numpy()
                thumb_p_best = torch.stack((thumb_best_traj_ee[t, :3], thumb_best_traj_ee[t + 1, :3]), dim=0).reshape(2,
                                                                                                                      3).cpu().numpy()
                gym.add_lines(viewer, e, 1, index_p_best, index_colors)
                gym.add_lines(viewer, e, 1, thumb_p_best, thumb_colors)
                # gym.add_lines(viewer, e, M, p, traj_line_colors)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)


if __name__ == "__main__":
    # get config
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro.yaml').read_text())
    from tqdm import tqdm

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
        goal = -0.5 * torch.tensor([np.pi])
        # goal = goal + 0.025 * torch.randn(1) + 0.2
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
            # final_distance_to_goal = turn(env, params, fpath)

            if controller not in results.keys():
                results[controller] = [final_distance_to_goal]
            else:
                results[controller].append(final_distance_to_goal)
        print(results)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
