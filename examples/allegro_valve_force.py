import numpy as np
from isaacgym.torch_utils import quat_apply
from isaac_victor_envs.tasks.allegro import AllegroValveTurningEnv, orientation_error, quat_change_convention
from isaac_victor_envs.utils import get_assets_dir

import torch
import time
import yaml
import pathlib
from functools import partial
from torch.func import vmap, jacrev, hessian, jacfwd
# from functorch import vmap, jacrev, hessian, jacfwd

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
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
# import pytorch3d.transforms as tf

from pytorch_volumetric import RobotSDF, RobotScene, MeshSDF
import matplotlib.pyplot as plt

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

valve_location = torch.tensor([0.85, 0.70, 1.405]).to('cuda:0') # the root of the valve
# instantiate environment
friction_coefficient = 0.95 # this one is used for planning, not simulation
# env = AllegroValveTurningEnv(1, control_mode='joint_torque_position',
#                              viewer=True, steps_per_action=60)
env = AllegroValveTurningEnv(1, control_mode='joint_impedance', use_cartesian_controller=False,
                             viewer=True, steps_per_action=60, valve_velocity_in_state=False, friction_coefficient=1.0)
world_trans = env.world_trans


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
def state2ee_pos(state, finger_name):
    """
    :params state: B x 8 joint configuration for full hand
    :return ee_pos: B x 3 position of ee

    """
    fk_dict = chain.forward_kinematics(partial_to_full_state(state), frame_indices=frame_indices)
    m = world_trans.compose(fk_dict[finger_name])
    points_finger_frame = torch.tensor([0.00, 0.03, 0.00], device=m.device).unsqueeze(0)
    ee_p = m.transform_points(points_finger_frame)
    return ee_p

class PositionControlConstrainedSteinTrajOpt(ConstrainedSteinTrajOpt):
    def __init__(self, problem, params):
        super().__init__(problem, params)
    def _clamp_in_bounds(self, xuz):
        N = xuz.shape[0]
        min_x = self.problem.x_min.reshape(1, 1, -1).repeat(1, self.problem.T, 1)
        max_x = self.problem.x_max.reshape(1, 1, -1).repeat(1, self.problem.T, 1)
        if self.problem.dz > 0:
            min_x = torch.cat((min_x, -1e3 * torch.ones(1, self.problem.T, self.problem.dz)), dim=-1)
            max_x = torch.cat((max_x, 1e3 * torch.ones(1, self.problem.T, self.problem.dz)), dim=-1)

        torch.clamp_(xuz, min=min_x.to(device=xuz.device).reshape(1, -1),
                     max=max_x.to(device=xuz.device).reshape(1, -1))
        xuz_copy = xuz.reshape((N, self.problem.T, -1))
        robot_joint_angles = xuz_copy[:, :-1, :8]
        robot_joint_angles = torch.cat((self.problem.start[:8].reshape((1, 1, 8)).repeat((N, 1, 1)), robot_joint_angles), dim=1)
        min_u = self.problem.robot_joint_x_min.repeat((N, self.problem.T, 1)).to(xuz.device) - robot_joint_angles
        max_u = self.problem.robot_joint_x_max.repeat((N, self.problem.T, 1)).to(xuz.device) - robot_joint_angles
        min_x = min_x.repeat((N,1,1)).to(device=xuz.device)
        max_x = max_x.repeat((N,1,1)).to(device=xuz.device)
        min_x[:, :, 9:17] = min_u
        max_x[:, :, 9:17] = max_u
        torch.clamp_(xuz, min=min_x.reshape((N,-1)), max=max_x.reshape((N, -1)))

class PositionControlConstrainedSVGDMPC(Constrained_SVGD_MPC):

    def __init__(self, problem, params):
        super().__init__(problem, params)
        self.solver = PositionControlConstrainedSteinTrajOpt(problem, params)
    

class AllegroValveProblem(ConstrainedSVGDProblem):

    def __init__(self, start, goal, T, chain, valve_location, finger_name, initial_valve_angle=0, collision_checking=False, device='cuda:0'):
        """
        valve location: the root location of the valve
        initial_valve_angle: it is designed for continuously turning the valve. For each turn, 
        the valve might not be exactly at 0 degree, we need to subtract that out. 
        If we only care about one turn, we can leave it to 0
        """
        super().__init__(start, goal, T, device)
        if collision_checking:
            self.dz = (4+1) * 2 # 1 means the collision checking
        else:
            self.dz = 4 * 2 
        self.dh = self.dz * T # inequality
        self.dg = 4 * 2 * T #equality  # though dynamics function constraints only have 2*(T-1), we make it 2 * T to be compatible with the code style
        self.dx = 9 # position of finger joints and theta and theta dot.
        self.du = 8 # finger joints delta action. 
        # NOTE: the decision variable x means x_1 to x_T, but for the action u, it means u_0 to u_{T-1}. 
        # NOTE: The timestep of x and u doesn't match
        self.dt = 0.1
        self.T = T
        self.start = start
        self.goal = goal
        self.K = rbf_kernel
        self.squared_slack = True
        self.compute_hess = True
        self._collision_checking_flag = collision_checking

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


        # add collision checking
        # collision check all of the non-finger tip links
        collision_check_oya = ['allegro_hand_oya_finger_link_13',
                               'allegro_hand_oya_finger_link_14',
                               # 'allegro_hand_oya_finger_link_15'
                               ]
        collision_check_hitosashi = [  # 'allegro_hand_hitosashi_finger_finger_link_3',
            'allegro_hand_hitosashi_finger_finger_link_2',
            'allegro_hand_hitosashi_finger_finger_link_1'
        ]

        asset_valve = get_assets_dir() + '/valve/valve.urdf'
        chain_valve = pk.build_chain_from_urdf(open(asset_valve).read())
        chain_valve = chain_valve.to(device=device)
        valve_sdf = pv.RobotSDF(chain_valve, path_prefix=get_assets_dir() + '/valve')
        robot_sdf = pv.RobotSDF(chain, path_prefix=get_assets_dir() + '/xela_models')

        # TODO: retrieve transformations from environment rather than hard-coded
        p = [0.89, 0.52, 1.375]
        r = [0.2425619, 0.2423688, 0.6639723, 0.6645012]
        rob_trans = pk.Transform3d(pos=torch.tensor(p, device=device),
                                   rot=torch.tensor([r[3], r[0], r[1], r[2]], device=device),
                                   device=device)

        scene_trans = rob_trans.inverse().compose(pk.Transform3d(device=device).translate(0.85, 0.75, 0.705))

        # TODO: right now we are using seperate collision checkers for each finger to avoid gradients swapping
        # between fingers - alteratively we can get the collision checker to return a list of collisions and gradients batched
        self.index_scene = pv.RobotScene(robot_sdf, valve_sdf, scene_trans,
                                         collision_check_links=collision_check_hitosashi,
                                         softmin_temp=100.0)
        self.thumb_scene = pv.RobotScene(robot_sdf, valve_sdf, scene_trans,
                                         collision_check_links=collision_check_oya,
                                         softmin_temp=100.0)
        # for honda hand
        index_x_max = torch.tensor([0.47, 1.6099999999, 1.7089999, 1.61799999])
        index_x_min = torch.tensor([-0.47, -0.195999999999, -0.174000000, -0.227])
        thumb_x_max = torch.tensor([1.396, 1.1629999999999, 1.644, 1.71899999])
        thumb_x_min = torch.tensor([0.26, -0.1049999999, -0.1889999999, -0.162])

        valve_x_max = torch.tensor([10.0 * np.pi])
        valve_x_min = torch.tensor([-10.0 * np.pi])
        self.x_max = torch.cat((index_x_max, thumb_x_max, valve_x_max))
        self.x_min = torch.cat((index_x_min, thumb_x_min, valve_x_min))
        self.robot_joint_x_max = torch.cat([index_x_max, thumb_x_max])
        self.robot_joint_x_min = torch.cat([index_x_min, thumb_x_min])

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
        J, grad_J, hess_J = self.cost(x), self.grad_cost(x), self.hess_cost(x)

        N = x.shape[0]
        return (self.alpha * J.reshape(N),
                self.alpha * grad_J.reshape(N, -1),
                self.alpha * hess_J.reshape(N, self.T * (self.dx+self.du), self.T * (self.dx+self.du)))
    
    def _con_eq(self, xu, compute_grads=True, compute_hess=False):
        N = xu.shape[0]
        T = xu.shape[1]
        g, grad_g, hess_g = self._equality_constraints.eval(xu.reshape(N, T, self.dx+self.du), compute_grads, compute_hess=compute_hess)

        g = g.reshape(N, -1)

        if not compute_grads:
            return g, None, None
            # Expand gradient to include time dimensions
        grad_g = grad_g.reshape(N, g.shape[1], self.T * (self.dx + self.du))
        hess_g = hess_g.reshape(N, g.shape[1], self.T * (self.dx + self.du), self.T * (self.dx + self.du))
        return g, grad_g, hess_g
    
    def _collision_check_finger(self, x, finger_scene, compute_grads=True, compute_hess=True):
        N = x.shape[0]
        q = x[:, :8]
        theta = x[:, -1].unsqueeze(-1)
        full_q = partial_to_full_state(q)
        ret_scene = finger_scene.scene_collision_check(full_q, theta,
                                                       compute_gradient=compute_grads,
                                                       compute_hessian=compute_hess)
        h = -ret_scene.get('sdf').unsqueeze(1)
        grad_h_q = ret_scene.get('grad_sdf', None)
        hess_h_q = ret_scene.get('hess_sdf', None)
        grad_h_theta = ret_scene.get('grad_env_sdf', None)
        hess_h_theta = ret_scene.get('hess_env_sdf', None)
        grad_h = None
        hess_h = None
        if grad_h_q is not None:
            grad_h_action = torch.zeros((h.shape[0], self.du)).to(grad_h_q.device)
            grad_h = -torch.cat((grad_h_q[:, (0, 1, 2, 3, 12, 13, 14, 15)], grad_h_theta, grad_h_action), dim=-1).unsqueeze(1)
        if hess_h_q is not None:
            hess_h = torch.zeros(N, 1, self.dx + self.du, self.dx + self.du, device=x.device)
            hess_h_q = hess_h_q[:, (0, 1, 2, 3, 12, 13, 14, 15)]
            hess_h_q = hess_h_q[:, :, (0, 1, 2, 3, 12, 13, 14, 15)]
            hess_h[:, :, :8, :8] = -hess_h_q.unsqueeze(1)
            hess_h[:, :, 8, 8] = -hess_h_theta.reshape(N, -1)
        return h, grad_h, hess_h

    def _collision_check(self, x, compute_grads=True, compute_hess=True):
        h_thumb, grad_h_thumb, hess_h_thumb = self._collision_check_finger(x,
                                                                           self.thumb_scene,
                                                                           compute_grads=compute_grads,
                                                                           compute_hess=compute_hess)

        h_index, grad_h_index, hess_h_index = self._collision_check_finger(x,
                                                                           self.index_scene,
                                                                           compute_grads=compute_grads,
                                                                           compute_hess=compute_hess)
        # for debug only
        # h_thumb += 0.04
        # h_index += 0.04
        h = torch.cat((h_thumb, h_index), dim=1)
        if not compute_grads:
            return h, None, None

        grad_h = torch.cat((grad_h_thumb, grad_h_index), dim=1)

        if not compute_hess:
            return h, grad_h, None

        hess_h = torch.cat((hess_h_thumb, hess_h_index), dim=1)

        return h, grad_h, hess_h
    
    def _con_ineq(self, xu, compute_grads=True, compute_hess=False):
        N = xu.shape[0]
        T = xu.shape[1]
        h, grad_h, hess_h = self._inequality_constraints.eval(xu.reshape(-1, T, self.dx+self.du), compute_grads, compute_hess=compute_hess)
        non_collision_con_dim = h.shape[1]        
        if self._collision_checking_flag:
            # TODO: hard code the hessian for collision check for now, need to fix it in the future
            h_collision, grad_h_collision, hess_h_collision = self._collision_check(xu.reshape(-1, self.dx+self.du),
                                                                                    compute_grads=compute_grads,
                                                                                    compute_hess=False)
            h_collision = h_collision.reshape(N, T, 2)
            h_collision = h_collision.reshape(N, T * 2)
            h = torch.cat((h, h_collision), dim=1)

        h = h.reshape(N, -1)
        N = xu.shape[0]
        if not compute_grads:
            return h, None, None
            # Expand gradient to include time dimensions
        if compute_grads:
            grad_h = grad_h.reshape(N, non_collision_con_dim, self.T * (self.dx + self.du))
            if self._collision_checking_flag:
                # grad_h_collision = torch.cat((grad_h_collision, torch.zeros((grad_h_collision.shape[0], grad_h_collision.shape[1], self.du)).to(grad_h_collision.device)), dim=-1)
                grad_h_collision = grad_h_collision.reshape(N, self.T, -1, self.dx+self.du).permute(0, 2, 3, 1)
                grad_h_collision = torch.diag_embed(grad_h_collision)  # (N, n_constraints, dx + du, T, T)
                grad_h_collision = grad_h_collision.permute(0, 3, 1, 4, 2).reshape(N, -1, self.T * (self.dx+self.du))
                grad_h = torch.cat((grad_h, grad_h_collision), dim=1)
                grad_h = grad_h.reshape(N, h.shape[1], self.T * (self.dx + self.du))
        if compute_hess: 
            # TODO: use zero for now, we might need to fix it if we want to use hessian
            # hess_h_collision = hess_h_collision.reshape(N, self.T, -1, self.dx, self.dx).permute(0, 2, 3, 4, 1)
            # hess_h_collision = torch.diag_embed(torch.diag_embed(hess_h_collision))  # (N, n_constraints, dx + du, dx + du, T, T, T)
            # hess_h_collision = hess_h_collision.permute(0, 4, 1, 5, 2, 6, 3).reshape(N, -1,
            #                                                     self.T * (self.dx),
            #                                                     self.T * (self.dx))
            if self._collision_checking_flag:
                hess_h_collision = torch.zeros((N, 2 * self.T, self.T, self.dx+self.du, self.T, self.dx+self.du)).to(hess_h.device)
                hess_h = torch.cat((hess_h, hess_h_collision), dim=1)
            hess_h = hess_h.reshape(N, h.shape[1], self.T * (self.dx + self.du), self.T * (self.dx + self.du))
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
        G, dG, hessG = self.combined_constraints(augmented_trajectory, compute_hess=self.compute_hess)

        if hess_J is not None:
            hess_J_ext = torch.zeros(N, self.T, self.dx + self.du + self.dz, self.T, self.dx + self.du + self.dz,
                                     device=x.device)
            hess_J_ext[:, :, :self.dx + self.du, :, :self.dx + self.du] = hess_J.reshape(N, self.T, self.dx + self.du,
                                                                                         self.T, self.dx + self.du)
            hess_J = hess_J_ext.reshape(N, self.T * (self.dx + self.du + self.dz),
                                        self.T * (self.dx + self.du + self.dz))

        if hessG is not None:
            hessG.detach_()
        return grad_J.detach(), hess_J, K.detach(), grad_K.detach(), G.detach(), dG.detach(), hessG

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
        u = torch.randn(N, self.T, self.du, device=self.device) / 20
        # u = torch.zeros((N, self.T, self.du), device=self.device)
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
    state = xu[:, :9] # state dim = 9
    state = torch.cat((start.reshape(1, 9), state), dim=0) # combine the first time step into it

    action = xu[:, 9: 9+6] # action dim = 6
    action_cost = torch.sum(action ** 2)

    smoothness_cost = torch.sum((state[1:] - state[:-1])**2)
    smoothness_cost += 20 * torch.sum((state[1:, -1] - state[:-1, -1])**2) # weight the smoothness of theta more

    goal_cost = (50 * (state[-1, -1] - goal)**2)[0]

    # this goal cost is not very informative and takes a long time for the gradients to back propagate to each state. 
    # thus, a better initialization is necessary for faster convergence
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

    def eval(self, qu, compute_grads=True, compute_hess=False):
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
        # if compute_hess:
        #     ddq = self.hess_constraint(qu)
        # else:
        #     ddq = torch.zeros(dq.shape + dq.shape[2:])
        # TODO: this is a hack for now to make it work. We should compute hessian in the future
        ddq = torch.zeros(dq.shape + dq.shape[2:])

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
    # radii = [0.040, 0.038]
    # radii = [0.033, 0.035]
    radii = [0.041, 0.038]
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

        "1 st equality constraints, the finger tip movement in the tangential direction should match with the commanded movement"
        # jac = jac[:, :T] # do not consider the last state, since no action matches the last state
        # translation_jac = jac[:, :, :3, :]
        # delta_p_action = torch.matmul(translation_jac, action.unsqueeze(-1)).squeeze(-1) # FK(q_{i,t) + delta q_{i,t}} - FK(q_{i,t})
        # NOTE: we could use jacobian to simulate the delta action to speed up, but we use ground truth now
        m_desired = world_trans.compose(fk_desired_dict[finger_name])
        ee_p_desired = m_desired.transform_points(points_finger_frame).reshape((N, T, 1, 3)).squeeze(-2)
        delta_p_desired = ee_p_desired - ee_p[:, :-1]
        
        delta_p = ee_p[:, 1:] - ee_p[:, :-1]
        # get the surface normal
        radial_vec = ee_p[:, :-1] - valve_location.unsqueeze(0).unsqueeze(0) # do not consider action at the final timestep
        radial_vec[:,:,1] = 0 # do not consider y axis
        # in our simple task, might need to ignore one dimension
        # surface_normal = - radial_vec / torch.linalg.norm(radial_vec, dim=-1).unsqueeze(-1) # the normal goes inwards for friction cone computation
        # tan_delta_p_desired = delta_p_desired - (delta_p_desired.unsqueeze(-2)@surface_normal.unsqueeze(-1)).squeeze(-1) * surface_normal
        # tan_delta_p = delta_p - (delta_p.unsqueeze(-2) @ surface_normal.unsqueeze(-1)).squeeze(-1) * surface_normal
        # constraint_list.append((tan_delta_p_desired - tan_delta_p).reshape(N, -1))
        # This version specifies a redundant constraint, which might not fit well with CSVTO
        surface_normal = - radial_vec / torch.linalg.norm(radial_vec, dim=-1).unsqueeze(-1) # the normal goes inwards for friction cone computation
        tan_delta_p = delta_p - (delta_p.unsqueeze(-2) @ surface_normal.unsqueeze(-1)).squeeze(-1) * surface_normal
        base_x = torch.tensor([1, 0, 0]).float().to(surface_normal.device)
        base_x_hat = base_x - (base_x.repeat((surface_normal.shape[0], surface_normal.shape[1], 1)).unsqueeze(-2) @ surface_normal.unsqueeze(-1)).squeeze(-1) * surface_normal
        # base_x_hat = tan_delta_p / torch.linalg.norm(tan_delta_p, dim=-1).unsqueeze(-1)
        base_y_hat = torch.cross(surface_normal, base_x_hat, dim=-1)
        tan_delta_p_desired = delta_p_desired - (delta_p_desired.unsqueeze(-2)@surface_normal.unsqueeze(-1)).squeeze(-1) * surface_normal
        tan_delta_p_desired_x_hat = (base_x_hat.unsqueeze(-2) @ tan_delta_p_desired.unsqueeze(-1)).squeeze((-1,-2))
        tan_delta_p_desired_y_hat = (base_y_hat.unsqueeze(-2) @ tan_delta_p_desired.unsqueeze(-1)).squeeze((-1,-2))
        tan_delta_p_x_hat = (base_x_hat.unsqueeze(-2) @ tan_delta_p.unsqueeze(-1)).squeeze((-1,-2))
        tan_delta_p_y_hat = (base_y_hat.unsqueeze(-2) @ tan_delta_p.unsqueeze(-1)).squeeze((-1,-2))
        constraint_list.append(tan_delta_p_x_hat - tan_delta_p_desired_x_hat)
        constraint_list.append(tan_delta_p_y_hat - tan_delta_p_desired_y_hat)

        "2nd constraint, finger tip has to be on the surface"
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
    # radii = [0.040, 0.038]
    # radii = [0.033, 0.035]
    radii = [0.041, 0.038]
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
        ee_p = m.transform_points(points_finger_frame).reshape((N, (T+1), 1, 3))
        ee_p = ee_p.squeeze(-2)
        # normals_world_frame = m.transform_normals(normals_finger_tip_frame).reshape(3)
        ee_mat = m.get_matrix().reshape(N, (T+1), 4, 4)[:, :, :3,:3]
        jac = chain.jacobian(partial_to_full_state(q[:, :, :8].reshape(-1, 8)), link_indices=ee_index) # pytorch_kinematics only supprts one additional dim
        jac = jac.reshape((N, (T+1), jac.shape[-2], jac.shape[-1]))
        jac = full_to_partial_state(jac)
        jac = rotate_jac(jac, env.world_trans.get_matrix()[0, :3, :3]) # rotate jac into world frame 
        
        " 1st constraint: friction cone constraints"
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
        constraint_list.append(torch.linalg.norm(f_tan, dim=-1) - friction_coefficient * torch.linalg.norm(f_normal, dim=-1))
        
        "2nd constraint y range of the finger tip should be within a range"
        constraint_y_1 = -(valve_location[1] - ee_p[:, 1:, 1]) + 0.02 # do not consider the current time step
        constraint_y_2 = (valve_location[1] - ee_p[:, 1:, 1]) - 0.065
        constraint_list.append(torch.cat((constraint_y_1, constraint_y_2), dim=-1))

        "3rd constraint: action should push in more than the actual movement"
        m_desired = world_trans.compose(fk_desired_dict[finger_name])
        ee_p_desired = m_desired.transform_points(points_finger_frame).reshape((N, T, 1, 3)).squeeze(-2)
        delta_p_desired = ee_p_desired - ee_p[:, :-1]
        delta_p = ee_p[:, 1:] - ee_p[:, :-1]
        normal_delta_p = (delta_p.unsqueeze(-2) @ surface_normal.unsqueeze(-1)).squeeze((-1, -2))
        normal_delta_p_desired = (delta_p_desired.unsqueeze(-2) @ surface_normal.unsqueeze(-1)).squeeze((-1,-2))
        constraint_list.append(normal_delta_p - normal_delta_p_desired)



    
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
                                      device = params['device'],
                                      chain = chain,
                                      finger_name = 'index',
                                      valve_location = valve_location,
                                      collision_checking = True)
        controller = PositionControlConstrainedSVGDMPC(problem, params)
    else:
        raise ValueError('Invalid controller')

    actual_trajectory = []
    duration = 0

    # debug: plot the thumb traj
    ax = plt.axes(projection='3d')
    ax.set_aspect('equal')
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)
    # ax.set_xlim((0.8, 0.8))
    # ax.set_ylim((0.6, 0.7))
    # ax.set_zlim((1.35, 1.45))
    thumb_traj_history = []
    index_traj_history = []
    state = env.get_state()
    start = state['q'].reshape(9).to(device=params['device'])
    thumb_ee = state2ee_pos(start[:8], thumb_ee_name).squeeze(0)
    thumb_traj_history.append(thumb_ee.detach().cpu().numpy())
    index_ee = state2ee_pos(start[:8], index_ee_name).squeeze(0)
    index_traj_history.append(index_ee.detach().cpu().numpy())

    for k in range(params['num_steps']):
        state = env.get_state()
        start = state['q'].reshape(9).to(device=params['device'])
        # for debugging
        current_theta = start[8]
        thumb_radial_vec = thumb_ee - valve_location # do not consider action at the final timestep
        thumb_radial_vec[1] = 0 # do not consider y axis
        # in our simple task, might need to ignore one dimension
        thumb_surface_normal = - thumb_radial_vec / torch.linalg.norm(thumb_radial_vec) /70 # the normal goes inwards for friction cone computation
        temp_for_plot = torch.stack((thumb_ee, thumb_ee + thumb_surface_normal), dim=0).detach().cpu().numpy()
        ax.plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'red')
        index_radial_vec = index_ee - valve_location # do not consider action at the final timestep
        index_radial_vec[1] = 0 # do not consider y axis
        # in our simple task, might need to ignore one dimension
        index_surface_normal = - index_radial_vec / torch.linalg.norm(index_radial_vec) / 70 # the normal goes inwards for friction cone computation
        temp_for_plot = torch.stack((index_ee, index_ee + index_surface_normal), dim=0).detach().cpu().numpy()
        ax.plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'red')


        fk_start = chain.forward_kinematics(partial_to_full_state(start[:8]), frame_indices=frame_indices)
        print(f"current theta: {current_theta}")

        actual_trajectory.append(state['q'].reshape(9).clone())
        # if k > 0:
        #     torch.cuda.synchronize()
        #     start_time = time.time()
        # if k == 0:
        # for debugging
        if controller.x is not None:
            initial_theta = controller.x[0, 0, 8]
            print(f"initial theta: {initial_theta}")
            # fk_initial = chain.forward_kinematics(partial_to_full_state(controller.x[0, :8]), frame_indices=frame_indices)
        start_time = time.time()
        best_traj, trajectories = controller.step(start)
        print(f"solve time: {time.time() - start_time}")
        # for debugging
        planned_theta = best_traj[0, 8]
        print(f"planned theta: {planned_theta}")

        # add trajectory lines to sim
        add_trajectories(trajectories, best_traj, chain, ax)

        # process the action
        ## end effector force to torque
        x = best_traj[0, :problem.dx+problem.du]
        x = x.reshape(1, problem.dx+problem.du)
        action = x[:, problem.dx:problem.dx+problem.du].to(device=env.device)
        action = action + start.unsqueeze(0)[:, :8] # NOTE: this is required since we define action as delta action
        env.step(action)
        # TODO: need to fix the compute hessian part
        # print(best_traj[:, 8])
        equality_eval, _, _ = problem._con_eq(best_traj.unsqueeze(0), compute_hess=True)
        print(f"max equality constraint violation: {equality_eval.max()}")
        inequality_eval, _, _ = problem._con_ineq(best_traj.unsqueeze(0), compute_hess=True)
        # print(f"max inequality constraint violation: {inequality_eval.max()}")
        # distance2surface = torch.sqrt((best_traj_ee[:, 2] - valve_location[2].unsqueeze(0)) ** 2 + (best_traj_ee[:, 0] - valve_location[0].unsqueeze(0))**2)
        distance2goal = (env.get_state()['q'][:, -1] - params['goal']).detach().cpu().item()
        print(distance2goal)

        gym.clear_lines(viewer)
        # for debugging
        state = env.get_state()
        start = state['q'].reshape(9).to(device=params['device'])
        thumb_ee = state2ee_pos(start[:8], thumb_ee_name).squeeze(0)
        thumb_traj_history.append(thumb_ee.detach().cpu().numpy())
        temp_for_plot = np.stack(thumb_traj_history, axis=0)
        ax.plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'gray')
        index_ee = state2ee_pos(start[:8], index_ee_name).squeeze(0)
        index_traj_history.append(index_ee.detach().cpu().numpy())
        temp_for_plot = np.stack(index_traj_history, axis=0)
        ax.plot3D(temp_for_plot[:, 0], temp_for_plot[:, 1], temp_for_plot[:, 2], 'gray')
    # plt.show()



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


def add_trajectories(trajectories, best_traj, chain, ax=None):
    M = len(trajectories)
    if M > 0:
        # print(partial_to_full_state(best_traj[:, :8]).shape)
        # frame_indices = torch.tensor([index_ee_link, thumb_ee_link])
        best_traj_ee_fk_dict = chain.forward_kinematics(partial_to_full_state(best_traj[:, :8]),
                                                        frame_indices=frame_indices)
        initial_state = env.get_state()['q'][:, :8]
        whole_state = torch.cat((initial_state, best_traj[:-1, :8]), dim=0)
        desired_state = whole_state + best_traj[:, 9:17]
        desired_traj_ee_fk_dict = chain.forward_kinematics(partial_to_full_state(desired_state),
                                                        frame_indices=frame_indices)

        index_best_traj_ee = world_trans.compose(best_traj_ee_fk_dict[index_ee_name])  # .get_matrix()
        thumb_best_traj_ee = world_trans.compose(best_traj_ee_fk_dict[thumb_ee_name])  # .get_matrix()

        index_desired_best_traj_ee = world_trans.compose(desired_traj_ee_fk_dict[index_ee_name])
        thumb_desired_best_traj_ee = world_trans.compose(desired_traj_ee_fk_dict[thumb_ee_name])
        # index_best_traj_ee = best_traj_ee_fk_dict[index_ee_name].compose(world_trans).get_matrix()
        # thumb_best_traj_ee = best_traj_ee_fk_dict[thumb_ee_name].compose(world_trans).get_matrix()
        points_finger_frame = torch.tensor([0.00, 0.03, 0.00], device=best_traj.device).unsqueeze(0)
        index_best_traj_ee = index_best_traj_ee.transform_points(points_finger_frame).squeeze(1)
        thumb_best_traj_ee = thumb_best_traj_ee.transform_points(points_finger_frame).squeeze(1)
        desired_index_best_traj_ee = index_desired_best_traj_ee.transform_points(points_finger_frame).squeeze(1)
        desired_thumb_best_traj_ee = thumb_desired_best_traj_ee.transform_points(points_finger_frame).squeeze(1)
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
                desired_index_p_best = torch.stack((index_best_traj_ee[t, :3], desired_index_best_traj_ee[t + 1, :3]), dim=0).reshape(2,
                                                                                                                      3).cpu().numpy()
                desired_thumb_p_best = torch.stack((thumb_best_traj_ee[t, :3], desired_thumb_best_traj_ee[t + 1, :3]), dim=0).reshape(2,
                                                                                                                      3).cpu().numpy()
                if t == 0:
                    initial_thumb_ee = state2ee_pos(initial_state, thumb_ee_name).squeeze(0)
                    thumb_state_traj = torch.stack((initial_thumb_ee, thumb_best_traj_ee[0]), dim=0).cpu().numpy()
                    thumb_action_traj = torch.stack((initial_thumb_ee, desired_thumb_best_traj_ee[0]), dim=0).cpu().numpy()
                    ax.plot3D(thumb_state_traj[:, 0], thumb_state_traj[:, 1], thumb_state_traj[:, 2], 'blue')
                    ax.plot3D(thumb_action_traj[:, 0], thumb_action_traj[:, 1], thumb_action_traj[:, 2], 'green')
                    initial_index_ee = state2ee_pos(initial_state, index_ee_name).squeeze(0)
                    index_state_traj = torch.stack((initial_index_ee, index_best_traj_ee[0]), dim=0).cpu().numpy()
                    index_action_traj = torch.stack((initial_index_ee, desired_index_best_traj_ee[0]), dim=0).cpu().numpy()
                    ax.plot3D(index_state_traj[:, 0], index_state_traj[:, 1], index_state_traj[:, 2], 'blue')
                    ax.plot3D(index_action_traj[:, 0], index_action_traj[:, 1], index_action_traj[:, 2], 'green')
                gym.add_lines(viewer, e, 1, index_p_best, index_colors)
                gym.add_lines(viewer, e, 1, thumb_p_best, thumb_colors)
                gym.add_lines(viewer, e, 1, desired_index_p_best, np.array([0, 1, 1]).astype(np.float32))
                gym.add_lines(viewer, e, 1, desired_thumb_p_best, np.array([1, 1, 1]).astype(np.float32))
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
