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
from ccai.mpc.mppi import MPPI
from ccai.mpc.svgd import SVMPC
from ccai.mpc.ipopt import IpoptMPC
import time
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

asset = f'{get_assets_dir()}/victor/allegro.urdf'
index_ee_name = 'index_ee'
thumb_ee_name = 'thumb_ee'
# combined chain
chain = pk.build_chain_from_urdf(open(asset).read())
index_ee_link = chain.frame_to_idx[index_ee_name]
thumb_ee_link = chain.frame_to_idx[thumb_ee_name]

valve_location = torch.tensor([0.85, 0.70, 1.405]).to('cuda:0')
# instantiate environment
env = AllegroValveTurningEnv(1, control_mode='joint_impedance', use_cartesian_controller=False,
                             viewer=True, steps_per_action=30)
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


class AllegroValveProblem(ConstrainedSVGDProblem):

    def __init__(self, start, goal, T, chain, valve_location, finger_name, initial_valve_angle=0, device='cuda:0'):
        """
        valve location: the root location of the valve
        initial_valve_angle: it is designed for continuously turning the valve. For each turn, 
        the valve might not be exactly at 0 degree, we need to subtract that out. 
        If we only care about one turn, we can leave it to 0
        """
        super().__init__(start, goal, T, device)
        self.dz = 4
        self.dh = self.dz * T
        self.dg = 4 * T  # + 2
        self.dx = 9
        self.du = 0
        self.dt = 0.1
        self.T = T
        self.start = start
        self.goal = goal
        self.K = rbf_kernel
        # self.K = structured_rbf_kernel

        self.finger_name = finger_name
        self.valve_location = valve_location
        self.initial_valve_angle = initial_valve_angle

        self.chain = chain
        self.grad_kernel = jacrev(rbf_kernel, argnums=0)
        self.alpha = 10

        self._equality_constraints = JointConstraint(
            partial(get_joint_constraint, chain=self.chain, constraint_function=finger_tip_equality_constraints,
                    initial_valve_angle=initial_valve_angle)
        )

        self._inequality_constraints = JointConstraint(
            partial(get_joint_constraint, chain=self.chain, constraint_function=finger_tip_inequality_constraint,
                    initial_valve_angle=initial_valve_angle)
        )

        self._terminal_constraints = JointConstraint(
            partial(joint_terminal_constraint, goal=self.goal)
        )
        # for nvidia hand
        index_x_max = torch.tensor([0.558488888889, 1.727825, 1.727825, 1.727825])
        index_x_min = torch.tensor([-0.558488888889, -0.279244444444, -0.279244444444, -0.279244444444])
        thumb_x_max = torch.tensor([1.57075, 1.15188333333, 1.727825, 1.76273055556])
        thumb_x_min = torch.tensor([0.279244444444, -0.331602777778, -0.279244444444, -0.279244444444])
        # for honda hand
        # index_x_max = torch.tensor([0.47, 1.6099999999, 1.7089999, 1.61799999])
        # index_x_min = torch.tensor([-0.47, -0.195999999999, -0.174000000, -0.227])
        # thumb_x_max = torch.tensor([1.396, 1.1629999999999, 1.644, 1.71899999])
        # thumb_x_min = torch.tensor([0.26, -0.1049999999, -0.1889999999, -0.162])

        valve_x_max = torch.tensor([10.0 * np.pi])
        valve_x_min = torch.tensor([-10.0 * np.pi])
        self.x_max = torch.cat((index_x_max, thumb_x_max, valve_x_max))
        self.x_min = torch.cat((index_x_min, thumb_x_min, valve_x_min))

        self.dynamics_constraint = vmap(self._dynamics_constraint)
        self.grad_dynamics_constraint = vmap(jacrev(self._dynamics_constraint))
        self.hess_dynamics_constraint = vmap(hessian(self._dynamics_constraint))

        self.cost = vmap(partial(cost, start=self.start))
        self.grad_cost = vmap(jacrev(partial(cost, start=self.start)))
        self.hess_cost = vmap(hessian(partial(cost, start=self.start)))

    def dynamics(self, x, u):
        N = x.shape[0]
        x_finger_prime = x[:, :-1] + self.dt * u
        x_full = partial_to_full_state(x_finger_prime)
        frame_indices = torch.tensor([index_ee_link, thumb_ee_link])

        # assume valve turns with the index finger
        m_prime = self.chain.forward_kinematics(x_full, frame_indices=frame_indices)[index_ee_name]
        m_prime = world_trans.compose(m_prime).get_matrix()
        p_prime, mat_prime = m_prime[:, :3, 3], m_prime[:, :3, :3]

        p_prime_vec = p_prime - self.valve_location
        theta = torch.atan2(p_prime_vec[:, 0], p_prime_vec[:, 2])

        # TODO: Update the dynamics
        return torch.cat((x_finger_prime, theta.unsqueeze(-1)), dim=-1)

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
        # term_grad_g_extended = torch.zeros(N, self.T, self.dx, device=self.device)
        # term_hess_g_extended = torch.zeros(N, self.T, self.dx, self.T, self.dx, device=self.device)
        # term_grad_g_extended[:, -1, :] = term_grad_g.reshape(N, -1)
        # term_hess_g_extended[:, -1, :, -1, :] = term_hess_g.reshape(N, self.dx, self.dx)
        # J, grad_J, hess_J = self.cost(x)
        J, grad_J, hess_J = self.cost(x), self.grad_cost(x), self.hess_cost(x)

        if term_grad_g is not None:
            with torch.no_grad():
                term_grad_g_extended = term_grad_g.reshape(N, self.T, self.dx)
                term_hess_g_extended = term_hess_g.reshape(N, self.T, self.dx, self.dx).permute(0, 2, 3, 1)
                # term_grad_g_extended[:, -1] *= 10
                term_hess_g_extended = torch.diag_embed(term_hess_g_extended).permute(0, 3, 1, 4, 2)
                # term_hess_g_extended[:, -1, :, -1] *= 10

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

    def _con_ineq(self, x, compute_grads=True):
        x = x[:, :, :self.dx]
        N = x.shape[0]
        h, grad_h, hess_h = self._inequality_constraints.eval(x.reshape(-1, self.dx), compute_grads)

        h = h.reshape(N, -1)
        N = x.shape[0]
        if not compute_grads:
            return h, None, None
            # Expand gradient to include time dimensions

        grad_h = grad_h.reshape(N, self.T, -1, self.dx).permute(0, 2, 3, 1)
        grad_h = torch.diag_embed(grad_h)  # (N, n_constraints, dx + du, T, T)
        grad_h = grad_h.permute(0, 3, 1, 4, 2).reshape(N, -1, self.T * (self.dx))

        # Now do hessian
        hess_h = hess_h.reshape(N, self.T, -1, self.dx, self.dx).permute(0, 2, 3, 4, 1)
        hess_h = torch.diag_embed(torch.diag_embed(hess_h))  # (N, n_constraints, dx + du, dx + du, T, T, T)
        hess_h = hess_h.permute(0, 4, 1, 5, 2, 6, 3).reshape(N, -1,
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

            self._terminal_constraints = JointConstraint(
                self.chain, partial(joint_terminal_constraint, goal=self.goal)
            )

        if T is not None:
            self.T = T
            self.dh = self.dz * T
            self.dg = 2 * T

    def get_initial_xu(self, N):

        u = torch.randn(N, self.T, 8, device=self.device)
        x = [self.start.reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            x.append(self.dynamics(x[-1], u[:, t]))

        # particles = torch.cumsum(particles, dim=1) + self.start.reshape(1, 1, self.dx)
        x = torch.stack(x[1:], dim=1)
        xu = torch.cat((x, u), dim=2)
        return x


def cost(x, start):
    x = torch.cat((start.reshape(1, 9), x[:, :9]), dim=0)
    weight = torch.tensor([
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 999], device=x.device, dtype=torch.float32)
    weight = 1.0 / weight
    diff = x[1:] - x[:-1]
    weighted_diff = diff.reshape(-1, 1, 9) @ torch.diag(weight).unsqueeze(0) @ diff.reshape(-1, 9, 1)
    return 10 * torch.sum(weighted_diff)


class JointConstraint:

    def __init__(self, joint_constraint_function):
        self._fn = joint_constraint_function
        self.joint_constraint_fn = vmap(joint_constraint_function)

        self._grad_fn = jacrev(joint_constraint_function, argnums=0)

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
        constraints = self.joint_constraint_fn(q)
        if not compute_grads:
            return constraints, None, None

        dq = self.grad_constraint(q)
        ddq = self.hess_constraint(q)

        return constraints, dq, ddq

    def reset(self):
        self._J, self._h, self._dH = None, None, None


def joint_terminal_constraint(q, goal, initial_valve_angle=0):
    """

    :param p:
    :param mat:
    :return:
    """
    return 5*torch.sum(((q[-1] - initial_valve_angle) - goal.reshape(1)) ** 2).reshape(-1)


def get_joint_constraint(q, chain, constraint_function, initial_valve_angle=0):
    """

    :param q: torch.Tensor (N, DoF) joint positions
    :param chain: pytorch kinematics chain

    :return constraints: torch.Tensor(N, 1) contsraints as specified above

    """
    valve_offset_for_index = 0*np.pi / 2.0
    valve_offset_for_thumb = np.pi / 2.0# * 0.99
    theta = q[-1]
    index_q = q[:4]
    thumb_q = q[4:]
    finger_qs = [index_q, thumb_q]
    finger_names = [index_ee_name, thumb_ee_name]
    constraint_list = []
    frame_indices = torch.tensor([index_ee_link, thumb_ee_link])
    fk_dict = chain.forward_kinematics(partial_to_full_state(q[:8]), frame_indices=frame_indices)

    for finger_q, finger_name in zip(finger_qs, finger_names):
        m = world_trans.compose(fk_dict[finger_name]).get_matrix()
        m = m[0]
        p, mat = m[:3, 3], m[:3, :3]
        if finger_name == finger_names[0]:
            # for thumb, it is tracking a differnet theta angles, which has an offset.
            # NOTE: make leads to bug if we want to use the original theta 
            constraint_list.append(constraint_function(p, theta - initial_valve_angle - valve_offset_for_index))
        elif finger_name == finger_names[1]:
            constraint_list.append(constraint_function(p, theta - initial_valve_angle - valve_offset_for_thumb))
    return torch.cat(constraint_list, dim=0)


def finger_tip_equality_constraints(p, theta):
    """
    sub function called in the joint_constraint function. It takes in the end effector of a single 
    finger and output the equality constraint. 
    : params p: finger tip position
    : params theta: the desired valve angle we needs to track. Note that it is different for different fingers.
    """
    # we use theta to compute desired finger x,z positions
    r = 0.0275
    x_pos = torch.sin(theta) * r + valve_location[0]
    z_pos = torch.cos(theta) * r + valve_location[2]
    return torch.cat((
        p[0] - x_pos.reshape(-1),
        p[2] - z_pos.reshape(-1))
    )


def finger_tip_inequality_constraint(p, theta):
    constraint_y_1 = -(valve_location[1] - p[1]) + 0.005
    constraint_y_2 = (valve_location[1] - p[1]) - 0.055
    return torch.cat((constraint_y_1.reshape(-1), constraint_y_2.reshape(-1)), dim=0)


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
    start = state['q'].reshape(9).to(device=params['device'])
    chain = chain.to(device=params['device'])

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

        actual_trajectory.append(state['q'].reshape(9).clone())
        if k > 0:
            torch.cuda.synchronize()
            start_time = time.time()
        if k == 0:
            best_traj, trajectories = controller.step(start)
        if k > 0:
            torch.cuda.synchronize()
            duration += time.time() - start_time

        x = best_traj[k, :9]

        # add trajectory lines to sim
        # TODO: fix the plotting for different fingers
        # add_trajectories(trajectories, best_traj, chain)
        action = x.reshape(1, 9)[:, :8].to(device=env.device)
        # distance2surface = torch.sqrt((best_traj_ee[:, 2] - valve_location[2].unsqueeze(0)) ** 2 + (best_traj_ee[:, 0] - valve_location[0].unsqueeze(0))**2)
        env.step(action)
        distance2goal = (env.get_state()['q'][:, -1] - params['goal']).detach().cpu().item()
        print(distance2goal)

        gym.clear_lines(viewer)

    state = env.get_state()
    state = state['q'].reshape(9).to(device=params['device'])
    actual_trajectory.append(state.clone())
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


def turn(env, params, fpath):
    state = env.get_state()
    if params['visualize']:
        env.frame_fpath = fpath
        env.frame_id = 0
    else:
        env.frame_fpath = None
        env.frame_id = None

    # ee_pos, ee_ori = state['ee_pos'], state['ee_ori']
    # start = torch.cat((ee_pos, ee_ori), dim=-1).reshape(7).to(device=params['device'])
    start = state['q'].reshape(9).to(device=params['device'])
    chain.to(device=params['device'])

    problem = AllegroValveProblem(start,
                                  params['goal'],
                                  params['T'],
                                  device=params['device'],
                                  chain=chain,
                                  finger_name='index',
                                  valve_location=valve_location,
                                  initial_valve_angle=0)
    controller = Constrained_SVGD_MPC(problem, params)

    actual_trajectory = []
    num_turns = 0
    for k in range(params['num_steps']):
        state = env.get_state()
        start = state['q'].reshape(9).to(device=params['device'])

        actual_trajectory.append(state['q'].reshape(9).clone())
        #if k == 0:
        best_traj, trajectories = controller.step(start)
        x = best_traj[0, :9]
        print(best_traj[:, -1])

        # add trajectory lines to sim
        # TODO: fix the plotting for different fingers
        # add_trajectories(trajectories, best_traj, chain)
        action = x.reshape(1, 9)[:, :8].to(device=env.device)
        # distance2surface = torch.sqrt((best_traj_ee[:, 2] - valve_location[2].unsqueeze(0)) ** 2 + (best_traj_ee[:, 0] - valve_location[0].unsqueeze(0))**2)
        env.step(action)
        distance2goal = (env.get_state()['q'][:, -1] - (- np.pi / 2 * num_turns) - params['goal']).detach().cpu().item() ** 2
        print(distance2goal, env.get_state()['q'][0, -1])
        if distance2goal < np.pi * 0.01:
            action = torch.tensor([[0.3, 0, 0, 0, 1.0, 0, 0, 0]]).to(device=env.device)
            print("resetting")
            env.step(action)
            env.step(torch.cat((env.default_dof_pos[:, :4], env.default_dof_pos[:, 12:16]), dim=1))
            num_turns += 1
            problem = AllegroValveProblem(start,
                                          params['goal'],
                                          params['T'],
                                          device=params['device'],
                                          chain=chain,
                                          finger_name='index',
                                          valve_location=valve_location,
                                          initial_valve_angle=- np.pi / 2 * num_turns)
            controller = Constrained_SVGD_MPC(problem, params)
            print("reset done")

        gym.clear_lines(viewer)

    state = env.get_state()
    state = state['q'].reshape(9).to(device=params['device'])
    actual_trajectory.append(state.clone())
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 9)
    problem.T = actual_trajectory.shape[0]
    constraint_val = problem._con_eq(actual_trajectory.unsqueeze(0))[0].squeeze(0)
    return 0


def add_trajectories(trajectories, best_traj, chain):
    M = len(trajectories)
    if M > 0:
        trajectories = chain.forward_kinematics(trajectories[:, :, :4].reshape(-1, 4), world=world_trans).reshape(M, -1,
                                                                                                                  4, 4)
        trajectories = trajectories[:, :, :3, 3]
        best_traj_ee = chain.forward_kinematics(best_traj[:, :4].reshape(-1, 4), world=world_trans).reshape(-1, 4, 4)
        best_traj_ee = best_traj_ee[:, :3, 3]

        traj_line_colors = np.random.random((3, M)).astype(np.float32)
        best_traj_line_colors = np.array([0, 1, 0]).astype(np.float32)

        for e in env.envs:
            s = env.get_state()['index_pos'].reshape(1, 3).to(device=params['device'])
            # breakpoint()
            p = torch.stack((s[:3].reshape(1, 3).repeat(M, 1),
                             trajectories[:, 0, :3]), dim=1).reshape(2 * M, 3).cpu().numpy()
            # p_best = torch.stack((s[:3].reshape(1, 3).repeat(1, 1), best_traj_ee[0, :3].unsqueeze(0)), dim=1).reshape(2, 3).cpu().numpy()
            # p[:, 2] += 0.005
            # gym.add_lines(viewer, e, 1, p_best, best_traj_line_colors)
            # gym.add_lines(viewer, e, M, p, traj_line_colors)
            T = trajectories.shape[1]
            for t in range(T - 1):
                p = torch.stack((trajectories[:, t, :3], trajectories[:, t + 1, :3]), dim=1).reshape(2 * M, 3)
                p = p.cpu().numpy()
                # p[:, 2] += 0.01
                p_best = torch.stack((best_traj_ee[t, :3], best_traj_ee[t + 1, :3]), dim=0).reshape(2, 3).cpu().numpy()
                gym.add_lines(viewer, e, 1, p_best, best_traj_line_colors)
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
        goal = -torch.tensor([np.pi * 0.5])
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
            # final_distance_to_goal = do_trial(env, params, fpath)
            final_distance_to_goal = turn(env, params, fpath)

            if controller not in results.keys():
                results[controller] = [final_distance_to_goal]
            else:
                results[controller].append(final_distance_to_goal)
        print(results)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
