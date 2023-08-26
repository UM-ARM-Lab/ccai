import numpy as np
from isaacgym.torch_utils import quat_apply
from isaac_victor_envs.tasks.victor import VictorPuckPillarEnv, orientation_error, quat_change_convention
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
from ccai.mpc.csvgd import Constrained_SVGD_MPC
from ccai.mpc.mppi import MPPI
from ccai.mpc.svgd import SVMPC
from ccai.mpc.ipopt import IpoptMPC
import time
import pytorch_kinematics as pk

from quadrotor_learn_to_sample import TrajectorySampler
from victor_table_surface_jointspace import EndEffectorConstraint
from ccai.env import generate_random_sphere_world

import pytorch_volumetric as pv

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
asset_dir = get_assets_dir()
asset = asset_dir + '/victor/victor_mallet.urdf'

ee_name = 'victor_left_arm_striker_mallet_tip'
chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name)
chain_cc = pk.build_chain_from_urdf(open(asset).read())

robot_scene = None


class VictorReacherProblem(ConstrainedSVGDProblem):

    def __init__(self, start, goal, T, robot_scene, device='cuda:0'):
        super().__init__(start, goal, T, device)
        # one for z for self collision and one for scene collision
        self.dz = 1
        self.dh = self.dz * T
        self.dg = 0
        self.dx = 7
        self.du = 0
        self.dt = 0.1
        self.T = T
        self.start = start
        self.goal = goal
        # self.K = rbf_kernel
        self.robot_scene = robot_scene
        self.K = structured_rbf_kernel

        self._equality_constraints = None

        self.grad_kernel = jacrev(rbf_kernel, argnums=0)
        self.alpha = 10

        # not actually a constraint, instead a cost? Could make constraint
        self._terminal_constraints = EndEffectorConstraint(
            chain, partial(ee_terminal_constraint, goal=self.goal)
        )

        self.right_arm = torch.tensor([1.144, -1.189, 0.590, 0.292, 0.296, 0.265, -0.809], device=self.device)
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

    def _inequality_constraints(self, x, compute_grads=True):
        N = x.shape[0]

        q = torch.cat((x, self.right_arm.expand(N, -1)), dim=1)

        ret_scene = self.robot_scene.scene_collision_check(q, compute_grads, compute_hessian=False)

        h = -ret_scene.get('sdf') + 0.05
        grad_h = ret_scene.get('grad_sdf', None)
        hess_h = ret_scene.get('hess_sdf', None)

        if grad_h is not None:
            grad_h = -grad_h[:, :7].unsqueeze(1)
            if hess_h is not None:
                hess_h = -hess_h[:, :7, :7].unsqueeze(1)
            else:
                hess_h = torch.zeros(x.shape[0], self.dz, self.dx, self.dx, device=self.device)

        return h, grad_h, hess_h

    def _con_ineq(self, x, compute_grads=True):
        z = x[:, :, -1]
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

        print(G.abs().max(), G.abs().mean(), J)
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

    def get_initial_xu(self, N):

        u = torch.randn(N, self.T, 7, device=self.device)
        x = [self.start.reshape(1, self.dx).expand(N, -1)]
        for t in range(self.T):
            x.append(self.dynamics(x[-1], u[:, t]))

        # particles = torch.cumsum(particles, dim=1) + self.start.reshape(1, 1, self.dx)
        x = torch.stack(x[1:], dim=1)
        xu = torch.cat((x, u), dim=2)
        return x


class VictorTableIpoptProblem(VictorReacherProblem, IpoptProblem):

    def __init__(self, start, goal, T, **kwargs):
        super().__init__(start, goal, T, device='cpu', **kwargs)


class VictorTableUnconstrainedProblem(VictorReacherProblem, UnconstrainedPenaltyProblem):

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
        0.1, 0.15, 0.4, 0.4, 0.6, 0.75, 1.0], device=x.device, dtype=torch.float32)
    weight = 1.0 / weight
    diff = x[1:] - x[:-1]
    weighted_diff = diff.reshape(-1, 1, 7) @ torch.diag(weight).unsqueeze(0) @ diff.reshape(-1, 7, 1)
    return 10 * torch.sum(weighted_diff)


def ee_terminal_constraint(p, mat, goal):
    """

    :param p:
    :param mat:
    :return:
    """

    return 5 * torch.sum((p[:3] - goal.reshape(3)) ** 2).reshape(-1)


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
    start = params.get('start', None)
    if start is None:
        start = state['q'].reshape(7).to(device=params['device'])

    chain.to(device=params['device'])
    chain_cc.to(device=params['device'])
    if 'csvgd' in params['controller']:
        flow_model = None
        params['flow_model'] = flow_model
        problem = VictorReacherProblem(start, params['goal'], params['T'], device=params['device'],
                                       robot_scene=params['robot_scene'])
        controller = Constrained_SVGD_MPC(problem, params)
    elif params['controller'] == 'ipopt':
        problem = VictorTableIpoptProblem(start, params['goal'], params['T'], robot_scene=robot_scene)
        controller = IpoptMPC(problem, params)
    elif 'svgd' in params['controller']:
        problem = VictorTableUnconstrainedProblem(start, params['goal'], params['T'], device=params['device'],
                                                  penalty=params['penalty'], robot_scene=robot_scene)
        controller = SVMPC(problem, params)
    elif 'mppi' in params['controller']:
        problem = VictorTableUnconstrainedProblem(start, params['goal'], params['T'], device=params['device'],
                                                  penalty=params['penalty'], robot_scene=robot_scene)
        controller = MPPI(problem, params)
    else:
        raise ValueError('Invalid controller')

    actual_trajectory = []
    planned_trajectories = []
    duration = 0
    constr_params = []

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
                [goal[0].item() - 0.025, goal[1].item() - 0.025, goal[2].item()],
                [goal[0].item() + 0.025, goal[1].item() + 0.025, goal[2].item()],
                [goal[0].item() - 0.025, goal[1].item() + 0.025, goal[2].item()],
                [goal[0].item() + 0.025, goal[1].item() - 0.025, goal[2].item()],
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
                    p = torch.stack((s[:3].reshape(1, 3).expand(M, -1),
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

        full_x = torch.cat((x, problem.right_arm), dim=0).reshape(1, 14)
        problem.robot_scene.visualize_robot(full_x)

        if params['simulate']:
            env.step(x.reshape(1, 7).to(device=env.device))
        if params['visualize']:
            gym.clear_lines(viewer)

    state = env.get_state()
    state = state['q'].reshape(7).to(device=params['device'])

    actual_trajectory.append(state.clone())
    actual_trajectory = torch.stack(actual_trajectory, dim=0).reshape(-1, 7)
    planned_trajectories = torch.stack(planned_trajectories, dim=0)

    problem.T = actual_trajectory.shape[0]

    # print(f'Controller: {params["controller"]} Final distance to goal: {torch.min(final_distance_to_goal)}')
    final_distance_to_goal = torch.linalg.norm(
        chain.forward_kinematics(actual_trajectory[:, :7].reshape(-1, 7)).get_matrix().reshape(-1, 4, 4)[:, :3, 3] -
        params[
            'goal'].unsqueeze(0),
        dim=1
    )
    # print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')
    np.savez(f'{fpath.resolve()}/trajectory.npz', x=actual_trajectory.cpu().numpy(),
             traj=planned_trajectories.cpu().numpy(),
             goal=params['goal'].cpu().numpy(),
             sdf_grid=params['sdf_grid'],
             )
    return torch.min(final_distance_to_goal).cpu().numpy()


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    # get config
    config = yaml.safe_load(
        pathlib.Path(f'{CCAI_PATH}/config/planning_configs/victor_reacher_jointspace.yaml').read_text())
    from tqdm import tqdm

    # instantiate environment
    env = VictorPuckPillarEnv(1, control_mode='joint_impedance',
                              viewer=config['visualize'])
    sim, gym, viewer = env.get_sim()
    from isaacgym import gymapi

    cam_pos = gymapi.Vec3(-0.5, 2, 2)
    cam_target = gymapi.Vec3(0.6, 0.3, 0.75)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
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
    right_arm = torch.tensor([1.144, -1.189, 0.590, 0.292, 0.296, 0.265, -0.809])
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


    results = {}
    device = 'cuda:0'
    for i in tqdm(range(config['num_trials'])):

        if config['random_sphere_world']:
            while True:
                attempts = 0
                sdf_grid, sdf_mesh = generate_random_sphere_world(7,
                                                                  3,
                                                                  0.2,
                                                                  0.1,
                                                                  np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]),
                                                                  64,
                                                                  device=device)

                T = torch.tensor([[[1., 0., 0., 0.75],
                                   [0., 1., 0., 0.25],
                                   [0., 0., 1., 1.],
                                   [0., 0., 0., 1.]]], device=device)
                chain_cc.to(device=device)
                sdf = pv.RobotSDF(chain_cc, path_prefix=asset_dir + '/victor')
                robot_scene = pv.RobotScene(sdf, sdf_mesh, pk.Transform3d(matrix=T),
                                            collision_check_links=collision_check_links)
                # need to randomly generate start and goal
                done = False
                while not done:
                    start = (2 * torch.rand(7) - 1) * torch.tensor([2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05])
                    q = torch.cat((start, right_arm), dim=-1)
                    q = q.to(device=device)
                    goal = torch.rand(3) - 0.5  # + torch.tensor([0.75, 0.25, 1.0])
                    start_sdf = robot_scene.scene_collision_check(q.unsqueeze(0), compute_gradient=False)['sdf']
                    #robot_scene.visualize_robot(q.unsqueeze(0))
                    goal_sdf, _ = sdf_mesh(goal.unsqueeze(0).to(device=device))
                    if start_sdf[0] > 0.05 and goal_sdf[0] > 0.05:
                        done = True
                        goal += torch.tensor([0.75, 0.25, 1.0])

                    attempts += 1
                    if attempts > 100:
                        break

                if done:
                    break

        else:
            sdf_grid, sdf_mesh = None, None
            env.reset()
            goal = torch.tensor([1.25, 0.5, 1.25]) + torch.rand(3) * 0.1
            start = None

        state = env.get_state()

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
            params['robot_scene'] = robot_scene
            params['sdf_grid'] = sdf_grid
            final_distance_to_goal = do_trial(env, params, fpath)

            if controller not in results.keys():
                results[controller] = [final_distance_to_goal]
            else:
                results[controller].append(final_distance_to_goal)
        # print(results)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
