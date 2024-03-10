import numpy as np

np.float = np.float64

from isaac_victor_envs.utils import get_assets_dir
from isaac_victor_envs.tasks.kuka import KukaBaseEnv
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
# from ccai.mpc.diffusion_mpc import Diffusion_MPC
import time
import pytorch_kinematics as pk

from ccai.tsr import TSR

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
asset_dir = get_assets_dir()
asset = asset_dir + '/kuka_allegro/kuka.urdf'
ee_name = 'iiwa7_link_7'
chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name)
chain_cc = pk.build_chain_from_urdf(open(asset).read())


def update_chain(device):
    chain.to(device=device)
    chain_cc.to(device=device)


class KukaProblem(ConstrainedSVGDProblem):

    def __init__(self, start, goal, T, device='cuda:0', base_pose=None):
        super().__init__(start, goal, T, device)
        self.dx = 7
        self.du = 0
        self.dt = 0.1

        # have step-size constraints
        self.dz = 2 * self.dx

        # have TSR constraints per timestep (dz is really quite large now)
        self.dz += 12

        self.dh = self.dz * T

        self.slack_variable = False
        if not self.slack_variable:
            self.dz = 0

        self.dg = 0

        self.T = T
        self.start = start
        self.goal = goal
        # self.K = rbf_kernel
        self.K = structured_rbf_kernel
        self.squared_slack = True
        self.grad_kernel = jacrev(rbf_kernel, argnums=0)
        self.alpha = 1

        self.x_max = torch.tensor([2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05])
        self.x_min = -self.x_max

        if base_pose is None:
            base_pose = torch.eye(4, device=self.device)
        self.base_transform = pk.Transform3d(matrix=base_pose.reshape(-1, 4, 4))
        # transform goal into robot frame
        self.goal = self.base_transform.inverse().transform_points(self.goal.unsqueeze(0)).reshape(-1)
        self._terminal_constraints = EndEffectorConstraint(
            chain, partial(ee_terminal_constraint, goal=self.goal)
        )

        # Do the TSR constraints - in robot frame
        transform_to_tsr_frame = pk.Transform3d(
            rot=torch.eye(3, device=device),
            pos=torch.tensor([0.0, 0.0, 0.0], device=device),
            device=device
        )
        end_effector_offset = pk.Transform3d(
            rot=torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], device=device),
            pos=torch.zeros(3, device=device),
            device=device
        )

        bounds = 10 * torch.tensor([
            [-2.0, 2.0],
            [-2.0, 2.0],
            [-2.0, 2.0],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi]
        ], device=device)

        self._tsr = TSR(transform_to_tsr_frame=transform_to_tsr_frame,
                        end_effector_offset=end_effector_offset,
                        bounds=bounds,
                        kinematic_chain=chain)
        self._tsr_data = {}

    def _preprocess(self, xu, compute_grads=True, compute_hess=False):
        N = xu.shape[0]
        x = xu.reshape(N, self.T, -1)[:, :, :self.dx]
        T = x.shape[1]
        x = x.reshape(N * T, -1)
        g, grad_g, hess_g, h, grad_h, hess_h = self._tsr.eval(x.reshape(-1, self.dx), compute_grads, compute_hess)

        if g is not None:
            g = g.reshape(N, -1)
        if grad_g is not None:
            grad_g = grad_g.reshape(N, self.T, -1, self.dx).permute(0, 2, 3, 1)
            grad_g = torch.diag_embed(grad_g)  # (N, n_constraints, dx + du, T, T)
            grad_g = grad_g.permute(0, 3, 1, 4, 2).reshape(N, -1, self.T * (self.dx))
        if hess_g is not None:
            hess_g = hess_g.reshape(N, self.T, -1, self.dx, self.dx).permute(0, 2, 3, 4, 1)
            hess_g = torch.diag_embed(torch.diag_embed(hess_g))  # (N, n_constraints, dx + du, dx + du, T, T, T)
            hess_g = hess_g.permute(0, 4, 1, 5, 2, 6, 3).reshape(N, -1,
                                                                 self.T * (self.dx),
                                                                 self.T * (self.dx)
                                                                 )
        if h is not None:
            h = h.reshape(N, -1)
        if grad_h is not None:
            grad_h = grad_h.reshape(N, self.T, -1, self.dx).permute(0, 2, 3, 1)
            grad_h = torch.diag_embed(grad_h)  # (N, n_constraints, dx + du, T, T)
            grad_h = grad_h.permute(0, 3, 1, 4, 2).reshape(N, -1, self.T * (self.dx))
        if hess_h is not None:
            hess_h = hess_h.reshape(N, self.T, -1, self.dx, self.dx).permute(0, 2, 3, 4, 1)
            hess_h = torch.diag_embed(torch.diag_embed(hess_h))  # (N, n_constraints, dx + du, dx + du, T, T, T)
            hess_h = hess_h.permute(0, 4, 1, 5, 2, 6, 3).reshape(N, -1,
                                                                 self.T * (self.dx),
                                                                 self.T * (self.dx)
                                                                 )

        #print(h.max())
        self._tsr_data['equality'] = (g, grad_g, hess_g)
        self._tsr_data['inequality'] = (h, grad_h, hess_h)

    def _stepsize_constraint(self, trajectory, compute_grads=True, compute_hess=False):
        step_size_lim = 0.1
        N, T, _ = trajectory.shape
        x = trajectory[:, :, :self.dx]
        current_x = torch.cat((self.start.reshape(1, 1, self.dx).repeat(N, 1, 1), x[:, :-1]), dim=1)

        next_x = x

        delta = next_x - current_x
        h = torch.stack((delta - step_size_lim, -delta - step_size_lim), dim=1)  # N x 2 x T x dx

        grad_h, hess_h = None, None
        if compute_grads:
            eye = torch.eye(self.dx, device=self.device)  # .reshape(1, 1, self.dx, self.dx)#.repeat(N, 1, 1, 1)
            grad_h = torch.zeros(N, 2, T, self.dx, T, self.dx, device=self.device)

            T_range = torch.arange(T, device=self.device)
            T_plus = torch.arange(1, T, device=self.device)
            T_neg = torch.arange(T - 1, device=self.device)

            grad_h[:, 0, T_range, :, T_range] = eye  # .repereshape(N, T * self.dx, self.dx)
            grad_h[:, 1, T_range, :, T_range] = -eye  # .reshape(N, T * self.dx, self.dx)
            grad_h[:, 0, T_plus, :, T_neg] = -eye  # .reshape(N, T * self.dx, self.dx)
            grad_h[:, 1, T_plus, :, T_neg] = eye  # .reshape(N, T * self.dx, self.dx)
        if compute_hess:
            hess_h = torch.zeros(N, 2 * T * self.dx, T * self.dx, T * self.dx, device=self.device)

        return h.reshape(N, -1), grad_h.reshape(N, -1, T * self.dx), hess_h

    def _objective(self, x):
        x = x[:, :, :self.dx]
        N = x.shape[0]
        J, grad_J, hess_J = self._terminal_constraints.eval(x.reshape(-1, self.dx))
        J = J.reshape(N, -1)
        grad_J = grad_J.reshape(N, -1, self.dx)

        J = torch.sum(J, dim=1)

        # hess_J = hess_J.reshape(N, -1, self.dx, self.dx)
        hess_J = torch.zeros(N, self.T * self.dx, self.T * self.dx, device=x.device)
        N = x.shape[0]
        return (self.alpha * J.reshape(N),
                self.alpha * grad_J.reshape(N, -1),
                self.alpha * hess_J.reshape(N, self.T * self.dx, self.T * self.dx))

    def _con_eq(self, x, compute_grads=True, compute_hess=True):
        return None, None, None

    def _con_ineq(self, x, compute_grads=True, compute_hess=True):
        h_tsr, grad_h_tsr, hess_h_tsr = self._tsr_data['inequality']
        h, grad_h, hess_h = self._stepsize_constraint(x)
        h = torch.cat((h, h_tsr), dim=1)
        if not compute_grads:
            return h, None, None

        grad_h = torch.cat((grad_h, grad_h_tsr), dim=1)
        if not compute_hess:
            return h, grad_h, None

        hess_h = torch.cat((hess_h, hess_h_tsr), dim=1)
        return h, grad_h, hess_h

    def eval(self, augmented_trajectory):
        self._preprocess(augmented_trajectory, compute_grads=True, compute_hess=False)
        N = augmented_trajectory.shape[0]
        augmented_trajectory = augmented_trajectory.clone().reshape(N, self.T, -1)
        x = augmented_trajectory[:, :, :self.dx + self.du]

        J, grad_J, hess_J = self._objective(x)
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
        G, dG, hessG = self.combined_constraints(augmented_trajectory, compute_grads=True, compute_hess=False)

        #print(G.max(), G.mean(), J.mean())
        if hessG is not None:
            hessG.detach_()
        return grad_J.detach(), hess_J, K.detach(), grad_K.detach(), G.detach(), dG.detach(), hessG

    def update(self, start, goal=None, T=None):
        self.start = start

        if goal is not None:
            self.goal = self.base_transform.inverse().transform_points(goal.unsqueeze(0)).reshape(-1)
            self._terminal_constraints = EndEffectorConstraint(
                chain, partial(ee_terminal_constraint, goal=self.goal)
            )

        if T is not None:
            self.T = T
            self.dh = self.dz * T
            self.dg = 0

    def get_initial_xu(self, N):
        u = 0.2 * (torch.rand(N, self.T, 7, device=self.device) - 0.5)
        x = torch.cumsum(u, dim=1) + self.start.reshape(1, 1, self.dx)
        return x


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

    def eval(self, q, compute_grads=True, compute_hess=False):
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
        self._J = self.chain.jacobian(joint_config)

        # Use Jacobian to get derivative wrt joint configuration
        Dg = (dpose.unsqueeze(-2) @ self._J.unsqueeze(1)).squeeze(-2)
        """
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
        """
        if compute_hess:
            DDg = torch.zeros(T, n_constraints, 7, 7, device=q.device)
        else:
            DDg = None
        return constraints, Dg, DDg

    def reset(self):
        self._J, self._h, self._dH = None, None, None


def ee_terminal_constraint(p, mat, goal):
    """

    :param p:
    :param mat:
    :return:
    """

    return 10 * torch.sum((p - goal.reshape(-1)) ** 2).reshape(-1)


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

    if 'csvgd' in params['controller'] or 'diffmpc' in params['controller']:
        problem = KukaProblem(start, params['goal'], params['T'], device=params['device'],
                              base_pose=env.robot_base_pose)
        if 'csvgd' in params['controller']:
            controller = Constrained_SVGD_MPC(problem, params)

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
                [params['goal'][0].item() - 0.025, params['goal'][1].item() - 0.025, params['goal'][2].item()],
                [params['goal'][0].item() + 0.025, params['goal'][1].item() + 0.025, params['goal'][2].item()],
                [params['goal'][0].item() - 0.025, params['goal'][1].item() + 0.025, params['goal'][2].item()],
                [params['goal'][0].item() + 0.025, params['goal'][1].item() - 0.025, params['goal'][2].item()],
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

                # transform trajectories into world pose
                trajectories = problem.base_transform.transform_points(trajectories.reshape(-1, 3)).reshape(M, -1, 3)

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

    final_distance_to_goal = torch.linalg.norm(
        chain.forward_kinematics(actual_trajectory[:, :7].reshape(-1, 7)).get_matrix().reshape(-1, 4, 4)[:, :3, 3] -
        params[
            'goal'].unsqueeze(0),
        dim=1
    )
    # print(f'Controller: {params["controller"]} Final distance to goal: {torch.min(final_distance_to_goal)}')
    print(f'{params["controller"]}, Average time per step: {duration / (params["num_steps"] - 1)}')
    if params['visualize']:
        env.gym.write_viewer_image_to_file(env.viewer, f'{env.frame_fpath}/frame_{env.frame_id + 1:06d}.png')

    return torch.min(final_distance_to_goal).cpu().numpy()


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    # get config
    config = yaml.safe_load(
        pathlib.Path(f'{CCAI_PATH}/config/kuka.yaml').read_text())
    from tqdm import tqdm

    env = KukaBaseEnv(1, control_mode='joint_impedance', viewer=config['visualize'], use_cartesian_controller=False)
    sim, gym, viewer = env.get_sim()
    results = {}

    for i in tqdm(range(config['num_trials']), initial=config['start_trial']):

        goal = (torch.rand(3) - 0.5)
        goal[2] += 1.5
        for controller in config['controllers'].keys():
            # env.reset()
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
