import numpy as np
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion, quaternion_multiply
import torch
from functools import partial
from functorch import vmap, jacrev, hessian, jacfwd
import pytorch_kinematics as pk
from ccai.kernels import rbf_kernel, structured_rbf_kernel
from ccai.problem import ConstrainedSVGDProblem, IpoptProblem, UnconstrainedPenaltyProblem, NLOptProblem
from isaac_victor_envs.utils import get_assets_dir


class VictorWrenchProblem(ConstrainedSVGDProblem):

    def __init__(self, start, goal, T, wrench_centre, wrench_length, device='cuda:0', chain=None):
        """
            Victor Wrench Problem
                Used to evaluate costs & constraints, and their respective Jacobians and Hessians

        :param start: torch.Tensor of shape (7,) containing start robot joint configuration
        :param goal: torch.Tensor of shape (1,) containing goal wrench angle
        :param T: Planning horizon (int)
        :param wrench_centre: list of length 3 containing [x, y, z] position of wrench axis
        :param wrench_length: length of wrench
        :param device: torch.device, typically cuda:0
        :param chain: kinematic chain as a pytorch_kinematics.chain
        """
        super().__init__(start, goal, T, device)
        self.squared_slack = False
        # Problem dimensions setup
        self.dz = 14
        self.dh = self.dz * T
        self.dg = 4 * T + 1
        self.dx = 9
        self.du = 0
        self.T = T
        self.start = start
        self.goal = goal
        self.compute_hessian = False
        self.device = device
        self.chain = chain
        self.alpha = 1
        if chain is None:
            asset = f'{get_assets_dir()}/victor/victor_grippers.urdf'
            ee_name = 'l_palm'
            self.chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name)

        self.chain.to(device=device)
        self.prior_sigma = torch.tensor([0.05, 0.075, 0.075, 0.1, 0.1, 0.15, 0.2],
                                        device=self.device,
                                        dtype=torch.float32)

        self.prior_sigma = 0.1 * torch.ones_like(self.prior_sigma)

        self.cost = vmap(partial(_cost_steps, start=self.start, weight=1 / self.prior_sigma ** 2))
        self.grad_cost = vmap(jacrev(partial(_cost_steps, start=self.start, weight=1 / self.prior_sigma ** 2)))
        self.hess_cost = vmap(hessian(partial(_cost_steps, start=self.start, weight=1 / self.prior_sigma ** 2)))

        self.kernel = structured_rbf_kernel
        self.grad_kernel = jacrev(self.kernel, argnums=0)

        self.wrench_centre = wrench_centre
        self.wrench_length = wrench_length

        self.ee_constraint = EndEffectorConstraint(chain=self.chain, ee_constraint_function=ee_constraint,
                                                   wrench_length=wrench_length, wrench_centre=wrench_centre,
                                                   start=start)

        # Bounds on joint values
        self.x_max = torch.tensor([2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05, 2, torch.pi])
        self.x_min = -self.x_max

    def _objective(self, x):
        return self.alpha * self.cost(x), self.alpha * self.grad_cost(x), None  # , self.hess_cost(x)

    def _con_ineq(self, x, compute_grads=True, compute_hess=True):
        N = x.shape[0]
        q_theta = x.reshape(N, self.T, -1)[:, :, :self.dx]
        z = x.reshape(N, self.T, -1)[:, :, self.dx:]

        # Compute inequality and gradients
        h, Dh, DDh = torque_limits_modified(q_theta.reshape(-1, self.dx),
                                            self.ee_constraint._J,
                                            self.ee_constraint._H,
                                            self.ee_constraint._dH,
                                            self.wrench_length,
                                            self.chain,
                                            compute_grads=compute_grads,
                                            compute_hess=compute_hess,
                                            )

        if compute_grads is False:
            return h.reshape(N, -1), None, None

        Dh = Dh.reshape(N, self.T, self.dz, self.dx)
        Dh = torch.diag_embed(Dh.permute(0, 2, 3, 1)).permute(0, 3, 1, 4, 2).reshape(N, self.T * self.dz, -1)

        if not compute_hess:
            return h.reshape(N, -1), Dh, None

        DDh = DDh.reshape(N, self.T, self.dz, self.dx, self.dx)
        DDh = torch.diag_embed(torch.diag_embed(DDh.permute(0, 2, 3, 4, 1)))
        DDh = DDh.permute(0, 4, 1, 5, 2, 6, 3).reshape(N, self.T * self.dz, self.T * self.dx, -1)

        return h.reshape(N, -1), Dh, DDh

    def _con_eq(self, q_theta, compute_grads=True, compute_hess=True):
        """
            Computes equality constraints and
            returns their first and second derivatives

        :param q_theta: Trajectory of shape (N, T, 9)
        :return g: torch.Tensor of shape (N, dg) containing constraint values
        :return Dg: torch.Tensor of shape (N, dg, T*(dx+du+dz)) containing constraint gradient
        :return DDg: torch.Tensor of shape (N, dg, T*(dx+du+dz)) containing constraint Hessian
        """
        N, T, _ = q_theta.shape
        q_theta = q_theta[:, :, :self.dx]

        # Compute end effector constraints
        g_ee, Dg_ee, DDg_ee = self.ee_constraint.eval(q_theta.reshape(N * T, -1),
                                                      compute_grads=compute_grads,
                                                      compute_hess=compute_hess)
        n_ee = g_ee.shape[1]
        g_ee = g_ee.reshape(N, -1)

        # now we need the goal constraint
        final_wrench_angle = q_theta[:, -1, -1]
        goal_constr = final_wrench_angle.reshape(-1, 1) - self.goal.reshape(-1, 1)

        g = torch.cat((g_ee, goal_constr), dim=1)

        if not compute_grads:
            return g, None, None

        # Currently computed with N T as batch dimensions, convert so T is decision variable
        Dg_ee = Dg_ee.reshape(N, T, n_ee, self.dx)
        # N x n_ee, dx, T
        Dg_ee = torch.diag_embed(Dg_ee.permute(0, 2, 3, 1)).permute(0, 3, 1, 4, 2).reshape(N, self.T * n_ee, -1)
        # then to N, T, n_ee, T, dx
        Dgoal_constr = torch.zeros(N, 1, self.T, self.dx, device=q_theta.device)
        Dgoal_constr[:, :, -1, 8] = 1
        Dgoal_constr = Dgoal_constr.reshape(N, 1, -1)
        Dg = torch.cat((Dg_ee, Dgoal_constr), dim=1)

        if not compute_hess:
            return g, Dg, None

        DDg_ee = DDg_ee.reshape(N, self.T, n_ee, self.dx, self.dx)
        DDg_ee = DDg_ee.permute(0, 2, 3, 4, 1)  # permute to be (N, n_ee, dx, dx, T)

        # diagonalize and make (N, T*n_ee, T*dx, T*dx)
        DDg_ee_extended = torch.diag_embed(torch.diag_embed(DDg_ee)).permute(0, 4, 1, 5, 2, 6, 3)
        DDg_ee_extended = DDg_ee_extended.reshape(N, self.T * n_ee, self.T * self.dx, -1)

        DDgoal_constr = torch.zeros(N, 1, self.T * self.dx, self.T * self.dx,
                                    device=q_theta.device)
        # Combine
        DDg = torch.cat((DDg_ee_extended, DDgoal_constr), dim=1)

        return g, Dg, DDg

    def eval(self, augmented_trajectory):
        """

        :param augmented_trajectory: set of augmented trajectories of shape (N, T * (dx+du) + num_slack)
        :return:

        """
        N = augmented_trajectory.shape[0]

        q_theta = augmented_trajectory.reshape(N, self.T, -1)[:, :, :self.dx]

        # compute gradient of cost first
        grad_cost = self.grad_cost(q_theta)
        grad_cost = torch.cat((grad_cost, torch.zeros(N, self.T, self.dz, device=q_theta.device)), dim=2).reshape(N, -1)

        # compute kernel and grad kernel
        Xk = q_theta#.reshape(N, -1)
        K = self.kernel(Xk, Xk)
        grad_K = -self.grad_kernel(Xk, Xk).reshape(N, N, N, self.T, -1)
        grad_K = torch.einsum('nmmij->nmij', grad_K)
        grad_K = torch.cat((grad_K.reshape(N, N, self.T, self.dx),
                            torch.zeros(N, N, self.T, self.dz, device=q_theta.device)), dim=-1)
        grad_K = grad_K.reshape(N, N, -1)

        # Now we need to compute constraints and their first and second partial derivatives
        g, Dg, DDg = self.combined_constraints(augmented_trajectory,
                                               compute_grads=True,
                                               compute_hess=self.compute_hessian)
        if DDg is not None:
            DDg = DDg.detach_()
        self.ee_constraint.reset()
        #print(self.cost(q_theta).mean(), g.abs().max(), g.abs().mean())

        return grad_cost.detach(), None, K.detach(), grad_K.detach(), g.detach(), Dg.detach(), DDg

    def get_initial_xu(self, N):
        particles = torch.randn(N, self.T, self.dx, device=self.device)
        particles[:, :, :7] *= self.prior_sigma
        particles[:, :, 7:] *= 0.1
        particles = torch.cumsum(particles, dim=1) + self.start.reshape(1, 1, 9)
        return particles

    def update(self, start, T=None, goal=None):
        self.start = start
        self.ee_constraint = EndEffectorConstraint(chain=self.chain, ee_constraint_function=ee_constraint,
                                                   wrench_length=self.wrench_length, wrench_centre=self.wrench_centre,
                                                   start=start)
        if goal is not None:
            self.goal = goal
        if T is not None:
            self.T = T
            self.dh = self.dz * self.T
            self.dg = 4 * T + 1
        self.cost = vmap(partial(_cost_steps, start=self.start, weight=1 / self.prior_sigma ** 2))
        self.grad_cost = vmap(jacrev(partial(_cost_steps, start=self.start, weight=1 / self.prior_sigma ** 2)))


def _cost_steps(x, start, weight):
    x = torch.cat((start[:7].reshape(1, 7), x[:, :7]), dim=0)
    diff = x[1:] - x[:-1]
    weighted_diff = diff.reshape(-1, 1, 7) @ torch.diag(weight).unsqueeze(0) @ diff.reshape(-1, 7, 1)
    #angle = x[:, -1]
    #angle_diff = angle[1:] - angle[:-1]
    return torch.sum(weighted_diff)# + 100 * torch.sum(angle_diff ** 2)


class EndEffectorConstraint:

    def __init__(self, chain, ee_constraint_function, wrench_centre, wrench_length, start):
        self.chain = chain
        self.ee_idx = self.chain.frame_to_idx['l_palm']

        self._fn = partial(ee_constraint_function, wrench_centre=wrench_centre, wrench_length=wrench_length)
        start_pos = self.chain.forward_kinematics(start.reshape(1, 9)[:, :7]).get_matrix()[:, :3, 3]
        start_pos = torch.cat((start_pos, start.reshape(1, 9)[:, 7:]), dim=1).reshape(5)

        self.ee_constraint_fn = vmap(partial(ee_constraint_function,
                                             wrench_centre=wrench_centre,
                                             wrench_length=wrench_length,
                                             start=start_pos))

        self._grad_fn = jacrev(partial(ee_constraint_function,
                                       wrench_centre=wrench_centre,
                                       wrench_length=wrench_length,
                                       start=start_pos), argnums=(0, 1, 2))

        self.grad_constraint = vmap(self._grad_g)
        self.hess_constraint = vmap(jacfwd(self._grad_g, argnums=(0, 1, 2)))

        self._J, self._H, self._dH = None, None, None

    def _grad_g(self, p, mat, theta):
        dp, dmat, dtheta = self._grad_fn(p, mat, theta)

        dmat = dmat @ mat.reshape(1, 3, 3).permute(0, 2, 1)
        # project dmat to be skew-symmetric
        dmat = 0.5 * (dmat - dmat.permute(0, 2, 1))

        omega1 = torch.stack((-dmat[:, 1, 2], dmat[:, 0, 2], -dmat[:, 0, 1]), dim=-1)
        omega2 = torch.stack((dmat[:, 2, 1], -dmat[:, 2, 0], dmat[:, 1, 0]), dim=-1)
        omega = 0.5 * (omega1 + omega2)  # this doesn't seem correct? Surely I should be halfing it
        return dp, omega, dtheta

    def eval(self, q, compute_grads=True, compute_hess=True):
        """

        :param q: torch.Tensor of shape (N, 9) containing set of robot joint config + wrench joint configs
        :return g: constraint values
        :return Dg: constraint gradient
        :return DDg: constraint hessian
        """

        T = q.shape[0]

        # robot joint configuration
        joint_config = q[:, :7]

        # wrench joint configuration
        theta = q[:, 7:].reshape(T, -1)

        # Get end effector pose
        m = self.chain.forward_kinematics(joint_config).get_matrix()
        p, mat = m[:, :3, 3], m[:, :3, :3]

        # Compute constraints
        constraints = self.ee_constraint_fn(p, mat, theta)
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
        dp, omega, dtheta = self.grad_constraint(p, mat, theta)

        # derivative of constraint wrt pose as a (N, num_constraints, 6) tensor
        dpose = torch.cat((dp, omega), dim=-1)
        link_indices = torch.ones(T, device=q.device, dtype=torch.long) * self.ee_idx

        if compute_hess:
            ddp, domega, ddtheta = self.hess_constraint(p, mat, theta)

            ddp, dp_dmat, dp_dtheta = ddp
            domega_dp, domega, omega_dtheta = domega
            dtheta_dp, dtheta_omega, ddtheta = ddtheta

            dp_omega = domega_dp
            dtheta_omega = omega_dtheta

            tmp = domega @ mat.reshape(-1, 1, 1, 3, 3).permute(0, 1, 2, 4, 3)
            domega1 = torch.stack((-tmp[:, :, :, 1, 2], tmp[:, :, :, 0, 2], -tmp[:, :, :, 0, 1]), dim=-1)
            domega2 = torch.stack((tmp[:, :, :, 2, 1], -tmp[:, :, :, 2, 0], tmp[:, :, :, 1, 0]), dim=-1)
            domega = 0.5 * (domega1 + domega2)

            # cache computation for later
            self._J, self._H, self._dH = self.chain.jacobian_and_hessian_and_dhessian(joint_config,
                                                                                      link_indices=link_indices)
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
            dtheta_dpose = torch.cat((dtheta_dp, dtheta_omega.permute(0, 1, 3, 2)), dim=-1)

            dtheta_dq = (dtheta_dpose.unsqueeze(2) @ self._J.unsqueeze(1).unsqueeze(1)).squeeze(2)

            DDg = torch.cat((
                torch.cat((DDg, dtheta_dq.permute(0, 1, 3, 2)), dim=-1),
                torch.cat((dtheta_dq, ddtheta), dim=-1)),
                dim=-2
            )
        else:
            self._J, self._H = self.chain.jacobian_and_hessian(joint_config, link_indices=link_indices)
            DDg = None

        # Use Jacobian to get derivative wrt joint configuration
        djoint_config = (dpose.unsqueeze(-2) @ self._J.unsqueeze(1)).squeeze(-2)
        Dg = torch.cat((djoint_config, dtheta), dim=-1)

        return constraints, Dg, DDg

    def reset(self):
        self._J, self._h, self._dH = None, None, None


def ee_constraint(p, mat, theta, wrench_centre, wrench_length, start):
    """
    End effector constraint for the wrench
     - the end effector should be at the same height as the wrench
     - the end effector xy position should be at the point on the arc defined by wrench angle and the
       radius length wrench_length + wrench_offset
    - The end-effector should be orientated so that the grippers are either side of the wrench, facing down

    :param p: torch.Tensor (N, 3) end effector position
    :param mat: torch.Tensor (N, 3, 3) end effector rotation matrix
    :param theta: torch.tensor(N, 2) wrench angle and offset
    :param wrench_centre: list of length [3] with wrench centre
    :param wrench_length: wrench length (int)

    :return constraints: torch.Tensor(N, 4) contsraints as specified above

    """
    height = wrench_centre[2]
    wrench_angle = theta[-1]
    wrench_offset = theta[-2]
    l = wrench_length + wrench_offset / 100

    slack = -torch.atan2(start[0] - wrench_centre[0], start[1] - wrench_centre[1]) - start[4]
    wrench_angle = wrench_angle + slack
    # constraint on position
    cx = p[0] + l * torch.sin(wrench_angle) - wrench_centre[0]
    cy = p[1] - l * torch.cos(wrench_angle) - wrench_centre[1]
    cz = p[2] - height

    desired_mat = torch.tensor([
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0]
    ], device=p.device)

    desired_euler = torch.cat((torch.zeros(2, device=p.device), -wrench_angle.reshape(1)), dim=0)
    desired_mat2 = euler_angles_to_matrix(desired_euler, 'XYZ')

    desired_mat = desired_mat @ desired_mat2

    mat_diff = desired_mat @ mat
    cosangle = (torch.trace(mat_diff) - 1) / 2
    #cosangle = torch.clamp(cosangle, min=-1 + 1e-4, max=1 - 1e-4)
    cosangle = torch.where(cosangle.abs() > 1.0 - 1e-6, cosangle-1e-4, cosangle)

    constraint_ori = torch.arccos(cosangle)

    constraints = torch.cat((
        cx.reshape(-1),
        cy.reshape(-1),
        cz.reshape(-1),
        constraint_ori.reshape(-1)),
        dim=0
    )
    return constraints


def torque_limits_modified(x, J, H, dH, wrench_length, chain, compute_grads=True, compute_hess=True):
    """
        Computes torque limit inequality constraint as
        torque_min <= J.T F <= torque_max

        Where F is a desired end effector wrench, and J is the maniplator Jacobian

        Returns constraint value & first and second derivatives

    :param x: torch.Tensor of shape (N, 9) joint configuration + wrench configuration
    :param J: Kinematic Jacobian
    :param H: Kinematic Hessian
    :param dH: partial derivative of kinematic hessian wrt joint config (3rd derivative)
    :param wrench_length: wrench length
    :param chain: kinematic chain

    :return constr: constraint values
    :return grad_constr: constraint gradients
    :return hess_constr: constraint hessian
    """
    N = x.shape[0]
    q = x[:, :7]  # joint config
    theta = x[:, -1]  # wrench config

    # hardcoded desired torque
    desired_torque = 1
    # desired force comes from wrench length and desired torque to turn wrench
    desired_force = desired_torque / wrench_length
    max_torque = 20

    # Compute end-effector wrench
    F = desired_force * torch.stack((
        torch.sin(theta),
        torch.cos(theta),
        torch.zeros_like(theta),
        torch.zeros_like(theta),
        torch.zeros_like(theta),
        torch.zeros_like(theta),
    ), dim=1)

    # if pre-computed values were not given then we compute them
    if J is None:
        ee_idx = chain.frame_to_idx['l_palm']
        link_indices = torch.ones(N, device=q.device, dtype=torch.long) * ee_idx

        if not compute_grads:
            J = chain.jacobian(q, link_indices=link_indices)
        elif not compute_hess:
            J, h = chain.jacobian_and_dhessian(q, link_indices=link_indices)
        else:
            J, H, dH = chain.jacobian_and_hessian_and_dhessian(q, link_indices=link_indices)

        # Compute joint torques via J.T @ F
    joint_torques = J.permute(0, 2, 1) @ F.reshape(-1, 6, 1)

    # Constraint values
    constr_upper = joint_torques.reshape(N, -1) - max_torque
    constr_lower = -max_torque - joint_torques.reshape(N, -1)
    constr = torch.cat((constr_lower, constr_upper), dim=1)

    if not compute_grads:
        return constr, None, None

    # Derivative of end effector wrench wrt theta
    dF = desired_force * torch.stack((
        torch.cos(theta),
        -torch.sin(theta),
        torch.zeros_like(theta),
        torch.zeros_like(theta),
        torch.zeros_like(theta),
        torch.zeros_like(theta),
    ), dim=1)

    # jacobian of constraint wrt joint_config
    grad_constr_upper = (H.permute(0, 2, 3, 1) @ F.reshape(-1, 1, 6, 1)).reshape(-1, 7, 7)
    grad_constr_lower = -grad_constr_upper

    # Jacobian of constraint wrt theta
    grad_constr_upper_theta = (J.permute(0, 2, 1) @ dF.reshape(-1, 6, 1)).reshape(-1, 7, 1)
    grad_constr_lower_theta = -grad_constr_upper_theta
    grad_constr_theta = torch.cat((grad_constr_lower_theta, grad_constr_upper_theta), dim=1)
    grad_constr = torch.cat((grad_constr_lower, grad_constr_upper), dim=1)
    # add zero for wrench angle
    grad_constr = torch.cat((grad_constr, torch.zeros_like(grad_constr_theta), grad_constr_theta), dim=2)
    if not compute_hess:
        return constr, grad_constr, None

    # 2nd derivative of end-effector wrench wrt theta
    ddF = -F

    # hessian of the constraint wrt q
    hess_constr_upper = (dH.permute(0, 2, 3, 4, 1) @ F.reshape(-1, 1, 1, 6, 1)).reshape(-1, 7, 7, 7)
    hess_constr_lower = -hess_constr_upper

    # hessian of constr wrt theta
    hess_constr_upper_theta = (J.permute(0, 2, 1) @ ddF.reshape(-1, 6, 1)).reshape(-1, 7, 1)
    hess_constr_lower_theta = -hess_constr_upper_theta
    hess_constr_theta = torch.cat((hess_constr_lower_theta, hess_constr_upper_theta), dim=1).reshape(-1, 14, 1, 1)

    # partial derivates of constraint wrt theta & q
    partial_constr_q_theta_upper = (H.permute(0, 2, 3, 1) @ dF.reshape(-1, 1, 6, 1)).reshape(-1, 7, 7, 1)
    partial_constr_q_theta_lower = - partial_constr_q_theta_upper

    # combine them all together
    hess_constr = torch.cat((hess_constr_lower, hess_constr_upper), dim=1)
    partial_q_theta = torch.cat((partial_constr_q_theta_lower, partial_constr_q_theta_upper), dim=1)

    # add zeros for wrench angle
    hess_constr = torch.cat((
        torch.cat((hess_constr, torch.zeros_like(partial_q_theta), partial_q_theta), dim=3),
        torch.cat((torch.zeros_like(partial_q_theta.permute(0, 1, 3, 2)), torch.zeros_like(hess_constr_theta),
                   torch.zeros_like(hess_constr_theta)), dim=3),
        torch.cat((partial_q_theta.permute(0, 1, 3, 2), torch.zeros_like(hess_constr_theta), hess_constr_theta), dim=3)
    ), dim=2)

    return constr, grad_constr, hess_constr


class VictorWrenchIpoptProblem(VictorWrenchProblem, IpoptProblem):
    def __init__(self, start, goal, T, wrench_centre, wrench_length, chain):
        super().__init__(start, goal, T, wrench_centre, wrench_length, device='cpu', chain=chain)


class VictorWrenchSQPProblem(VictorWrenchProblem, NLOptProblem):
    def __init__(self, start, goal, T, *args, **kwargs):
        super().__init__(start, goal, T, device='cpu', *args, **kwargs)


class VictorUnconstrainedPenaltyProblem(VictorWrenchProblem, UnconstrainedPenaltyProblem):
    def __init__(self, start, goal, T, wrench_centre, wrench_length, chain, penalty):
        super().__init__(start, goal, T, wrench_centre, wrench_length, device='cuda:0', chain=chain)
        self.penalty = penalty
        self.du = 7
        self.dt = 0.1
        self.x_min = torch.cat((self.x_min, -torch.ones(7)))
        self.x_max = torch.cat((self.x_max, torch.ones(7)))

    def dynamics(self, x, u):
        q = x[:, :7]
        next_q = q + self.dt * u

        # compute next wrench angle and next wrench length offset
        # NOTE: this is difficult because we don't actually know the full dynamics of the wrench
        # When the constraint is satisfied, then the dynamics follow a certain path,
        # but when the constraint is not satisfied,
        # they may not.
        # For now we just assume constraint satisfied and see how it breaks
        next_mat = self.chain.forward_kinematics(next_q).get_matrix()
        p = next_mat[:, :3, 3]
        # wrench length is new distance between p and wrench centre
        c = torch.tensor(self.wrench_centre, dtype=torch.float32, device=self.device).reshape(1, 3)
        next_offset = 100 * (torch.norm(p - c, dim=1) - self.wrench_length)

        # next angle is angle between px, py and [0, 1]
        # norm_p = (p[:, :2] / torch.norm(p[:, :2], dim=1, keepdim=True)).reshape(-1, 2)
        # e = torch.tensor([0, 1], dtype=torch.float32, device=self.device).reshape(1, 2)
        # cos_angle = torch.sum(norm_p * e, dim=1)

        # next_angle = torch.arccos(cos_angle)
        next_angle = -torch.atan2(p[:, 0], p[:, 1])
        next_x = torch.cat((next_q, next_offset.reshape(-1, 1), next_angle.reshape(-1, 1)), dim=1)
        return next_x
