import torch
from functools import partial
from pytorch_kinematics.transforms import Transform3d
from torch.func import vmap, jacrev, jacfwd


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

        # now to compute hessian
        """
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
        DDg = torch.zeros(T, n_constraints, 7, 7, device=q.device)
        return constraints, Dg, DDg

    def reset(self):
        self._J, self._h, self._dH = None, None, None


class TSR:

    def __init__(self, transform_to_tsr_frame: Transform3d,
                 end_effector_offset: Transform3d,
                 bounds,
                 kinematic_chain):

        self.T_0_w = transform_to_tsr_frame
        self.T_w_ee = end_effector_offset

        self.B = bounds  # 6 x 2
        self.chain = kinematic_chain  # for performing forward kinematics

        # separate bounds to no bounds, equality and inequality
        equality_indices = torch.where(self.B[:, 0] == self.B[:, 1])[0]
        upper_bound_indices = torch.where(torch.logical_and(self.B[:, 0] < self.B[:, 1],
                                                            torch.logical_not(torch.isinf(self.B[:, 1]))))[0]

        lower_bound_indices = torch.where(torch.logical_and(self.B[:, 0] < self.B[:, 1],
                                                            torch.logical_not(torch.isinf(self.B[:, 0]))))[0]

        self.equality_indices = equality_indices
        self.upper_bound_indices = upper_bound_indices
        self.lower_bound_indices = lower_bound_indices

        # number of constraints defined by TSR
        self.num_equality = equality_indices.shape[0]
        self.num_inequality = upper_bound_indices.shape[0] + lower_bound_indices.shape[0]

        if self.num_equality + self.num_inequality < 1:
            raise ValueError('TSR must define at least one constraint')

        self.constraint_eval = EndEffectorConstraint(self.chain, ee_constraint_function=self._tsr_eval_ee)

    def get_eq_and_ineq(self, d, grad_d, hess_d, get_grads=True):
        eq, ineq = None, None
        grad_eq, grad_ineq = None, None
        hess_eq, hess_ineq = None, None

        # get equality constraints
        if self.num_equality > 0:
            eq = d[:, self.equality_indices] - self.B[self.equality_indices, 0]
            if get_grads:
                grad_eq = grad_d[:, self.equality_indices, :]
                hess_eq = hess_d[:, self.equality_indices, :, :]

        # get inequality
        if self.num_inequality > 0:
            ineq = []
            if get_grads:
                grad_ineq = []
                hess_ineq = []
            if self.upper_bound_indices.shape[0] > 0:
                print(self.upper_bound_indices)
                print(self.B.shape)
                ineq.append(d[:, self.upper_bound_indices] - self.B[self.upper_bound_indices, 1])
                if get_grads:
                    grad_ineq.append(grad_d[:, self.upper_bound_indices, :])
                    hess_ineq.append(hess_d[:, self.upper_bound_indices, :, :])
            if self.lower_bound_indices.shape[0] > 0:
                ineq.append(-d[:, self.upper_bound_indices] + self.B[self.lower_bound_indices, 0])
                if get_grads:
                    grad_ineq.append(-grad_d[:, self.lower_bound_indices, :])
                    hess_ineq.append(-hess_d[:, self.lower_bound_indices, :, :])

            grad_ineq = torch.cat(grad_ineq, dim=1)
            hess_ineq = torch.cat(hess_ineq, dim=1)
        return eq, grad_eq, hess_eq, ineq, grad_ineq, hess_ineq

    def _tsr_eval_ee(self, p, mat):
        T_0_ee = Transform3d(rot=mat, pos=p, device=p.device)
        T_0_what = self.T_0_w.inverse().compose(T_0_ee, self.T_w_ee.inverse())
        d = get_displacement(T_0_what)
        return d

    def eval(self, q, compute_grads=True):
        d, grad_d, hess_d = self.constraint_eval.eval(q, compute_grads)
        return self.get_eq_and_ineq(d, grad_d, hess_d, compute_grads)


def get_displacement(T):
    mat = T.get_matrix()
    t = mat[:, :3, 3]
    R = mat[:, :3, :3]

    r = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    p = torch.asin(R[:, 2, 0])
    y = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    return torch.cat([t, r.reshape(-1, 1), p.reshape(-1, 1), y.reshape(-1, 1)], dim=1).reshape(-1)


# test
if __name__ == "__main__":
    # let's make equivalent TSR to both the wrench task and the table end-effector task

    # take table on end-effector first
    from isaac_victor_envs.utils import get_assets_dir
    import pytorch_kinematics as pk

    asset = get_assets_dir() + '/victor/victor_mallet.urdf'
    ee_name = 'victor_left_arm_striker_mallet_tip'
    chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name)

    transform_to_tsr_frame = Transform3d(
        rot=torch.eye(3),
        pos=torch.tensor([0.0, 0.0, 0.8])
    )
    end_effector_offset = Transform3d(
        rot=torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ]),
        pos=torch.zeros(3)
    )

    bounds = torch.tensor([
        [-torch.inf, torch.inf],
        [-torch.inf, torch.inf],
        [0.0, 0.0],
        [0, 0],
        [0, 0],
        [-torch.inf, torch.inf]
    ])

    ee_se2_constraint = TSR(transform_to_tsr_frame=transform_to_tsr_frame,
                            end_effector_offset=end_effector_offset,
                            bounds=bounds,
                            kinematic_chain=chain)

    # get random config
    # q = torch.randn(10, 7)
    q = torch.tensor([[2.9501, -0.2883, -2.5463, -1.5888, -1.6216, -1.1591, -0.7822]])
    eq, grad_eq, hess_eq, ineq, grad_ineq, hess_ineq = ee_se2_constraint.eval(q, compute_grads=True)

    print(eq)
    print(grad_eq)
    # print(hess_eq)

    # Now let's the wrench constraint

    asset = f'{get_assets_dir()}/victor/victor_grippers.urdf'
    ee_name = 'l_palm'
    chain = pk.build_serial_chain_from_urdf(open(asset).read(), ee_name)
    transform_to_tsr_frame = Transform3d(
        rot=torch.eye(3),
        pos=torch.tensor([0.6, 0.15, 0.975])
    )
    end_effector_offset1 = Transform3d(
        rot=torch.tensor([
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0]
        ]),
    )
    end_effector_offset2 = Transform3d(
        pos=torch.tensor([0.0, 0.2, 0.0])
    )

    end_effector_offset = end_effector_offset2.compose(end_effector_offset1)
    bounds = torch.tensor([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [-torch.pi, torch.pi]
    ])

    q = torch.tensor([[2.9501, -0.2883, -2.5463, -1.5888, -1.6216, -1.1591, 0.0]])
    q = torch.tensor([[0.0534, -0.0635, 0.1700, -1.5892, -1.5442, -1.3911, 1.5009]])
    wrench_constraint = TSR(transform_to_tsr_frame=transform_to_tsr_frame,
                        end_effector_offset=end_effector_offset,
                        bounds=bounds,
                        kinematic_chain=chain)
    eq, grad_eq, hess_eq, ineq, grad_ineq, hess_ineq = wrench_constraint.eval(q, compute_grads=True)
    print(eq)
    print(grad_eq)

    print(ineq)
    # print(hess_eq)
