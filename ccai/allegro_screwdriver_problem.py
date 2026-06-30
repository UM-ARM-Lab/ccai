from ccai.allegro_contact import AllegroManipulationProblem, PositionControlConstrainedSVGDMPC, add_trajectories, \
    add_trajectories_hardware
import torch


PROTO5_FULL_JOINT_INDEX = {
    'wrist': [0, 1],
    'index': [2, 3, 4, 5],
    'middle': [6, 7, 8, 9],
    'ring': [10, 11, 12, 13],
    'thumb': [14, 15, 16, 17],
}
PROTO5_FINGER_JOINT_INDEX = {
    key: value for key, value in PROTO5_FULL_JOINT_INDEX.items() if key != 'wrist'
}
PROTO5_EE_NAMES = {
    'index': 'RHand_I6AF_LINK',
    'middle': 'RHand_M6AF_LINK',
    'ring': 'RHand_R6AF_LINK',
    'thumb': 'RHand_T6AF_LINK',
}
PROTO5_COLLISION_LINK_NAMES = {
    'index': ['RHand_I6AF_LINK', 'RHand_I3Y_LINK', 'RHand_I2Y_LINK', 'RHand_I1Y_LINK', 'RHand_I1Z_LINK'],
    'middle': ['RHand_M6AF_LINK', 'RHand_M3Y_LINK', 'RHand_M2Y_LINK', 'RHand_M1Y_LINK', 'RHand_M1Z_LINK'],
    'ring': ['RHand_R6AF_LINK', 'RHand_R3Y_LINK', 'RHand_R2Y_LINK', 'RHand_R1Y_LINK', 'RHand_R1Z_LINK'],
    'thumb': ['RHand_T6AF_LINK', 'RHand_T3Y_LINK', 'RHand_T2Y_LINK', 'RHand_T1Y_LINK', 'RHand_T1Z_LINK'],
}
PROTO5_ACTIVE_JOINT_MIN = {
    'wrist': torch.tensor([-0.78539816, -0.95993109], dtype=torch.float32),
    'index': torch.tensor([-0.34906585, 0.0, 0.0, -0.17453293], dtype=torch.float32),
    'middle': torch.tensor([-0.34906585, 0.0, 0.0, -0.17453293], dtype=torch.float32),
    'ring': torch.tensor([-0.34906585, 0.0, 0.0, -0.17453293], dtype=torch.float32),
    'thumb': torch.tensor([-1.57079633, 0.0, 0.0, -0.17453293], dtype=torch.float32),
}
PROTO5_ACTIVE_JOINT_MAX = {
    'wrist': torch.tensor([0.43633231, 0.78539816], dtype=torch.float32),
    'index': torch.tensor([0.34906585, 1.57079633, 1.57079633, 1.57079633], dtype=torch.float32),
    'middle': torch.tensor([0.34906585, 1.57079633, 1.57079633, 1.57079633], dtype=torch.float32),
    'ring': torch.tensor([0.34906585, 1.57079633, 1.57079633, 1.57079633], dtype=torch.float32),
    'thumb': torch.tensor([0.17453293, 1.22173048, 1.57079633, 1.57079633], dtype=torch.float32),
}


class AllegroScrewdriver(AllegroManipulationProblem):
    def __init__(self,
                 start,
                 goal,
                 T,
                 chain,
                 object_location,
                 object_type,
                 world_trans,
                 object_asset_pos,
                 regrasp_fingers=[],
                 contact_fingers=['index', 'middle', 'ring', 'thumb'],
                 friction_coefficient=0.95,
                 yaw_joint_friction=0.0,
                 obj_dof=1,
                 obj_ori_rep='euler',
                 obj_joint_dim=0,
                 optimize_force=False,
                 turn=False,
                 obj_gravity=False,
                 min_force_dict=None,
                 device='cuda:0',
                 proj_path=None,
                 full_dof_goal=False, 
                 project=False,
                 default_dof_pos=None,
                 contact_constraint_only=False,
                 tactile_controller=False,
                 skip_csvto=False,
                 **kwargs):
        self.tactile_controller = tactile_controller
        self.skip_csvto = skip_csvto
        self.obj_mass = float(kwargs.pop('object_mass', 0.0851))
        self.obj_dof_type = None
        self.object_type = 'screwdriver'
        object_link_name = 'screwdriver_body'
        self.obj_translational_dim = 0
        self.obj_rotational_dim = 3
        self.obj_link_name = object_link_name

        self.contact_points = None
        contact_points_object = None
        if proj_path is not None:
            self.proj_path = proj_path.to(device=device)
        else:
            self.proj_path = None

        # Set default DOF positions if not provided
        if default_dof_pos is None:
            self.default_dof_pos = torch.cat((torch.tensor([[0.1,  0.6, 0.6, 0.6]]).float().to(device=device),
                                            torch.tensor([[-0.1, 0.5, 0.9, 0.9]]).float().to(device=device),
                                            torch.tensor([[0., 0, 0, 0]]).float().to(device=device),
                                            torch.tensor([[1.2, 0.3, 0.3, 1.2]]).float().to(device=device)),
                                            dim=1).to(device).reshape(-1)

        else:
            self.default_dof_pos = default_dof_pos

        super(AllegroScrewdriver, self).__init__(start=start, goal=goal, T=T, chain=chain,
                                                 object_location=object_location,
                                                 object_type=object_type, world_trans=world_trans,
                                                 object_asset_pos=object_asset_pos,
                                                 regrasp_fingers=regrasp_fingers,
                                                 contact_fingers=contact_fingers,
                                                 friction_coefficient=friction_coefficient,
                                                 yaw_joint_friction=yaw_joint_friction,
                                                 obj_dof=obj_dof,
                                                 obj_ori_rep=obj_ori_rep, obj_joint_dim=1,
                                                 optimize_force=optimize_force, device=device,
                                                 turn=turn, obj_gravity=obj_gravity,
                                                 min_force_dict=min_force_dict, 
                                                 full_dof_goal=full_dof_goal,
                                                  contact_points_object=contact_points_object,
                                                  contact_points_dict = self.contact_points,
                                                  project=project,
                                                  contact_constraint_only=contact_constraint_only,
                                                   **kwargs)
        self.friction_coefficient = friction_coefficient
        self.yaw_joint_friction = float(yaw_joint_friction)
        self.object_smoothness_cost_weight = float(kwargs.pop('object_smoothness_cost_weight', 1.0))
        self.upright_cost_weight = float(kwargs.pop('upright_cost_weight', 500.0))

    def _cost(self, xu, rob_link_pts, nearest_robot_pts, start, goal, projected_diffusion=False):
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it

        # Smoothness cost for object degrees of freedom
        smoothness_cost = self.object_smoothness_cost_weight * torch.sum(
            (state[1:, -self.obj_dof:] - state[:-1, -self.obj_dof:]) ** 2
        )
        
        upright_cost = 0
        if not self.project:
            upright_cost = self.upright_cost_weight * torch.sum(
                (state[:, -self.obj_dof:-1] + goal[-self.obj_dof:-1]) ** 2)  # the screwdriver should only rotate in z direction
        return smoothness_cost + upright_cost + super()._cost(xu, rob_link_pts, nearest_robot_pts, start, goal, projected_diffusion=projected_diffusion)


class Proto5Screwdriver(AllegroScrewdriver):
    def __init__(
        self,
        *args,
        full_dof_reference=None,
        robot_sdf_path_prefix=None,
        control_wrist=False,
        **kwargs,
    ):
        if full_dof_reference is None:
            full_dof_reference = torch.zeros(18, dtype=torch.float32)
        device = kwargs.get('device', 'cuda:0')
        kwargs.setdefault(
            'default_dof_pos',
            torch.as_tensor(full_dof_reference, dtype=torch.float32, device=device).reshape(18),
        )
        kwargs.setdefault('full_robot_dof', 18)
        kwargs.setdefault('joint_index', PROTO5_FULL_JOINT_INDEX)
        if control_wrist:
            controlled_joint_groups = ('index', 'middle', 'thumb', 'wrist')
            controlled_joint_index = sum([PROTO5_FULL_JOINT_INDEX[group] for group in controlled_joint_groups], [])
            kwargs.setdefault('robot_dof', len(controlled_joint_index))
            kwargs.setdefault('controlled_joint_index', controlled_joint_index)
            kwargs.setdefault(
                'controlled_joint_min',
                torch.cat([PROTO5_ACTIVE_JOINT_MIN[group] for group in controlled_joint_groups], dim=0),
            )
            kwargs.setdefault(
                'controlled_joint_max',
                torch.cat([PROTO5_ACTIVE_JOINT_MAX[group] for group in controlled_joint_groups], dim=0),
            )
        kwargs.setdefault('ee_names', PROTO5_EE_NAMES)
        kwargs.setdefault('collision_link_names', PROTO5_COLLISION_LINK_NAMES)
        kwargs.setdefault('fingertip_contact_only', True)
        kwargs.setdefault('filter_self_collision_query_points', False)
        kwargs.setdefault('joint_min', PROTO5_ACTIVE_JOINT_MIN)
        kwargs.setdefault('joint_max', PROTO5_ACTIVE_JOINT_MAX)
        kwargs.setdefault('full_dof_reference', full_dof_reference)
        kwargs.setdefault('contact_patch_link_frame_z_max', -0.003)
        if robot_sdf_path_prefix is not None:
            kwargs.setdefault('robot_sdf_path_prefix', robot_sdf_path_prefix)
        super().__init__(*args, **kwargs)
