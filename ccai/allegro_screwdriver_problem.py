from ccai.allegro_contact import AllegroManipulationProblem, PositionControlConstrainedSVGDMPC, add_trajectories, \
    add_trajectories_hardware
import torch
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
        # Mass of the object. Hardcoded for now.
        self.obj_mass = 0.0851
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

    def _cost(self, xu, rob_link_pts, nearest_robot_pts, start, goal, projected_diffusion=False):
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it

        # Smoothness cost for object degrees of freedom
        smoothness_cost = torch.sum((state[1:, -self.obj_dof:] - state[:-1, -self.obj_dof:]) ** 2)
        
        upright_cost = 0
        if not self.project:
            upright_cost = 500 * torch.sum(
                (state[:, -self.obj_dof:-1] + goal[-self.obj_dof:-1]) ** 2)  # the screwdriver should only rotate in z direction
        return smoothness_cost + upright_cost + super()._cost(xu, rob_link_pts, nearest_robot_pts, start, goal, projected_diffusion=projected_diffusion)