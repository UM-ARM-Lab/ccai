from ccai.allegro_contact_diffusion import AllegroManipulationProblemDiff
import torch

class AllegroScrewdriverDiff(AllegroManipulationProblemDiff):
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
                 device='cuda:0', **kwargs):
        super(AllegroScrewdriverDiff, self).__init__(start=start, goal=goal, T=T, chain=chain,
                                                 object_location=object_location,
                                                 object_type=object_type, world_trans=world_trans,
                                                 object_asset_pos=object_asset_pos,
                                                 regrasp_fingers=regrasp_fingers,
                                                 contact_fingers=contact_fingers,
                                                 friction_coefficient=friction_coefficient,
                                                 obj_dof=obj_dof,
                                                 obj_ori_rep=obj_ori_rep, obj_joint_dim=1,
                                                 optimize_force=optimize_force, device=device)
        self.friction_coefficient = friction_coefficient

    def _cost(self, xu, start, goal):
        # TODO: check if the addtional term of the smoothness cost and running goal cost is necessary
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it

        smoothness_cost = torch.sum((state[1:, -self.obj_dof:] - state[:-1, -self.obj_dof:]) ** 2)
        upright_cost = 500 * torch.sum(
            (state[:, -self.obj_dof:-1]) ** 2)  # the screwdriver should only rotate in z direction
        return smoothness_cost + upright_cost + super()._cost(xu, start, goal)

