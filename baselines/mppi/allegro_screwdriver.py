import torch
import numpy as np
from utils.allegro_utils import get_screwdriver_top_in_world


class RunningCost:
    def __init__(self, start, goal, include_velocity=False):
        self.start = start
        self.goal = goal
        self.obj_dof = 3
        self.obj_translational_dim = 0
        self.obj_rotational_dim = 3
        self.include_velocity = include_velocity
    
    def __call__(self, state, action):
        state = state.to(action.device)
        N = action.shape[0]
        if self.include_velocity:
            state = state.reshape(N, -1 ,2)
            state = state[:, :, 0] # no the cost is only on the position
            obj_orientation = state[:, -4:-1]
        else:
            obj_orientation = state[:, -4:-1]
                
        action_cost = torch.sum(action ** 2, dim=-1)

        goal_cost = 0
        # terminal cost
        # obj_orientation = tf.euler_angles_to_matrix(obj_orientation, convention='XYZ')
        # obj_orientation = tf.matrix_to_rotation_6d(obj_orientation)
        # goal_orientation = tf.euler_angles_to_matrix(self.goal[-self.obj_rotational_dim:], convention='XYZ')
        # goal_orientation = tf.matrix_to_rotation_6d(goal_orientation)
        # terminal cost
        goal_cost = goal_cost + torch.sum((20 * (obj_orientation - self.goal.unsqueeze(0)) ** 2), dim=-1)

        #upright cost
        upright_cost = 10000 * torch.sum(obj_orientation[:, :-1] ** 2, dim=-1)
        # dropping cost
        cost = action_cost + goal_cost + upright_cost
        cost = torch.nan_to_num(cost, nan=1e6)

        return cost

class ValidityCheck:
    def __init__(self, obj_chain, obj_dof, world_trans, obj_pose):
        self.nominal_screwdriver_top = np.array([0, 0, 1.405])
        self.obj_chain = obj_chain
        self.obj_dof = obj_dof
        self.world_trans = world_trans
        self.obj_pose = obj_pose

    def check_validity(self, state):
        screwdriver_top_pos = get_screwdriver_top_in_world(state[0, -self.obj_dof:], self.obj_chain, self.world_trans, self.obj_pose)
        screwdriver_top_pos = screwdriver_top_pos.detach().cpu().numpy()
        distance2nominal = np.linalg.norm(screwdriver_top_pos - self.nominal_screwdriver_top)
        if distance2nominal > 0.02:
            validity_flag = False
        else:
            validity_flag = True
        return validity_flag


   