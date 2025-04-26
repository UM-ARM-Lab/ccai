import torch
class DynamicsModel:
    "This uses the simulation environment as the dynamcis model"
    def __init__(self, env, num_fingers, include_velocity=False, obj_joint_dim=0, hardware=False):
        self.env = env
        self.num_fingers = num_fingers
        self.include_velocity = include_velocity
        self.obj_joint_dim = obj_joint_dim
        self.hardware = hardware
    def __call__(self, state, action):
        N = action.shape[0]
        # for the 1st env, action is to repeat the current state
        if self.include_velocity:
            tmp_obj_joint = torch.zeros((state.shape[0], self.obj_joint_dim, 2)).to(device=state.device)
            state = state.reshape((N, -1, 2))
            # full_state = torch.cat((state, tmp_obj_joint), dim=-2)
            if self.hardware:
                self.env.set_pose(state, semantic_order=True, zero_velocity=False, ignore_img=True)
            else:
                self.env.set_pose(state, semantic_order=False, zero_velocity=False, ignore_img=True)
            action = self.env.get_state()['q'][0, : 4 * self.num_fingers].to(action.device) + action
            action = action.to(self.env.device)
            
            # action += torch.randn_like(action) * .03
            self.env.step(action, ignore_img=True)
            ret = self.env.dof_states.clone().reshape(N, -1)
        else:
            if self.obj_joint_dim > 0:
                tmp_obj_joint = torch.zeros((state.shape[0], self.obj_joint_dim)).to(device=state.device)
                state = torch.cat((state, tmp_obj_joint), dim=-1)
            self.env.set_pose(state, semantic_order=True, zero_velocity=True, ignore_img=True)
            action = state[:, :4 * self.num_fingers] + action.to(state.device)
            ret = self.env.step(action, ignore_img=True)
        return ret