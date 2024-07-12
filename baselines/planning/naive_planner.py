from isaacgym.torch_utils import quat_apply, quat_mul, quat_conjugate

import numpy as np

import torch

import time
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
# import pytorch3d.transforms as tf

from utils.allegro_utils import partial_to_full_state, full_to_partial_state
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class NaivePlanner:
    def __init__(self, chain, fingers, ee_peg_frame, T, world2robot, goal, obj_dof_code, device, obj_pos=None) -> None:
        self.fingers = fingers
        self.num_fingers = len(fingers)
        self.chain = chain
        self.ee_peg_frame = ee_peg_frame
        self.T = T
        self.world2robot = world2robot
        self.goal = goal
        self.obj_dof_code = obj_dof_code
        self.translational_dim = sum(obj_dof_code[:3])
        self.rotational_dim = sum(obj_dof_code[3:])
        self.obj_dof = self.translational_dim + self.rotational_dim
        self.num_particles = 100
        self.obj_pos = obj_pos
        if self.translational_dim > 0:
            self.goal_pos = self.goal[:self.translational_dim]
        if self.rotational_dim > 0:
            self.goal_R = R.from_euler('XYZ', goal[self.translational_dim:].detach().cpu().numpy())
        self.device = device
        self.finger2index = {'index': [0, 1, 2, 3],
                             'middle': [4, 5, 6, 7],
                             'ring': [8, 9, 10, 11],
                             'thumb': [12, 13, 14, 15]}
        self.ee_names = {
            'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
            }
        frame_indices = [self.chain.frame_to_idx[self.ee_names[finger]] for finger in self.fingers]    # combined chain
        self.frame_indices = torch.tensor(frame_indices)
    def step(self, state):
        self.current_robot_q = state[:4*self.num_fingers]
        next_pos = np.zeros(3)
        next_quat = np.zeros(4)
        next_quat[-1] = 1
        obj_state = state[-self.obj_dof:]
        if self.translational_dim > 0:
            obj_pos = obj_state[:self.translational_dim]
            interp_position = np.linspace(obj_pos.cpu().numpy(), self.goal_pos.cpu().numpy(), self.T + 1)
            next_pos = interp_position[1]
        else:
            next_pos = self.obj_pos
        if self.rotational_dim > 0:
            obj_ori = obj_state[self.translational_dim:]
            obj_R = R.from_euler('XYZ', obj_ori.detach().cpu().numpy())
            obj_quat = obj_R.as_quat()
            key_times = [0, self.T]
            times = np.linspace(0, self.T, self.T + 1)
            slerp = Slerp(key_times, R.concatenate([obj_R, self.goal_R]))
            interp_rots = slerp(times)
            interp_rots = interp_rots.as_quat()
            next_quat = interp_rots[1]
        obj_trans = tf.Transform3d(pos=torch.tensor(next_pos, device=self.device).float(),
                                    rot=torch.tensor(
                                        [next_quat[3], next_quat[0], next_quat[1], next_quat[2]],
                                        device=self.device).float(), device=self.device)
        ee_world = [obj_trans.compose(ee) for ee in self.ee_peg_frame]
        ee_robot = [self.world2robot.inverse().compose(ee) for ee in ee_world]
        action = self.do_IK(ee_robot, from_current=True)
        return action    

    def do_IK(self, target_pose, ignore_dims=[3, 4, 5], from_current=False):
        """
        Perform IK using damped least squares
        :param target_pose: target pose in world frame (x, y, z, quat)
        :param from_current: whether to start from current configuration, if False start from zero vector
        :param ignore_dims: dimensions to ignore in IK, i.e. [3, 4, 5] would do position IK ignoring orientation
        :return:
        """

        _lambda = 1e-6
        eye = torch.eye(6, device=self.device)
        ret = torch.zeros(16)
        num_iter = 2000
        for i, finger in enumerate(self.fingers):
            tmp_target_pose = target_pose[i].get_matrix()
            tmp_target_position = tmp_target_pose[:, :3, 3]
            tmp_target_quat = pk.matrix_to_quaternion(tmp_target_pose[:, :3, :3])
            if from_current:
                q = self.current_robot_q.clone()
            else:
                q = torch.zeros(4*self.num_fingers, device=target_pose.device)
            q = partial_to_full_state(q, fingers=self.fingers)
            q = q.unsqueeze(0)
            q = q.repeat(self.num_particles, 1)
            q = q + torch.randn_like(q) * 0.2
            for j in range(num_iter):
                mat = self.chain.forward_kinematics(q)[self.ee_names[finger]].get_matrix()
                ee_pos = mat[:, :3, 3]
                ee_orn = pk.matrix_to_quaternion(mat[:, :3, :3])
                J = self.chain.jacobian(q, link_indices=self.frame_indices[i]) # TODO: check if this is correct
                J = J[:, :, self.finger2index[finger]]
                ee_orn = self.quat_change_convention(ee_orn, current='wxyz')
                orn_error = self.orientation_error(tmp_target_quat.repeat(self.num_particles, 1), ee_orn)
                pos_error = tmp_target_position - ee_pos
                error = torch.cat([pos_error, orn_error], dim=1)

                # ignore dimensions
                error[:, ignore_dims] = 0
                # print(error)

                pseudo_inv = J.permute(0, 2, 1) @ torch.linalg.inv(J @ J.permute(0, 2, 1) + _lambda * eye)
                qdot = pseudo_inv @ error.unsqueeze(-1)

                q[:, self.finger2index[finger]] = q[:, self.finger2index[finger]] + qdot.squeeze(-1)

                # clamp joint angles
                q = torch.clamp(q, min=-2.8973, max=2.8973)
                max_lims = torch.tensor([0.47, 1.6099999999999999, 1.7089999999999999, 1.6179999999999999,
                                        0.47, 1.6099999999999999, 1.7089999999999999, 1.6179999999999999,
                                        0.47, 1.6099999999999999, 1.7089999999999999, 1.6179999999999999,
                                        1.5, 1.1629999999999998, 1.644, 1.7189999999999999])
                min_lims = torch.tensor([-0.47, -0.19599999999999998, -0.17400000000000002, -0.227,
                                        -0.47, -0.19599999999999998, -0.17400000000000002, -0.227,
                                        -0.47, -0.19599999999999998, -0.17400000000000002, -0.227,
                                        0.263, -0.105,-0.18899999999999997, -0.162])
                q = torch.clamp(q, min=min_lims, max=max_lims)
                if torch.linalg.norm(error, dim=-1).min() < 1e-3:
                    ret[self.finger2index[finger]] = q[torch.argmin(torch.linalg.norm(error, dim=-1))][self.finger2index[finger]]
                    break
                if j == num_iter - 1:
                    print(f'{finger} Failed to converge')
                    print(error[torch.argmin(torch.linalg.norm(error, dim=-1))])
                    ret[self.finger2index[finger]] = q[torch.argmin(torch.linalg.norm(error, dim=-1))][self.finger2index[finger]]

        ret = full_to_partial_state(ret, fingers=self.fingers)
        return ret
    @staticmethod
    def orientation_error(desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    @staticmethod
    def quat_change_convention(q, current='xyzw'):
        if current == 'xyzw':
            return torch.stack(
                (q[:, 3], q[:, 0], q[:, 1], q[:, 2]), dim=-1)

        if current == 'wxyz':
            return torch.stack((
                q[:, 1], q[:, 2], q[:, 3], q[:, 0]), dim=-1)
    @staticmethod
    def quat_conjugate(q):
        con_quat = - q # conjugate
        con_quat[..., 0] = q[..., 0]
        return con_quat
