from isaacgym.torch_utils import quat_apply, quat_mul, quat_conjugate
import numpy as np
import torch
import time
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf

from utils.allegro_utils import partial_to_full_state, full_to_partial_state
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class IKSolver:
    def __init__(self, chain, fingers, device) -> None:
        self.fingers = fingers
        self.num_fingers = len(fingers)
        self.chain = chain
        self.num_particles = 100
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

    def do_IK(self, target_pose, ignore_dims=[3, 4, 5], from_current=False, current = None):

        _lambda = 1e-6
        eye = torch.eye(6, device=self.device)
        ret = torch.zeros(16)
        num_iter = 2000
        for i, finger in enumerate(self.fingers):
            tmp_target_pose = target_pose[i].get_matrix()
            tmp_target_position = tmp_target_pose[:, :3, 3]
            tmp_target_quat = pk.matrix_to_quaternion(tmp_target_pose[:, :3, :3])
            if from_current:
                q = current.clone()
            else:
                q = torch.zeros(4*self.num_fingers, device=self.device)
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
                #print(torch.linalg.norm(error, dim=-1).min())

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
                threshold = 1e-3
                if torch.linalg.norm(error, dim=-1).min() < threshold:
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
