import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from .allegro_ros import RosNode
import torch
import rospy

class ObjectPoseReader:
    def __init__(self, obj='valve') -> None:
        self.obj = obj
        self.__intrinsic = np.array([[621.80984333,   0.,         651.09118583],
        [  0.,         621.66658768, 352.88384525],
        [  0.,           0. ,          1.,        ]])
        self.__dist = np.array([[0.07583722,  0.00042308, -0.00245659 , 0.00797877 , 0.01058895]])
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        self.__detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        self.__cap = cv2.VideoCapture(0)
        if self.obj == 'valve':
            self.__objp = np.array([[0., 0., 0.],
                                [1., 0., 0.],
                                [1., 1., 0.],
                                [0., 1., 0.]])
            self.__axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)
            self.__initial_pose = None
            while True:
                self.__initial_pose = self._get_valve_rot_vec()
                if self.__initial_pose is not None:
                    break
        elif self.obj == 'screwdriver':
            self.__objp = np.array([[-0.5, -0.5, 0.],
                    [-0.5, 0.5, 0.],
                    [0.5, 0.5, 0.],
                    [0.5, -0.5, 0.]]) * 35
            global_rvecs = np.load('rvecs.npy')
            global_tvecs = np.load('tvecs.npy')
            rotation = R.from_rotvec(global_rvecs[:,0])
            rotation_mat = rotation.as_matrix()
            translation = global_tvecs
            world2cam = np.concatenate((rotation_mat, translation), axis=1) # transform from world coordinate to camera coordinate
            world2cam = np.concatenate((world2cam, np.array([[0, 0, 0, 1]])), axis=0)
            global_trans = np.array([[0, 1, 0], 
                                    [1, 0, 0],
                                    [0, 0, -1]]).T
            global_trans = np.concatenate((global_trans, np.array([[0, 0, 0]]).T), axis=1)
            global_trans = np.concatenate((global_trans, np.array([[0, 0, 0, 1]])), axis=0)
            cam2world = np.linalg.inv(world2cam)
            self.__cam2world = global_trans @ cam2world

            self.__screwdriver2marker = []
            rotate_90_trans = np.array([[1, 0, 0],
                                        [0, 0, 1],
                                        [0, -1, 0]])
            for i in range(4):
                trans = np.array([[0, 0, 1],
                                [-1, 0, 0],
                                [0, -1, 0]])
                for j in range(i):
                    trans = rotate_90_trans @ trans
                trans = np.concatenate((trans, np.array([[0, 0, 0]]).T), axis=1)
                trans = np.concatenate((trans, np.array([[0, 0, 0, 1]])), axis=0)
                self.__screwdriver2marker.append(trans)

    def _get_valve_rot_vec(self):
        ret, frame = self.__cap.read()
        markerCorners, markerIds, rejectedCandidates = self.__detector.detectMarkers(frame)
        if len(markerCorners) == 0:
            print("No readings from camera.")
            return None
        ret,rvecs, tvecs = cv2.solvePnP(self.__objp, markerCorners[0], self.__intrinsic, self.__dist) 
        pose = R.from_rotvec(rvecs[:,0]).inv()
        return pose
    def get_valve_angle(self):
        while True:
            for i in range(7):
                # clear up the buffer
                current_pose = self._get_valve_rot_vec()
            current_pose = self._get_valve_rot_vec()
            if current_pose is not None:
                break
        rot_diff = current_pose * self.__initial_pose.inv()
        diff_vec = rot_diff.as_rotvec()
        angle = np.linalg.norm(diff_vec) * np.sign(diff_vec[2])
        return angle
    def get_screwdriver_state(self):
        center_coors = []
        screwdriver_ori_eulers = []
        ret, frame = self.__cap.read()
        markerCorners, markerIds, rejectedCandidates = self.__detector.detectMarkers(frame)
        if len(markerCorners) > 0 and markerIds.max() < 4:
            cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            for i in range(len(markerCorners)):

                ret, rvecs, tvecs = cv2.solvePnP(self.__objp, markerCorners[i], self.__intrinsic, self.__dist) 
                cv2.drawFrameAxes(frame, self.__intrinsic, self.__dist, rvecs, tvecs, 35 / 2) # debug
                rotation = R.from_rotvec(rvecs[:,0])
                rotation_mat = rotation.as_matrix()
                translation = tvecs
                marker2cam = np.concatenate((rotation_mat, translation), axis=1)
                marker2cam = np.concatenate((marker2cam, np.array([[0, 0, 0, 1]])), axis=0)
                marker2world = self.__cam2world @ marker2cam
                world_coor = marker2world[:3, 3]
                centroid_direction = marker2world[:3, 2]
                center_coor = world_coor - centroid_direction * 21
                # print(markerIds[i], center_coor)
                trans_mat = self.__screwdriver2marker[markerIds[i].item()]
                trans_mat = marker2world @ trans_mat
                scrwedriver_ori_mat = trans_mat[:3, :3]
                screwdriver_ori_euler = R.from_matrix(scrwedriver_ori_mat).as_euler('XYZ', degrees=True)
                # print(markerIds[i].item(), screwdriver_ori_euler)
                center_coors.append(center_coor)
                screwdriver_ori_eulers.append(screwdriver_ori_euler)
            center_coor = np.mean(center_coors, axis=0)
            screwdriver_ori_euler = np.mean(screwdriver_ori_eulers, axis=0)
            if np.isnan(center_coor).any() or np.isnan(screwdriver_ori_euler).any():
                breakpoint()
            print(center_coor, screwdriver_ori_euler)
            cv2.imshow('frame', frame) # debug
            return center_coor, screwdriver_ori_euler
        else:
            return None, None
    def get_state(self):
        if self.obj == 'valve':
            return self.get_valve_angle()
        elif self.obj == 'screwdriver':
            while True:
                center_coor, screwdriver_ori_euler = self.get_screwdriver_state()
                if center_coor is not None:
                    break
            return center_coor, screwdriver_ori_euler

class HardwareEnv:
    def __init__(self, default_pos, finger_list=['index', 'middle', 'ring', 'thumb'], kp=4, obj='valve', ori_only=True, device='cuda:0'):
        self.__all_finger_list = ['index', 'middle', 'ring', 'thumb']
        self.__finger_list = finger_list
        self.__ros_node = RosNode(kp=kp)

        self.obj_reader = ObjectPoseReader(obj=obj)

        self.device = device
        self.default_pos = default_pos
        self.ori_only = ori_only
    
    def get_state(self):
        robot_state = self.__ros_node.allegro_joint_pos.float()
        robot_state = robot_state.to(self.device)
        index, mid, ring, thumb = torch.chunk(robot_state, chunks=4, dim=-1)
        state = {}
        state['index'] = index
        state['middle'] = mid
        state['ring'] = ring
        state['thumb'] = thumb
        q = []
        for finger_name in self.__finger_list:
            q.append(state[finger_name])
        if self.obj == 'valve':
            ori = self.obj_reader.get_state()
            ori = torch.tensor([ori]).float().to(self.device)
            q.append(ori)
        elif self.obj == 'screwdriver':
            pos, ori = self.obj_reader.get_state()
            pos = torch.tensor(pos).float().to(self.device)
            ori = torch.tensor(ori).float().to(self.device)
            if self.ori_only:
                q.append(ori)
            else:
                raise NotImplementedError
        all_state = torch.cat((robot_state, ori), dim=-1)
        state['all_state'] = all_state
        state['q'] = torch.cat(q).unsqueeze(0)
        state['theta'] = ori
        return state
    def step(self, action):
        action = self.partial_to_full_state(action)
        # action[:, -2] += 0.25
        if len(action.shape) == 2:
            action = action.squeeze(0)
        self.__ros_node.apply_action(action)
        return self.get_state()
    def reset(self):
        return self.__ros_node.apply_action(self.default_pos.squeeze(0))
    def partial_to_full_state(self, partial):
        """
        :params partial: B x 8 joint configurations for index and thumb
        :return full: B x 16 joint configuration for full hand

        # assume that default is zeros, but could change
        """
        finger_data = torch.chunk(partial, chunks=len(self.__finger_list), dim=-1)
        full = []
        ctr = 0
        for finger_name in self.__all_finger_list:
            if finger_name not in self.__finger_list:
                full.append(torch.zeros_like(finger_data[0]))
            else:
                full.append(finger_data[ctr])
                ctr += 1
        full = torch.cat(full, dim=-1)
        return full

if __name__ == "__main__":
    # env = HardwareEnv(default_pos=torch.zeros(16), finger_list=['index', 'thumb'])
    # while True:
    #     action = torch.randn(8) / 5
    #     state = env.step(action)
    #     print(state)
    #     rospy.sleep(1)
    reader = ObjectPoseReader(obj='screwdriver')
    while True:
        center_coor, screwdriver_ori_euler = reader.get_state()
        print(center_coor, screwdriver_ori_euler)
        cv2.waitKey(1)
        
