import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from .allegro_ros import RosNode
import torch
import rospy

class ObjectPoseReader:
    def __init__(self, obj='valve', mode='relative') -> None:
        # in this file, the world frame is defined as the root of the robot hand, but the axis aligns with the actual world frame
        self.mode = mode
        self.obj = obj
        # self.__intrinsic = np.array([[621.80984333,   0.,         651.09118583],
        # [  0.,         621.66658768, 352.88384525],
        # [  0.,           0. ,          1.,        ]])
        # self.__dist = np.array([[0.07583722,  0.00042308, -0.00245659 , 0.00797877 , 0.01058895]])

        self.__intrinsic = np.array([[606.42074687,   0.,         648.68751226],
                                    [0.,         606.09387277, 369.09085521],
                                    [0.,           0.,           1.        ]])
        self.__dist = np.array([[ 0.11198521, -0.15500009,  0.00198888,  0.0008579,   0.08273805]])
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
            if mode == 'absolute': # not working any more, since the obj_p has been changed 
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
            elif mode == 'relative':
                # the following transformation is not following the definition step by step. 
                # instead, it's directly finding the world frame based on the tag
                self.__objp_palm = np.array([[-0.5, 0, -0.5],
                                            [0.5, 0, -0.5],
                                            [0.5, 0, 0.5],
                                            [-0.5, 0, 0.5]]) * 35
                # self.__world2palm_marker = np.array([[1, 0, 0, 20],
                #                                     [0, 1, 0, 50],
                #                                     [0, 0, 1, -110],
                #                                     [0, 0, 0, 1]])
                hand_center2palm_marker= np.array([[1, 0, 0, 15],
                                                    [0, 1, 0, 55],
                                                    [0, 0, 1, -120],
                                                    [0, 0, 0, 1]])
                # assume 40 degrees rotation around y axis
                world2hand_center = np.array([[0.7660444, 0, 0.6427876, 0],
                                            [0, 1, 0, 0],
                                            [-0.6427876, 0, 0.7660444, 0],
                                            [0, 0, 0, 1]])
                # assume 45 degrees rotation around y axis
                # world2hand_center = np.array([[0.7071068, 0, 0.7071068, 0],
                #                             [0, 1, 0, 0],
                #                             [-0.7071068, 0, 0.7071068, 0],
                #                             [0, 0, 0, 1]])

                # assume 50 degrees rotation around y axis
                # world2hand_center = np.array([[0.6427876,  0.0000000,  0.7660444],
                #                         [0.0000000,  1.0000000,  0.0000000],
                #                         [-0.7660444,  0.0000000,  0.6427876]])

                self.__world2palm_marker =  hand_center2palm_marker @ world2hand_center
                # self.__world2palm_marker = np.array([[0.7071068, 0, -0.7071068, -25],
                #                                     [0, 1, 0, -50],
                #                                     [0.7071068, 0, 0.7071068, 110],
                #                                     [0, 0, 0, 1]])
                # self.__world2palm_marker = np.linalg.inv(self.__world2palm_marker)
                # self.__world2palm_marker = np.array([[0.7071068, 0, 0.7071068, 63.63960861],
                #                                     [0, 1, 0, 50],
                #                                     [-0.7071068, 0, 0.7071068, -91.92387911],
                #                                     [0, 0, 0, 1]])
                while True:
                    flag = self.get_robot_frame()
                    if flag:
                        break
            else:
                raise NotImplementedError

            self.__objp = np.array([[0.5, 0.0, -0.5],
                    [-0.5, 0.0, -0.5],
                    [-0.5, 0.0, 0.5],
                    [0.5, 0.0, 0.5]]) * 35
            self.__root2marker = np.array([[1, 0, 0, 0], # the root of the screwdriver to the screwdriver marker
                                    [0, 1, 0, 20.5],
                                    [0, 0, 1, -80],
                                    [0, 0, 0, 1]])
            self.__nominal_root2root = []
            rotate_90_trans = np.array([[0, 1, 0, 0],
                                        [-1, 0, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
            for i in range(4):
                trans = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
                for _ in range(i):
                    trans = rotate_90_trans @ trans
                self.__nominal_root2root.append(trans)
        elif self.obj == 'peg':
            self.__objp_palm = np.array([[-0.5, 0, -0.5],
                                        [0.5, 0, -0.5],
                                        [0.5, 0, 0.5],
                                        [-0.5, 0, 0.5]]) * 35
            hand_center2palm_marker= np.array([[1, 0, 0, 15],
                                                [0, 1, 0, 55],
                                                [0, 0, 1, -80],
                                                [0, 0, 0, 1]])
            # assume 50 degrees rotation around y axis
            world_temp2hand_center = np.array([[0.6427876,  0.0000000,  0.7660444, 0.0],
                                        [0.0000000,  1.0000000,  0.0000000, 0.0],
                                        [-0.7660444,  0.0000000,  0.6427876, 0.0],
                                        [0.0, 0.0, 0.0, 1.0]])
            # rotate around z axis for - 90 degrees
            world2world_temp = np.array([[0, 1, 0, 0],
                                        [-1, 0, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
            world2hand_center = world_temp2hand_center @ world2world_temp

            self.__world2palm_marker =  hand_center2palm_marker @ world2hand_center
            # read the palm marker
            while True:
                flag = self.get_robot_frame()
                if flag:
                    break

            self.__objp = np.array([[0, -0.5, 0.5],
                                    [0, -0.5, -0.5],
                                    [0, 0.5, -0.5],
                                    [0.0, 0.5, 0.5]]) * 35
            
            self.__root2marker = np.array([[1, 0, 0, 20.0], # the root of the screwdriver to the screwdriver marker
                                    [0, 1, 0, 0.0],
                                    [0, 0, 1, 0.0],
                                    [0, 0, 0, 1]])
            # the following are stats in isaac gym, needs to be updated whenever you change the envs
            self.robot_root_p = np.array([0.11, -0.023, 0.3]) * 1000
            self.obj_root_p = np.array([0.05, 0, 0.205]) * 1000
            wall_corner_p = np.array([0, 0, 0.25])

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
        root_coors = []
        screwdriver_ori_eulers = []
        ctr = 0
        while True:
            ret, frame = self.__cap.read()
            ctr += 1
            if ctr >= 7 and ret:
                break
        markerCorners, markerIds, rejectedCandidates = self.__detector.detectMarkers(frame)
        if len(markerCorners) > 0 and markerIds.max() <= 4:
            # cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            for marker_corner, marker_id in zip(markerCorners, markerIds):
                if marker_id.item() >= 4:
                    continue

                ret, rvecs, tvecs = cv2.solvePnP(self.__objp, marker_corner, self.__intrinsic, self.__dist) 
                # cv2.drawFrameAxes(frame, self.__intrinsic, self.__dist, rvecs, tvecs, 35 / 2) # debug
                rotation = R.from_rotvec(rvecs[:,0])
                rotation_mat = rotation.as_matrix()
                translation = tvecs
                marker2cam = np.concatenate((rotation_mat, translation), axis=1)
                marker2cam = np.concatenate((marker2cam, np.array([[0, 0, 0, 1]])), axis=0)
                marker2world = self.__cam2world @ marker2cam
                root2world = marker2world @ self.__root2marker
                trans_mat = self.__nominal_root2root[marker_id.item()]
                root2world = root2world @ trans_mat 
                root_coor = root2world[:3, 3]
                scrwedriver_ori_mat = root2world[:3, :3]
                screwdriver_ori_euler = R.from_matrix(scrwedriver_ori_mat).as_euler('XYZ', degrees=False)
                # print(markerIds[i].item(), screwdriver_ori_euler)
                root_coors.append(root_coor)
                screwdriver_ori_eulers.append(screwdriver_ori_euler)
            root_coor = np.mean(root_coors, axis=0)
            screwdriver_ori_euler = np.mean(screwdriver_ori_eulers, axis=0)
            # cv2.imshow('frame', frame) # debug
            return root_coor, screwdriver_ori_euler
        else:
            print("No readings from camera.")
            return None, None
        
    def get_peg_state(self):
        ctr = 0
        while True:
            ret, frame = self.__cap.read()
            ctr += 1
            if ctr >= 7 and ret:
                break
        markerCorners, markerIds, rejectedCandidates = self.__detector.detectMarkers(frame)
        peg_sim_world_coor, peg_ori_euler = None, None
        if len(markerCorners) > 0 and markerIds.max() <= 4:
            # cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            for marker_corner, marker_id in zip(markerCorners, markerIds):
                if marker_id.item() == 1:
                    ret, rvecs, tvecs = cv2.solvePnP(self.__objp, marker_corner, self.__intrinsic, self.__dist) 
                    # cv2.drawFrameAxes(frame, self.__intrinsic, self.__dist, rvecs, tvecs, 35 / 2) # debug
                    rotation = R.from_rotvec(rvecs[:,0])
                    rotation_mat = rotation.as_matrix()
                    translation = tvecs
                    marker2cam = np.concatenate((rotation_mat, translation), axis=1)
                    marker2cam = np.concatenate((marker2cam, np.array([[0, 0, 0, 1]])), axis=0)
                    marker2world = self.__cam2world @ marker2cam
                    root2world = marker2world @ self.__root2marker
                    root_coor = root2world[:3, 3]
                    peg_ori_mat = root2world[:3, :3]
                    peg_ori_euler = R.from_matrix(peg_ori_mat).as_euler('XYZ', degrees=False)

                    peg_sim_world_coor = root_coor + self.robot_root_p - self.obj_root_p
                    # peg_sim_world_coor = root_coor
            # cv2.imshow('frame', frame) # debug
            if peg_sim_world_coor is None:
                print("No readings from camera.")
            return peg_sim_world_coor / 1000, peg_ori_euler
        else:
            cv2.imshow('frame', frame) # debug
            print("No readings from camera.")
            return None, None
    
    def get_robot_frame(self):
        
        ctr = 0
        while True:
            ret, frame = self.__cap.read()
            ctr += 1
            if ctr >= 7 and ret:
                break
        palm_marker_flag = False
        markerCorners, markerIds, rejectedCandidates = self.__detector.detectMarkers(frame)
        if len(markerCorners) > 0 and markerIds.max() <= 4:
            # cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            for marker_corner, marker_id in zip(markerCorners, markerIds):
                if marker_id.item() == 4:
                    ret, rvecs, tvecs = cv2.solvePnP(self.__objp_palm, marker_corner, self.__intrinsic, self.__dist) 
                    cv2.drawFrameAxes(frame, self.__intrinsic, self.__dist, rvecs, tvecs, 35 / 2)
                    rotation = R.from_rotvec(rvecs[:,0])
                    rotation_mat = rotation.as_matrix()
                    translation = tvecs
                    palm_marker2cam = np.concatenate((rotation_mat, translation), axis=1)
                    palm_marker2cam = np.concatenate((palm_marker2cam, np.array([[0, 0, 0, 1]])), axis=0)
                    # the origin of the world frame is defined as the root of the robot hand
                    world2cam = palm_marker2cam @ self.__world2palm_marker 
                    self.__cam2world = np.linalg.inv(world2cam)
                    palm_marker_flag = True
        return palm_marker_flag

    def get_state(self):
        if self.obj == 'valve':
            return self.get_valve_angle()
        elif self.obj == 'screwdriver':
            while True:
                center_coor, screwdriver_ori_euler = self.get_screwdriver_state()
                # print(center_coor, screwdriver_ori_euler)
                if center_coor is not None:
                    break
            return center_coor, screwdriver_ori_euler
        elif self.obj == 'peg':
            while True:
                center_coor, peg_ori_euler = self.get_peg_state()
                # print(center_coor, peg_ori_euler)
                if center_coor is not None:
                    break
            return center_coor, peg_ori_euler

class HardwareEnv:
    def __init__(self, default_pos, num_repeat=1, gradual_control=False, 
                 finger_list=['index', 'middle', 'ring', 'thumb'], kp=4, 
                 obj='valve', ori_only=True, mode='relative', device='cuda:0'):
        self.__all_finger_list = ['index', 'middle', 'ring', 'thumb']
        self.obj = obj
        self.__finger_list = finger_list
        self.__ros_node = RosNode(kp=kp, num_repeat=num_repeat, gradual_control=gradual_control)

        self.obj_reader = ObjectPoseReader(obj=obj, mode=mode)

        self.device = device
        self.default_pos = default_pos.clone()
        self.ori_only = ori_only
    
    def get_state(self, return_dict=False):
        rospy.sleep(0.5)
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
            obj_config = ori
            q.append(ori)
        elif self.obj == 'screwdriver':
            pos, ori = self.obj_reader.get_state()
            pos = torch.tensor(pos).float().to(self.device)
            ori = torch.tensor(ori).float().to(self.device)
            # ori = ori * 0 # debug
            if self.ori_only:
                obj_cofig = ori
                q.append(ori)
                # q.append(torch.zeros(1).float().to(self.device)) # add the screwdriver cap angle
            else:
                raise NotImplementedError
        elif self.obj == 'peg':
            pos, ori = self.obj_reader.get_state()
            pos = torch.tensor(pos).float().to(self.device)
            ori = torch.tensor(ori).float().to(self.device)
            obj_config = torch.cat((pos, ori))
            q.append(obj_config)
        all_state = torch.cat((robot_state, obj_config), dim=-1)
        state['all_state'] = all_state.unsqueeze(0)
        state['q'] = torch.cat(q).unsqueeze(0)
        if not return_dict:
            state = state['q']
        # state['theta'] = ori
        # state = q
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
    reader = ObjectPoseReader(obj='peg')
    while True:
        center_coor, screwdriver_ori_euler = reader.get_state()
        print(center_coor, screwdriver_ori_euler)
        cv2.waitKey(1)
        
