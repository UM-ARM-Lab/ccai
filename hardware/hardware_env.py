import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from .allegro_ros import RosNode
import torch
import rospy
from shapely.geometry import Polygon

class ObjectPoseReader:
    def __init__(self, obj='valve', mode='relative') -> None:
        self.mode = mode
        self.obj = obj
        # self.__intrinsic = np.array([[621.80984333,   0.,         651.09118583],
        # [  0.,         621.66658768, 352.88384525],
        # [  0.,           0. ,          1.,        ]])
        # self.__dist = np.array([[0.07583722,  0.00042308, -0.00245659 , 0.00797877 , 0.01058895]])
        
        # self.__intrinsic = np.array([[620.01947778,   0.,         634.39543476],
        # [  0.,         621.01114219, 373.00241614],
        # [  0.,           0. ,          1.,        ]])
        # self.__dist = np.array([[0.10693622,  -0.11111605, 0.00523414 , -0.00723972 , 0.0405243]])

        # self.__intrinsic = np.array([[627.356434,     0.     ,    648.64167401],
        # [  0.,         626.97086464 ,353.32197956],
        # [  0.     ,      0.   ,        1.        ]])
        # self.__dist = np.array([[0.11001195, -0.10927342, -0.0065648,   0.00226242 , 0.0547425]])

        # self.__intrinsic = np.array([[628.31507209 ,  0.        , 637.88670275],
        # [  0.    ,     628.08092551, 351.3041728 ],
        # [  0.     ,      0.         ,  1.        ]])
        # self.__dist = np.array([[ 0.10698373, -0.08771528, -0.00798322, -0.00410741,  0.04082028]])

        # self.__intrinsic = np.array(    [[631.51928386 ,  0.   ,      631.83484949],
        #     [  0.    ,     632.21401224, 345.20179321],
        #     [  0.   ,        0. ,          1.        ]])
        # self.__dist = np.array([[ 0.12413314, -0.11324339 ,-0.01188807 ,-0.00755358 , 0.05668905]])

        self.__intrinsic = np.array(
            [[613.51871107 ,  0.         ,636.90898042],
            [  0.       ,  615.81195063 ,366.01265147],
            [  0.        ,   0.         ,  1.        ]]
        )
        self.__dist = np.array([[ 0.13515046, -0.17302101, -0.00296737, -0.00497764,  0.08863282]])
        
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
                world2hand_center = np.array([[0.7071068, 0, 0.7071068, 0],
                                            [0, 1, 0, 0],
                                            [-0.7071068, 0, 0.7071068, 0],
                                            [0, 0, 0, 1]])
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
                # while True:
                #     flag = self.get_robot_frame()
                #     if flag:
                #         break

                
                # self.__cam2world = np.array(
                #     [[-9.55239386e-01, -5.31440129e-02 ,-2.91021263e-01 , 2.74029745e+01],
                #     [ 2.91324818e-01 , 2.11427720e-03, -9.56621858e-01,  4.25566549e+02],
                #     [ 5.14540240e-02 ,-9.98584594e-01,  1.34625290e-02, -4.58435857e+01],
                #     [ 0.00000000e+00,  0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]]
                #  )


                # self.__cam2world = np.array(
                #     [[-9.46183190e-01 ,-4.59892001e-02, -3.20347173e-01 , 4.20749042e+01],
                #     [ 3.21853710e-01, -3.01083015e-02, -9.46310562e-01,  4.13520006e+02],
                #     [ 3.38749565e-02, -9.98488072e-01,  4.32897635e-02 ,-5.99224171e+01],
                #     [ 0.00000000e+00,  0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]]
                #  )
                   
                # self.__cam2world = np.array(
                #     [[-9.60303369e-01 ,-5.40745965e-02, -2.73666446e-01 , 3.25387834e+00],
                #     [ 2.74022475e-01 , 8.61456392e-04, -9.61722902e-01 , 4.32167441e+02],
                #     [ 5.22405296e-02, -9.98536500e-01,  1.39903953e-02 ,-4.82987398e+01],
                #     [ 0.00000000e+00,  0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]]
                #  )

                # self.__cam2world = np.array(
                #     [[-9.51957181e-01, -5.66268316e-02 ,-3.00949954e-01 , 8.34654917e+00],
                #     [ 3.01873872e-01 ,-8.33467526e-03 ,-9.53311439e-01,  3.93356872e+02],
                #     [ 5.14746861e-02, -9.98360597e-01,  2.50284148e-02 ,-5.18497540e+01],
                #     [ 0.00000000e+00,  0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]]
                # )

                self.__cam2world = np.array(
                    [[-9.56905016e-01 ,-5.27570113e-02 ,-2.85568617e-01, -1.49615486e+01],
                    [ 2.85823353e-01 , 2.81250537e-03, -9.58278196e-01,  3.86554300e+02],
                    [ 5.13590569e-02 ,-9.98603392e-01,  1.23878853e-02, -5.50173840e+01],
                    [ 0.00000000e+00,  0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]]
                )

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

        # while True:
        #     flag = self.get_robot_frame()
        #     if flag:
        #         break
        # print(self.__cam2world)

        while True:
            ret, frame = self.__cap.read()
            ctr += 1
            if ctr >= 7 and ret:
                break
        markerCorners, markerIds, rejectedCandidates = self.__detector.detectMarkers(frame)
        if len(markerCorners) > 0 and markerIds.max() <= 4:
            cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            areas = []
                
            for marker_corner, marker_id in zip(markerCorners, markerIds):
                if marker_id.item() >= 4:
                    continue
                
                # Use the marker_corners to calculate the area of the marker
                polygon = Polygon(np.array(marker_corner).squeeze())
                area = polygon.area
                areas.append(area)
                ret, rvecs, tvecs = cv2.solvePnP(self.__objp, marker_corner, self.__intrinsic, self.__dist) 
                cv2.drawFrameAxes(frame, self.__intrinsic, self.__dist, rvecs, tvecs, 35 / 2) # debug
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
                screwdriver_ori_euler = R.from_matrix(scrwedriver_ori_mat).as_euler('XYZ', degrees=True)
                # print(markerIds[i].item(), screwdriver_ori_euler)
                root_coors.append(root_coor)
                screwdriver_ori_eulers.append(screwdriver_ori_euler)
            areas = np.array(areas)
            # print(areas)
            # print(root_coors)
            # print(screwdriver_ori_eulers)
            areas = areas / areas.sum()
            areas = areas[:, None]
            # root_coor is weighted average of the root coor of the screwdriver
            root_coor = (np.array(root_coors) * areas).sum(axis=0)

            # root_coor += np.array([-5, -15, 0])
            # root_coor += np.array([-5, -15, 2])

            root_coor += np.array([7, -3, 7])


            screwdriver_ori_euler = (np.array(screwdriver_ori_eulers) * areas).sum(axis=0)
            screwdriver_ori_euler = screwdriver_ori_euler / 180 * np.pi # change to radian
            screwdriver_ori_euler += np.array([0, -.05, 0])
            cv2.imshow('frame', frame) # debug
            return root_coor, screwdriver_ori_euler
        else:
            if markerCorners is None or markerIds is None:
                return None, None
            print(f"No readings from camera, {len(markerCorners), markerIds.max()}.")
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

class HardwareEnv:
    def __init__(self, default_pos, num_repeat=1, gradual_control=False, finger_list=['index', 'middle', 'ring', 'thumb'], kp=4, obj='valve', ori_only=True, mode='relative', device='cuda:0'):
        self.__all_finger_list = ['index', 'middle', 'ring', 'thumb']
        self.obj = obj
        self.__finger_list = finger_list
        self.__ros_node = RosNode(kp=kp, num_repeat=num_repeat, gradual_control=gradual_control)

        self.obj_reader = ObjectPoseReader(obj=obj, mode=mode)

        self.device = device
        self.default_dof_pos = default_pos.clone()
        self.ori_only = ori_only
    
    def get_state(self):
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
            q.append(ori)
        elif self.obj == 'screwdriver':
            pos, ori = self.obj_reader.get_state()
            pos = torch.tensor(pos).float().to(self.device)
            ori = torch.tensor(ori).float().to(self.device)
            # ori = ori * 0 # debug
            if self.ori_only:
                q.append(ori)
                q.append(torch.zeros(1).float().to(self.device)) # add the screwdriver cap angle
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
        return self.__ros_node.apply_action(self.default_dof_pos.squeeze(0))
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
        