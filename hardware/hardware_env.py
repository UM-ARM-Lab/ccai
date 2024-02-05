import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from allegro_ros import Ros_Node
import torch
import rospy

class HarwareEnv:
    def __init__(self, finger_list=['index', 'middle', 'ring', 'thumb']):
        self.__finger_list = finger_list
        self.__ros_node = Ros_Node()
        self.__intrinsic = np.array([[742.76562131, 0., 627.8765477],
                          [0.,734.42057297, 310.25507405],
                          [0., 0.,1.]])
        self.__dist = np.array([[ 3.28057832e-01, -5.85426548e+00, -1.64381997e-02, -6.97016057e-03, 5.58445122e+01]])
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        self.__detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        self.__cap = cv2.VideoCapture(0)
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
            current_pose = self._get_valve_rot_vec()
            if current_pose is not None:
                break
        rot_diff = current_pose * self.__initial_pose.inv()
        diff_vec = rot_diff.as_rotvec()
        angle = np.linalg.norm(diff_vec) * np.sign(diff_vec[2])
        return angle
    def get_state(self):
        robot_state = self.__ros_node.allegro_joint_pos
        index, mid, ring, thumb = torch.chunk(robot_state, chunks=4, dim=-1)
        state = {}
        state['index'] = index
        state['middle'] = mid
        state['ring'] = ring
        state['thumb'] = thumb
        q = []
        for finger_name in self.__finger_list:
            q.append(state[finger_name])
        valve_angle = self.get_valve_angle()
        q.append(torch.tensor([valve_angle]))
        state['q'] = torch.cat(q)
        state['theta'] = torch.tensor([valve_angle])
        return state
    def step(self, action):
        self.__ros_node.apply_action(action)
        return self.get_state()

if __name__ == "__main__":
    env = HarwareEnv(finger_list=['index', 'thumb'])
    while True:
        action = torch.randn(16) / 3
        state = env.step(action)
        print(state)
        rospy.sleep(1)
