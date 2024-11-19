import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)
    # img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 5)
    return img
if __name__ == "__main__":
    intrinsic = np.array([[621.80984333,   0.,         651.09118583],
    [  0.,         621.66658768, 352.88384525],
    [  0.,           0. ,          1.,        ]])
    dist = np.array([[0.07583722,  0.00042308, -0.00245659 , 0.00797877 , 0.01058895]])
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    cap = cv2.VideoCapture(0)
    # matrix = np.array([[904.5715942382812, 0.000000, 635.9815063476562],
    #                                     [0.000000,905.2954711914062, 353.06036376953125],
    #                                     [0.000000, 0.000000, 1.000000]])
    
    # objp = np.array([[0., 0., 0.],
    #                 [1., 0., 0.],
    #                 [1., 1., 0.],
    #                 [0., 1., 0.]])
    objp = np.array([[0, -0.5, 0.5],
                    [0, -0.5, -0.5],
                    [0, 0.5, -0.5],
                    [0.0, 0.5, 0.5]])

    axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)
    ctr = 0
    while True:
        ret, frame = cap.read()
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)
        # print(markerCorners)
        if len(markerCorners) > 0:
            cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            for i in range(len(markerCorners)):

                ret, rvecs, tvecs = cv2.solvePnP(objp, markerCorners[i], intrinsic, dist) 
                # if ctr == 0:
                #     initial_r = rvecs
                #     initial_pose = R.from_rotvec(initial_r[:,0]).inv()
                # else:
                #     current_pose = R.from_rotvec(rvecs[:,0]).inv()
                #     rot_diff = current_pose * initial_pose.inv()
                #     diff_vec = rot_diff.as_rotvec()
                #     angle = np.linalg.norm(diff_vec) * np.sign(diff_vec[2])
                #     print(angle)
                cv2.drawFrameAxes(frame, intrinsic, dist, rvecs, tvecs, 0.5)
                # imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, intrinsic, dist) 
                # # print(imgpts[-1, 0]-markerCorners[0][0][0])
                # frame = draw(frame, markerCorners[i][0], imgpts)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ctr += 1
    cap.release()
    cv2.destroyAllWindows()
