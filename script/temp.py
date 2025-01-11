import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
if __name__ == "__main__":
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    cap = cv2.VideoCapture(0)
    # matrix = np.array([[904.5715942382812, 0.000000, 635.9815063476562],
    #                                     [0.000000,905.2954711914062, 353.06036376953125],
    #                                     [0.000000, 0.000000, 1.000000]])
    
    objp = np.array([[-0.5, -0.5, 0.],
                    [-0.5, 0.5, 0.],
                    [0.5, 0.5, 0.],
                    [0.5, -0.5, 0.]]) * 35
    intrinsic = np.array([[621.80984333,   0.,         651.09118583],
    [  0.,         621.66658768, 352.88384525],
    [  0.,           0. ,          1.,        ]])
    dist = np.array([[0.07583722,  0.00042308, -0.00245659 , 0.00797877 , 0.01058895]])

    # axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)
    while True:
        ret, frame = cap.read()
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)
        cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        for i in range(len(markerCorners)):
            ret, rvecs, tvecs = cv2.solvePnP(objp, markerCorners[i], intrinsic, dist) 
            cv2.drawFrameAxes(frame, intrinsic, dist, rvecs, tvecs, 35 / 2)

        # print(markerCorners)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
