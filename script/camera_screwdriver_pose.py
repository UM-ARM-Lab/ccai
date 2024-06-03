import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
if __name__ == "__main__":
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
    cam2world = global_trans @ cam2world

    screwdriver2marker = []
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
        screwdriver2marker.append(trans)


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
    
    objp = np.array([[-0.5, -0.5, 0.],
                    [-0.5, 0.5, 0.],
                    [0.5, 0.5, 0.],
                    [0.5, -0.5, 0.]]) * 35

    # axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)
    while True:
        ret, frame = cap.read()
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)
        # print(markerCorners)
        if len(markerCorners) > 0:
            cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            for i in range(len(markerCorners)):
                if markerIds[i].item() >= 4:
                    continue

                ret, rvecs, tvecs = cv2.solvePnP(objp, markerCorners[i], intrinsic, dist) 
                cv2.drawFrameAxes(frame, intrinsic, dist, rvecs, tvecs, 35 / 2)
                rotation = R.from_rotvec(rvecs[:,0])
                rotation_mat = rotation.as_matrix()
                translation = tvecs
                marker2cam = np.concatenate((rotation_mat, translation), axis=1)
                marker2cam = np.concatenate((marker2cam, np.array([[0, 0, 0, 1]])), axis=0)
                marker2world = cam2world @ marker2cam
                world_coor = marker2world[:3, 3]
                centroid_direction = marker2world[:3, 2]
                center_coor = world_coor - centroid_direction * 21
                # print(markerIds[i], center_coor)
                trans_mat = screwdriver2marker[markerIds[i].item()]
                trans_mat = marker2world @ trans_mat
                scrwedriver_ori_mat = trans_mat[:3, :3]
                screwdriver_ori_euler = R.from_matrix(scrwedriver_ori_mat).as_euler('XYZ', degrees=True)
                print(markerIds[i].item(), screwdriver_ori_euler)
                # imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, intrinsic, dist) 
                # # print(imgpts[-1, 0]-markerCorners[0][0][0])
                # frame = draw(frame, markerCorners[i][0], imgpts)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
