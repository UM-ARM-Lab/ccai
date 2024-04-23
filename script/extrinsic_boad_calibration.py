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
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    charuco_params = cv2.aruco.CharucoParameters()
    # detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    detector_params = cv2.aruco.DetectorParameters()
    charuco_board = cv2.aruco.CharucoBoard((5,7), 30, 15, aruco_dict)
    detector = cv2.aruco.CharucoDetector(charuco_board, charuco_params, detector_params)
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        charuco_corners, charuco_Ids, marker_corners, marker_ids = detector.detectBoard(frame)
        if charuco_corners is not None and charuco_corners.shape[0] > 5:
            obj_points, img_points = charuco_board.matchImagePoints(charuco_corners, charuco_Ids)
            ret, rvecs, tvecs  = cv2.solvePnP(obj_points, img_points, intrinsic, dist)
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_Ids)
            cv2.drawFrameAxes(frame, intrinsic, dist, rvecs, tvecs, 30)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            np.save('./rvecs.npy', rvecs)
            np.save('./tvecs.npy', tvecs)
            break
    cap.release()
    cv2.destroyAllWindows()
