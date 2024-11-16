import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

h = 13
w = 18
objp = np.zeros((h*w,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
# objp = np.zeros((5*7,3), np.float32)
# objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')
cap = cv.VideoCapture(0)
while True:
    ret, img = cap.read()

    # for fname in images:
    # img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (w,h), None)
    # If found, add object points, image points (after refining them)
    # cv.imshow('frame', img)
    if cv.waitKey(1) & 0xFF == ord('r'):
        if ret == True:
            print("recorded", len(objpoints))
            objpoints.append(objp)
            # corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            # cv.drawChessboardCorners(img, (w,h), corners2, ret)
            # cv.imshow('img', img)
            # cv.waitKey(1000)
    if corners is not None:
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (w,h), corners2, ret)
    cv.imshow('frame', img)
    if len(objpoints) > 20:
        break
cv.destroyAllWindows()
cap.release()
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx)
print(dist)