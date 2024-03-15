import numpy as np
import cv2
import glob
import os


ChessBoardSize = (8, 5)
frameSize = (1920, 1080)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((ChessBoardSize[0] * ChessBoardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:ChessBoardSize[0],0:ChessBoardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

images = glob.glob('ChessboardImages/*.png')

for image in images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, ChessBoardSize, None)

    if ret == True:

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, ChessBoardSize, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(27)

cv2.destroyAllWindows()

ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


print("Camera Calibrated: ", ret)
print("\nCamera Matrix: \n", cameraMatrix)
print("\nDistortion Parameters: \n", dist)
print("\n Rotation Vectors: \n", rvecs)
print("\n Translation Vectors: \n", tvecs)

np.save('CameraMatrix', cameraMatrix)
np.save('Distortion', dist)


img_test = cv2.imread("ChessboardImages/Chessboard0.png")
h, w = img_test.shape[:2]

newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))


undst = cv2.undistort(img_test, cameraMatrix, dist, None, newCameraMatrix)

x,y,w,h = roi
undst = undst[y:y+h, x:x+w]

cv2.imwrite("UndisChessboardImages/Chessboard0.png", undst)

mean_error = 0

for i in range(len(objpoints)):
    imgPoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv2.norm(imgpoints[i], imgPoints2, cv2.NORM_L2)/len(imgPoints2)

    mean_error += error

print("total error: {}".format(mean_error/len(objpoints)))
