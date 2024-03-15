import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

charuco_board = cv2.aruco.CharucoBoard((10, 7), 15, 11, aruco_dict)
image = charuco_board.generateImage((800, 600))

#cv2.imwrite("test.png", image)

charuco_param = cv2.aruco.CharucoParameters()
detector = cv2.aruco.CharucoDetector(charuco_board, charuco_param,)

cameraMatrix = np.load("CameraMatrix.npy")
distortionCoef = np.load("Distortion.npy")

cap = cv2.VideoCapture(0)

while True:
    rect, frame = cap.read()
    if not rect:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, rejected_corners, rejected_ids = detector.detectBoard(gray_frame)

    if charuco_ids is not None:
        frame = cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners)

        obj_points = charuco_board.getChessboardCorners()
        charuco_corners = np.reshape(charuco_corners, (len(charuco_corners), 2))
        if len(obj_points) == len(charuco_corners):
            retval, rvec, tvec = cv2.solvePnP(obj_points, charuco_corners, cameraMatrix, distortionCoef)


    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

    #if len(aruco_ids) > 0:
        #charuco_retval, charuco_corners, charuco_ids = cv2.aruco.estimate()
    
cap.release()
cv2.destroyAllWindows()