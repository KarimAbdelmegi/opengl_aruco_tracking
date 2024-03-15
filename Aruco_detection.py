import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

aruco_param = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_param)

cap = cv2.VideoCapture(0)

while True:
    rect, frame = cap.read()
    if not rect:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_corners, aruco_ids, reject = detector.detectMarkers(frame)

    if np.shape(aruco_corners) == (1, 1, 4, 2):
        for ids, corners in zip(aruco_ids, aruco_corners):
            corners = np.array(corners).astype(int)
            cv2.polylines(frame, [corners], True, (0, 255, 255), thickness=20)
            temp = aruco_corners
        if np.shape(temp) == (1, 1, 4, 2):
            aruco_corners = np.reshape(aruco_corners, (4,2))
            print(aruco_corners[0])

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()