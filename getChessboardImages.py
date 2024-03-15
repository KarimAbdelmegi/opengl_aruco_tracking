import cv2
import os


cap = cv2.VideoCapture(0)

num = 0

os.chdir('ChessboardImages')

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('Chessboard' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img 1',img)