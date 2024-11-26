import cv2 as cv
import numpy as np
import os

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Capture failed")
        break

    red = frame[:, :, 2]
    green = frame[:, :, 1]
    blue = frame[:, :, 0]

    red_only = np.int16(red) - np.int16(green) - np.int16(blue)
    red_only[red_only < 0] = 0
    red_only[red_only > 255] = 255
    red_only = np.uint8(red_only)

    blue_only = np.int16(blue) - np.int16(red) - np.int16(green)
    blue_only[blue_only < 0] = 0
    blue_only[blue_only > 255] = 255
    blue_only = np.uint8(blue_only)

    green_only = np.int16(green) - np.int16(red) - np.int16(blue)
    green_only[green_only < 0] = 0
    green_only[green_only > 255] = 255
    green_only = np.uint8(green_only)

    cv.imshow('Bokhari-RGB', frame)
    cv.imshow('Bokhari-Red Layer', red)
    cv.imshow('Bokhari-Green Layer', green)
    cv.imshow('Bokhari-Blue Layer', blue)
    cv.imshow('Bokhari-Red Only', red_only)
    cv.imshow('Bokhari-Green Only', green_only)
    cv.imshow('Bokhari-Blue Only', blue_only)

    k = cv.waitKey(5)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
