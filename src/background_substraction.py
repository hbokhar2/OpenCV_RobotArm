import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

while(1):
    _,frame = cap.read()

    gray_image = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cv.imshow('background', gray_image)
    k = cv.waitKey(5)
    if k == 27:
        break

while(1):
    _,frame = cap.read()

    gray_image_2 = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cv.imshow('background', gray_image_2)

    difference = np.absolute(np.matrix(np.int16(gray_image)) - np.matrix(np.int16(gray_image_2)))
    difference[difference > 255] = 255

    difference = np.uint8(difference)

    cv.imshow('difference', difference)

    k = cv.waitKey(5)
    if k == 27:
        break