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

#Center of object in X coordinate
column_sums = np.matrix(np.sum(difference,0))
column_numbers = np.matrix(np.arange(640))
column_mult = np.multiply(column_sums, column_numbers)
column_total = np.sum(column_mult)
total_total = np.sum(np.sum(difference))
difference_column_location = column_total/total_total

print('Object column ("X") location: ', difference_column_location)

#Center of object in Y coordinate
row_sums = np.matrix(np.sum(difference,1))
row_sums = row_sums.transpose()
row_numbers = np.matrix(np.arange(480))
row_mult = np.multiply(row_sums,row_numbers)
row_total = np.sum(row_mult)
difference_row_location = row_total/total_total

print('Object column ("Y") location: ', difference_row_location)

#Crosshairs
color = (255,255,255)

thickness = 1

diff_x_location = np.uint16(difference_column_location)
diff_y_location = np.uint16(difference_row_location)

difference = cv.line(difference, (diff_x_location,0),(diff_x_location,480),color, thickness)
difference = cv.line(difference, (0, diff_y_location),(640, diff_y_location),color, thickness)

cv.imshow('difference',difference)