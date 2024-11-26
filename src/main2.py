import numpy as np
import cv2

cap = cv2.VideoCapture(0)

threshold = 50

while (1):
    _,frame = cap.read()

    red = frame[:,:,2] # : means all [row, column, layer] third layer
    green = frame[:,:,1]  # : means all [row, column, layer] second layer
    blue = frame[:,:,0]  # : means all [row, column, layer] first layer

    # make value a 16 bit that can be positive or negative
    red_only = np.int16(red) - np.int16(green) - np.int16(blue)

    # numbers to achieve valid matrix ranging from 0 to 255
    red_only[red_only < threshold] = 0 # sets #'s lower than 0 to 0
    red_only[red_only >= threshold] = 255 # sets #'s greater than 255 down to 255


    cv2.imshow('RO-RGB',frame)
    cv2.imshow('RO-Red',red)
    #cv2.imshow('RO-Green',green)
    #cv2.imshow('RO-Blue',blue)

    red_only = np.uint8(red_only)

    cv2.imshow('RO-Red Only', red_only)

    k = cv2.waitKey(5) # number is ms delay to check if
    # the number 27 is the ASCII control character value for the esc key

    if k == 27:
        break

#finding the center of the object for x coordinate
column_sums = np.matrix(np.sum(red_only, 0)) # 0 sums the columns
column_numbers = np.matrix(np.arange(640))
column_mult = np.multiply(column_sums, column_numbers) #element wise multiplication
column_total = np.sum(column_mult)
total_total = np.sum(np.sum(red_only)) #sums all values in the image
red_column_location = column_total/total_total
print('Red column ("X") location: ', red_column_location)

#finding the center of the object for Y coordinate
row_sums = np.matrix(np.sum(red_only, 1))
#row_sums = row_sums.transpose()
row_numbers = np.matrix(np.arange(480))
row_mult = np.multiply(row_sums, row_numbers)
row_total = np.sum(row_mult)
red_row_location = row_total / total_total
print('Red row ("Y") location: ', red_row_location)

red_only = np.uint8(red_only) # ensure red_only is a unsigned interger


cv2.destroyAllWindows() #closes openned python windows
 