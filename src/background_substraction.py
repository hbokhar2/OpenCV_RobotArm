import numpy as np
import cv2

cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_SIMPLEX  # Font for overlay text

# Define HSV ranges for red, green, and blue
lower_red_1 = np.array([0, 50, 50])   # First range for red (hue ~ 0-10)
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([170, 50, 50]) # Second range for red (hue ~ 170-180)
upper_red_2 = np.array([180, 255, 255])

lower_green = np.array([35, 50, 50])  # Range for green (hue ~ 35-85)
upper_green = np.array([85, 255, 255])

lower_blue = np.array([100, 50, 50])  # Range for blue (hue ~ 100-140)
upper_blue = np.array([140, 255, 255])

while True:
    _, frame = cap.read()

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Isolate red shades
    mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    red_only = cv2.bitwise_or(mask_red_1, mask_red_2)

    # Isolate green shades
    green_only = cv2.inRange(hsv, lower_green, upper_green)

    # Isolate blue shades
    blue_only = cv2.inRange(hsv, lower_blue, upper_blue)

    # Add text to the main frame
    cv2.putText(frame, "Bokhari - Lab08", (10, 30), font, 1, (255, 255, 255), 2)

    # Add text to each isolated color frame
    cv2.putText(red_only, "Bokhari - Lab08", (10, 30), font, 1, 255, 2)
    cv2.putText(green_only, "Bokhari - Lab08", (10, 30), font, 1, 255, 2)
    cv2.putText(blue_only, "Bokhari - Lab08", (10, 30), font, 1, 255, 2)

    # Display the results
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Red Only', red_only)
    cv2.imshow('Green Only', green_only)
    cv2.imshow('Blue Only', blue_only)

    # Stop processing when ESC is pressed
    k = cv2.waitKey(5)
    if k == 27:  # ESC key
        break

# Destroy all windows when exiting
cv2.destroyAllWindows()
