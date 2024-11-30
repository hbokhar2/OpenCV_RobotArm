import cv2 as cv
import numpy as np

cap = cv.VideoCapture(1)

# Initialize tracker (Mean Shift or KLT)
tracker = cv.TrackerMIL_create()

# Function to perform noise reduction and object detection
def image_processing(frame):
    global hsv_feed
    hsv_feed = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define HSV range for grey color (silver/gray detection)
    lower_grey = np.array([0, 0, 150])   # Low saturation, high brightness for grey
    upper_grey = np.array([180, 50, 255])  # Upper threshold

    # Create a mask based on HSV range
    mask = cv.inRange(hsv_feed, lower_grey, upper_grey)
    grey_filtered = cv.bitwise_and(frame, frame, mask=mask)

    # Convert the filtered image to grayscale and apply Gaussian blur
    grayscale_feed = cv.cvtColor(grey_filtered, cv.COLOR_BGR2GRAY)
    blurred_feed = cv.GaussianBlur(grayscale_feed, (5, 5), 0)

    # Detect edges using Canny edge detection
    edges = cv.Canny(blurred_feed, 50, 150)
    return edges, mask

# Function to filter and process contours
def detect_nut(frame, edges, mask):
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours to detect objects
    for contour, hier in zip(contours, hierarchy[0]):
        # Skip small contours (noise)
        if cv.contourArea(contour) < 200:  # Increase this threshold as needed
            continue

        # Approximate the contour to a polygon to check for a hexagon (6 sides)
        epsilon = 0.04 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        if len(approx) == 6:  # Check if the contour is a hexagon
            # Calculate the center of the hexagon using moments
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw the center of the hexagon on the image
                cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Red center

            # Draw the hexagon contour on the image
            cv.drawContours(frame, [approx], 0, (0, 255, 0), 2)  # Green hexagon

            # Return the bounding box for tracking
            x, y, w, h = cv.boundingRect(contour)
            return (x, y, w, h)  # Return bounding box

    return None

# Function to initialize tracking
def init_tracking(frame):
    ret, bbox = cv.selectROI("Select Nut", frame, fromCenter=False, showCrosshair=True)
    if ret:
        tracker.init(frame, bbox)
    return bbox

while True:
    ret, frame = cap.read()

    if not ret:
        print('Failed to capture camera.')
        break

    if 'bbox' in locals():
        # If the tracker is initialized, update it
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            # Draw the tracked object rectangle
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate the center of the object
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)

            # Draw the center point on the image
            cv.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red center point

            # Optionally, display the coordinates of the center
            cv.putText(frame, f'Center: ({center_x}, {center_y})', (center_x + 10, center_y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Smooth the object's position with exponential smoothing
        if success:
            smoothed_x = int(center_x * 0.7 + previous_x * 0.3)
            smoothed_y = int(center_y * 0.7 + previous_y * 0.3)
            previous_x, previous_y = smoothed_x, smoothed_y
            cv.circle(frame, (smoothed_x, smoothed_y), 5, (0, 0, 255), -1)
        
    else:
        # If tracking hasn't been initialized, find the nut and start tracking
        edges, mask = image_processing(frame)
        bbox = detect_nut(frame, edges, mask)
        
        if bbox is not None:
            # Start tracking after the first detection
            tracker.init(frame, bbox)
            previous_x, previous_y = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2

    # Display the frames
    cv.imshow('HSV Feed', hsv_feed)
    cv.imshow('Bokhari - Camera (Original)', frame)
    cv.imshow('Bokhari - Edge Detection', edges)

    # Exit on pressing 'ESC'
    k = cv.waitKey(1)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
