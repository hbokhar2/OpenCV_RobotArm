import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(1)  # Change to the appropriate index if necessary

# Previous center to track the object's position
previous_center = None

# Threshold for movement sensitivity (distance between previous and current center)
movement_threshold = 30

# Variable to store the last detected center
last_detected_center = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Thresholding: dark objects (nut) will have low intensity values
    _, binary = cv2.threshold(blurred_frame, 100, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours of detected objects
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store possible centers
    possible_centers = []

    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.04 * cv2.arcLength(contour, True)  # Adjust epsilon for accuracy
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Only consider hexagonal shapes (6 vertices)
        if len(approx) == 6:
            # Draw the contour (hexagon)
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

            # Compute moments to get the center of the hexagon
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])  # X-coordinate of the center
                cy = int(M["m01"] / M["m00"])  # Y-coordinate of the center

                # Add to list of possible centers
                possible_centers.append((cx, cy))

    # If we have possible centers, apply motion filtering
    if possible_centers:
        # Initialize the best center to track the stable object
        best_center = None
        min_distance = float('inf')

        # Check the movement of each detected center and pick the most stable one
        for center in possible_centers:
            # If previous center exists, calculate the movement
            if previous_center is not None:
                distance = np.linalg.norm(np.array(center) - np.array(previous_center))
            else:
                distance = 0  # For the first frame, there's no movement

            # Compare movement, the less the movement, the more stable it is
            if distance < min_distance and distance < movement_threshold:
                best_center = center
                min_distance = distance

        # If a stable center is found, update and use it
        if best_center is not None:
            previous_center = best_center
            cx, cy = best_center

            # Draw the center point
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Store the center value for further processing
            last_detected_center = (cx, cy)

            # Display the center text just above the center point, with small font
            font_scale = 0.4  # Smaller font size
            text = f"Center: ({cx}, {cy})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            text_x = cx - text_size[0] // 2  # Center the text horizontally
            text_y = cy - 10  # Place text above the dot

            # Ensure text stays within frame boundaries
            text_x = max(0, min(text_x, frame.shape[1] - text_size[0]))
            text_y = max(0, min(text_y, frame.shape[0] - 10))

            # Draw the text on the frame
            cv2.putText(frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1)

    # Display the resulting frame
    cv2.imshow("Detected Hex Nut", frame)

    # Break the loop on pressing 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# After the loop ends, print the last detected center
print("Last Detected Center:", last_detected_center)

cap.release()
cv2.destroyAllWindows()
