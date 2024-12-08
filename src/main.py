import cv2
import numpy as np
import math

# Initialize camera
cap = cv2.VideoCapture(1)  # Change to the appropriate index if necessary

# Define camera's field of view (21.6 mm x 16.5 mm)
fov_width = 21.6  # mm
fov_height = 16.5  # mm

# Image resolution (640x480)
image_width = 640  # pixels
image_height = 480  # pixels

# New camera-origin (344, 8) corresponds to (0 mm, -2 mm) in the real world
camera_origin_x = 341
camera_origin_y = 13
real_origin_offset_y = 2  # Positive 2 mm (real-world y-offset)

# Calculate scaling factors
scaling_factor_x = fov_width / image_width  # mm/pixel
scaling_factor_y = fov_height / image_height  # mm/pixel

# Previous center to track the object's position
previous_center = None
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

    possible_centers = []

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)  # Adjust epsilon for accuracy
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Only consider hexagonal shapes (6 vertices)
        if len(approx) == 6:
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])  # X-coordinate of the center
                cy = int(M["m01"] / M["m00"])  # Y-coordinate of the center
                possible_centers.append((cx, cy))

    # If we have possible centers, apply motion filtering
    if possible_centers:
        best_center = None
        min_distance = float('inf')

        for center in possible_centers:
            if previous_center is not None:
                distance = np.linalg.norm(np.array(center) - np.array(previous_center))
            else:
                distance = 0  # First frame, no movement

            if distance < min_distance and distance < movement_threshold:
                best_center = center
                min_distance = distance

        if best_center is not None:
            previous_center = best_center
            cx, cy = best_center

            # Convert to real-world coordinates (mm) with the real-world y-origin offset
            real_world_x = (cx - camera_origin_x) * scaling_factor_x
            real_world_y = (cy - camera_origin_y) * scaling_factor_y + real_origin_offset_y  # Apply +2 mm offset for y

            # Invert the sign of x-coordinate, but keep y positive
            real_world_x = -real_world_x  # Flip the sign of x
            # No inversion of y; it stays positive

            # Display the real-world coordinates in mm
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Center: ({real_world_x:.2f} mm, {real_world_y:.2f} mm)", 
                        (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Store the center value
            last_detected_center = (real_world_x, real_world_y)

            # Optionally, you can print or log these coordinates
            print(f"Real-world center: ({real_world_x:.2f} mm, {real_world_y:.2f} mm)")

    cv2.imshow("Detected Hex Nut", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# After loop ends, print the last detected center
print("Last Detected Center:", last_detected_center)

cap.release()
cv2.destroyAllWindows()

def inverse_kinematics_2d(x, y, L1=6.2, L2=3.8):
    """
    Calculate the angles for a 2-DOF manipulator arm in a 2D plane.
    
    Args:
        x (float): Target x-coordinate in cm.
        y (float): Target y-coordinate in cm.
        L1 (float): Length of the first link in cm. Default is 6.2 cm.
        L2 (float): Length of the second link in cm. Default is 3.8 cm.
    
    Returns:
        tuple: Angles (theta1, theta2) in degrees.
    """
    # Check if the target is reachable
    distance = math.sqrt(x**2 + y**2)
    if distance > (L1 + L2):
        raise ValueError("Target is out of reach.")

    # Calculate theta2 (elbow angle)
    cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    sin_theta2 = math.sqrt(1 - cos_theta2**2)  # Positive root
    theta2 = math.atan2(sin_theta2, cos_theta2)

    # Calculate theta1 (base angle)
    k1 = L1 + L2 * cos_theta2
    k2 = L2 * sin_theta2
    theta1 = math.atan2(y, x) - math.atan2(k2, k1)

    # Convert radians to degrees
    theta1_deg = math.degrees(theta1)
    theta2_deg = math.degrees(theta2)

    return theta1_deg, theta2_deg

coordinates = inverse_kinematics_2d(real_world_x, real_world_y, 6.2, 3.8)

Theta_1 = coordinates[0]
Theta_2 = coordinates[1]

print(Theta_1)
print(Theta_2)


# Close the serial connection if it was used
# ser.close()  # Commented out for now

# Close the serial connection if it was used
# ser.close()  # Commented out for now
