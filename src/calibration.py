import cv2

# Initialize camera
cap = cv2.VideoCapture(1)  # Change to the appropriate index if necessary

# Variable to store the coordinates of the clicked point
clicked_point = None

# Mouse callback function to capture the pixel coordinates
def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
        clicked_point = (x, y)  # Store the pixel coordinates
        print(f"Pixel Coordinates: ({x}, {y})")  # Print the coordinates

# Set up the window and the callback function
cv2.namedWindow("Camera Feed")
cv2.setMouseCallback("Camera Feed", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the camera feed
    cv2.imshow("Camera Feed", frame)

    # If a point is selected, draw a circle at the selected point
    if clicked_point:
        cv2.circle(frame, clicked_point, 5, (0, 0, 255), -1)  # Red circle

    # Press 'Esc' to exit
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
