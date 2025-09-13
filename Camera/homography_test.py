import cv2
import numpy as np

# Pick 4 points on the plane (replace with your own)
src_points = np.array([
    [600, 78],   # top-left
    [1421, 80],   # top-right
    [1622, 823],    # bottom-right 
    [314, 821],   # bottom-left
], dtype=np.float32)

# Destination rectangle
dst_points = np.array([
    [0, 0],
    [400, 0],
    [400, 400],
    [0, 400]
], dtype=np.float32)

H = cv2.getPerspectiveTransform(src_points, dst_points)

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    warped = cv2.warpPerspective(frame, H, (400, 400))

    h = warped.shape[0]
    scale = h / frame.shape[0]
    frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)

    both = np.hstack((frame_resized, warped))
    cv2.imshow("Original (left) | Bird's-eye (right)", both)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
