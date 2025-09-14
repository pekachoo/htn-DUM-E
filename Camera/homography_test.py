import cv2
import numpy as np

# --- CAMERA INTRINSICS (replace with your calibration) ---
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float32)
# Use 4, 5, 8, or 14 coeffs depending on your model (k1,k2,p1,p2[,k3,...])
dist = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# Destination rectangle on the plane (your choice)
dst_points = np.array([
    [0,   0],
    [400, 0],
    [400, 400],
    [0,   400]
], dtype=np.float32)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    raise SystemExit("Camera not available")

h0, w0 = frame.shape[:2]

# Rectification (choose alpha in [0..1]: 0 = minimal black borders)
K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w0, h0), alpha=0.0)
map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K_new, (w0, h0), cv2.CV_16SC2)

# ---- If you already clicked 4 src points on the *distorted* image: undistort them ----
src_points_dist = np.array([
    [600,  78],   # top-left
    [1421, 80],   # top-right
    [1622, 823],  # bottom-right
    [314,  821],  # bottom-left
], dtype=np.float32)

# Get their locations in the rectified (undistorted) pixel space
undist_src = cv2.undistortPoints(src_points_dist.reshape(-1,1,2), K, dist, P=K_new).reshape(-1,2).astype(np.float32)

# Homography from rectified image -> plane coords
H = cv2.getPerspectiveTransform(undist_src, dst_points)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) Undistort/rectify frame
    rectified = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    # 2) Bird’s-eye warp from rectified image
    warped = cv2.warpPerspective(rectified, H, (400, 400))

    # Side-by-side view (match heights)
    h = 400
    w_rect = int(rectified.shape[1] * (h / rectified.shape[0]))
    rect_small = cv2.resize(rectified, (w_rect, h))
    both = np.hstack((rect_small, warped))

    cv2.imshow("Rectified (left) | Bird's-eye (right)", both)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
