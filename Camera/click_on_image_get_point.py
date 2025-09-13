import cv2

def get_external_camera_index():
    """
    Returns the index of the first external camera it can open.
    Assumes built-in cam is index 0; tries higher indexes.
    """
    for i in range(0, 10):  # start from 1 to skip the built-in cam
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    raise RuntimeError("No external camera found")

# Click callback
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at (x={x}, y={y})")

def main():
    cam_idx = 0 #get_external_camera_index()
    print(f"Using camera index: {cam_idx}")

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("Could not open external camera.")
        return

    cv2.namedWindow("External Camera")
    cv2.setMouseCallback("External Camera", click_event)

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("External Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
