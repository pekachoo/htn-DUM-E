import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from homography_cv import CompletePipeline


def capture_with_detection(
    camera_id: int = 0, output_path: str = "current_task_image.jpg"
) -> Tuple[Optional[str], List[Tuple[int, float, float]], Optional[Tuple[float, float]]]:
    """
    Simple function to capture image with detection, save it with bounding boxes, and return coordinates
    Returns: (image_path, detections_list, hand_cm) or (None, [], None) if failed
    """
    # Initialize pipeline and camera
    pipeline = CompletePipeline()
    camera = None

    try:
        # Initialize camera
        camera = cv2.VideoCapture(camera_id)
        if not camera.isOpened():
            # Try alternative camera IDs
            for alt_id in [1, 2, 0]:
                camera = cv2.VideoCapture(alt_id)
                if camera.isOpened():
                    ret, test_frame = camera.read()
                    if ret:
                        break
                    else:
                        camera.release()
                        camera = None

        if camera is None or not camera.isOpened():
            print("Failed to initialize camera")
            return None, [], None

        # Capture frame
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame")
            return None, [], None

        # Apply homography transformation
        warped = cv2.warpPerspective(frame, pipeline.H, (400, 400))
        warped_with_boxes = warped.copy()

        # Draw detections on the warped image
        pipeline.draw_automatic_detections(warped, warped_with_boxes)

        # Get detection coordinates using the same method as homography_cv.py
        # Apply homography to get warped frame for detection
        warped_for_detection = cv2.warpPerspective(frame, pipeline.H, (400, 400))
        rects = pipeline._find_filtered_rects(warped_for_detection)
        if isinstance(rects, tuple):
            rects = rects[0]

        detections = []
        for i, (x, y, w, h) in enumerate(rects):
            cx = x + w // 2
            cy = y + h // 2
            # Convert to (x, 400-y), then scale to cm - same as homography_cv.py
            cx_cm = (cx * 30.0) / 400.0
            cy_cm = ((400 - cy) * 30.0) / 400.0
            detections.append((i + 1, cx_cm, cy_cm))  # These are in cm, with y flipped

        # Hand detection (in original frame), then map to cm
        hand_info = pipeline.get_hand_in_cm(frame)
        hand_cm: Optional[Tuple[float, float]] = None
        if hand_info is not None:
            (hx, hy), (wx, wy), (x_cm, y_cm) = hand_info
            hand_cm = (x_cm, y_cm)
            # Draw marker on warped image where the hand maps
            cv2.circle(warped_with_boxes, (int(wx), int(wy)), 8, (255, 0, 0), -1)
            cv2.putText(warped_with_boxes, f"HAND ({x_cm:.1f},{y_cm:.1f})cm", (int(wx)+10, int(wy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        # Save the image with bounding boxes
        cv2.imwrite(output_path, warped_with_boxes)
        print(f"Image saved: {output_path}")

        # Print detections
        if detections:
            print(
                "Detections (cm):",
                ", ".join([f"({n},{x:.2f},{y:.2f})" for n, x, y in detections]),
            )
        else:
            print("Detections (cm): []")

        # Print hand coordinates if present
        if hand_cm is not None:
            print(f"Hand (cm): ({hand_cm[0]:.2f}, {hand_cm[1]:.2f})")
        else:
            print("Hand (cm): None")

        return output_path, detections, hand_cm

    except Exception as e:
        print(f"Error in capture: {e}")
        return None, [], None
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test the function
    image_path, detections = capture_with_detection()
    if image_path:
        print(f"Success! Image: {image_path}, Detections: {detections}")
    else:
        print("Failed to capture")
