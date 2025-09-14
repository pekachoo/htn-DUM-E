import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict, Any
from homography_cv import CompletePipeline


class DetectionCapture:
    def __init__(self, src_points=None, dst_points=None):
        """
        Initialize the detection capture system with homography pipeline
        """
        self.pipeline = CompletePipeline(src_points, dst_points)
        self.camera = None

    def initialize_camera(self, camera_id: int = 0) -> bool:
        """
        Initialize camera and return True if successful
        """
        try:
            self.camera = cv2.VideoCapture(camera_id)
            if not self.camera.isOpened():
                # Try alternative camera IDs
                for alt_id in [1, 2, 0]:
                    self.camera = cv2.VideoCapture(alt_id)
                    if self.camera.isOpened():
                        ret, test_frame = self.camera.read()
                        if ret:
                            break
                        else:
                            self.camera.release()
                            self.camera = None
            return self.camera is not None and self.camera.isOpened()
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False

    def capture_frame_with_detection(
        self,
    ) -> Tuple[Optional[np.ndarray], List[Tuple[int, float, float]]]:
        """
        Capture a single frame and return both the warped image with detections and coordinates
        Returns: (warped_image_with_boxes, detections_list)
        """
        if self.camera is None or not self.camera.isOpened():
            print("Camera not initialized or not available")
            return None, []

        ret, frame = self.camera.read()
        if not ret:
            print("Failed to capture frame")
            return None, []

        # Apply homography transformation
        warped = cv2.warpPerspective(frame, self.pipeline.H, (400, 400))
        warped_with_boxes = warped.copy()

        # Draw detections on the warped image
        self.pipeline.draw_automatic_detections(warped, warped_with_boxes)

        # Get detection coordinates
        detections = self.pipeline.get_detections(frame)

        return warped_with_boxes, detections

    def format_detection_text(self, detections: List[Tuple[int, float, float]]) -> str:
        """
        Format detection coordinates into a readable text string
        """
        if not detections:
            return "No objects detected"

        detection_strings = []
        for obj_num, x, y in detections:
            detection_strings.append(f"Object {obj_num}: ({x:.2f}cm, {y:.2f}cm)")

        return "Detected objects:\n" + "\n".join(detection_strings)

    def capture_and_save(self, output_path: str = "new_img.jpg") -> Dict[str, Any]:
        """
        Main function to capture image with detection and save it
        Returns dictionary with image path, detections, and formatted text
        """
        # Capture frame with detection
        warped_image, detections = self.capture_frame_with_detection()

        if warped_image is None:
            return {
                "success": False,
                "error": "Failed to capture frame",
                "image_path": None,
                "detections": [],
                "formatted_text": "No data available",
            }

        # Save the image
        try:
            cv2.imwrite(output_path, warped_image)
            print(f"Image saved as: {output_path}")
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to save image: {e}",
                "image_path": None,
                "detections": detections,
                "formatted_text": self.format_detection_text(detections),
            }

        # Format the detection text
        formatted_text = self.format_detection_text(detections)

        # Print detections to console (matching original format)
        if detections:
            print(
                "Detections (cm):",
                ", ".join([f"({n},{x:.2f},{y:.2f})" for n, x, y in detections]),
            )
        else:
            print("Detections (cm): []")

        return {
            "success": True,
            "error": None,
            "image_path": output_path,
            "detections": detections,
            "formatted_text": formatted_text,
        }

    def cleanup(self):
        """
        Clean up camera resources
        """
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()


def prompt_user_and_capture(camera_id: int = 0) -> Dict[str, Any]:
    """
    Main function that prompts user and captures detection
    This is the main function that can be called by other files
    """
    print("=" * 50)
    print("Object Detection Capture System")
    print("=" * 50)

    # Initialize capture system
    capture_system = DetectionCapture()

    # Initialize camera
    if not capture_system.initialize_camera(camera_id):
        return {
            "success": False,
            "error": "Failed to initialize camera",
            "image_path": None,
            "detections": [],
            "formatted_text": "Camera initialization failed",
        }

    try:
        # Prompt user
        print("\nCamera initialized successfully!")
        print("Position objects in the detection area...")
        input("Press Enter when ready to capture...")

        # Capture and save
        result = capture_system.capture_and_save()

        if result["success"]:
            print(f"\nCapture successful!")
            print(f"Image saved: {result['image_path']}")
            print(f"📊 {result['formatted_text']}")
        else:
            print(f"\nCapture failed: {result['error']}")

        return result

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        return {
            "success": False,
            "error": "Operation cancelled by user",
            "image_path": None,
            "detections": [],
            "formatted_text": "Operation cancelled",
        }
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return {
            "success": False,
            "error": f"Unexpected error: {e}",
            "image_path": None,
            "detections": [],
            "formatted_text": "Error occurred",
        }
    finally:
        # Clean up
        capture_system.cleanup()


def main():
    """
    Command line interface for the capture system
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Capture object detection with homography"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera ID to use")
    parser.add_argument(
        "--output", type=str, default="new_img.jpg", help="Output image filename"
    )
    args = parser.parse_args()

    # Create capture system
    capture_system = DetectionCapture()

    # Initialize camera
    if not capture_system.initialize_camera(args.camera):
        print("Failed to initialize camera. Exiting.")
        return

    try:
        # Prompt user
        print("\nCamera initialized successfully!")
        print("Position objects in the detection area...")
        input("Press Enter when ready to capture...")

        # Capture and save
        result = capture_system.capture_and_save(args.output)

        if result["success"]:
            print(f"\nCapture successful!")
            print(f"Image saved: {result['image_path']}")
            print(f"📊 {result['formatted_text']}")
        else:
            print(f"\nCapture failed: {result['error']}")

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        # Clean up
        capture_system.cleanup()


if __name__ == "__main__":
    main()
