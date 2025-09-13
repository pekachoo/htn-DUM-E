#!/usr/bin/env python3
"""
Complete Pipeline: Homography + Object Detection
1. Camera input -> Homography transformation (bird's eye view)
2. Transformed image -> Object detection with bounding boxes
"""

import cv2
import numpy as np
import argparse
import os
from typing import List, Tuple, Optional

class CompletePipeline:
    def __init__(self, src_points=None, dst_points=None):
        """
        Initialize the complete pipeline
        
        Args:
            src_points: Source points for homography (4 corners of the plane)
            dst_points: Destination points for homography (rectangle corners)
        """
        # Default homography points (you can modify these)
        self.src_points = src_points or np.array([
            [478, 602],   # top-left
            [731, 264],   # top-right
            [1344, 290],  # bottom-right
            [1471, 640],  # bottom-left
        ], dtype=np.float32)
        
        self.dst_points = dst_points or np.array([
            [0, 0],
            [400, 0],
            [400, 400],
            [0, 400]
        ], dtype=np.float32)
        
        # Calculate homography matrix
        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        
        # Detection parameters
        self.detection_enabled = True
        
    def detect_objects(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """Detect objects using edge detection"""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            if area > 1000:  # Lower area threshold
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Basic size filtering
                if w > 30 and h > 30:  # Lower size threshold
                    confidence = min(1.0, area / 50000)  # Lower confidence threshold
                    detections.append((x, y, x+w, y+h, confidence, "Object"))
        
        return detections
    
    def transform_coordinates_to_original(self, detections: List[Tuple[int, int, int, int, float, str]], 
                                        original_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int, float, str]]:
        """Transform detection coordinates back to original image space"""
        if not detections:
            return []
        
        # Calculate inverse homography
        H_inv = np.linalg.inv(self.H)
        
        transformed_detections = []
        for x1, y1, x2, y2, confidence, label in detections:
            # Transform corners back to original space
            corners = np.array([
                [[x1, y1]],
                [[x2, y2]]
            ], dtype=np.float32)
            
            # Apply inverse transformation
            transformed_corners = cv2.perspectiveTransform(corners, H_inv)
            
            # Extract transformed coordinates
            tx1, ty1 = transformed_corners[0][0]
            tx2, ty2 = transformed_corners[1][0]
            
            # Ensure proper ordering
            tx1, tx2 = min(tx1, tx2), max(tx1, tx2)
            ty1, ty2 = min(ty1, ty2), max(ty1, ty2)
            
            # Clamp to image boundaries
            tx1 = max(0, min(tx1, original_shape[1]))
            ty1 = max(0, min(ty1, original_shape[0]))
            tx2 = max(0, min(tx2, original_shape[1]))
            ty2 = max(0, min(ty2, original_shape[0]))
            
            transformed_detections.append((int(tx1), int(ty1), int(tx2), int(ty2), confidence, label))
        
        return transformed_detections
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
        """
        Process a single frame through the complete pipeline
        
        Returns:
            original_with_boxes: Original frame with bounding boxes
            warped: Bird's eye view frame
            detections: List of (object_number, center_x, center_y) tuples
        """
        # Step 1: Apply homography transformation
        warped = cv2.warpPerspective(frame, self.H, (400, 400))
        
        # Step 2: Detect objects in the warped image
        detections = self.detect_objects(warped)
        
        # Step 3: Transform coordinates back to original image space
        original_detections = self.transform_coordinates_to_original(detections, frame.shape[:2])
        
        # Step 4: Convert to simplified format (object_number, center_x, center_y)
        simplified_detections = []
        for i, (x1, y1, x2, y2, confidence, label) in enumerate(original_detections):
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            simplified_detections.append((i + 1, center_x, center_y))
        
        # Step 5: Draw bounding boxes on original image
        original_with_boxes = frame.copy()
        for i, (x1, y1, x2, y2, confidence, label) in enumerate(original_detections):
            cv2.rectangle(original_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_with_boxes, f"Object {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Step 6: Draw bounding boxes on warped image
        warped_with_boxes = warped.copy()
        for i, (x1, y1, x2, y2, confidence, label) in enumerate(detections):
            cv2.rectangle(warped_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(warped_with_boxes, f"Object {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return original_with_boxes, warped_with_boxes, simplified_detections
    
    def get_detections_only(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Get only the detection results without visual processing
        
        Returns:
            List of (object_number, center_x, center_y) tuples
        """
        # Step 1: Apply homography transformation
        warped = cv2.warpPerspective(frame, self.H, (400, 400))
        
        # Step 2: Detect objects in the warped image
        detections = self.detect_objects(warped)
        
        # Step 3: Transform coordinates back to original image space
        original_detections = self.transform_coordinates_to_original(detections, frame.shape[:2])
        
        # Step 4: Convert to simplified format (object_number, center_x, center_y)
        simplified_detections = []
        for i, (x1, y1, x2, y2, confidence, label) in enumerate(original_detections):
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            simplified_detections.append((i + 1, center_x, center_y))
        
        return simplified_detections
    
    def run_pipeline(self, camera_id: int = 0):
        """Run the complete pipeline"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Complete Pipeline: Homography + Object Detection")
        print("- Left: Original view with bounding boxes")
        print("- Right: Bird's eye view with bounding boxes")
        print("- 'q': Quit")
        print("- 's': Save current results")
        print("- 'c': Calibrate homography points")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame through pipeline
            original_with_boxes, warped_with_boxes, detections = self.process_frame(frame)
            
            # Resize frames to fit side by side
            h = 400
            scale = h / frame.shape[0]
            original_resized = cv2.resize(original_with_boxes, None, fx=scale, fy=scale)
            
            # Combine frames side by side
            combined = np.hstack((original_resized, warped_with_boxes))
            
            # Add status text
            cv2.putText(combined, f"Detected: {len(detections)} objects", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, "Original (left) | Bird's-eye (right)", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Print object coordinates
            if detections:
                print("Detected objects:")
                for obj_num, center_x, center_y in detections:
                    print(f"  Object {obj_num}: center at ({center_x}, {center_y})")
            
            # Display combined frame
            cv2.imshow("Complete Pipeline", combined)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_results(original_with_boxes, warped_with_boxes, detections)
            elif key == ord('c'):
                self.calibrate_homography(frame)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def calibrate_homography(self, frame: np.ndarray):
        """Interactive homography calibration"""
        print("Homography Calibration Mode")
        print("Click 4 points in order: top-left, top-right, bottom-right, bottom-left")
        print("Press 'Enter' when done, 'Esc' to cancel")
        
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"{len(points)}", (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("Calibration", frame)
        
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_callback)
        
        while True:
            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                if len(points) == 4:
                    self.src_points = np.array(points, dtype=np.float32)
                    self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
                    print("Homography calibrated successfully!")
                    break
                else:
                    print("Please select exactly 4 points")
            elif key == 27:  # Esc
                print("Calibration cancelled")
                break
        
        cv2.destroyWindow("Calibration")
    
    def save_results(self, original: np.ndarray, warped: np.ndarray, detections: List[Tuple[int, int, int]]):
        """Save pipeline results"""
        os.makedirs("pipeline_output", exist_ok=True)
        
        # Save images
        cv2.imwrite("pipeline_output/original_with_boxes.jpg", original)
        cv2.imwrite("pipeline_output/birdseye_with_boxes.jpg", warped)
        
        # Save detection data
        with open("pipeline_output/detections.txt", "w") as f:
            f.write("Object Detections (object_number, center_x, center_y):\n")
            for obj_num, center_x, center_y in detections:
                f.write(f"Object {obj_num}: center at ({center_x}, {center_y})\n")
        
        print(f"Results saved to pipeline_output/ - {len(detections)} objects detected")

def main():
    parser = argparse.ArgumentParser(description="Complete Pipeline: Homography + Object Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--calibrate", action="store_true", help="Start in calibration mode")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CompletePipeline()
    
    if args.calibrate:
        # Run calibration first
        cap = cv2.VideoCapture(args.camera)
        ret, frame = cap.read()
        if ret:
            pipeline.calibrate_homography(frame)
        cap.release()
    
    # Run the complete pipeline
    pipeline.run_pipeline(args.camera)

if __name__ == "__main__":
    main()
