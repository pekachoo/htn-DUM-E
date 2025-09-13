import cv2
import numpy as np
import argparse
import os
from typing import List, Tuple, Optional

class CompletePipeline:
    def __init__(self, src_points=None, dst_points=None):
        self.src_points = src_points or np.array([
            [600, 78], #topleft
            [1421, 80], #top right
            [1622, 823], # bottom right
            [314, 821], # bottomleft
        ], dtype=np.float32)
        
        self.dst_points = dst_points or np.array([
            [0, 0],
            [400, 0],
            [400, 400],
            [0, 400]
        ], dtype=np.float32)
        
        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        
    def detect_objects_contours(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 30 and h > 30:
                    confidence = min(1.0, area / 50000)
                    detections.append((x, y, x+w, y+h, confidence, "Object"))
        return detections
    
    def transform_coordinates_to_original(self, detections: List[Tuple[int, int, int, int, float, str]], 
                                        original_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int, float, str]]:
        if not detections:
            return []
        
        H_inv = np.linalg.inv(self.H)
        transformed_detections = []
        for x1, y1, x2, y2, confidence, label in detections:
            corners = np.array([
                [[x1, y1]],
                [[x2, y2]]
            ], dtype=np.float32)
            
            transformed_corners = cv2.perspectiveTransform(corners, H_inv)
            
            tx1, ty1 = transformed_corners[0][0]
            tx2, ty2 = transformed_corners[1][0]
            
            tx1, tx2 = min(tx1, tx2), max(tx1, tx2)
            ty1, ty2 = min(ty1, ty2), max(ty1, ty2)
            
            tx1 = max(0, min(tx1, original_shape[1]))
            ty1 = max(0, min(ty1, original_shape[0]))
            tx2 = max(0, min(tx2, original_shape[1]))
            ty2 = max(0, min(ty2, original_shape[0]))
            
            transformed_detections.append((int(tx1), int(ty1), int(tx2), int(ty2), confidence, label))
        
        return transformed_detections
    
    def draw_automatic_detections(self, frame: np.ndarray, result: np.ndarray):
        detections = self.detect_objects_contours(frame)
        for i, (x1, y1, x2, y2, confidence, label) in enumerate(detections):
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result, f"Object {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def get_detections(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        warped = cv2.warpPerspective(frame, self.H, (400, 400))
        detections = self.detect_objects_contours(warped)
        original_detections = self.transform_coordinates_to_original(detections, frame.shape[:2])
        
        simplified_detections = []
        for i, (x1, y1, x2, y2, confidence, label) in enumerate(original_detections):
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            simplified_detections.append((i + 1, center_x, center_y))
        
        return simplified_detections
    
    def run_pipeline(self, camera_id: int = 0):
        cap = None
        for cam_id in [camera_id, 1, 2, 0]:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret:
                    break
                else:
                    cap.release()
                    cap = None
            else:
                cap.release()
                cap = None
        
        if cap is None or not cap.isOpened():
            self.run_test_mode()
            return
        
        frame_count = 0
        consecutive_failures = 0
        max_failures = 10
        
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                if consecutive_failures % 5 == 0:
                    cap.release()
                    cap = cv2.VideoCapture(camera_id)
                    if not cap.isOpened():
                        break
                continue
            
            consecutive_failures = 0
            frame_count += 1
            
            detections = self.get_detections(frame)
            
            warped = cv2.warpPerspective(frame, self.H, (400, 400))
            warped_with_boxes = warped.copy()
            self.draw_automatic_detections(warped, warped_with_boxes)
            
            h = 400
            scale = h / frame.shape[0]
            original_resized = cv2.resize(frame, None, fx=scale, fy=scale)
            combined = np.hstack((original_resized, warped_with_boxes))
            
            cv2.imshow("Complete Pipeline", combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_test_mode(self):
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), -1)
        cv2.circle(test_image, (400, 150), 50, (255, 0, 0), -1)
        cv2.rectangle(test_image, (300, 300), (450, 400), (0, 0, 255), -1)
        
        detections = self.get_detections(test_image)
        
        h = 400
        scale = h / test_image.shape[0]
        original_resized = cv2.resize(test_image, None, fx=scale, fy=scale)
        warped = cv2.warpPerspective(test_image, self.H, (400, 400))
        combined = np.hstack((original_resized, warped))
        
        cv2.imshow("Test Mode - Complete Pipeline", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Complete Pipeline: Homography + Object Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    args = parser.parse_args()
    
    pipeline = CompletePipeline()
    pipeline.run_pipeline(args.camera)

if __name__ == "__main__":
    main()