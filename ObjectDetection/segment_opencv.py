#!/usr/bin/env python3
"""
Lightweight Live Image Segmentation using OpenCV
Real-time object segmentation with minimal dependencies
"""

import cv2
import numpy as np
import argparse
import os
from typing import List, Tuple, Optional

class LightweightSegmentation:
    def __init__(self):
        """Initialize the lightweight segmentation system"""
        # Segmentation state
        self.points = []
        self.mask = None
        self.current_mask = None
        self.mask_mode = "add"  # "add" or "remove"
        
        # UI state
        self.drawing = False
        self.last_point = None
        self.brush_size = 20
        
        # Bounding box state
        self.bounding_boxes = []
        self.current_bbox = None
        self.bbox_mode = False
        self.bbox_start = None
        self.bbox_end = None
        
        # Segmentation parameters
        self.grabcut_iterations = 5
        self.use_grabcut = True
        
        # Object detection parameters
        self.detection_enabled = True  # Always on
        self.detection_confidence = 0.5
        self.detection_threshold = 0.3
        
        # Filtering parameters
        self.detection_history = []
        self.max_history = 5
        self.min_detections = 2  # Minimum detections to show box
        self.stability_threshold = 0.3  # Max movement between frames
        
        # Background for subtraction
        self.background = None
        self.background_captured = False
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for interactive segmentation"""
        if self.bbox_mode:
            # Bounding box mode
            if event == cv2.EVENT_LBUTTONDOWN:
                self.bbox_start = (x, y)
                self.bbox_end = (x, y)
                print(f"Started bounding box at ({x}, {y})")
                
            elif event == cv2.EVENT_MOUSEMOVE and self.bbox_start is not None:
                self.bbox_end = (x, y)
                
            elif event == cv2.EVENT_LBUTTONUP and self.bbox_start is not None:
                # Finalize bounding box
                x1, y1 = self.bbox_start
                x2, y2 = self.bbox_end
                # Ensure proper rectangle
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:  # Minimum size
                    self.bounding_boxes.append((x1, y1, x2, y2))
                    print(f"Added bounding box: ({x1}, {y1}, {x2}, {y2})")
                else:
                    print("Bounding box too small, not added")
                
                self.bbox_start = None
                self.bbox_end = None
        else:
            # Point/mask mode
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.points.append((x, y))
                self.last_point = (x, y)
                
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.last_point = (x, y)
                # Update mask in real-time
                self.update_mask_drawing(x, y)
                
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.update_segmentation()
                
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right click to remove area
                self.points.append((x, y))
                self.mask_mode = "remove"
                self.update_mask_drawing(x, y)
                self.update_segmentation()
                self.mask_mode = "add"
    
    def update_mask_drawing(self, x, y):
        """Update mask while drawing"""
        if self.mask is None:
            return
            
        if self.mask_mode == "add":
            cv2.circle(self.mask, (x, y), self.brush_size, 1, -1)
        else:
            cv2.circle(self.mask, (x, y), self.brush_size, 0, -1)
    
    def update_segmentation(self):
        """Update segmentation based on current points and mask"""
        if len(self.points) == 0 and self.mask is None:
            return
            
        # This will be called after we have the frame
        pass
    
    def clear_segmentation(self):
        """Clear all segmentation data"""
        self.points = []
        self.mask = None
        self.current_mask = None
        self.bounding_boxes = []
        self.bbox_start = None
        self.bbox_end = None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return segmented result"""
        height, width = frame.shape[:2]
        
        # Initialize mask if needed
        if self.mask is None:
            self.mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create visualization
        result = frame.copy()
        
        # Draw points
        for i, point in enumerate(self.points):
            color = (0, 255, 0) if self.mask_mode == "add" else (0, 0, 255)
            cv2.circle(result, point, 5, color, -1)
            cv2.putText(result, f"{i+1}", (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Apply segmentation methods
        if self.use_grabcut and len(self.points) > 0:
            # Use GrabCut for better segmentation
            self.apply_grabcut(frame, result)
        elif self.mask is not None:
            # Use simple mask-based segmentation
            self.apply_mask_segmentation(result)
        
        # Draw current brush while dragging
        if self.last_point and self.drawing:
            color = (0, 255, 0) if self.mask_mode == "add" else (0, 0, 255)
            cv2.circle(result, self.last_point, self.brush_size, color, 2)
        
        # Draw manual bounding boxes
        for i, (x1, y1, x2, y2) in enumerate(self.bounding_boxes):
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(result, f"Manual {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw current bounding box being drawn
        if self.bbox_start is not None and self.bbox_end is not None:
            x1, y1 = self.bbox_start
            x2, y2 = self.bbox_end
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Always draw automatic detection boxes
        self.draw_automatic_detections(frame, result)
        
        return result
    
    def apply_grabcut(self, frame: np.ndarray, result: np.ndarray):
        """Apply GrabCut algorithm for segmentation"""
        if len(self.points) < 2:
            return
            
        # Create bounding box from points
        points_array = np.array(self.points)
        x1, y1 = np.min(points_array, axis=0)
        x2, y2 = np.max(points_array, axis=0)
        
        # Add some padding
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        rect = (x1, y1, x2-x1, y2-y1)
        
        # Initialize mask for GrabCut
        mask = np.zeros(frame.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            # Apply GrabCut
            cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 
                       self.grabcut_iterations, cv2.GC_INIT_WITH_RECT)
            
            # Create final mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Apply mask to result
            mask_colored = np.zeros_like(frame)
            mask_colored[mask2 == 1] = [0, 255, 0]
            result[:] = cv2.addWeighted(result, 0.7, mask_colored, 0.3, 0)
            
            self.current_mask = mask2
            
        except Exception as e:
            print(f"GrabCut error: {e}")
    
    def apply_mask_segmentation(self, result: np.ndarray):
        """Apply simple mask-based segmentation"""
        if self.mask is None:
            return
            
        # Create colored mask overlay
        mask_colored = np.zeros_like(result)
        mask_colored[self.mask == 1] = [0, 255, 0]
        result[:] = cv2.addWeighted(result, 0.7, mask_colored, 0.3, 0)
        self.current_mask = self.mask.copy()
    
    def draw_automatic_detections(self, frame: np.ndarray, result: np.ndarray):
        """Draw automatic object detection bounding boxes"""
        # Simple detection
        detections = self.detect_objects_contours(frame)
        
        # Draw detections
        for i, (x1, y1, x2, y2, confidence, label) in enumerate(detections):
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result, f"Object {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def apply_nms(self, detections: List[Tuple[int, int, int, int, float, str]], 
                  overlap_threshold: float = 0.3) -> List[Tuple[int, int, int, int, float, str]]:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if not detections:
            return []
        
        # Convert to numpy arrays for easier processing
        boxes = np.array([(x1, y1, x2, y2) for x1, y1, x2, y2, _, _ in detections])
        scores = np.array([conf for _, _, _, _, conf, _ in detections])
        labels = [label for _, _, _, _, _, label in detections]
        
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by confidence
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            # Pick the detection with highest confidence
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            remaining = indices[1:]
            
            # Calculate intersection
            x1 = np.maximum(boxes[current, 0], boxes[remaining, 0])
            y1 = np.maximum(boxes[current, 1], boxes[remaining, 1])
            x2 = np.minimum(boxes[current, 2], boxes[remaining, 2])
            y2 = np.minimum(boxes[current, 3], boxes[remaining, 3])
            
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            union = areas[current] + areas[remaining] - intersection
            iou = intersection / union
            
            # Remove boxes with high overlap
            indices = remaining[iou <= overlap_threshold]
        
        # Return filtered detections
        return [detections[i] for i in keep]
    
    def filter_detections(self, detections: List[Tuple[int, int, int, int, float, str]]) -> List[Tuple[int, int, int, int, float, str]]:
        """Filter and stabilize detections to reduce twitching"""
        if not detections:
            return []
        
        # Add current detections to history
        self.detection_history.append(detections)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        # If we don't have enough history, return current detections
        if len(self.detection_history) < self.min_detections:
            return detections
        
        # Find stable detections across frames
        stable_detections = []
        
        for detection in detections:
            x1, y1, x2, y2, confidence, label = detection
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Count how many times this detection appears in history
            detection_count = 0
            for hist_detections in self.detection_history:
                for hist_det in hist_detections:
                    hx1, hy1, hx2, hy2, hconf, hlabel = hist_det
                    h_center_x = (hx1 + hx2) / 2
                    h_center_y = (hy1 + hy2) / 2
                    
                    # Check if detection is close to historical detection
                    distance = ((center_x - h_center_x) ** 2 + (center_y - h_center_y) ** 2) ** 0.5
                    if distance < 50 and abs((x2-x1) - (hx2-hx1)) < 30:  # Similar size and position
                        detection_count += 1
                        break
            
            # Only show detection if it appears in multiple frames
            if detection_count >= self.min_detections:
                stable_detections.append(detection)
        
        return stable_detections
    
    def detect_objects_contours(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """Simple, clean object detection using standard OpenCV"""
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
    
    def detect_by_edges_improved(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """Improved edge detection with better accuracy"""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(filtered, (5, 5), 0)
        
        # Use adaptive thresholding for better edge detection
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Edge detection with optimized parameters
        edges = cv2.Canny(thresh, 30, 100)
        
        # Morphological operations to connect broken edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 800:  # Minimum area
                x, y, w, h = cv2.boundingRect(contour)
                
                # Better size and shape filtering
                if w > 25 and h > 25:
                    # Calculate aspect ratio and solidity
                    aspect_ratio = w / h
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    # More lenient but still reasonable filtering
                    if 0.2 < aspect_ratio < 5.0 and solidity > 0.3:
                        confidence = min(1.0, area / 15000)
                        detections.append((x, y, x+w, y+h, confidence, "Edge"))
        
        return detections
    
    def detect_by_color_improved(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """Improved color-based detection with better accuracy"""
        detections = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define more precise color ranges
        color_ranges = [
            # Red objects (two ranges for red)
            (np.array([0, 30, 30]), np.array([10, 255, 255]), "Red"),
            (np.array([170, 30, 30]), np.array([180, 255, 255]), "Red"),
            # Blue objects
            (np.array([100, 30, 30]), np.array([130, 255, 255]), "Blue"),
            # Green objects
            (np.array([40, 30, 30]), np.array([80, 255, 255]), "Green"),
            # Yellow objects
            (np.array([20, 30, 30]), np.array([40, 255, 255]), "Yellow"),
            # Orange objects
            (np.array([10, 30, 30]), np.array([20, 255, 255]), "Orange"),
            # Purple objects
            (np.array([130, 30, 30]), np.array([170, 255, 255]), "Purple"),
        ]
        
        for lower, upper, color_name in color_ranges:
            # Create mask for this color
            mask = cv2.inRange(hsv, lower, upper)
            
            # Clean up the mask with better morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 400:  # Lower threshold for color detection
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 15 and h > 15:  # Lower size threshold
                        # Calculate confidence based on area and color saturation
                        confidence = min(1.0, area / 8000)
                        detections.append((x, y, x+w, y+h, confidence, color_name))
        
        return detections
    
    def detect_by_contours(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """Detect objects using contour analysis on different color spaces"""
        detections = []
        
        # Try different color spaces for better detection
        color_spaces = [
            (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), "Gray"),
            (cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:,:,0], "LAB-L"),
            (cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,2], "HSV-V"),
        ]
        
        for gray, space_name in color_spaces:
            # Apply different preprocessing
            if space_name == "Gray":
                # For grayscale, use bilateral filter
                processed = cv2.bilateralFilter(gray, 9, 75, 75)
            else:
                # For other spaces, use Gaussian blur
                processed = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 600:  # Minimum area
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 20 and h > 20:  # Minimum size
                        # Calculate confidence
                        aspect_ratio = w / h
                        if 0.3 < aspect_ratio < 3.0:  # Reasonable aspect ratio
                            confidence = min(1.0, area / 12000)
                            detections.append((x, y, x+w, y+h, confidence, f"Contour-{space_name}"))
        
        return detections
    
    def detect_by_color(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """Detect objects by color segmentation"""
        detections = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common objects
        color_ranges = [
            # Red objects
            (np.array([0, 50, 50]), np.array([10, 255, 255]), "Red"),
            (np.array([170, 50, 50]), np.array([180, 255, 255]), "Red"),
            # Blue objects
            (np.array([100, 50, 50]), np.array([130, 255, 255]), "Blue"),
            # Green objects
            (np.array([40, 50, 50]), np.array([80, 255, 255]), "Green"),
            # Yellow objects
            (np.array([20, 50, 50]), np.array([40, 255, 255]), "Yellow"),
        ]
        
        for lower, upper, color_name in color_ranges:
            # Create mask for this color
            mask = cv2.inRange(hsv, lower, upper)
            
            # Clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 20 and h > 20:  # Minimum size
                        confidence = min(1.0, area / 10000)
                        detections.append((x, y, x+w, y+h, confidence, color_name))
        
        return detections
    
    def detect_by_edges(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """Detect objects using simple edge detection"""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small objects
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Basic size filtering
                if w > 30 and h > 30:
                    confidence = min(1.0, area / 20000)
                    detections.append((x, y, x+w, y+h, confidence, "Object"))
        
        return detections
    
    def detect_by_background_subtraction(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """Detect objects by background subtraction"""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(bg_gray, gray)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 600:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 30 and h > 30:
                    confidence = min(1.0, area / 8000)
                    detections.append((x, y, x+w, y+h, confidence, "Background"))
        
        return detections
    
    def detect_motion(self, current_frame: np.ndarray, prev_frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """Detect motion between frames with improved filtering"""
        detections = []
        
        # Convert to grayscale
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray_current = cv2.GaussianBlur(gray_current, (5, 5), 0)
        gray_prev = cv2.GaussianBlur(gray_prev, (5, 5), 0)
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(gray_prev, gray_current)
        
        # Apply higher threshold for more stable motion detection
        _, thresh = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
        
        # Stronger morphological operations to clean up
        kernel = np.ones((7, 7), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 800:  # Increased minimum area for stability
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # More strict size filtering
            if w < 40 or h < 40:
                continue
            
            # Calculate confidence based on area and shape
            aspect_ratio = w / h
            if 0.2 < aspect_ratio < 5.0:  # Reasonable aspect ratio
                confidence = min(1.0, area / 8000)  # Adjusted normalization
                detections.append((x, y, x+w, y+h, confidence, "Motion"))
        
        return detections
    
    def run_live_segmentation(self, camera_id: int = 0):
        """Run live segmentation from camera"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Create window and set mouse callback
        window_name = "Lightweight Live Segmentation"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("Edge Detection - Stationary Object Detection")
        print("- Automatically detects and draws bounding boxes around objects using edge detection")
        print("- 'q': Quit")
        print("- 's': Save current detection results")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            result = self.process_frame(frame)
            
            # Add UI text
            cv2.putText(result, "EDGE DETECTION - STATIONARY OBJECTS", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result, "Press 'q' to quit, 's' to save", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow(window_name, result)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_segmentation(frame)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_segmentation(self, frame: np.ndarray):
        """Save current segmentation results"""
        if self.current_mask is None and len(self.bounding_boxes) == 0:
            print("No segmentation or bounding boxes to save")
            return
        
        # Create output directory
        os.makedirs("segmentation_output", exist_ok=True)
        
        # Save original frame
        cv2.imwrite("segmentation_output/original.jpg", frame)
        
        # Save mask if available
        if self.current_mask is not None:
            mask_uint8 = (self.current_mask * 255).astype(np.uint8)
            cv2.imwrite("segmentation_output/mask.png", mask_uint8)
        
        # Save bounding boxes info
        if self.bounding_boxes:
            with open("segmentation_output/bounding_boxes.txt", "w") as f:
                f.write("Bounding Boxes (x1, y1, x2, y2):\n")
                for i, (x1, y1, x2, y2) in enumerate(self.bounding_boxes):
                    f.write(f"Box {i+1}: ({x1}, {y1}, {x2}, {y2})\n")
                    f.write(f"  Width: {x2-x1}, Height: {y2-y1}\n")
        
        # Save segmented result
        result = self.process_frame(frame)
        cv2.imwrite("segmentation_output/result.jpg", result)
        
        print("Segmentation saved to segmentation_output/")
        if self.bounding_boxes:
            print(f"Saved {len(self.bounding_boxes)} bounding boxes")

def main():
    parser = argparse.ArgumentParser(description="Lightweight Live Image Segmentation")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera ID")
    
    args = parser.parse_args()
    
    # Initialize and run segmentation
    try:
        segmenter = LightweightSegmentation()
        segmenter.run_live_segmentation(args.camera)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have OpenCV installed:")
        print("pip install opencv-python")

if __name__ == "__main__":
    main()
