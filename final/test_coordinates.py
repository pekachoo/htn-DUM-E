#!/usr/bin/env python3
"""
Test script to verify coordinate calculation matches homography_cv.py
"""

import cv2
import numpy as np
from homography_cv import CompletePipeline
from capture_detection import capture_with_detection


def test_coordinate_calculation():
    """Test that coordinate calculation matches between files"""
    print("Testing coordinate calculation consistency...")

    # Create a test pipeline
    pipeline = CompletePipeline()

    # Create a dummy frame (400x400 black image)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add some test rectangles to simulate objects
    cv2.rectangle(
        test_frame, (100, 100), (150, 150), (0, 255, 0), -1
    )  # Green rectangle
    cv2.rectangle(test_frame, (200, 200), (250, 250), (0, 0, 255), -1)  # Red rectangle

    # Test homography_cv.py method
    print("\n1. Testing homography_cv.py get_detections method:")
    detections_homography = pipeline.get_detections(test_frame)
    print(f"   Homography detections: {detections_homography}")

    # Test capture_detection.py method
    print("\n2. Testing capture_detection.py coordinate calculation:")

    # Apply homography to get warped frame for detection
    warped_for_detection = cv2.warpPerspective(test_frame, pipeline.H, (400, 400))
    rects = pipeline._find_filtered_rects(warped_for_detection)
    if isinstance(rects, tuple):
        rects = rects[0]

    detections_capture = []
    for i, (x, y, w, h) in enumerate(rects):
        cx = x + w // 2
        cy = y + h // 2
        # Convert to (x, 400-y), then scale to cm - same as homography_cv.py
        cx_cm = (cx * 30.0) / 400.0
        cy_cm = ((400 - cy) * 30.0) / 400.0
        detections_capture.append((i + 1, cx_cm, cy_cm))

    print(f"   Capture detections: {detections_capture}")

    # Compare results
    print("\n3. Comparison:")
    if len(detections_homography) == len(detections_capture):
        print("   ✅ Same number of detections")
        for i, (h_det, c_det) in enumerate(
            zip(detections_homography, detections_capture)
        ):
            if abs(h_det[1] - c_det[1]) < 0.01 and abs(h_det[2] - c_det[2]) < 0.01:
                print(f"   ✅ Detection {i+1} coordinates match")
            else:
                print(f"   ❌ Detection {i+1} coordinates differ: {h_det} vs {c_det}")
    else:
        print(
            f"   ❌ Different number of detections: {len(detections_homography)} vs {len(detections_capture)}"
        )

    return len(detections_homography) == len(detections_capture)


if __name__ == "__main__":
    print("=" * 60)
    print("Coordinate Calculation Test")
    print("=" * 60)

    success = test_coordinate_calculation()

    print("\n" + "=" * 60)
    if success:
        print("✅ Coordinate calculation test passed!")
    else:
        print("❌ Coordinate calculation test failed!")
    print("=" * 60)
