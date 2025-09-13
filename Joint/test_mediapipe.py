#!/usr/bin/env python3
"""
Test script to check MediaPipe installation and camera access
"""

try:
    import cv2
    print("✅ OpenCV imported successfully")
    print(f"OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")
    exit(1)

try:
    import mediapipe as mp
    print("✅ MediaPipe imported successfully")
    print(f"MediaPipe version: {mp.__version__}")
except ImportError as e:
    print(f"❌ MediaPipe import failed: {e}")
    print("Install with: pip install mediapipe")
    exit(1)

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")
    exit(1)

# Test camera access
print("\n🎥 Testing camera access...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open camera 0")
    # Try other camera IDs
    for cam_id in [1, 2]:
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            print(f"✅ Camera {cam_id} opened successfully")
            break
        else:
            print(f"❌ Could not open camera {cam_id}")
else:
    print("✅ Camera 0 opened successfully")

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("✅ Frame captured successfully")
        print(f"Frame shape: {frame.shape}")
    else:
        print("❌ Could not read frame")
    cap.release()
else:
    print("❌ No camera available")

print("\n🧪 Testing MediaPipe Hands...")
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    print("✅ MediaPipe Hands initialized successfully")
except Exception as e:
    print(f"❌ MediaPipe Hands initialization failed: {e}")

print("\n✅ All tests completed!")
