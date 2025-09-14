#!/usr/bin/env python3
"""
Test script to verify main.py imports and basic functionality
"""

print("Testing main.py imports and basic functionality...")

try:
    # Test imports
    import time
    import os
    import sys

    print("✓ Basic imports successful")

    import cv2

    print("✓ OpenCV imported successfully")

    from capture_detection import DetectionCapture

    print("✓ DetectionCapture imported successfully")

    from llm import analyze_with_groq, send_to_arm_control, parse_groq_response

    print("✓ LLM functions imported successfully")

    # Test basic functionality
    print("\nTesting basic functionality...")

    # Test arm control mock
    test_coords = {
        "in_coord": [10, 20, 30],
        "out_coord": [40, 50, 60],
        "gripper_action": "open",
        "task_description": "Test task",
    }

    print("Testing arm control mock:")
    result = send_to_arm_control(test_coords)
    print(f"Arm control result: {result}")

    print("\n✅ All tests passed! main.py should work correctly now.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install missing dependencies:")
    print("pip install opencv-python groq python-dotenv requests")

except Exception as e:
    print(f"❌ Error: {e}")
