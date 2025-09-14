#!/usr/bin/env python3
"""
Simple test script for the simplified DUM-E system
"""

import os
from capture_detection import capture_with_detection


def test_capture():
    """Test the capture function"""
    print("Testing capture_with_detection...")

    # Test capture
    image_path, detections = capture_with_detection()

    if image_path:
        print(f"✅ Success! Image saved: {image_path}")
        print(f"✅ Detections: {detections}")
        return True
    else:
        print("❌ Failed to capture")
        return False


def test_main_import():
    """Test that main.py can be imported without errors"""
    try:
        import main

        print("✅ main.py imports successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing main.py: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Simplified DUM-E System")
    print("=" * 50)

    # Test imports
    print("\n1. Testing imports...")
    import_success = test_main_import()

    # Test capture (only if camera is available)
    print("\n2. Testing capture...")
    capture_success = test_capture()

    print("\n" + "=" * 50)
    if import_success and capture_success:
        print("✅ All tests passed! System is ready.")
        print("\nTo run the full system:")
        print('python main.py "Pick up the red object and move it to the left"')
    else:
        print("❌ Some tests failed. Check the errors above.")
    print("=" * 50)
