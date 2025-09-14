#!/usr/bin/env python3
"""
Test script to demonstrate how to use the capture_detection module
"""

from capture_detection import prompt_user_and_capture, DetectionCapture


def test_main_function():
    """
    Test the main prompt_user_and_capture function
    """
    print("Testing main capture function...")
    result = prompt_user_and_capture(camera_id=0)

    print("\n" + "=" * 50)
    print("RESULT SUMMARY:")
    print("=" * 50)
    print(f"Success: {result['success']}")
    print(f"Image Path: {result['image_path']}")
    print(f"Number of Detections: {len(result['detections'])}")
    print(f"Formatted Text:\n{result['formatted_text']}")

    return result


def test_direct_usage():
    """
    Test using the DetectionCapture class directly
    """
    print("\nTesting direct class usage...")

    # Create capture system
    capture_system = DetectionCapture()

    # Initialize camera
    if not capture_system.initialize_camera(0):
        print("Failed to initialize camera")
        return None

    try:
        # Capture without user prompt (for automated testing)
        result = capture_system.capture_and_save("test_direct.jpg")

        print(f"Direct capture result: {result['success']}")
        if result["success"]:
            print(f"Detections: {result['detections']}")
            print(f"Text: {result['formatted_text']}")

        return result

    finally:
        capture_system.cleanup()


if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Interactive mode (with user prompt)")
    print("2. Direct mode (automated)")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        test_main_function()
    elif choice == "2":
        test_direct_usage()
    else:
        print("Invalid choice. Running interactive mode...")
        test_main_function()
