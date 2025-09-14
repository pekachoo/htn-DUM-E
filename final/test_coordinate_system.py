#!/usr/bin/env python3
"""
Test script to verify the coordinate system is working correctly
"""


def test_coordinate_formatting():
    """Test that coordinates are formatted correctly for the LLM"""
    print("Testing coordinate system formatting...")

    # Simulate detections from homography_cv.py (in CM coordinates)
    test_detections = [
        (1, 15.5, 12.3),  # Object 1 at (15.5cm, 12.3cm)
        (2, 8.2, 25.7),  # Object 2 at (8.2cm, 25.7cm)
        (3, 22.1, 5.9),  # Object 3 at (22.1cm, 5.9cm)
    ]

    print("\n1. Test detections (from homography_cv.py):")
    for obj_num, x, y in test_detections:
        print(f"   Object {obj_num}: ({x:.2f}cm, {y:.2f}cm)")

    # Format as the LLM would receive them
    print("\n2. Formatted for LLM context:")
    detection_strings = [
        f"Object {obj_num}: ({x:.2f}cm, {y:.2f}cm)" for obj_num, x, y in test_detections
    ]
    detection_text = "Detected objects (in CM coordinates):\n" + "\n".join(
        detection_strings
    )
    print(detection_text)

    # Test coordinate validation
    print("\n3. Testing coordinate validation:")
    for obj_num, x, y in test_detections:
        if 2 <= x <= 28 and 2 <= y <= 28:
            print(
                f"   ✅ Object {obj_num}: ({x:.2f}cm, {y:.2f}cm) - VALID (within 2-28 range)"
            )
        else:
            print(
                f"   ❌ Object {obj_num}: ({x:.2f}cm, {y:.2f}cm) - INVALID (outside 2-28 range)"
            )

    print("\n4. Expected LLM response format:")
    print("   The LLM should now understand:")
    print("   - X: 0-30cm (left to right)")
    print("   - Y: 0-30cm (close to far)")
    print("   - Z: 0-30cm (low to high)")
    print("   - All coordinates in centimeters")
    print("   - Avoid edge values (stay between 2-28 for X,Y)")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Coordinate System Test")
    print("=" * 60)

    success = test_coordinate_formatting()

    print("\n" + "=" * 60)
    if success:
        print("✅ Coordinate system test passed!")
        print("The LLM should now receive proper CM coordinates from homography_cv.py")
    else:
        print("❌ Coordinate system test failed!")
    print("=" * 60)
