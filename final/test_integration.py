#!/usr/bin/env python3
"""
Test script for the complete DUM-E integration
"""

from main import DUMESystem
import time


def test_system_initialization():
    """Test if the system initializes properly"""
    print("Testing system initialization...")

    dume = DUMESystem(camera_id=0)
    success = dume.initialize()

    if success:
        print("System initialization successful")
        dume.cleanup()
        return True
    else:
        print("System initialization failed")
        return False


def test_capture_only():
    """Test just the capture functionality"""
    print("\nTesting capture functionality...")

    dume = DUMESystem(camera_id=0)

    if not dume.initialize():
        print("Failed to initialize system")
        return False

    try:
        # Test capture
        image_path, detections, formatted_text = dume.capture_current_scene()

        if image_path:
            print("Capture test successful")
            print(f"   Image saved: {image_path}")
            print(f"   Detections: {len(detections)} objects")
            return True
        else:
            print("Capture test failed")
            return False

    finally:
        dume.cleanup()


def test_quick_task():
    """Test a quick task execution"""
    print("\nTesting quick task execution...")

    dume = DUMESystem(camera_id=0)

    if not dume.initialize():
        print("Failed to initialize system")
        return False

    try:
        # Test with a simple prompt
        test_prompt = "Look at the objects in the scene"
        print(f"Testing with prompt: '{test_prompt}'")

        # This will run the full pipeline but with sleep simulation
        success = dume.execute_task(test_prompt)

        if success:
            print("Task execution test successful")
        else:
            print("Task execution test failed")

        return success

    finally:
        dume.cleanup()


def main():
    """Run all tests"""
    print("DUM-E Integration Test Suite")
    print("=" * 40)

    tests = [
        ("System Initialization", test_system_initialization),
        ("Capture Functionality", test_capture_only),
        ("Quick Task Execution", test_quick_task),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test '{test_name}' failed with error: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*20} Test Summary {'='*20}")
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{len(results)} tests")

    if passed == len(results):
        print("All tests passed! System is ready to use.")
    else:
        print("Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
