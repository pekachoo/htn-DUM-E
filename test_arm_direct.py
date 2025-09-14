#!/usr/bin/env python3
"""
Direct arm control test - bypasses LLM and directly tests arm functions
"""

import requests
import json
import time

# Configuration
ARM_SERVER_URL = "http://10.37.101.152:5000"  # Change this to your server IP


def test_arm_action(action_data, description):
    """Test a single arm action"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Action: {action_data['action']}")
    print(f"Data: {json.dumps(action_data, indent=2)}")
    print(f"{'='*60}")

    try:
        response = requests.post(
            f"{ARM_SERVER_URL}/arm_control", json=action_data, timeout=15
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        if response.status_code == 200:
            print("✅ SUCCESS!")
            return True
        else:
            print("❌ FAILED!")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR - Server not reachable")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_all_arm_functions():
    """Test all available arm functions"""

    print("🤖 DUM-E Direct Arm Control Test")
    print("=" * 60)
    print(f"Server URL: {ARM_SERVER_URL}")
    print("=" * 60)

    # Test health first
    print("\n1. Testing server health...")
    try:
        health_response = requests.get(f"{ARM_SERVER_URL}/health", timeout=5)
        print(f"Health Status: {health_response.status_code}")
        print(f"Health Response: {health_response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return

    # Test cases
    test_cases = [
        # Gesture tests (no coordinates needed)
        {
            "data": {
                "action": "move_to_idle",
                "task_description": "Move to idle position",
            },
            "description": "Move to Idle Position",
        },
        {
            "data": {"action": "wave_bye", "task_description": "Wave goodbye"},
            "description": "Wave Goodbye Gesture",
        },
        {
            "data": {"action": "shake_yes", "task_description": "Nod yes"},
            "description": "Shake Yes Gesture",
        },
        {
            "data": {"action": "shake_no", "task_description": "Shake no"},
            "description": "Shake No Gesture",
        },
        {
            "data": {"action": "shake_hand", "task_description": "Handshake"},
            "description": "Shake Hand Gesture",
        },
        # Coordinate-based tests
        {
            "data": {
                "action": "grab",
                "x": 15.0,
                "y": 10.0,
                "phi": 270,
                "x2": 25.0,
                "y2": 20.0,
                "task_description": "Grab object at (15, 10) and move to (25, 20)",
            },
            "description": "Grab Object at (15, 10) and move to (25, 20)",
        },
        {
            "data": {
                "action": "move",
                "x": 20.0,
                "y": 15.0,
                "z": 5.0,
                "phi": 45,
                "claw_open": 1,
                "roll_angle": 0,
                "elbow": "up",
                "task_description": "Move to (20, 15, 5)",
            },
            "description": "Move to (20, 15, 5)",
        },
        {
            "data": {
                "action": "move_to_hold",
                "x": 12.0,
                "y": 8.0,
                "task_description": "Move to hold position",
            },
            "description": "Move to Hold Position",
        },
        {
            "data": {
                "action": "hold",
                "x": 12.0,
                "y": 8.0,
                "task_description": "Hold at position",
            },
            "description": "Hold at Position",
        },
        {
            "data": {
                "action": "drop_off",
                "x": 25.0,
                "y": 20.0,
                "z": 0,
                "phi": 0,
                "task_description": "Drop object at (25, 20)",
            },
            "description": "Drop Off at (25, 20)",
        },
    ]

    # Run tests
    success_count = 0
    total_tests = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{total_tests}] {test_case['description']}")

        success = test_arm_action(test_case["data"], test_case["description"])
        if success:
            success_count += 1

        # Wait between tests to avoid overwhelming the arm
        if i < total_tests:
            print("⏳ Waiting 2 seconds before next test...")
            time.sleep(2)

    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_tests - success_count}")
    print(f"Success Rate: {(success_count/total_tests)*100:.1f}%")

    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  Some tests failed - check the arm connection and server status")


def interactive_test():
    """Interactive mode for custom testing"""
    print("\n🎮 Interactive Arm Control")
    print("=" * 40)
    print("Available actions:")
    print("1. grab - Pick up object and move to drop-off location")
    print("2. move - Move to position")
    print("3. wave_bye - Wave goodbye")
    print("4. shake_yes - Nod yes")
    print("5. shake_no - Shake no")
    print("6. shake_hand - Handshake")
    print("7. move_to_idle - Go to idle")
    print("8. move_to_hold - Move to hold position")
    print("9. hold - Hold at position")
    print("10. drop_off - Drop object")
    print("0. Exit")

    while True:
        try:
            choice = input("\nEnter action number (0-10): ").strip()

            if choice == "0":
                print("Goodbye!")
                break
            elif choice == "1":  # grab
                x = float(input("Enter X coordinate: "))
                y = float(input("Enter Y coordinate: "))
                phi = float(
                    input("Enter phi angle (270 for top-down, 0 for side): ") or "270"
                )
                x2 = float(input("Enter drop-off X coordinate: "))
                y2 = float(input("Enter drop-off Y coordinate: "))
                test_arm_action(
                    {
                        "action": "grab",
                        "x": x,
                        "y": y,
                        "phi": phi,
                        "x2": x2,
                        "y2": y2,
                        "task_description": f"Grab at ({x}, {y}) and move to ({x2}, {y2})",
                    },
                    f"Grab at ({x}, {y}) and move to ({x2}, {y2})",
                )
            elif choice == "2":  # move
                x = float(input("Enter X coordinate: "))
                y = float(input("Enter Y coordinate: "))
                z = float(input("Enter Z coordinate (0): ") or "0")
                phi = float(input("Enter phi angle (0): ") or "0")
                claw_open = int(input("Claw open? (1=yes, 0=no): ") or "1")
                test_arm_action(
                    {
                        "action": "move",
                        "x": x,
                        "y": y,
                        "z": z,
                        "phi": phi,
                        "claw_open": claw_open,
                        "roll_angle": 0,
                        "elbow": "up",
                        "task_description": f"Move to ({x}, {y}, {z})",
                    },
                    f"Move to ({x}, {y}, {z})",
                )
            elif choice in ["3", "4", "5", "6", "7"]:  # gestures
                actions = {
                    "3": "wave_bye",
                    "4": "shake_yes",
                    "5": "shake_no",
                    "6": "shake_hand",
                    "7": "move_to_idle",
                }
                action = actions[choice]
                test_arm_action(
                    {"action": action, "task_description": f"Execute {action}"},
                    f"Execute {action}",
                )
            elif choice in ["8", "9"]:  # hold functions
                x = float(input("Enter X coordinate: "))
                y = float(input("Enter Y coordinate: "))
                action = "move_to_hold" if choice == "8" else "hold"
                test_arm_action(
                    {
                        "action": action,
                        "x": x,
                        "y": y,
                        "task_description": f"{action} at ({x}, {y})",
                    },
                    f"{action} at ({x}, {y})",
                )
            elif choice == "10":  # drop_off
                x = float(input("Enter X coordinate: "))
                y = float(input("Enter Y coordinate: "))
                z = float(input("Enter Z coordinate (0): ") or "0")
                phi = float(input("Enter phi angle (0): ") or "0")
                test_arm_action(
                    {
                        "action": "drop_off",
                        "x": x,
                        "y": y,
                        "z": z,
                        "phi": phi,
                        "task_description": f"Drop off at ({x}, {y}, {z})",
                    },
                    f"Drop off at ({x}, {y}, {z})",
                )
            else:
                print("Invalid choice. Please enter 0-10.")

        except ValueError:
            print("Invalid input. Please enter numbers only.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Run all tests automatically")
    print("2. Interactive mode")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        test_all_arm_functions()
    elif choice == "2":
        interactive_test()
    else:
        print("Invalid choice. Running all tests...")
        test_all_arm_functions()
