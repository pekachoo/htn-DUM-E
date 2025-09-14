#!/usr/bin/env python3
"""
Test script for the new action-based LLM system
Tests the updated prompt and arm control integration
"""

import requests
import json


def test_action_based_system():
    """Test the action-based system with sample LLM responses"""

    print("Testing Action-Based DUM-E System")
    print("=" * 50)

    # Sample LLM responses that would come from the new prompt
    sample_responses = [
        {
            "action": "grab",
            "x": 15.0,
            "y": 10.0,
            "phi": 0,
            "task_description": "Pick up the red object",
            "task_complete": False,
        },
        {
            "action": "move",
            "x": 25.0,
            "y": 20.0,
            "z": 5.0,
            "phi": 45,
            "claw_open": 0,
            "roll_angle": 0,
            "elbow": "up",
            "task_description": "Move the object to the top right corner",
            "task_complete": False,
        },
        {
            "action": "wave_bye",
            "task_description": "Wave goodbye to the user",
            "task_complete": True,
        },
        {
            "action": "move_to_idle",
            "task_description": "Return to safe position",
            "task_complete": False,
        },
    ]

    base_url = "http://localhost:5000"

    for i, response in enumerate(sample_responses, 1):
        print(f"\n--- Test {i}: {response['action'].upper()} ---")
        print(f"LLM Response: {json.dumps(response, indent=2)}")

        try:
            # Send to arm control server
            arm_response = requests.post(
                f"{base_url}/arm_control", json=response, timeout=10
            )
            print(f"Arm Server Response: {arm_response.json()}")

            if arm_response.status_code == 200:
                print("✅ Action executed successfully")
            else:
                print("❌ Action failed")

        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to arm server")
            print("Make sure to start the server with: python arm_control_server.py")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "=" * 50)
    print("Action-based system test completed!")


if __name__ == "__main__":
    print("This test demonstrates the new action-based LLM system.")
    print("The LLM now specifies which action to take instead of just coordinates.")
    print("\nPress Enter to start the test...")
    input()

    test_action_based_system()
