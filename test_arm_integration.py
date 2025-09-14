#!/usr/bin/env python3
"""
Test script for DUM-E arm integration
Tests the Flask server and main.py integration
"""

import requests
import time
import json


def test_arm_server():
    """Test the arm control server endpoints"""
    base_url = "http://localhost:5000"

    print("Testing DUM-E Arm Control Server")
    print("=" * 40)

    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
        return False

    # Test arm status
    print("\n2. Testing arm status...")
    try:
        response = requests.get(f"{base_url}/arm_status", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
        return False

    # Test move to idle
    print("\n3. Testing move to idle...")
    try:
        data = {"action": "move_to_idle"}
        response = requests.post(f"{base_url}/arm_control", json=data, timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
        return False

    # Test grab action
    print("\n4. Testing grab action...")
    try:
        data = {"action": "grab", "x": 10.0, "y": 10.0, "phi": 0}
        response = requests.post(f"{base_url}/arm_control", json=data, timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
        return False

    # Test move action
    print("\n5. Testing move action...")
    try:
        data = {
            "action": "move",
            "x": 15.0,
            "y": 15.0,
            "z": 5.0,
            "phi": 45,
            "claw_open": 1,
            "roll_angle": 0,
        }
        response = requests.post(f"{base_url}/arm_control", json=data, timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
        return False

    # Test gesture
    print("\n6. Testing wave gesture...")
    try:
        data = {"action": "wave_bye"}
        response = requests.post(f"{base_url}/arm_control", json=data, timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
        return False

    print("\nAll tests completed!")
    return True


if __name__ == "__main__":
    print("Make sure the arm control server is running:")
    print("python arm_control_server.py")
    print("\nPress Enter to start tests...")
    input()

    test_arm_server()
