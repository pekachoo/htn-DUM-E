#!/usr/bin/env python3
"""
Debug version of main.py to test the connection
"""

import requests
import json


def test_arm_connection():
    """Test connection to arm server"""

    # Test data
    test_data = {
        "action": "move_to_idle",
        "task_description": "Test connection",
        "task_complete": False,
    }

    arm_server_url = "http://10.37.101.152:5000"

    print("Testing arm server connection...")
    print(f"URL: {arm_server_url}")
    print(f"Data: {json.dumps(test_data, indent=2)}")

    try:
        # Test health first
        print("\n1. Testing health endpoint...")
        health_response = requests.get(f"{arm_server_url}/health", timeout=5)
        print(f"Health status: {health_response.status_code}")
        print(f"Health response: {health_response.json()}")

        # Test arm control
        print("\n2. Testing arm control endpoint...")
        control_response = requests.post(
            f"{arm_server_url}/arm_control", json=test_data, timeout=10
        )
        print(f"Control status: {control_response.status_code}")
        print(f"Control response: {control_response.json()}")

        if control_response.status_code == 200:
            print("✅ Connection successful!")
            return True
        else:
            print("❌ Connection failed")
            return False

    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        print("Make sure the Flask server is running and accessible")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    test_arm_connection()
