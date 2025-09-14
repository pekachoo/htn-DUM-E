#!/usr/bin/env python3
"""
Simple test of the simplified system
"""

import requests
import json


def test_simple():
    """Test the simplified arm system"""

    # Test data
    test_cases = [
        {"action": "wave_bye", "task_description": "Wave goodbye"},
        {"action": "shake_yes", "task_description": "Nod yes"},
        {"action": "move_to_idle", "task_description": "Go to idle"},
        {
            "action": "grab",
            "x": 15.0,
            "y": 10.0,
            "phi": 270,
            "x2": 25.0,
            "y2": 20.0,
            "task_description": "Grab and move",
        },
        {
            "action": "move_to_hold",
            "x": 12.0,
            "y": 8.0,
            "task_description": "Move to hold",
        },
    ]

    base_url = "http://10.37.101.152:5000"

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['action']}")
        try:
            response = requests.post(f"{base_url}/arm_control", json=test, timeout=10)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    test_simple()
