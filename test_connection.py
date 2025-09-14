#!/usr/bin/env python3
"""
Quick test to verify Flask server connection
"""

import requests
import json


def test_connection():
    """Test connection to the Flask server"""

    # Test both localhost and the IP address
    urls = ["http://localhost:5000", "http://10.37.101.152:5000"]

    for url in urls:
        print(f"\nTesting connection to {url}")
        print("-" * 50)

        try:
            # Test health endpoint
            response = requests.get(f"{url}/health", timeout=5)
            print(f"Health check: {response.status_code}")
            if response.status_code == 200:
                print(f"Response: {response.json()}")
            else:
                print(f"Error: {response.text}")

        except requests.exceptions.ConnectionError:
            print("❌ Connection failed - server not reachable")
        except Exception as e:
            print(f"❌ Error: {e}")

        try:
            # Test arm control endpoint
            test_data = {"action": "move_to_idle"}
            response = requests.post(f"{url}/arm_control", json=test_data, timeout=5)
            print(f"Arm control test: {response.status_code}")
            if response.status_code == 200:
                print(f"Response: {response.json()}")
            else:
                print(f"Error: {response.text}")

        except requests.exceptions.ConnectionError:
            print("❌ Arm control connection failed")
        except Exception as e:
            print(f"❌ Arm control error: {e}")


if __name__ == "__main__":
    print("Testing Flask server connection...")
    test_connection()
