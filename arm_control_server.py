#!/usr/bin/env python3
"""
Flask server for DUM-E robotic arm control
Provides a single endpoint that routes to appropriate arm functions based on action type
"""

from flask import Flask, request, jsonify
import serial
import time
import math
import threading
from pi_to_arduino import (
    IKSolver,
    grab,
    move_to_idle_position,
    move,
    wave_bye,
    shake_yes,
    shake_no,
    shake_hand,
    move_to_hold,
    hold,
)

app = Flask(__name__)

# Global variables for arm control
ser = None
ikSolver = None
arm_ready = False


def initialize_arm():
    """Initialize the arm connection and IK solver"""
    global ser, ikSolver, arm_ready
    try:
        ser = serial.Serial(
            "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0", 9600, timeout=1
        )
        ikSolver = IKSolver(P1=12.0, P2=12.3, P3=8.0)
        time.sleep(3)  # Wait for arm to initialize
        arm_ready = True
        print("Arm initialized successfully")
    except Exception as e:
        print(f"Failed to initialize arm: {e}")
        arm_ready = False


def validate_coordinates(data):
    """Validate coordinate data"""
    required_coords = ["x", "y"]
    for coord in required_coords:
        if coord not in data:
            return False, f"Missing required coordinate: {coord}"
        if not isinstance(data[coord], (int, float)):
            return False, f"Invalid {coord} coordinate type"
    return True, "Valid"


@app.route("/arm_control", methods=["POST"])
def arm_control():
    """Simple arm control endpoint"""
    global ser, ikSolver, arm_ready

    if not arm_ready:
        return jsonify({"success": False, "error": "Arm not ready"}), 500

    try:
        data = request.get_json()
        action = data.get("action", "").lower()

        print(f"Executing: {action}")

        if action == "grab":
            x = float(data["x"])
            y = float(data["y"])
            phi = float(data.get("phi", 270))
            x2 = float(data.get("x2", x))
            y2 = float(data.get("y2", y))
            grab(ikSolver, x, y, phi, ser, x2, y2)

        elif action == "move":
            x = float(data["x"])
            y = float(data["y"])
            z = float(data.get("z", 0))
            phi = float(data.get("phi", 0))
            claw_open = int(data.get("claw_open", 1))
            roll_angle = float(data.get("roll_angle", 0))
            elbow = data.get("elbow", "up")
            move(ikSolver, x, y, z, phi, ser, claw_open, roll_angle, elbow)

        elif action == "move_to_hold":
            x = float(data["x"])
            y = float(data["y"])
            move_to_hold(ikSolver, ser, x, y)

        elif action == "hold":
            x = float(data["x"])
            y = float(data["y"])
            hold(ikSolver, ser, x, y)

        elif action == "wave_bye":
            wave_bye(ikSolver, ser)

        elif action == "shake_yes":
            shake_yes(ikSolver, ser)

        elif action == "shake_no":
            shake_no(ikSolver, ser)

        elif action == "shake_hand":
            shake_hand(ikSolver, ser)

        elif action == "move_to_idle":
            move_to_idle_position(ikSolver, ser)

        else:
            return (
                jsonify({"success": False, "error": f"Unknown action: {action}"}),
                400,
            )

        return jsonify({"success": True, "action": action})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/arm_status", methods=["GET"])
def arm_status():
    """Check if arm is ready"""
    return jsonify(
        {
            "ready": arm_ready,
            "message": "Arm ready" if arm_ready else "Arm not initialized",
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "arm_ready": arm_ready})


if __name__ == "__main__":
    print("Initializing DUM-E Arm Control Server...")

    # Initialize arm in a separate thread to avoid blocking
    arm_thread = threading.Thread(target=initialize_arm)
    arm_thread.daemon = True
    arm_thread.start()

    print("Starting Flask server on http://0.0.0.0:5000")
    print("Available endpoints:")
    print("  POST /arm_control - Main arm control endpoint")
    print("  GET  /arm_status - Check arm status")
    print("  GET  /health - Health check")
    print("\nRequired JSON format for /arm_control:")
    print(
        "  action: string (required) - one of: grab, move_to_idle, move, wave_bye, shake_yes, shake_no, shake_hand, move_to_hold, hold, drop_off"
    )
    print("  x: float (required for most actions) - X coordinate in cm")
    print("  y: float (required for most actions) - Y coordinate in cm")
    print("  z: float (optional) - Z coordinate in cm, defaults to 0")
    print("  phi: float (optional) - angle in degrees, defaults to 0")
    print("  claw_open: int (optional) - 1 for open, 0 for closed, defaults to 1")
    print("  roll_angle: float (optional) - roll angle in degrees, defaults to 0")
    print("  elbow: string (optional) - 'up' or 'down', defaults to 'up'")

    app.run(host="0.0.0.0", port=5000, debug=True)
