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
    drop_off,
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
    """
    Single endpoint for all arm control operations
    Routes to appropriate function based on action type
    """
    global ser, ikSolver, arm_ready

    if not arm_ready:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Arm not initialized. Please check connection.",
                }
            ),
            500,
        )

    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        action = data.get("action", "").lower()

        # Route to appropriate function based on action
        if action == "grab":
            # grab(ikSolver, x, y, phi, ser)
            valid, msg = validate_coordinates(data)
            if not valid:
                return jsonify({"success": False, "error": msg}), 400

            x = float(data["x"])
            y = float(data["y"])
            phi = float(data.get("phi", 0))  # Default to 0 if not provided

            print(f"Executing GRAB: x={x}, y={y}, phi={phi}")
            grab(ikSolver, x, y, phi, ser)

        elif action == "move_to_idle":
            # move_to_idle_position(ikSolver, ser)
            print("Executing MOVE_TO_IDLE")
            move_to_idle_position(ikSolver, ser)

        elif action == "move":
            # move(ikSolver, x, y, z, phi, ser, claw_open=1, roll_angle=0, elbow='up')
            valid, msg = validate_coordinates(data)
            if not valid:
                return jsonify({"success": False, "error": msg}), 400

            x = float(data["x"])
            y = float(data["y"])
            z = float(data.get("z", 0))  # Default to 0
            phi = float(data.get("phi", 0))  # Default to 0
            claw_open = int(data.get("claw_open", 1))  # Default to open
            roll_angle = float(data.get("roll_angle", 0))  # Default to 0
            elbow = data.get("elbow", "up")  # Default to 'up'

            print(
                f"Executing MOVE: x={x}, y={y}, z={z}, phi={phi}, claw_open={claw_open}, roll_angle={roll_angle}, elbow={elbow}"
            )
            move(ikSolver, x, y, z, phi, ser, claw_open, roll_angle, elbow)

        elif action == "wave_bye":
            # wave_bye(ikSolver, ser)
            print("Executing WAVE_BYE")
            wave_bye(ikSolver, ser)

        elif action == "shake_yes":
            # shake_yes(ikSolver, ser)
            print("Executing SHAKE_YES")
            shake_yes(ikSolver, ser)

        elif action == "shake_no":
            # shake_no(ikSolver, ser)
            print("Executing SHAKE_NO")
            shake_no(ikSolver, ser)

        elif action == "shake_hand":
            # shake_hand(ikSolver, ser)
            print("Executing SHAKE_HAND")
            shake_hand(ikSolver, ser)

        elif action == "move_to_hold":
            # move_to_hold(ikSolver, ser, x, y)
            valid, msg = validate_coordinates(data)
            if not valid:
                return jsonify({"success": False, "error": msg}), 400

            x = float(data["x"])
            y = float(data["y"])
            print(f"Executing MOVE_TO_HOLD: x={x}, y={y}")
            move_to_hold(ikSolver, ser, x, y)

        elif action == "hold":
            # hold(ikSolver, ser, x, y)
            valid, msg = validate_coordinates(data)
            if not valid:
                return jsonify({"success": False, "error": msg}), 400

            x = float(data["x"])
            y = float(data["y"])
            print(f"Executing HOLD: x={x}, y={y}")
            hold(ikSolver, ser, x, y)

        elif action == "drop_off":
            # drop_off(ikSolver, ser, x, y, z, phi)
            valid, msg = validate_coordinates(data)
            if not valid:
                return jsonify({"success": False, "error": msg}), 400

            x = float(data["x"])
            y = float(data["y"])
            z = float(data.get("z", 0))  # Default to 0
            phi = float(data.get("phi", 0))  # Default to 0

            print(f"Executing DROP_OFF: x={x}, y={y}, z={z}, phi={phi}")
            drop_off(ikSolver, ser, x, y, z, phi)

        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Unknown action: {action}. Valid actions: grab, move_to_idle, move, wave_bye, shake_yes, shake_no, shake_hand, move_to_hold, hold, drop_off",
                    }
                ),
                400,
            )

        return jsonify(
            {
                "success": True,
                "message": f"Action {action} executed successfully",
                "action": action,
            }
        )

    except Exception as e:
        print(f"Error executing arm control: {e}")
        return jsonify({"success": False, "error": f"Execution error: {str(e)}"}), 500


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

    print("Starting Flask server on http://localhost:5000")
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
