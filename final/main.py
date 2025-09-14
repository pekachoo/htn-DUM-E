#!/usr/bin/env python3
"""
DUM-E robotic arm system (lenient, multi-object, human-like)
Takes CLI prompt, captures image with detection, sends to Groq, loops until task complete.
If there are multiple things to do, just do the next one that makes sense (don't repeat the same one over and over).
Be lenient: if it looks good enough, call it done!
"""

import sys
import time
import json
import base64
import os
import requests
from groq import Groq
from dotenv import load_dotenv
from capture_detection import capture_with_detection

load_dotenv()

# === Set your Flask server URL here ===
ARM_SERVER_URL = "http://10.37.101.152:5000"
# ======================================


def encode_image(image_path):
    """Encode image to base64 for Groq API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_groq_response(response_text):
    """Parse Groq response to extract coordinate dictionary"""
    try:
        # Try to find a JSON object in the response
        lines = response_text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        # Fallback: try to extract JSON from the whole response
        if "{" in response_text and "}" in response_text:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_str = response_text[start:end]
            return json.loads(json_str)

        print("No valid JSON found in Groq response")
        return None

    except Exception as e:
        print(f"Error parsing Groq response: {e}")
        return None


def analyze_with_groq(image_path, user_prompt, detections):
    """Analyze image with Groq and return coordinate dictionary (lenient, multi-object, human-like)"""
    try:
        base64_image = encode_image(image_path)
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # Format detections for context
        if detections:
            detection_strings = [
                f"Object {obj_num}: ({x:.2f}cm, {y:.2f}cm)"
                for obj_num, x, y in detections
            ]
            detection_text = "Detected objects (in CM coordinates):\n" + "\n".join(
                detection_strings
            )
        else:
            detection_text = "No objects detected"

        # Prompt: action-based system with coordinates
        prompt_text = f"""
You are DUM-E, a robotic arm assistant. Your job is to analyze the image and context, and determine the next action and coordinates needed to complete the user's request.

USER REQUEST (text): {user_prompt}
DETECTION CONTEXT (text): {detection_text}

INSTRUCTIONS:
- If there are multiple things to do (like multiple objects), just do the next one you think is needed. Don't keep doing the same one over and over. Just eyeball it by seeing the picture.
- Be lenient: if the object is anywhere close to the intended area or quadrant, and it looks "about right" or "good enough", then the task is complete. Don't require precision—if it looks like the goal is basically achieved, set "task_complete": true and explain in "task_description".
- If the user request is ambiguous or already satisfied, or if the objects are already in a reasonable position, mark the task as complete.
- If you are on a later step and can infer what happened before (even if you didn't see the first request), use your best judgment based on the current image and context.
- The computer vision system has already detected and localized all objects for you. You do NOT need to do any image analysis or object detection yourself, but you should use the image to help you decide if the task is done or if the object is in the right place.
- Decide what the arm should do next, one action at a time, until the task is complete. After each action, a new image and updated context will be provided.
- If the user wants multiple objects moved, do them one by one, but don't just keep doing the same one. Pick the next one that makes sense.
- If the object is in the general area or quadrant described by the user (for example, "top right" means the upper right quarter of the workspace), and it looks "good enough" or "about right", then the task is complete. Do not require precision—if it looks like the goal is basically achieved, set "task_complete": true and explain in "task_description".
- If the user request is ambiguous or already satisfied, or if the objects are already in a reasonable position, mark the task as complete.

IMPORTANT:
- Only return valid JSON as plain text in the format below. Do NOT include any explanation, reasoning, or extra text.
- Do NOT output any step-by-step thinking or commentary.
- If the task is already complete or cannot be performed, set "task_complete": true and provide an explanation in the "task_description" or "error" field.
- If you do not understand the request, respond with: {{"task_complete": true, "error": "Cannot understand request"}}
- Be flexible and human-like in your interpretation, and err on the side of considering the task complete if the object is anywhere close to the area the user described. Only if the object is clearly far from the intended area should the task be considered incomplete.

COORDINATE SYSTEM (IMPORTANT!):
- X: left/right (0 = far left, 30 = far right), in centimeters (cm)
- Y: forward/backward (0 = closest to arm base, 30 = farthest away), in centimeters (cm)
- Z: always 0 (table surface level)
- Workspace grid: (0,0) is bottom left, (30,30) is top right.
- DO NOT use values below 2 or above 28 for X or Y (avoid edge cases).
- All coordinates must be in centimeters (cm), not millimeters.

AVAILABLE ACTIONS:
- "grab": Pick up an object and move it to drop-off location (requires x, y, phi, x2, y2)
- "move": Move to coordinates with orientation (requires x, y, z, phi, claw_open, roll_angle)
- "move_to_idle": Move to safe idle position (no coordinates needed)
- "wave_bye": Wave goodbye gesture (no coordinates needed)
- "shake_yes": Nod yes gesture (no coordinates needed)
- "shake_no": Shake no gesture (no coordinates needed)
- "shake_hand": Handshake gesture (no coordinates needed)
- "move_to_hold": Move to hold position (requires x, y)
- "hold": Hold at position (requires x, y)
- "drop_off": Drop object at coordinates (requires x, y, z, phi)

RESPONSE FORMAT (JSON as plain text only, no markdown, no code block, no explanation):
{{
    "action": "grab",                // Action to perform (see AVAILABLE ACTIONS above)
    "x": float,                      // X coordinate in cm (required for most actions)
    "y": float,                      // Y coordinate in cm (required for most actions)
    "z": float,                      // Z coordinate in cm (optional, defaults to 0)
    "phi": float,                    // Claw orientation in degrees (270=top-down, 0=side approach)
    "x2": float,                     // Drop-off X coordinate in cm (required for grab action)
    "y2": float,                     // Drop-off Y coordinate in cm (required for grab action)
    "claw_open": 1,                  // 1 for open, 0 for closed (optional, defaults to 1)
    "roll_angle": float,             // Roll angle in degrees (optional, defaults to 0)
    "elbow": "up",                   // "up" or "down" (optional, defaults to "up")
    "task_description": "Brief description of what the arm will do",
    "task_complete": false           // Set to true only when the entire task is finished or cannot be done. Use reasonable, human-like judgment.
}}

REMEMBER:
- Only output valid JSON as plain text in the specified format.
- Do not include any explanation, reasoning, or extra text.
- Be lenient and human-like: If the object is anywhere close to the general area/quadrant the user described, that's good enough and the task is complete. Only if it's clearly far from the intended area should the task be considered incomplete.
- If there are multiple things to do, do the next one that makes sense (don't just keep doing the same one).
- All coordinates must be in centimeters (cm), and avoid using values near 0 or 30 for X and Y (stay between 2 and 28).
- Z should always be 0.
- You don't need to be accurate—close enough is good enough!
- Choose the appropriate action based on what needs to be done (grab for picking up, move for moving, gestures for communication, etc.)
- For grab action: Use phi=270 for top-down approach (default), phi=0 for side approach
- For grab action: Always specify x2, y2 for where to drop the object after picking it up
"""

        # Make the API call
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            temperature=0.2,  # Low temperature for precise coordinates
            max_completion_tokens=512,
            top_p=0.8,
            stream=False,
        )

        response = completion.choices[0].message.content
        print("DUM-E Analysis:")
        print(response)

        # Parse the response
        coordinate_dict = parse_groq_response(response)

        if coordinate_dict is None:
            print("Failed to parse coordinate dictionary from Groq response")
            return None

        return coordinate_dict

    except Exception as e:
        print(f"Error in Groq analysis: {e}")
        return None


def send_to_arm_control(action_dict, arm_server_url=None):
    """Send instructions to the arm control server"""
    if arm_server_url is None:
        arm_server_url = ARM_SERVER_URL
    try:
        print(f"ARM CONTROL INSTRUCTIONS:")
        print(f"   Action: {action_dict.get('action', 'N/A')}")
        print(f"   X: {action_dict.get('x', 'N/A')}")
        print(f"   Y: {action_dict.get('y', 'N/A')}")
        print(f"   Z: {action_dict.get('z', 'N/A')}")
        print(f"   Phi: {action_dict.get('phi', 'N/A')}")
        print(f"   Claw Open: {action_dict.get('claw_open', 'N/A')}")
        print(f"   Roll Angle: {action_dict.get('roll_angle', 'N/A')}")
        print(f"   Elbow: {action_dict.get('elbow', 'N/A')}")
        print(f"   Task: {action_dict.get('task_description', 'N/A')}")
        print(f"   Complete: {action_dict.get('task_complete', False)}")

        # Extract action and parameters
        action = action_dict.get("action", "").lower()

        # Build the request data based on the action
        request_data = {"action": action}

        # Add coordinates and parameters based on action requirements
        if action in ["grab", "move", "move_to_hold", "hold", "drop_off"]:
            if "x" in action_dict:
                request_data["x"] = float(action_dict["x"])
            if "y" in action_dict:
                request_data["y"] = float(action_dict["y"])

        if action in ["move", "drop_off"]:
            if "z" in action_dict:
                request_data["z"] = float(action_dict["z"])

        if action in ["grab", "move", "drop_off"]:
            if "phi" in action_dict:
                request_data["phi"] = float(action_dict["phi"])

        # Special handling for grab action - needs drop-off coordinates
        if action == "grab":
            if "x2" in action_dict:
                request_data["x2"] = float(action_dict["x2"])
            if "y2" in action_dict:
                request_data["y2"] = float(action_dict["y2"])

        if action == "move":
            if "claw_open" in action_dict:
                request_data["claw_open"] = int(action_dict["claw_open"])
            if "roll_angle" in action_dict:
                request_data["roll_angle"] = float(action_dict["roll_angle"])
            if "elbow" in action_dict:
                request_data["elbow"] = action_dict["elbow"]

        print(f"Executing {action.upper()} action...")
        print(f"Request data: {request_data}")

        # Send the request to the arm control server
        response = requests.post(
            f"{arm_server_url}/arm_control", json=request_data, timeout=10
        )

        if response.status_code != 200:
            print(f"Action failed: {response.json()}")
            return False

        print(f"Action response: {response.json()}")
        print("Arm control completed successfully")
        return True

    except requests.exceptions.ConnectionError:
        print(
            f"Error: Could not connect to arm control server. Make sure it's running on {arm_server_url}"
        )
        return False
    except Exception as e:
        print(f"Error in arm control: {e}")
        return False


def check_arm_server(arm_server_url=None):
    """Check if the arm control server is running"""
    if arm_server_url is None:
        arm_server_url = ARM_SERVER_URL
    try:
        response = requests.get(f"{arm_server_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"Arm server status: {data.get('status', 'unknown')}")
            print(f"Arm ready: {data.get('arm_ready', False)}")
            return data.get("arm_ready", False)
        else:
            print(f"Arm server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(
            f"Arm server is not running. Please start it with: python arm_control_server.py (URL: {arm_server_url})"
        )
        return False
    except Exception as e:
        print(f"Error checking arm server: {e}")
        return False


def execute_task(user_prompt, camera_id=0):
    """Main function to execute a complete arm task (lenient, multi-object, human-like)"""
    print("=" * 60)
    print("DUM-E Robotic Arm System")
    print("=" * 60)
    print(f"User Request: {user_prompt}")
    print("=" * 60)

    # Check if arm server is running
    print("\nChecking arm control server...")
    if not check_arm_server():
        print("Warning: Arm server is not ready. Continuing with simulation mode...")
        print(
            "To use real arm control, start the server with: python arm_control_server.py"
        )
        print("=" * 60)

    try:
        # Step 1: Capture image with detection
        print("\nCapturing image with object detection...")
        image_path, detections = capture_with_detection(camera_id)

        if image_path is None:
            print("Failed to capture image")
            return False

        # Step 2: Analyze with Groq
        print("\nAnalyzing scene with Groq...")
        coordinate_dict = analyze_with_groq(image_path, user_prompt, detections)

        if coordinate_dict is None:
            print("Failed to get coordinates from Groq")
            return False

        # Step 3: Execute arm movements in a loop
        print("\nStarting arm execution loop...")
        task_complete = False
        iteration = 0
        max_iterations = 10  # Safety limit

        while not task_complete and iteration < max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # Send coordinates to arm control
            success = send_to_arm_control(coordinate_dict)

            if not success:
                print("Arm control failed, stopping task")
                return False

            # Check if task is complete
            task_complete = coordinate_dict.get("task_complete", False)
            print(f" Task complete status: {task_complete}")
            print(
                f" Task description: {coordinate_dict.get('task_description', 'No description')}"
            )

            if task_complete:
                print("Task completed successfully!")
                break

            # Wait a bit before next iteration (shorter wait for demo)
            time.sleep(3)

            # Re-analyze if task is not complete
            if not task_complete and iteration < max_iterations:
                print("Re-capturing scene for next step...")
                image_path, detections = capture_with_detection(camera_id)

                if image_path is None:
                    print("Failed to re-capture, stopping task")
                    return False

                print("Re-analyzing for next step...")
                coordinate_dict = analyze_with_groq(image_path, user_prompt, detections)

                if coordinate_dict is None:
                    print("Failed to re-analyze, stopping task")
                    return False

        if iteration >= max_iterations:
            print("Reached maximum iterations, stopping task")

        return task_complete

    except Exception as e:
        print(f"Error executing task: {e}")
        return False


def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print('Usage: python main.py "your task description"')
        print(
            'Example: python main.py "Pick up the red object and move it to the left"'
        )
        sys.exit(1)

    user_prompt = sys.argv[1]

    # Check for Groq API key
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set")
        print("Please set your Groq API key: export GROQ_API_KEY=your_key_here")
        sys.exit(1)

    success = execute_task(user_prompt)

    if success:
        print("\nTask completed successfully!")
    else:
        print("\nTask failed or was incomplete")


if __name__ == "__main__":
    main()
