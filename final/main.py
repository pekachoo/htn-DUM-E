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

        # Simple prompt
        prompt_text = f"""
You are DUM-E, a robotic arm. Look at the image and do what the user asks.

USER REQUEST: {user_prompt}
DETECTED OBJECTS: {detection_text}

COORDINATE SYSTEM: X=0-30cm (left-right), Y=0-30cm (front-back), Z=0 (table level)

AVAILABLE ACTIONS:
- "grab": Pick up object at (x,y) and move to (x2,y2) - use phi=270 for top-down
- "move": Move to (x,y,z) with orientation
- "move_to_hold": Move to hold position at (x,y)
- "hold": Hold at position (x,y)
- "wave_bye": Wave goodbye
- "shake_yes": Nod yes  
- "shake_no": Shake no
- "shake_hand": Handshake
- "move_to_idle": Go to safe position

RESPONSE (JSON only):
{{
    "action": "grab",
    "x": 15.0,
    "y": 10.0, 
    "phi": 270,
    "x2": 25.0,
    "y2": 20.0,
    "task_description": "What I'm doing"
}}
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
    """Send action to arm control server"""
    if arm_server_url is None:
        arm_server_url = ARM_SERVER_URL

    try:
        print(f"Sending action: {action_dict.get('action')}")
        response = requests.post(
            f"{arm_server_url}/arm_control", json=action_dict, timeout=10
        )

        if response.status_code == 200:
            print("Action completed successfully")
            return True
        else:
            print(f"Action failed: {response.json()}")
            return False

    except requests.exceptions.ConnectionError:
        print("Cannot connect to arm server")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def execute_task(user_prompt, camera_id=0):
    """Simple task execution: capture image, analyze with LLM, execute arm action"""
    print("=" * 60)
    print("DUM-E Robotic Arm System")
    print("=" * 60)
    print(f"User Request: {user_prompt}")
    print("=" * 60)

    try:
        # Step 1: Capture image
        print("Capturing image...")
        image_path, detections = capture_with_detection(camera_id)
        if image_path is None:
            print("Failed to capture image")
            return False

        # Step 2: Analyze with LLM
        print("Analyzing with LLM...")
        action_dict = analyze_with_groq(image_path, user_prompt, detections)
        if action_dict is None:
            print("Failed to get action from LLM")
            return False

        # Step 3: Execute arm action
        print("Executing arm action...")
        success = send_to_arm_control(action_dict)
        if success:
            print("Task completed!")
        return success

    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print('Usage: python main.py "your task"')
        print('Example: python main.py "wave goodbye"')
        sys.exit(1)

    user_prompt = sys.argv[1]
    success = execute_task(user_prompt)

    if success:
        print("Done!")
    else:
        print("Failed!")


if __name__ == "__main__":
    main()
