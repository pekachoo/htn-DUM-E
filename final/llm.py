from groq import Groq
import base64
import os
import json
import requests
import time
from dotenv import load_dotenv

load_dotenv()

# Configuration
FLASK_IMAGE_ENDPOINT = (
    "http://localhost:5000/get_image"  # Change this to your Flask endpoint
)
FLASK_ARM_ENDPOINT = (
    "http://localhost:5000/arm_control"  # Change this to your arm control endpoint
)
LOCAL_IMAGE_PATH = "current_image.jpg"  # Local path to save the image


def encode_image(image_path):
    """Encode image to base64 for Groq API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_groq_response(response_text):
    """
    Parse Groq response to extract coordinate dictionary
    Expected format: JSON with in_coord, out_coord, direction, yaw, etc.
    """
    try:
        # Look for JSON in the response
        lines = response_text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        # If no JSON found, try to extract from the entire response
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


def send_to_arm_control(coordinate_dict):
    """
    Mock arm control function - just prints the instructions
    Returns: True if successful, False otherwise
    """
    try:
        print(f"🤖 ARM CONTROL INSTRUCTIONS:")
        print(f"   Pick up at: {coordinate_dict.get('in_coord', 'N/A')}")
        print(f"   Place at: {coordinate_dict.get('out_coord', 'N/A')}")
        print(f"   Direction: {coordinate_dict.get('direction', 'N/A')}")
        print(f"   Gripper: {coordinate_dict.get('gripper_action', 'N/A')}")
        print(f"   Yaw: {coordinate_dict.get('yaw', 'N/A')}")
        print(f"   Pitch: {coordinate_dict.get('pitch', 'N/A')}")
        print(f"   Roll: {coordinate_dict.get('roll', 'N/A')}")
        print(f"   Task: {coordinate_dict.get('task_description', 'N/A')}")
        print(f"   Complete: {coordinate_dict.get('task_complete', False)}")
        print("✅ Arm control simulation completed successfully")
        return True

    except Exception as e:
        print(f"Error in arm control simulation: {e}")
        return False


def analyze_with_groq(image_path, user_prompt, additional_text=""):
    """
    Analyze image with Groq and return coordinate dictionary
    """
    try:
        # Encode the image
        base64_image = encode_image(image_path)

        # Create the client
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # Comprehensive prompt for coordinate-based actions
        prompt_text = f"""
You are DUM-E, a robotic arm assistant. Analyze the image and determine the coordinates needed to complete the user's request.

USER REQUEST: {user_prompt}
ADDITIONAL CONTEXT: {additional_text}

INSTRUCTIONS:
You are DUM-E, a highly capable robotic arm assistant. The computer vision (CV) pipeline has already detected and localized all objects in the scene, and you are provided with their bounding boxes and coordinates. You do not need to do any image analysis or object detection yourself—just use the information given.

Your job is to decide what the arm should do next, one action at a time, until the task is complete. After each action, a new image and updated context will be provided.

IMPORTANT:
- Only return valid JSON in the specified format below. Do not include any explanation, reasoning, or extra text.
- Do not output any step-by-step thinking or commentary.
- If the task is already complete or cannot be performed, set "task_complete": true and provide an explanation in the "task_description" or "error" field.
- If you do not understand the request, respond with: {{"task_complete": true, "error": "Cannot understand request"}}
- BE FLEXIBLE WITH PRECISION: You don't need to be super precise with coordinates. Close enough is good enough for most tasks.

COORDINATE SYSTEM (IMPORTANT!):
- X: left/right (0 = far left, 30 = far right), in centimeters (cm)
- Y: forward/backward (0 = closest to arm base, 30 = farthest away), in centimeters (cm)
- Z: just use 0 (table surface level)
- The workspace grid is (0,0) at the bottom left and (30,30) at the top right.
- DO NOT use values below 2 or above 28 for X or Y (avoid edge cases).
- Z should always be 0 - don't worry about height.
- All coordinates should be in centimeters (cm), not millimeters.
- BE FLEXIBLE: You don't need to be super accurate - close enough is good enough!

RESPONSE FORMAT (JSON only):
{{
    "in_coord": [x, y, 0],           // Where to pick up or start the action (in cm, Z=0, avoid edge values)
    "out_coord": [x, y, 0],          // Where to place, move, or end the action (in cm, Z=0, avoid edge values)
    "direction": [x, y, 0],          // Direction of movement or approach vector (in cm, Z=0, can be approximate)
    "yaw": float,                    // Yaw angle for the end effector (approximate is fine)
    "pitch": float,                  // Pitch angle for the end effector (approximate is fine)
    "roll": float,                   // Roll angle for the end effector (approximate is fine)
    "gripper_action": "open" or "close", // Whether to open or close the gripper
    "task_description": "Brief description of what the arm will do",
    "task_complete": false           // Set to true only when the entire task is finished or cannot be done
}}

REMEMBER:
- Only output valid JSON in the specified format.
- Do not include any explanation, reasoning, or extra text.
- Be flexible with precision - close enough is good enough.
- All coordinates must be in centimeters (cm), and avoid using values near 0 or 30 for X and Y (stay between 2 and 28).
- Z should always be 0 - don't worry about height.
- You don't need to be super accurate - close enough is good enough!
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
        print("🧠 DUM-E Analysis:")
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
