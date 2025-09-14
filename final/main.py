#!/usr/bin/env python3
"""
Simplified DUM-E robotic arm system
Takes CLI prompt, captures image with detection, sends to Groq, loops until task complete
"""

import sys
import time
import json
import base64
import os
from groq import Groq
from dotenv import load_dotenv
from capture_detection import capture_with_detection

load_dotenv()


def encode_image(image_path):
    """Encode image to base64 for Groq API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_groq_response(response_text):
    """Parse Groq response to extract coordinate dictionary"""
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


def analyze_with_groq(image_path, user_prompt, detections):
    """Analyze image with Groq and return coordinate dictionary"""
    try:
        # Encode the image
        base64_image = encode_image(image_path)

        # Create the client
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # Format detections for context - these are already in CM coordinates from homography_cv.py
        detection_text = ""
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

        # Comprehensive prompt for coordinate-based actions
        prompt_text = f"""
You are DUM-E, a robotic arm assistant. Analyze the image and determine the coordinates needed to complete the user's request.

USER REQUEST: {user_prompt}
DETECTION CONTEXT: {detection_text}

INSTRUCTIONS:
You are DUM-E, a highly capable robotic arm assistant. The computer vision (CV) pipeline has already detected and localized all objects in the scene, and you are provided with their bounding boxes and coordinates. You do not need to do any image analysis or object detection yourself—just use the information given.

Your job is to decide what the arm should do next, one action at a time, until the task is complete. After each action, a new image and updated context will be provided.

IMPORTANT:
- Only return valid JSON in the specified format below. Do not include any explanation, reasoning, or extra text.
- Do not output any step-by-step thinking or commentary.
- If the task is already complete or cannot be performed, set "task_complete": true and provide an explanation in the "task_description" or "error" field.
- If you do not understand the request, respond with: {{"task_complete": true, "error": "Cannot understand request"}}
- BE FLEXIBLE WITH PRECISION: You don't need to be super precise with coordinates. Close enough is good enough for most tasks.

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

COORDINATE SYSTEM (IMPORTANT!):
- X: left/right (0 = far left, 30 = far right), in centimeters (cm)
- Y: forward/backward (0 = closest to arm base, 30 = farthest away), in centimeters (cm)
- Z: just use 0 (table surface level)
- The workspace grid is (0,0) at the bottom left and (30,30) at the top right.
- DO NOT use values below 2 or above 28 for X or Y (avoid edge cases).
- Z should always be 0 - don't worry about height.
- All coordinates should be in centimeters (cm), not millimeters.
- BE FLEXIBLE: You don't need to be super accurate - close enough is good enough!

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


def send_to_arm_control(coordinate_dict):
    """Mock arm control function - just prints the instructions"""
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


def execute_task(user_prompt, camera_id=0):
    """Main function to execute a complete arm task"""
    print("=" * 60)
    print("DUM-E Robotic Arm System")
    print("=" * 60)
    print(f"User Request: {user_prompt}")
    print("=" * 60)

    try:
        # Step 1: Capture image with detection
        print("\n📸 Capturing image with object detection...")
        image_path, detections = capture_with_detection(camera_id)

        if image_path is None:
            print("❌ Failed to capture image")
            return False

        # Step 2: Analyze with Groq
        print("\n🧠 Analyzing scene with Groq...")
        coordinate_dict = analyze_with_groq(image_path, user_prompt, detections)

        if coordinate_dict is None:
            print("❌ Failed to get coordinates from Groq")
            return False

        # Step 3: Execute arm movements in a loop
        print("\n🤖 Starting arm execution loop...")
        task_complete = False
        iteration = 0
        max_iterations = 10  # Safety limit

        while not task_complete and iteration < max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # Send coordinates to arm control
            success = send_to_arm_control(coordinate_dict)

            if not success:
                print("❌ Arm control failed, stopping task")
                return False

            # Check if task is complete
            task_complete = coordinate_dict.get("task_complete", False)

            if task_complete:
                print("✅ Task completed successfully!")
                break

            # Wait a bit before next iteration
            time.sleep(2)

            # Re-analyze if task is not complete
            if not task_complete and iteration < max_iterations:
                print("📸 Re-capturing scene for next step...")
                image_path, detections = capture_with_detection(camera_id)

                if image_path is None:
                    print("❌ Failed to re-capture, stopping task")
                    return False

                print("🧠 Re-analyzing for next step...")
                coordinate_dict = analyze_with_groq(image_path, user_prompt, detections)

                if coordinate_dict is None:
                    print("❌ Failed to re-analyze, stopping task")
                    return False

        if iteration >= max_iterations:
            print("⚠️ Reached maximum iterations, stopping task")

        return task_complete

    except Exception as e:
        print(f"❌ Error executing task: {e}")
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
        print("❌ Error: GROQ_API_KEY environment variable not set")
        print("Please set your Groq API key: export GROQ_API_KEY=your_key_here")
        sys.exit(1)

    success = execute_task(user_prompt)

    if success:
        print("\n🎉 Task completed successfully!")
    else:
        print("\n💥 Task failed or was incomplete")


if __name__ == "__main__":
    main()
