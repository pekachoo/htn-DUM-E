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


def get_image_from_flask():
    """
    Get image and additional text from Flask endpoint
    Returns: (image_saved_path, additional_text)
    """
    try:
        print(f"📡 Requesting image from Flask endpoint")

        response = requests.get(FLASK_IMAGE_ENDPOINT)
        response.raise_for_status()

        data = response.json()

        # Save the image locally
        image_data = base64.b64decode(data["image"])
        with open(LOCAL_IMAGE_PATH, "wb") as f:
            f.write(image_data)

        print(f"✅ Image saved to {LOCAL_IMAGE_PATH}")
        return LOCAL_IMAGE_PATH, data.get("text", "")

    except Exception as e:
        print(f"❌ Error getting image from Flask: {e}")
        raise


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

        print("❌ No valid JSON found in Groq response")
        return None

    except Exception as e:
        print(f"❌ Error parsing Groq response: {e}")
        return None


def send_to_arm_control(coordinate_dict):
    """
    Send coordinate dictionary to arm control endpoint
    Returns: True if successful, False otherwise
    """
    try:
        print(f"🤖 Sending coordinates to arm: {coordinate_dict}")

        response = requests.post(
            FLASK_ARM_ENDPOINT,
            json=coordinate_dict,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        result = response.json()
        success = result.get("success", False)

        if success:
            print("✅ Arm action completed successfully")
        else:
            print(f"⚠️ Arm action status: {result.get('message', 'Unknown status')}")

        return success

    except Exception as e:
        print(f"❌ Error sending to arm control: {e}")
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
        
You are DUM-E, a robotic arm assistant. Analyze the image and determine the exact coordinates needed to complete the user's request.

USER REQUEST: {user_prompt}
ADDITIONAL CONTEXT: {additional_text}

INSTRUCTIONS:
You are DUM-E, a highly capable robotic arm assistant. Your job is to take the user's natural language request and, using the provided image and additional context, reason step-by-step about how to accomplish the task using the robotic arm. The computer vision (CV) pipeline has already detected and localized all objects in the scene, and you are provided with their bounding boxes and coordinates (from the flask_get_image endpoint). You do not need to do any image analysis or object detection yourself—just use the information given.

Your main responsibility is to plan and decide what the arm should do next, one action at a time, until the task is complete. After each action, a new image and updated context will be provided, and you will repeat the process until you determine the task is finished.

EXAMPLES:
- If the user asks to "sort the objects into two bins by color," you should:
    1. Identify all objects and their colors using the provided context.
    2. For each object, decide which bin it belongs to and generate a pick-and-place action for that object (e.g., pick up object 1 at [x1, y1, z1], move to bin A at [x2, y2, z2], and release).
    3. Only output one action at a time (e.g., pick and place for a single object), then wait for the next image/context update before proceeding to the next object.
    4. When all objects are sorted, set "task_complete": true.

- If the user asks to "play a specific song," you have access to a tool that provides the first 10 notes of the song. Use this information to reason about which keys to press, and generate actions for the arm to press the correct keys in sequence, one at a time.

GENERAL GUIDELINES:
- Use the provided object coordinates and context to plan the arm's actions.
- You are responsible for all high-level reasoning and planning. The CV and inverse kinematics (IK) systems will handle the low-level movement and perception.
- For each step, output a single action (e.g., pick up an object, move to a location, press a key, etc.).
- After each action, a new image and updated context will be provided. Continue until the task is complete.
- If the task is already complete or cannot be performed, set "task_complete": true and provide an explanation in the "task_description" or "error" field.
- If you do not understand the request, respond with: {"task_complete": true, "error": "Cannot understand request"}

RESPONSE FORMAT (JSON only):
{{
    "in_coord": [x, y, z],           // Where to pick up or start the action
    "out_coord": [x, y, z],          // Where to place, move, or end the action
    "direction": [x, y, z],          // Direction of movement or approach vector
    "yaw": float,                    // Yaw angle for the end effector
    "pitch": float,                  // Pitch angle for the end effector
    "roll": float,                   // Roll angle for the end effector
    "gripper_action": "open" or "close", // Whether to open or close the gripper
    "task_description": "Brief description of what the arm will do",
    "task_complete": false           // Set to true only when the entire task is finished or cannot be done
}}

COORDINATE SYSTEM:
- X: left/right (negative = left, positive = right)
- Y: forward/backward (negative = backward, positive = forward)
- Z: up/down (negative = down, positive = up)
- All coordinates are in millimeters
- The origin (0,0,0) is at the base of the arm

REMEMBER:
- Only output valid JSON in the specified format.
- Plan and reason about the task step-by-step, one action at a time.
- Use the provided context and coordinates for all decisions.
- Wait for new context after each action before proceeding."""

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
            print("❌ Failed to parse coordinate dictionary from Groq response")
            return None

        return coordinate_dict

    except Exception as e:
        print(f"❌ Error in Groq analysis: {e}")
        return None


def execute_arm_task(user_prompt):
    """
    Main function to execute a complete arm task
    """
    print("🤖 DUM-E Robotic Arm Task Executor")
    print("=" * 50)
    print(f"📝 User Request: {user_prompt}")
    print("=" * 50)

    try:
        # Step 1: Get image from Flask endpoint
        image_path, additional_text = get_image_from_flask(user_prompt)

        # Step 2: Analyze with Groq (single call)
        print("\n🔍 Analyzing image with Groq...")
        coordinate_dict = analyze_with_groq(image_path, user_prompt, additional_text)

        if coordinate_dict is None:
            print("❌ Failed to get coordinates from Groq")
            return False

        # Step 3: Execute arm movements in a loop
        print("\n🔄 Starting arm execution loop...")
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
            time.sleep(1)

            # Re-analyze if task is not complete (optional - for complex tasks)
            if not task_complete and iteration < max_iterations:
                print("🔄 Re-analyzing for next step...")
                coordinate_dict = analyze_with_groq(
                    image_path, user_prompt, additional_text
                )
                if coordinate_dict is None:
                    print("❌ Failed to re-analyze, stopping task")
                    return False

        if iteration >= max_iterations:
            print("⚠️ Reached maximum iterations, stopping task")

        return task_complete

    except Exception as e:
        print(f"❌ Error executing arm task: {e}")
        return False


# Main execution
if __name__ == "__main__":
    # Example usage
    test_prompts = [
        "Pick up the red apple and move it to the left corner",
        "Sort the objects by color - red to left, blue to right",
        "Wave to the camera",
        "Pick up the tool and hold it steady",
    ]

    # Use the first test prompt or get user input
    user_prompt = test_prompts[0]  # Change this or make it interactive

    success = execute_arm_task(user_prompt)

    if success:
        print("\n🎉 Task completed successfully!")
    else:
        print("\n💥 Task failed or was incomplete")
