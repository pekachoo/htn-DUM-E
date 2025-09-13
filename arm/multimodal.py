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


def get_image_from_flask(user_prompt):
    """
    Get image and additional text from Flask endpoint
    Returns: (image_saved_path, additional_text)
    """
    try:
        print(f"📡 Requesting image from Flask endpoint with prompt: '{user_prompt}'")

        response = requests.get(FLASK_IMAGE_ENDPOINT, params={"prompt": user_prompt})
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
        prompt_text = f"""You are DUM-E, a robotic arm assistant. Analyze the image and determine the exact coordinates needed to complete the user's request.

USER REQUEST: {user_prompt}
ADDITIONAL CONTEXT: {additional_text}

TASK: Based on the image and request, provide a JSON response with the exact coordinates and parameters needed for the robotic arm to complete the task.

RESPONSE FORMAT (JSON only):
{{
    "in_coord": [x, y, z],
    "out_coord": [x, y, z], 
    "direction": [x, y, z],
    "yaw": float,
    "pitch": float,
    "roll": float,
    "gripper_action": "open" or "close",
    "task_description": "Brief description of what the arm will do",
    "task_complete": false
}}

COORDINATE SYSTEM:
- X: left/right (negative = left, positive = right)
- Y: forward/backward (negative = backward, positive = forward)  
- Z: up/down (negative = down, positive = up)
- All coordinates in millimeters
- Origin (0,0,0) is at the base of the arm

IMPORTANT:
- Only respond with valid JSON
- If the task is complete or cannot be done, set "task_complete": true
- Be precise with coordinates based on what you see in the image
- If you cannot understand the request, respond with: {{"task_complete": true, "error": "Cannot understand request"}}"""

        # Make the API call
        completion = client.chat.completions.create(
            model="llava-v1.5-7b-4096-preview",
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
