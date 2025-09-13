from groq import Groq
import base64
import os
import json
from dotenv import load_dotenv

load_dotenv()


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Function to make the arm say "no" for out-of-scope requests
def arm_say_no():
    # This will be implemented to call an endpoint that makes the arm gesture "no"
    print(
        "DUM-E: *shakes head* I don't understand that request yet. I can only help with sorting tasks for now!"
    )
    return {"action": "no", "message": "Out of scope request"}


# Function to parse LLM response and extract sorting instructions
def parse_sorting_response(response_text):
    """
    Parse the LLM response to extract sorting instructions
    Returns a list of tuples: (object_description, position)
    """
    try:
        # Look for the structured response pattern
        lines = response_text.split("\n")
        sorting_instructions = []

        for line in lines:
            if "->" in line and ("left" in line.lower() or "right" in line.lower()):
                # Extract object and position
                parts = line.split("->")
                if len(parts) == 2:
                    object_desc = parts[0].strip()
                    position = parts[1].strip().lower()
                    sorting_instructions.append((object_desc, position))

        return sorting_instructions
    except Exception as e:
        print(f"Error parsing response: {e}")
        return []


# Main sorting function
def perform_sorting_analysis(image_path, user_request=""):
    """
    Analyze an image and provide binary sorting instructions for DUM-E
    """
    try:
        # Encode the image
        base64_image = encode_image(image_path)

        # Create the client
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # Comprehensive prompt for binary sorting
        prompt_text = f"""You are DUM-E, a helpful robotic arm assistant. Your primary function is binary sorting - organizing objects into two categories based on user requests.

USER REQUEST: {user_request if user_request else "Please sort the objects in this image into two groups"}

TASK: Analyze the image and identify all visible objects. For each object, determine which of two positions it should be moved to:
- LEFT CORNER: For objects that match the first category
- RIGHT CORNER: For objects that match the second category

RESPONSE FORMAT:
For each object, provide exactly this format:
[Object description] -> [LEFT CORNER or RIGHT CORNER]

REASONING: After listing all objects, provide a brief explanation of your sorting criteria.

EXAMPLE:
Red apple -> LEFT CORNER
Blue pen -> RIGHT CORNER
Green book -> LEFT CORNER

REASONING: I sorted by color - red and green objects go to the left corner, blue objects go to the right corner.

IMPORTANT: 
- Only provide sorting instructions for objects you can clearly see
- Use LEFT CORNER and RIGHT CORNER exactly as written
- If you cannot understand the user's request or it's outside sorting scope, respond with: "SCOPE_ERROR: This request is outside my sorting capabilities"
- Be specific in object descriptions (color, size, type, etc.)"""

        # Make the API call
        completion = client.chat.completions.create(
            model="llava-v1.5-7b-4096-preview",  # Correct Groq vision model
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
            temperature=0.3,  # Lower temperature for more consistent sorting
            max_completion_tokens=1024,
            top_p=0.9,
            stream=False,
        )

        response = completion.choices[0].message.content
        print("DUM-E Analysis:")
        print(response)

        # Check if the request is out of scope
        if "SCOPE_ERROR:" in response:
            print("\n" + "=" * 50)
            arm_say_no()
            return {"status": "out_of_scope", "message": response}

        # Parse the sorting instructions
        sorting_instructions = parse_sorting_response(response)

        if sorting_instructions:
            print(f"\n" + "=" * 50)
            print("SORTING INSTRUCTIONS FOR DUM-E:")
            for i, (obj, pos) in enumerate(sorting_instructions, 1):
                print(f"{i}. Move {obj} to {pos}")

            return {
                "status": "success",
                "instructions": sorting_instructions,
                "raw_response": response,
            }
        else:
            print("\n" + "=" * 50)
            print("Could not parse sorting instructions from response")
            return {"status": "parse_error", "message": response}

    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found")
        return {"status": "error", "message": "Image file not found"}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}


# Main execution
if __name__ == "__main__":
    # Default image path - change this to your actual image
    image_path = "image.png"

    # Example usage with different sorting requests
    print("🤖 DUM-E Robotic Arm Sorting System")
    print("=" * 50)

    # Test with different sorting criteria
    test_requests = [
        "Sort by color - red objects to left, blue objects to right",
        "Sort by size - large objects to left, small objects to right",
        "Sort by type - tools to left, other objects to right",
        "Sort the objects into two groups based on what makes sense",
    ]

    # Use the first test request or allow custom input
    user_request = test_requests[0]  # Change this or make it interactive

    result = perform_sorting_analysis(image_path, user_request)

    if result["status"] == "success":
        print(
            f"\n✅ Successfully generated {len(result['instructions'])} sorting instructions"
        )
    elif result["status"] == "out_of_scope":
        print("\n❌ Request is outside DUM-E's current capabilities")
    else:
        print(f"\n❌ Error: {result['message']}")
