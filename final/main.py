#!/usr/bin/env python3
"""
Main integration script for DUM-E robotic arm system
Combines capture_detection.py with llm.py for complete pipeline
"""

import time
import os
import sys

# Try to import required packages with error handling
try:
    import cv2

    print("✓ OpenCV imported successfully")
except ImportError as e:
    print(f"✗ Failed to import OpenCV: {e}")
    print("Please install opencv-python: pip install opencv-python")
    sys.exit(1)

try:
    from capture_detection import DetectionCapture

    print("✓ DetectionCapture imported successfully")
except ImportError as e:
    print(f"✗ Failed to import DetectionCapture: {e}")
    sys.exit(1)

try:
    from llm import analyze_with_groq, send_to_arm_control, parse_groq_response

    print("✓ LLM functions imported successfully")
except ImportError as e:
    print(f"✗ Failed to import LLM functions: {e}")
    print("Please install required packages: pip install groq python-dotenv requests")
    sys.exit(1)


class DUMESystem:
    def __init__(self, camera_id=0):
        """
        Initialize the DUM-E system with camera and detection pipeline
        """
        self.capture_system = DetectionCapture()
        self.camera_id = camera_id
        self.current_image_path = "current_task_image.jpg"
        self.user_prompt = ""  # Store the original user prompt

    def initialize(self):
        """
        Initialize the camera and detection system
        """
        print("Initializing DUM-E System...")

        if not self.capture_system.initialize_camera(self.camera_id):
            print("Failed to initialize camera")
            return False

        print("DUM-E System initialized successfully!")
        return True

    def capture_current_scene(self):
        """
        Capture current scene with object detection
        Returns: (image_path, detections, formatted_text)
        """
        print("\nCapturing current scene...")

        # Capture frame with detection
        warped_image, detections = self.capture_system.capture_frame_with_detection()

        if warped_image is None:
            print("Failed to capture scene")
            return None, [], "No data available"

        # Save the image
        try:
            cv2.imwrite(self.current_image_path, warped_image)
            print(f"Scene captured and saved as: {self.current_image_path}")
        except Exception as e:
            print(f"Failed to save image: {e}")
            return None, [], "Save failed"

        # Format detection text
        if not detections:
            formatted_text = "No objects detected"
        else:
            detection_strings = []
            for obj_num, x, y in detections:
                detection_strings.append(f"Object {obj_num}: ({x:.2f}cm, {y:.2f}cm)")
            formatted_text = "Detected objects:\n" + "\n".join(detection_strings)

        # Print detections to console (matching original format)
        if detections:
            print(
                "Detections (cm):",
                ", ".join([f"({n},{x:.2f},{y:.2f})" for n, x, y in detections]),
            )
        else:
            print("Detections (cm): []")

        return self.current_image_path, detections, formatted_text

    def execute_task(self, user_prompt):
        """
        Execute a complete task with the robotic arm
        This follows the same structure as the original llm.py execute_arm_task function
        """
        # Store the user prompt for the entire task
        self.user_prompt = user_prompt

        print("DUM-E Robotic Arm Task Executor")
        print("=" * 50)
        print(f"User Request: {user_prompt}")
        print("=" * 50)

        try:
            # Step 1: Capture initial scene
            print("\nCapturing initial scene...")
            image_path, detections, additional_text = self.capture_current_scene()

            if image_path is None:
                print("Failed to capture initial scene")
                return False

            # Step 2: Analyze with Groq (single call)
            print("\nAnalyzing scene with Groq...")
            coordinate_dict = analyze_with_groq(
                image_path, user_prompt, additional_text
            )

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

                # Display the planned action
                task_desc = coordinate_dict.get("task_description", "Unknown action")
                print(f"Planned Action: {task_desc}")

                # Display coordinates for debugging
                if "in_coord" in coordinate_dict:
                    print(f"Pick up location: {coordinate_dict['in_coord']}")
                if "out_coord" in coordinate_dict:
                    print(f"Place location: {coordinate_dict['out_coord']}")
                if "gripper_action" in coordinate_dict:
                    print(f"Gripper action: {coordinate_dict['gripper_action']}")

                # For now, simulate arm control with sleep
                print("Simulating arm movement...")
                print("Waiting 10 seconds for manual object movement...")
                print(
                    "   (In real implementation, this would send coordinates to arm control)"
                )
                time.sleep(10)

                # In real implementation, this would be:
                # success = send_to_arm_control(coordinate_dict)
                # if not success:
                #     print("Arm control failed, stopping task")
                #     return False

                # Check if task is complete
                task_complete = coordinate_dict.get("task_complete", False)

                if task_complete:
                    print("Task completed successfully!")
                    break

                # Wait a bit before next iteration
                time.sleep(1)

                # Re-analyze if task is not complete (following original llm.py structure)
                if not task_complete and iteration < max_iterations:
                    print("Re-analyzing for next step...")

                    # Capture new scene
                    print("Capturing updated scene...")
                    image_path, detections, additional_text = (
                        self.capture_current_scene()
                    )

                    if image_path is None:
                        print("Failed to capture updated scene")
                        return False

                    # Re-analyze with new scene - IMPORTANT: Pass the original user_prompt
                    coordinate_dict = analyze_with_groq(
                        image_path, self.user_prompt, additional_text
                    )
                    if coordinate_dict is None:
                        print("Failed to re-analyze, stopping task")
                        return False

            if iteration >= max_iterations:
                print("Reached maximum iterations, stopping task")

            return task_complete

        except Exception as e:
            print(f"Error executing task: {e}")
            return False

    def cleanup(self):
        """
        Clean up resources
        """
        self.capture_system.cleanup()
        print("DUM-E System cleaned up")


def main():
    """
    Main function to run the DUM-E system
    """
    print("Welcome to DUM-E Robotic Arm System!")
    print("=" * 50)

    # Initialize system
    dume = DUMESystem(camera_id=0)
    print("running!?!?!")

    if not dume.initialize():
        print("Failed to initialize DUM-E system")
        return

    try:
        # Get user prompt
        print("\nPlease describe what you want the robotic arm to do:")
        print("Examples:")
        print("- 'Pick up the red object and move it to the left'")
        print("- 'Sort the objects by color'")
        print("- 'Wave to the camera'")
        print("- 'Pick up the tool and hold it steady'")
        print("- 'Move all objects to the right side'")

        user_prompt = input("\nYour request: ").strip()

        if not user_prompt:
            print("No request provided, exiting")
            return

        # Execute the task
        success = dume.execute_task(user_prompt)

        if success:
            print("\nTask completed successfully!")
        else:
            print("\nTask failed or was incomplete")

    except KeyboardInterrupt:
        print("\n\nTask interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        dume.cleanup()


if __name__ == "__main__":
    main()
