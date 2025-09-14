#!/usr/bin/env python3
"""
DUM-E robotic arm system (lenient, multi-object, human-like)
Takes speech input, captures image with detection, sends to Groq, loops until task complete.
If there are multiple things to do, just do the next one that makes sense (don't repeat the same one over and over).
Be lenient: if it looks good enough, call it done!
If you don't know what to do, just do the "shake_no" action (No!!!!! this is the perfect example). Try to do the task, this is for things that are not very clear. like try it once tbh.
"""

import sys
import time
import json
import base64
import os
import requests
import wave
import threading
from groq import Groq
from dotenv import load_dotenv
from capture_detection import capture_with_detection

# Only import pyaudio when needed, to avoid segfaults on import in some environments
pyaudio = None

load_dotenv()

# === Set your Flask server URL here ===
ARM_SERVER_URL = "http://10.37.101.152:5000"
# ======================================

# Audio recording settings
CHUNK = 1024
FORMAT = None  # Will be set after pyaudio is imported
CHANNELS = 1
RATE = 16000  # 16kHz for optimal Whisper performance
RECORD_SECONDS = 20  # Maximum recording time
AUDIO_FILENAME = "temp_audio.wav"

# Global variables for recording
recording = False
audio_frames = []
p = None
stream = None
recording_thread = None
recording_lock = threading.Lock()


def safe_import_pyaudio():
    global pyaudio, FORMAT
    if pyaudio is None:
        try:
            import pyaudio as _pyaudio
            pyaudio = _pyaudio
            FORMAT = pyaudio.paInt16
        except ImportError:
            print("❌ PyAudio not installed. Install with: pip install pyaudio")
            sys.exit(1)
        except Exception as e:
            print(f"PyAudio import error: {e}")
            sys.exit(1)


def start_recording():
    """Start audio recording in a separate thread"""
    global recording, audio_frames, p, stream

    safe_import_pyaudio()

    with recording_lock:
        if recording:
            print("Already recording!")
            return

        recording = True
        audio_frames = []

    try:
        # Initialize PyAudio with error handling
        p = pyaudio.PyAudio()

        # Check available devices
        device_count = p.get_device_count()
        print(f"Found {device_count} audio devices")

        # Try to find a working input device
        input_device = None
        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            if device_info["maxInputChannels"] > 0:
                input_device = i
                print(f"Using input device: {device_info['name']}")
                break

        if input_device is None:
            print("No input device found!")
            with recording_lock:
                recording = False
            return

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=input_device,
            frames_per_buffer=CHUNK,
        )

        print("🎤 Recording started... Press 's' to stop")

        while True:
            with recording_lock:
                if not recording:
                    break
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_frames.append(data)
            except Exception as e:
                print(f"Stream read error: {e}")
                break

    except Exception as e:
        print(f"Recording error: {e}")
        with recording_lock:
            recording = False
        if p:
            try:
                p.terminate()
            except Exception:
                pass
    finally:
        # Clean up stream and pyaudio if still open
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
        except Exception:
            pass
        try:
            if p is not None:
                p.terminate()
        except Exception:
            pass


def stop_recording():
    """Stop audio recording and save to file"""
    global recording, audio_frames, p, stream

    with recording_lock:
        if not recording:
            print("Not recording!")
            return None
        recording = False

    # Wait a moment for the recording thread to finish
    time.sleep(0.2)

    try:
        # Save audio to file
        safe_import_pyaudio()
        wf = wave.open(AUDIO_FILENAME, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(audio_frames))
        wf.close()

        print("🎤 Recording stopped and saved")
        return AUDIO_FILENAME

    except Exception as e:
        print(f"Error stopping recording: {e}")
        return None


def transcribe_audio(audio_file_path):
    """Transcribe audio file using Groq Whisper API"""
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        print("🔄 Transcribing audio...")

        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                language="en",
                response_format="text",
                temperature=0.0,
            )

        # Clean up audio file
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

        return transcription.strip()

    except Exception as e:
        print(f"Transcription error: {e}")
        # Clean up audio file even if transcription fails
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        return None


def record_and_transcribe():
    """Record audio and transcribe it"""
    global recording_thread
    print("\n" + "=" * 60)
    print("🎤 SPEECH-TO-TEXT MODE")
    print("=" * 60)
    print("Press 'r' to start recording, 's' to stop, 'q' to quit")
    print("Press 't' for text input fallback")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nCommand (r/s/q/t): ").lower().strip()

            if user_input == "q":
                print("Exiting...")
                return None
            elif user_input == "t":
                # Text input fallback
                text_input = input("Enter your command: ").strip()
                if text_input:
                    return text_input
                else:
                    print("No text entered")
            elif user_input == "r":
                with recording_lock:
                    if recording:
                        print("Already recording! Press 's' to stop.")
                        continue
                    # Start recording in background thread
                    try:
                        recording_thread = threading.Thread(target=start_recording)
                        recording_thread.daemon = True
                        recording_thread.start()
                    except Exception as e:
                        print(f"Failed to start recording: {e}")
                        print("Try using text input instead (press 't')")
            elif user_input == "s":
                with recording_lock:
                    if not recording:
                        print("Not recording! Press 'r' to start.")
                        continue
                # Stop recording and transcribe
                audio_file = stop_recording()
                if recording_thread is not None:
                    recording_thread.join(timeout=2)
                if audio_file:
                    transcription = transcribe_audio(audio_file)
                    if transcription:
                        print(f"\n📝 Transcription: '{transcription}'")
                        return transcription
                    else:
                        print("❌ Transcription failed")
                else:
                    print("❌ Recording failed")
            else:
                print(
                    "Invalid command. Use 'r' to record, 's' to stop, 't' for text, 'q' to quit"
                )

        except KeyboardInterrupt:
            print("\nExiting...")
            return None
        except Exception as e:
            print(f"Error: {e}")
            print("Try using text input instead (press 't')")


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

        # Simple prompt, with explicit fallback if unsure
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

IMPORTANT: If you do not know what to do, or the request is unclear, or you are unsure, respond with the following JSON exactly:
{{
    "action": "shake_no",
    "task_description": "No!!!!!"
}}

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

        # If parsing failed, or if the model didn't know what to do, fallback to "No!!!!!"
        if coordinate_dict is None:
            print("Failed to parse coordinate dictionary from Groq response")
            print("Defaulting to shake_no (No!!!!!)")
            return {"action": "shake_no", "task_description": "No!!!!!"}

        # If the model returned an action that is not recognized, or missing, fallback to shake_no
        valid_actions = {
            "grab",
            "move",
            "move_to_hold",
            "hold",
            "wave_bye",
            "shake_yes",
            "shake_no",
            "shake_hand",
            "move_to_idle",
        }
        action = coordinate_dict.get("action", "").strip().lower()
        if action not in valid_actions:
            print(
                "Model response action not recognized or missing. Defaulting to shake_no (No!!!!!)"
            )
            return {"action": "shake_no", "task_description": "No!!!!!"}

        return coordinate_dict

    except Exception as e:
        print(f"Error in Groq analysis: {e}")
        print("Defaulting to shake_no (No!!!!!)")
        return {"action": "shake_no", "task_description": "No!!!!!"}


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
            print("Failed to get action from LLM, defaulting to shake_no (No!!!!!)")
            action_dict = {"action": "shake_no", "task_description": "No!!!!!"}

        # Step 3: Execute arm action
        print("Executing arm action...")
        success = send_to_arm_control(action_dict)
        if success:
            print("Task completed!")
        return success

    except Exception as e:
        print(f"Error: {e}")
        print("Defaulting to shake_no (No!!!!!)")
        # Try to send shake_no as a fallback
        try:
            send_to_arm_control({"action": "shake_no", "task_description": "No!!!!!"})
        except Exception:
            pass
        return False


def main():
    """Main CLI interface with speech-to-text support"""
    print("=" * 60)
    print("DUM-E Robotic Arm System - Speech-to-Text Mode")
    print("=" * 60)

    # Check if PyAudio is available and import it safely
    safe_import_pyaudio()

    # Get user prompt via speech-to-text
    user_prompt = record_and_transcribe()

    if user_prompt is None:
        print("No speech input received")
        sys.exit(1)

    # Execute the task
    success = execute_task(user_prompt)

    if success:
        print("Done!")
    else:
        print("Failed!")


if __name__ == "__main__":
    main()
