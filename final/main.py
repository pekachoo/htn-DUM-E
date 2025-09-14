#!/usr/bin/env python3
"""
DUM-E robotic arm system (lenient, multi-object, human-like)
Takes speech input, transcribes with Groq Whisper, captures image with detection, sends to Groq, loops until task complete.
If there are multiple things to do, just do the next one that makes sense (don't repeat the same one over and over).
Be lenient: if it looks good enough, call it done!
"""

import sys
import time
import json
import base64
import os
import requests
import pyaudio
import wave
import threading
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


def analyze_with_groq(image_path, user_prompt, detections, hand_cm=None):
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

        # Add hand coordinate context if available
        if hand_cm is not None:
            hand_text = f"\nHAND POSITION (cm): ({hand_cm[0]:.2f}, {hand_cm[1]:.2f})"
        else:
            hand_text = "\nHAND POSITION: None"

        # Simple prompt
        prompt_text = f"""
You are DUM-E, a robotic arm. Look at the image and do what the user asks.

USER REQUEST: {user_prompt}
DETECTED OBJECTS: {detection_text}{hand_text}

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


class AudioRecorder:
    """Simple audio recorder with start/stop functionality"""

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.recording = False
        self.frames = []
        self.stream = None
        self.record_thread = None
        self._stop_event = threading.Event()

        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # Whisper works well with 16kHz

    def start_recording(self):
        """Start recording audio"""
        if self.recording:
            return

        self.recording = True
        self.frames = []
        self._stop_event.clear()

        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        print("🎤 Recording... Press 's' to stop")

        def record():
            while not self._stop_event.is_set():
                try:
                    data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    print(f"Audio read error: {e}")
                    break

        self.record_thread = threading.Thread(target=record)
        self.record_thread.start()

    def stop_recording(self):
        """Stop recording and return audio data"""
        if not self.recording:
            return None

        self.recording = False
        self._stop_event.set()

        if self.record_thread:
            self.record_thread.join()
            self.record_thread = None

        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"Stream close error: {e}")
            self.stream = None

        print("⏹️  Recording stopped")
        return b"".join(self.frames)

    def save_audio(self, audio_data, filename="temp_audio.wav"):
        """Save audio data to file"""
        if not audio_data:
            return None

        filepath = os.path.join(os.getcwd(), filename)

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(audio_data)

        return filepath

    def cleanup(self):
        """Clean up audio resources"""
        if self.stream:
            try:
                self.stream.close()
            except Exception:
                pass
        self.audio.terminate()


def transcribe_audio_with_groq(audio_file_path):
    """Transcribe audio using Groq Whisper Large v3 Turbo"""
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        print("🔄 Transcribing audio...")

        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                language="en",  # You can change this or remove for auto-detection
            )

        transcript = transcription.text.strip()
        print(f"📝 Transcription: '{transcript}'")
        return transcript

    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None


def get_speech_input():
    """Get speech input from user with keyboard controls"""
    recorder = AudioRecorder()

    try:
        print("\n" + "=" * 60)
        print("🎤 DUM-E Speech Interface")
        print("=" * 60)
        print("Press 'r' to start recording, 's' to stop, 'q' to quit")
        print("=" * 60)

        while True:
            try:
                key = input().strip().lower()

                if key == "q":
                    if recorder.recording:
                        recorder.stop_recording()
                    print("👋 Goodbye!")
                    return None
                elif key == "r":
                    if not recorder.recording:
                        recorder.start_recording()
                    else:
                        print("Already recording! Press 's' to stop.")
                elif key == "s":
                    if recorder.recording:
                        audio_data = recorder.stop_recording()
                        if audio_data:
                            # Save audio file
                            audio_file = recorder.save_audio(audio_data)
                            if audio_file:
                                # Transcribe
                                transcript = transcribe_audio_with_groq(audio_file)
                                if transcript:
                                    # Clean up audio file
                                    try:
                                        os.remove(audio_file)
                                    except:
                                        pass
                                    return transcript
                                else:
                                    print("❌ Transcription failed. Try again.")
                        else:
                            print("❌ No audio recorded. Try again.")
                    else:
                        print("Not recording! Press 'r' to start.")
                else:
                    print(
                        "Unknown command. Use 'r' to record, 's' to stop, 'q' to quit."
                    )

            except KeyboardInterrupt:
                if recorder.recording:
                    recorder.stop_recording()
                print("\n👋 Goodbye!")
                return None
            except EOFError:
                if recorder.recording:
                    recorder.stop_recording()
                print("\n👋 Goodbye!")
                return None

    finally:
        recorder.cleanup()


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
        image_path, detections, hand_cm = capture_with_detection(camera_id)
        if image_path is None:
            print("Failed to capture image")
            return False

        # Step 2: Analyze with LLM
        print("Analyzing with LLM...")
        action_dict = analyze_with_groq(image_path, user_prompt, detections, hand_cm)
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
    """Main speech interface"""
    print("🤖 DUM-E Robotic Arm with Speech Control")
    print("Starting speech interface...")

    while True:
        # Get speech input
        user_prompt = get_speech_input()

        if user_prompt is None:  # User quit
            break

        if not user_prompt.strip():
            print("No speech detected. Try again.")
            continue

        # Execute the task
        print(f"\n🎯 Executing task: '{user_prompt}'")
        success = execute_task(user_prompt)

        if success:
            print("✅ Task completed!")
        else:
            print("❌ Task failed!")

        print("\nPress 'r' to record another command, or 'q' to quit.")


if __name__ == "__main__":
    main()
