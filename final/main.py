#!/usr/bin/env python3
"""
dum-e robotic arm system (lenient, multi-object, human-like)
takes speech input, transcribes with groq whisper, captures image with detection, sends to groq, does one action at a time.
if there are multiple things to do, just do the next one that makes sense (don't repeat the same one over and over).
be lenient: if it looks good enough, call it done!
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

# === set your flask server url here ===
arm_server_url = "http://10.37.101.152:5000"
# ======================================


def encode_image(image_path):
    """encode image to base64 for groq api"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_groq_response(response_text):
    """parse groq response to extract coordinate dictionary"""
    try:
        # try to find a json object in the response
        lines = response_text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        # fallback: try to extract json from the whole response
        if "{" in response_text and "}" in response_text:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_str = response_text[start:end]
            return json.loads(json_str)

        print("no valid json found in groq response")
        return None

    except Exception as e:
        print(f"error parsing groq response: {e}")
        return None


def analyze_with_groq(image_path, user_prompt, detections, hand_cm=None):
    """Analyze image with Groq and return coordinate dictionary (lenient, multi-object, human-like)"""
    try:
        base64_image = encode_image(image_path)
        client = Groq(api_key=os.environ.get("groq_api_key"))

        # format detections for context
        if detections:
            detection_strings = [
                f"object {obj_num}: ({x:.2f}cm, {y:.2f}cm)"
                for obj_num, x, y in detections
            ]
            detection_text = "detected objects (in cm coordinates):\n" + "\n".join(
                detection_strings
            )
        else:
            detection_text = "no objects detected"

        # Add hand coordinate context if available
        if hand_cm is not None:
            hand_text = f"\nHAND POSITION (cm): ({hand_cm[0]:.2f}, {hand_cm[1]:.2f})"
        else:
            hand_text = "\nHAND POSITION: None"

        # Simple prompt
        prompt_text = f"""
you are dum-e, a robotic arm. look at the image and do what the user asks.

USER REQUEST: {user_prompt}
DETECTED OBJECTS: {detection_text}{hand_text}

coordinate system: x=0-30cm (left-right), y=0-30cm (front-back), z=0 (table level)

available actions:
- "grab": pick up object at (x,y) and move to (x2,y2) - use phi=270 for top-down. 
  grab --> for moving objects
- "move_to_hold": move to hold position at (x,y)
  move to hold --> for asking it to hold things for us in place. it just helps to clarify to the DUM-E arm what we want out of it
- "wave_bye": wave goodbye
- "shake_yes": nod yes  
- "shake_no": shake no
- "shake_hand": handshake
  handshake --> for handshaekes
- "move_to_idle": go to safe position

Do just one action at a time. If there are multiple things to do, just do the next one that makes sense (don't repeat the same one over and over). Be lenient and don't be too strict. For example, if you have to move something, if its approx in the desired area then you are good. No need to be perfect for cords.

response (json only): MAKE SURE ITS VALID PYTHON JSON AND CAN PARSE. NOTHING ELSE. 
{{
    "action": "grab",
    "x": 15.0,
    "y": 10.0, 
    "phi": 270,
    "x2": 25.0,
    "y2": 20.0,
    "task_description": "what i'm doing"
}}
"""

        # make the api call
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
            temperature=0.2,  # low temperature for precise coordinates
            max_completion_tokens=512,
            top_p=0.8,
            stream=False,
        )

        response = completion.choices[0].message.content
        print("dum-e analysis:")
        print(response)

        # parse the response
        coordinate_dict = parse_groq_response(response)

        if coordinate_dict is None:
            print("failed to parse coordinate dictionary from groq response")
            return None

        return coordinate_dict

    except Exception as e:
        print(f"error in groq analysis: {e}")
        return None


def send_to_arm_control(action_dict, arm_server_url=None):
    """send action to arm control server"""
    if arm_server_url is None:
        arm_server_url = arm_server_url

    try:
        print(f"sending action: {action_dict.get('action')}")
        response = requests.post(
            f"{arm_server_url}/arm_control", json=action_dict, timeout=10
        )

        if response.status_code == 200:
            print("action completed successfully")
            return True
        else:
            print(f"action failed: {response.json()}")
            return False

    except requests.exceptions.ConnectionError:
        print("cannot connect to arm server")
        return False
    except Exception as e:
        print(f"error: {e}")
        return False


class AudioRecorder:
    """simple audio recorder with start/stop functionality"""

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.recording = False
        self.frames = []
        self.stream = None
        self.record_thread = None
        self._stop_event = threading.Event()

        # audio settings
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # whisper works well with 16khz

    def start_recording(self):
        """start recording audio"""
        if self.recording:
            return

        self.recording = True
        self.frames = []
        self._stop_event.clear()

        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

        print("🎤 recording... (press 's' to stop)")

        def record():
            while not self._stop_event.is_set():
                try:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    print(f"audio read error: {e}")
                    break

        self.record_thread = threading.Thread(target=record)
        self.record_thread.start()

    def stop_recording(self):
        """stop recording and return audio data"""
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
                print(f"stream close error: {e}")
            self.stream = None

        print("⏹️  recording stopped")
        return b"".join(self.frames)

    def save_audio(self, audio_data, filename="temp_audio.wav"):
        """save audio data to file"""
        if not audio_data:
            return None

        filepath = os.path.join(os.getcwd(), filename)

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(audio_data)

        return filepath

    def cleanup(self):
        """clean up audio resources"""
        if self.stream:
            try:
                self.stream.close()
            except Exception:
                pass
        self.audio.terminate()


def transcribe_audio_with_groq(audio_file_path):
    """transcribe audio using groq whisper large v3 turbo"""
    try:
        client = Groq(api_key=os.environ.get("groq_api_key"))

        print("🔄 transcribing audio...")

        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                language="en",  # you can change this or remove for auto-detection
            )

        transcript = transcription.text.strip()
        print(f"📝 transcription: '{transcript}'")
        return transcript

    except Exception as e:
        print(f"error transcribing audio: {e}")
        return None


def get_speech_input():
    """get speech input from user with 'r' to record, 's' to stop, 'q' to quit"""
    recorder = AudioRecorder()
    print("\n" + "=" * 60)
    print("🎤 dum-e speech interface")
    print("=" * 60)
    print("Press 'r' to start recording, 's' to stop, 'q' to quit")
    print("=" * 60)

    result = None
    recording = False

    try:
        while True:
            key = input(">> ").strip().lower()
            if key == "q":
                if recorder.recording:
                    recorder.stop_recording()
                print("👋 goodbye!")
                return None
            elif key == "r":
                if not recording:
                    recorder.start_recording()
                    recording = True
                else:
                    print("Already recording. Press 's' to stop.")
            elif key == "s":
                if recording:
                    audio_data = recorder.stop_recording()
                    recording = False
                    if audio_data:
                        audio_file = recorder.save_audio(audio_data)
                        if audio_file:
                            transcript = transcribe_audio_with_groq(audio_file)
                            try:
                                os.remove(audio_file)
                            except:
                                pass
                            if transcript:
                                result = transcript
                                break
                            else:
                                print("❌ transcription failed. try again.")
                        else:
                            print("❌ failed to save audio. try again.")
                    else:
                        print("❌ no audio recorded. try again.")
                else:
                    print("Not recording. Press 'r' to start.")
            else:
                print("Press 'r' to record, 's' to stop, or 'q' to quit.")
        return result
    finally:
        recorder.cleanup()


def execute_task(user_prompt, camera_id=0):
    """task execution: capture image, analyze with llm, execute arm action (one step only)"""
    print("=" * 60)
    print("dum-e robotic arm system")
    print("=" * 60)
    print(f"user request: {user_prompt}")
    print("=" * 60)

    try:
        # Step 1: Capture image
        print("Capturing image...")
        image_path, detections, hand_cm = capture_with_detection(camera_id)
        if image_path is None:
            print("failed to capture image")
            return False

        # Step 2: Analyze with LLM
        print("Analyzing with LLM...")
        action_dict = analyze_with_groq(
            image_path, user_prompt, detections, hand_cm
        )
        if action_dict is None:
            print("failed to get action from llm")
            return False

        # Step 3: Execute arm action
        print("executing arm action...")
        success = send_to_arm_control(action_dict, arm_server_url)
        if not success:
            print("arm action failed")
            return False

        print(f"✅ action executed: {action_dict.get('action', 'unknown')}")
        return True

    except Exception as e:
        print(f"error: {e}")
        return False


def main():
    """main speech interface"""
    print("🤖 dum-e robotic arm with speech control")
    print("starting speech interface...")

    while True:
        # get speech input
        user_prompt = get_speech_input()

        if user_prompt is None:  # user quit
            break

        if not user_prompt.strip():
            print("no speech detected. try again.")
            continue

        # execute the task
        print(f"\n🎯 executing task: '{user_prompt}'")
        success = execute_task(user_prompt)

        if success:
            print("✅ action completed!")
        else:
            print("❌ action failed!")

        print("\nPress 'r' to record another command, or 'q' to quit.")


if __name__ == "__main__":
    main()
