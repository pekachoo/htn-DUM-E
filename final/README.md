# DUM-E Robotic Arm System (Simplified)

Streamlined integration of computer vision, object detection, and LLM-based robotic arm control.

## 🚀 Quick Start

### 1. Set up environment

```bash
cd final
export GROQ_API_KEY=your_groq_api_key_here
```

### 2. Run the Main System

```bash
python main.py "Pick up the red object and move it to the left"
```

### 3. Test the System

```bash
python test_simple.py
```

## 📁 Files Overview

- **`main.py`** - Complete system with CLI interface and Groq integration
- **`capture_detection.py`** - Simple function to capture image with detection
- **`homography_cv.py`** - Computer vision pipeline with homography transformation
- **`test_simple.py`** - Simple test script

## 🔧 How It Works

1. **CLI Input**: Takes task description as command line argument
2. **Scene Capture**: Captures image with object detection and bounding boxes
3. **LLM Analysis**: Sends image + task to Groq for coordinate planning
4. **Action Loop**:
   - Displays planned action and coordinates
   - Simulates arm movement (2-second pause)
   - Captures updated scene
   - Re-analyzes with LLM
   - Repeats until task is complete

## 🎯 Key Features

- **Simple CLI**: Just pass task as command line argument
- **Single Function Capture**: One function call captures, saves, and returns coordinates
- **Integrated Groq**: All LLM functionality built into main.py
- **Real-time Updates**: Captures new scenes after each action
- **Task Completion Detection**: LLM determines when task is finished

## 🔄 Task Execution Flow

```
CLI Input → Capture → Groq Analysis → Action Planning →
Simulation → New Capture → Re-analysis → Continue until Complete
```

## 🛠️ Configuration

### Camera Settings

- Default camera ID: 0
- Automatically tries alternative cameras (1, 2) if needed

### LLM Settings

- Model: `meta-llama/llama-4-maverick-17b-128e-instruct`
- Temperature: 0.2 (for precise coordinates)
- Max iterations: 10 (safety limit)

### Coordinate System

- X: left/right (negative = left, positive = right)
- Y: forward/backward (negative = backward, positive = forward)
- Z: up/down (negative = down, positive = up)
- Units: millimeters
- Origin: base of the arm

## 📝 Example Usage

```bash
# Basic pick and place
python main.py "Pick up the red object and move it to the left"

# Sorting task
python main.py "Sort the objects by color - red to left, blue to right"

# Simple movement
python main.py "Move all objects to the right side"
```

## 🧪 Testing

Run the simple test to verify everything works:

```bash
python test_simple.py
```

This will test:

- Import functionality
- Image capture with detection
- System readiness

## 📋 Requirements

- OpenCV
- Groq API key (set in environment variables)
- Camera access
- All dependencies from `requirements.txt`

## 🚨 Troubleshooting

- **Camera not found**: Try different camera IDs (0, 1, 2)
- **LLM errors**: Check GROQ_API_KEY environment variable
- **Detection issues**: Adjust parameters in `homography_cv.py`
- **Task not completing**: Check LLM response format and task_complete flag
