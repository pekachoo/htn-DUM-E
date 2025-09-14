# DUM-E Robotic Arm System

Complete integration of computer vision, object detection, and LLM-based robotic arm control.

## 🚀 Quick Start

### 1. Run the Main System

```bash
cd final
python main.py
```

### 2. Test the System

```bash
python test_integration.py
```

## 📁 Files Overview

- **`main.py`** - Main integration script that runs the complete pipeline
- **`capture_detection.py`** - Object detection and image capture system
- **`homography_cv.py`** - Computer vision pipeline with homography transformation
- **`llm.py`** - LLM integration for task planning and coordinate generation
- **`test_integration.py`** - Test suite for the complete system

## 🔧 How It Works

1. **Initialization**: Sets up camera and detection system
2. **User Input**: Prompts for natural language task description
3. **Scene Capture**: Takes image with object detection using homography
4. **LLM Analysis**: Sends image + task to Groq for coordinate planning
5. **Action Loop**:
   - Displays planned action and coordinates
   - Simulates arm movement (10-second sleep)
   - Captures updated scene
   - Re-analyzes with LLM
   - Repeats until task is complete

## 🎯 Key Features

- **Persistent Task Context**: Remembers original user prompt throughout execution
- **Real-time Updates**: Captures new scenes after each action
- **Coordinate Display**: Shows pick/place locations and gripper actions
- **Task Completion Detection**: LLM determines when task is finished
- **Error Handling**: Robust error handling and cleanup

## 🔄 Task Execution Flow

```
User Input → Scene Capture → LLM Analysis → Action Planning →
Simulation → New Scene → Re-analysis → Continue until Complete
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

## 📝 Example Tasks

- "Pick up the red object and move it to the left"
- "Sort the objects by color"
- "Wave to the camera"
- "Pick up the tool and hold it steady"
- "Move all objects to the right side"

## 🔧 For Real Arm Integration

To connect to actual arm hardware, uncomment these lines in `main.py`:

```python
# Replace the sleep simulation with:
success = send_to_arm_control(coordinate_dict)
if not success:
    print("❌ Arm control failed, stopping task")
    return False
```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
python test_integration.py
```

This will test:

- System initialization
- Image capture functionality
- Complete task execution pipeline

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
