# DUM-E Arm Integration

This document explains how to use the integrated DUM-E robotic arm system with LLM control.

## Overview

The system consists of two main components:

1. **Flask Arm Control Server** (`arm_control_server.py`) - Controls the physical arm
2. **Main LLM System** (`final/main.py`) - Processes natural language commands and coordinates with the arm

## Setup

### 1. Install Dependencies

For the arm control server:

```bash
pip install -r arm_requirements.txt
```

For the main system (if not already installed):

```bash
cd final
pip install -r requirements.txt
```

### 2. Hardware Setup

- Connect the Arduino to your computer via USB
- Note the serial port (usually `/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0` on Linux)
- Update the serial port in `arm_control_server.py` if needed

### 3. Environment Variables

Make sure you have your Groq API key set:

```bash
export GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### Starting the System

1. **Start the Arm Control Server** (in one terminal):

```bash
python arm_control_server.py
```

This will start the Flask server on `http://localhost:5000`

2. **Run the Main System** (in another terminal):

```bash
cd final
python main.py "Pick up the red object and move it to the left"
```

### Available Arm Actions

The system supports the following arm actions:

| Action         | Parameters                                 | Description                                    |
| -------------- | ------------------------------------------ | ---------------------------------------------- |
| `grab`         | x, y, phi                                  | Pick up an object at coordinates               |
| `move`         | x, y, z, phi, claw_open, roll_angle, elbow | Move to coordinates with specified orientation |
| `move_to_idle` | None                                       | Move to default idle position                  |
| `wave_bye`     | None                                       | Wave goodbye gesture                           |
| `shake_yes`    | None                                       | Nod yes gesture                                |
| `shake_no`     | None                                       | Shake no gesture                               |
| `shake_hand`   | None                                       | Handshake gesture                              |
| `move_to_hold` | x, y                                       | Move to hold position                          |
| `hold`         | x, y                                       | Hold at position                               |
| `drop_off`     | x, y, z, phi                               | Drop object at coordinates                     |

### API Endpoints

#### POST `/arm_control`

Main endpoint for arm control. Send JSON with:

```json
{
  "action": "grab",
  "x": 10.0,
  "y": 10.0,
  "phi": 0
}
```

#### GET `/arm_status`

Check if the arm is ready and initialized.

#### GET `/health`

Health check endpoint.

## How It Works

1. **User Input**: User provides a natural language command
2. **Image Capture**: System captures an image with object detection
3. **LLM Analysis**: Groq LLM analyzes the image and determines which action to take and coordinates
4. **Action Routing**: System sends action-based commands to the Flask server
5. **Execution**: Flask server routes commands to the appropriate arm functions based on action type
6. **Feedback**: System provides real-time feedback on arm movements

## New Action-Based System

The system now uses an **action-based approach** where the LLM specifies:

- **Action type**: `grab`, `move`, `wave_bye`, etc.
- **Coordinates**: Only when needed for the specific action
- **Parameters**: Action-specific parameters like `claw_open`, `roll_angle`, etc.

This is more intuitive and efficient than the previous coordinate-only system.

### LLM Response Format

The LLM now returns JSON in this format:

```json
{
  "action": "grab", // Action to perform
  "x": 15.0, // X coordinate in cm (when needed)
  "y": 10.0, // Y coordinate in cm (when needed)
  "z": 0.0, // Z coordinate in cm (when needed)
  "phi": 0, // Angle in degrees (when needed)
  "claw_open": 1, // 1 for open, 0 for closed (when needed)
  "roll_angle": 0, // Roll angle in degrees (when needed)
  "elbow": "up", // "up" or "down" (when needed)
  "task_description": "Pick up the red object",
  "task_complete": false
}
```

## Example Commands

```bash
# Pick up and move objects
python main.py "Pick up the red block and move it to the top right corner"

# Gestures
python main.py "Wave goodbye"

# Complex tasks
python main.py "Organize all the objects by color"

# Simple movements
python main.py "Move the blue object to the center"
```

## Troubleshooting

### Arm Server Not Starting

- Check if the Arduino is connected
- Verify the serial port path
- Ensure no other programs are using the serial port

### Connection Errors

- Make sure the Flask server is running on port 5000
- Check firewall settings
- Verify the arm server is responding to health checks

### Arm Not Moving

- Check Arduino connection
- Verify servo connections
- Check power supply
- Review serial communication logs

## Testing

Run the test script to verify the integration:

```bash
python test_arm_integration.py
```

## Safety Notes

- Always ensure the workspace is clear before running commands
- Keep hands away from the arm during operation
- Use appropriate safety measures when working with the robotic arm
- The system includes safety limits, but always supervise operation

## File Structure

```
├── arm_control_server.py      # Flask server for arm control
├── arm_requirements.txt       # Dependencies for arm server
├── pi_to_arduino.py          # Original arm control functions
├── final/
│   ├── main.py               # Main LLM system (modified)
│   └── capture_detection.py  # Image capture and detection
├── test_arm_integration.py   # Test script
└── ARM_INTEGRATION_README.md # This file
```
