# Backend Flask Server

Flask server that processes images using the complete pipeline (homography + object detection).

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### GET /capture
Capture an image from camera, process it through complete pipeline, and return detected objects with their center coordinates.

**Request:**
- Method: GET
- No body required

**Response:**
```json
{
  "success": true,
  "image": "base64_encoded_image",
  "objects": [
    {
      "object_number": 1,
      "center_x": 150,
      "center_y": 200
    },
    {
      "object_number": 2,
      "center_x": 300,
      "center_y": 250
    }
  ],
  "object_count": 2
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## Usage Example

```bash
curl http://localhost:5000/capture
```
