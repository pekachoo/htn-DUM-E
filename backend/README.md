# Backend WebSocket Server

WebSocket server that captures a camera frame, runs the complete pipeline (homography + object detection), and returns a base64 image (with boxes) and object centers.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python app.py
```

The server starts at `ws://localhost:5000`.

## Protocol
- Send any JSON message (or empty string). The server captures a new frame, processes it, and sends a JSON response.

### Response JSON
```json
{
  "success": true,
  "image": "base64_jpeg",
  "objects": [
    {"object_number": 1, "center_x": 150, "center_y": 200}
  ],
  "object_count": 1
}
```

## Minimal Python Client
```python
import asyncio, websockets, json, base64

async def main():
    uri = "ws://localhost:5000"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"capture": True}))
        msg = await ws.recv()
        data = json.loads(msg)
        print("objects:", data.get("objects"))
        if data.get("image"):
            with open("result.jpg", "wb") as f:
                f.write(base64.b64decode(data["image"]))

asyncio.run(main())
```

## Notes
- The server opens the camera per request (tries IDs 0,1,2). Adjust if needed.
- Response includes a processed bird's-eye view image with bounding boxes.
