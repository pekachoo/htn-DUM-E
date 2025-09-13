import asyncio
import base64
import json
import os
import sys
import cv2
import numpy as np
import websockets

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from complete_pipeline import CompletePipeline

pipeline = CompletePipeline()

async def capture_frame() -> np.ndarray:
    cap = None
    for cam_id in [0, 1, 2]:
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cap.release()
                return frame
            cap.release()
    return None

async def process_request(message: str) -> str:
    try:
        data = json.loads(message) if message else {}
    except Exception:
        data = {}

    frame = await capture_frame()
    if frame is None:
        return json.dumps({'success': False, 'error': 'Could not capture image from camera'})

    detections = pipeline.get_detections(frame)
    warped = cv2.warpPerspective(frame, pipeline.H, (400, 400))
    warped_with_boxes = warped.copy()
    pipeline.draw_automatic_detections(warped, warped_with_boxes)

    _, buffer = cv2.imencode('.jpg', warped_with_boxes)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    objects = [{'object_number': n, 'center_x': x, 'center_y': y} for n, x, y in detections]

    return json.dumps({
        'success': True,
        'image': image_base64,
        'objects': objects,
        'object_count': len(objects)
    })

async def handler(websocket):
    async for message in websocket:
        response = await process_request(message)
        await websocket.send(response)

async def main():
    async with websockets.serve(handler, '0.0.0.0', 5000, ping_interval=20, ping_timeout=20):
        print('WebSocket server running on ws://0.0.0.0:5000')
        await asyncio.Future()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
