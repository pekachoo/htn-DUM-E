from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from complete_pipeline import CompletePipeline

app = Flask(__name__)
pipeline = CompletePipeline()

def capture_image():
    cap = None
    for cam_id in [0, 1, 2]:
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cap.release()
                return frame
            else:
                cap.release()
                cap = None
        else:
            cap.release()
            cap = None
    return None

@app.route('/capture', methods=['GET'])
def capture_and_process():
    try:
        frame = capture_image()
        if frame is None:
            return jsonify({'error': 'Could not capture image from camera'}), 500
        
        detections = pipeline.get_detections(frame)
        
        warped = cv2.warpPerspective(frame, pipeline.H, (400, 400))
        warped_with_boxes = warped.copy()
        pipeline.draw_automatic_detections(warped, warped_with_boxes)
        
        # Save image with boxes temporarily
        import time
        timestamp = int(time.time())
        filename = f"temp_image_{timestamp}.jpg"
        cv2.imwrite(filename, warped_with_boxes)
        
        _, buffer = cv2.imencode('.jpg', warped_with_boxes)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        objects = []
        for obj_num, center_x, center_y in detections:
            objects.append({
                'object_number': obj_num,
                'center_x': center_x,
                'center_y': center_y
            })
        
        return jsonify({
            'success': True,
            'image': image_base64,
            'objects': objects,
            'object_count': len(objects),
            'saved_file': filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
