import cv2
import numpy as np
import argparse
import os
from typing import List, Tuple, Optional
from collections import deque

class CompletePipeline:
    def __init__(self, src_points=None, dst_points=None):
        self.src_points = src_points or np.array([
            [600, 78], #topleft
            [1421, 80], #top right
            [1622, 823], # bottom right
            [314, 821], # bottomleft
        ], dtype=np.float32)
        
        self.dst_points = dst_points or np.array([
            [0, 0],
            [400, 0],
            [400, 400],
            [0, 400]
        ], dtype=np.float32)
        
        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.center_history: deque = deque(maxlen=5)
    
    def _apply_nms(self, rects_scores: List[Tuple[int,int,int,int,float]], iou_thr: float = 0.35) -> List[Tuple[int,int,int,int,float]]:
        if not rects_scores:
            return []
        boxes = np.array([[x, y, x+w, y+h] for (x,y,w,h,_) in rects_scores], dtype=np.float32)
        scores = np.array([s for (*_, s) in rects_scores], dtype=np.float32)
        idxs = scores.argsort()[::-1]
        keep = []
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            if len(idxs) == 1:
                break
            rest = idxs[1:]
            x1 = np.maximum(boxes[i,0], boxes[rest,0])
            y1 = np.maximum(boxes[i,1], boxes[rest,1])
            x2 = np.minimum(boxes[i,2], boxes[rest,2])
            y2 = np.minimum(boxes[i,3], boxes[rest,3])
            inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
            area_i = (boxes[i,2]-boxes[i,0]) * (boxes[i,3]-boxes[i,1])
            area_r = (boxes[rest,2]-boxes[rest,0]) * (boxes[rest,3]-boxes[rest,1])
            iou = inter / (area_i + area_r - inter + 1e-6)
            idxs = rest[iou <= iou_thr]
        return [rects_scores[i] for i in keep]

    def _persistence_filter(self, rects: List[Tuple[int,int,int,int]]) -> List[Tuple[int,int,int,int]]:
        if not self.center_history:
            centers = [((x + x+w)//2, (y + y+h)//2) for (x,y,w,h) in rects]
            self.center_history.append(centers)
            return rects
        prev_centers_flat = [c for hist in self.center_history for c in hist]
        kept = []
        cur_centers = []
        for (x,y,w,h) in rects:
            cx, cy = (x + w//2), (y + h//2)
            cur_centers.append((cx, cy))
            ok = False
            for (px, py) in prev_centers_flat:
                if (cx-px)*(cx-px) + (cy-py)*(cy-py) <= 25*25:
                    ok = True
                    break
            if ok:
                kept.append((x,y,w,h))
        self.center_history.append(cur_centers)
        return kept

    def _red_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 80, 50], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 80, 50], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        red = cv2.bitwise_or(mask1, mask2)
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        return red

    def _find_filtered_rects(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (3,3))
        canny = cv2.Canny(gray, 100, 200)
        canny = cv2.dilate(canny, np.ones((3,3), np.uint8), iterations=1)
        red = self._red_mask(frame)
        combo = cv2.bitwise_or(canny, red)
        contours, _ = cv2.findContours(combo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Tuple[int,int,int,int,float]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 1400:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 26 or h < 26:
                continue
            rect_area = float(w * h)
            extent = area / (rect_area + 1e-6)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull) + 1e-6
            solidity = area / hull_area
            ar = w / float(h)
            peri = cv2.arcLength(c, True)
            circularity = 4.0 * np.pi * area / (peri*peri + 1e-6)
            roi_edges = canny[y:y+h, x:x+w]
            edge_density = float(np.count_nonzero(roi_edges)) / (rect_area + 1e-6)
            roi_red = red[y:y+h, x:x+w]
            red_cov = float(np.count_nonzero(roi_red)) / (rect_area + 1e-6)
            if extent < 0.17:
                continue
            if solidity < 0.33:
                continue
            if ar < 0.20 or ar > 6.0:
                continue
            if circularity < 0.035:
                continue
            if edge_density < 0.012 and red_cov < 0.07:
                continue
            score = min(1.0, (area / 30000.0) + 0.7*red_cov + 0.4*edge_density + 0.2*extent)
            candidates.append((x, y, w, h, float(score)))
        candidates = self._apply_nms(candidates, 0.35)
        rects = [(x,y,w,h) for (x,y,w,h,_) in candidates]
        rects = self._persistence_filter(rects)
        return rects

    def draw_automatic_detections(self, frame: np.ndarray, result: np.ndarray):
        rects = self._find_filtered_rects(frame)
        if isinstance(rects, tuple):
            rects = rects[0]
        for i, (x, y, w, h) in enumerate(rects):
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx, cy = x + w//2, y + h//2
            cv2.putText(result, f"{i+1}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.circle(result, (cx, cy), 3, (0,255,255), -1)

    def get_detections(self, frame: np.ndarray) -> List[Tuple[int, float, float]]:
        # Get detections in the warped (homography) space, and return their coordinates in that space
        warped = cv2.warpPerspective(frame, self.H, (400, 400))
        rects = self._find_filtered_rects(warped)
        if isinstance(rects, tuple):
            rects = rects[0]
        out = []
        for i, (x, y, w, h) in enumerate(rects):
            cx = x + w // 2
            cy = y + h // 2
            # Convert to (x, 400-y), then scale to cm
            cx_cm = (cx * 30.0) / 400.0
            cy_cm = ((400 - cy) * 30.0) / 400.0
            out.append((i+1, cx_cm, cy_cm))  # These are in cm, with y flipped
        return out

    def save_centers(self, centers: List[Tuple[int,float,float]], path: str = "pipeline_output/centers.txt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for obj_num, cx, cy in centers:
                f.write(f"{obj_num},{cx:.2f},{cy:.2f}\n")

    def run_pipeline(self, camera_id: int = 0):
        cap = None
        for cam_id in [camera_id, 1, 2, 0]:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret:
                    break
                else:
                    cap.release()
                    cap = None
            else:
                cap.release()
                cap = None
        if cap is None or not cap.isOpened():
            return
        frame_count = 0
        consecutive_failures = 0
        max_failures = 10
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                if consecutive_failures % 5 == 0:
                    cap.release()
                    cap = cv2.VideoCapture(camera_id)
                    if not cap.isOpened():
                        break
                continue
            consecutive_failures = 0
            frame_count += 1
            warped = cv2.warpPerspective(frame, self.H, (400, 400))
            warped_with_boxes = warped.copy()
            self.draw_automatic_detections(warped, warped_with_boxes)
            h = 400
            scale = h / frame.shape[0]
            original_resized = cv2.resize(frame, None, fx=scale, fy=scale)
            combined = np.hstack((original_resized, warped_with_boxes))
            centers = self.get_detections(frame)
            if centers:
                print("Detections (cm):", ", ".join([f"({n},{x:.2f},{y:.2f})" for n,x,y in centers]))
            else:
                print("Detections (cm): []")
            cv2.imshow("Complete Pipeline", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.save_centers(centers)
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Complete Pipeline: Homography + Object Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    args = parser.parse_args()
    pipeline = CompletePipeline()
    pipeline.run_pipeline(args.camera)

if __name__ == "__main__":
    main()