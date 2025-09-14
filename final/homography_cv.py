import cv2
import numpy as np
import argparse
import os
from typing import List, Tuple, Optional
from collections import deque
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class CompletePipeline:
    # ---------- Hand helpers ----------
    def _detect_hand_all(self, frame: np.ndarray
                         ) -> Optional[Tuple[Tuple[int,int], Tuple[float,float], Tuple[float,float]]]:
        """
        Internal: detect wrist center and return:
        - (hx, hy) in original pixels
        - (wx, wy) in warped pixels (400x400)
        - (x_cm, y_cm) in centimeters (30cm x 30cm board, y up)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        # Wrist is landmark 0
        hx = int(hand_landmarks.landmark[0].x * w)
        hy = int(hand_landmarks.landmark[0].y * h)

        # Optional: draw landmarks on the original frame for debugging
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Map to warped pixels, then to cm
        wx, wy = self.pixel_to_warped(hx, hy)
        x_cm, y_cm = self.warped_to_cm(wx, wy)
        return (hx, hy), (wx, wy), (x_cm, y_cm)

    def detect_hand(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        PUBLIC API (CHANGED): returns the wrist center in **centimetres**
        in the homography (warped) coordinate system, just like object detections.
        Output: (x_cm, y_cm) or None.
        """
        found = self._detect_hand_all(frame)
        if not found:
            return None
        _, _, (x_cm, y_cm) = found
        return (x_cm, y_cm)

    # ---------- Init / geometry ----------
    def __init__(self, src_points=None, dst_points=None):
        self.src_points = src_points or np.array([
            [528, 99],    # top-left
            [1373, 136],  # top-right
            [1601, 922],  # bottom-right
            [239, 866],   # bottom-left
        ], dtype=np.float32)

        self.dst_points = dst_points or np.array([
            [0, 0],
            [400, 0],
            [400, 400],
            [0, 400]
        ], dtype=np.float32)

        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.center_history: deque = deque(maxlen=5)

        # min center threshold in centimeters (warped frame is 400x400 -> 30cm x 30cm)
        self.min_center_cm = (6.0, 6.0)  # (min_x_cm, min_y_cm)

    def pixel_to_warped(self, x: int, y: int) -> Tuple[float, float]:
        """Map original pixel -> warped (400x400) subpixel coordinate."""
        pts = np.array([[[float(x), float(y)]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts, self.H)
        return float(warped[0, 0, 0]), float(warped[0, 0, 1])

    def warped_to_cm(self, x_warped: float, y_warped: float) -> Tuple[float, float]:
        """Convert warped 400x400 (y down) -> centimetres (y up)."""
        cx_cm = (x_warped * 30.0) / 400.0
        cy_cm = ((400.0 - y_warped) * 30.0) / 400.0
        return cx_cm, cy_cm

    # ---------- Detection utilities ----------
    def _apply_nms(self, rects_scores: List[Tuple[int,int,int,int,float]], iou_thr: float = 0.35
                   ) -> List[Tuple[int,int,int,int,float]]:
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
            centers = [((x + w)//2, (y + h)//2) for (x,y,w,h) in rects]
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

    def _white_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 50, 255], dtype=np.uint8)
        white = cv2.inRange(hsv, lower_white, upper_white)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        return white

    def _black_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([180, 255, 50], dtype=np.uint8)
        black = cv2.inRange(hsv, lower_black, upper_black)
        black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
        black = cv2.morphologyEx(black, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        return black

    def _find_filtered_rects(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (3,3))
        canny = cv2.Canny(gray, 100, 200)
        canny = cv2.dilate(canny, np.ones((3,3), np.uint8), iterations=1)
        red = self._red_mask(frame)
        white = self._white_mask(frame)
        black = self._black_mask(frame)
        combo = cv2.bitwise_or(canny, cv2.bitwise_or(red, cv2.bitwise_or(white, black)))
        contours, _ = cv2.findContours(combo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Tuple[int,int,int,int,float]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 2000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 30 or h < 30:
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
            roi_white = white[y:y+h, x:x+w]
            white_cov = float(np.count_nonzero(roi_white)) / (rect_area + 1e-6)
            roi_black = black[y:y+h, x:x+w]
            black_cov = float(np.count_nonzero(roi_black)) / (rect_area + 1e-6)
            if extent < 0.20:
                continue
            if solidity < 0.38:
                continue
            if ar < 0.25 or ar > 5.0:
                continue
            if circularity < 0.040:
                continue
            if edge_density < 0.015 and red_cov < 0.10 and white_cov < 0.10 and black_cov < 0.10:
                continue
            score = (area / 30000.0) + 0.6*red_cov + 0.6*white_cov + 0.6*black_cov + 0.4*edge_density + 0.2*extent
            score = float(min(1.0, score))
            candidates.append((x, y, w, h, float(score)))
        candidates = self._apply_nms(candidates, 0.35)
        rects = [(x,y,w,h) for (x,y,w,h,_) in candidates]
        rects = self._persistence_filter(rects)
        return rects

    def draw_automatic_detections(self, frame: np.ndarray, result: np.ndarray):
        """
        frame/result: warped 400x400 view.
        Skips drawing if center is inside the 6cm x 6cm corner.
        """
        rects = self._find_filtered_rects(frame)
        if isinstance(rects, tuple):
            rects = rects[0]
        for i, (x, y, w, h) in enumerate(rects):
            cx, cy = x + w // 2, y + h // 2
            cx_cm, cy_cm = self.warped_to_cm(float(cx), float(cy))
            if cx_cm < self.min_center_cm[0] and cy_cm < self.min_center_cm[1]:
                continue
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, f"{i+1}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.circle(result, (cx, cy), 3, (0,255,255), -1)

    def get_detections(self, frame: np.ndarray) -> List[Tuple[int, float, float]]:
        """Return list of (index, x_cm, y_cm) for objects, in warped cm coords."""
        warped = cv2.warpPerspective(frame, self.H, (400, 400))
        rects = self._find_filtered_rects(warped)
        if isinstance(rects, tuple):
            rects = rects[0]
        out = []
        for i, (x, y, w, h) in enumerate(rects):
            cx = x + w // 2
            cy = y + h // 2
            cx_cm, cy_cm = self.warped_to_cm(float(cx), float(cy))
            if cx_cm < self.min_center_cm[0] and cy_cm < self.min_center_cm[1]:
                continue
            out.append((i+1, cx_cm, cy_cm))
        return out

    def get_hand_in_cm(self, frame: np.ndarray
                       ) -> Optional[Tuple[Tuple[int,int], Tuple[float,float], Tuple[float,float]]]:
        """
        Returns ((hx,hy) original px, (wx,wy) warped px, (x_cm,y_cm) centimetres).
        """
        return self._detect_hand_all(frame)

    def save_centers(self, centers: List[Tuple[int,float,float]], path: str = "pipeline_output/centers.txt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for obj_num, cx, cy in centers:
                f.write(f"{obj_num},{cx:.2f},{cy:.2f}\n")

    # ---------- Main loop ----------
    def run_pipeline(self, camera_id: int = 0):
        # Try a few common camera IDs, starting with the requested one
        try_ids = [camera_id, 1, 2, 0]
        cap = None
        for cam_id in try_ids:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret:
                    break
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

            # Warp once for the right panel
            warped = cv2.warpPerspective(frame, self.H, (400, 400))
            warped_with_boxes = warped.copy()
            self.draw_automatic_detections(warped, warped_with_boxes)

            # ---- HAND (in cm) ----
            hand_cm = self.detect_hand(frame)  # now returns (x_cm, y_cm)
            if hand_cm is not None:
                x_cm, y_cm = hand_cm
                print(f"Hand (cm): ({x_cm:.2f}, {y_cm:.2f})")

                # draw on the warped panel at the corresponding pixel
                wx = int((x_cm * 400.0) / 30.0)
                wy = int(400.0 - (y_cm * 400.0) / 30.0)
                if 0 <= wx < 400 and 0 <= wy < 400:
                    cv2.circle(warped_with_boxes, (wx, wy), 8, (255, 0, 0), -1)
                    cv2.putText(warped_with_boxes, "HAND", (wx+8, wy-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            # Left = resized original, Right = warped with boxes/hand
            h = 400
            scale = h / frame.shape[0]
            original_resized = cv2.resize(frame, None, fx=scale, fy=scale)
            combined = np.hstack((original_resized, warped_with_boxes))

            # Objects (cm)
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
    parser = argparse.ArgumentParser(description="Complete Pipeline: Homography + Object Detection + Hand (cm)")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    args = parser.parse_args()
    pipeline = CompletePipeline()
    pipeline.run_pipeline(args.camera)

if __name__ == "__main__":
    main()
