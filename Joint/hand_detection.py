import cv2
import mediapipe as mp
import numpy as np
import math
import time

class HandRobotController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        # Baseline orientation (assume initial hand faces camera)
        self.baseline_pitch_deg = 0.0
        self.baseline_yaw_deg = 0.0
        self.baseline_roll_deg = 0.0
        self.deadzone_deg = 5.0
        
    def calculate_hand_orientation(self, landmarks):
        # Key landmarks
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        middle_mcp = landmarks[9]
        
        # Vectors with y-up convention
        fwd = np.array([middle_mcp.x - wrist.x, -(middle_mcp.y - wrist.y), middle_mcp.z - wrist.z])
        across3 = np.array([index_mcp.x - pinky_mcp.x, -(index_mcp.y - pinky_mcp.y), index_mcp.z - pinky_mcp.z])
        
        # Normalize
        fwd_n = fwd / (np.linalg.norm(fwd) + 1e-9)
        across_n = across3 / (np.linalg.norm(across3) + 1e-9)
        
        # For a hand facing the camera, forward points toward camera (negative Z in many CV setups).
        # Use -fwd_z as reference so yaw/pitch=0 when hand faces camera.
        yaw_deg = math.degrees(math.atan2(fwd_n[0], -fwd_n[2]))   # left/right
        pitch_deg = math.degrees(math.atan2(fwd_n[1], -fwd_n[2])) # up/down
        # Roll from across vector projected to image plane
        roll_deg = math.degrees(math.atan2(across_n[1], across_n[0]))
        
        return pitch_deg, yaw_deg, roll_deg

    def calculate_finger_pitches(self, landmarks):
        # Build palm normal from fwd and across
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        middle_mcp = landmarks[9]
        fwd = np.array([middle_mcp.x - wrist.x, -(middle_mcp.y - wrist.y), middle_mcp.z - wrist.z])
        across3 = np.array([index_mcp.x - pinky_mcp.x, -(index_mcp.y - pinky_mcp.y), index_mcp.z - pinky_mcp.z])
        fwd_n = fwd / (np.linalg.norm(fwd) + 1e-9)
        across_n = across3 / (np.linalg.norm(across3) + 1e-9)
        palm_normal = np.cross(across_n, fwd_n)
        palm_normal_n = palm_normal / (np.linalg.norm(palm_normal) + 1e-9)
        
        # Fingers indices (MCP, PIP, DIP, TIP)
        fingers = [
            (5, 6, 7, 8),   # index
            (9, 10, 11, 12),# middle
            (13, 14, 15, 16),# ring
            (17, 18, 19, 20) # pinky
        ]
        prox_pitches = []
        tip_pitches = []
        for mcp_i, pip_i, dip_i, tip_i in fingers:
            mcp = landmarks[mcp_i]
            pip = landmarks[pip_i]
            tip = landmarks[tip_i]
            # y-up vectors
            prox_dir = np.array([pip.x - mcp.x, -(pip.y - mcp.y), pip.z - mcp.z])
            tip_dir = np.array([tip.x - mcp.x, -(tip.y - mcp.y), tip.z - mcp.z])
            prox_dir_n = prox_dir / (np.linalg.norm(prox_dir) + 1e-9)
            tip_dir_n = tip_dir / (np.linalg.norm(tip_dir) + 1e-9)
            # Angle to palm normal (0=straight out, 90=in-plane)
            dot_prox = float(np.clip(np.dot(prox_dir_n, palm_normal_n), -1.0, 1.0))
            dot_tip = float(np.clip(np.dot(tip_dir_n, palm_normal_n), -1.0, 1.0))
            angle_prox = math.degrees(math.acos(dot_prox))
            angle_tip = math.degrees(math.acos(dot_tip))
            # Convert to pitch away from palm: 0..90 (0=in-plane, 90=straight out)
            pitch2_val = max(0.0, 90.0 - angle_prox)
            pitch3_val = max(0.0, 90.0 - angle_tip)
            prox_pitches.append(pitch2_val)
            tip_pitches.append(pitch3_val)
        # Average across fingers
        pitch2 = float(np.mean(prox_pitches)) if prox_pitches else 0.0
        pitch3 = float(np.mean(tip_pitches)) if tip_pitches else 0.0
        return pitch2, pitch3
    
    def normalize_to_servo_range(self, angle_deg: float, min_deg: float = -90.0, max_deg: float = 90.0) -> int:
        angle_clamped = max(min_deg, min(max_deg, angle_deg))
        normalized = (angle_clamped - min_deg) / (max_deg - min_deg)
        return int(round(normalized * 180.0))
    
    def calculate_claw_state(self, landmarks):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        distances = []
        for tip in [index_tip, middle_tip, ring_tip, pinky_tip]:
            dist = math.sqrt((thumb_tip.x - tip.x)**2 + (thumb_tip.y - tip.y)**2 + (thumb_tip.z - tip.z)**2)
            distances.append(dist)
        avg_distance = float(np.mean(distances)) if distances else 0.0
        claw_threshold = 0.05
        claw_open = avg_distance > claw_threshold
        return claw_open, avg_distance
    
    def _apply_baseline_and_deadzone(self, angle_deg: float, baseline_deg: float) -> float:
        delta = angle_deg - baseline_deg
        if abs(delta) < self.deadzone_deg:
            return 0.0
        return delta
    
    def process_hand_data(self, landmarks):
        pitch1_raw, yaw_deg_raw, roll_deg_raw = self.calculate_hand_orientation(landmarks)
        # Finger pitches
        pitch2_raw, pitch3_raw = self.calculate_finger_pitches(landmarks)
        
        # Apply baseline/deadzone to pitch1 only (hand tilt)
        pitch1 = self._apply_baseline_and_deadzone(pitch1_raw, self.baseline_pitch_deg)
        yaw_deg = self._apply_baseline_and_deadzone(yaw_deg_raw, self.baseline_yaw_deg)
        roll_deg_int = int(round(roll_deg_raw))
        
        # Servo mapping (pitch1/yaw only)
        pitch_servo = self.normalize_to_servo_range(pitch1)
        yaw_servo = self.normalize_to_servo_range(yaw_deg)
        
        claw_open, claw_distance = self.calculate_claw_state(landmarks)
        
        wrist = landmarks[0]
        h, w = 480, 640
        x = int(wrist.x * w)
        y = int(wrist.y * h)
        z = int((1 - wrist.z) * 100)
        
        return {
            'position': {'x': x, 'y': y, 'z': z},
            'pitches_deg': {'p1_hand': pitch1, 'p2_fingers': pitch2_raw, 'p3_tips': pitch3_raw},
            'orientation_raw_deg': {'pitch1': pitch1_raw, 'yaw': yaw_deg_raw, 'roll': roll_deg_raw},
            'orientation_servo': {'pitch1': pitch_servo, 'yaw': yaw_servo},
            'roll_int': roll_deg_int,
            'claw': {'open': claw_open, 'distance': claw_distance}
        }

    def _open_camera(self, preferred_id: int = 0):
        for cam_id in [preferred_id, 1, 2, 0]:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap
                cap.release()
        return None
    
    def run_hand_tracking(self):
        cap = self._open_camera(0)
        if cap is None:
            err_img = np.zeros((200, 600, 3), dtype=np.uint8)
            cv2.putText(err_img, "Could not open any camera (0/1/2)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Hand Robot Controller - Error", err_img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            return
        
        cv2.namedWindow("Hand Robot Controller", cv2.WINDOW_NORMAL)
        consecutive_failures = 0
        max_failures = 10
        latest_raw = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    cap.release()
                    cap = self._open_camera(0)
                    if cap is None:
                        break
                    consecutive_failures = 0
                time.sleep(0.05)
                continue
            consecutive_failures = 0
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                    )
                    hand_data = self.process_hand_data(hand_landmarks.landmark)
                    latest_raw = hand_data['orientation_raw_deg']
                    cv2.putText(frame, f"X:{hand_data['position']['x']} Y:{hand_data['position']['y']} Z:{hand_data['position']['z']}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Pitches P1/P2/P3: {hand_data['pitches_deg']['p1_hand']:.1f}/{hand_data['pitches_deg']['p2_fingers']:.1f}/{hand_data['pitches_deg']['p3_tips']:.1f}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Yaw(deg):{hand_data['orientation_raw_deg']['yaw']:.1f}  Roll(int):{hand_data['roll_int']}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 2)
                    cv2.putText(frame, f"Servo P1/Y:{hand_data['orientation_servo']['pitch1']}/{hand_data['orientation_servo']['yaw']}",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
                    cv2.putText(frame, f"Claw: {'OPEN' if hand_data['claw']['open'] else 'CLOSED'} (d={hand_data['claw']['distance']:.3f})",
                                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, "Press 'b' to set baseline (zero)", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 180), 2)
            else:
                cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Hand Robot Controller", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b') and latest_raw is not None:
                self.baseline_pitch_deg = latest_raw['pitch1']
                self.baseline_yaw_deg = latest_raw['yaw']
                self.baseline_roll_deg = latest_raw['roll']
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandRobotController()
    controller.run_hand_tracking()