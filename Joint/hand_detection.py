import cv2
import mediapipe as mp
import numpy as np
import math

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
        
    def calculate_hand_orientation(self, landmarks):
        # Get key landmarks
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        middle_tip = landmarks[12]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate palm normal vector (wrist to middle finger)
        palm_vector = np.array([middle_mcp.x - wrist.x, middle_mcp.y - wrist.y, middle_mcp.z - wrist.z])
        
        # Calculate pitch (up/down)
        pitch = math.atan2(palm_vector[1], palm_vector[2])
        
        # Calculate yaw (left/right)
        yaw = math.atan2(palm_vector[0], palm_vector[2])
        
        # Calculate roll (rotation around palm axis)
        # Use the angle between thumb and index finger projection
        thumb_to_index = np.array([index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y])
        roll = math.atan2(thumb_to_index[1], thumb_to_index[0])
        
        return pitch, yaw, roll
    
    def calculate_claw_state(self, landmarks):
        # Get finger tips
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Calculate distances from thumb to other fingertips
        distances = []
        for tip in [index_tip, middle_tip, ring_tip, pinky_tip]:
            dist = math.sqrt((thumb_tip.x - tip.x)**2 + (thumb_tip.y - tip.y)**2 + (thumb_tip.z - tip.z)**2)
            distances.append(dist)
        
        # Average distance indicates claw state
        avg_distance = np.mean(distances)
        
        # Threshold for open/close (adjust based on testing)
        claw_threshold = 0.05
        claw_open = avg_distance > claw_threshold
        
        return claw_open, avg_distance
    
    def normalize_to_servo_range(self, angle, min_angle=-math.pi/2, max_angle=math.pi/2):
        # Normalize angle to 0-180 servo range
        normalized = (angle - min_angle) / (max_angle - min_angle)
        servo_angle = int(normalized * 180)
        return max(0, min(180, servo_angle))
    
    def process_hand_data(self, landmarks):
        # Calculate orientation
        pitch, yaw, roll = self.calculate_hand_orientation(landmarks)
        
        # Calculate claw state
        claw_open, claw_distance = self.calculate_claw_state(landmarks)
        
        # Get hand center position
        wrist = landmarks[0]
        h, w, c = 480, 640, 3  # Default frame size
        
        # Convert to pixel coordinates
        x = int(wrist.x * w)
        y = int(wrist.y * h)
        z = int((1 - wrist.z) * 100)  # Convert z to 0-100 range
        
        # Normalize angles to servo range
        pitch_servo = self.normalize_to_servo_range(pitch)
        yaw_servo = self.normalize_to_servo_range(yaw)
        roll_servo = self.normalize_to_servo_range(roll)
        
        return {
            'position': {'x': x, 'y': y, 'z': z},
            'orientation': {'pitch': pitch_servo, 'yaw': yaw_servo, 'roll': roll_servo},
            'claw': {'open': claw_open, 'distance': claw_distance}
        }
    
    def run_hand_tracking(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Hand Robot Controller Started")
        print("Press 'q' to quit")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                    )
                    
                    # Process hand data for robot control
                    hand_data = self.process_hand_data(hand_landmarks.landmark)
                    
                    # Display robot control data
                    cv2.putText(frame, f"Position: X:{hand_data['position']['x']} Y:{hand_data['position']['y']} Z:{hand_data['position']['z']}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Orientation: Pitch:{hand_data['orientation']['pitch']} Yaw:{hand_data['orientation']['yaw']} Roll:{hand_data['orientation']['roll']}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Claw: {'OPEN' if hand_data['claw']['open'] else 'CLOSED'} ({hand_data['claw']['distance']:.3f})", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Print robot commands (for debugging)
                    print(f"Robot Commands - X:{hand_data['position']['x']} Y:{hand_data['position']['y']} Z:{hand_data['position']['z']} "
                          f"Pitch:{hand_data['orientation']['pitch']} Yaw:{hand_data['orientation']['yaw']} Roll:{hand_data['orientation']['roll']} "
                          f"Claw:{'OPEN' if hand_data['claw']['open'] else 'CLOSED'}")
            
            cv2.imshow("Hand Robot Controller", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandRobotController()
    controller.run_hand_tracking()