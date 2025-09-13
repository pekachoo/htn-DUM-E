import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configure hand detection
hands = mp_hands.Hands(
    static_image_mode=False,       # Video feed
    max_num_hands=2,               # Detect up to 2 hands
    min_detection_confidence=0.7,  # Confidence threshold
    min_tracking_confidence=0.7    # Tracking threshold
)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the image horizontally for a selfie-view
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw joints
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )
            
            # Extract landmark coordinates
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
    
    cv2.imshow("Hand Joint Pose Estimation", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
