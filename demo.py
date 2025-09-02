import cv2
import mediapipe as mp
import numpy as np

class SimpleGestureDemo:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def count_fingers(self, landmarks):
        """Simple finger counting gesture recognition"""
        # Finger tip and pip landmarks
        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [3, 6, 10, 14, 18]
        
        fingers = []
        
        # Thumb (different logic due to orientation)
        if landmarks[tip_ids[0]].x > landmarks[pip_ids[0]].x:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers
        for i in range(1, 5):
            if landmarks[tip_ids[i]].y < landmarks[pip_ids[i]].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return sum(fingers)
    
    def detect_gesture(self, landmarks):
        """Detect basic gestures"""
        finger_count = self.count_fingers(landmarks)
        
        if finger_count == 0:
            return "Fist"
        elif finger_count == 1:
            return "One"
        elif finger_count == 2:
            return "Two/Peace"
        elif finger_count == 3:
            return "Three"
        elif finger_count == 4:
            return "Four"
        elif finger_count == 5:
            return "Open Palm"
        else:
            return "Unknown"
    
    def run_demo(self):
        """Run the gesture recognition demo"""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    gesture = self.detect_gesture(hand_landmarks.landmark)
                    
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Hand Gesture Demo', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = SimpleGestureDemo()
    print("Starting Hand Gesture Recognition Demo...")
    print("Show your hand to the camera and try different gestures!")
    demo.run_demo()