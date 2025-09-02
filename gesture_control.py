import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.last_action_time = 0
        self.action_cooldown = 1.0  # 1 second cooldown between actions
        
    def get_finger_status(self, landmarks):
        """Get status of each finger (up/down)"""
        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [3, 6, 10, 14, 18]
        
        fingers = []
        
        # Thumb
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
        
        return fingers
    
    def detect_control_gesture(self, landmarks):
        """Detect control gestures and return action"""
        fingers = self.get_finger_status(landmarks)
        finger_count = sum(fingers)
        
        current_time = time.time()
        if current_time - self.last_action_time < self.action_cooldown:
            return None, "Cooldown"
        
        # Define gesture controls
        if finger_count == 0:  # Fist
            action = "pause_play"
            description = "Play/Pause"
        elif finger_count == 1 and fingers[1] == 1:  # Index finger
            action = "volume_up"
            description = "Volume Up"
        elif finger_count == 1 and fingers[4] == 1:  # Pinky
            action = "volume_down"
            description = "Volume Down"
        elif finger_count == 2 and fingers[1] == 1 and fingers[2] == 1:  # Peace sign
            action = "next_track"
            description = "Next Track"
        elif finger_count == 1 and fingers[0] == 1:  # Thumbs up
            action = "previous_track"
            description = "Previous Track"
        elif finger_count == 5:  # Open palm
            action = "mute"
            description = "Mute/Unmute"
        else:
            return None, "Unknown"
        
        self.last_action_time = current_time
        return action, description
    
    def execute_action(self, action):
        """Execute the control action"""
        try:
            if action == "pause_play":
                pyautogui.press('space')
            elif action == "volume_up":
                pyautogui.press('volumeup')
            elif action == "volume_down":
                pyautogui.press('volumedown')
            elif action == "next_track":
                pyautogui.press('nexttrack')
            elif action == "previous_track":
                pyautogui.press('prevtrack')
            elif action == "mute":
                pyautogui.press('volumemute')
            return True
        except:
            return False
    
    def run_controller(self):
        """Run the gesture controller"""
        cap = cv2.VideoCapture(0)
        
        print("Gesture Controller Started!")
        print("Gestures:")
        print("- Fist: Play/Pause")
        print("- Index finger: Volume Up")
        print("- Pinky: Volume Down")
        print("- Peace sign: Next Track")
        print("- Thumbs up: Previous Track")
        print("- Open palm: Mute/Unmute")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            action_text = "No hand detected"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    action, description = self.detect_control_gesture(hand_landmarks.landmark)
                    
                    if action:
                        success = self.execute_action(action)
                        action_text = f"Action: {description} {'✓' if success else '✗'}"
                    else:
                        action_text = f"Gesture: {description}"
            
            # Display UI
            cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 100), (0, 0, 0), -1)
            cv2.putText(frame, "Gesture Controller", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, action_text, (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Gesture Controller', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = GestureController()
    controller.run_controller()