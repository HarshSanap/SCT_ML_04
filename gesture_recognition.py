import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.model = None
        self.gesture_labels = ['fist', 'open_palm', 'thumbs_up', 'peace', 'pointing']
        
    def extract_landmarks(self, hand_landmarks):
        """Extract normalized hand landmarks as features"""
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks)
    
    def collect_training_data(self, gesture_name, num_samples=100):
        """Collect training data for a specific gesture"""
        cap = cv2.VideoCapture(0)
        data = []
        count = 0
        
        print(f"Collecting data for '{gesture_name}'. Press 's' to save sample, 'q' to quit")
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        landmarks = self.extract_landmarks(hand_landmarks)
                        data.append(landmarks)
                        count += 1
                        print(f"Saved sample {count}/{num_samples}")
            
            cv2.putText(frame, f"{gesture_name}: {count}/{num_samples}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Data Collection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return np.array(data)
    
    def train_model(self):
        """Train the gesture recognition model"""
        X, y = [], []
        
        # Collect data for each gesture
        for i, gesture in enumerate(self.gesture_labels):
            print(f"\nCollecting data for gesture: {gesture}")
            data = self.collect_training_data(gesture, 50)
            if len(data) > 0:
                X.extend(data)
                y.extend([i] * len(data))
        
        if len(X) == 0:
            print("No training data collected!")
            return
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")
        
        # Save model
        with open('gesture_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print("Model saved as 'gesture_model.pkl'")
    
    def load_model(self, model_path='gesture_model.pkl'):
        """Load pre-trained model"""
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully")
            return True
        return False
    
    def predict_gesture(self, hand_landmarks):
        """Predict gesture from hand landmarks"""
        if self.model is None:
            return "No model loaded"
        
        landmarks = self.extract_landmarks(hand_landmarks)
        prediction = self.model.predict([landmarks])[0]
        confidence = max(self.model.predict_proba([landmarks])[0])
        
        return self.gesture_labels[prediction], confidence
    
    def real_time_recognition(self):
        """Real-time gesture recognition"""
        if self.model is None:
            print("Please load or train a model first!")
            return
        
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
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    gesture, confidence = self.predict_gesture(hand_landmarks)
                    
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Hand Gesture Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    recognizer = HandGestureRecognizer()
    
    while True:
        print("\nHand Gesture Recognition System")
        print("1. Train new model")
        print("2. Load existing model")
        print("3. Start real-time recognition")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            recognizer.train_model()
        elif choice == '2':
            if recognizer.load_model():
                print("Model loaded successfully!")
            else:
                print("Model file not found!")
        elif choice == '3':
            recognizer.real_time_recognition()
        elif choice == '4':
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()