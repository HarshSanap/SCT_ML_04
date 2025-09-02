try:
    import cv2
    import numpy as np
    
    def detect_hand_gesture(frame):
        # Convert to HSV for better hand detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (hand)
            hand_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(hand_contour) > 3000:
                # Get convex hull and defects
                hull = cv2.convexHull(hand_contour, returnPoints=False)
                defects = cv2.convexityDefects(hand_contour, hull)
                
                if defects is not None:
                    finger_count = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        if d > 10000:  # Filter small defects
                            finger_count += 1
                    
                    # Map finger count to gestures
                    gestures = {0: "Fist", 1: "One", 2: "Two/Peace", 3: "Three", 4: "Four", 5: "Five"}
                    return gestures.get(min(finger_count + 1, 5), "Unknown")
        
        return "No Hand"
    
    def main():
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Camera opened! Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally (mirror effect)
            frame = cv2.flip(frame, 1)
            
            # Detect gesture
            gesture = detect_hand_gesture(frame)
            
            # Draw gesture text on frame
            cv2.putText(frame, f"Gesture: {gesture}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(frame, "Show your hand to camera", (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show YOUR live video feed
            cv2.imshow('Real Camera - Gesture Detection', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    if __name__ == "__main__":
        main()

except ImportError:
    print("OpenCV not installed. Installing now...")
    import subprocess
    import sys
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
        print("OpenCV installed! Please run the script again.")
    except:
        print("Could not install OpenCV automatically.")
        print("Please install manually: pip install opencv-python")