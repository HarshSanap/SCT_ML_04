import tkinter as tk
from tkinter import messagebox
import threading
import time
import random

class GestureCameraApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Live Gesture Recognition")
        self.root.geometry("600x400")
        self.root.configure(bg='black')
        
        self.running = False
        self.gestures = ["Fist", "One", "Two", "Three", "Four", "Five", "Peace", "Thumbs Up"]
        
        # Camera simulation frame
        self.camera_frame = tk.Frame(self.root, bg='gray', width=400, height=300)
        self.camera_frame.pack(pady=20)
        
        self.camera_label = tk.Label(self.camera_frame, text="CAMERA FEED", 
                                   bg='gray', fg='white', font=('Arial', 16))
        self.camera_label.pack(expand=True)
        
        # Gesture display
        self.gesture_label = tk.Label(self.root, text="No gesture detected", 
                                    bg='black', fg='green', font=('Arial', 20))
        self.gesture_label.pack(pady=10)
        
        # Control buttons
        button_frame = tk.Frame(self.root, bg='black')
        button_frame.pack(pady=10)
        
        self.start_btn = tk.Button(button_frame, text="Start Camera", 
                                 command=self.start_detection, bg='green', fg='white')
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(button_frame, text="Stop Camera", 
                                command=self.stop_detection, bg='red', fg='white')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_label = tk.Label(self.root, text="Camera: OFF", 
                                   bg='black', fg='red', font=('Arial', 12))
        self.status_label.pack(pady=5)
    
    def start_detection(self):
        self.running = True
        self.status_label.config(text="Camera: ON - Detecting gestures...", fg='green')
        self.camera_label.config(text="🎥 LIVE FEED ACTIVE", bg='darkgreen')
        
        # Start gesture detection thread
        threading.Thread(target=self.detect_gestures, daemon=True).start()
    
    def stop_detection(self):
        self.running = False
        self.status_label.config(text="Camera: OFF", fg='red')
        self.camera_label.config(text="CAMERA FEED", bg='gray')
        self.gesture_label.config(text="No gesture detected")
    
    def detect_gestures(self):
        while self.running:
            # Simulate gesture detection
            gesture = random.choice(self.gestures)
            confidence = random.uniform(0.75, 0.98)
            
            self.gesture_label.config(text=f"Detected: {gesture} ({confidence:.1%})")
            
            # Simulate processing time
            time.sleep(1.5)
    
    def run(self):
        messagebox.showinfo("Gesture Recognition", 
                          "This simulates live camera gesture detection.\n\n" +
                          "In real implementation:\n" +
                          "• Camera would capture your hand\n" +
                          "• AI would analyze hand landmarks\n" +
                          "• Gestures detected in real-time")
        self.root.mainloop()

if __name__ == "__main__":
    app = GestureCameraApp()
    app.run()