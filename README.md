# Hand Gesture Recognition Model

A real-time hand gesture recognition system using MediaPipe and OpenCV for gesture-based control systems and human-computer interaction.

## Features

- Real-time hand detection and tracking
- Custom gesture training capability
- Pre-built finger counting recognition
- Machine learning-based gesture classification
- Easy-to-use interface for data collection and training

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Demo (Finger Counting)
```bash
python demo.py
```
This runs a simple finger counting demo that recognizes:
- Fist (0 fingers)
- One finger
- Two fingers/Peace sign
- Three fingers
- Four fingers
- Open palm (5 fingers)

### Full Gesture Recognition System
```bash
python gesture_recognition.py
```

#### Training Custom Gestures:
1. Select option 1 to train a new model
2. For each gesture, position your hand and press 's' to save samples
3. Collect 50+ samples per gesture for better accuracy
4. The model will be automatically trained and saved

#### Real-time Recognition:
1. Load an existing model (option 2) or train a new one
2. Select option 3 for real-time recognition
3. Show gestures to the camera for classification

## Supported Gestures (Default)

- **Fist**: Closed hand
- **Open Palm**: All fingers extended
- **Thumbs Up**: Thumb pointing up
- **Peace**: Two fingers in V shape
- **Pointing**: Index finger extended

## Technical Details

### Architecture
- **Hand Detection**: MediaPipe Hands solution
- **Feature Extraction**: 21 hand landmarks (x, y, z coordinates)
- **Classification**: Random Forest Classifier
- **Real-time Processing**: OpenCV for video capture and display

### Key Components

1. **HandGestureRecognizer Class**:
   - `extract_landmarks()`: Converts hand landmarks to feature vectors
   - `collect_training_data()`: Interactive data collection
   - `train_model()`: Trains Random Forest classifier
   - `predict_gesture()`: Real-time gesture prediction

2. **SimpleGestureDemo Class**:
   - Basic finger counting without ML training
   - Immediate testing capability

## Performance

- **Detection Confidence**: 70% minimum
- **Tracking Confidence**: 50% minimum
- **Model Accuracy**: Typically 85-95% with good training data
- **Real-time FPS**: 15-30 FPS depending on hardware

## Customization

### Adding New Gestures:
1. Modify `gesture_labels` list in `HandGestureRecognizer`
2. Run training mode to collect data for new gestures
3. Retrain the model

### Adjusting Parameters:
- Detection confidence: Modify `min_detection_confidence`
- Tracking confidence: Modify `min_tracking_confidence`
- Model parameters: Adjust RandomForestClassifier settings

## Applications

- **Gesture-based Control**: Control applications with hand movements
- **Sign Language Recognition**: Extend for sign language interpretation
- **Gaming**: Gesture-controlled games
- **Accessibility**: Hands-free computer interaction
- **Smart Home**: Control IoT devices with gestures

## Troubleshooting

1. **Camera not detected**: Ensure webcam is connected and not used by other applications
2. **Poor recognition**: Ensure good lighting and clear hand visibility
3. **Low accuracy**: Collect more training samples and ensure consistent hand positioning
4. **Performance issues**: Reduce video resolution or adjust confidence thresholds

## Future Enhancements

- Dynamic gesture recognition (gesture sequences)
- Multi-hand gesture support
- Deep learning models (CNN/LSTM)
- Mobile deployment
- Integration with control systems