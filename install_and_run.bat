@echo off
echo Installing required packages...
pip install opencv-python
echo.
echo Starting live gesture recognition...
python live_gesture.py
pause