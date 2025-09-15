# Raspberry Pi Flame Detection - Performance Optimization Guide

## System Requirements
- Raspberry Pi 4 (recommended) or Pi 3B+
- USB Camera (preferably with MJPEG support)
- Arduino Uno connected via USB
- Python 3.7+

## Installation Steps

### 1. Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python packages
pip install ultralytics opencv-python pyserial numpy

# Install system dependencies for OpenCV
sudo apt install -y python3-opencv libopencv-dev

# For camera optimization
sudo apt install -y v4l-utils
```

### 2. Camera Optimization
```bash
# List available cameras
v4l2-ctl --list-devices

# Check camera capabilities
v4l2-ctl --device=/dev/video0 --list-formats-ext

# Set camera to MJPEG if supported
v4l2-ctl --device=/dev/video0 --set-fmt-video=width=640,height=480,pixelformat=MJPG
```

### 3. System Optimization
```bash
# Increase GPU memory split
sudo raspi-config
# Advanced Options > Memory Split > 128

# Enable camera
sudo raspi-config
# Interface Options > Camera > Enable

# Optimize boot config
echo 'gpu_mem=128' | sudo tee -a /boot/config.txt
echo 'dtparam=i2c_arm=on' | sudo tee -a /boot/config.txt
echo 'dtparam=spi=on' | sudo tee -a /boot/config.txt
```

## Performance Features

### 1. Multi-threading Architecture
- **Capture Thread**: Dedicated thread for camera capture
- **Serial Thread**: Non-blocking Arduino communication
- **Main Thread**: YOLO inference and display

### 2. Frame Processing Optimization
- **Frame Skipping**: Process every 2nd frame (configurable)
- **Fixed Input Size**: 640x480 for consistent performance
- **Buffer Management**: Small queue to prevent lag
- **MJPEG Codec**: Hardware-accelerated when available

### 3. YOLO Optimization
- **Model Fusion**: Fuses conv and bn layers for speed
- **CPU Inference**: Optimized for Raspberry Pi
- **Warm-up**: Pre-loads model for consistent timing
- **Confidence Threshold**: 0.4 for balance of speed/accuracy

### 4. Serial Communication
- **Non-blocking**: Won't freeze on Arduino issues
- **Queue-based**: Prevents data loss
- **Error Handling**: Continues operation if Arduino disconnects

## Configuration Options

### Adjust Performance (in optimized_flame_detection.py)
```python
# Target FPS (lower = more stable)
self.TARGET_FPS = 10

# Frame skip (higher = faster, lower accuracy)
self.FRAME_SKIP = 2

# Input size (smaller = faster)
self.INPUT_SIZE = (640, 480)

# Confidence threshold (higher = fewer false positives)
self.CONFIDENCE_THRESHOLD = 0.4
```

### Camera Settings
```python
# In _setup_camera method, adjust:
cap.set(cv2.CAP_PROP_FPS, 15)        # Camera FPS
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer size
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto-exposure
```

## Troubleshooting

### Low FPS Issues
1. **Reduce input size**: Change to (480, 360) or (320, 240)
2. **Increase frame skip**: Set FRAME_SKIP to 3 or 4
3. **Lower camera FPS**: Set to 10 or 5
4. **Check CPU usage**: `htop` - should be <80%

### Camera Issues
```bash
# Check camera connection
lsusb | grep -i camera

# Test camera directly
ffplay /dev/video0

# Check camera settings
v4l2-ctl --device=/dev/video0 --all
```

### Serial Issues
```bash
# Check Arduino connection
ls /dev/ttyUSB* /dev/ttyACM*

# Test serial communication
screen /dev/ttyUSB0 9600  # or /dev/ttyACM0
```

### Memory Issues
```bash
# Check memory usage
free -h

# Check swap
sudo dphys-swapfile swapoff
sudo dphys-swapfile swapon
```

## Expected Performance
- **Raspberry Pi 4**: 8-12 FPS
- **Raspberry Pi 3B+**: 5-8 FPS
- **Memory Usage**: ~200-300MB
- **CPU Usage**: 60-80%

## Running the System
```bash
# Navigate to project directory
cd /path/to/Fire_Detection

# Run optimized detection
python3 optimized_flame_detection.py

# Monitor performance
htop  # In another terminal
```

## Arduino Integration
1. Upload `arduino_flame_receiver.ino` to Arduino
2. Connect Arduino via USB
3. The Python script will auto-detect the Arduino
4. Flame positions (LEFT/RIGHT/CENTER/NONE) sent automatically

## Tips for Best Performance
1. Use a fast SD card (Class 10 or better)
2. Ensure adequate power supply (3A for Pi 4)
3. Keep the Pi cool with heatsinks/fan
4. Close unnecessary applications
5. Use wired connection instead of WiFi when possible
6. Position camera to minimize background motion
