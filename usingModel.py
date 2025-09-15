import cv2
from ultralytics import YOLO
import serial
import serial.tools.list_ports
import time

# Load your trained YOLO model
model = YOLO("best_v2.pt") 

def get_first_port(baudrate=9600, timeout=1):
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise Exception("No serial ports found!")

    for p in ports:
        print("Found port:", p.device)
    
    # Pick the first one
    return serial.Serial(port=ports[0].device, baudrate=baudrate, timeout=timeout)

# Usage
arduino = get_first_port()
print("Connected to:", arduino.port)

# Open serial connection to Arduino
time.sleep(2)  # wait for Arduino to reset


# Open webcam
cap = cv2.VideoCapture(0)   

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

def detect_flame_position(results, frame_width):
    """
    Detect flame position relative to center of frame
    Returns: 'LEFT', 'RIGHT', 'CENTER', or None if no flame detected
    """
    flame_positions = []
    
    # Get the detection results
    boxes = results[0].boxes
    
    if boxes is not None and len(boxes) > 0:
        # Calculate frame center
        center_x = frame_width // 2
        
        # Process each detection
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Calculate center of the detected flame
            flame_center_x = (x1 + x2) / 2
            
            # Determine position relative to frame center
            if flame_center_x < center_x - 50:  # Left side (with small buffer)
                flame_positions.append("Left")
            elif flame_center_x > center_x + 50:  # Right side (with small buffer)
                flame_positions.append("Right")
            else:
                flame_positions.append("Center")

    return flame_positions


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Run YOLO detection on the frame
    results = model.predict(frame, conf=0.4, verbose=False)  

    # Detect flame positions
    flame_positions = detect_flame_position(results, frame_width)
    
    # Draw detections on the frame
    annotated_frame = results[0].plot()
    
    # Draw center line for reference
    cv2.line(annotated_frame, (frame_width // 2, 0), (frame_width // 2, frame_height), (255, 255, 255), 2)
    cv2.putText(annotated_frame, "CENTER", (frame_width // 2 - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display flame positions and print to console
    if flame_positions:
        position = flame_positions[0]
        cv2.putText(annotated_frame, f"FLAME: {position}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        arduino.write((position + "\n").encode())
        # print("Sent to Arduino:", position)

        # Read echo from Arduino
        response = arduino.readline().decode().strip()
        if response:
            print("Arduino says:", response)
    else:
        arduino.write(("NONE\n").encode())
        response = arduino.readline().decode().strip()
        if response:
            print("Arduino says:", response)



    # Show the frame
    cv2.imshow("YOLO Fire Detection with Position", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
arduino.close()
