import cv2
from ultralytics import YOLO
import serial
import serial.tools.list_ports
import time

# ---------------- YOLO Model ---------------- #
model = YOLO("best_v2.pt")

# ---------------- Serial Setup ---------------- #
def get_first_port(baudrate=9600, timeout=0):  # non-blocking
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise Exception("No serial ports found!")
    for p in ports:
        print("Found port:", p.device)
    return serial.Serial(port=ports[0].device, baudrate=baudrate, timeout=timeout)

arduino = get_first_port()
print("Connected to:", arduino.port)
time.sleep(2)  # allow Arduino reset

# ---------------- Camera Setup ---------------- #
def get_camera():
    for i in range(5):  # try first 5 indexes
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)  # force V4L2 for Pi
        if cap.isOpened():
            print(f"Camera found at index {i}")
            # Optimize camera settings
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # if supported
            return cap
    raise RuntimeError("No camera found")

cap = get_camera()

# ---------------- Helper ---------------- #
def detect_flame_position(results, frame_width):
    flame_positions = []
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        center_x = frame_width // 2
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            flame_center_x = (x1 + x2) / 2
            if flame_center_x < center_x - 50:
                flame_positions.append("Left")
            elif flame_center_x > center_x + 50:
                flame_positions.append("Right")
            else:
                flame_positions.append("Center")
    return flame_positions

# ---------------- Main Loop ---------------- #
while True:
    ret, frame = cap.read()
    if not ret:
        continue  # don’t break, just skip frame

    # Resize before inference
    resized = cv2.resize(frame, (640, 480))

    # Faster inference call
    results = model(resized, conf=0.4, verbose=False)

    # Get detections
    frame_height, frame_width = resized.shape[:2]
    flame_positions = detect_flame_position(results, frame_width)

    # Draw results
    annotated_frame = results[0].plot()
    cv2.line(annotated_frame, (frame_width // 2, 0), (frame_width // 2, frame_height), (255, 255, 255), 2)
    cv2.putText(annotated_frame, "CENTER", (frame_width // 2 - 30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Send to Arduino (non-blocking)
    position = flame_positions[0] if flame_positions else "NONE"
    arduino.write((position + "\n").encode())

    # Non-blocking read
    response = arduino.readline().decode().strip()
    if response:
        print("Arduino:", response)

    # Show window
    cv2.imshow("YOLO Fire Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- Cleanup ---------------- #
cap.release()
cv2.destroyAllWindows()
arduino.close()
