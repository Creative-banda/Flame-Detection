# pip install tflite-runtime

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load TFLite model
interpreter = tflite.Interpreter(model_path="best_v2_float16.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Camera capture
cap = cv2.VideoCapture(0)  # adjust if not /dev/video0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found")
        break

    # Preprocess
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img, axis=0).astype(np.uint8)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get predictions
    output = interpreter.get_tensor(output_details[0]['index'])
    print("Detections:", output)

    cv2.imshow("Flame Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
