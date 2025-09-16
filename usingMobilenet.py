import torch
import cv2
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import torchvision.transforms as T

# --------------------------
# Load Model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2  # background + flame

# --------------------------
# Load Model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2  # background + flame

def create_model(num_classes):
    """Create SSD model with proper configuration"""
    return ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=num_classes)

try:
    print("Attempting to load custom checkpoint...")
    # First, try to load the checkpoint
    checkpoint = torch.load("mobilenet_ssd_candle_best.pth", map_location=device)
    print("Checkpoint loaded successfully")
    
    # Create model with our number of classes
    model = create_model(num_classes)
    model.to(device)
    
    # Try to load the state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Custom model loaded successfully!")
    except RuntimeError as e:
        print(f"⚠️  State dict loading failed: {e}")
        print("Attempting to load with strict=False...")
        
        # Load with strict=False to ignore mismatched layers
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)} keys")
        
        print("✅ Model loaded with some mismatches - proceeding with available weights")

except FileNotFoundError:
    print("❌ Custom checkpoint not found: mobilenet_ssd_candle_best.pth")
    print("📦 Creating model with COCO pretrained weights...")
    model = ssdlite320_mobilenet_v3_large(pretrained=True)
    model.to(device)
    print("✅ Pretrained model loaded (will detect general objects)")

except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    print("📦 Creating model with COCO pretrained weights...")
    model = ssdlite320_mobilenet_v3_large(pretrained=True)
    model.to(device)
    print("✅ Pretrained model loaded (will detect general objects)")

model.eval()
print(f"🔥 Model ready on device: {device}")

# --------------------------
# Transform for input images
# --------------------------
transform = T.Compose([
    T.ToTensor(),  # Convert OpenCV image (H,W,C) to tensor (C,H,W) and scale 0-1
])

# --------------------------
# OpenCV Video Capture
# --------------------------
cap = cv2.VideoCapture(0)  # use 0 for default camera

def detect_flame_position(boxes, frame_width):
    """Detect flame position relative to center"""
    if len(boxes) == 0:
        return "NONE"
    
    center_x = frame_width // 2
    positions = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        flame_center_x = (x1 + x2) / 2
        
        # Determine position with buffer zone
        if flame_center_x < center_x - 50:
            positions.append("LEFT")
        elif flame_center_x > center_x + 50:
            positions.append("RIGHT")
        else:
            positions.append("CENTER")
    
    return positions[0] if positions else "NONE"

# COCO class names (for pretrained model)
COCO_CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

print("Starting flame detection... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img).to(device)
    input_tensor = input_tensor.unsqueeze(0)  # batch dimension

    # --------------------------
    # Detection
    # --------------------------
    with torch.no_grad():
        outputs = model(input_tensor)

    # Extract boxes, labels, scores
    boxes = outputs[0]['boxes'].cpu()
    scores = outputs[0]['scores'].cpu()
    labels = outputs[0]['labels'].cpu()

    # --------------------------
    # Draw boxes and detect position
    # --------------------------
    valid_boxes = []
    for box, score, label in zip(boxes, scores, labels):
        # Convert tensors to Python values
        score_val = score.item()
        label_val = label.item()
        
        if score_val < 0.5:  # confidence threshold
            continue

        # Convert tensor coordinates to Python integers
        x1, y1, x2, y2 = box.int().tolist()
        
        # Determine if this is a flame detection
        is_flame = False
        label_text = ""
        
        if num_classes == 2:  # Custom flame model
            if label_val == 1:  # flame class
                is_flame = True
                label_text = f"Flame: {score_val:.2f}"
        else:  # Pretrained COCO model - look for fire-related objects
            class_name = COCO_CLASSES[label_val] if label_val < len(COCO_CLASSES) else f"Class {label_val}"
            # Consider certain objects as potential "flames" for demo purposes
            if any(keyword in class_name.lower() for keyword in ['fire', 'candle', 'light', 'torch']):
                is_flame = True
                label_text = f"Fire-like ({class_name}): {score_val:.2f}"
            else:
                # Show all detections but don't count as flame
                label_text = f"{class_name}: {score_val:.2f}"
                
        if is_flame:
            valid_boxes.append([x1, y1, x2, y2])
            color = (0, 255, 0)  # Green for flames
        else:
            color = (255, 0, 0)  # Blue for other objects
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label_text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Detect flame position (only for flame detections)
    position = detect_flame_position(valid_boxes, frame.shape[1])
    
    # Draw center line
    center_x = frame.shape[1] // 2
    cv2.line(frame, (center_x, 0), (center_x, frame.shape[0]), (255, 255, 255), 2)
    cv2.putText(frame, "CENTER", (center_x - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show position
    position_color = (0, 255, 0) if position != "NONE" else (0, 0, 255)
    cv2.putText(frame, f"FLAME: {position}", (10, frame.shape[0] - 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, position_color, 2)
    
    # Print position to console
    if position != "NONE":
        print(f"Flame detected: {position}")

    # Show image
    cv2.imshow("Flame Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
