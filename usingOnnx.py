#!/usr/bin/env python3
"""
Optimized YOLOv8 Flame Detection for Raspberry Pi
- Real-time inference with Arduino communication
- Optimized for 5-10 FPS stable performance
- Non-blocking serial communication
- Robust error handling
"""

import cv2
import numpy as np
import onnxruntime as ort
import serial
import serial.tools.list_ports
import time
import threading
from queue import Queue, Empty
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedFlameDetector:
    def __init__(self, model_path="best_v2.onnx", camera_index=0):
        # Performance settings
        self.TARGET_FPS = 10
        self.FRAME_SKIP = 2  # Process every 2nd frame for speed
        self.INPUT_SIZE = (640, 480)
        self.CONFIDENCE_THRESHOLD = 0.4
        self.NMS_THRESHOLD = 0.45
        
        # Initialize components
        self.session = self._load_onnx_model(model_path)
        self.camera = self._setup_camera(camera_index)
        self.arduino = self._setup_arduino()
        
        # Threading for non-blocking operations
        self.serial_queue = Queue(maxsize=10)
        self.frame_queue = Queue(maxsize=3)  # Small buffer to prevent lag
        self.running = True
        
        # Performance tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.last_fps_update = time.time()
        self.fps_frame_count = 0
        
        # Start background threads
        self._start_threads()
    
    def _load_onnx_model(self, model_path):
        """Load ONNX model with optimizations"""
        try:
            logger.info(f"Loading ONNX model: {model_path}")
            
            # ONNX Runtime providers for optimized performance
            providers = ['CPUExecutionProvider']
            
            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 2  # Optimize for Pi CPU cores
            sess_options.inter_op_num_threads = 2
            
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # Get model input/output info
            self.input_name = session.get_inputs()[0].name
            self.output_names = [output.name for output in session.get_outputs()]
            
            logger.info(f"ONNX model loaded successfully")
            logger.info(f"Input: {self.input_name}")
            logger.info(f"Outputs: {self.output_names}")
            
            # Warm up the model
            dummy_input = np.random.randn(1, 3, self.INPUT_SIZE[1], self.INPUT_SIZE[0]).astype(np.float32)
            _ = session.run(self.output_names, {self.input_name: dummy_input})
            logger.info("Model warmed up successfully")
            
            return session
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def _setup_camera(self, camera_index):
        """Setup camera with optimal settings for Raspberry Pi"""
        try:
            # Try multiple backends for best performance
            backends = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
            
            for backend in backends:
                cap = cv2.VideoCapture(camera_index, backend)
                if cap.isOpened():
                    logger.info(f"Camera opened with backend: {backend}")
                    break
            else:
                raise RuntimeError("Failed to open camera with any backend")
            
            # Optimize camera settings for performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.INPUT_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.INPUT_SIZE[1])
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
            
            # Try to set MJPEG codec for better performance
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # Disable auto-exposure and auto-white balance for consistent performance
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            
            logger.info(f"Camera configured: {self.INPUT_SIZE[0]}x{self.INPUT_SIZE[1]} @ 15 FPS")
            return cap
            
        except Exception as e:
            logger.error(f"Camera setup failed: {e}")
            raise
    
    def _setup_arduino(self):
        """Setup Arduino with simple serial communication"""
        try:
            ports = list(serial.tools.list_ports.comports())
            if not ports:
                logger.warning("No serial ports found - running without Arduino")
                return None
            
            # Try to connect to first available port (simpler approach)
            for port_info in ports:
                try:
                    arduino = serial.Serial(
                        port=port_info.device,
                        baudrate=9600,
                        timeout=0.1,  # Short timeout for non-blocking
                        write_timeout=0.1
                    )
                    time.sleep(2)  # Allow Arduino to reset
                    logger.info(f"Arduino connected on {port_info.device}")
                    return arduino
                except Exception as e:
                    logger.warning(f"Failed to connect to {port_info.device}: {e}")
                    continue
            
            logger.warning("No Arduino found - running without serial communication")
            return None
            
        except Exception as e:
            logger.error(f"Arduino setup failed: {e}")
            return None
    
    def _start_threads(self):
        """Start background threads for non-blocking operations"""
        # Serial communication thread
        if self.arduino:
            serial_thread = threading.Thread(target=self._serial_worker, daemon=True)
            serial_thread.start()
        
        # Camera capture thread
        capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        capture_thread.start()
    
    def _serial_worker(self):
        """Background thread for simple Arduino communication"""
        while self.running and self.arduino:
            try:
                # Send data to Arduino (simple and direct)
                if not self.serial_queue.empty():
                    try:
                        message = self.serial_queue.get_nowait()
                        self.arduino.write(f"{message}\n".encode('utf-8'))
                        self.arduino.flush()
                        logger.info(f"Sent to Arduino: {message}")
                    except Exception as e:
                        logger.warning(f"Serial write error: {e}")
                
                # Read any response from Arduino (optional)
                try:
                    if self.arduino.in_waiting > 0:
                        response = self.arduino.readline().decode('utf-8').strip()
                        if response:
                            logger.info(f"Arduino response: {response}")
                except Exception as e:
                    logger.warning(f"Serial read error: {e}")
                
                time.sleep(0.05)  # Small delay to prevent CPU overload
                
            except Exception as e:
                logger.error(f"Serial worker error: {e}")
                break
    
    def _capture_worker(self):
        """Background thread for camera capture"""
        while self.running:
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Keep only the latest frame to prevent lag
                    if not self.frame_queue.full():
                        try:
                            self.frame_queue.put_nowait(frame)
                        except:
                            pass  # Queue full, skip frame
                    else:
                        # Remove old frame and add new one
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame)
                        except:
                            pass
                else:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Capture worker error: {e}")
                break
    
    def preprocess_frame(self, frame):
        """Preprocess frame for ONNX inference"""
        # Resize to model input size
        resized = cv2.resize(frame, self.INPUT_SIZE)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        normalized = rgb.astype(np.float32) / 255.0
        
        # Convert to NCHW format (batch, channels, height, width)
        input_tensor = np.transpose(normalized, (2, 0, 1))  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
        
        return input_tensor
    
    def postprocess_outputs(self, outputs):
        """Process ONNX model outputs to get detections"""
        try:
            # YOLOv8 output format: [batch, 84, 8400] where 84 = 4 (bbox) + 80 (classes)
            predictions = outputs[0]  # Shape: [1, 84, 8400]
            predictions = predictions[0]  # Remove batch dimension: [84, 8400]
            predictions = predictions.T  # Transpose to [8400, 84]
            
            boxes = []
            confidences = []
            class_ids = []
            
            # Extract boxes and scores
            for detection in predictions:
                # First 4 values are box coordinates (center_x, center_y, width, height)
                center_x, center_y, width, height = detection[:4]
                
                # Remaining values are class probabilities
                class_scores = detection[4:]
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]
                
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    # Convert from center format to corner format
                    x1 = int(center_x - width / 2)
                    y1 = int(center_y - height / 2)
                    x2 = int(center_x + width / 2)
                    y2 = int(center_y + height / 2)
                    
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))
            
            # Apply Non-Maximum Suppression
            if len(boxes) > 0:
                boxes = np.array(boxes)
                confidences = np.array(confidences)
                
                # Convert to format expected by cv2.dnn.NMSBoxes
                nms_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    nms_boxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to x, y, w, h
                
                indices = cv2.dnn.NMSBoxes(nms_boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
                
                final_boxes = []
                final_confidences = []
                final_class_ids = []
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        final_boxes.append(boxes[i])
                        final_confidences.append(confidences[i])
                        final_class_ids.append(class_ids[i])
                
                return final_boxes, final_confidences, final_class_ids
            
            return [], [], []
            
        except Exception as e:
            logger.error(f"Postprocessing error: {e}")
            return [], [], []
    
    def detect_flame_position(self, boxes):
        """Detect flame position relative to center"""
        if not boxes:
            return "NONE"
        
        center_x = self.INPUT_SIZE[0] // 2
        positions = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            flame_center_x = (x1 + x2) / 2
            
            # Determine position with buffer zone
            if flame_center_x < center_x - 80:
                positions.append("LEFT")
            elif flame_center_x > center_x + 80:
                positions.append("RIGHT")
            else:
                positions.append("CENTER")
        
        # Return position of first detection
        return positions[0] if positions else "NONE"
    
    def draw_overlay(self, frame, boxes, confidences, position):
        """Draw detection overlay on frame"""
        # Draw center line
        height, width = frame.shape[:2]
        center_x = width // 2
        
        cv2.line(frame, (center_x, 0), (center_x, height), (255, 255, 255), 2)
        cv2.putText(frame, "CENTER", (center_x - 40, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw detections
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            confidence = confidences[i]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"Fire: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Show current position
        position_color = (0, 255, 0) if position != "NONE" else (0, 0, 255)
        cv2.putText(frame, f"FLAME: {position}", (10, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, position_color, 2)
        
        # Show FPS
        fps_text = f"FPS: {self.current_fps:.1f}" if self.current_fps > 0 else "FPS: Calculating..."
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    def update_fps(self):
        """Update FPS calculation - simple and accurate"""
        current_time = time.time()
        self.fps_frame_count += 1
        
        # Update FPS every second
        if current_time - self.last_fps_update >= 1.0:
            elapsed = current_time - self.last_fps_update
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.last_fps_update = current_time
    
    def run(self):
        """Main detection loop"""
        logger.info("Starting flame detection...")
        
        try:
            while self.running:
                # Get latest frame
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Skip frames for performance (process every nth frame)
                if self.frame_count % self.FRAME_SKIP != 0:
                    self.frame_count += 1
                    continue
                
                # Ensure frame is correct size
                if frame.shape[:2] != self.INPUT_SIZE[::-1]:
                    frame = cv2.resize(frame, self.INPUT_SIZE)
                
                # Preprocess frame for ONNX
                start_time = time.time()
                input_tensor = self.preprocess_frame(frame)
                
                # Run ONNX inference
                try:
                    outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
                    
                    # Process outputs
                    boxes, confidences, class_ids = self.postprocess_outputs(outputs)
                    
                    # Detect flame position
                    position = self.detect_flame_position(boxes)
                    
                    # Send position to Arduino (non-blocking)
                    if self.arduino:
                        try:
                            self.serial_queue.put_nowait(position)
                        except:
                            pass  # Queue full, skip this update
                    
                    # Print position for debugging
                    if position != "NONE":
                        logger.info(f"Flame detected: {position} (confidence: {max(confidences) if confidences else 0:.2f})")
                    
                    # Draw overlay and display
                    display_frame = self.draw_overlay(frame.copy(), boxes, confidences, position)
                    cv2.imshow("ONNX Flame Detection", display_frame)
                    
                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    cv2.imshow("ONNX Flame Detection", frame)
                
                inference_time = time.time() - start_time
                
                # Update performance metrics
                self.update_fps()
                
                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                
                # Target FPS control
                target_delay = 1.0 / self.TARGET_FPS - inference_time
                if target_delay > 0:
                    time.sleep(target_delay)
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        self.running = False
        
        if self.camera:
            self.camera.release()
        
        if self.arduino:
            self.arduino.close()
        
        cv2.destroyAllWindows()
        logger.info("Cleanup complete")

# Main execution
if __name__ == "__main__":
    try:
        detector = OptimizedFlameDetector(
            model_path="best_v2.onnx",
            camera_index=0
        )
        detector.run()
    except Exception as e:
        logger.error(f"Failed to start ONNX flame detector: {e}")
        print("\nMake sure you have:")
        print("1. best_v2.onnx model file in the current directory")
        print("2. Camera connected and working") 
        print("3. Required packages: pip install onnxruntime opencv-python pyserial numpy")
        print("4. Convert your .pt model to .onnx using: python -c \"from ultralytics import YOLO; YOLO('best_v2.pt').export(format='onnx')\")")
