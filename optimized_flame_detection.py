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
from ultralytics import YOLO
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
    def __init__(self, model_path="best_v2.pt", camera_index=0):
        # Performance settings
        self.TARGET_FPS = 10
        self.FRAME_SKIP = 2  # Process every 2nd frame for speed
        self.INPUT_SIZE = (640, 480)
        self.CONFIDENCE_THRESHOLD = 0.4
        
        # Initialize components
        self.model = self._load_model(model_path)
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
        
        # Start background threads
        self._start_threads()
    
    def _load_model(self, model_path):
        """Load YOLO model with optimizations"""
        try:
            logger.info(f"Loading YOLO model: {model_path}")
            model = YOLO(model_path)
            
            # Optimize model for inference
            model.fuse()  # Fuse conv and bn layers for speed
            
            # Warm up the model with dummy input
            dummy_input = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = model.predict(dummy_input, conf=self.CONFIDENCE_THRESHOLD, verbose=False)
            
            logger.info("Model loaded and warmed up successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
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
        """Setup Arduino with non-blocking serial communication"""
        try:
            ports = list(serial.tools.list_ports.comports())
            if not ports:
                logger.warning("No serial ports found - running without Arduino")
                return None
            
            # Try to connect to Arduino (usually on USB ports)
            for port_info in ports:
                try:
                    if 'USB' in port_info.description or 'Arduino' in port_info.description:
                        arduino = serial.Serial(
                            port=port_info.device,
                            baudrate=9600,
                            timeout=0.01,  # Very short timeout for non-blocking
                            write_timeout=0.01
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
        """Background thread for Arduino communication"""
        while self.running and self.arduino:
            try:
                # Send data to Arduino (non-blocking)
                if not self.serial_queue.empty():
                    try:
                        message = self.serial_queue.get_nowait()
                        self.arduino.write(f"{message}\n".encode('utf-8'))
                        self.arduino.flush()
                    except Exception as e:
                        logger.warning(f"Serial write error: {e}")
                
                # Read response from Arduino (non-blocking)
                if self.arduino.in_waiting:
                    try:
                        response = self.arduino.readline().decode('utf-8').strip()
                        if response:
                            logger.info(f"Arduino: {response}")
                    except Exception as e:
                        logger.warning(f"Serial read error: {e}")
                
                time.sleep(0.01)  # Small delay to prevent CPU overload
                
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
    
    def detect_flame_position(self, results, frame_width):
        """Detect flame position relative to center"""
        if not results or not results[0].boxes:
            return "NONE"
        
        boxes = results[0].boxes
        center_x = frame_width // 2
        positions = []
        
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            flame_center_x = (x1 + x2) / 2
            
            # Determine position with buffer zone
            if flame_center_x < center_x - 80:  # Increased buffer for stability
                positions.append("LEFT")
            elif flame_center_x > center_x + 80:
                positions.append("RIGHT")
            else:
                positions.append("CENTER")
        
        # Return the position of the largest detection
        if positions:
            return positions[0]  # Return first detection
        return "NONE"
    
    def draw_overlay(self, frame, results, position):
        """Draw detection overlay on frame"""
        # Draw center line
        height, width = frame.shape[:2]
        center_x = width // 2
        
        cv2.line(frame, (center_x, 0), (center_x, height), (255, 255, 255), 2)
        cv2.putText(frame, "CENTER", (center_x - 40, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw detections if any
        if results and results[0].boxes is not None:
            annotated_frame = results[0].plot()
            frame[:] = annotated_frame[:]
        
        # Show current position
        position_color = (0, 255, 0) if position != "NONE" else (0, 0, 255)
        cv2.putText(frame, f"FLAME: {position}", (10, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, position_color, 2)
        
        # Show FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:  # Update every 30 frames
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            if elapsed > 0:
                self.current_fps = 30 / elapsed
            self.fps_start_time = current_time
    
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
                
                # Run YOLO inference
                start_time = time.time()
                results = self.model.predict(
                    frame, 
                    conf=self.CONFIDENCE_THRESHOLD, 
                    verbose=False,
                    device='cpu'  # Force CPU for Raspberry Pi
                )
                inference_time = time.time() - start_time
                
                # Detect flame position
                position = self.detect_flame_position(results, self.INPUT_SIZE[0])
                
                # Send position to Arduino (non-blocking)
                if self.arduino:
                    try:
                        self.serial_queue.put_nowait(position)
                    except:
                        pass  # Queue full, skip this update
                
                # Print position for debugging
                if position != "NONE":
                    logger.info(f"Flame detected: {position}")
                
                # Draw overlay and display
                display_frame = self.draw_overlay(frame.copy(), results, position)
                cv2.imshow("Optimized Flame Detection", display_frame)
                
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
            model_path="best_v2.pt",
            camera_index=0
        )
        detector.run()
    except Exception as e:
        logger.error(f"Failed to start flame detector: {e}")
        print("Make sure your camera and model file are properly connected/available.")
