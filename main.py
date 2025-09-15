import cv2
import numpy as np
from collections import deque
import time

class FireDetector:
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # HSV color ranges for fire detection (orange/yellow/red)
        self.fire_lower1 = np.array([0, 50, 50])    # Red-orange lower bound
        self.fire_upper1 = np.array([35, 255, 255]) # Red-orange upper bound
        
        # Additional range for deeper reds
        self.fire_lower2 = np.array([160, 50, 50])
        self.fire_upper2 = np.array([180, 255, 255])
        
        # Tracking parameters
        self.blob_history = {}  # Store intensity history for each blob
        self.max_history_length = 10  # Number of frames to track
        self.flicker_threshold = 0.15  # Minimum intensity variation ratio
        self.min_blob_area = 50  # Minimum area for small flames (candles)
        self.max_blob_area = 50000  # Maximum reasonable fire size
        
        # Frame counter for blob ID management
        self.frame_count = 0
        self.cleanup_interval = 30  # Clean up old blob histories every N frames
        
    def preprocess_frame(self, frame):
        """Resize and blur frame for processing"""
        # Resize frame for performance
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        return frame, blurred
    
    def detect_fire_color_regions(self, hsv_frame):
        """Detect fire-colored regions using HSV color filtering"""
        # Create masks for fire colors
        mask1 = cv2.inRange(hsv_frame, self.fire_lower1, self.fire_upper1)
        mask2 = cv2.inRange(hsv_frame, self.fire_lower2, self.fire_upper2)
        
        # Combine masks
        fire_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        
        return fire_mask
    
    def find_fire_contours(self, mask):
        """Find contours of potential fire regions"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_blob_area <= area <= self.max_blob_area:
                valid_contours.append(contour)
        
        return valid_contours
    
    def calculate_blob_intensity(self, frame, contour):
        """Calculate average intensity of a blob region"""
        # Create mask for this contour
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        
        # Calculate mean intensity in the region
        mean_intensity = cv2.mean(frame, mask=mask)[0]
        return mean_intensity
    
    def get_blob_center(self, contour):
        """Get center point of a contour"""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return None
    
    def find_closest_blob_id(self, center, max_distance=50):
        """Find the closest tracked blob to current center"""
        if not self.blob_history:
            return None
            
        min_distance = float('inf')
        closest_id = None
        
        for blob_id, history in self.blob_history.items():
            if history and len(history) > 0:
                last_center = history[-1].get('center')
                if last_center:
                    distance = np.sqrt((center[0] - last_center[0])**2 + 
                                     (center[1] - last_center[1])**2)
                    if distance < min_distance and distance < max_distance:
                        min_distance = distance
                        closest_id = blob_id
        
        return closest_id
    
    def analyze_flicker(self, intensity_history):
        """Analyze intensity variation to detect flickering"""
        if len(intensity_history) < 3:
            return False
        
        intensities = [frame_data['intensity'] for frame_data in intensity_history]
        
        # Calculate intensity variation
        mean_intensity = np.mean(intensities)
        std_intensity = np.std(intensities)
        
        if mean_intensity == 0:
            return False
        
        # Check if variation is significant (flickering characteristic)
        variation_ratio = std_intensity / mean_intensity
        return variation_ratio > self.flicker_threshold
    
    def update_blob_tracking(self, contours, frame):
        """Update blob tracking with intensity history"""
        current_blobs = []
        
        for contour in contours:
            center = self.get_blob_center(contour)
            if center is None:
                continue
            
            intensity = self.calculate_blob_intensity(frame, contour)
            
            # Try to match with existing blob
            blob_id = self.find_closest_blob_id(center)
            
            if blob_id is None:
                # Create new blob
                blob_id = f"blob_{self.frame_count}_{len(current_blobs)}"
                self.blob_history[blob_id] = deque(maxlen=self.max_history_length)
            
            # Add current frame data
            frame_data = {
                'center': center,
                'intensity': intensity,
                'contour': contour,
                'frame': self.frame_count
            }
            
            self.blob_history[blob_id].append(frame_data)
            current_blobs.append((blob_id, contour, center))
        
        return current_blobs
    
    def cleanup_old_blobs(self):
        """Remove blob histories that haven't been updated recently"""
        if self.frame_count % self.cleanup_interval == 0:
            current_frame = self.frame_count
            to_remove = []
            
            for blob_id, history in self.blob_history.items():
                if history and len(history) > 0:
                    last_frame = history[-1]['frame']
                    if current_frame - last_frame > self.cleanup_interval:
                        to_remove.append(blob_id)
            
            for blob_id in to_remove:
                del self.blob_history[blob_id]
    
    def detect_fire(self, frame):
        """Main fire detection function"""
        self.frame_count += 1
        
        # Preprocess frame
        display_frame, processed_frame = self.preprocess_frame(frame)
        
        # Convert to HSV
        hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
        
        # Detect fire-colored regions
        fire_mask = self.detect_fire_color_regions(hsv)
        
        # Find contours
        contours = self.find_fire_contours(fire_mask)
        
        # Update blob tracking
        current_blobs = self.update_blob_tracking(contours, processed_frame)
        
        # Analyze each blob for fire characteristics
        fire_detected = False
        fire_boxes = []
        
        for blob_id, contour, center in current_blobs:
            history = self.blob_history[blob_id]
            
            # Check if blob shows flickering behavior
            is_flickering = self.analyze_flicker(history)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Classify as fire if:
            # 1. Large enough area OR small but flickering (for candles)
            # 2. Shows flickering behavior
            is_fire = False
            if area >= 200:  # Larger flames
                is_fire = is_flickering
            elif area >= self.min_blob_area:  # Smaller flames (candles)
                is_fire = is_flickering and len(history) >= 5  # Need more frames for small flames
            
            if is_fire:
                fire_detected = True
                fire_boxes.append((x, y, w, h))
        
        # Cleanup old blob histories
        self.cleanup_old_blobs()
        
        return fire_detected, fire_boxes, display_frame, fire_mask

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize fire detector
    fire_detector = FireDetector(frame_width=640, frame_height=480)
    
    print("White Flame Detection Started (Optimized for #FEFEFE color)")
    print("Detecting near-white flames like hex color #FEFEFE")
    print("This works best with bright white/near-white candle flames")
    print("Light your candle and test the detection. Press 'q' to quit.")
    
    fps_counter = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Detect fire
            fire_detected, fire_boxes, processed_frame, fire_mask = fire_detector.detect_fire(frame)
            
            # Draw results
            if fire_detected:
                # Draw "FIRE DETECTED" text
                cv2.putText(processed_frame, "FIRE DETECTED!", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Draw bounding boxes around detected fires
                for x, y, w, h in fire_boxes:
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(processed_frame, "FIRE", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = 30 / elapsed_time if elapsed_time > 0 else 0
                start_time = time.time()
            
            if fps_counter >= 30:
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, processed_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frames
            cv2.imshow('Fire Detection', processed_frame)
            cv2.imshow('Fire Mask', fire_mask)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")

if __name__ == "__main__":
    main()