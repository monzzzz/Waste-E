import sys
import os
import cv2
import time
import numpy as np
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# COCO Classes (YOLOv8 default)

SPEED_FORWARD = 40
SPEED_TURN = 35
SPEED_PIVOT = 40
CENTER_TOLERANCE_PERCENT = 0.15  # Keep object within center 15% of screen

class GarbageDetection:
    def __init__(self):
        print("Initializing GarbageDetection (Server/Cloud Mode)...")
        
        # 1. Setup AI Model
        print("Loading YOLO model...")
        try:
            self.model = YOLO('yolov8n.pt')
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
            
    def detect_garbage(self, frame, save_name=None):
        """
        Run YOLO detection on frame.
        Returns closest detection dict or None.
        Prioritizes the largest bounding box area (closest object).
        """
        if frame is None: return None
        
        results = self.model(frame, verbose=False)
        
        # Save visualization if requested
        if save_name:
            if not os.path.exists('result'):
                os.makedirs('result')
            annotated_frame = results[0].plot()
            cv2.imwrite(os.path.join('result', save_name), annotated_frame)
            
        frame_width = frame.shape[1]
        
        valid_detections = []
        
        for r in results:
            for box in r.boxes:
                # coordinate of boundsing box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                print("results:", x1, y1, x2, y2, conf)
                if (conf < 0.6):
                    continue
                area = (x2 - x1) * (y2 - y1)
                center_x = (x1 + x2) / 2
                    
                valid_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'center_x': center_x,
                        'area': area,
                        'frame_width': frame_width,
                        'relative_x': (center_x / frame_width) - 0.5, 
                        'conf': conf
                })
                    
        if not valid_detections:
            return None
            
        # Sort by area (descending) -> Biggest area (closest object) first
        valid_detections.sort(key=lambda x: x['area'], reverse=True)
        
        return valid_detections[0]

    def get_motor_speeds(self, front_img, left_img, right_img):
        """
        Captures frames, runs detection, and calculates motor speeds.
        Args:
            front_img: numpy array (image) from front camera
            left_img: numpy array (image) from left camera
            right_img: numpy array (image) from right camera
        Returns:
            dict: {'left': int, 'right': int, 'status': str}
        """
        frames = {
            'front': front_img,
            'left': left_img,
            'right': right_img
        }
        detections = {}
        
        # Run detection on all frames
        for name, frame in frames.items():
            if frame is not None:
                detections[name] = self.detect_garbage(frame, save_name=f"{name}_result.jpg")
            else:
                detections[name] = None
                
        # Decision Logic
        motor_left = 0
        motor_right = 0
        status_msg = "IDLE"

        # Priority 1: Tracking target in FRONT camera -> Approach
        if detections.get('front'):
            target = detections['front']
            rel_x = target['relative_x'] # -0.5 to 0.5
            
            if abs(rel_x) < (CENTER_TOLERANCE_PERCENT / 2):
                # Centered -> Move Forward
                status_msg = f"FRONT: Target Centered (x={rel_x:.2f}) -> FORWARD"
                motor_left = SPEED_FORWARD
                motor_right = SPEED_FORWARD
                
            elif rel_x < 0:
                # Target is left of center -> Turn Left slightly (Arcade drive style or differential)
                # Left motor slower, Right motor faster
                status_msg = f"FRONT: Target Left (x={rel_x:.2f}) -> ADJUST LEFT"
                motor_left = SPEED_FORWARD - SPEED_TURN
                motor_right = SPEED_FORWARD + SPEED_TURN  
                
            else:
                # Target is right of center -> Turn Right slightly
                status_msg = f"FRONT: Target Right (x={rel_x:.2f}) -> ADJUST RIGHT"
                motor_left = SPEED_FORWARD + SPEED_TURN
                motor_right = SPEED_FORWARD - SPEED_TURN

        # Priority 2: Detected in LEFT camera -> Pivot Left to bring into front view
        elif detections.get('left'):
            status_msg = "LEFT: Garbage detected on side -> PIVOT LEFT"
            motor_left = -SPEED_PIVOT
            motor_right = SPEED_PIVOT
        
        # Priority 3: Detected in RIGHT camera -> Pivot Right to bring into front view
        elif detections.get('right'):
            status_msg = "RIGHT: Garbage detected on side -> PIVOT RIGHT"
            motor_left = SPEED_PIVOT
            motor_right = -SPEED_PIVOT
            
        # Priority 4: Search / Idle -> Stop
        else:
            status_msg = "No garbage detected -> STOP"
            motor_left = 0
            motor_right = 0
            
        # Clamp speeds to -100 to 100
        motor_left = max(-100, min(100, int(motor_left)))
        motor_right = max(-100, min(100, int(motor_right)))
        
        return {
            'left': motor_left,
            'right': motor_right,
            'status': status_msg
        }
