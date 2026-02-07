#!/usr/bin/env python3
"""
Hand Tracking Backend for GideonMac
Uses MediaPipe for reliable hand detection and finger counting.
Communication via stdin/stdout JSON messages.
"""

import sys
import json
import cv2
import numpy as np
import mediapipe as mp
import base64


class HandTracker:
    """MediaPipe-based hand tracker matching live.py logic exactly."""
    
    FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Little
    FINGER_PIPS = [3, 6, 10, 14, 18]  # Thumb IP, then PIP joints
    
    def __init__(self):
        # Use legacy MediaPipe Hands (no external model needed)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("MediaPipe Hands initialized", file=sys.stderr)
    
    def process_frame(self, frame_data: str) -> dict:
        """Process base64-encoded frame and return finger count."""
        try:
            # Decode base64 image
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {"error": "Failed to decode frame", "count": 0}
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            total_fingers = 0
            
            # Use legacy MediaPipe Hands
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                    total_fingers += self.count_fingers(landmarks, frame.shape[:2])
            
            return {"count": total_fingers, "error": None}
            
        except Exception as e:
            return {"error": str(e), "count": 0}
    
    def count_fingers(self, landmarks, frame_shape):
        """Count extended fingers using exact logic from live.py"""
        if len(landmarks) < 21:
            return 0
        
        height, width = frame_shape
        finger_count = 0
        
        # Convert normalized coords to pixels
        pixels = [(int(x * width), int(y * height)) for x, y in landmarks]
        
        # Check thumb (x-axis distance from wrist)
        thumb_tip = pixels[self.FINGER_TIPS[0]]
        thumb_base = pixels[self.FINGER_PIPS[0]]  # IP joint for thumb
        wrist = pixels[0]
        
        # Thumb is extended if tip is further from wrist than base (x-axis)
        thumb_tip_dist = abs(thumb_tip[0] - wrist[0])
        thumb_base_dist = abs(thumb_base[0] - wrist[0])
        
        if thumb_tip_dist > thumb_base_dist:
            finger_count += 1
        
        # Check other four fingers (y-axis: tip above PIP)
        for i in range(1, 5):
            tip_idx = self.FINGER_TIPS[i]
            pip_idx = self.FINGER_PIPS[i]
            
            tip_y = pixels[tip_idx][1]
            pip_y = pixels[pip_idx][1]
            
            # In image coordinates, smaller y = higher position
            if tip_y < pip_y:
                finger_count += 1
        
        return finger_count


def main():
    """Main loop: read JSON commands from stdin, write results to stdout."""
    tracker = HandTracker()
    
    print(json.dumps({"status": "ready"}), flush=True)
    
    for line in sys.stdin:
        try:
            msg = json.loads(line.strip())
            
            if msg.get("command") == "process":
                result = tracker.process_frame(msg.get("frame", ""))
                print(json.dumps(result), flush=True)
            elif msg.get("command") == "quit":
                break
            else:
                print(json.dumps({"error": "Unknown command"}), flush=True)
                
        except json.JSONDecodeError:
            print(json.dumps({"error": "Invalid JSON"}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
