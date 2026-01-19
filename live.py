"""
PROFESSIONAL FINGER COUNTING SYSTEM WITH MNIST RECOGNITION
===========================================================

Advanced hand tracking system that:
1. Detects hand using MediaPipe (robust, no skin color dependency)
2. Counts extended fingers using MediaPipe landmarks
3. Renders the count as an MNIST-style digit image
4. Passes rendered digit to trained MNIST CNN for verification
5. Displays results with professional UI

Requirements:
    pip install torch torchvision opencv-python numpy mediapipe

Author: Professional AI Assistant
Version: 2.1 (MediaPipe Edition)
Date: 2026
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import deque
from typing import Optional, Tuple, List
import math

# MediaPipe imports for new API (v0.10+)
# MediaPipe imports
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration constants"""
    
    # MNIST Parameters
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    MNIST_SIZE = 28
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Camera
    CAM_WIDTH = 1280
    CAM_HEIGHT = 720
    CAM_FPS = 30
    
    # Colors (BGR format)
    COLOR_PRIMARY = (76, 175, 80)      # Material Green
    COLOR_SECONDARY = (33, 150, 243)   # Material Blue
    COLOR_ACCENT = (255, 152, 0)       # Material Orange
    COLOR_ERROR = (244, 67, 54)        # Material Red
    COLOR_TEXT = (255, 255, 255)       # White
    COLOR_BACKGROUND = (33, 33, 33)    # Dark Gray
    COLOR_DEFECTS = (0, 0, 255)        # Red for defects
    
    # UI
    FPS_WINDOW_SIZE = 30
    SMOOTHING_WINDOW = 7
    
    # MediaPipe Hand Detection
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5


# ============================================================================
# MNIST CNN MODEL
# ============================================================================

class MNIST_CNN(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.
    
    Architecture:
        Conv2D(1->32) -> ReLU -> MaxPool2D(2x2)
        Conv2D(32->64) -> ReLU -> MaxPool2D(2x2)
        Flatten -> Dense(3136->128) -> ReLU -> Dropout(0.2)
        Dense(128->10) -> Softmax
    """
    
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================================
# HAND TRACKING AND FINGER COUNTING (MEDIAPIPE)
# ============================================================================

class HandTracker:
    """
    Advanced hand tracking with finger counting using MediaPipe.
    
    Uses MediaPipe Hands for robust hand detection without skin color dependency.
    Works with any background and lighting conditions.
    """
    
    def __init__(self):
        """Initialize MediaPipe hand tracker."""
        self.finger_count_history = deque(maxlen=Config.SMOOTHING_WINDOW)
        
        # Standard stable import for MediaPipe 0.10.x
        import mediapipe as mp
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
        )
        
        self.FINGER_TIPS = [4, 8, 12, 16, 20]
        self.FINGER_PIPS = [2, 6, 10, 14, 18]  # Finger base joints (for comparison)
        
    def detect_hand(self, frame: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """
        Detect hand(s) using MediaPipe.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            (hand_landmarks_list, visualization_frame): List of detected hands and visualization
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Create visualization frame
        vis_frame = frame.copy()
        
        hand_landmarks_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_landmarks_list.append(hand_landmarks)
                
                # Draw hand landmarks on visualization
                self.mp_drawing.draw_landmarks(
                    vis_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return hand_landmarks_list, vis_frame
    
    def count_fingers_from_landmarks(self, hand_landmarks, frame_shape: Tuple[int, int]) -> int:
        """
        Count extended fingers using MediaPipe landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_shape: (height, width) of frame
            
        Returns:
            finger_count: Number of extended fingers (0-5)
        """
        h, w = frame_shape
        finger_count = 0
        
        # Get all landmark coordinates
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append((int(lm.x * w), int(lm.y * h)))
        
        # Check thumb (different logic - compare x-coordinate)
        # Thumb is extended if tip is further from wrist than base
        thumb_tip = landmarks[self.FINGER_TIPS[0]]
        thumb_base = landmarks[self.FINGER_PIPS[0]]
        wrist = landmarks[0]
        
        # Calculate if thumb is extended (based on hand orientation)
        thumb_tip_dist = abs(thumb_tip[0] - wrist[0])
        thumb_base_dist = abs(thumb_base[0] - wrist[0])
        
        if thumb_tip_dist > thumb_base_dist:
            finger_count += 1
        
        # Check other four fingers (compare y-coordinate)
        # Finger is extended if tip is above the PIP joint
        for i in range(1, 5):
            tip_idx = self.FINGER_TIPS[i]
            pip_idx = self.FINGER_PIPS[i]
            
            tip_y = landmarks[tip_idx][1]
            pip_y = landmarks[pip_idx][1]
            
            # Finger is extended if tip is higher (lower y value) than base
            if tip_y < pip_y:
                finger_count += 1
        
        return finger_count
    
    def get_smoothed_count(self, current_count: int) -> int:
        """
        Apply temporal smoothing to finger count.
        
        Args:
            current_count: Current frame finger count
            
        Returns:
            Smoothed finger count using mode (most common value)
        """
        self.finger_count_history.append(current_count)
        
        if len(self.finger_count_history) == 0:
            return current_count
        
        # Use mode for stability
        counts = list(self.finger_count_history)
        return max(set(counts), key=counts.count)
    
    def draw_hand_visualization(self, frame: np.ndarray, hand_landmarks_list: List, finger_counts: List[int]):
        """
        Draw hand landmarks and finger count.
        
        Args:
            frame: Image to draw on
            hand_landmarks_list: List of hand landmarks
            finger_counts: List of finger counts for each hand
        """
        # MediaPipe already draws landmarks in detect_hand()
        # Here we can add additional visualizations if needed
        
        for idx, hand_landmarks in enumerate(hand_landmarks_list):
            if idx < len(finger_counts):
                # Get wrist position for text
                h, w = frame.shape[:2]
                wrist = hand_landmarks.landmark[0]
                wrist_x = int(wrist.x * w)
                wrist_y = int(wrist.y * h)
                
                # Draw finger count near wrist
                cv2.putText(frame, f"{finger_counts[idx]}", 
                           (wrist_x - 20, wrist_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, Config.COLOR_ACCENT, 3)
    
    def cleanup(self):
        """Cleanup MediaPipe resources."""
        self.hands.close()


# ============================================================================
# DIGIT RENDERER
# ============================================================================

class DigitRenderer:
    """
    Renders digits in MNIST style for model input.
    """
    
    @staticmethod
    def render_digit(digit: int, size: int = 28) -> np.ndarray:
        """
        Render a digit in MNIST style.
        
        Creates a clean, centered digit image similar to MNIST training data.
        
        Args:
            digit: Digit to render (0-9)
            size: Output image size (default 28x28)
            
        Returns:
            Binary image of rendered digit (numpy array)
        """
        # Create blank canvas (larger for better quality)
        canvas = np.zeros((size * 4, size * 4), dtype=np.uint8)
        
        # Render digit with OpenCV
        text = str(digit)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4.0
        thickness = 12
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Center text
        x = (canvas.shape[1] - text_width) // 2
        y = (canvas.shape[0] + text_height) // 2
        
        # Draw white text on black background
        cv2.putText(canvas, text, (x, y), font, font_scale, 255, thickness)
        
        # Resize to MNIST size
        resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
        
        return resized
    
    @staticmethod
    def preprocess_for_model(digit_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess rendered digit for MNIST model.
        
        Args:
            digit_image: 28x28 grayscale image
            
        Returns:
            Preprocessed tensor (1, 1, 28, 28)
        """
        # Normalize
        img = digit_image.astype(np.float32) / 255.0
        img = (img - Config.MNIST_MEAN) / Config.MNIST_STD
        
        # Convert to tensor
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(Config.DEVICE)
        
        return tensor


# ============================================================================
# RECOGNITION SYSTEM
# ============================================================================

class RecognitionSystem:
    """
    Complete recognition pipeline: finger counting -> digit rendering -> MNIST prediction.
    """
    
    def __init__(self, model_path: str = "mnist_cnn.pt"):
        """
        Initialize recognition system.
        
        Args:
            model_path: Path to trained MNIST model
        """
        self.model = self._load_model(model_path)
        self.digit_renderer = DigitRenderer()
        self.last_prediction = None
        self.last_confidence = 0.0
        self.last_rendered_digit = None
    
    def _load_model(self, model_path: str) -> MNIST_CNN:
        """Load trained MNIST model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Please ensure mnist_cnn.pt is in the same directory."
            )
        
        model = MNIST_CNN().to(Config.DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        model.eval()
        
        return model
    
    def process_finger_count(self, finger_count: int) -> Tuple[int, float, np.ndarray]:
        """
        Process finger count through complete pipeline.
        
        Pipeline:
        1. Render finger count as MNIST-style digit
        2. Preprocess for model
        3. Run through MNIST CNN
        4. Return prediction and confidence
        
        Args:
            finger_count: Number of fingers detected (0-10)
            
        Returns:
            (prediction, confidence, rendered_digit_image)
        """
        # Handle counts > 9
        display_digit = finger_count % 10
        
        # Render digit
        rendered = self.digit_renderer.render_digit(display_digit)
        self.last_rendered_digit = rendered.copy()
        
        # Preprocess
        tensor = self.digit_renderer.preprocess_for_model(rendered)
        
        # Predict
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            prediction = int(probs.argmax(1).item())
            confidence = float(probs.max().item())
        
        self.last_prediction = prediction
        self.last_confidence = confidence
        
        return prediction, confidence, rendered


# ============================================================================
# UI AND VISUALIZATION
# ============================================================================

class UIManager:
    """
    Professional UI rendering and visualization.
    """
    
    @staticmethod
    def draw_info_panel(frame: np.ndarray, fps: float, finger_count: int, 
                       prediction: Optional[int], confidence: float) -> np.ndarray:
        """
        Draw information panel overlay.
        
        Args:
            frame: Base frame
            fps: Current FPS
            finger_count: Detected finger count
            prediction: Model prediction
            confidence: Prediction confidence
            
        Returns:
            Frame with info panel
        """
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        
        # Top panel
        cv2.rectangle(overlay, (0, 0), (w, 100), Config.COLOR_BACKGROUND, -1)
        
        # Bottom panel
        cv2.rectangle(overlay, (0, h - 80), (w, h), Config.COLOR_BACKGROUND, -1)
        
        # Blend
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Title
        cv2.putText(frame, " Gideon FINGER RECOGNITION SYSTEM", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, Config.COLOR_PRIMARY, 3)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_TEXT, 2)
        
        # Finger count
        cv2.putText(frame, f"Fingers: {finger_count}", (w - 200, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.COLOR_ACCENT, 2)
        
        # Prediction
        if prediction is not None:
            pred_text = f"Prediction: {prediction} ({confidence*100:.1f}%)"
            color = Config.COLOR_PRIMARY if confidence > 0.8 else Config.COLOR_ACCENT
            cv2.putText(frame, pred_text, (w - 350, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Bottom instructions
        instructions = "Controls: Q=Quit | H=Toggle Hand | D=Toggle Detection"
        cv2.putText(frame, instructions, (20, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.COLOR_TEXT, 1)
        
        return frame
    
    @staticmethod
    def draw_digit_preview(frame: np.ndarray, digit_image: Optional[np.ndarray], 
                          prediction: int, confidence: float, finger_count: int):
        """
        Draw MNIST digit preview and prediction box.
        
        Args:
            frame: Frame to draw on
            digit_image: Rendered 28x28 digit
            prediction: Model prediction
            confidence: Confidence score
            finger_count: Actual finger count
        """
        if digit_image is None:
            return
        
        h, w = frame.shape[:2]
        
        # Position for preview (top right)
        preview_size = 200
        x_start = w - preview_size - 20
        y_start = 120
        
        # Resize digit for display
        digit_display = cv2.resize(digit_image, (preview_size, preview_size),
                                  interpolation=cv2.INTER_NEAREST)
        digit_display = cv2.cvtColor(digit_display, cv2.COLOR_GRAY2BGR)
        
        # Add border
        border_color = Config.COLOR_PRIMARY if prediction == finger_count % 10 else Config.COLOR_ERROR
        cv2.rectangle(digit_display, (0, 0), (preview_size-1, preview_size-1),
                     border_color, 3)
        
        # Overlay on frame
        frame[y_start:y_start+preview_size, x_start:x_start+preview_size] = digit_display
        
        # Label
        cv2.putText(frame, "MNIST Input:", (x_start, y_start - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.COLOR_TEXT, 2)
        
        # Prediction box
        pred_y = y_start + preview_size + 20
        pred_text = f"Model Output: {prediction}"
        conf_text = f"Confidence: {confidence*100:.1f}%"
        match_text = f"Match: {'YES' if prediction == finger_count % 10 else 'NO'}"
        
        cv2.putText(frame, pred_text, (x_start, pred_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.COLOR_PRIMARY, 2)
        cv2.putText(frame, conf_text, (x_start, pred_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_TEXT, 2)
        
        match_color = Config.COLOR_PRIMARY if prediction == finger_count % 10 else Config.COLOR_ERROR
        cv2.putText(frame, match_text, (x_start, pred_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)


# ============================================================================
# FPS COUNTER
# ============================================================================

class FPSCounter:
    """Accurate FPS measurement with moving average."""
    
    def __init__(self, window_size: int = Config.FPS_WINDOW_SIZE):
        self.frame_times = deque(maxlen=window_size)
    
    def tick(self):
        """Record current frame time."""
        self.frame_times.append(time.time())
    
    def get_fps(self) -> float:
        """Calculate current FPS."""
        if len(self.frame_times) < 2:
            return 0.0
        
        elapsed = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / elapsed if elapsed > 0 else 0.0


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class FingerRecognitionApp:
    """
    Main application class coordinating all components.
    """
    
    def __init__(self, model_path: str = "mnist_cnn.pt", camera_index: int = 0):
        """
        Initialize application.
        
        Args:
            model_path: Path to MNIST model weights
            camera_index: Camera device index
        """
        print("\n" + "="*70)
        print("PROFESSIONAL FINGER RECOGNITION SYSTEM")
        print("="*70)
        
        # Initialize components
        self.hand_tracker = HandTracker()
        self.recognition_system = RecognitionSystem(model_path)
        self.ui_manager = UIManager()
        self.fps_counter = FPSCounter()
        
        # Camera setup
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.CAM_FPS)
        
        # Application state
        self.show_hand = True
        self.show_detection = True
        self.running = True
        
        print(f"✓ Camera initialized: {Config.CAM_WIDTH}x{Config.CAM_HEIGHT}@{Config.CAM_FPS}fps")
        print(f"✓ Device: {Config.DEVICE}")
        print(f"✓ MediaPipe Hands initialized")
        print(f"✓ All systems ready\n")
    
    def run(self):
        """Main application loop."""
        print("Starting recognition system...")
        print("Show your hand to the camera!")
        print("NOTE: Works with any background - no skin color dependency!\n")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            self.fps_counter.tick()
            
            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Detect hand(s) using MediaPipe
            hand_landmarks_list, vis_frame = self.hand_tracker.detect_hand(frame)
            
            total_finger_count = 0
            prediction = None
            confidence = 0.0
            rendered_digit = None
            finger_counts = []
            
            if hand_landmarks_list:
                # Count fingers for each hand
                h, w = frame.shape[:2]
                for hand_landmarks in hand_landmarks_list:
                    finger_count = self.hand_tracker.count_fingers_from_landmarks(
                        hand_landmarks, (h, w)
                    )
                    finger_counts.append(finger_count)
                    total_finger_count += finger_count
                
                # Smooth the total count
                total_finger_count = self.hand_tracker.get_smoothed_count(total_finger_count)
                
                # Draw hand visualization
                if self.show_hand:
                    self.hand_tracker.draw_hand_visualization(vis_frame, hand_landmarks_list, finger_counts)
                    frame = vis_frame
                
                # Process through recognition system
                if total_finger_count > 0:
                    prediction, confidence, rendered_digit = \
                        self.recognition_system.process_finger_count(total_finger_count)
            else:
                # No hands detected
                if self.show_hand:
                    frame = vis_frame
            
            # Draw UI
            frame = self.ui_manager.draw_info_panel(
                frame, self.fps_counter.get_fps(), total_finger_count, 
                prediction, confidence
            )
            
            if self.show_detection and rendered_digit is not None:
                self.ui_manager.draw_digit_preview(
                    frame, rendered_digit, prediction, confidence, total_finger_count
                )
            
            # Display window
            cv2.imshow("Gideon Finger Recognition System", frame)
            
            # Handle input
            self.handle_keyboard()
        
        # Cleanup
        self.cleanup()
    
    def handle_keyboard(self):
        """Handle keyboard input."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # Q or ESC
            self.running = False
        elif key == ord('h'):
            self.show_hand = not self.show_hand
            print(f"Hand visualization: {'ON' if self.show_hand else 'OFF'}")
        elif key == ord('d'):
            self.show_detection = not self.show_detection
            print(f"Detection box: {'ON' if self.show_detection else 'OFF'}")
    
    def cleanup(self):
        """Cleanup resources."""
        self.hand_tracker.cleanup()
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Application terminated successfully")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Application entry point."""
    try:
        app = FingerRecognitionApp(model_path="mnist_cnn.pt", camera_index=0)
        app.run()
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Please ensure mnist_cnn.pt is in the same directory.")
        
    except RuntimeError as e:
        print(f"\n✗ Error: {e}")
        
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user")
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()