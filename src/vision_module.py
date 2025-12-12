#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
from collections import deque, Counter
import time
import math


class VisionModule:
    # Board dimensions 
    MAP_WIDTH_CM = 28.7
    MAP_HEIGHT_CM = 30.5

    # Camera ROI (pixels)
    BOARD_X0_PX = 0
    BOARD_Y0_PX = 0
    BOARD_X1_PX = 640
    BOARD_Y1_PX = 480

    def __init__(self, model_path: str = 'models/best.pt',
                 performance_mode: str = 'balanced',
                 enable_white_balance: bool = True):
       
        print(f"🔧 Loading YOLOv8 model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Fuse model for faster inference
        try:
            self.model.fuse()
            print("   Model layers fused for optimization")
        except Exception as e:
            print(f"   Could not fuse model: {e}")

        # Class names (your 8 objects)
        self.class_names = [
            'aa_battery', 'charger_adapter', 'eraser', 'glue_stick',
            'highlighter', 'pen', 'sharpener', 'stapler'
        ]

        self.class_conf_boost = {
            'charger_adapter': 0.15,
        }

        # BGR colors for visualization
        self.color_map = {
            'red':     (0,   0, 255),
            'orange':  (0, 165, 255),
            'yellow':  (0, 255, 255),
            'green':   (0, 255,   0),
            'blue':    (255, 0,   0),
            'indigo':  (130, 0,  75),
            'violet':  (255, 0, 255),
            'neutral': (180, 180, 180),
            'white':   (255, 255, 255),
            'black':   (30, 30, 30),
        }

        # Performance settings
        self.performance_mode = performance_mode
        self.enable_white_balance = enable_white_balance
        self._configure_performance(performance_mode)

        # FPS tracking
        self.prev_frame_time = 0.0
        self.current_fps = 0.0

        self.detection_history: Dict[int, deque] = {}
        self.color_history: Dict[str, deque] = {}
        self.history_length = 3

        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        # COLOR DETECTION PARAMETERS
       
        self.COLOR_RANGES = {
            'red':     [(0, 10, 70, 50), (165, 180, 70, 50)],
            'orange':  [(10, 25, 80, 70)],
            'yellow':  [(25, 40, 70, 90)],
            'green':   [(40, 85, 50, 40)],
            'blue':    [(85, 115, 60, 40)],
            'indigo':  [(115, 135, 40, 30)],
            'violet':  [(135, 165, 35, 35)],
        }
        
        # Thresholds for special colors
        self.BLACK_V_THRESHOLD = 50      
        self.WHITE_V_THRESHOLD = 200     
        self.WHITE_S_THRESHOLD = 40      
        self.NEUTRAL_S_THRESHOLD = 35    
        self.MIN_COLORFUL_RATIO = 0.08   

        print(f" Vision module initialized ({performance_mode} mode)")

    def _configure_performance(self, mode: str):
        """Configure settings based on performance mode."""
        configs = {
            'fast': {
                'detection_size': (256, 192),
                'skip_enhancement': True,
                'smoothing_factor': 0.75,
                'color_samples': 3,
                'nms_conf': 0.4,
                'nms_iou': 0.5,
            },
            'balanced': {
                'detection_size': (320, 256),
                'skip_enhancement': False,
                'smoothing_factor': 0.65,
                'color_samples': 5,
                'nms_conf': 0.35,
                'nms_iou': 0.45,
            },
            'accurate': {
                'detection_size': (416, 320),
                'skip_enhancement': False,
                'smoothing_factor': 0.5,
                'color_samples': 7,
                'nms_conf': 0.3,
                'nms_iou': 0.4,
            }
        }
        
        config = configs.get(mode, configs['balanced'])
        self.detection_size = config['detection_size']
        self.skip_enhancement = config['skip_enhancement']
        self.smoothing_factor = config['smoothing_factor']
        self.color_samples = config['color_samples']
        self.nms_conf = config['nms_conf']
        self.nms_iou = config['nms_iou']

    # IMAGE PREPROCESSING
    def white_balance(self, img: np.ndarray) -> np.ndarray:
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        if self.skip_enhancement:
            return image

        if self.enable_white_balance:
            image = self.white_balance(image)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    # COORDINATE TRANSFORMATION
    def pixel_to_board_coords(self, u_px: int, v_px: int, 
                              img_width: int, img_height: int) -> Tuple[float, float]:
        board_w_px = self.BOARD_X1_PX - self.BOARD_X0_PX
        board_h_px = self.BOARD_Y1_PX - self.BOARD_Y0_PX

        if board_w_px <= 0 or board_h_px <= 0:
            board_w_px = img_width
            board_h_px = img_height
            x0, y0 = 0, 0
        else:
            x0 = self.BOARD_X0_PX
            y0 = self.BOARD_Y0_PX

        # 90° rotation mapping
        u_norm = (v_px - y0) / float(board_h_px)
        v_norm = (u_px - x0) / float(board_w_px)

        u_norm = max(0.0, min(1.0, u_norm))
        v_norm = max(0.0, min(1.0, v_norm))

        u_map = u_norm * self.MAP_WIDTH_CM
        v_map = v_norm * self.MAP_HEIGHT_CM

        return u_map, v_map
    
    # COLOR DETECTION
    def _extract_color_samples(self, frame: np.ndarray, 
                                detection: Dict) -> List[np.ndarray]:
        x1, y1, x2, y2 = detection["bbox"]
        w = x2 - x1
        h = y2 - y1
        
        if w <= 10 or h <= 10:
            return []

        cx, cy = detection["center"]
        samples = []

        # Sample regions
        regions = [
            (0.5, 0.5),   # Center
            (0.5, 0.3),   # Top
            (0.5, 0.7),   # Bottom
            (0.3, 0.5),   # Left
            (0.7, 0.5),   # Right
            (0.35, 0.35), # Top-left
            (0.65, 0.65), # Bottom-right
        ]

        sample_size = max(10, min(w, h) // 6)

        for rx, ry in regions[:self.color_samples]:
            # Calculate sample position
            sx = int(x1 + w * rx)
            sy = int(y1 + h * ry)

            # Extract small region
            sx1 = max(0, sx - sample_size // 2)
            sx2 = min(frame.shape[1], sx + sample_size // 2)
            sy1 = max(0, sy - sample_size // 2)
            sy2 = min(frame.shape[0], sy + sample_size // 2)

            roi = frame[sy1:sy2, sx1:sx2]
            if roi.size > 0:
                samples.append(roi)

        return samples

    def _analyze_hsv_histogram(self, hsv_roi: np.ndarray) -> Tuple[int, int, int]:
        h = hsv_roi[:, :, 0].flatten()
        s = hsv_roi[:, :, 1].flatten()
        v = hsv_roi[:, :, 2].flatten()

        # Use histogram peak for hue
        h_hist, h_bins = np.histogram(h, bins=36, range=(0, 180))
        h_peak = h_bins[np.argmax(h_hist)]

        # Mean for saturation and value
        s_mean = int(np.mean(s))
        v_mean = int(np.mean(v))

        return int(h_peak), s_mean, v_mean

    def _classify_color_from_hsv(self, h: int, s: int, v: int) -> str:
        # Priority 1: BLACK (very low brightness)
        if v < self.BLACK_V_THRESHOLD:
            return "black"

        # Priority 2: WHITE (high brightness, low saturation)
        if v > self.WHITE_V_THRESHOLD and s < self.WHITE_S_THRESHOLD:
            return "white"

        # Priority 3: NEUTRAL/GRAY (low saturation)
        if s < self.NEUTRAL_S_THRESHOLD:
            return "neutral"

        # Priority 4: Check against color ranges
        for color_name, ranges in self.COLOR_RANGES.items():
            for h_min, h_max, s_min, v_min in ranges:
                if h_min <= h <= h_max and s >= s_min and v >= v_min:
                    return color_name

        if h < 10 or h > 165:
            return "red"
        elif h < 25:
            return "orange"
        elif h < 40:
            return "yellow"
        elif h < 85:
            return "green"
        elif h < 115:
            return "blue"
        elif h < 135:
            return "indigo"
        elif h < 165:
            return "violet"

        return "neutral"

    def get_object_color(self, frame: np.ndarray, detection: Dict) -> str:
        # Get multiple samples from the object
        samples = self._extract_color_samples(frame, detection)
        
        if not samples:
            return "neutral"

        color_votes = []

        for roi in samples:
            if roi.size == 0:
                continue

            # Apply slight blur to reduce noise
            roi = cv2.GaussianBlur(roi, (3, 3), 0)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Get dominant HSV values
            h, s, v = self._analyze_hsv_histogram(hsv)

            # Classify this sample
            color = self._classify_color_from_hsv(h, s, v)
            color_votes.append(color)

        if not color_votes:
            return "neutral"

        # Voting: most common color wins
        color_counter = Counter(color_votes)
        best_color, count = color_counter.most_common(1)[0]

        # Require at least 40% agreement for confidence
        if count < len(color_votes) * 0.4:
            # If no clear winner, prefer chromatic over neutral
            for color, _ in color_counter.most_common():
                if color not in ['neutral', 'white', 'black']:
                    return color

        return best_color

    def get_object_color_with_temporal(self, frame: np.ndarray, 
                                        detection: Dict) -> str:
        # Get current color
        current_color = self.get_object_color(frame, detection)

        # Create key for this object (class + approximate position)
        cx, cy = detection["center"]
        key = f"{detection['class_id']}_{cx//50}_{cy//50}"

        # Initialize history if needed
        if key not in self.color_history:
            self.color_history[key] = deque(maxlen=self.history_length)

        # Add to history
        self.color_history[key].append(current_color)

        # Vote from history
        if len(self.color_history[key]) >= 2:
            counter = Counter(self.color_history[key])
            return counter.most_common(1)[0][0]

        return current_color

 
    # OBJECT DETECTION
    def detect_objects(self, frame: np.ndarray, 
                       conf_threshold: float = 0.5) -> List[Dict]:
        # FPS calculation
        current_time = time.time()
        dt = current_time - self.prev_frame_time
        if dt > 0:
            self.current_fps = 1.0 / dt
        self.prev_frame_time = current_time

        # Enhance image
        enhanced_frame = self.enhance_image(frame)
        orig_h, orig_w = frame.shape[:2]

        # Resize for detection
        det_w, det_h = self.detection_size
        resized_frame = cv2.resize(enhanced_frame, (det_w, det_h))
        scale_x = orig_w / det_w
        scale_y = orig_h / det_h

        # Run YOLO detection with optimized NMS
        results = self.model(
            resized_frame,
            conf=max(conf_threshold, self.nms_conf),
            iou=self.nms_iou,
            verbose=False
        )

        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())

                class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else 'unknown'
                boost = self.class_conf_boost.get(class_name, 0)
                effective_threshold = max(0.2, conf_threshold - boost)

                if conf < effective_threshold:
                    continue

                # Scale back to original size
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                # Ensure valid bbox
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(orig_w, x2)
                y2 = min(orig_h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Get board coordinates
                u_map, v_map = self.pixel_to_board_coords(cx, cy, orig_w, orig_h)

                detection = {
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'center': (cx, cy),
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'map_coord_cm': (u_map, v_map),
                }

                # Get color with temporal smoothing
                detection['color'] = self.get_object_color_with_temporal(
                    enhanced_frame, detection
                )

                detections.append(detection)

        # Apply temporal smoothing to detection positions
        detections = self._smooth_detections(detections)

        return detections

    def _smooth_detections(self, detections: List[Dict]) -> List[Dict]:
        smoothing = self.smoothing_factor

        for det in detections:
            cls_id = det['class_id']

            # Initialize history
            if cls_id not in self.detection_history:
                self.detection_history[cls_id] = deque(maxlen=self.history_length)

            history = self.detection_history[cls_id]

            # Find matching previous detection
            matched = False
            for prev in history:
                dx = abs(det['center'][0] - prev['center'][0])
                dy = abs(det['center'][1] - prev['center'][1])
                
                if dx < 60 and dy < 60:  # Same object
                    # Smooth confidence
                    det['confidence'] = (
                        prev['confidence'] * smoothing +
                        det['confidence'] * (1 - smoothing)
                    )

                    # Smooth coordinates
                    u_curr, v_curr = det['map_coord_cm']
                    u_prev, v_prev = prev['map_coord_cm']
                    
                    det['map_coord_cm'] = (
                        u_prev * smoothing + u_curr * (1 - smoothing),
                        v_prev * smoothing + v_curr * (1 - smoothing)
                    )
                    matched = True
                    break

            # Add to history
            history.append(det.copy())

        return detections

    def get_fps(self) -> float:
        return self.current_fps

    def set_performance_mode(self, mode: str):
        self._configure_performance(mode)
        self.performance_mode = mode
        print(f"Performance mode changed to: {mode}")

    def classify_by_color(self, frame: np.ndarray, 
                          detections: List[Dict]) -> Dict[str, List[Dict]]:
        bins: Dict[str, List[Dict]] = {}
        
        for det in detections:
            color = det.get('color', 'neutral')
            if color not in bins:
                bins[color] = []
            bins[color].append(det)
        
        return bins

    def draw_detections(self, frame: np.ndarray, 
                        detections: List[Dict],
                        show_colors: bool = True) -> np.ndarray:
        annotated = frame.copy()

        # FPS and mode display
        fps_text = f"FPS: {self.current_fps:.1f} | Mode: {self.performance_mode}"
        cv2.putText(annotated, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Detection count
        cv2.putText(annotated, f"Objects: {len(detections)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["center"]
            color_name = det.get("color", "neutral")
            conf = det["confidence"]
            name = det["class_name"]
            u_map, v_map = det.get("map_coord_cm", (0.0, 0.0))

            # Get visualization color
            box_color = self.color_map.get(color_name, (200, 200, 200))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Draw center point
            cv2.circle(annotated, (cx, cy), 4, box_color, -1)

            # Build label
            if show_colors:
                label = f"{name} {conf:.2f} | {color_name} | ({u_map:.1f}, {v_map:.1f})cm"
            else:
                label = f"{name} {conf:.2f}"

            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), box_color, -1)

            # Draw label text
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return annotated

    def calculate_object_orientation(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox

        # Extract object region with small padding
        padding = 5
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(frame.shape[1], x2 + padding)
        y2_pad = min(frame.shape[0], y2 + padding)

        roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]

        if roi.size == 0:
            return 90.0  # Default to vertical

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use adaptive thresholding for better edge detection
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fallback: use Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 90.0  # Default to vertical

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        if len(largest_contour) < 5:
            return 90.0

        # Method 1: Try fitting an ellipse
        try:
            ellipse = cv2.fitEllipse(largest_contour)
            center, axes, angle = ellipse

            # axes = (minor_axis, major_axis)
            # If object is more elongated, use ellipse angle
            if axes[0] > 0 and axes[1] / axes[0] > 1.3:  
                # OpenCV ellipse angle is 0-180
                return angle if angle <= 180 else angle - 180

        except cv2.error:
            pass

        # Method 2: Use minimum area rectangle
        try:
            rect = cv2.minAreaRect(largest_contour)
            center, size, angle = rect

            # size = (width, height)
            width, height = size

            # Determine if object is elongated
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)

                if aspect_ratio > 1.3:  # Object is elongated
                    # minAreaRect angle is -90 to 0
                    # Convert to 0-180 range
                    if width < height:
                        # Object is more vertical
                        angle = angle + 90
                    else:
                        # Object is more horizontal
                        angle = angle + 180 if angle < -45 else angle + 90

                    # Normalize to 0-180
                    angle = angle % 180
                    return angle

        except cv2.error:
            pass

        # Default: assume vertical orientation
        return 90.0

    def calibrate_colors(self, frame: np.ndarray) -> Dict:
        """
        Utility to help calibrate color thresholds.
        Shows HSV values at different points.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        
        # Sample 9 points in a grid
        points = []
        for i in range(3):
            for j in range(3):
                x = int(w * (0.25 + 0.25 * j))
                y = int(h * (0.25 + 0.25 * i))
                h_val = int(hsv[y, x, 0])
                s_val = int(hsv[y, x, 1])
                v_val = int(hsv[y, x, 2])
                points.append({
                    'position': (x, y),
                    'hsv': (h_val, s_val, v_val)
                })
        
        return {'sample_points': points}

# TEST CODE
if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced Vision Module Test")
    print("=" * 60)
    print("\nControls:")
    print("  Q - Quit")
    print("  1/2/3 - Switch mode (fast/balanced/accurate)")
    print("  W - Toggle white balance")
    print("  C - Print calibration info")
    print("=" * 60)

    vision = VisionModule(performance_mode='balanced')
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        exit(1)

    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Detect objects
        detections = vision.detect_objects(frame, conf_threshold=0.5)

        # Draw detections
        annotated = vision.draw_detections(frame, detections, show_colors=True)

        # Show color distribution
        color_bins = vision.classify_by_color(frame, detections)
        y_pos = 90
        for color, objs in sorted(color_bins.items()):
            text = f"{color}: {len(objs)}"
            cv2.putText(annotated, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 20

        # Display
        cv2.imshow('Enhanced Vision Module', annotated)

        # Handle input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('1'):
            vision.set_performance_mode('fast')
        elif key == ord('2'):
            vision.set_performance_mode('balanced')
        elif key == ord('3'):
            vision.set_performance_mode('accurate')
        elif key == ord('w'):
            vision.enable_white_balance = not vision.enable_white_balance
            print(f"White balance: {'ON' if vision.enable_white_balance else 'OFF'}")
        elif key == ord('c'):
            cal = vision.calibrate_colors(frame)
            print("\nCalibration samples:")
            for pt in cal['sample_points']:
                print(f"  Position {pt['position']}: HSV = {pt['hsv']}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Test complete")