import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict

class VisionModule:
    def __init__(self, model_path: str = 'models/best.pt'):
        print(f"🔧 Loading YOLOv8 model from: {model_path}")
        self.model = YOLO(model_path)
        
        self.class_names = [
            'aa_battery', 'charger_adapter', 'eraser', 'glue_stick', 
            'highlighter', 'pen', 'sharpener', 'stapler'
        ]
        
        # --- SHADE DEFINITIONS (Same as before) ---
        self.shade_definitions = {
            'black_pure':   ((0, 0, 0), (180, 255, 50), 'BLACK'),
            'white_pure':   ((0, 0, 200), (180, 50, 255), 'WHITE'),
            'gray_light':   ((0, 0, 150), (180, 60, 220), 'GRAY'),
            'gray_dark':    ((0, 0, 50), (180, 60, 150), 'GRAY'),
            'red_pure':     ((0, 100, 100), (10, 255, 255), 'RED'),
            'red_wrap':     ((175, 100, 100), (180, 255, 255), 'RED'),
            'maroon':       ((0, 50, 50), (10, 255, 150), 'RED'),
            'tomato':       ((0, 50, 150), (10, 120, 255), 'RED'),
            'orange_vibrant': ((11, 100, 100), (25, 255, 255), 'ORANGE'),
            'orange_pale':    ((11, 50, 150), (25, 120, 255), 'ORANGE'),
            'brown_dark':     ((10, 50, 20), (30, 255, 150), 'BROWN'),
            'yellow_pure':    ((26, 100, 100), (35, 255, 255), 'YELLOW'),
            'yellow_pale':    ((26, 40, 150), (35, 100, 255), 'YELLOW'),
            'green_pure':     ((36, 100, 50), (85, 255, 255), 'GREEN'),
            'lime_green':     ((36, 50, 200), (85, 255, 255), 'GREEN'),
            'forest_green':   ((36, 50, 20), (85, 255, 150), 'GREEN'),
            'blue_pure':      ((86, 100, 50), (125, 255, 255), 'BLUE'),
            'sky_blue':       ((86, 30, 150), (125, 100, 255), 'BLUE'),
            'navy_blue':      ((86, 50, 20), (125, 255, 150), 'BLUE'),
            'indigo_pure':    ((126, 50, 50), (140, 255, 255), 'INDIGO'),
            'violet_pure':    ((141, 50, 50), (160, 255, 255), 'VIOLET'),
            'pink_hot':       ((161, 50, 150), (174, 255, 255), 'VIOLET'),
        }

    def detect_objects(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        results = self.model(frame, conf=0.25, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                if conf < conf_threshold:
                    final_name = "Unknown"
                else:
                    final_name = self.class_names[cls_id]

                detections.append({
                    'class_id': cls_id,
                    'class_name': final_name,
                    'confidence': conf,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': (int((x1+x2)/2), int((y1+y2)/2)),
                    'width': int(x2-x1),
                    'height': int(y2-y1)
                })
        return detections

    def get_object_color(self, frame: np.ndarray, detection: Dict) -> str:
        x1, y1, x2, y2 = detection['bbox']
        w, h = detection['width'], detection['height']
        roi_x1 = x1 + int(w * 0.35)
        roi_y1 = y1 + int(h * 0.35)
        roi_x2 = x2 - int(w * 0.35)
        roi_y2 = y2 - int(h * 0.35)
        
        if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi.size == 0: return 'unknown'

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        parent_votes = {}
        
        for shade_name, (lower, upper, parent_category) in self.shade_definitions.items():
            lower_np = np.array(lower, dtype="uint8")
            upper_np = np.array(upper, dtype="uint8")
            mask = cv2.inRange(hsv_roi, lower_np, upper_np)
            count = cv2.countNonZero(mask)
            
            if count > 0:
                if parent_category in parent_votes:
                    parent_votes[parent_category] += count
                else:
                    parent_votes[parent_category] = count

        if not parent_votes:
            return 'GRAY'
            
        winner = max(parent_votes, key=parent_votes.get)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        if parent_votes[winner] < (total_pixels * 0.05):
            avg_val = np.mean(hsv_roi[:, :, 2])
            if avg_val > 200: return 'WHITE'
            if avg_val < 50: return 'BLACK'
            return 'GRAY'

        return winner

    def classify_by_color(self, frame: np.ndarray, detections: List[Dict]) -> None:
        for det in detections:
            det['color'] = self.get_object_color(frame, det)

    def draw_detections(self, frame: np.ndarray, detections: List[Dict], show_colors: bool = False) -> np.ndarray:
        annotated = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            name = detection['class_name']
            conf_percent = int(detection['confidence'] * 100) # Convert 0.85 to 85
            
            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            
            # Draw Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # --- LABEL FORMATTING UPDATED HERE ---
            # Format: "PEN (85%)"
            label = f"{name} ({conf_percent}%)"
            
            if show_colors and 'color' in detection and name != "Unknown":
                # Format: "PEN (85%) - RED"
                label += f" - {detection['color']}"
            
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
        return annotated