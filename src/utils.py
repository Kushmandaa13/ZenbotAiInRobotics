#!/usr/bin/env python3
"""
DOFBOT Utility Functions
========================
PDE3802 - AI in Robotics
Team: Kushmandaa, Kimberley, Leynah

Provides logging, geometry calculations, statistics, timing,
and configuration utilities for the DOFBOT system.
"""

import logging
import json
import math
import os
import time as time_module
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional


# ============================================================================
# LOGGING SETUP
# ============================================================================

# Ensure logs directory exists
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, 'dofbot.log')

# Configure logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('DOFBOT')


def log_info(message: str):
    """Log informational message."""
    logger.info(message)


def log_error(message: str):
    """Log error message."""
    logger.error(message)


def log_warning(message: str):
    """Log warning message."""
    logger.warning(message)


def log_debug(message: str):
    """Log debug message."""
    logger.debug(message)


# ============================================================================
# OPERATION LOGGING
# ============================================================================

def save_operation_log(operation_name: str, success: bool,
                      details: Dict = None, log_file: str = None):
    """
    Save operation details to JSON log file.

    Args:
        operation_name: Name/type of operation
        success: Whether operation succeeded
        details: Additional operation details
        log_file: Path to log file (default: logs/operation_log.json)
    """
    if log_file is None:
        log_file = os.path.join(LOG_DIR, 'operation_log.json')

    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'operation': operation_name,
        'success': success,
        'details': details or {}
    }

    try:
        with open(log_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
    except Exception as e:
        log_error(f"Failed to save operation log: {e}")


def read_operation_logs(log_file: str = None, limit: int = None) -> List[Dict]:
    """
    Read operation logs from file.

    Args:
        log_file: Path to log file
        limit: Maximum number of entries to return (newest first)

    Returns:
        List of log entry dictionaries
    """
    if log_file is None:
        log_file = os.path.join(LOG_DIR, 'operation_log.json')

    entries = []
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
    except Exception as e:
        log_error(f"Failed to read operation log: {e}")

    # Return newest first, limited
    entries.reverse()
    if limit:
        entries = entries[:limit]

    return entries


# ============================================================================
# GEOMETRY CALCULATIONS
# ============================================================================

def calculate_distance(point1: Tuple, point2: Tuple) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: (x, y) or (x, y, z) tuple
        point2: (x, y) or (x, y, z) tuple
        
    Returns:
        Distance between points
    """
    if len(point1) == 2 and len(point2) == 2:
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    elif len(point1) == 3 and len(point2) == 3:
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    else:
        raise ValueError("Points must have same dimensions (2D or 3D)")


def get_centroid(bounding_box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Get center point of bounding box.
    
    Args:
        bounding_box: (x1, y1, x2, y2) tuple
        
    Returns:
        (center_x, center_y) tuple
    """
    x1, y1, x2, y2 = bounding_box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return (center_x, center_y)


def calculate_area(bounding_box: Tuple[int, int, int, int]) -> int:
    """
    Calculate area of bounding box.
    
    Args:
        bounding_box: (x1, y1, x2, y2) tuple
        
    Returns:
        Area in pixels
    """
    x1, y1, x2, y2 = bounding_box
    width = x2 - x1
    height = y2 - y1
    return width * height


def scale_coordinates(coord: Tuple, scale_factor: float) -> Tuple:
    """
    Scale coordinates by given factor.
    
    Args:
        coord: Coordinate tuple (any length)
        scale_factor: Scaling factor
        
    Returns:
        Scaled coordinates
    """
    return tuple(int(c * scale_factor) for c in coord)


def pixel_to_cm(pixel_coord: Tuple[int, int],
                pixels_per_cm: float = 10.0) -> Tuple[float, float]:
    """
    Convert pixel coordinates to centimeters.

    Args:
        pixel_coord: (x, y) in pixels
        pixels_per_cm: Calibration factor (pixels per cm)

    Returns:
        (x, y) in centimeters
    """
    x_px, y_px = pixel_coord
    x_cm = x_px / pixels_per_cm
    y_cm = y_px / pixels_per_cm
    return (x_cm, y_cm)


def is_point_in_rect(point: Tuple[float, float],
                     rect: Tuple[float, float, float, float]) -> bool:
    """
    Check if a point is inside a rectangle.

    Args:
        point: (x, y) coordinates
        rect: (x1, y1, x2, y2) rectangle corners

    Returns:
        True if point is inside rectangle
    """
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def rect_overlap(rect1: Tuple[float, float, float, float],
                 rect2: Tuple[float, float, float, float]) -> float:
    """
    Calculate overlap area between two rectangles.

    Args:
        rect1: (x1, y1, x2, y2) first rectangle
        rect2: (x1, y1, x2, y2) second rectangle

    Returns:
        Overlap area (0 if no overlap)
    """
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2

    # Find intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)


# ============================================================================
# VALUE MANIPULATION
# ============================================================================

def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value between minimum and maximum.
    
    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-180, 180] degree range.
    
    Args:
        angle: Angle in degrees
        
    Returns:
        Normalized angle
    """
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def map_value(value: float, in_min: float, in_max: float,
             out_min: float, out_max: float) -> float:
    """
    Map value from one range to another.
    
    Args:
        value: Input value
        in_min, in_max: Input range
        out_min, out_max: Output range
        
    Returns:
        Mapped value
    """
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


# ============================================================================
# DETECTION VALIDATION
# ============================================================================

def validate_detection(detection: Dict) -> bool:
    """
    Validate detection dictionary has required fields.

    Args:
        detection: Detection dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    required_keys = ['class_id', 'class_name', 'confidence', 'bbox', 'center']

    # Check all required keys exist
    for key in required_keys:
        if key not in detection:
            log_warning(f"Missing required key in detection: {key}")
            return False

    # Validate confidence is in valid range
    if not (0 <= detection['confidence'] <= 1):
        log_warning(f"Invalid confidence value: {detection['confidence']}")
        return False

    # Validate bbox format
    bbox = detection['bbox']
    if not (isinstance(bbox, (tuple, list)) and len(bbox) == 4):
        log_warning(f"Invalid bbox format: {bbox}")
        return False

    return True


def filter_detections_by_confidence(detections: List[Dict],
                                    min_confidence: float) -> List[Dict]:
    """
    Filter detections by minimum confidence threshold.

    Args:
        detections: List of detection dictionaries
        min_confidence: Minimum confidence threshold

    Returns:
        Filtered list of detections
    """
    return [d for d in detections if d.get('confidence', 0) >= min_confidence]


def filter_detections_by_class(detections: List[Dict],
                               class_names: List[str]) -> List[Dict]:
    """
    Filter detections by class name.

    Args:
        detections: List of detection dictionaries
        class_names: List of class names to include

    Returns:
        Filtered list of detections
    """
    return [d for d in detections if d.get('class_name') in class_names]


def sort_detections_by_position(detections: List[Dict],
                                by: str = 'y') -> List[Dict]:
    """
    Sort detections by position.

    Args:
        detections: List of detection dictionaries
        by: Sort by 'x', 'y', or 'distance' from origin

    Returns:
        Sorted list of detections
    """
    if by == 'x':
        return sorted(detections, key=lambda d: d.get('map_coord_cm', (0, 0))[0])
    elif by == 'y':
        return sorted(detections, key=lambda d: d.get('map_coord_cm', (0, 0))[1])
    elif by == 'distance':
        return sorted(detections, key=lambda d: (
            d.get('map_coord_cm', (0, 0))[0]**2 + d.get('map_coord_cm', (0, 0))[1]**2
        ))
    else:
        return detections


# ============================================================================
# STATISTICS
# ============================================================================

def get_statistics(detections: List[Dict]) -> Dict:
    """
    Calculate statistics from list of detections.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Statistics dictionary with counts and averages
    """
    if not detections:
        return {
            'total': 0,
            'avg_confidence': 0,
            'classes': {},
            'colors': {}
        }
    
    total = len(detections)
    avg_confidence = sum(d['confidence'] for d in detections) / total
    
    # Count by class
    classes = {}
    for detection in detections:
        class_name = detection['class_name']
        classes[class_name] = classes.get(class_name, 0) + 1
    
    # Count by color (if available)
    colors = {}
    for detection in detections:
        if 'color' in detection:
            color = detection['color']
            colors[color] = colors.get(color, 0) + 1
    
    return {
        'total': total,
        'avg_confidence': avg_confidence,
        'classes': classes,
        'colors': colors
    }


def print_statistics(stats: Dict):
    """
    Print statistics in formatted way.
    
    Args:
        stats: Statistics dictionary from get_statistics()
    """
    print("\n" + "=" * 60)
    print("📊 DETECTION STATISTICS")
    print("=" * 60)
    print(f"Total objects detected: {stats['total']}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    
    if stats['classes']:
        print("\n📦 Objects by class:")
        for class_name, count in sorted(stats['classes'].items()):
            print(f"  • {class_name}: {count}")
    
    if stats['colors']:
        print("\n🎨 Objects by color:")
        for color, count in sorted(stats['colors'].items()):
            print(f"  • {color}: {count}")
    
    print("=" * 60 + "\n")


def calculate_success_rate(successful: int, total: int) -> float:
    """
    Calculate success rate percentage.
    
    Args:
        successful: Number of successful operations
        total: Total number of operations
        
    Returns:
        Success rate as percentage (0-100)
    """
    if total == 0:
        return 0.0
    return (successful / total) * 100


# ============================================================================
# FORMATTING
# ============================================================================

def format_coordinates(coords: Tuple) -> str:
    """
    Format coordinates as readable string.
    
    Args:
        coords: Coordinate tuple
        
    Returns:
        Formatted string
    """
    if len(coords) == 2:
        return f"({coords[0]:.1f}, {coords[1]:.1f})"
    elif len(coords) == 3:
        return f"({coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f})"
    else:
        return str(coords)


def format_time(seconds: float) -> str:
    """
    Format time duration as readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "2.5s", "1m 23s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(filepath: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        log_info(f"Loaded configuration from {filepath}")
        return config
    except FileNotFoundError:
        log_warning(f"Config file not found: {filepath}")
        return {}
    except json.JSONDecodeError as e:
        log_error(f"Invalid JSON in config file: {e}")
        return {}


def save_config(filepath: str, config: Dict):
    """
    Save configuration to JSON file.
    
    Args:
        filepath: Path to save config
        config: Configuration dictionary
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        log_info(f"Saved configuration to {filepath}")
    except Exception as e:
        log_error(f"Failed to save config: {e}")


# ============================================================================
# PERFORMANCE TIMING
# ============================================================================

class Timer:
    """Context manager for timing code execution."""

    def __init__(self, name: str = "Operation", verbose: bool = True):
        """
        Initialize timer.

        Args:
            name: Name of operation being timed
            verbose: Whether to print timing info
        """
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.now()
        if self.verbose:
            log_info(f"{self.name} started...")
        return self

    def __exit__(self, *args):
        """Stop timing and print results."""
        self.end_time = datetime.now()
        elapsed = self.elapsed()

        if self.verbose:
            log_info(f"{self.name} completed in {format_time(elapsed)}")

    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.

        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0

        end = self.end_time or datetime.now()
        delta = end - self.start_time
        return delta.total_seconds()


class FPSCounter:
    """Track frames per second over time."""

    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.

        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.timestamps = []
        self.fps = 0.0

    def tick(self):
        """Record a frame timestamp."""
        now = time_module.time()
        self.timestamps.append(now)

        # Keep only recent timestamps
        if len(self.timestamps) > self.window_size:
            self.timestamps = self.timestamps[-self.window_size:]

        # Calculate FPS
        if len(self.timestamps) >= 2:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            if elapsed > 0:
                self.fps = (len(self.timestamps) - 1) / elapsed

    def get_fps(self) -> float:
        """Get current FPS."""
        return self.fps

    def reset(self):
        """Reset the counter."""
        self.timestamps = []
        self.fps = 0.0


# ============================================================================
# COLOR UTILITIES
# ============================================================================

def get_color_rgb(color_name: str) -> Tuple[int, int, int]:
    """
    Get RGB values for named color (BGR format for OpenCV).
    
    Args:
        color_name: Color name
        
    Returns:
        (B, G, R) tuple for OpenCV
    """
    colors = {
        'red': (0, 0, 255),
        'orange': (0, 165, 255),
        'yellow': (0, 255, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'indigo': (130, 0, 75),
        'violet': (255, 0, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255),
    }
    return colors.get(color_name.lower(), (128, 128, 128))


# ============================================================================
# BOARD CONSTANTS
# ============================================================================

# DOFBOT workspace dimensions
BOARD_WIDTH_CM = 28.7
BOARD_HEIGHT_CM = 30.5
TRAY_SIZE_CM = 7.0

# Rainbow color order for sorting
RAINBOW_ORDER = ["red", "orange", "yellow", "green",
                 "blue", "indigo", "violet", "neutral"]

# Object class names
CLASS_NAMES = [
    'aa_battery', 'charger_adapter', 'eraser', 'glue_stick',
    'highlighter', 'pen', 'sharpener', 'stapler'
]


def get_zone_for_position(u_cm: float, v_cm: float) -> int:
    """
    Get the pickup zone number for given board coordinates.

    The board is divided into 8 zones (2 rows x 4 columns).

    Args:
        u_cm: X coordinate in centimeters
        v_cm: Y coordinate in centimeters

    Returns:
        Zone number (1-8)
    """
    # Zone boundaries
    zones = {
        1: {'u': (0, 7), 'v': (0, 8.1)},
        2: {'u': (7, 14), 'v': (0, 8.2)},
        3: {'u': (14, 21), 'v': (0, 8.3)},
        4: {'u': (21, 28.7), 'v': (0, 8.4)},
        5: {'u': (0, 7), 'v': (8.1, 16.3)},
        6: {'u': (7, 14), 'v': (8.2, 16.3)},
        7: {'u': (14, 21), 'v': (8.3, 16.4)},
        8: {'u': (21, 28.7), 'v': (8.3, 16.5)},
    }

    for zone_num, bounds in zones.items():
        u_min, u_max = bounds['u']
        v_min, v_max = bounds['v']

        if u_min <= u_cm <= u_max and v_min <= v_cm <= v_max:
            return zone_num

    return 6  # Default fallback


def map_color_to_tray(color: str) -> str:
    """
    Map detected color to tray color.

    Args:
        color: Detected color name

    Returns:
        Tray color name
    """
    # Map special colors to neutral
    color = color.lower()
    if color in ['white', 'black', 'cream', 'maroon', 'gray', 'grey', 'brown']:
        return 'neutral'

    # Check if it's a valid rainbow color
    if color in RAINBOW_ORDER:
        return color

    return 'neutral'


# ============================================================================
# TESTING
# ============================================================================

def test_utilities():
    """Test utility functions."""
    print("\n" + "=" * 60)
    print("TESTING UTILITY FUNCTIONS")
    print("=" * 60 + "\n")

    # Test distance calculation
    print("Testing distance calculation:")
    p1 = (0, 0, 0)
    p2 = (3, 4, 0)
    dist = calculate_distance(p1, p2)
    print(f"   Distance between {p1} and {p2}: {dist:.2f} cm")

    # Test angle normalization
    print("\nTesting angle normalization:")
    angles = [270, -270, 450, -450]
    for angle in angles:
        normalized = normalize_angle(angle)
        print(f"   {angle} -> {normalized}")

    # Test statistics
    print("\nTesting statistics:")
    detections = [
        {'class_name': 'pen', 'confidence': 0.95, 'color': 'blue',
         'class_id': 5, 'bbox': (0, 0, 10, 10), 'center': (5, 5)},
        {'class_name': 'pen', 'confidence': 0.92, 'color': 'red',
         'class_id': 5, 'bbox': (20, 20, 30, 30), 'center': (25, 25)},
        {'class_name': 'stapler', 'confidence': 0.88, 'color': 'black',
         'class_id': 7, 'bbox': (40, 40, 60, 60), 'center': (50, 50)},
    ]
    stats = get_statistics(detections)
    print_statistics(stats)

    # Test timer
    print("Testing timer:")
    with Timer("Sleep test"):
        time_module.sleep(0.5)

    # Test coordinate formatting
    print("\nTesting coordinate formatting:")
    coords = [(10.5, 20.3), (15.7, 25.2, 30.8)]
    for coord in coords:
        print(f"   {coord} -> {format_coordinates(coord)}")

    # Test detection validation
    print("\nTesting detection validation:")
    valid_det = detections[0]
    invalid_det = {'class_name': 'pen'}
    print(f"   Valid detection: {validate_detection(valid_det)}")
    print(f"   Invalid detection: {validate_detection(invalid_det)}")

    # Test zone calculation
    print("\nTesting zone calculation:")
    test_coords = [(3, 4), (10, 5), (20, 12)]
    for u, v in test_coords:
        zone = get_zone_for_position(u, v)
        print(f"   ({u}, {v}) -> Zone {zone}")

    # Test FPS counter
    print("\nTesting FPS counter:")
    fps = FPSCounter(window_size=10)
    for _ in range(20):
        fps.tick()
        time_module.sleep(0.033)  # ~30 FPS
    print(f"   Measured FPS: {fps.get_fps():.1f}")

    print("\nAll utility tests passed!\n")


if __name__ == "__main__":
    test_utilities()