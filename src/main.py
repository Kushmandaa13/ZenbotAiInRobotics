#!/usr/bin/env python3
import cv2
import time
import sys
import os
import numpy as np
from typing import Tuple, List, Dict, Optional

# Handle imports from different locations
try:
    from vision_module import VisionModule
    from arm_controller import ArmController
    from utils import (log_info, log_error, log_warning, save_operation_log,
                       get_statistics, print_statistics, Timer, format_time,
                       validate_detection, calculate_success_rate)
except ImportError:
    from src.vision_module import VisionModule
    from src.arm_controller import ArmController
    from src.utils import (log_info, log_error, log_warning, save_operation_log,
                           get_statistics, print_statistics, Timer, format_time,
                           validate_detection, calculate_success_rate)


class DOFBOTSystem:
    # Rainbow order for sorting
    RAINBOW_ORDER = ["red", "orange", "yellow", "green",
                     "blue", "indigo", "violet", "neutral"]

    def __init__(self, model_path: str = None,
                 arm_port: str = None,
                 camera_id: int = 0,
                 performance_mode: str = 'balanced'):
        
        # Resolve model path
        if model_path is None:
            # Try common locations
            possible_paths = [
                'models/best.pt',
                '../models/best.pt',
                os.path.join(os.path.dirname(__file__), '..', 'models', 'best.pt')
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            else:
                model_path = 'models/best.pt'  # Default

        self._print_banner()

        log_info("Initializing DOFBOT System...")

        # Initialize vision module
        print(f"Loading vision module (model: {model_path})...")
        try:
            self.vision = VisionModule(model_path, performance_mode=performance_mode)
            self.vision_ready = True
        except Exception as e:
            log_error(f"Failed to load vision module: {e}")
            self.vision = None
            self.vision_ready = False

        # Initialize arm controller
        print("Loading arm controller...")
        try:
            self.arm = ArmController(port=arm_port)
            self.arm_ready = True
        except Exception as e:
            log_error(f"Failed to load arm controller: {e}")
            self.arm = None
            self.arm_ready = False

        # Camera settings
        self.camera_id = camera_id
        self.cap = None

        # Operation statistics
        self.operation_count = 0
        self.successful_operations = 0
        self.failed_operations = 0

        # Current state
        self.current_frame = None
        self.current_detections = []

        if self.vision_ready:
            log_info("DOFBOT System initialized successfully")
        else:
            log_warning("DOFBOT System initialized with errors")

    def _print_banner(self):
        """Print startup banner."""
        print("\n" + "=" * 70)
        print("DOFBOT SMART DESK-TIDYING ROBOT")
        print("=" * 70)
        print("Project: PDE3802 - AI in Robotics")
        print("Team: Kushmandaa, Kimberley, Leynah")
        print("=" * 70)
        print("\nYour 8 Object Classes:")
        print("   1. AA Battery       5. Highlighter")
        print("   2. Charger Adapter  6. Pen")
        print("   3. Eraser           7. Sharpener")
        print("   4. Glue Stick       8. Stapler")
        print("=" * 70 + "\n")
    
    def connect_arm(self) -> bool:
        """
        Connect to robotic arm.

        Returns:
            True if connected successfully
        """
        if not self.arm_ready or self.arm is None:
            log_error("Arm controller not initialized")
            return False

        log_info("Connecting to arm...")

        if self.arm.connect():
            log_info("Arm connected successfully")
            return True
        else:
            log_error("Failed to connect to arm")
            return False

    def open_camera(self) -> bool:
        """
        Open camera for capturing frames.

        Returns:
            True if camera opened successfully
        """
        if self.cap is not None and self.cap.isOpened():
            return True

        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            log_error(f"Cannot open camera {self.camera_id}")
            return False

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        log_info(f"Camera {self.camera_id} opened")
        return True

    def close_camera(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            log_info("Camera released")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from camera.

        Returns:
            Frame as numpy array, or None if failed
        """
        if not self.open_camera():
            return None

        ret, frame = self.cap.read()

        if not ret:
            log_error("Failed to capture frame")
            return None

        self.current_frame = frame
        return frame

    def scan_desk(self, conf_threshold: float = 0.5,
                  show_preview: bool = False) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """
        Scan desk and detect all objects using camera.

        Args:
            conf_threshold: Confidence threshold for detection
            show_preview: Whether to show preview window

        Returns:
            Tuple of (frame, detections)
        """
        if not self.vision_ready:
            log_error("Vision module not ready")
            return None, []

        log_info("Scanning desk...")

        # Capture frame
        frame = self.capture_frame()

        if frame is None:
            return None, []

        # Detect objects
        with Timer("Object detection", verbose=False):
            detections = self.vision.detect_objects(frame, conf_threshold)

        self.current_detections = detections
        log_info(f"Detected {len(detections)} objects")

        # Show preview if requested
        if show_preview and detections:
            annotated = self.vision.draw_detections(frame, detections, show_colors=True)
            cv2.imshow('Desk Scan', annotated)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

        return frame, detections
    
    def classify_by_color(self, frame: np.ndarray,
                         detections: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Organize detected objects by color for bin sorting.

        Args:
            frame: Camera frame
            detections: List of detections

        Returns:
            Dictionary mapping colors to lists of objects
        """
        log_info("Classifying objects by color...")

        color_bins = self.vision.classify_by_color(frame, detections)

        # Log distribution
        for color, objects in color_bins.items():
            log_info(f"   {color}: {len(objects)} object(s)")

        return color_bins
    
    def sort_single_detection(self, detection: Dict) -> bool:
        """
        Sort a single detected object using the arm controller.

        Uses the detection's map coordinates (already in cm) and color
        to perform a pick-and-place operation.

        Args:
            detection: Detection dictionary with 'map_coord_cm', 'color', 'class_name'

        Returns:
            True if operation successful
        """
        if not self.arm_ready or self.arm is None:
            log_error("Arm not ready")
            return False

        self.operation_count += 1

        try:
            # Get object information
            class_name = detection.get('class_name', 'unknown')
            confidence = detection.get('confidence', 0)
            color = detection.get('color', 'neutral')
            u_map, v_map = detection.get('map_coord_cm', (0, 0))

            log_info(f"\n[Operation #{self.operation_count}]")
            log_info(f"Object: {class_name} ({color})")
            log_info(f"Confidence: {confidence:.2f}")
            log_info(f"Position: ({u_map:.1f}, {v_map:.1f}) cm")

            # Perform pick and place using arm controller
            with Timer(f"Pick-and-place {class_name}"):
                success = self.arm.pick_and_place(u_map, v_map, color, class_name)

            # Log operation
            if success:
                self.successful_operations += 1
                save_operation_log(
                    operation_name=f"pick_and_place_{class_name}",
                    success=True,
                    details={
                        'object': class_name,
                        'color': color,
                        'confidence': confidence,
                        'position_cm': (u_map, v_map)
                    }
                )
                log_info(f"Successfully sorted {class_name}")
            else:
                self.failed_operations += 1
                save_operation_log(
                    operation_name=f"pick_and_place_{class_name}",
                    success=False,
                    details={
                        'object': class_name,
                        'error': 'Movement failed'
                    }
                )
                log_error(f"Failed to sort {class_name}")

            return success

        except Exception as e:
            self.failed_operations += 1
            log_error(f"Error during pick and place: {e}")
            save_operation_log(
                operation_name=f"pick_and_place_{detection.get('class_name', 'unknown')}",
                success=False,
                details={'error': str(e)}
            )
            return False
    
    def sort_all_by_position(self, detections: List[Dict],
                              max_objects: int = None) -> Dict:
        """
        Sort all detected objects by position (closest first).

        Args:
            detections: List of detection dictionaries
            max_objects: Maximum number to sort (None for all)

        Returns:
            Statistics dictionary with results
        """
        # Sort by Y coordinate (v_map) - closest to arm first
        sorted_dets = sorted(detections, key=lambda d: d['map_coord_cm'][1])

        if max_objects:
            sorted_dets = sorted_dets[:max_objects]

        successful = 0
        failed = 0

        for det in sorted_dets:
            if self.sort_single_detection(det):
                successful += 1
            else:
                failed += 1
            time.sleep(0.3)

        return {'total': len(sorted_dets), 'successful': successful, 'failed': failed}

    def sort_by_rainbow_order(self, color_bins: Dict[str, List[Dict]],
                               max_objects: int = None) -> Dict:
        """
        Sort objects in rainbow color order.

        Args:
            color_bins: Dictionary mapping colors to detection lists
            max_objects: Maximum total objects to sort

        Returns:
            Statistics dictionary with results
        """
        successful = 0
        failed = 0
        total = 0

        for color in self.RAINBOW_ORDER:
            if color not in color_bins or not color_bins[color]:
                continue

            log_info(f"\nProcessing {color.upper()} objects...")

            # Sort objects in this color by position
            objects = sorted(color_bins[color], key=lambda d: d['map_coord_cm'][1])

            for det in objects:
                if max_objects and total >= max_objects:
                    break

                if self.sort_single_detection(det):
                    successful += 1
                else:
                    failed += 1

                total += 1
                time.sleep(0.3)

            if max_objects and total >= max_objects:
                break

        return {'total': total, 'successful': successful, 'failed': failed}

    def tidy_desk(self, max_objects: int = None, conf_threshold: float = 0.5,
                  rainbow_order: bool = True):
        """
        Main desk tidying operation - detect and sort all objects.

        Args:
            max_objects: Maximum number of objects to sort (None for all)
            conf_threshold: Detection confidence threshold
            rainbow_order: If True, sort by color order; if False, by position
        """
        print("\n" + "=" * 70)
        print("STARTING DESK TIDYING OPERATION")
        print("=" * 70 + "\n")

        start_time = time.time()

        try:
            # Step 1: Move arm to home position
            log_info("[Step 1/5] Moving arm to home position...")
            if self.arm:
                self.arm.move_home()
            time.sleep(1)

            # Step 2: Scan desk
            log_info("[Step 2/5] Scanning desk for objects...")
            frame, detections = self.scan_desk(
                conf_threshold=conf_threshold,
                show_preview=False
            )

            if not detections:
                print("\n" + "=" * 70)
                print("DESK IS ALREADY CLEAN!")
                print("=" * 70 + "\n")
                return

            # Step 3: Classify by color
            log_info("[Step 3/5] Classifying objects by color...")
            color_bins = self.classify_by_color(frame, detections)

            # Show statistics
            stats = get_statistics(detections)
            print_statistics(stats)

            # Step 4: Sort objects
            log_info("[Step 4/5] Sorting objects into trays...")
            print("\n" + "-" * 70)

            if rainbow_order:
                results = self.sort_by_rainbow_order(color_bins, max_objects)
            else:
                results = self.sort_all_by_position(detections, max_objects)

            print("-" * 70)

            # Step 5: Return to home
            log_info("[Step 5/5] Returning arm to home position...")
            if self.arm:
                self.arm.move_home()

            # Calculate statistics
            elapsed_time = time.time() - start_time
            success_rate = calculate_success_rate(
                self.successful_operations, self.operation_count
            )

            # Print final summary
            print("\n" + "=" * 70)
            print("DESK TIDYING COMPLETE")
            print("=" * 70)
            print(f"Operations: {self.operation_count}")
            print(f"Successful: {self.successful_operations}")
            print(f"Failed: {self.failed_operations}")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Total time: {format_time(elapsed_time)}")
            print("=" * 70 + "\n")

        except KeyboardInterrupt:
            log_warning("Operation interrupted by user")
            if self.arm:
                self.arm.emergency_stop()
            print("\nEmergency stop activated")

        except Exception as e:
            log_error(f"Unexpected error: {e}")
            if self.arm:
                self.arm.emergency_stop()
            print("\nEmergency stop activated")
    
    def demo_mode(self, conf_threshold: float = 0.5):
        """
        Demo mode - vision only without arm control.
        Shows real-time detection and color classification.

        Args:
            conf_threshold: Detection confidence threshold
        """
        if not self.vision_ready:
            log_error("Vision module not ready")
            return

        print("\n" + "=" * 70)
        print("DEMO MODE - Vision Only")
        print("=" * 70)
        print("Your 8 Objects:")
        print("  aa_battery, charger_adapter, eraser, glue_stick")
        print("  highlighter, pen, sharpener, stapler")
        print("\nControls:")
        print("  q - Quit")
        print("  s - Show statistics")
        print("  c - Capture frame")
        print("  1/2/3 - Performance mode (fast/balanced/accurate)")
        print("=" * 70 + "\n")

        if not self.open_camera():
            return

        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                log_error("Failed to capture frame")
                break

            self.current_frame = frame

            # Detect objects
            detections = self.vision.detect_objects(frame, conf_threshold)
            self.current_detections = detections

            # Classify by color
            color_bins = self.vision.classify_by_color(frame, detections)

            # Draw annotations
            annotated = self.vision.draw_detections(frame, detections, show_colors=True)

            # Display
            cv2.imshow('DOFBOT Demo Mode', annotated)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                log_info("Exiting demo mode...")
                break

            elif key == ord('s'):
                stats = get_statistics(detections)
                print_statistics(stats)

            elif key == ord('c'):
                os.makedirs('snapshots', exist_ok=True)
                filename = f'snapshots/demo_capture_{frame_count:04d}.jpg'
                cv2.imwrite(filename, annotated)
                log_info(f"Saved: {filename}")
                frame_count += 1

            elif key == ord('1'):
                self.vision.set_performance_mode('fast')

            elif key == ord('2'):
                self.vision.set_performance_mode('balanced')

            elif key == ord('3'):
                self.vision.set_performance_mode('accurate')

        self.close_camera()
        cv2.destroyAllWindows()
        log_info("Demo mode ended")
    
    def calibrate_camera(self):
        """
        Camera calibration mode to determine pixel-to-cm conversion.
        Shows grid overlay and allows capturing calibration frames.
        """
        print("\n" + "=" * 70)
        print("CAMERA CALIBRATION MODE")
        print("=" * 70)
        print("Instructions:")
        print("  1. Place an object of known size (e.g., 10cm ruler)")
        print("  2. Press 'c' to capture and measure")
        print("  3. Press 'q' to quit")
        print("=" * 70 + "\n")

        if not self.open_camera():
            return

        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Show frame with grid
            height, width = frame.shape[:2]

            # Draw center crosshair
            cv2.line(frame, (width//2, 0), (width//2, height), (0, 255, 0), 1)
            cv2.line(frame, (0, height//2), (width, height//2), (0, 255, 0), 1)

            # Draw grid lines
            for i in range(1, 4):
                x = int(width * i / 4)
                y = int(height * i / 4)
                cv2.line(frame, (x, 0), (x, height), (50, 50, 50), 1)
                cv2.line(frame, (0, y), (width, y), (50, 50, 50), 1)

            cv2.putText(frame, "Place calibration object and press 'c'",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Calibration', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                os.makedirs('calibration', exist_ok=True)
                filename = f'calibration/calib_{frame_count:04d}.jpg'
                cv2.imwrite(filename, frame)
                log_info(f"Calibration captured: {filename}")
                frame_count += 1

        self.close_camera()
        cv2.destroyAllWindows()
    
    def shutdown(self):
        """Safely shutdown the system."""
        log_info("Shutting down DOFBOT system...")

        # Release camera
        self.close_camera()

        # Disconnect arm
        if self.arm and self.arm.connected:
            self.arm.disconnect()

        log_info("System shutdown complete")

    def get_status(self) -> Dict:
        """
        Get current system status.

        Returns:
            Dictionary with status information
        """
        return {
            'vision_ready': self.vision_ready,
            'arm_ready': self.arm_ready,
            'arm_connected': self.arm.connected if self.arm else False,
            'camera_open': self.cap is not None and self.cap.isOpened(),
            'operation_count': self.operation_count,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'success_rate': calculate_success_rate(
                self.successful_operations, self.operation_count
            )
        }

    def reset_statistics(self):
        """Reset operation statistics."""
        self.operation_count = 0
        self.successful_operations = 0
        self.failed_operations = 0
        log_info("Statistics reset")


def main():
    """Main entry point for CLI interface."""
    print("\n")

    # Initialize system
    system = DOFBOTSystem()

    # Main menu loop
    while True:
        print("\n" + "=" * 70)
        print("DOFBOT MAIN MENU")
        print("=" * 70)
        print("1. Tidy Desk (Rainbow Order)")
        print("2. Tidy Desk (Position Order)")
        print("3. Demo Mode (Vision Only)")
        print("4. Calibrate Camera")
        print("5. Test Arm")
        print("6. Show Statistics")
        print("7. Exit")
        print("=" * 70)

        try:
            choice = input("\nSelect option (1-7): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == '1':
            # Tidy desk with rainbow order
            if system.connect_arm():
                max_obj = input("Max objects to sort (Enter for all): ").strip()
                max_obj = int(max_obj) if max_obj else None
                system.tidy_desk(max_objects=max_obj, rainbow_order=True)
            else:
                print("Arm connection failed. Try demo mode instead.")

        elif choice == '2':
            # Tidy desk by position
            if system.connect_arm():
                max_obj = input("Max objects to sort (Enter for all): ").strip()
                max_obj = int(max_obj) if max_obj else None
                system.tidy_desk(max_objects=max_obj, rainbow_order=False)
            else:
                print("Arm connection failed.")

        elif choice == '3':
            # Demo mode (vision only)
            system.demo_mode()

        elif choice == '4':
            # Camera calibration
            system.calibrate_camera()

        elif choice == '5':
            # Test arm movements
            if system.connect_arm():
                print("\nArm Test Options:")
                print("  1. Test all trays")
                print("  2. Test all pickup zones")
                print("  3. Calibration mode")
                print("  4. Single pick & place test")

                test_choice = input("Select test (1-4): ").strip()

                if test_choice == '1':
                    system.arm.test_all_trays()
                elif test_choice == '2':
                    system.arm.test_all_zones()
                elif test_choice == '3':
                    system.arm.calibration_mode()
                elif test_choice == '4':
                    zone = int(input("Zone (1-8): ") or "6")
                    color = input("Target tray color: ") or "neutral"
                    system.arm.test_single_pick_place(zone, color)
            else:
                print("Arm connection failed")

        elif choice == '6':
            # Show statistics
            status = system.get_status()
            print("\n" + "=" * 50)
            print("SYSTEM STATUS")
            print("=" * 50)
            print(f"Vision Ready: {status['vision_ready']}")
            print(f"Arm Ready: {status['arm_ready']}")
            print(f"Arm Connected: {status['arm_connected']}")
            print(f"Camera Open: {status['camera_open']}")
            print("-" * 50)
            print(f"Total Operations: {status['operation_count']}")
            print(f"Successful: {status['successful_operations']}")
            print(f"Failed: {status['failed_operations']}")
            print(f"Success Rate: {status['success_rate']:.1f}%")
            print("=" * 50)

        elif choice == '7':
            # Exit
            system.shutdown()
            print("Goodbye!\n")
            break

        else:
            print("Invalid option")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
        print("Goodbye!\n")
        sys.exit(0)