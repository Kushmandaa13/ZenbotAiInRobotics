import sys
import os
import cv2
import time
from datetime import datetime

# Smart path resolution 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir if os.path.exists(os.path.join(current_dir, 'src')) else os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')

if os.path.exists(src_path):
    sys.path.insert(0, src_path)
    print(f"Using src path: {src_path}")
else:
    print(f"WARNING: src/ directory not found at {src_path}")

try:
    from vision_module import VisionModule
    VISION_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import vision_module: {e}")
    VISION_MODULE_AVAILABLE = False


class CameraVisionTester:
    def __init__(self, project_root):
        self.test_results = {}
        self.project_root = project_root
        
    def test_camera_access(self, camera_id=0):
        """Test if camera can be accessed."""
        print("\n" + "=" * 70)
        print("TEST 1: CAMERA ACCESS")
        print("=" * 70)
        
        try:
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                print(f"FAIL: Cannot open camera {camera_id}")
                self.test_results['camera_access'] = False
                return None
            
            # Try to read a frame
            ret, frame = cap.read()
            
            if not ret:
                print("FAIL: Cannot read frame from camera")
                cap.release()
                self.test_results['camera_access'] = False
                return None
            
            height, width = frame.shape[:2]
            print(f"PASS: Camera opened successfully")
            print(f"   Resolution: {width}x{height}")
            print(f"   Frame shape: {frame.shape}")
            
            # Test FPS
            print("\n   Testing camera FPS...")
            start_time = time.time()
            frame_count = 0
            test_duration = 2  # seconds
            
            while (time.time() - start_time) < test_duration:
                ret, _ = cap.read()
                if ret:
                    frame_count += 1
            
            fps = frame_count / test_duration
            print(f"   Camera FPS: {fps:.2f}")
            
            cap.release()
            self.test_results['camera_access'] = True
            self.test_results['camera_fps'] = fps
            self.test_results['camera_resolution'] = (width, height)
            
            return frame
            
        except Exception as e:
            print(f"FAIL: Error accessing camera: {e}")
            self.test_results['camera_access'] = False
            return None
    
    def test_opencv_installation(self):
        """Test if OpenCV is properly installed."""
        print("\n" + "=" * 70)
        print("TEST 2: OPENCV INSTALLATION")
        print("=" * 70)
        
        try:
            print(f"OpenCV version: {cv2.__version__}")
            
            # Create a simple test image
            import numpy as np
            test_array = np.zeros((100, 100, 3), dtype=np.uint8)
            test_gray = cv2.cvtColor(test_array, cv2.COLOR_BGR2GRAY)
            
            print("PASS: OpenCV basic operations working")
            self.test_results['opencv'] = True
            
        except Exception as e:
            print(f"FAIL: OpenCV error: {e}")
            self.test_results['opencv'] = False
    
    def test_model_loading(self, model_path=None):
        """Test if YOLOv8 model can be loaded."""
        print("\n" + "=" * 70)
        print("TEST 3: MODEL LOADING")
        print("=" * 70)
        
        if not VISION_MODULE_AVAILABLE:
            print("FAIL: VisionModule not available")
            self.test_results['model_loading'] = False
            return None
        
        # Try to find model in multiple locations
        if model_path is None:
            possible_paths = [
                os.path.join(self.project_root, 'models', 'best.pt'),
                os.path.join(self.project_root, 'models', 'yolov8_dofbot.pt'),
                'models/best.pt',
                '../models/best.pt'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if model_path is None or not os.path.exists(model_path):
            print(f"FAIL: Model file not found")
            print(f"   Searched in: {self.project_root}/models/")
            print(f"   Please ensure your trained model is at: models/best.pt")
            self.test_results['model_loading'] = False
            return None
        
        try:
            print(f"Loading model from: {model_path}")
            vision = VisionModule(model_path=model_path)
            
            print(f"PASS: Model loaded successfully")
            print(f"   Classes ({len(vision.class_names)}): {vision.class_names}")
            
            self.test_results['model_loading'] = True
            self.test_results['num_classes'] = len(vision.class_names)
            
            return vision
            
        except Exception as e:
            print(f"FAIL: Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['model_loading'] = False
            return None
    
    def test_detection_speed(self, vision, camera_id=0):
        """Test detection speed with live camera."""
        print("\n" + "=" * 70)
        print("TEST 4: DETECTION SPEED")
        print("=" * 70)
        
        if vision is None:
            print("SKIP: Model not loaded")
            return
        
        try:
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                print("FAIL: Cannot open camera")
                return
            
            print("Running detection speed test (5 seconds)...")
            print("Place objects in front of camera for testing...")
            
            start_time = time.time()
            frame_count = 0
            detection_times = []
            
            while (time.time() - start_time) < 5:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Time the detection
                det_start = time.time()
                detections = vision.detect_objects(frame, conf_threshold=0.5)
                det_time = time.time() - det_start
                
                detection_times.append(det_time)
                frame_count += 1
            
            cap.release()
            
            # Calculate statistics
            avg_time = sum(detection_times) / len(detection_times)
            avg_fps = 1.0 / avg_time
            
            print(f"PASS: Detection speed test complete")
            print(f"   Frames processed: {frame_count}")
            print(f"   Average detection time: {avg_time*1000:.2f} ms")
            print(f"   Average FPS: {avg_fps:.2f}")
            print(f"   Min time: {min(detection_times)*1000:.2f} ms")
            print(f"   Max time: {max(detection_times)*1000:.2f} ms")
            
            self.test_results['detection_fps'] = avg_fps
            self.test_results['avg_detection_time'] = avg_time
            
            if avg_fps < 5:
                print("   WARNING: FPS below target (5 FPS for Assessment 2)")
            else:
                print("   PASS: FPS meets Assessment 2 requirement (>5 FPS)")
            
        except Exception as e:
            print(f"FAIL: Detection speed test error: {e}")
            import traceback
            traceback.print_exc()
    
    def test_live_detection(self, vision, camera_id=0, duration=10):
        """Test live detection with visualization."""
        print("\n" + "=" * 70)
        print("TEST 5: LIVE DETECTION (Interactive)")
        print("=" * 70)
        print(f"\nRunning live detection for {duration} seconds...")
        print("Press 'q' to quit early, 's' to save frame")
        print("=" * 70)
        
        if vision is None:
            print("SKIP: Model not loaded")
            return
        
        try:
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                print("FAIL: Cannot open camera")
                return
            
            start_time = time.time()
            frame_count = 0
            total_detections = 0
            saved_frames = 0
            
            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Detect objects
                detections = vision.detect_objects(frame, conf_threshold=0.5)
                total_detections += len(detections)
                
                # Classify by color
                color_bins = vision.classify_by_color(frame, detections)
                
                # Draw detections
                annotated = vision.draw_detections(frame, detections, show_colors=True)
                
                # Add info overlay
                fps = frame_count / (time.time() - start_time + 0.001)
                cv2.putText(annotated, f"FPS: {fps:.1f} | Objects: {len(detections)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                y_pos = 60
                for color, objects in sorted(color_bins.items()):
                    cv2.putText(annotated, f"{color}: {len(objects)}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 255, 0), 2)
                    y_pos += 25
                
                # Show frame
                cv2.imshow('Camera Vision Test - Press q to quit, s to save', annotated)
                
                frame_count += 1
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nUser quit")
                    break
                elif key == ord('s'):
                    filename = f"test_frame_{saved_frames:03d}.jpg"
                    cv2.imwrite(filename, annotated)
                    print(f"\nSaved: {filename}")
                    saved_frames += 1
            
            cap.release()
            cv2.destroyAllWindows()
            
            avg_detections = total_detections / frame_count if frame_count > 0 else 0
            
            print(f"\nPASS: Live detection test complete")
            print(f"   Total frames: {frame_count}")
            print(f"   Total detections: {total_detections}")
            print(f"   Average detections per frame: {avg_detections:.2f}")
            print(f"   Frames saved: {saved_frames}")
            
            self.test_results['live_detection'] = True
            
        except Exception as e:
            print(f"FAIL: Live detection error: {e}")
            import traceback
            traceback.print_exc()
            cv2.destroyAllWindows()
    
    def test_classification_accuracy(self, vision, camera_id=0):
        """Test classification on a single captured frame."""
        print("\n" + "=" * 70)
        print("TEST 6: CLASSIFICATION TEST")
        print("=" * 70)
        print("\nPlace an object in front of the camera...")
        print("Capturing frame in 3 seconds...")
        
        if vision is None:
            print("SKIP: Model not loaded")
            return
        
        try:
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                print("FAIL: Cannot open camera")
                return
            
            # Countdown
            for i in range(3, 0, -1):
                print(f"{i}...")
                time.sleep(1)
            
            print("Capturing!")
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print("FAIL: Cannot capture frame")
                return
            
            # Detect objects
            detections = vision.detect_objects(frame, conf_threshold=0.5)
            
            print(f"\nDetection Results:")
            print(f"   Objects found: {len(detections)}")
            
            if len(detections) == 0:
                print("   WARNING: No objects detected")
                print("   Try:")
                print("      - Moving object closer to camera")
                print("      - Improving lighting")
                print("      - Using objects from your training classes")
            else:
                # Classify by color
                color_bins = vision.classify_by_color(frame, detections)
                
                print("\n   Detected objects:")
                for i, det in enumerate(detections, 1):
                    print(f"   {i}. {det['class_name']}")
                    print(f"      Confidence: {det['confidence']:.2%}")
                    if 'color' in det:
                        print(f"      Color: {det['color']}")
                    print(f"      BBox: {det['bbox']}")
                
                # Save annotated image
                annotated = vision.draw_detections(frame, detections, show_colors=True)
                filename = f"classification_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"\n   Saved annotated image: {filename}")
            
            self.test_results['classification'] = True
            
        except Exception as e:
            print(f"FAIL: Classification test error: {e}")
            import traceback
            traceback.print_exc()
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for v in self.test_results.values() if v is True)
        total = len([k for k in self.test_results.keys() if isinstance(self.test_results[k], bool)])
        
        for test_name, result in self.test_results.items():
            if isinstance(result, bool):
                status = "PASS" if result else "FAIL"
                print(f"{status}: {test_name}")
        
        print("=" * 70)
        print(f"Results: {passed}/{total} tests passed")
        
        if 'camera_fps' in self.test_results:
            print(f"Camera FPS: {self.test_results['camera_fps']:.2f}")
        if 'detection_fps' in self.test_results:
            print(f"Detection FPS: {self.test_results['detection_fps']:.2f}")
            if self.test_results['detection_fps'] >= 5:
                print("PASS: Meets Assessment 2 requirement (>5 FPS)")
            else:
                print("WARNING: Below Assessment 2 requirement (>5 FPS)")
        if 'num_classes' in self.test_results:
            print(f"Model classes: {self.test_results['num_classes']}")
        
        print("=" * 70)
        
        if passed == total:
            print("\nALL TESTS PASSED!")
            print("Your camera and vision system are working correctly!")
        else:
            print("\nSome tests failed. Please review the errors above.")
        
        print()


def main():
    """Run all camera and vision tests."""
    print("\n" + "=" * 70)
    print("RASPBERRY PI CAMERA & VISION TEST SUITE")
    print("=" * 70)
    print("This will test:")
    print("  1. Camera access")
    print("  2. OpenCV installation")
    print("  3. Model loading")
    print("  4. Detection speed")
    print("  5. Live detection")
    print("  6. Classification accuracy")
    print("=" * 70)
    
    # Ask for camera ID
    camera_input = input("\nEnter camera ID (default 0): ").strip()
    camera_id = int(camera_input) if camera_input else 0
    
    # Ask for model path (optional)
    model_input = input("Enter model path (or press Enter to auto-detect): ").strip()
    model_path = model_input if model_input else None
    
    # Create tester
    tester = CameraVisionTester(project_root)
    
    # Run tests
    tester.test_opencv_installation()
    frame = tester.test_camera_access(camera_id)
    vision = tester.test_model_loading(model_path)
    
    if vision is not None:
        tester.test_detection_speed(vision, camera_id)
        
        # Ask if user wants to run interactive tests
        response = input("\nRun live detection test? (y/n, default y): ").strip().lower()
        if response != 'n':
            tester.test_live_detection(vision, camera_id, duration=10)
        
        response = input("\nRun classification test? (y/n, default y): ").strip().lower()
        if response != 'n':
            tester.test_classification_accuracy(vision, camera_id)
    
    # Print summary
    tester.print_summary()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        print("Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)