#!/usr/bin/env python3
import time
from typing import Dict, Tuple, Optional, List

#import DOFBOT library
try:
    from Arm_Lib import Arm_Device
    ARM_LIB_AVAILABLE = True
except ImportError:
    ARM_LIB_AVAILABLE = False
    print("Arm_Lib not found - running in SIMULATION mode")


class ArmController:
    # BOARD CONFIGURATION
    
    MAP_WIDTH_CM = 28.7
    MAP_HEIGHT_CM = 30.5
    TRAY_SIZE_CM = 7.0

    RAINBOW_ORDER = [
        "red", "orange", "yellow", "green",
        "blue", "indigo", "violet", "neutral"
    ]
    # HARDCODED SERVO POSITIONS
    
    HOME = (90, 90, 90, 90, 90, 90)
    READY = (90, 70, 80, 50, 90, 60)  # Gripper open (60)
    SAFE_TRAVEL = (90, 80, 90, 45, 90, 90)

    # GRIPPER SETTINGS
    GRIPPER_OPEN = 60
    GRIPPER_CLOSED = 180

    GRIPPER_BY_OBJECT = {
        'aa_battery': 85,
        'charger_adapter': 115,
        'eraser': 155,
        'glue_stick': 158,
        'highlighter': 148,
        'pen': 180,
        'sharpener': 144,
        'stapler': 160
    }

    # TRAY POSITIONS 
    TRAY_POSITIONS = {
        "red": (108, 0, 80, 80, 90, 149),
        "orange": (96, 0, 81, 80, 90, 88),
        "yellow": (83, 0, 80, 84, 90, 135),
        "green": (70, 0, 83, 75, 90, 134),
        "blue": (112, 48, 25, 65, 90, 126),
        "indigo": (99, 30, 75, 20, 90, 150),
        "violet": (81, 30, 70, 30, 90, 143),
        "neutral": (66, 33, 60, 40, 90, 110),
    }

    TRAY_HOVER_POSITIONS = {
        "red": (108, 5, 80, 80, 90, 139),
        "orange": (96, 0, 81, 90, 90, 140),
        "yellow": (83, 0, 80, 90, 90, 180),
        "green": (70, 0, 78, 90, 90, 180),
        "blue": (112, 63, 0, 85, 90, 180),
        "indigo": (99, 35, 75, 20, 90, 180),
        "violet": (80, 35, 70, 30, 90, 180),
        "neutral": (66, 40, 60, 40, 90, 180),
    }

    # PICKUP ZONES 
    PICKUP_ZONES = {
        1: {
            'u_range': (0, 7),
            'v_range': (0, 8.1),
            'hover': (130, 91, 5, 15, 180, 105),
            'pickup': (130, 81, 5, 15, 180, 155),
        },
        2: {
            'u_range': (7, 14),
            'v_range': (0, 8.2),
            'hover': (100, 90, 10, 0, 115, 110),
            'pickup': (100, 75, 10, 0, 115, 150),
        },
        3: {
            'u_range': (14, 21),
            'v_range': (0, 8.3),
            'hover': (70, 85, 15, 0, 45, 115),
            'pickup': (65, 80, 15, 3, 45, 145),
        },
        4: {
            'u_range': (21, 28.7),
            'v_range': (0, 8.4),
            'hover': (40, 55, 45, 0, 10, 55),
            'pickup': (40, 55, 45, 0, 10, 115),
        },
        5: {
            'u_range': (0, 7),
            'v_range': (8.1, 16.3),
            'hover': (116, 36, 78, 1, 80, 40),
            'pickup': (116, 31, 78, 1, 80, 85),
        },
        6: {
            'u_range': (7, 14),
            'v_range': (8.2, 16.3),
            'hover': (94, 65, 29, 22, 150, 115),
            'pickup': (94, 65, 29, 22, 150, 180),
        },
        7: {
            'u_range': (14, 21),
            'v_range': (8.3, 16.4),
            'hover': (78, 70, 29, 22, 115, 105),
            'pickup': (78, 65, 29, 22, 115, 158),
        },
        8: {
            'u_range': (21, 28.7),
            'v_range': (8.3, 16.5),
            'hover': (48, 60, 39, 32, 180, 63),
            'pickup': (48, 60, 39, 32, 180, 160),
        },
    }

    # TIMING

    MOVE_TIME_FAST = 1000        
    MOVE_TIME_NORMAL = 1500      
    MOVE_TIME_SLOW = 2000        
    MOVE_TIME_VERY_SLOW = 2500  

    GRIPPER_TIME = 800           
    SETTLE_TIME = 0.5         

    # INITIALIZATION

    def __init__(self, port: str = None, simulation: bool = False):
        self.simulation = simulation or not ARM_LIB_AVAILABLE
        self.connected = False
        self.arm = None
        self.current_position = self.HOME
        
        # Statistics
        self.operations_count = 0
        self.successful_picks = 0
        self.failed_picks = 0

        print("=" * 60)
        print("DOFBOT Arm Controller (FIXED - Gripper Preserved)")
        print("=" * 60)
        
        if self.simulation:
            print("Running in SIMULATION mode")
        else:
            print("Arm_Lib available")

    def connect(self) -> bool:
        if self.simulation:
            print("[SIM] Connected to simulated arm")
            self.connected = True
            return True

        try:
            print("Connecting to DOFBOT...")
            self.arm = Arm_Device()
            time.sleep(0.5)
            
            self.move_to_position(self.HOME, time_ms=1000)
            time.sleep(1)
            
            self.connected = True
            print("Connected to DOFBOT successfully")
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            print("Switching to simulation mode")
            self.simulation = True
            self.connected = True
            return True

    def disconnect(self):
        if self.connected:
            print("Disconnecting...")
            try:
                self.move_home()
                time.sleep(0.5)
            except:
                pass
            self.connected = False
            print("Disconnected")

    # LOW-LEVEL MOVEMENT

    def move_to_position(self, position: Tuple[int, ...], 
                         time_ms: int = None) -> bool:
        """Move arm to specified servo position."""
        if time_ms is None:
            time_ms = self.MOVE_TIME_NORMAL

        s1, s2, s3, s4, s5, s6 = position

        # Clamp values
        s1 = max(0, min(180, s1))
        s2 = max(0, min(180, s2))
        s3 = max(0, min(180, s3))
        s4 = max(0, min(180, s4))
        s5 = max(0, min(180, s5))
        s6 = max(0, min(180, s6))

        if self.simulation:
            print(f"   [SIM] Move: ({s1}, {s2}, {s3}, {s4}, {s5}, {s6}) @ {time_ms}ms")
        else:
            try:
                self.arm.Arm_serial_servo_write6(s1, s2, s3, s4, s5, s6, time_ms)
            except Exception as e:
                print(f"Movement error: {e}")
                return False

        time.sleep(time_ms / 1000.0 + self.SETTLE_TIME)
        self.current_position = (s1, s2, s3, s4, s5, s6)
        return True

     # Move arm keeping current gripper value.
    def move_to_position_keep_gripper(self, position: Tuple[int, ...], 
                                       time_ms: int = None) -> bool:
   
        current_grip = self.current_position[5]
        s1, s2, s3, s4, s5, _ = position
        new_position = (s1, s2, s3, s4, s5, current_grip)
        return self.move_to_position(new_position, time_ms)

    def move_single_servo(self, servo_id: int, angle: int, 
                          time_ms: int = 500) -> bool:
        angle = max(0, min(180, angle))

        if self.simulation:
            print(f"   [SIM] Servo {servo_id} → {angle}° @ {time_ms}ms")
        else:
            try:
                self.arm.Arm_serial_servo_write(servo_id, angle, time_ms)
            except Exception as e:
                print(f"Servo {servo_id} error: {e}")
                return False

        time.sleep(time_ms / 1000.0 + 0.1)
        
        # Update internal position
        pos_list = list(self.current_position)
        pos_list[servo_id - 1] = angle
        self.current_position = tuple(pos_list)
        
        return True

    def move_home(self) -> bool:
        """Move arm to home position."""
        print(" Moving to HOME position...")
        return self.move_to_position(self.HOME, self.MOVE_TIME_NORMAL)

    def move_ready(self) -> bool:
        print(" Moving to READY position...")
        return self.move_to_position(self.READY, self.MOVE_TIME_NORMAL)

    # GRIPPER CONTROL

    def open_gripper(self, time_ms: int = None) -> bool:
        if time_ms is None:
            time_ms = self.GRIPPER_TIME
        print(f" Opening gripper (value: {self.GRIPPER_OPEN})...")
        return self.move_single_servo(6, self.GRIPPER_OPEN, time_ms)

    def close_gripper(self, grip_value: int = None, time_ms: int = None) -> bool:
        if grip_value is None:
            grip_value = self.GRIPPER_CLOSED
        if time_ms is None:
            time_ms = self.GRIPPER_TIME
        print(f" Closing gripper (value: {grip_value})...")
        return self.move_single_servo(6, grip_value, time_ms)

    def close_gripper_for_object(self, object_class: str) -> bool:
        grip_value = self.GRIPPER_BY_OBJECT.get(object_class, self.GRIPPER_CLOSED)
        print(f"Gripping {object_class} (grip: {grip_value})...")
        return self.move_single_servo(6, grip_value, self.GRIPPER_TIME)
    
    # ZONE-BASED PICKING

    def get_zone_for_coordinates(self, u_map: float, v_map: float) -> int:
        for zone_num, zone_data in self.PICKUP_ZONES.items():
            u_min, u_max = zone_data['u_range']
            v_min, v_max = zone_data['v_range']
            
            if u_min <= u_map <= u_max and v_min <= v_map <= v_max:
                return zone_num
            
        # Default fallback
        print(f"Coordinates ({u_map:.1f}, {v_map:.1f}) outside defined zones")
        return 6  

   # Get hover and pickup positions for coordinates
    def get_pickup_positions(self, u_map: float, v_map: float) -> Dict:
        
        zone = self.get_zone_for_coordinates(u_map, v_map)
        zone_data = self.PICKUP_ZONES[zone]
        
        print(f"Zone {zone} selected for ({u_map:.1f}, {v_map:.1f}) cm")
        
        return {
            'hover': zone_data['hover'],
            'pickup': zone_data['pickup'],
            'zone': zone
        }

    # PICK AND PLACE OPERATIONS

    def pick_object_at_coordinates(self, u_map: float, v_map: float,
                                   object_class: str = None) -> bool:
   
        print(f"\n PICK OBJECT at ({u_map:.1f}, {v_map:.1f}) cm")
        if object_class:
            print(f"   Object: {object_class}")

        try:
            # Get positions for this location
            positions = self.get_pickup_positions(u_map, v_map)
            hover_pos = positions['hover']
            pickup_pos = positions['pickup']

            # Open gripper first
            print("   [1/5] Opening gripper...")
            self.open_gripper()

            #  Move to hover position with gripper open
            print("   [2/5] Moving above object...")
            h1, h2, h3, h4, h5, _ = hover_pos
            hover_open = (h1, h2, h3, h4, h5, self.GRIPPER_OPEN)
            if not self.move_to_position(hover_open, self.MOVE_TIME_NORMAL):
                return False

            #  Lower to pickup position with gripper still open
            print("   [3/5] Lowering to object...")
            p1, p2, p3, p4, p5, _ = pickup_pos
            pickup_open = (p1, p2, p3, p4, p5, self.GRIPPER_OPEN)
            if not self.move_to_position(pickup_open, self.MOVE_TIME_SLOW):
                return False

            #Close gripper to grab object
            print("   [4/5] Grabbing object...")
            if object_class:
                self.close_gripper_for_object(object_class)
            else:
                self.close_gripper()
            
            # Extra delay to ensure grip is secure
            time.sleep(0.3)

            # Capture grip value AFTER closing
            current_grip = self.current_position[5]
            print(f"   ✓ Grip secured at value: {current_grip}")

            # Lift object with closed gripper
            print("   [5/5] Lifting object (keeping grip)...")
            hover_with_grip = (h1, h2, h3, h4, h5, current_grip)
            if not self.move_to_position(hover_with_grip, self.MOVE_TIME_NORMAL):
                return False

            print("Object picked successfully!")
            return True

        except Exception as e:
            print(f"Pick failed: {e}")
            return False

     # Get hover and pickup positions for coordinates
    def place_in_tray(self, color: str) -> bool:
        color = color.lower()
        if color in ['white', 'black', 'cream', 'maroon', 'gray', 'grey', 'brown']:
            print(f"  Mapping '{color}' to 'neutral' tray")
            color = 'neutral'

        if color not in self.TRAY_POSITIONS:
            print(f" Unknown color '{color}', using 'neutral' tray")
            color = 'neutral'

        print(f"\n PLACE in {color.upper()} tray")

        try:
            hover_pos = self.TRAY_HOVER_POSITIONS[color]
            tray_pos = self.TRAY_POSITIONS[color]
            
            # Keep current grip during movement 
            current_grip = self.current_position[5]
            print(f"  Maintaining grip at value: {current_grip}")

            # Move to safe travel position
            print("   [1/5] Moving to safe position...")
            s1, s2, s3, s4, s5, _ = self.SAFE_TRAVEL
            safe_with_grip = (s1, s2, s3, s4, s5, current_grip)
            if not self.move_to_position(safe_with_grip, self.MOVE_TIME_FAST):
                return False

            # Move above tray
            print(f"   [2/5] Moving above {color} tray...")
            t1, t2, t3, t4, t5, _ = hover_pos
            hover_with_grip = (t1, t2, t3, t4, t5, current_grip)
            if not self.move_to_position(hover_with_grip, self.MOVE_TIME_NORMAL):
                return False

            # Lower into tray 
            print("   [3/5] Lowering into tray...")
            d1, d2, d3, d4, d5, _ = tray_pos
            tray_with_grip = (d1, d2, d3, d4, d5, current_grip)
            if not self.move_to_position(tray_with_grip, self.MOVE_TIME_SLOW):
                return False

            # open gripper to release
            print("   [4/5] Releasing object...")
            self.open_gripper()

            # Small delay after release
            time.sleep(0.2)

            # Lift  
            print("   [5/5] Lifting away...")
            h1, h2, h3, h4, h5, _ = hover_pos
            hover_open = (h1, h2, h3, h4, h5, self.GRIPPER_OPEN)
            if not self.move_to_position(hover_open, self.MOVE_TIME_NORMAL):
                return False

            print(f"  Object placed in {color} tray!")
            return True

        except Exception as e:
            print(f"   Place failed: {e}")
            return False

    def pick_and_place(self, u_map: float, v_map: float, 
                       color: str, object_class: str = None) -> bool:
     #   complete pick-and-place operation
        self.operations_count += 1
        
        print("\n" + "=" * 60)
        print(f" PICK AND PLACE OPERATION #{self.operations_count}")
        print("=" * 60)
        print(f"   From: ({u_map:.1f}, {v_map:.1f}) cm")
        print(f"   To: {color} tray")
        if object_class:
            print(f"   Object: {object_class}")
        print("-" * 60)

        try:
            # Pick the object
            if not self.pick_object_at_coordinates(u_map, v_map, object_class):
                self.failed_picks += 1
                print("PICK FAILED")
                return False

            # Place in tray
            if not self.place_in_tray(color):
                self.failed_picks += 1
                print("PLACE FAILED")
                self.open_gripper()  # Drop object safely
                return False

            # Return to home position
            print("\n Returning to home position...")
            self.move_to_position(self.HOME, self.MOVE_TIME_NORMAL)

            self.successful_picks += 1
            print("\n" + "=" * 60)
            print(" PICK AND PLACE COMPLETE!")
            print("=" * 60)
            return True

        except Exception as e:
            self.failed_picks += 1
            print(f"\n Operation failed: {e}")
            self.emergency_stop()
            return False

    # VISION INTEGRATION

    def pick_and_place_detection(self, detection: Dict) -> bool:
        # Pick and place using detection from vision module
        u_map, v_map = detection['map_coord_cm']
        color = detection.get('color', 'neutral')
        object_class = detection.get('class_name', None)
        
        return self.pick_and_place(u_map, v_map, color, object_class)

    def sort_all_detections(self, detections: List[Dict]) -> Dict:
        """Sort all detected objects by color."""
        print("\n" + "=" * 60)
        print(f" SORTING {len(detections)} OBJECTS")
        print("=" * 60)

        sorted_detections = sorted(detections, 
                                   key=lambda d: d['map_coord_cm'][1])

        successful = 0
        failed = 0

        for i, det in enumerate(sorted_detections):
            print(f"\n--- Object {i+1}/{len(sorted_detections)} ---")
            
            if self.pick_and_place_detection(det):
                successful += 1
            else:
                failed += 1
            
            time.sleep(0.3)

        self.move_home()

        print("\n" + "=" * 60)
        print(" SORTING COMPLETE")
        print(f"    Successful: {successful}")
        print(f"    Failed: {failed}")
        print("=" * 60)

        return {'total': len(detections), 'successful': successful, 'failed': failed}

   #Sort objects in rainbow order
    def sort_by_rainbow_order(self, color_bins: Dict[str, List[Dict]]) -> Dict:
        print("\n" + "=" * 60)
        print(" RAINBOW SORTING")
        print("=" * 60)

        successful = 0
        failed = 0
        total = 0

        for color in self.RAINBOW_ORDER:
            if color not in color_bins or not color_bins[color]:
                continue

            print(f"\n{'='*40}")
            print(f"Processing {color.upper()} objects...")
            print(f"{'='*40}")

            objects = sorted(color_bins[color], 
                           key=lambda d: d['map_coord_cm'][1])

            for det in objects:
                total += 1
                if self.pick_and_place_detection(det):
                    successful += 1
                else:
                    failed += 1
                time.sleep(0.3)

        self.move_home()

        print("\n" + "=" * 60)
        print(" RAINBOW SORTING COMPLETE")
        print(f"    Successful: {successful}/{total}")
        print(f"   Failed: {failed}/{total}")
        print("=" * 60)

        return {'total': total, 'successful': successful, 'failed': failed}


    # Emergency Stop
    def emergency_stop(self):
        """Emergency stop - open gripper and return home."""
        print("\n EMERGENCY STOP!")
        try:
            self.open_gripper()
            time.sleep(0.3)
            self.move_home()
        except:
            pass

    
    # CALIBRATION & TESTING
    # Interactive calibration 
    def calibration_mode(self):
        print("\n" + "=" * 60)
        print("CALIBRATION MODE")
        print("=" * 60)
        print("\nControls:")
        print("  1-6: Select servo")
        print("  +/= : +5 degrees")
        print("  -   : -5 degrees")
        print("  [/] : ±1 degree (fine)")
        print("  h   : HOME")
        print("  r   : READY")
        print("  o   : Open gripper")
        print("  c   : Close gripper")
        print("  p   : Print position")
        print("  s   : Save note")
        print("  q   : Quit")
        print("=" * 60)

        current_servo = 1
        position = list(self.HOME)
        saved_positions = []

        self.move_to_position(tuple(position), 1000)

        while True:
            print(f"\n Position: ({position[0]}, {position[1]}, {position[2]}, "
                  f"{position[3]}, {position[4]}, {position[5]})")
            print(f"   Selected: S{current_servo} = {position[current_servo-1]}°")
            
            try:
                cmd = input("Command: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if cmd == 'q':
                break
            elif cmd == 'h':
                position = list(self.HOME)
                self.move_to_position(tuple(position), 800)
            elif cmd == 'r':
                position = list(self.READY)
                self.move_to_position(tuple(position), 800)
            elif cmd == 'o':
                self.open_gripper()
                position[5] = self.GRIPPER_OPEN
            elif cmd == 'c':
                self.close_gripper()
                position[5] = self.GRIPPER_CLOSED
            elif cmd == 'p':
                pos_str = f"({position[0]}, {position[1]}, {position[2]}, {position[3]}, {position[4]}, {position[5]})"
                print(f"\nCurrent position: {pos_str}")
            elif cmd == 's':
                note = input("Note: ")
                pos_str = f"({position[0]}, {position[1]}, {position[2]}, {position[3]}, {position[4]}, {position[5]})"
                saved_positions.append(f"{note}: {pos_str}")
                print(f" Saved!")
            elif cmd in ['1', '2', '3', '4', '5', '6']:
                current_servo = int(cmd)
            elif cmd in ['+', '=']:
                position[current_servo-1] = min(180, position[current_servo-1] + 5)
                self.move_single_servo(current_servo, position[current_servo-1], 300)
            elif cmd == '-':
                position[current_servo-1] = max(0, position[current_servo-1] - 5)
                self.move_single_servo(current_servo, position[current_servo-1], 300)
            elif cmd == ']':
                position[current_servo-1] = min(180, position[current_servo-1] + 1)
                self.move_single_servo(current_servo, position[current_servo-1], 200)
            elif cmd == '[':
                position[current_servo-1] = max(0, position[current_servo-1] - 1)
                self.move_single_servo(current_servo, position[current_servo-1], 200)

        if saved_positions:
            print("\n SAVED POSITIONS:")
            for pos in saved_positions:
                print(f"  {pos}")

        self.move_home()

    # Test movement to all tray positions
    def test_all_trays(self):
        print("\n TESTING ALL TRAYS")
        self.move_home()
        time.sleep(0.5)

        for color in self.RAINBOW_ORDER:
            print(f"\n {color.upper()} tray...")
            hover_pos = self.TRAY_HOVER_POSITIONS[color]
            tray_pos = self.TRAY_POSITIONS[color]
            
            self.move_to_position(hover_pos, self.MOVE_TIME_NORMAL)
            time.sleep(0.3)
            self.move_to_position(tray_pos, self.MOVE_TIME_SLOW)
            time.sleep(0.5)
            self.move_to_position(hover_pos, self.MOVE_TIME_NORMAL)
            time.sleep(0.3)

        self.move_home()
        print("\n Tray test complete!")

    def test_all_zones(self):
        """Test movement to all pickup zones."""
        print("\n TESTING ALL ZONES")
        self.move_home()
        self.open_gripper()
        time.sleep(0.5)

        for zone_num in sorted(self.PICKUP_ZONES.keys()):
            zone = self.PICKUP_ZONES[zone_num]
            print(f"\n  Zone {zone_num}...")
            
            # Move with gripper open
            h1, h2, h3, h4, h5, _ = zone['hover']
            self.move_to_position((h1, h2, h3, h4, h5, self.GRIPPER_OPEN), self.MOVE_TIME_NORMAL)
            time.sleep(0.3)
            
            p1, p2, p3, p4, p5, _ = zone['pickup']
            self.move_to_position((p1, p2, p3, p4, p5, self.GRIPPER_OPEN), self.MOVE_TIME_SLOW)
            time.sleep(0.3)
            
            self.close_gripper()
            time.sleep(0.3)
            
            self.move_to_position((h1, h2, h3, h4, h5, self.current_position[5]), self.MOVE_TIME_NORMAL)
            self.open_gripper()
            time.sleep(0.3)

        self.move_home()
        print("\n Zone test complete!")

   # Test a single pick and place
    def test_single_pick_place(self, zone: int = 6, color: str = "neutral"):
        print(f"\n Test: Zone {zone} → {color} tray")
        zone_data = self.PICKUP_ZONES[zone]
        u_center = (zone_data['u_range'][0] + zone_data['u_range'][1]) / 2
        v_center = (zone_data['v_range'][0] + zone_data['v_range'][1]) / 2
        return self.pick_and_place(u_center, v_center, color, "test_object")

    def print_statistics(self):

        print("\n STATISTICS")
        print(f"   Total: {self.operations_count}")
        print(f"   Success: {self.successful_picks}")
        print(f"   Failed: {self.failed_picks}")
        if self.operations_count > 0:
            rate = (self.successful_picks / self.operations_count) * 100
            print(f"   Rate: {rate:.1f}%")



# MAIN

if __name__ == "__main__":
    print("\n DOFBOT ARM CONTROLLER TEST")
    
    arm = ArmController(simulation=False)
    
    if not arm.connect():
        print(" Failed to connect")
        exit(1)

    while True:
        print("\n" + "-" * 40)
        print("MENU:")
        print("  1. Test all trays")
        print("  2. Test all zones")
        print("  3. Test single pick & place")
        print("  4. Calibration mode")
        print("  5. Home position")
        print("  6. Statistics")
        print("  7. Quit")
        print("-" * 40)

        try:
            choice = input("Select: ").strip()
        except:
            break

        if choice == '1':
            arm.test_all_trays()
        elif choice == '2':
            arm.test_all_zones()
        elif choice == '3':
            zone = int(input("Zone (1-8): ") or "6")
            color = input("Color: ") or "neutral"
            arm.test_single_pick_place(zone, color)
        elif choice == '4':
            arm.calibration_mode()
        elif choice == '5':
            arm.move_home()
        elif choice == '6':
            arm.print_statistics()
        elif choice == '7':
            break

    arm.disconnect()
    print("\n Goodbye!")
