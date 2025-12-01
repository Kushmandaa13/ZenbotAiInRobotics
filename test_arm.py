#!/usr/bin/env python3
"""
Raspberry Pi Robotic Arm Test Script
Tests DOFBOT arm connection and basic movements
"""

import sys
import os
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
    from arm_controller import ArmController
    ARM_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import arm_controller: {e}")
    ARM_MODULE_AVAILABLE = False


class ArmTester:
    """Test robotic arm capabilities on Raspberry Pi."""
    
    def __init__(self):
        self.test_results = {}
        self.arm = None
        
    def test_arm_import(self):
        """Test if arm controller module can be imported."""
        print("\n" + "=" * 70)
        print("TEST 1: ARM CONTROLLER MODULE")
        print("=" * 70)
        
        if ARM_MODULE_AVAILABLE:
            print("PASS: ArmController module imported successfully")
            self.test_results['module_import'] = True
            return True
        else:
            print("FAIL: Could not import ArmController module")
            self.test_results['module_import'] = False
            return False
    
    def test_serial_connection(self, port=None):
        """Test serial connection to arm."""
        print("\n" + "=" * 70)
        print("TEST 2: SERIAL CONNECTION")
        print("=" * 70)
        
        if not ARM_MODULE_AVAILABLE:
            print("SKIP: Module not available")
            return None
        
        try:
            # Try to import pyserial
            try:
                import serial
                print("PASS: pyserial is installed")
            except ImportError:
                print("WARNING: pyserial not installed")
                print("   Install with: pip install pyserial")
                print("   Running in simulation mode...")
            
            # Initialize arm
            print(f"\nInitializing arm controller...")
            if port:
                print(f"Using port: {port}")
            else:
                print("No port specified - will run in simulation mode")
            
            self.arm = ArmController(port=port)
            
            print("PASS: Arm controller initialized")
            self.test_results['initialization'] = True
            
            # Try to connect
            print("\nAttempting connection...")
            connected = self.arm.connect()
            
            if connected:
                if port:
                    print(f"PASS: Connected to arm on {port}")
                else:
                    print("PASS: Running in simulation mode")
                self.test_results['connection'] = True
            else:
                print("WARNING: Connection failed, running in simulation mode")
                self.test_results['connection'] = False
            
            return self.arm
            
        except Exception as e:
            print(f"FAIL: Error during connection: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['initialization'] = False
            self.test_results['connection'] = False
            return None
    
    def test_home_position(self):
        """Test moving to home position."""
        print("\n" + "=" * 70)
        print("TEST 3: HOME POSITION")
        print("=" * 70)
        
        if self.arm is None:
            print("SKIP: Arm not initialized")
            return
        
        try:
            print("Moving to home position...")
            success = self.arm.move_home()
            
            if success:
                print("PASS: Moved to home position successfully")
                self.test_results['home_position'] = True
            else:
                print("FAIL: Failed to move to home position")
                self.test_results['home_position'] = False
            
            time.sleep(1)
            
        except Exception as e:
            print(f"FAIL: Error during home position test: {e}")
            self.test_results['home_position'] = False
    
    def test_joint_movements(self):
        """Test individual joint movements."""
        print("\n" + "=" * 70)
        print("TEST 4: JOINT MOVEMENTS")
        print("=" * 70)
        
        if self.arm is None:
            print("SKIP: Arm not initialized")
            return
        
        try:
            # Test base rotation (J1)
            print("\n1. Testing base rotation (J1)...")
            print("   Rotating left...")
            self.arm.move_to_position({'j1': -30})
            time.sleep(1)
            
            print("   Rotating right...")
            self.arm.move_to_position({'j1': 30})
            time.sleep(1)
            
            print("   Returning to center...")
            self.arm.move_to_position({'j1': 0})
            time.sleep(1)
            
            # Test shoulder (J2)
            print("\n2. Testing shoulder (J2)...")
            print("   Moving up...")
            self.arm.move_to_position({'j2': 30})
            time.sleep(1)
            
            print("   Returning to neutral...")
            self.arm.move_to_position({'j2': 0})
            time.sleep(1)
            
            # Test elbow (J3)
            print("\n3. Testing elbow (J3)...")
            print("   Bending...")
            self.arm.move_to_position({'j3': 45})
            time.sleep(1)
            
            print("   Straightening...")
            self.arm.move_to_position({'j3': 0})
            time.sleep(1)
            
            print("\nPASS: All joint movements completed")
            self.test_results['joint_movements'] = True
            
        except Exception as e:
            print(f"FAIL: Error during joint movements: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['joint_movements'] = False
    
    def test_gripper(self):
        """Test gripper operations."""
        print("\n" + "=" * 70)
        print("TEST 5: GRIPPER OPERATIONS")
        print("=" * 70)
        
        if self.arm is None:
            print("SKIP: Arm not initialized")
            return
        
        try:
            print("\n1. Opening gripper...")
            self.arm.open_gripper()
            time.sleep(1)
            
            print("2. Closing gripper...")
            self.arm.close_gripper()
            time.sleep(1)
            
            print("3. Opening gripper...")
            self.arm.open_gripper()
            time.sleep(1)
            
            print("\nPASS: Gripper operations completed")
            self.test_results['gripper'] = True
            
        except Exception as e:
            print(f"FAIL: Error during gripper test: {e}")
            self.test_results['gripper'] = False
    
    def test_coordinate_conversion(self):
        """Test cartesian to angles conversion."""
        print("\n" + "=" * 70)
        print("TEST 6: COORDINATE CONVERSION")
        print("=" * 70)
        
        if self.arm is None:
            print("SKIP: Arm not initialized")
            return
        
        try:
            # Test positions
            test_positions = [
                (15, 0, 5, "Center front"),
                (15, -10, 5, "Right side"),
                (15, 10, 5, "Left side"),
                (20, 0, 10, "Front high"),
            ]
            
            print("\nTesting coordinate conversions:")
            for x, y, z, description in test_positions:
                angles = self.arm.cartesian_to_angles(x, y, z)
                print(f"\n   {description}: ({x}, {y}, {z}) cm")
                print(f"   -> Angles: J1={angles['j1']}°, J2={angles['j2']}°, J3={angles['j3']}°")
                
                # Validate angles are within limits
                valid = self.arm.validate_angles(angles)
                if valid:
                    print(f"   PASS: Angles within limits")
                else:
                    print(f"   WARNING: Angles outside safe limits")
            
            print("\nPASS: Coordinate conversion test completed")
            self.test_results['coordinate_conversion'] = True
            
        except Exception as e:
            print(f"FAIL: Error during coordinate conversion: {e}")
            self.test_results['coordinate_conversion'] = False
    
    def test_bin_positions(self):
        """Test bin position retrieval."""
        print("\n" + "=" * 70)
        print("TEST 7: BIN POSITIONS")
        print("=" * 70)
        
        if self.arm is None:
            print("SKIP: Arm not initialized")
            return
        
        try:
            colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'white']
            
            print("\nConfigured bin positions:")
            for color in colors:
                pos = self.arm.get_bin_position(color)
                print(f"   {color.capitalize():8s}: {pos}")
            
            print("\nPASS: All bin positions retrieved")
            self.test_results['bin_positions'] = True
            
        except Exception as e:
            print(f"FAIL: Error retrieving bin positions: {e}")
            self.test_results['bin_positions'] = False
    
    def test_pick_and_place(self):
        """Test a complete pick and place operation."""
        print("\n" + "=" * 70)
        print("TEST 8: PICK AND PLACE OPERATION")
        print("=" * 70)
        
        if self.arm is None:
            print("SKIP: Arm not initialized")
            return
        
        response = input("\nRun pick and place test? This will move the arm. (y/n): ").strip().lower()
        if response != 'y':
            print("SKIPPED by user")
            return
        
        try:
            pick_pos = (15, 0, 5)
            place_pos = self.arm.get_bin_position('red')
            
            print(f"\nPick position: {pick_pos}")
            print(f"Place position: {place_pos}")
            print("\nExecuting pick and place...")
            
            success = self.arm.pick_and_place(pick_pos, place_pos)
            
            if success:
                print("\nPASS: Pick and place completed successfully")
                self.test_results['pick_and_place'] = True
            else:
                print("\nFAIL: Pick and place failed")
                self.test_results['pick_and_place'] = False
            
        except Exception as e:
            print(f"FAIL: Error during pick and place: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['pick_and_place'] = False
    
    def test_emergency_stop(self):
        """Test emergency stop functionality."""
        print("\n" + "=" * 70)
        print("TEST 9: EMERGENCY STOP")
        print("=" * 70)
        
        if self.arm is None:
            print("SKIP: Arm not initialized")
            return
        
        try:
            print("\nTriggering emergency stop...")
            success = self.arm.emergency_stop()
            
            if success:
                print("PASS: Emergency stop executed")
                self.test_results['emergency_stop'] = True
            else:
                print("WARNING: Emergency stop may not have executed")
                self.test_results['emergency_stop'] = False
            
            time.sleep(1)
            
        except Exception as e:
            print(f"FAIL: Error during emergency stop: {e}")
            self.test_results['emergency_stop'] = False
    
    def interactive_mode(self):
        """Interactive control mode."""
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE")
        print("=" * 70)
        
        if self.arm is None:
            print("SKIP: Arm not initialized")
            return
        
        print("\nCommands:")
        print("  h - Move to home")
        print("  o - Open gripper")
        print("  c - Close gripper")
        print("  1-7 - Move to bin position (1=red, 2=orange, etc.)")
        print("  j - Test joint movement")
        print("  e - Emergency stop")
        print("  q - Quit interactive mode")
        print("=" * 70 + "\n")
        
        while True:
            cmd = input("Enter command: ").strip().lower()
            
            if cmd == 'q':
                print("Exiting interactive mode...")
                break
            
            elif cmd == 'h':
                print("Moving to home...")
                self.arm.move_home()
            
            elif cmd == 'o':
                print("Opening gripper...")
                self.arm.open_gripper()
            
            elif cmd == 'c':
                print("Closing gripper...")
                self.arm.close_gripper()
            
            elif cmd in '1234567':
                colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
                color = colors[int(cmd) - 1]
                bin_pos = self.arm.get_bin_position(color)
                print(f"Moving to {color} bin at {bin_pos}...")
                angles = self.arm.cartesian_to_angles(*bin_pos)
                self.arm.move_to_position(angles)
            
            elif cmd == 'j':
                joint = input("  Enter joint (j1-j6): ").strip().lower()
                angle = input("  Enter angle: ").strip()
                try:
                    angle = int(angle)
                    print(f"Moving {joint} to {angle}°...")
                    self.arm.move_to_position({joint: angle})
                except ValueError:
                    print("Invalid angle")
            
            elif cmd == 'e':
                print("EMERGENCY STOP!")
                self.arm.emergency_stop()
            
            else:
                print("Unknown command")
    
    def cleanup(self):
        """Cleanup and return to home."""
        if self.arm and self.arm.connected:
            print("\nReturning to home position...")
            self.arm.move_home()
            time.sleep(1)
            self.arm.disconnect()
    
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
        print("=" * 70)
        
        if passed == total and total > 0:
            print("\nALL TESTS PASSED!")
            print("Your robotic arm is working correctly!")
        elif total > 0:
            print("\nSome tests failed. Please review the errors above.")
        
        print()


def main():
    """Run all arm tests."""
    print("\n" + "=" * 70)
    print("RASPBERRY PI ROBOTIC ARM TEST SUITE")
    print("=" * 70)
    print("This will test:")
    print("  1. Module import")
    print("  2. Serial connection")
    print("  3. Home position")
    print("  4. Joint movements")
    print("  5. Gripper operations")
    print("  6. Coordinate conversion")
    print("  7. Bin positions")
    print("  8. Pick and place (optional)")
    print("  9. Emergency stop")
    print("=" * 70)
    
    # Ask for serial port
    print("\nIMPORTANT: If you don't have the DOFBOT arm connected,")
    print("   just press Enter to run in simulation mode.")
    port_input = input("\nEnter serial port (e.g., /dev/ttyUSB0 or COM3, or Enter for simulation): ").strip()
    port = port_input if port_input else None
    
    # Create tester
    tester = ArmTester()
    
    try:
        # Run tests
        if not tester.test_arm_import():
            print("\nCannot proceed without arm controller module")
            return
        
        arm = tester.test_serial_connection(port)
        
        if arm is not None:
            tester.test_home_position()
            
            response = input("\nRun movement tests? (y/n, default y): ").strip().lower()
            if response != 'n':
                tester.test_joint_movements()
                tester.test_gripper()
            
            tester.test_coordinate_conversion()
            tester.test_bin_positions()
            tester.test_pick_and_place()
            tester.test_emergency_stop()
            
            # Interactive mode
            response = input("\nEnter interactive mode? (y/n): ").strip().lower()
            if response == 'y':
                tester.interactive_mode()
        
        # Print summary
        tester.print_summary()
        
    finally:
        # Cleanup
        tester.cleanup()


if __name__ == "__main__":
    try:
        main()
        print("Test complete!\n")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        print("Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)