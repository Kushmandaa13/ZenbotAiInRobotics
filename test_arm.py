import time
import sys

# Try to import the library
try:
    from Arm_Lib import Arm_Device
    ARM_LIBRARY_AVAILABLE = True
except ImportError:
    print("CRITICAL ERROR: 'Arm_Lib' not found.")
    ARM_LIBRARY_AVAILABLE = False

class DofbotTester:
    def __init__(self):
        self.arm = None
        self.results = {}

    def setup(self):
        print("\n" + "="*60)
        print("DOFBOT HARDWARE TEST SUITE")
        print("="*60)
        
        if not ARM_LIBRARY_AVAILABLE:
            return False

        try:
            self.arm = Arm_Device()
            time.sleep(0.1)
            print("PASS: Arm_Lib driver loaded successfully.")
            return True
        except Exception as e:
            print(f"FAIL: Could not connect to arm driver. Error: {e}")
            return False

    def test_buzzer(self):
        """Beep to prove we have control"""
        print("\n[TEST 1] Internal Communication")
        try:
            id_read = self.arm.Arm_serial_servo_read(1)
            if id_read == 1:
                print(f"PASS: Read Servo ID 1 successfully.")
                self.results['comm'] = True
                
                self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
                time.sleep(1)
            else:
                print(f"FAIL: Cannot read Servo ID 1. Check power.")
                self.results['comm'] = False
        except Exception as e:
            print(f"FAIL: Error reading servo: {e}")
            self.results['comm'] = False

    def test_joints(self):
        """Test each joint individually"""
        print("\n[TEST 2] Joint Movement (Dance)")
        if not self.results.get('comm'):
            print("SKIP: No connection.")
            return

        joints = [
            (1, "Base", 45, 135),
            (3, "Elbow", 45, 135),
            (4, "Wrist Pitch", 45, 135),
            (5, "Wrist Roll", 45, 135)
        ]

        try:
            # Reset
            self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
            time.sleep(1.2)

            for j_id, name, angle_a, angle_b in joints:
                print(f"  Testing ID {j_id} ({name})...", end="", flush=True)
                self.arm.Arm_serial_servo_write(j_id, angle_a, 500)
                time.sleep(0.6)
                self.arm.Arm_serial_servo_write(j_id, angle_b, 500)
                time.sleep(0.6)
                self.arm.Arm_serial_servo_write(j_id, 90, 500) # Return to center
                time.sleep(0.5)
                print(" OK")
            
            self.results['joints'] = True
        except Exception as e:
            print(f"\nFAIL: Joint test error: {e}")
            self.results['joints'] = False

    def test_gripper(self):
        print("\n[TEST 3] Gripper (ID 6)")
        try:
            print("  Clenching...", end="", flush=True)
            self.arm.Arm_serial_servo_write(6, 160, 500) # Close/Tight
            time.sleep(0.6)
            print(" Opening...", end="", flush=True)
            self.arm.Arm_serial_servo_write(6, 30, 500)  # Open
            time.sleep(0.6)
            print(" Relaxing...", end="", flush=True)
            self.arm.Arm_serial_servo_write(6, 90, 500)  # Neutral
            time.sleep(0.6)
            print(" OK")
            self.results['gripper'] = True
        except Exception as e:
            print(f"\nFAIL: Gripper error: {e}")
            self.results['gripper'] = False

    def print_summary(self):
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        all_passed = True
        for name, passed in self.results.items():
            status = "PASS" if passed else "FAIL"
            print(f"{name.upper():<10} : {status}")
            if not passed: all_passed = False
        
        print("-" * 60)
        if all_passed and len(self.results) > 0:
            print("SUCCESS: Your DOFBOT is fully operational!")
        else:
            print("WARNING: Some tests failed. Check power cable.")

if __name__ == "__main__":
    tester = DofbotTester()
    if tester.setup():
        tester.test_buzzer()
        tester.test_joints()
        tester.test_gripper()
        tester.print_summary()