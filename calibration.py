#!/usr/bin/env python3
"""
DOFBOT Calibration Helper
=========================
Interactive tool to find and save correct servo positions for:
- Pickup zones on the board
- Tray positions for each color
- Gripper values for each object type

Run this ON your Raspberry Pi with the DOFBOT connected.

Usage:
    python3 calibration_helper.py

The tool will guide you through calibrating each position
and generate the correct values to paste into arm_controller.py
"""

import time
import sys

try:
    from Arm_Lib import Arm_Device
    ARM_AVAILABLE = True
except ImportError:
    ARM_AVAILABLE = False
    print("⚠️  Arm_Lib not found - Cannot calibrate without hardware")
    print("   This script must run on the Raspberry Pi with DOFBOT")


class CalibrationHelper:
    """Interactive DOFBOT calibration tool."""

    # Servo limits
    SERVO_MIN = 0
    SERVO_MAX = 180
    SERVO_CENTER = 90

    def __init__(self):
        self.arm = None
        self.current_position = [90, 90, 90, 90, 90, 180]  # S1-S6
        self.calibrated_positions = {
            'trays': {},
            'tray_hovers': {},
            'pickup_zones': {},
            'gripper_values': {}
        }

    def connect(self):
        """Connect to DOFBOT."""
        if not ARM_AVAILABLE:
            print("❌ Cannot connect - Arm_Lib not available")
            return False

        try:
            print("🔌 Connecting to DOFBOT...")
            self.arm = Arm_Device()
            time.sleep(0.5)
            
            # Move to home
            self.move_all(90, 90, 90, 90, 90, 180, 1000)
            time.sleep(1)
            
            print("✅ Connected!")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def move_all(self, s1, s2, s3, s4, s5, s6, time_ms=500):
        """Move all servos."""
        self.current_position = [s1, s2, s3, s4, s5, s6]
        if self.arm:
            self.arm.Arm_serial_servo_write6(s1, s2, s3, s4, s5, s6, time_ms)
            time.sleep(time_ms / 1000.0 + 0.1)

    def move_servo(self, servo_id, angle, time_ms=300):
        """Move single servo."""
        angle = max(self.SERVO_MIN, min(self.SERVO_MAX, angle))
        self.current_position[servo_id - 1] = angle
        if self.arm:
            self.arm.Arm_serial_servo_write(servo_id, angle, time_ms)
            time.sleep(time_ms / 1000.0 + 0.1)

    def home(self):
        """Go to home position."""
        self.move_all(90, 90, 90, 90, 90, 180, 800)

    def get_position_tuple(self):
        """Get current position as tuple string."""
        return f"({self.current_position[0]}, {self.current_position[1]}, {self.current_position[2]}, {self.current_position[3]}, {self.current_position[4]}, {self.current_position[5]})"

    def interactive_adjust(self, name: str) -> tuple:
        """
        Interactive servo adjustment.
        Returns the final position as tuple.
        """
        print(f"\n🎯 Adjusting position for: {name}")
        print("-" * 50)
        print("Controls:")
        print("  1-6    : Select servo")
        print("  +/=    : +5 degrees")
        print("  -      : -5 degrees")
        print("  ]/[    : ±1 degree (fine)")
        print("  w/s    : +10/-10 degrees (fast)")
        print("  o      : Open gripper")
        print("  c      : Close gripper")
        print("  h      : Home position")
        print("  p      : Print position")
        print("  ENTER  : Save and continue")
        print("  q      : Cancel")
        print("-" * 50)

        selected = 1

        while True:
            pos_str = self.get_position_tuple()
            print(f"\r📍 S{selected}: {self.current_position[selected-1]:3d}° | All: {pos_str}    ", end='')
            sys.stdout.flush()

            try:
                import termios
                import tty
                
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    ch = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except:
                ch = input("\nCommand: ").strip()
                if not ch:
                    ch = '\r'
                else:
                    ch = ch[0]

            if ch in '\r\n':
                print(f"\n✅ Saved: {name} = {self.get_position_tuple()}")
                return tuple(self.current_position)
            elif ch == 'q':
                print("\n❌ Cancelled")
                return None
            elif ch in '123456':
                selected = int(ch)
            elif ch in '+=':
                self.move_servo(selected, self.current_position[selected-1] + 5)
            elif ch == '-':
                self.move_servo(selected, self.current_position[selected-1] - 5)
            elif ch == ']':
                self.move_servo(selected, self.current_position[selected-1] + 1)
            elif ch == '[':
                self.move_servo(selected, self.current_position[selected-1] - 1)
            elif ch == 'w':
                self.move_servo(selected, self.current_position[selected-1] + 10)
            elif ch == 's':
                self.move_servo(selected, self.current_position[selected-1] - 10)
            elif ch == 'o':
                self.move_servo(6, 180)
            elif ch == 'c':
                self.move_servo(6, 60)
            elif ch == 'h':
                self.home()
            elif ch == 'p':
                print(f"\n{name} = {self.get_position_tuple()}")

    def calibrate_trays(self):
        """Calibrate all tray positions."""
        print("\n" + "=" * 60)
        print("🎨 TRAY POSITION CALIBRATION")
        print("=" * 60)
        print("\nYou'll calibrate 2 positions for each tray:")
        print("  1. HOVER: Position above the tray (for approach)")
        print("  2. DROP:  Position inside the tray (for dropping)")
        print("\nTray layout (as seen from above):")
        print("  ┌───────┬───────┬───────┬───────┐")
        print("  │  RED  │ORANGE │YELLOW │ GREEN │ ← Top row")
        print("  ├───────┼───────┼───────┼───────┤")
        print("  │ BLUE  │INDIGO │VIOLET │NEUTRAL│ ← Second row")
        print("  └───────┴───────┴───────┴───────┘")
        print("=" * 60)

        colors = [
            ("red", "Far left, top row"),
            ("orange", "Center-left, top row"),
            ("yellow", "Center-right, top row"),
            ("green", "Far right, top row"),
            ("blue", "Far left, second row"),
            ("indigo", "Center-left, second row"),
            ("violet", "Center-right, second row"),
            ("neutral", "Far right, second row"),
        ]

        input("\nPress ENTER to start calibration...")

        for color, description in colors:
            print(f"\n{'='*50}")
            print(f"🎨 {color.upper()} TRAY ({description})")
            print(f"{'='*50}")

            self.home()
            time.sleep(0.5)

            # Calibrate hover position
            print(f"\n1️⃣  First, position the gripper ABOVE the {color} tray")
            hover = self.interactive_adjust(f"{color}_hover")
            if hover:
                self.calibrated_positions['tray_hovers'][color] = hover

            # Calibrate drop position
            print(f"\n2️⃣  Now, lower the gripper INTO the {color} tray")
            drop = self.interactive_adjust(f"{color}_drop")
            if drop:
                self.calibrated_positions['trays'][color] = drop

        self.home()
        self.print_tray_calibration()

    def calibrate_pickup_zones(self):
        """Calibrate pickup zone positions."""
        print("\n" + "=" * 60)
        print("📍 PICKUP ZONE CALIBRATION")
        print("=" * 60)
        print("\nThe board is divided into 12 zones (3 rows × 4 columns):")
        print()
        print("  Zone layout:")
        print("  ┌────┬────┬────┬────┐")
        print("  │  9 │ 10 │ 11 │ 12 │ ← Far (trays area)")
        print("  ├────┼────┼────┼────┤")
        print("  │  5 │  6 │  7 │  8 │ ← Middle")
        print("  ├────┼────┼────┼────┤")
        print("  │  1 │  2 │  3 │  4 │ ← Close (near robot)")
        print("  └────┴────┴────┴────┘")
        print("    ↑    ↑    ↑    ↑")
        print("   Left      →    Right")
        print()
        print("For each zone, you'll calibrate:")
        print("  1. HOVER: Position above the zone")
        print("  2. PICKUP: Position at object height")
        print("=" * 60)

        # Define zones with their descriptions
        zones = [
            (1, "Bottom-left (close to robot, left side)"),
            (2, "Bottom center-left"),
            (3, "Bottom center-right"),
            (4, "Bottom-right (close to robot, right side)"),
            (5, "Middle-left"),
            (6, "Middle center-left (CENTER)"),
            (7, "Middle center-right"),
            (8, "Middle-right"),
            (9, "Top-left (far, near trays)"),
            (10, "Top center-left"),
            (11, "Top center-right"),
            (12, "Top-right (far, near trays)"),
        ]

        input("\nPress ENTER to start... (you can skip zones with 'q')")

        for zone_num, description in zones:
            print(f"\n{'='*50}")
            print(f"📍 ZONE {zone_num}: {description}")
            print(f"{'='*50}")

            self.home()
            self.move_servo(6, 180)  # Open gripper
            time.sleep(0.5)

            # Hover position
            print(f"\n1️⃣  Position gripper ABOVE zone {zone_num}")
            hover = self.interactive_adjust(f"zone_{zone_num}_hover")
            
            if hover is None:
                print(f"⏭️  Skipping zone {zone_num}")
                continue

            # Pickup position
            print(f"\n2️⃣  Lower gripper to PICKUP height in zone {zone_num}")
            pickup = self.interactive_adjust(f"zone_{zone_num}_pickup")

            if pickup is None:
                continue

            self.calibrated_positions['pickup_zones'][zone_num] = {
                'hover': hover,
                'pickup': pickup
            }

        self.home()
        self.print_zone_calibration()

    def calibrate_gripper(self):
        """Calibrate gripper values for different objects."""
        print("\n" + "=" * 60)
        print("✊ GRIPPER CALIBRATION")
        print("=" * 60)
        print("\nFor each object type, find the grip value that holds it securely.")
        print("Lower values = tighter grip")
        print()
        print("Objects to calibrate:")

        objects = [
            ("aa_battery", "Small cylindrical"),
            ("charger_adapter", "Rectangular, larger"),
            ("eraser", "Rectangular, medium"),
            ("glue_stick", "Cylindrical, medium"),
            ("highlighter", "Cylindrical, medium-large"),
            ("pen", "Thin cylindrical"),
            ("sharpener", "Small, irregular"),
            ("stapler", "Large, rectangular"),
        ]

        for obj, desc in objects:
            print(f"  • {obj}: {desc}")

        print("=" * 60)
        input("\nPlace objects nearby. Press ENTER to start...")

        for obj_name, description in objects:
            print(f"\n{'='*50}")
            print(f"✊ {obj_name.upper()} ({description})")
            print(f"{'='*50}")
            print(f"\nPlace a {obj_name} in the gripper")
            print("Controls: +/- to adjust, ENTER to save, q to skip")

            grip_value = 90  # Start at middle
            self.move_servo(6, 180)  # Open
            
            input("Press ENTER when object is in gripper...")
            
            while True:
                self.move_servo(6, grip_value)
                print(f"\rGrip value: {grip_value}  ", end='')
                
                try:
                    import termios, tty
                    fd = sys.stdin.fileno()
                    old = termios.tcgetattr(fd)
                    try:
                        tty.setraw(fd)
                        ch = sys.stdin.read(1)
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old)
                except:
                    ch = input("\n(+/-/enter/q): ")
                    if not ch:
                        ch = '\r'
                    else:
                        ch = ch[0]

                if ch in '\r\n':
                    print(f"\n✅ {obj_name} grip = {grip_value}")
                    self.calibrated_positions['gripper_values'][obj_name] = grip_value
                    break
                elif ch == 'q':
                    print("\n⏭️  Skipped")
                    break
                elif ch in '+=':
                    grip_value = min(180, grip_value + 5)
                elif ch == '-':
                    grip_value = max(0, grip_value - 5)

            self.move_servo(6, 180)  # Open

        self.print_gripper_calibration()

    def print_tray_calibration(self):
        """Print calibrated tray positions."""
        print("\n" + "=" * 60)
        print("📋 TRAY POSITIONS - Copy to arm_controller.py:")
        print("=" * 60)
        
        print("\nTRAY_POSITIONS = {")
        for color in ["red", "orange", "yellow", "green", "blue", "indigo", "violet", "neutral"]:
            if color in self.calibrated_positions['trays']:
                pos = self.calibrated_positions['trays'][color]
                print(f'    "{color}": {pos},')
            else:
                print(f'    # "{color}": NOT CALIBRATED,')
        print("}")

        print("\nTRAY_HOVER_POSITIONS = {")
        for color in ["red", "orange", "yellow", "green", "blue", "indigo", "violet", "neutral"]:
            if color in self.calibrated_positions['tray_hovers']:
                pos = self.calibrated_positions['tray_hovers'][color]
                print(f'    "{color}": {pos},')
            else:
                print(f'    # "{color}": NOT CALIBRATED,')
        print("}")
        print("=" * 60)

    def print_zone_calibration(self):
        """Print calibrated zone positions."""
        print("\n" + "=" * 60)
        print("📋 PICKUP ZONES - Copy to arm_controller.py:")
        print("=" * 60)
        
        print("\nPICKUP_ZONES = {")
        for zone_num in range(1, 13):
            if zone_num in self.calibrated_positions['pickup_zones']:
                data = self.calibrated_positions['pickup_zones'][zone_num]
                print(f"    {zone_num}: {{")
                print(f"        'hover': {data['hover']},")
                print(f"        'pickup': {data['pickup']},")
                print(f"    }},")
            else:
                print(f"    # {zone_num}: NOT CALIBRATED,")
        print("}")
        print("=" * 60)

    def print_gripper_calibration(self):
        """Print calibrated gripper values."""
        print("\n" + "=" * 60)
        print("📋 GRIPPER VALUES - Copy to arm_controller.py:")
        print("=" * 60)
        
        print("\nGRIPPER_BY_OBJECT = {")
        for obj in ["aa_battery", "charger_adapter", "eraser", "glue_stick", 
                    "highlighter", "pen", "sharpener", "stapler"]:
            if obj in self.calibrated_positions['gripper_values']:
                val = self.calibrated_positions['gripper_values'][obj]
                print(f'    "{obj}": {val},')
            else:
                print(f'    # "{obj}": NOT CALIBRATED,')
        print("}")
        print("=" * 60)

    def print_all_calibration(self):
        """Print all calibrated values."""
        self.print_tray_calibration()
        self.print_zone_calibration()
        self.print_gripper_calibration()

    def quick_test(self):
        """Quick movement test."""
        print("\n🧪 Quick movement test...")
        
        positions = [
            ("Home", (90, 90, 90, 90, 90, 180)),
            ("Forward reach", (90, 50, 50, 80, 90, 180)),
            ("Left", (130, 70, 70, 50, 90, 180)),
            ("Right", (50, 70, 70, 50, 90, 180)),
            ("Home", (90, 90, 90, 90, 90, 180)),
        ]

        for name, pos in positions:
            print(f"  → {name}: {pos}")
            self.move_all(*pos, 800)
            time.sleep(0.5)

        print("✅ Test complete")


def main():
    """Main calibration program."""
    print("\n" + "=" * 60)
    print("🔧 DOFBOT CALIBRATION HELPER")
    print("=" * 60)

    if not ARM_AVAILABLE:
        print("\n❌ This tool requires Arm_Lib")
        print("   Run this script on your Raspberry Pi with DOFBOT connected")
        return

    cal = CalibrationHelper()
    
    if not cal.connect():
        return

    while True:
        print("\n" + "-" * 40)
        print("CALIBRATION MENU:")
        print("  1. Quick movement test")
        print("  2. Calibrate TRAY positions")
        print("  3. Calibrate PICKUP zones")
        print("  4. Calibrate GRIPPER values")
        print("  5. Free adjustment mode")
        print("  6. Print all calibration")
        print("  7. Home position")
        print("  8. Exit")
        print("-" * 40)

        try:
            choice = input("Select (1-8): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == '1':
            cal.quick_test()
        elif choice == '2':
            cal.calibrate_trays()
        elif choice == '3':
            cal.calibrate_pickup_zones()
        elif choice == '4':
            cal.calibrate_gripper()
        elif choice == '5':
            cal.interactive_adjust("Free adjustment")
        elif choice == '6':
            cal.print_all_calibration()
        elif choice == '7':
            cal.home()
        elif choice == '8':
            break

    cal.home()
    print("\n👋 Calibration complete!")


if __name__ == "__main__":
    main()