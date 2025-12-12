#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
from collections import deque
import os

# Import vision module
try:
    from src.vision_module import VisionModule
except ImportError:
    from vision_module import VisionModule

# Import DOFBOT arm controller
ARM_AVAILABLE = False
try:
    from src.arm_controller import ArmController
    ARM_AVAILABLE = True
    print("Arm controller imported successfully")
except ImportError:
    try:
        from arm_controller import ArmController
        ARM_AVAILABLE = True
        print("Arm controller imported successfully")
    except ImportError:
        print("Arm controller not available - Running in simulation mode")
        ARM_AVAILABLE = False


class DOFBotUI:
    """Main UI for DOFBOT Robot Control."""

    def __init__(self, root):
        self.root = root
        self.root.title("DOFBOT Control - Detection & Manipulation")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)

        # Colors
        self.bg_dark = "#1a1a1a"
        self.bg_panel = "#2d2d2d"
        self.bg_accent = "#3d3d3d"
        self.text_color = "#ffffff"
        self.accent_green = "#00ff88"
        self.accent_red = "#ff4444"
        self.accent_blue = "#4488ff"
        self.accent_orange = "#ff8800"
        self.accent_purple = "#aa00ff"

        self.root.configure(bg=self.bg_dark)

        # Vision module
        self.vision = None
        self.vision_ready = False

        # Arm control
        self.arm = None
        self.arm_connected = False

        # Camera state
        self.cap = None
        self.camera_running = False
        self.current_frame = None
        self.detections = []

        # Performance
        self.fps = 0.0
        self.fps_history = deque(maxlen=30)
        self.prev_time = 0

        # UI State
        self.conf_threshold = 0.5
        self.performance_mode = "balanced"
        self.camera_id = 0

        # Statistics
        self.total_detections = 0
        self.total_sorted = 0
        self.is_sorting = False

        # Setup UI
        self.setup_ui()
        self.init_vision()
        self.init_arm()

        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        """Create the UI layout with fixed panels."""
        top_bar = tk.Frame(self.root, bg=self.bg_panel, height=60)
        top_bar.pack(side=tk.TOP, fill=tk.X)
        top_bar.pack_propagate(False)

        # Title
        title = tk.Label(top_bar, text="DOFBOT ROBOT CONTROL",
                        bg=self.bg_panel, fg=self.accent_green,
                        font=("Arial", 20, "bold"))
        title.pack(side=tk.LEFT, padx=20, pady=15)

        # Status indicators
        status_frame = tk.Frame(top_bar, bg=self.bg_panel)
        status_frame.pack(side=tk.RIGHT, padx=20, pady=10)

        self.vision_status = tk.Label(status_frame, text="Vision",
                                     bg=self.bg_panel, fg=self.accent_red,
                                     font=("Arial", 10, "bold"))
        self.vision_status.pack(side=tk.LEFT, padx=10)

        self.arm_status = tk.Label(status_frame, text="Arm",
                                  bg=self.bg_panel, fg=self.accent_red,
                                  font=("Arial", 10, "bold"))
        self.arm_status.pack(side=tk.LEFT, padx=10)

        self.fps_status = tk.Label(status_frame, text="FPS: 0",
                                  bg=self.bg_panel, fg=self.accent_blue,
                                  font=("Arial", 10, "bold"))
        self.fps_status.pack(side=tk.LEFT, padx=10)

        main_container = tk.Frame(self.root, bg=self.bg_dark)
        main_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LEFT: Video Feed
        self.setup_video_panel(main_container)

        # RIGHT: Controls
        self.setup_control_panel(main_container)

        bottom_bar = tk.Frame(self.root, bg=self.bg_panel, height=30)
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)
        bottom_bar.pack_propagate(False)

        self.status_label = tk.Label(bottom_bar, text="Ready",
                                    bg=self.bg_panel, fg=self.text_color,
                                    font=("Arial", 9), anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

    def setup_video_panel(self, parent):
        """Setup video display panel."""
        video_container = tk.Frame(parent, bg=self.bg_panel)
        video_container.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)

        # Video display area
        video_frame = tk.Frame(video_container, bg="#000000", width=640, height=480)
        video_frame.pack(padx=10, pady=10)
        video_frame.pack_propagate(False)

        self.video_canvas = tk.Label(video_frame,
                                     text="CAMERA STANDBY\n\nPress START to begin",
                                     bg="#000000", fg="#666666",
                                     font=("Arial", 14, "bold"))
        self.video_canvas.place(x=0, y=0, width=640, height=480)

        # Control buttons below video
        btn_frame = tk.Frame(video_container, bg=self.bg_panel)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        # Row 1: Camera controls
        row1 = tk.Frame(btn_frame, bg=self.bg_panel)
        row1.pack(fill=tk.X, pady=(0, 10))

        tk.Label(row1, text="Camera ID:", bg=self.bg_panel, fg=self.text_color,
                font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

        self.cam_entry = tk.Entry(row1, width=5, font=("Arial", 10))
        self.cam_entry.insert(0, "0")
        self.cam_entry.pack(side=tk.LEFT, padx=5)

        self.btn_start = tk.Button(row1, text="START", command=self.start_camera,
                                   bg="#00aa00", fg="white", font=("Arial", 10, "bold"),
                                   width=15, height=1, relief=tk.FLAT, cursor="hand2")
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = tk.Button(row1, text="STOP", command=self.stop_camera,
                                 bg="#aa0000", fg="white", font=("Arial", 10, "bold"),
                                 width=15, height=1, relief=tk.FLAT, cursor="hand2",
                                 state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.btn_snapshot = tk.Button(row1, text="SAVE", command=self.save_snapshot,
                                     bg="#666666", fg="white", font=("Arial", 10, "bold"),
                                     width=15, height=1, relief=tk.FLAT, cursor="hand2")
        self.btn_snapshot.pack(side=tk.LEFT, padx=5)

    def setup_control_panel(self, parent):
        """Setup control panel - COMPACT SIZE."""
        control_container = tk.Frame(parent, bg=self.bg_dark, width=330)
        control_container.pack(side=tk.RIGHT, fill=tk.Y)
        control_container.pack_propagate(False)

        # ========== DETECTION SETTINGS ==========
        det_frame = self.create_panel(control_container, "DETECTION SETTINGS", 150)

        # Confidence
        conf_row = tk.Frame(det_frame, bg=self.bg_accent)
        conf_row.pack(fill=tk.X, padx=5, pady=3)

        tk.Label(conf_row, text="Conf:", bg=self.bg_accent, fg=self.text_color,
                font=("Arial", 8)).pack(side=tk.LEFT)

        self.conf_scale = tk.Scale(conf_row, from_=0.1, to=1.0, resolution=0.05,
                                  orient=tk.HORIZONTAL, bg=self.bg_accent, fg=self.text_color,
                                  troughcolor=self.bg_dark, highlightthickness=0,
                                  length=150, command=self.on_conf_change)
        self.conf_scale.set(0.5)
        self.conf_scale.pack(side=tk.LEFT, padx=5)

        self.conf_label = tk.Label(conf_row, text="0.50", bg=self.bg_accent,
                                  fg=self.accent_green, font=("Arial", 9, "bold"),
                                  width=4)
        self.conf_label.pack(side=tk.LEFT)

        # Performance mode
        perf_row = tk.Frame(det_frame, bg=self.bg_accent)
        perf_row.pack(fill=tk.X, padx=5, pady=3)

        tk.Label(perf_row, text="Perf:", bg=self.bg_accent, fg=self.text_color,
                font=("Arial", 8)).pack(side=tk.LEFT)

        self.perf_var = tk.StringVar(value="balanced")
        perf_combo = ttk.Combobox(perf_row, textvariable=self.perf_var,
                                 values=["fast", "balanced", "accurate"],
                                 state="readonly", width=10, font=("Arial", 8))
        perf_combo.pack(side=tk.LEFT, padx=5)
        perf_combo.bind('<<ComboboxSelected>>', self.on_perf_change)

        # Stats
        stats_row = tk.Frame(det_frame, bg=self.bg_accent)
        stats_row.pack(fill=tk.X, padx=5, pady=5)

        self.det_count_label = tk.Label(stats_row, text="Det: 0",
                                       bg=self.bg_accent, fg=self.accent_orange,
                                       font=("Arial", 9, "bold"))
        self.det_count_label.pack(side=tk.LEFT, padx=5)

        self.sorted_label = tk.Label(stats_row, text="Sort: 0",
                                    bg=self.bg_accent, fg=self.accent_green,
                                    font=("Arial", 9, "bold"))
        self.sorted_label.pack(side=tk.LEFT, padx=5)

        list_frame = self.create_panel(control_container, "DETECTED OBJECTS", 500)

        # Instructions
        instruction_label = tk.Label(list_frame,
                                    text="Click selection to single sort or use Auto Sort",
                                    bg=self.bg_accent, fg=self.accent_orange,
                                    font=("Arial", 7, "italic"))
        instruction_label.pack(padx=5, pady=(0, 5))

        # Listbox with scrollbar
        list_container = tk.Frame(list_frame, bg=self.bg_accent)
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = tk.Scrollbar(list_container, bg=self.bg_accent)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.detection_listbox = tk.Listbox(list_container,
                                           bg=self.bg_dark, fg=self.text_color,
                                           font=("Courier New", 8),
                                           selectbackground=self.accent_blue,
                                           selectforeground="white",
                                           yscrollcommand=scrollbar.set,
                                           highlightthickness=0,
                                           relief=tk.FLAT)
        self.detection_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.detection_listbox.bind('<Double-Button-1>', self.on_detection_select)

        scrollbar.config(command=self.detection_listbox.yview)

        # Manual Sort button
        tk.Button(list_frame, text="SORT SELECTED ONE", command=self.sort_selected,
                 bg="#444444", fg="white", font=("Arial", 8),
                 width=30, height=1, relief=tk.FLAT, cursor="hand2").pack(
            padx=5, pady=(0, 5))

        # === NEW AUTO SORT BUTTON ===
        self.btn_auto_sort = tk.Button(list_frame, text="AUTO SORT (RAINBOW)", command=self.start_auto_sort,
                                      bg=self.accent_purple, fg="white", font=("Arial", 10, "bold"),
                                      width=30, height=2, relief=tk.RAISED, cursor="hand2")
        self.btn_auto_sort.pack(padx=5, pady=(5, 10))

    def create_panel(self, parent, title, height):
        """Create a titled panel with fixed height."""
        frame = tk.Frame(parent, bg=self.bg_panel, height=height)
        frame.pack(fill=tk.X, pady=(0, 10))
        frame.pack_propagate(False)

        # Title
        title_bar = tk.Frame(frame, bg=self.bg_dark, height=30)
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)

        tk.Label(title_bar, text=title, bg=self.bg_dark, fg=self.accent_green,
                font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=10, pady=5)

        # Content area
        content = tk.Frame(frame, bg=self.bg_accent)
        content.pack(fill=tk.BOTH, expand=True)

        return content

    # ==================== INITIALIZATION ====================

    def init_vision(self):
        """Initialize vision module."""
        try:
            self.log("Loading vision model...")
            self.vision = VisionModule()
            self.vision_ready = True
            self.vision_status.config(text="Vision", fg=self.accent_green)
            self.log("Vision module ready")
        except Exception as e:
            self.vision_ready = False
            self.log(f"Vision error: {e}")
            messagebox.showerror("Vision Error", f"Failed to load vision module:\n{e}")

    def init_arm(self):
        """Initialize and connect to arm automatically."""
        if not ARM_AVAILABLE:
            self.log("Arm controller not available")
            messagebox.showwarning("Warning", "Arm controller not available! Running without arm control.")
            return

        try:
            self.log("Connecting to DOFBOT arm...")
            self.arm = ArmController()

            if self.arm.connect():
                self.arm_connected = True
                self.arm_status.config(text="Arm", fg=self.accent_green)
                self.log("Arm connected successfully")
            else:
                raise Exception("Connection failed")

        except Exception as e:
            self.log(f"Arm connection failed: {e}")
            messagebox.showerror("Error", f"Failed to connect arm:\n{e}")

    # Camera Functions

    def start_camera(self):
        """Start camera capture."""
        if not self.vision_ready:
            messagebox.showerror("Error", "Vision module not ready!")
            return

        try:
            self.camera_id = int(self.cam_entry.get())
        except:
            messagebox.showerror("Error", "Invalid camera ID!")
            return

        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open camera {self.camera_id}")
            return

        # Set resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.camera_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.cam_entry.config(state=tk.DISABLED)

        self.log(f"Camera {self.camera_id} started")

        # Start video thread
        threading.Thread(target=self.video_loop, daemon=True).start()

    def stop_camera(self):
        """Stop camera capture."""
        self.camera_running = False

        if self.cap:
            self.cap.release()

        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.cam_entry.config(state=tk.NORMAL)

        self.video_canvas.config(image='', text="CAMERA STOPPED")
        self.log("Camera stopped")

    def video_loop(self):
        """Main video processing loop."""
        while self.camera_running:
            ret, frame = self.cap.read()

            if not ret:
                self.log("Failed to read frame")
                break

            # Store current frame for sorting logic
            self.current_frame = frame.copy()

            # Detect objects
            self.detections = self.vision.detect_objects(frame, self.conf_threshold)

            # Classify colors
            self.vision.classify_by_color(frame, self.detections)

            # Draw annotations
            annotated = self.vision.draw_detections(frame, self.detections, True)

            # Calculate FPS
            current_time = time.time()
            if self.prev_time > 0:
                self.fps = 1.0 / (current_time - self.prev_time)
                self.fps_history.append(self.fps)
            self.prev_time = current_time

            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

            # Resize to exactly 640x480
            display_frame = cv2.resize(annotated, (640, 480))

            # Convert to PhotoImage
            img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            # Update display (thread-safe)
            try:
                self.video_canvas.img = img_tk
                self.video_canvas.config(image=img_tk)
            except:
                pass

            # Update UI
            self.fps_status.config(text=f"FPS: {avg_fps:.1f}")
            self.det_count_label.config(text=f"Det: {len(self.detections)}")
            self.total_detections = len(self.detections)

            # Update detection list only if not sorting (to prevent flashing)
            if not self.is_sorting:
                self.update_detection_list()

            # Minimal sleep
            time.sleep(0.001)

    def save_snapshot(self):
        """Save current frame."""
        if self.current_frame is not None:
            os.makedirs("snapshots", exist_ok=True)
            filename = f"snapshots/snapshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.log(f"Saved: {filename}")
            messagebox.showinfo("Saved", f"Snapshot saved:\n{filename}")
        else:
            messagebox.showwarning("Warning", "No frame to save")

    def update_detection_list(self):
        """Update the detection listbox."""
        self.detection_listbox.delete(0, tk.END)

        # Sort for display only
        disp_dets = sorted(self.detections, key=lambda d: d['map_coord_cm'][1])

        for i, det in enumerate(disp_dets):
            u, v = det['map_coord_cm']
            text = f"{det['class_name'][:10]:10s} | {det['color'][:7]:7s} | ({u:4.1f}, {v:4.1f})"
            self.detection_listbox.insert(tk.END, text)

    # ARM FUNCTIONS

    def start_auto_sort(self):
        """Start autonomous rainbow sorting."""
        if not self.arm_connected:
            messagebox.showwarning("Warning", "Arm not connected!")
            return
            
        if not self.detections:
            messagebox.showwarning("Warning", "No objects detected!")
            return

        if self.is_sorting:
            messagebox.showinfo("Info", "Sorting already in progress")
            return

        if messagebox.askyesno("Auto Sort", "Start autonomous rainbow sorting?\nOrder: Red -> Orange -> Yellow -> Green -> Blue -> Indigo -> Violet"):
            self.is_sorting = True
            self.btn_auto_sort.config(state=tk.DISABLED, text="SORTING...", bg="#555555")
            threading.Thread(target=self.run_auto_sort, daemon=True).start()

    def run_auto_sort(self):
        """Execute the sorting logic in RAINBOW ORDER."""
        self.log("Starting Auto Sort (Rainbow Order)...")
        
        if self.current_frame is None:
            self.log("No frame available")
            self.finish_sorting()
            return

        # Snapshot of detections to process
        snapshot_detections = list(self.detections)
        
        # Group by color
        color_bins = {}
        for det in snapshot_detections:
            c = det.get('color', 'neutral')
            if c not in color_bins: color_bins[c] = []
            color_bins[c].append(det)

        # THE RAINBOW ORDER
        rainbow_order = ["red", "orange", "yellow", "green", 
                        "blue", "indigo", "violet", "neutral"]
        
        processed_count = 0

        # Iterate through colors in specific order
        for color in rainbow_order:
            if not self.camera_running: break

            if color in color_bins:
                objects = color_bins[color]
                
                # Sort objects by Y-coordinate (Closest first)
                objects.sort(key=lambda d: d['map_coord_cm'][1])
                
                self.log(f"Processing {color.upper()} ({len(objects)} items)...")
                
                for det in objects:
                    if not self.camera_running: break
                    
                    self.execute_sort(det)
                    processed_count += 1
                    time.sleep(0.5) # Pause between objects

        self.log(f"Auto Sort Complete! Sorted {processed_count} items.")
        self.finish_sorting()
        messagebox.showinfo("Complete", f"Sorting Finished!\nProcessed {processed_count} objects.")

    def finish_sorting(self):
        """Reset UI state after sorting."""
        self.is_sorting = False
        self.btn_auto_sort.config(state=tk.NORMAL, text="AUTO SORT (RAINBOW)", bg=self.accent_purple)
        # Move arm home
        if self.arm:
            self.arm.move_home()

    def sort_selected(self):
        """Sort selected detection (Single Item)."""
        if self.is_sorting: return

        selection = self.detection_listbox.curselection()

        if not selection:
            messagebox.showwarning("Warning", "Please select an object to sort")
            return

        if not self.arm_connected:
            messagebox.showwarning("Warning", "Arm not connected!")
            return

        # Get the sorted list that matches the listbox
        disp_dets = sorted(self.detections, key=lambda d: d['map_coord_cm'][1])
        
        idx = selection[0]
        if idx >= len(disp_dets):
            return

        det = disp_dets[idx]

        # Confirm
        result = messagebox.askyesno("Confirm Sort",
            f"Sort {det['class_name']} ({det['color']})?\n"
            f"Position: {det['map_coord_cm']}")

        if result:
            threading.Thread(target=self.execute_sort, args=(det,), daemon=True).start()

    def execute_sort(self, detection):
        """Execute pick and place operation."""
        try:
            self.log(f"Sorting {detection['class_name']} ({detection['color']})...")

            # Execute Arm Action
            success = self.arm.pick_and_place_detection(detection)

            if success:
                self.total_sorted += 1
                self.sorted_label.config(text=f"Sort: {self.total_sorted}")
                self.log(f"Sorted {detection['class_name']}")
            else:
                self.log(f"Sort failed for {detection['class_name']}")

        except Exception as e:
            self.log(f"Sort failed: {e}")
            print(f"ERROR: {e}")

    def on_detection_select(self, _event):
        """Handle double-click on detection."""
        self.sort_selected()

    # ==================== UI CALLBACKS ====================

    def on_conf_change(self, value):
        """Handle confidence threshold change."""
        self.conf_threshold = float(value)
        self.conf_label.config(text=f"{self.conf_threshold:.2f}")

    def on_perf_change(self, _event):
        """Handle performance mode change."""
        mode = self.perf_var.get()
        self.performance_mode = mode
        self.log(f"Performance mode: {mode.upper()}")

    def log(self, message):
        """Log message to status bar."""
        timestamp = time.strftime("%H:%M:%S")
        self.status_label.config(text=f"[{timestamp}] {message}")
        print(f"[{timestamp}] {message}")

    def on_closing(self):
        """Handle window closing."""
        if self.camera_running:
            self.stop_camera()

        if self.arm_connected and self.arm:
            self.arm.disconnect()

        self.root.destroy()


def main():
    """Main entry point."""
    root = tk.Tk()
    app = DOFBotUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()