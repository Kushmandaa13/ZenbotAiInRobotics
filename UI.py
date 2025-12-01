import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import time
import numpy as np
import os

try:
    from src.vision_module import VisionModule
except ImportError:
    try:
        from vision_module import VisionModule
    except ImportError:
        print("❌ Critical Error: Could not find vision_module.py")

class RobotGUI:
    def __init__(self, root, window_title="Zenbot Control Panel"):
        self.root = root
        self.root.title(window_title)
        self.root.geometry("1000x750")
        
        # Initialize Vision
        try:
            self.vision = VisionModule()
        except Exception as e:
            print(f"⚠️ Vision Init Error: {e}")
            self.vision = None

        self.cap = None
        self.running = False
        self.current_frame = None
        self.last_processed_frame = None
        
        # ==========================================================
        #  LAYOUT FIX: BUILD THE BOTTOM CONTROLS FIRST
        #  (This guarantees they stay on screen)
        # ==========================================================
        
        self.controls_frame = tk.Frame(root, bg="#e1e1e1", bd=1, relief=tk.RAISED)
        self.controls_frame.pack(side=tk.BOTTOM, fill=tk.X, ipady=10)

        # --- CENTER CONTAINER (To hold buttons in the middle) ---
        self.center_box = tk.Frame(self.controls_frame, bg="#e1e1e1")
        self.center_box.pack(expand=True)

        # Start Button
        self.btn_start = tk.Button(self.center_box, text="▶ START", 
                                   command=self.start_camera, 
                                   bg="#4CAF50", fg="white", font=("Segoe UI", 10, "bold"), 
                                   width=12, relief=tk.FLAT, cursor="hand2")
        self.btn_start.pack(side=tk.LEFT, padx=10)

        # Stop Button
        self.btn_stop = tk.Button(self.center_box, text="⏹ STOP", 
                                  command=self.stop_camera, 
                                  bg="#F44336", fg="white", font=("Segoe UI", 10, "bold"), 
                                  width=10, relief=tk.FLAT, cursor="hand2")
        self.btn_stop.pack(side=tk.LEFT, padx=10)

        # Snapshot Button
        self.btn_save = tk.Button(self.center_box, text="📸 SAVE", 
                                  command=self.save_frame, 
                                  bg="#2196F3", fg="white", font=("Segoe UI", 10, "bold"), 
                                  width=10, relief=tk.FLAT, cursor="hand2")
        self.btn_save.pack(side=tk.LEFT, padx=10)

        # Divider Line
        tk.Frame(self.center_box, width=2, bg="#cccccc").pack(side=tk.LEFT, fill=tk.Y, padx=15)

        # Mode Selector
        tk.Label(self.center_box, text="MODE:", bg="#e1e1e1", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.mode_var = tk.StringVar(value="Object Detection")
        self.mode_menu = ttk.Combobox(self.center_box, textvariable=self.mode_var, 
                                      values=["Object Detection", "Color Debug"], 
                                      state="readonly", width=17)
        self.mode_menu.pack(side=tk.LEFT, padx=5)

        # Debug Hint (Bottom Right)
        self.hint_label = tk.Label(self.controls_frame, text="Click video for Color Info", 
                                   bg="#e1e1e1", fg="#666", font=("Segoe UI", 8))
        self.hint_label.place(relx=0.98, rely=0.5, anchor="e")

        # ==========================================================
        #  2. THE VIDEO SCREEN (Build this LAST)
        #  (It takes whatever space is left)
        # ==========================================================
        self.video_frame = tk.Frame(root, bg="#1a1a1a")
        self.video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(self.video_frame, text="[ Camera Feed Standby ]\n\nPress START to begin", 
                                    bg="#1a1a1a", fg="#888888", font=("Segoe UI", 14))
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.video_label.bind('<Button-1>', self.get_pixel_color)


    def get_pixel_color(self, event):
        if self.current_frame is not None:
            label_w = self.video_label.winfo_width()
            label_h = self.video_label.winfo_height()
            img_h, img_w, _ = self.current_frame.shape
            
            x = int(event.x * (img_w / label_w))
            y = int(event.y * (img_h / label_h))
            
            if 0 <= x < img_w and 0 <= y < img_h:
                bgr = self.current_frame[y, x]
                hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                print(f"🎨 CLICKED ({x},{y}) -> HSV: {hsv}")
                self.hint_label.config(text=f"HSV: {hsv}", fg="blue")

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened(): self.cap = cv2.VideoCapture(1)
            
            if self.cap.isOpened():
                self.running = True
                self.video_label.config(text="")
                threading.Thread(target=self.video_loop, daemon=True).start()
            else:
                print("Error: Camera not found")

    def stop_camera(self):
        self.running = False
        if self.cap: self.cap.release()
        self.video_label.config(image='', text="[ Camera Feed Stopped ]")

    def save_frame(self):
        if self.last_processed_frame is not None:
            if not os.path.exists("snapshots"):
                os.makedirs("snapshots")
            filename = f"snapshots/zenbot_{int(time.time())}.jpg"
            cv2.imwrite(filename, self.last_processed_frame)
            print(f"💾 Saved: {filename}")
            self.hint_label.config(text="Snapshot Saved!", fg="green")

    def video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                annotated = frame.copy()

                try:
                    if self.mode_var.get() == "Object Detection" and self.vision:
                        detections = self.vision.detect_objects(frame)
                        self.vision.classify_by_color(frame, detections)
                        annotated = self.vision.draw_detections(frame, detections, show_colors=True)
                    elif self.mode_var.get() == "Color Debug":
                        h, w = frame.shape[:2]
                        cv2.circle(annotated, (w//2, h//2), 20, (0,255,255), 2)
                        hsv_center = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[h//2, w//2]
                        cv2.putText(annotated, f"Center HSV: {hsv_center}", (20, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                except Exception as e:
                    print(f"Error: {e}")

                self.last_processed_frame = annotated.copy()
                
                # Intelligent Resizing to prevent window expansion
                gui_h = self.video_label.winfo_height()
                if gui_h > 10:
                    ratio = gui_h / frame.shape[0]
                    gui_w = int(frame.shape[1] * ratio)
                    
                    # Ensure we don't resize if the window is collapsed
                    if gui_w > 10:
                        resized = cv2.resize(annotated, (gui_w, gui_h))
                        img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
                        self.video_label.imgtk = img
                        self.video_label.configure(image=img)
            time.sleep(0.01)

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotGUI(root)
    root.mainloop()