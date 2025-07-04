import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from ultralytics import YOLO
from datetime import timedelta
import numpy as np
import math
import threading
from PIL import Image, ImageTk
import os

class TollBoothGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Highway Toll Booth Vehicle Detection")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)
        
        # Variables
        self.video_path = tk.StringVar()
        self.log_path = tk.StringVar()
        self.processing = False
        self.cap = None
        self.model = None
        
        # Processing parameters
        self.TARGET_WIDTH = 1280
        self.TARGET_HEIGHT = 720
        self.MIN_STATIONARY_DURATION_SECONDS = 5.0
        self.PIXEL_MOVEMENT_TOLERANCE = 75
        self.VEHICLE_CLASS_IDS = [2, 3, 5, 7]
        self.CONFIDENCE_THRESHOLD = 0.4
        self.VEHICLE_TYPE_MAP = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        self.FRONT_VEHICLE_HEURISTIC = "highest_y2"
        
        # Table formatting
        self.COLUMN_WIDTH = 30
        self.COLUMN_WIDTHS = {
            "id": self.COLUMN_WIDTH,
            "type": self.COLUMN_WIDTH,
            "from": self.COLUMN_WIDTH,
            "to": self.COLUMN_WIDTH,
            "duration": self.COLUMN_WIDTH,
        }
        self.TABLE_HEADERS = [
            "Vehicle ID",
            "Vehicle Type", 
            "Stationary From",
            "Stationary To",
            "Duration (s)",
        ]
        
        self.setup_gui()
        self.load_model()
        
    def setup_gui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Highway Toll Booth Vehicle Detection", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # Video file selection
        ttk.Label(file_frame, text="Video File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.video_entry = ttk.Entry(file_frame, textvariable=self.video_path, width=50)
        self.video_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(file_frame, text="Browse", 
                  command=self.browse_video_file).grid(row=0, column=2)
        
        # Log file selection
        ttk.Label(file_frame, text="Log File:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.log_entry = ttk.Entry(file_frame, textvariable=self.log_path, width=50)
        self.log_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(10, 0))
        ttk.Button(file_frame, text="Browse", 
                  command=self.browse_log_file).grid(row=1, column=2, pady=(10, 0))
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        # Process button
        self.process_btn = ttk.Button(control_frame, text="Start Processing", 
                                     command=self.start_processing)
        self.process_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Stop button
        self.stop_btn = ttk.Button(control_frame, text="Stop Processing", 
                                  command=self.stop_processing, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        # Progress bar
        self.progress_label = ttk.Label(control_frame, text="Ready")
        self.progress_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.progress_bar = ttk.Progressbar(control_frame, length=300, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT)
        
        # Main content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Video display frame
        video_frame = ttk.LabelFrame(content_frame, text="Video Processing", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        # Video canvas
        self.video_canvas = tk.Canvas(video_frame, bg="black", width=640, height=360)
        self.video_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video info
        self.video_info_label = ttk.Label(video_frame, text="No video loaded")
        self.video_info_label.grid(row=1, column=0, pady=(5, 0))
        
        # Results frame
        results_frame = ttk.LabelFrame(content_frame, text="Detection Results", padding="5")
        results_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Results treeview
        columns = ("ID", "Type", "From", "To", "Duration")
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=100)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Please select a video file and log location")
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def load_model(self):
        try:
            self.model = YOLO("yolov8n.pt")
            self.status_var.set("YOLO model loaded successfully")
        except Exception as e:
            self.status_var.set(f"Error loading YOLO model: {e}")
            messagebox.showerror("Model Error", f"Failed to load YOLO model: {e}")
    
    def browse_video_file(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.video_path.set(filename)
            self.status_var.set(f"Video selected: {os.path.basename(filename)}")
    
    def browse_log_file(self):
        filename = filedialog.asksaveasfilename(
            title="Save Log File As",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.log_path.set(filename)
            self.status_var.set(f"Log location: {os.path.basename(filename)}")
    
    def start_processing(self):
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file")
            return
        
        if not self.log_path.get():
            messagebox.showerror("Error", "Please select a log file location")
            return
        
        if not self.model:
            messagebox.showerror("Error", "YOLO model not loaded")
            return
        
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Update UI
        self.processing = True
        self.process_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.progress_bar['value'] = 0
        self.status_var.set("Starting video processing...")
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=self.process_video)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_processing(self):
        self.processing = False
        self.process_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set("Processing stopped by user")
        
        if self.cap:
            self.cap.release()
    
    def process_video(self):
        try:
            self.cap = cv2.VideoCapture(self.video_path.get())
            if not self.cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Error", "Could not open video file"))
                return
            
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                fps = 30.0
            
            original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Update video info
            self.root.after(0, lambda: self.video_info_label.configure(
                text=f"Original: {original_width}x{original_height} | Target: {self.TARGET_WIDTH}x{self.TARGET_HEIGHT} | FPS: {fps:.2f}"))
            
            # Calculate scaling factors
            width_scale = self.TARGET_WIDTH / original_width
            height_scale = self.TARGET_HEIGHT / original_height
            
            # Get scaled detection line coordinates
            LINE_P1, LINE_P2 = self.get_scaled_detection_line(original_width, original_height)
            
            # Scale movement tolerance proportionally
            scaled_movement_tolerance = int(self.PIXEL_MOVEMENT_TOLERANCE * min(width_scale, height_scale))
            
            # Processing variables
            stationary_log = []
            is_tracking_stationary_period = False
            current_stationary_start_time_sec_video = 0.0
            current_stationary_start_center = None
            current_stationary_vehicle_type = "Unknown"
            front_vehicle_event_id_counter = 0
            frame_idx = 0
            
            while self.processing:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Resize frame
                resized_frame = self.resize_frame(frame, self.TARGET_WIDTH, self.TARGET_HEIGHT)
                current_timestamp_sec_video = frame_idx / fps
                
                # YOLO detection
                results = self.model.predict(resized_frame, verbose=False, conf=self.CONFIDENCE_THRESHOLD)
                
                detected_vehicles_info = []
                if results and results[0].boxes:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    clss = results[0].boxes.cls.cpu().numpy()
                    for i in range(len(boxes)):
                        class_id = int(clss[i])
                        if class_id in self.VEHICLE_CLASS_IDS:
                            bbox = boxes[i]
                            center = self.calculate_bbox_center(bbox)
                            if self.is_point_before_line(center, LINE_P1, LINE_P2):
                                detected_vehicles_info.append((bbox, class_id))
                
                # Find front vehicle
                front_vehicle_info = None
                front_vehicle_bbox = None
                front_vehicle_center = None
                front_vehicle_class_id = None
                
                if detected_vehicles_info:
                    if self.FRONT_VEHICLE_HEURISTIC == "highest_y2":
                        detected_vehicles_info.sort(key=lambda item: item[0][3], reverse=True)
                        if detected_vehicles_info:
                            front_vehicle_info = detected_vehicles_info[0]
                    
                    if front_vehicle_info:
                        front_vehicle_bbox = front_vehicle_info[0]
                        front_vehicle_class_id = front_vehicle_info[1]
                        front_vehicle_center = self.calculate_bbox_center(front_vehicle_bbox)
                
                # Stationary tracking logic
                if front_vehicle_center:
                    if is_tracking_stationary_period:
                        distance_from_start = self.calculate_distance(
                            front_vehicle_center, current_stationary_start_center
                        )
                        if distance_from_start > scaled_movement_tolerance:
                            stationary_end_time_sec_video = (frame_idx - 1) / fps
                            duration = stationary_end_time_sec_video - current_stationary_start_time_sec_video
                            if duration >= self.MIN_STATIONARY_DURATION_SECONDS:
                                front_vehicle_event_id_counter += 1
                                event = {
                                    "vehicle_id": front_vehicle_event_id_counter,
                                    "vehicle_type": current_stationary_vehicle_type,
                                    "start_sec_video": current_stationary_start_time_sec_video,
                                    "end_sec_video": stationary_end_time_sec_video,
                                    "duration_sec": duration,
                                }
                                stationary_log.append(event)
                                # Add to GUI
                                self.root.after(0, lambda e=event: self.add_result_to_tree(e))
                            
                            is_tracking_stationary_period = True
                            current_stationary_start_time_sec_video = current_timestamp_sec_video
                            current_stationary_start_center = front_vehicle_center
                            current_stationary_vehicle_type = self.VEHICLE_TYPE_MAP.get(front_vehicle_class_id, "Unknown")
                    else:
                        is_tracking_stationary_period = True
                        current_stationary_start_time_sec_video = current_timestamp_sec_video
                        current_stationary_start_center = front_vehicle_center
                        current_stationary_vehicle_type = self.VEHICLE_TYPE_MAP.get(front_vehicle_class_id, "Unknown")
                else:
                    if is_tracking_stationary_period:
                        stationary_end_time_sec_video = (frame_idx - 1) / fps
                        duration = stationary_end_time_sec_video - current_stationary_start_time_sec_video
                        if duration >= self.MIN_STATIONARY_DURATION_SECONDS:
                            front_vehicle_event_id_counter += 1
                            event = {
                                "vehicle_id": front_vehicle_event_id_counter,
                                "vehicle_type": current_stationary_vehicle_type,
                                "start_sec_video": current_stationary_start_time_sec_video,
                                "end_sec_video": stationary_end_time_sec_video,
                                "duration_sec": duration,
                            }
                            stationary_log.append(event)
                            # Add to GUI
                            self.root.after(0, lambda e=event: self.add_result_to_tree(e))
                        
                        is_tracking_stationary_period = False
                        current_stationary_start_center = None
                        current_stationary_vehicle_type = "Unknown"
                
                # Create display frame
                display_frame = resized_frame.copy()
                
                # Draw detections
                for bbox_info_tuple in detected_vehicles_info:
                    x1, y1, x2, y2 = map(int, bbox_info_tuple[0])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                # Draw detection line
                cv2.line(display_frame, LINE_P1, LINE_P2, (255, 0, 255), 2)
                
                # Draw front vehicle
                if front_vehicle_bbox is not None:
                    x1, y1, x2, y2 = map(int, front_vehicle_bbox)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    if front_vehicle_center:
                        cv2.circle(display_frame, front_vehicle_center, 5, (255, 0, 0), -1)
                    type_to_display = self.VEHICLE_TYPE_MAP.get(front_vehicle_class_id, "Unknown")
                    cv2.putText(display_frame, type_to_display, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Draw stationary info
                if is_tracking_stationary_period and current_stationary_start_center:
                    display_stationary_duration = current_timestamp_sec_video - current_stationary_start_time_sec_video
                    cv2.circle(display_frame, current_stationary_start_center, 
                              scaled_movement_tolerance, (255, 255, 0), 1)
                    cv2.putText(display_frame, 
                               f"Stationary ({current_stationary_vehicle_type}): {display_stationary_duration:.1f}s",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Update GUI display
                self.root.after(0, lambda f=display_frame: self.update_video_display(f))
                
                # Update progress
                if total_frames > 0:
                    progress = (frame_idx / total_frames) * 100
                    self.root.after(0, lambda p=progress: self.update_progress(p, frame_idx, total_frames))
                
                frame_idx += 1
            
            # Handle final stationary period
            if is_tracking_stationary_period and current_stationary_start_center:
                end_timestamp_sec_video = (frame_idx - 1) / fps if frame_idx > 0 else current_stationary_start_time_sec_video
                duration = end_timestamp_sec_video - current_stationary_start_time_sec_video
                if duration >= self.MIN_STATIONARY_DURATION_SECONDS:
                    front_vehicle_event_id_counter += 1
                    event = {
                        "vehicle_id": front_vehicle_event_id_counter,
                        "vehicle_type": current_stationary_vehicle_type,
                        "start_sec_video": current_stationary_start_time_sec_video,
                        "end_sec_video": end_timestamp_sec_video,
                        "duration_sec": duration,
                    }
                    stationary_log.append(event)
                    self.root.after(0, lambda e=event: self.add_result_to_tree(e))
            
            # Save log file
            self.save_log_file(stationary_log)
            
            # Update UI
            self.root.after(0, lambda: self.processing_complete(len(stationary_log)))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Processing Error", f"An error occurred: {str(e)}"))
        finally:
            if self.cap:
                self.cap.release()
            self.processing = False
            self.root.after(0, self.reset_ui)
    
    def update_video_display(self, frame):
        # Resize frame for display (smaller than processing resolution)
        display_frame = cv2.resize(frame, (640, 360))
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        self.video_canvas.delete("all")
        self.video_canvas.create_image(320, 180, anchor=tk.CENTER, image=photo)
        self.video_canvas.image = photo  # Keep a reference
    
    def update_progress(self, progress, current_frame, total_frames):
        self.progress_bar['value'] = progress
        self.progress_label.configure(text=f"Frame {current_frame}/{total_frames} ({progress:.1f}%)")
        self.status_var.set(f"Processing... {progress:.1f}% complete")
    
    def add_result_to_tree(self, event):
        self.results_tree.insert("", "end", values=(
            event["vehicle_id"],
            event["vehicle_type"],
            self.format_time(event["start_sec_video"]),
            self.format_time(event["end_sec_video"]),
            f"{event['duration_sec']:.1f}s"
        ))
        
        # Scroll to the bottom
        children = self.results_tree.get_children()
        if children:
            self.results_tree.see(children[-1])
    
    def processing_complete(self, event_count):
        self.status_var.set(f"Processing complete! Found {event_count} stationary events")
        self.progress_bar['value'] = 100
        self.progress_label.configure(text="Complete")
        messagebox.showinfo("Processing Complete", 
                           f"Video processing completed successfully!\n\n"
                           f"Detected {event_count} stationary events.\n"
                           f"Results saved to: {self.log_path.get()}")
    
    def reset_ui(self):
        self.process_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.processing = False
    
    def save_log_file(self, stationary_log):
        try:
            col_keys_ordered_for_table = ["id", "type", "from", "to", "duration"]
            
            with open(self.log_path.get(), "w") as f:
                # Write the header
                f.write(self.create_table_line(self.COLUMN_WIDTHS, col_keys_ordered_for_table) + "\n")
                f.write(self.format_table_row(self.TABLE_HEADERS, self.COLUMN_WIDTHS, col_keys_ordered_for_table) + "\n")
                f.write(self.create_table_line(self.COLUMN_WIDTHS, col_keys_ordered_for_table) + "\n")
                
                # Write each stationary event
                for event in stationary_log:
                    f.write(self.format_table_row([
                        event["vehicle_id"],
                        event["vehicle_type"],
                        self.format_time(event["start_sec_video"]),
                        self.format_time(event["end_sec_video"]),
                        f"{event['duration_sec']:.1f}",
                    ], self.COLUMN_WIDTHS, col_keys_ordered_for_table) + "\n")
                
                f.write(self.create_table_line(self.COLUMN_WIDTHS, col_keys_ordered_for_table) + "\n")
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Save Error", f"Error saving log file: {e}"))
    
    # Utility functions
    def resize_frame(self, frame, target_width, target_height):
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    def calculate_bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    def calculate_distance(self, p1, p2):
        if p1 is None or p2 is None:
            return float("inf")
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def is_point_before_line(self, point, line_p1, line_p2):
        x, y = point
        x1, y1 = line_p1
        x2, y2 = line_p2
        return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1) < 0
    
    def get_scaled_detection_line(self, original_width, original_height):
        original_line_p1 = (374, 95)
        original_line_p2 = (1254, 398)
        
        assumed_original_width = 1920
        assumed_original_height = 1080
        
        x_scale = self.TARGET_WIDTH / assumed_original_width
        y_scale = self.TARGET_HEIGHT / assumed_original_height
        
        scaled_p1 = (int(original_line_p1[0] * x_scale), int(original_line_p1[1] * y_scale))
        scaled_p2 = (int(original_line_p2[0] * x_scale), int(original_line_p2[1] * y_scale))
        
        return scaled_p1, scaled_p2
    
    def format_time(self, total_seconds):
        if total_seconds < 0:
            total_seconds = 0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        secs = int(total_seconds % 60)
        milliseconds = int((total_seconds - int(total_seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"
    
    def create_table_line(self, widths_dict, header_keys_ordered):
        char = "+"
        parts = [char]
        for key in header_keys_ordered:
            parts.append("-" * widths_dict.get(key, self.COLUMN_WIDTH))
            parts.append(char)
        return "".join(parts)
    
    def format_table_row(self, data_tuple, widths_dict, col_keys_ordered):
        if len(data_tuple) != len(col_keys_ordered):
            return "| Error: Data mismatch |"
        parts = ["|"]
        for i, item in enumerate(data_tuple):
            item_str = str(item)
            width_key = col_keys_ordered[i]
            col_width = widths_dict.get(width_key, self.COLUMN_WIDTH)
            parts.append(f" {item_str:<{col_width - 2}} ")
            parts.append("|")
        return "".join(parts)

def main():
    root = tk.Tk()
    app = TollBoothGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
