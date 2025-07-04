import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
from video_processor import VideoProcessor
from utils import format_time
from config import *


class TollBoothGUI:
    """Main GUI application for the toll booth vehicle detection system."""
    
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.resizable(True, True)
        
        # Variables
        self.video_path = tk.StringVar()
        self.log_path = tk.StringVar()
        self.video_start_time = tk.StringVar()
        self.status_var = tk.StringVar()
        
        # Set default start time
        self.video_start_time.set(DEFAULT_VIDEO_START_TIME)
        
        # Video processor
        self.video_processor = VideoProcessor(gui_callback=self.handle_processor_callback)
        
        # Setup GUI
        self.setup_gui()
        self.check_model_status()
        self.ensure_logs_directory()
        
    def setup_gui(self):
        """Setup the main GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Create GUI sections
        self._create_header(main_frame)
        self._create_file_selection(main_frame)
        self._create_controls(main_frame)
        self._create_main_content(main_frame)
        self._create_status_bar(main_frame)
        
    def _create_header(self, parent):
        """Create the header section."""
        title_label = ttk.Label(parent, text="Highway Toll Booth Vehicle Detection", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
    def _create_file_selection(self, parent):
        """Create the file selection section."""
        file_frame = ttk.LabelFrame(parent, text="File Selection", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # Video file selection
        ttk.Label(file_frame, text="Video File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.video_entry = ttk.Entry(file_frame, textvariable=self.video_path, width=50)
        self.video_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(file_frame, text="Browse", 
                  command=self.browse_video_file).grid(row=0, column=2)
        
        # Video start time selection
        ttk.Label(file_frame, text="Video Start Time:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        start_time_frame = ttk.Frame(file_frame)
        start_time_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(10, 0))
        
        self.start_time_entry = ttk.Entry(start_time_frame, textvariable=self.video_start_time, width=15)
        self.start_time_entry.grid(row=0, column=0, sticky=tk.W)
        
        ttk.Label(start_time_frame, text="(HH:MM:SS format - when the video recording actually started)", 
                 font=("Arial", 8)).grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Log file selection
        ttk.Label(file_frame, text="Log File:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.log_entry = ttk.Entry(file_frame, textvariable=self.log_path, width=50)
        self.log_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(10, 0))
        ttk.Button(file_frame, text="Browse", 
                  command=self.browse_log_file).grid(row=2, column=2, pady=(10, 0))
        
    def _create_controls(self, parent):
        """Create the control buttons and progress section."""
        control_frame = ttk.Frame(parent)
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
        
    def _create_main_content(self, parent):
        """Create the main content area with video display and results."""
        content_frame = ttk.Frame(parent)
        content_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Video display
        self._create_video_display(content_frame)
        
        # Results display
        self._create_results_display(content_frame)
        
    def _create_video_display(self, parent):
        """Create the video display section."""
        video_frame = ttk.LabelFrame(parent, text="Video Processing", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        # Video canvas
        self.video_canvas = tk.Canvas(video_frame, bg="black", 
                                     width=VIDEO_DISPLAY_SIZE[0], 
                                     height=VIDEO_DISPLAY_SIZE[1])
        self.video_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add default text to video canvas
        self.show_default_video_message()
        
        # Video info
        self.video_info_label = ttk.Label(video_frame, text="No video loaded")
        self.video_info_label.grid(row=1, column=0, pady=(5, 0))
        
    def _create_results_display(self, parent):
        """Create the results display section."""
        results_frame = ttk.LabelFrame(parent, text="Detection Results", padding="5")
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
        
    def _create_status_bar(self, parent):
        """Create the status bar."""
        self.status_var.set("Ready - Please select a video file (log will be auto-generated in logs folder)")
        self.status_bar = ttk.Label(parent, textvariable=self.status_var, 
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def check_model_status(self):
        """Check and display model loading status."""
        if self.video_processor.is_model_loaded():
            self.status_var.set("YOLO model loaded successfully")
        else:
            self.status_var.set("Error: YOLO model not loaded")
            messagebox.showerror("Model Error", "Failed to load YOLO model")
    
    def browse_video_file(self):
        """Browse for video file."""
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
            self.set_default_log_filename(filename)
            self.load_first_frame(filename)
    
    def browse_log_file(self):
        """Browse for log file location."""
        # Default to logs folder
        initial_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        
        # Get current log path for initial filename
        current_log = self.log_path.get()
        initial_filename = os.path.basename(current_log) if current_log else "detection_log.txt"
        
        filename = filedialog.asksaveasfilename(
            title="Save Log File As",
            initialdir=initial_dir,
            initialfile=initial_filename,
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.log_path.set(filename)
            self.status_var.set(f"Log location: {os.path.basename(filename)}")
    
    def set_default_log_filename(self, video_path):
        """Set default log filename based on the selected video."""
        try:
            # Get video filename without extension
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            # Generate default log filename
            log_filename = f"{video_name}_detection_log.txt"
            default_log_path = os.path.join(logs_dir, log_filename)
            
            # Set the log path
            self.log_path.set(default_log_path)
            self.status_var.set(f"Video selected: {os.path.basename(video_path)} | Default log: {log_filename}")
            
        except Exception as e:
            self.status_var.set(f"Video selected: {os.path.basename(video_path)} | Error setting default log: {str(e)}")
    
    def start_processing(self):
        """Start video processing."""
        # Validation
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file")
            return
        
        if not self.log_path.get():
            # Auto-generate log path if not set
            if self.video_path.get():
                self.set_default_log_filename(self.video_path.get())
            else:
                messagebox.showerror("Error", "Please select a video file first")
                return
        
        # Validate start time format
        start_time_str = self.video_start_time.get().strip()
        if not self.validate_time_format(start_time_str):
            messagebox.showerror("Error", 
                               "Invalid start time format. Please use HH:MM:SS format (e.g., 14:30:25)")
            return
        
        if not self.video_processor.is_model_loaded():
            messagebox.showerror("Error", "YOLO model not loaded")
            return
        
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Clear preview frame and show processing message
        self.video_canvas.delete("all")
        self.video_canvas.create_text(
            VIDEO_DISPLAY_SIZE[0]//2, VIDEO_DISPLAY_SIZE[1]//2,
            text="Processing...", 
            fill="yellow", 
            font=("Arial", 14, "bold")
        )
        
        # Update UI
        self.process_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.progress_bar['value'] = 0
        self.status_var.set("Starting video processing...")
        
        # Start processing
        self.video_processor.start_processing(
            self.video_path.get(), 
            self.log_path.get(), 
            self.video_start_time.get().strip()
        )
    
    def stop_processing(self):
        """Stop video processing."""
        self.video_processor.stop_processing()
        self.reset_ui()
        self.status_var.set("Processing stopped by user")
        
        # Restore preview if video is selected
        if self.video_path.get():
            self.load_first_frame(self.video_path.get())
    
    def reset_ui(self):
        """Reset UI to initial state."""
        self.process_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
    
    def handle_processor_callback(self, event_type, data):
        """Handle callbacks from the video processor."""
        if event_type == "error":
            self.root.after(0, lambda: messagebox.showerror("Error", data))
        elif event_type == "video_info":
            self.root.after(0, lambda: self.update_video_info(data))
        elif event_type == "frame_update":
            self.root.after(0, lambda: self.update_video_display(data))
        elif event_type == "progress_update":
            self.root.after(0, lambda: self.update_progress(data))
        elif event_type == "new_event":
            self.root.after(0, lambda: self.add_result_to_tree(data))
        elif event_type == "processing_complete":
            self.root.after(0, lambda: self.processing_complete(data))
        elif event_type == "processing_finished":
            self.root.after(0, self.reset_ui)
    
    def update_video_info(self, info):
        """Update video information display."""
        text = (f"Processing: {info['original']} → {info['target']} | "
                f"FPS: {info['fps']:.2f} | Tolerance: {info['tolerance']}px | "
                f"Start Time: {self.video_start_time.get()}")
        self.video_info_label.configure(text=text)
    
    def update_video_display(self, frame):
        """Update the video display with new frame."""
        # Resize frame for display
        display_frame = cv2.resize(frame, VIDEO_DISPLAY_SIZE)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and then PhotoImage
        pil_image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        self.video_canvas.delete("all")
        self.video_canvas.create_image(
            VIDEO_DISPLAY_SIZE[0]//2, VIDEO_DISPLAY_SIZE[1]//2, 
            anchor=tk.CENTER, image=photo
        )
        self.video_canvas.image = photo  # Keep a reference
    
    def update_progress(self, progress_data):
        """Update progress bar and labels."""
        progress = progress_data["progress"]
        current_frame = progress_data["current_frame"]
        total_frames = progress_data["total_frames"]
        
        self.progress_bar['value'] = progress
        self.progress_label.configure(
            text=f"Frame {current_frame}/{total_frames} ({progress:.1f}%)"
        )
        self.status_var.set(f"Processing... {progress:.1f}% complete")
    
    def add_result_to_tree(self, event):
        """Add a new detection event to the results tree."""
        # Use real time if available, otherwise fall back to video time
        start_time = event.get("real_start_time", format_time(event["start_sec_video"]))
        end_time = event.get("real_end_time", format_time(event["end_sec_video"]))
        
        self.results_tree.insert("", "end", values=(
            event["vehicle_id"],
            event["vehicle_type"],
            start_time,
            end_time,
            f"{event['duration_sec']:.1f}s"
        ))
        
        # Scroll to the bottom
        children = self.results_tree.get_children()
        if children:
            self.results_tree.see(children[-1])
    
    def processing_complete(self, event_count):
        """Handle processing completion."""
        self.status_var.set(f"Processing complete! Found {event_count} stationary events")
        self.progress_bar['value'] = 100
        self.progress_label.configure(text="Complete")
        
        # Show completion message in video canvas temporarily
        self.video_canvas.delete("all")
        self.video_canvas.create_text(
            VIDEO_DISPLAY_SIZE[0]//2, VIDEO_DISPLAY_SIZE[1]//2 - 20,
            text="Processing Complete!", 
            fill="green", 
            font=("Arial", 14, "bold")
        )
        self.video_canvas.create_text(
            VIDEO_DISPLAY_SIZE[0]//2, VIDEO_DISPLAY_SIZE[1]//2 + 10,
            text=f"Found {event_count} stationary events", 
            fill="white", 
            font=("Arial", 10)
        )
        
        # Restore preview after 3 seconds
        if self.video_path.get():
            self.root.after(3000, lambda: self.load_first_frame(self.video_path.get()))
        
        # Show completion dialog
        result = messagebox.askquestion(
            "Processing Complete", 
            f"Video processing completed successfully!\n\n"
            f"Detected {event_count} stationary events.\n"
            f"Results saved to: {os.path.basename(self.log_path.get())}\n\n"
            f"Would you like to open the logs folder?",
            icon='question'
        )
        
        # Open logs folder if user wants to
        if result == 'yes':
            self.open_logs_folder()
    
    def load_first_frame(self, video_path):
        """Load and display the first frame of the selected video with detection line."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.status_var.set("Error: Could not open video file")
                return
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                fps = 30.0
            
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Read first frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                self.status_var.set("Error: Could not read first frame")
                return
            
            # Resize frame to target resolution for preview
            resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
            
            # Get detection line for the preview
            line_p1, line_p2 = self.video_processor.detection_engine.get_scaled_detection_line(
                TARGET_WIDTH, TARGET_HEIGHT
            )
            
            # Draw detection line on preview
            preview_frame = resized_frame.copy()
            cv2.line(preview_frame, line_p1, line_p2, (255, 0, 255), 2)
            
            # Add preview text
            cv2.putText(preview_frame, "PREVIEW - Detection Line Shown", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(preview_frame, "Click 'Start Processing' to begin", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the preview frame
            self.update_video_display(preview_frame)
            
            # Update video info
            self.video_info_label.configure(
                text=f"Preview: {original_width}x{original_height} → {TARGET_WIDTH}x{TARGET_HEIGHT} | "
                     f"FPS: {fps:.2f} | Frames: {total_frames} | Start Time: {self.video_start_time.get()} | Ready to process"
            )
            
        except Exception as e:
            self.status_var.set(f"Error loading video preview: {str(e)}")
    
    def show_default_video_message(self):
        """Show default message in video canvas when no video is loaded."""
        self.video_canvas.delete("all")
        # Add text to indicate no video loaded
        self.video_canvas.create_text(
            VIDEO_DISPLAY_SIZE[0]//2, VIDEO_DISPLAY_SIZE[1]//2 - 20,
            text="No Video Selected", 
            fill="white", 
            font=("Arial", 14, "bold")
        )
        self.video_canvas.create_text(
            VIDEO_DISPLAY_SIZE[0]//2, VIDEO_DISPLAY_SIZE[1]//2 + 10,
            text="Click 'Browse' to select a video file", 
            fill="gray", 
            font=("Arial", 10)
        )
    
    def validate_time_format(self, time_str):
        """Validate time format (HH:MM:SS)."""
        if not time_str:
            return False
        try:
            parts = time_str.split(":")
            if len(parts) != 3:
                return False
            h, m, s = map(int, parts)
            # Check valid ranges
            if 0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59:
                return True
            return False
        except (ValueError, AttributeError):
            return False
    
    def open_logs_folder(self):
        """Open the logs folder in the file explorer."""
        try:
            log_file_path = self.log_path.get()
            if log_file_path and os.path.exists(log_file_path):
                # Open the folder containing the log file
                folder_path = os.path.dirname(log_file_path)
                
                # Platform-specific folder opening
                import platform
                system = platform.system()
                
                if system == "Windows":
                    # Use explorer to open folder and select the file
                    os.system(f'explorer /select,"{log_file_path}"')
                elif system == "Darwin":  # macOS
                    os.system(f'open -R "{log_file_path}"')
                else:  # Linux and others
                    os.system(f'xdg-open "{folder_path}"')
                    
            else:
                # Fallback: open logs directory
                logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
                if os.path.exists(logs_dir):
                    import platform
                    system = platform.system()
                    
                    if system == "Windows":
                        os.system(f'explorer "{logs_dir}"')
                    elif system == "Darwin":  # macOS
                        os.system(f'open "{logs_dir}"')
                    else:  # Linux and others
                        os.system(f'xdg-open "{logs_dir}"')
                        
        except Exception as e:
            print(f"Error opening logs folder: {e}")
            # Show message to user
            messagebox.showinfo("Info", f"Processing complete! Log saved to:\n{self.log_path.get()}")
    
    def ensure_logs_directory(self):
        """Ensure the logs directory exists."""
        try:
            logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            os.makedirs(logs_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create logs directory: {e}")
    
def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = TollBoothGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
