import cv2
import threading
from tkinter import messagebox
from detection_engine import DetectionEngine, StationaryTracker
from utils import format_time, create_table_line, format_table_row
from config import *


class VideoProcessor:
    """Handles video processing logic and file operations."""
    
    def __init__(self, gui_callback=None):
        self.gui_callback = gui_callback
        self.detection_engine = DetectionEngine()
        self.stationary_tracker = StationaryTracker()
        self.processing = False
        self.cap = None
    
    def start_processing(self, video_path, log_path):
        """Start video processing in a separate thread."""
        self.processing = True
        self.stationary_tracker.reset()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._process_video_thread, 
            args=(video_path, log_path)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop video processing."""
        self.processing = False
        if self.cap:
            self.cap.release()
    
    def _process_video_thread(self, video_path, log_path):
        """Main video processing loop (runs in separate thread)."""
        try:
            # Open video
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                self._gui_callback("error", "Could not open video file")
                return
            
            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                fps = 30.0
            
            original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate scaling factors
            width_scale = TARGET_WIDTH / original_width
            height_scale = TARGET_HEIGHT / original_height
            scaled_tolerance = int(PIXEL_MOVEMENT_TOLERANCE * min(width_scale, height_scale))
            
            # Get detection line
            line_p1, line_p2 = self.detection_engine.get_scaled_detection_line(
                TARGET_WIDTH, TARGET_HEIGHT
            )
            
            # Update GUI with video info
            self._gui_callback("video_info", {
                "original": f"{original_width}x{original_height}",
                "target": f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
                "fps": fps,
                "scaling": (width_scale, height_scale),
                "tolerance": scaled_tolerance
            })
            
            # Processing loop
            stationary_log = []
            frame_idx = 0
            
            while self.processing:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Resize frame
                resized_frame = self._resize_frame(frame)
                current_time = frame_idx / fps
                
                # Detect vehicles
                all_vehicles = self.detection_engine.detect_vehicles(resized_frame)
                filtered_vehicles = self.detection_engine.filter_vehicles_by_line(
                    all_vehicles, line_p1, line_p2
                )
                
                # Find front vehicle
                front_vehicle = self.detection_engine.find_front_vehicle(filtered_vehicles)
                
                # Update stationary tracking
                events = self.stationary_tracker.update(
                    front_vehicle, current_time, scaled_tolerance
                )
                
                # Add events to log and GUI
                for event in events:
                    stationary_log.append(event)
                    self._gui_callback("new_event", event)
                
                # Create display frame
                display_frame = self._create_display_frame(
                    resized_frame, filtered_vehicles, front_vehicle, 
                    line_p1, line_p2, current_time, scaled_tolerance
                )
                
                # Update GUI
                self._gui_callback("frame_update", display_frame)
                self._gui_callback("progress_update", {
                    "progress": (frame_idx / total_frames) * 100 if total_frames > 0 else 0,
                    "current_frame": frame_idx,
                    "total_frames": total_frames
                })
                
                frame_idx += 1
            
            # Finalize any ongoing tracking
            final_events = self.stationary_tracker.finalize((frame_idx - 1) / fps)
            for event in final_events:
                stationary_log.append(event)
                self._gui_callback("new_event", event)
            
            # Save log file
            self._save_log_file(stationary_log, log_path)
            
            # Processing complete
            self._gui_callback("processing_complete", len(stationary_log))
            
        except Exception as e:
            self._gui_callback("error", f"Processing error: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
            self.processing = False
            self._gui_callback("processing_finished", None)
    
    def _resize_frame(self, frame):
        """Resize frame to target resolution."""
        return cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    
    def _create_display_frame(self, frame, all_vehicles, front_vehicle, 
                             line_p1, line_p2, current_time, scaled_tolerance):
        """Create the display frame with all overlays."""
        display_frame = frame.copy()
        
        # Draw all vehicle detections
        for bbox, class_id in all_vehicles:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Draw detection line
        cv2.line(display_frame, line_p1, line_p2, (255, 0, 255), 2)
        
        # Draw front vehicle
        if front_vehicle:
            bbox, class_id = front_vehicle
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            from utils import calculate_bbox_center
            center = calculate_bbox_center(bbox)
            cv2.circle(display_frame, center, 5, (255, 0, 0), -1)
            
            vehicle_type = VEHICLE_TYPE_MAP.get(class_id, "Unknown")
            cv2.putText(display_frame, vehicle_type, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw stationary tracking info
        tracking_info = self.stationary_tracker.get_current_info()
        if tracking_info["is_tracking"] and tracking_info["start_center"]:
            duration = current_time - tracking_info["start_time"]
            cv2.circle(display_frame, tracking_info["start_center"], 
                      scaled_tolerance, (255, 255, 0), 1)
            cv2.putText(display_frame,
                       f"Stationary ({tracking_info['vehicle_type']}): {duration:.1f}s",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return display_frame
    
    def _save_log_file(self, stationary_log, log_path):
        """Save the stationary events to a log file."""
        try:
            col_keys_ordered = ["id", "type", "from", "to", "duration"]
            
            with open(log_path, "w") as f:
                # Write header
                f.write(create_table_line(COLUMN_WIDTHS, col_keys_ordered) + "\n")
                f.write(format_table_row(TABLE_HEADERS, COLUMN_WIDTHS, col_keys_ordered) + "\n")
                f.write(create_table_line(COLUMN_WIDTHS, col_keys_ordered) + "\n")
                
                # Write events
                for event in stationary_log:
                    f.write(format_table_row([
                        event["vehicle_id"],
                        event["vehicle_type"],
                        format_time(event["start_sec_video"]),
                        format_time(event["end_sec_video"]),
                        f"{event['duration_sec']:.1f}",
                    ], COLUMN_WIDTHS, col_keys_ordered) + "\n")
                
                f.write(create_table_line(COLUMN_WIDTHS, col_keys_ordered) + "\n")
                
        except Exception as e:
            self._gui_callback("error", f"Error saving log file: {e}")
    
    def _gui_callback(self, event_type, data):
        """Send callback to GUI if available."""
        if self.gui_callback:
            self.gui_callback(event_type, data)
    
    def is_model_loaded(self):
        """Check if the detection model is loaded."""
        return self.detection_engine.model is not None
