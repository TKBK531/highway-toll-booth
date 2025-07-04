import cv2
from ultralytics import YOLO
from utils import calculate_bbox_center, is_point_before_line, scale_coordinates
from config import *


class DetectionEngine:
    """Handles YOLO detection and vehicle tracking logic."""
    
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model."""
        try:
            self.model = YOLO(MODEL_PATH)
            return True, "YOLO model loaded successfully"
        except Exception as e:
            return False, f"Error loading YOLO model: {e}"
    
    def get_scaled_detection_line(self, target_width, target_height):
        """Get detection line coordinates scaled for the target resolution."""
        x_scale = target_width / ASSUMED_ORIGINAL_WIDTH
        y_scale = target_height / ASSUMED_ORIGINAL_HEIGHT
        
        scaled_p1 = (int(ORIGINAL_LINE_P1[0] * x_scale), int(ORIGINAL_LINE_P1[1] * y_scale))
        scaled_p2 = (int(ORIGINAL_LINE_P2[0] * x_scale), int(ORIGINAL_LINE_P2[1] * y_scale))
        
        return scaled_p1, scaled_p2
    
    def detect_vehicles(self, frame):
        """Detect vehicles in a frame and return filtered detections."""
        if not self.model:
            return []
        
        results = self.model.predict(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        
        detected_vehicles_info = []
        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            for i in range(len(boxes)):
                class_id = int(clss[i])
                if class_id in VEHICLE_CLASS_IDS:
                    bbox = boxes[i]
                    detected_vehicles_info.append((bbox, class_id))
        
        return detected_vehicles_info
    
    def filter_vehicles_by_line(self, detected_vehicles, line_p1, line_p2):
        """Filter vehicles that are before the detection line."""
        filtered_vehicles = []
        for bbox, class_id in detected_vehicles:
            center = calculate_bbox_center(bbox)
            if is_point_before_line(center, line_p1, line_p2):
                filtered_vehicles.append((bbox, class_id))
        return filtered_vehicles
    
    def find_front_vehicle(self, detected_vehicles):
        """Find the front vehicle based on the configured heuristic."""
        if not detected_vehicles:
            return None
        
        if FRONT_VEHICLE_HEURISTIC == "lowest_y1":
            detected_vehicles.sort(key=lambda item: item[0][1])
            return detected_vehicles[0] if detected_vehicles else None
        elif FRONT_VEHICLE_HEURISTIC == "highest_y2":
            detected_vehicles.sort(key=lambda item: item[0][3], reverse=True)
            return detected_vehicles[0] if detected_vehicles else None
        elif FRONT_VEHICLE_HEURISTIC == "closest_to_bottom_edge":
            vehicle_centers_with_info = [
                (info, calculate_bbox_center(info[0]))
                for info in detected_vehicles
            ]
            vehicle_centers_with_info.sort(key=lambda item: item[1][1], reverse=True)
            return vehicle_centers_with_info[0][0] if vehicle_centers_with_info else None
        elif FRONT_VEHICLE_HEURISTIC == "closest_to_top_edge":
            vehicle_centers_with_info = [
                (info, calculate_bbox_center(info[0]))
                for info in detected_vehicles
            ]
            vehicle_centers_with_info.sort(key=lambda item: item[1][1])
            return vehicle_centers_with_info[0][0] if vehicle_centers_with_info else None
        
        return None


class StationaryTracker:
    """Tracks stationary periods of vehicles."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset tracking state."""
        self.is_tracking = False
        self.start_time = 0.0
        self.start_center = None
        self.vehicle_type = "Unknown"
        self.event_counter = 0
    
    def update(self, front_vehicle_info, current_time, scaled_tolerance):
        """Update tracking state and return any completed events."""
        events = []
        
        if front_vehicle_info:
            bbox, class_id = front_vehicle_info
            center = calculate_bbox_center(bbox)
            vehicle_type = VEHICLE_TYPE_MAP.get(class_id, "Unknown")
            
            if self.is_tracking:
                # Check if vehicle moved beyond tolerance
                from utils import calculate_distance
                distance = calculate_distance(center, self.start_center)
                
                if distance > scaled_tolerance:
                    # Vehicle moved, end current tracking period
                    duration = current_time - self.start_time
                    if duration >= MIN_STATIONARY_DURATION_SECONDS:
                        self.event_counter += 1
                        events.append({
                            "vehicle_id": self.event_counter,
                            "vehicle_type": self.vehicle_type,
                            "start_sec_video": self.start_time,
                            "end_sec_video": current_time,
                            "duration_sec": duration,
                        })
                    
                    # Start new tracking period
                    self.start_time = current_time
                    self.start_center = center
                    self.vehicle_type = vehicle_type
            else:
                # Start tracking
                self.is_tracking = True
                self.start_time = current_time
                self.start_center = center
                self.vehicle_type = vehicle_type
        else:
            # No vehicle detected
            if self.is_tracking:
                # End current tracking period
                duration = current_time - self.start_time
                if duration >= MIN_STATIONARY_DURATION_SECONDS:
                    self.event_counter += 1
                    events.append({
                        "vehicle_id": self.event_counter,
                        "vehicle_type": self.vehicle_type,
                        "start_sec_video": self.start_time,
                        "end_sec_video": current_time,
                        "duration_sec": duration,
                    })
                
                self.is_tracking = False
                self.start_center = None
                self.vehicle_type = "Unknown"
        
        return events
    
    def finalize(self, final_time):
        """Finalize any ongoing tracking period."""
        events = []
        if self.is_tracking and self.start_center:
            duration = final_time - self.start_time
            if duration >= MIN_STATIONARY_DURATION_SECONDS:
                self.event_counter += 1
                events.append({
                    "vehicle_id": self.event_counter,
                    "vehicle_type": self.vehicle_type,
                    "start_sec_video": self.start_time,
                    "end_sec_video": final_time,
                    "duration_sec": duration,
                })
        return events
    
    def get_current_info(self):
        """Get current tracking information for display."""
        return {
            "is_tracking": self.is_tracking,
            "start_center": self.start_center,
            "vehicle_type": self.vehicle_type,
            "start_time": self.start_time
        }
