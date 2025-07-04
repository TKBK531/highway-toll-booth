# Configuration parameters for the toll booth detection system

# Video processing settings
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Detection parameters
MIN_STATIONARY_DURATION_SECONDS = 3.5
PIXEL_MOVEMENT_TOLERANCE = 75
CONFIDENCE_THRESHOLD = 0.4

# Vehicle classification
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
VEHICLE_TYPE_MAP = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Detection behavior
FRONT_VEHICLE_HEURISTIC = "highest_y2"  # Options: "lowest_y1", "highest_y2", "closest_to_bottom_edge", "closest_to_top_edge"

# Detection line coordinates (will be scaled based on video resolution)
ORIGINAL_LINE_P1 = (374, 95)
ORIGINAL_LINE_P2 = (1254, 398)
ASSUMED_ORIGINAL_WIDTH = 1920
ASSUMED_ORIGINAL_HEIGHT = 1080

# Model settings
MODEL_PATH = "yolov8n.pt"

# Output formatting
COLUMN_WIDTH = 30
COLUMN_WIDTHS = {
    "id": COLUMN_WIDTH,
    "type": COLUMN_WIDTH,
    "from": COLUMN_WIDTH,
    "to": COLUMN_WIDTH,
    "duration": COLUMN_WIDTH,
}
TABLE_HEADERS = [
    "Vehicle ID",
    "Vehicle Type", 
    "Stationary From",
    "Stationary To",
    "Duration (s)",
]

# GUI settings
WINDOW_TITLE = "Highway Toll Booth Vehicle Detection"
WINDOW_SIZE = "1400x900"
VIDEO_DISPLAY_SIZE = (640, 360)
