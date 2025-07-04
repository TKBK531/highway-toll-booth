import math
from datetime import timedelta
from config import COLUMN_WIDTH


def calculate_bbox_center(bbox):
    """Calculate the center point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    if p1 is None or p2 is None:
        return float("inf")
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def is_point_before_line(point, line_p1, line_p2):
    """Check if a point is on the 'before' side of a line (left of the line vector)."""
    x, y = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1) < 0


def scale_coordinates(original_coords, original_width, original_height, target_width, target_height):
    """Scale coordinates from original resolution to target resolution."""
    x_scale = target_width / original_width
    y_scale = target_height / original_height
    
    if len(original_coords) == 2:  # Point (x, y)
        x, y = original_coords
        return (int(x * x_scale), int(y * y_scale))
    elif len(original_coords) == 4:  # Bounding box (x1, y1, x2, y2)
        x1, y1, x2, y2 = original_coords
        return (int(x1 * x_scale), int(y1 * y_scale), int(x2 * x_scale), int(y2 * y_scale))
    else:
        return original_coords


def format_time(total_seconds):
    """Format seconds into HH:MM:SS.mmm format."""
    if total_seconds < 0:
        total_seconds = 0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"


def create_table_line(widths_dict, header_keys_ordered):
    """Create a table separator line."""
    char = "+"
    parts = [char]
    for key in header_keys_ordered:
        parts.append("-" * widths_dict.get(key, COLUMN_WIDTH))
        parts.append(char)
    return "".join(parts)


def format_table_row(data_tuple, widths_dict, col_keys_ordered):
    """Format a data tuple as a table row."""
    if len(data_tuple) != len(col_keys_ordered):
        return "| Error: Data mismatch |"
    parts = ["|"]
    for i, item in enumerate(data_tuple):
        item_str = str(item)
        width_key = col_keys_ordered[i]
        col_width = widths_dict.get(width_key, COLUMN_WIDTH)
        parts.append(f" {item_str:<{col_width - 2}} ")
        parts.append("|")
    return "".join(parts)
