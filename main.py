import cv2
from ultralytics import YOLO
from datetime import timedelta
import numpy as np
import math

VIDEO_PATH = r"resources/toll.mp4"  # Path to the video file
VIDEO_START_TIME_STR = "17:13:18"  # HH:MM:SS 24 hour format
OUTPUT_FILE = "log.txt"
MODEL_PATH = "yolov8n.pt"

MIN_STATIONARY_DURATION_SECONDS = (
    5.0  # Minimum duration in seconds to consider a vehicle stationary
)
PIXEL_MOVEMENT_TOLERANCE = (
    75  # Pixel distance tolerance to consider a vehicle stationary
)
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
CONFIDENCE_THRESHOLD = 0.4

VEHICLE_TYPE_MAP = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FRONT_VEHICLE_HEURISTIC = "highest_y2"  # Options: "lowest_y1", "highest_y2", "closest_to_bottom_edge", "closest_to_top_edge"

COLUMN_WIDTH = 30  # Width of each column in the output table
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


def parse_time_to_seconds(time_str):
    if not time_str:
        return 0
    try:
        h, m, s = map(int, time_str.split(":"))
        return timedelta(hours=h, minutes=m, seconds=s).total_seconds()
    except ValueError:
        print(
            f"Warning: Invalid time format for VIDEO_START_TIME_STR: '{time_str}'. Using 0 offset."
        )
        return 0


def format_time(total_seconds):
    if total_seconds < 0:
        total_seconds = 0
    td = timedelta(seconds=total_seconds)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"


def calculate_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def calculate_distance(p1, p2):
    if p1 is None or p2 is None:
        return float("inf")
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def create_table_line(widths_dict, header_keys_ordered):
    char = "+"
    parts = [char]
    for key in header_keys_ordered:
        parts.append("-" * widths_dict.get(key, COLUMN_WIDTH))
        parts.append(char)
    return "".join(parts)


def format_table_row(data_tuple, widths_dict, col_keys_ordered):
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


def main():
    video_start_offset_seconds = parse_time_to_seconds(VIDEO_START_TIME_STR)
    if video_start_offset_seconds > 0:
        print(
            f"Applying video start time offset: {format_time(video_start_offset_seconds)} ({video_start_offset_seconds} seconds)"
        )

    try:
        model = YOLO(MODEL_PATH)
        print(f"YOLO model '{MODEL_PATH}' loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(
        f"Video Info: {frame_width}x{frame_height} @ {fps:.2f} FPS, Total Frames (reported): {total_frames}"
    )

    stationary_log = []
    is_tracking_stationary_period = False
    current_stationary_start_time_sec_video = 0.0
    current_stationary_start_center = None
    current_stationary_vehicle_type = "Unknown"

    front_vehicle_event_id_counter = 0

    frame_idx = 0
    processed_frames_count = 0

    col_keys_ordered_for_table = ["id", "type", "from", "to", "duration"]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream reached.")
                break

            current_timestamp_sec_video = frame_idx / fps
            processed_frames_count += 1

            results = model.predict(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)

            detected_vehicles_info = []
            if results and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()
                for i in range(len(boxes)):
                    class_id = int(clss[i])
                    if class_id in VEHICLE_CLASS_IDS:
                        detected_vehicles_info.append((boxes[i], class_id))

            front_vehicle_info = None
            front_vehicle_bbox = None
            front_vehicle_center = None
            front_vehicle_class_id = None

            if detected_vehicles_info:
                if FRONT_VEHICLE_HEURISTIC == "lowest_y1":
                    detected_vehicles_info.sort(key=lambda item: item[0][1])
                    if detected_vehicles_info:
                        front_vehicle_info = detected_vehicles_info[0]
                elif FRONT_VEHICLE_HEURISTIC == "highest_y2":
                    detected_vehicles_info.sort(
                        key=lambda item: item[0][3], reverse=True
                    )
                    if detected_vehicles_info:
                        front_vehicle_info = detected_vehicles_info[0]
                elif FRONT_VEHICLE_HEURISTIC == "closest_to_bottom_edge":
                    vehicle_centers_with_info = [
                        (info, calculate_bbox_center(info[0]))
                        for info in detected_vehicles_info
                    ]
                    vehicle_centers_with_info.sort(
                        key=lambda item: item[1][1], reverse=True
                    )
                    if vehicle_centers_with_info:
                        front_vehicle_info = vehicle_centers_with_info[0][0]
                elif FRONT_VEHICLE_HEURISTIC == "closest_to_top_edge":
                    vehicle_centers_with_info = [
                        (info, calculate_bbox_center(info[0]))
                        for info in detected_vehicles_info
                    ]
                    vehicle_centers_with_info.sort(key=lambda item: item[1][1])
                    if vehicle_centers_with_info:
                        front_vehicle_info = vehicle_centers_with_info[0][0]

                if front_vehicle_info:
                    front_vehicle_bbox = front_vehicle_info[0]
                    front_vehicle_class_id = front_vehicle_info[1]
                    front_vehicle_center = calculate_bbox_center(front_vehicle_bbox)

            if front_vehicle_center:
                if is_tracking_stationary_period:
                    distance_from_start = calculate_distance(
                        front_vehicle_center, current_stationary_start_center
                    )
                    if distance_from_start > PIXEL_MOVEMENT_TOLERANCE:
                        stationary_end_time_sec_video = (frame_idx - 1) / fps
                        duration = (
                            stationary_end_time_sec_video
                            - current_stationary_start_time_sec_video
                        )
                        if duration >= MIN_STATIONARY_DURATION_SECONDS:
                            front_vehicle_event_id_counter += 1
                            stationary_log.append(
                                {
                                    "vehicle_id": front_vehicle_event_id_counter,
                                    "vehicle_type": current_stationary_vehicle_type,
                                    "start_sec_video": current_stationary_start_time_sec_video,
                                    "end_sec_video": stationary_end_time_sec_video,
                                    "duration_sec": duration,
                                }
                            )
                        is_tracking_stationary_period = True
                        current_stationary_start_time_sec_video = (
                            current_timestamp_sec_video
                        )
                        current_stationary_start_center = front_vehicle_center
                        current_stationary_vehicle_type = VEHICLE_TYPE_MAP.get(
                            front_vehicle_class_id, "Unknown"
                        )
                else:
                    is_tracking_stationary_period = True
                    current_stationary_start_time_sec_video = (
                        current_timestamp_sec_video
                    )
                    current_stationary_start_center = front_vehicle_center
                    current_stationary_vehicle_type = VEHICLE_TYPE_MAP.get(
                        front_vehicle_class_id, "Unknown"
                    )
            else:
                if is_tracking_stationary_period:
                    stationary_end_time_sec_video = (frame_idx - 1) / fps
                    duration = (
                        stationary_end_time_sec_video
                        - current_stationary_start_time_sec_video
                    )
                    if duration >= MIN_STATIONARY_DURATION_SECONDS:
                        front_vehicle_event_id_counter += 1
                        stationary_log.append(
                            {
                                "vehicle_id": front_vehicle_event_id_counter,
                                "vehicle_type": current_stationary_vehicle_type,
                                "start_sec_video": current_stationary_start_time_sec_video,
                                "end_sec_video": stationary_end_time_sec_video,
                                "duration_sec": duration,
                            }
                        )
                    is_tracking_stationary_period = False
                    current_stationary_start_center = None
                    current_stationary_vehicle_type = "Unknown"

            display_frame = frame.copy()
            for bbox_info_tuple in detected_vehicles_info:
                x1, y1, x2, y2 = map(int, bbox_info_tuple[0])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            if front_vehicle_bbox is not None:
                x1, y1, x2, y2 = map(int, front_vehicle_bbox)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                if front_vehicle_center:
                    cv2.circle(display_frame, front_vehicle_center, 5, (255, 0, 0), -1)
                type_to_display = VEHICLE_TYPE_MAP.get(
                    front_vehicle_class_id, "Unknown"
                )
                cv2.putText(
                    display_frame,
                    type_to_display,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            if is_tracking_stationary_period and current_stationary_start_center:
                display_stationary_duration = (
                    current_timestamp_sec_video
                    - current_stationary_start_time_sec_video
                )
                cv2.circle(
                    display_frame,
                    current_stationary_start_center,
                    PIXEL_MOVEMENT_TOLERANCE,
                    (255, 255, 0),
                    1,
                )
                cv2.putText(
                    display_frame,
                    f"Stationary ({current_stationary_vehicle_type}): {display_stationary_duration:.1f}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Frame", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Processing stopped by user ('q' pressed).")
                break

            frame_idx += 1
            if frame_idx % (int(fps) * 10) == 0:
                progress_str = (
                    f"{frame_idx}/{total_frames}"
                    if total_frames > 0
                    else f"{frame_idx}"
                )
                print(
                    f"Processed {progress_str} frames ({format_time(current_timestamp_sec_video)})..."
                )

    except KeyboardInterrupt:
        print("Processing interrupted by user (Ctrl+C).")
    finally:
        if is_tracking_stationary_period and current_stationary_start_center:
            end_timestamp_sec_video = (
                (processed_frames_count - 1) / fps
                if processed_frames_count > 0
                else current_stationary_start_time_sec_video
            )
            if end_timestamp_sec_video < current_stationary_start_time_sec_video:
                end_timestamp_sec_video = current_stationary_start_time_sec_video

            duration = end_timestamp_sec_video - current_stationary_start_time_sec_video
            if duration >= MIN_STATIONARY_DURATION_SECONDS:
                front_vehicle_event_id_counter += 1
                stationary_log.append(
                    {
                        "vehicle_id": front_vehicle_event_id_counter,
                        "vehicle_type": current_stationary_vehicle_type,
                        "start_sec_video": current_stationary_start_time_sec_video,
                        "end_sec_video": end_timestamp_sec_video,
                        "duration_sec": duration,
                    }
                )

        cap.release()
        cv2.destroyAllWindows()

        with open(OUTPUT_FILE, "w") as f:
            f.write("Front Vehicle Stationary Times Log\n")
            f.write("==================================\n")
            f.write(f"Stationary Threshold: {MIN_STATIONARY_DURATION_SECONDS:.1f}s\n")
            f.write("==================================\n")

            total_table_width = (
                sum(COLUMN_WIDTHS[key] for key in col_keys_ordered_for_table)
                + len(col_keys_ordered_for_table)
                + 1
            )
            f.write("-" * total_table_width + "\n\n")

            line_sep = create_table_line(COLUMN_WIDTHS, col_keys_ordered_for_table)
            f.write(line_sep + "\n")

            header_data_tuple = tuple(TABLE_HEADERS)
            f.write(
                format_table_row(
                    header_data_tuple, COLUMN_WIDTHS, col_keys_ordered_for_table
                )
                + "\n"
            )
            f.write(line_sep + "\n")

            if not stationary_log:
                empty_row_data = tuple(["-" for _ in col_keys_ordered_for_table])
                f.write(
                    format_table_row(
                        empty_row_data, COLUMN_WIDTHS, col_keys_ordered_for_table
                    )
                    + "\n"
                )
                f.write(line_sep + "\n")
                f.write("No stationary periods detected meeting the criteria.\n\n")
            else:
                for record in stationary_log:
                    start_time_with_offset = (
                        record["start_sec_video"] + video_start_offset_seconds
                    )
                    end_time_with_offset = (
                        record["end_sec_video"] + video_start_offset_seconds
                    )

                    data_tuple = (
                        record["vehicle_id"],
                        record["vehicle_type"],
                        format_time(start_time_with_offset),
                        format_time(end_time_with_offset),
                        f"{record['duration_sec']:.2f}",
                    )
                    f.write(
                        format_table_row(
                            data_tuple, COLUMN_WIDTHS, col_keys_ordered_for_table
                        )
                        + "\n"
                    )
                f.write(line_sep + "\n")

        print(f"\nProcessing complete. Stationary times recorded in '{OUTPUT_FILE}'")
        print(f"Total frames processed: {processed_frames_count}")


if __name__ == "__main__":
    main()
