# Highway Toll Booth Vehicle Detection

This project detects vehicles at a toll booth in a video, tracks when the front vehicle stops and starts, and logs the stationary periods. It uses YOLOv8 for vehicle detection and allows the user to define a detection line interactively.

## Features
- Detects vehicles (car, motorcycle, bus, truck) in a video.
- Lets the user define an angled detection line by clicking two points on the first frame.
- Only vehicles before the line are considered for detection and logging.
- Tracks when the front vehicle stops and starts moving.
- Logs each stationary period (with vehicle type, time, and duration) to a text file in tabular format.
- Visualizes detections, the detection line, and stationary status in a window.

## Requirements
- Python 3.8+
- Windows OS (tested)
- [YOLOv8 weights file](https://github.com/ultralytics/ultralytics) (e.g., `yolov8n.pt`)

### Python Packages
Install all dependencies with:

```
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install the main dependencies manually:

```
pip install ultralytics opencv-python numpy
```

## Setup
1. **Add Your Video**
   - Place your video file (e.g., `toll.mp4`) in the `resources/` folder.
   - Update the `VIDEO_PATH` variable in `main.py` if your video has a different name or location.

. **(Optional) Edit Configuration**
   - You can change detection thresholds, vehicle types, and other parameters at the top of `main.py`.

## How to Run
1. Open a terminal in the project directory.
2. Run the main script:

```
python main.py
```

4. **Select the Detection Line**
   - The first frame of the video will appear.
   - Click two points to define the detection line (e.g., across the toll booth).
   - The script will use this line to filter vehicles.

5. The script will process the video, display detections, and log stationary periods.
6. When finished, results are saved in `log.txt`.

## Output
- **log.txt**: Contains a table of all detected stationary periods for the front vehicle, including vehicle ID, type, start/end time, and duration.

## Notes
- Press `q` in the video window to stop processing early.
- The detection line can be angled and is defined by your two clicks.
- Only vehicles before the line are considered for logging.

## Troubleshooting
- If you see errors about missing packages, install them with `pip` as shown above.
- If you have issues with OpenCV windows, ensure you are not running in a headless environment.
- For GPU acceleration, ensure you have the correct CUDA drivers and a compatible PyTorch/Ultralytics install.

## License
This project is for educational and research purposes.
