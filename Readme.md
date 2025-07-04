# Highway Toll Booth Vehicle Detection

This project detects vehicles at a toll booth in a video, tracks when vehicles stop and start, and logs the stationary periods. It features a user-friendly GUI application built with modern, modular Python code.

## Features
- **User-Friendly GUI**: Easy-to-use interface for selecting videos and viewing results
- **Real-Time Processing**: Live video display with detection overlays
- **Vehicle Detection**: Detects cars, motorcycles, buses, and trucks using YOLOv8
- **Stationary Tracking**: Monitors when vehicles stop and calculates duration
- **Automatic Scaling**: Processes videos at optimal 1280x720 resolution regardless of input size
- **Progress Monitoring**: Real-time progress tracking with visual feedback
- **Comprehensive Logging**: Detailed tabular logs of all stationary events

## Project Structure
```
├── main_gui.py           # Main GUI application (entry point)
├── video_processor.py    # Video processing and file operations
├── detection_engine.py   # YOLO detection and vehicle tracking logic
├── config.py            # Configuration parameters
├── utils.py             # Utility functions
├── requirements.txt     # Python dependencies
├── yolov8n.pt          # YOLO model weights
└── resources/          # Video files directory
```

## Requirements
- Python 3.8+
- Windows OS (tested)

## Installation

1. **Clone or download the project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure YOLO model is present**:
   - The `yolov8n.pt` file should be in the project root
   - It will be downloaded automatically if missing when you first run the application

## Usage

### GUI Application (Recommended)
1. **Run the application**:
   ```bash
   python main_gui.py
   ```

2. **Select files**:
   - Click "Browse" next to "Video File" to select your toll booth video
   - Click "Browse" next to "Log File" to choose where to save results

3. **Start processing**:
   - Click "Start Processing" to begin analysis
   - Watch the live video feed showing detections in real-time
   - Monitor progress and view results as they appear

4. **View results**:
   - Detection events appear in the results table on the right
   - Final log file is saved to your chosen location
   - Use "Stop Processing" if you need to halt early

### Visual Indicators
- **Green boxes**: All detected vehicles
- **Red box**: Front vehicle being tracked
- **Purple line**: Detection boundary
- **Yellow circle**: Stationary tolerance zone
- **Text overlays**: Vehicle types and stationary duration

## Configuration
Edit `config.py` to customize:
- Detection thresholds and confidence levels
- Stationary duration requirements
- Video processing resolution
- Vehicle types to detect
- Front vehicle selection method

## Output
The application generates a detailed log file with:
- Vehicle ID and type
- Start and end timestamps
- Duration of stationary periods
- Formatted as a readable table

## Troubleshooting
- **Model loading errors**: Ensure `yolov8n.pt` is present and accessible
- **Video playback issues**: Check that your video format is supported (MP4, AVI, MOV, etc.)
- **Performance issues**: The application automatically scales videos to 1280x720 for optimal performance
- **GUI not responsive**: Processing runs in background threads to keep the interface responsive

## Notes
- The application automatically handles different video resolutions
- Detection coordinates are dynamically scaled
- All processing maintains aspect ratios for accurate detection
- Press "Stop Processing" to halt analysis at any time

## License
This project is for educational and research purposes.
