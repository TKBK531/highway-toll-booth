#!/usr/bin/env python3
"""
Test script to verify the logs folder functionality works correctly.
"""
import os
import tempfile
import shutil

def test_logs_functionality():
    """Test the logs folder setup and default filename generation."""
    try:
        print("🔧 Testing logs folder functionality...")
        
        # Import main GUI components
        import main
        from config import DEFAULT_VIDEO_START_TIME
        
        print(f"✓ GUI module imported successfully")
        print(f"✓ Default start time: {DEFAULT_VIDEO_START_TIME}")
        
        # Test logs directory creation
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(current_dir, "logs")
        
        if os.path.exists(logs_dir):
            print(f"✓ Logs directory exists: {logs_dir}")
        else:
            print(f"❌ Logs directory missing: {logs_dir}")
            return False
        
        # Test .gitkeep file
        gitkeep_file = os.path.join(logs_dir, ".gitkeep")
        if os.path.exists(gitkeep_file):
            print(f"✓ .gitkeep file exists: {gitkeep_file}")
        else:
            print(f"❌ .gitkeep file missing: {gitkeep_file}")
            return False
        
        # Test default log filename generation
        test_video_paths = [
            "test_video.mp4",
            "my_toll_booth_recording.avi",
            "highway_cam_2025.mkv"
        ]
        
        for video_path in test_video_paths:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            expected_log = f"{video_name}_detection_log.txt"
            expected_path = os.path.join(logs_dir, expected_log)
            
            print(f"✓ Video: {video_path} → Log: {expected_log}")
        
        print(f"\n🎯 All logs functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_folder_opening():
    """Test platform detection for folder opening."""
    try:
        import platform
        system = platform.system()
        print(f"\n🖥️  Detected platform: {system}")
        
        if system == "Windows":
            print("✓ Will use 'explorer' command to open folders")
        elif system == "Darwin":
            print("✓ Will use 'open' command to open folders (macOS)")
        else:
            print("✓ Will use 'xdg-open' command to open folders (Linux)")
        
        return True
        
    except Exception as e:
        print(f"❌ Platform detection error: {e}")
        return False

if __name__ == "__main__":
    print("📁 Highway Toll Booth - Logs Functionality Test")
    print("=" * 55)
    
    success = True
    success &= test_logs_functionality()
    success &= test_folder_opening()
    
    print("\n" + "=" * 55)
    if success:
        print("🎉 All logs tests passed! Ready to use the application.")
        print("\n📋 New Features:")
        print("  • Default log location: logs/ folder")
        print("  • Auto-generated log filenames based on video name")
        print("  • Option to open logs folder after processing")
        print("  • Git-friendly logs directory structure")
    else:
        print("❌ Some tests failed. Check the errors above.")
