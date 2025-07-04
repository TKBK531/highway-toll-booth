#!/usr/bin/env python3
"""
Simple test script to verify the modular toll booth GUI works correctly.
"""

def test_imports():
    """Test that all modules can be imported successfully."""
    try:
        print("Testing imports...")
        
        import config
        print("✓ config.py imported")
        
        import utils
        print("✓ utils.py imported")
        
        import detection_engine
        print("✓ detection_engine.py imported")
        
        import video_processor
        print("✓ video_processor.py imported")
        
        import main_gui
        print("✓ main_gui.py imported")
        
        print("\n🎉 All modules imported successfully!")
        print("✓ Modular structure is working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration values."""
    try:
        import config
        
        print("\nTesting configuration...")
        print(f"✓ Target resolution: {config.TARGET_WIDTH}x{config.TARGET_HEIGHT}")
        print(f"✓ Min stationary duration: {config.MIN_STATIONARY_DURATION_SECONDS}s")
        print(f"✓ Vehicle types: {list(config.VEHICLE_TYPE_MAP.values())}")
        print(f"✓ Window title: {config.WINDOW_TITLE}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Highway Toll Booth Detection - Module Test")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_config()
    
    print("\n" + "=" * 50)
    if success:
        print("🎯 All tests passed! Ready to run: python main_gui.py")
    else:
        print("❌ Some tests failed. Check the errors above.")
