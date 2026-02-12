"""
Check detailed specifications of each camera.
"""
import cv2
import subprocess

def check_camera_specs():
    """Check specifications for each camera."""
    
    camera_indices = [6, 2, 4]
    
    print("\n" + "="*60)
    print("CAMERA SPECIFICATIONS CHECKER")
    print("="*60)
    
    for idx in camera_indices:
        print(f"\n{'='*60}")
        print(f"CAMERA {idx} - /dev/video{idx}")
        print('='*60)
        
        # Method 1: Using v4l2-ctl (most detailed)
        print("\n--- V4L2 Device Info ---")
        try:
            result = subprocess.run(
                ['v4l2-ctl', '-d', f'/dev/video{idx}', '--all'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse relevant info
                for line in result.stdout.split('\n'):
                    if any(keyword in line.lower() for keyword in 
                          ['card', 'driver', 'bus', 'capabilities', 'video capture']):
                        print(f"  {line.strip()}")
            else:
                print(f"  ❌ Could not get v4l2 info")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # Method 2: Check supported formats
        print("\n--- Supported Formats ---")
        try:
            result = subprocess.run(
                ['v4l2-ctl', '-d', f'/dev/video{idx}', '--list-formats-ext'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(result.stdout[:500])  # First 500 chars
            else:
                print(f"  ❌ Could not list formats")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # Method 3: Using OpenCV
        print("\n--- OpenCV Properties ---")
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        
        if not cap.isOpened():
            print(f"  ❌ Could not open camera {idx} with OpenCV")
            continue
        
        # Try to get backend name
        backend = cap.getBackendName()
        print(f"  Backend: {backend}")
        
        # Test different resolutions and FPS
        test_resolutions = [
            (640, 480, "VGA"),
            (1280, 720, "HD 720p"),
            (1920, 1080, "Full HD 1080p"),
        ]
        
        print("\n  Supported Resolutions:")
        supported_res = []
        
        for width, height, name in test_resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            if actual_width == width and actual_height == height:
                supported_res.append((width, height, name, actual_fps))
                print(f"    ✓ {name}: {width}x{height} @ {actual_fps} FPS")
            else:
                print(f"    ❌ {name}: requested {width}x{height}, got {actual_width}x{actual_height}")
        
        # Get additional properties
        print("\n  Camera Properties:")
        properties = {
            'FRAME_WIDTH': cv2.CAP_PROP_FRAME_WIDTH,
            'FRAME_HEIGHT': cv2.CAP_PROP_FRAME_HEIGHT,
            'FPS': cv2.CAP_PROP_FPS,
            'FOURCC': cv2.CAP_PROP_FOURCC,
            'BRIGHTNESS': cv2.CAP_PROP_BRIGHTNESS,
            'CONTRAST': cv2.CAP_PROP_CONTRAST,
            'SATURATION': cv2.CAP_PROP_SATURATION,
            'HUE': cv2.CAP_PROP_HUE,
            'GAIN': cv2.CAP_PROP_GAIN,
            'EXPOSURE': cv2.CAP_PROP_EXPOSURE,
            'AUTO_EXPOSURE': cv2.CAP_PROP_AUTO_EXPOSURE,
            'AUTOFOCUS': cv2.CAP_PROP_AUTOFOCUS,
        }
        
        for name, prop in properties.items():
            value = cap.get(prop)
            if value != -1:  # -1 means not supported
                if name == 'FOURCC':
                    # Decode FOURCC
                    fourcc = int(value)
                    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                    print(f"    {name}: {fourcc_str}")
                else:
                    print(f"    {name}: {value}")
        
        # Try to capture a test frame
        print("\n  Capture Test:")
        warmup_frames = 10
        for _ in range(warmup_frames):
            cap.read()
        
        ret, frame = cap.read()
        if ret:
            brightness = frame.mean()
            print(f"    ✓ Successfully captured frame")
            print(f"    Frame shape: {frame.shape}")
            print(f"    Brightness: {brightness:.1f}")
            
            # Save test image
            filename = f"camera_{idx}_spec_test.jpg"
            cv2.imwrite(filename, frame)
            print(f"    Saved test image: {filename}")
        else:
            print(f"    ❌ Failed to capture frame")
        
        cap.release()
        
        # Summary for this camera
        if supported_res:
            print(f"\n  📋 RECOMMENDED SETTINGS FOR CAMERA {idx}:")
            best = supported_res[-1]  # Highest resolution
            print(f"    Resolution: {best[0]}x{best[1]} ({best[2]})")
            print(f"    FPS: {best[3]}")
            print(f'    Config: "index_or_path":{idx},"width":{best[0]},"height":{best[1]},"fps":{int(best[3])}')
    
    print("\n" + "="*60)
    print("SUMMARY - RECOMMENDED LEROBOT CONFIG")
    print("="*60)
    print("\nBased on the specs above, use these settings in your lerobot-record command.")


if __name__ == "__main__":
    try:
        check_camera_specs()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()