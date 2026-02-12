"""
Headless camera viewer - optimized for Innomaker USB cameras.
"""
import cv2
import time
import os

def headless_camera_viewer():
    """Save periodic snapshots from all cameras."""
    
    # Your available cameras
    camera_indices = [6, 2, 4]
    camera_names = {6: "Camera_0", 2: "Camera_2", 4: "Camera_4"}
    captures = {}
    
    print("\n" + "="*60)
    print("HEADLESS CAMERA VIEWER - INNOMAKER OPTIMIZED")
    print("="*60)
    print("\nOpening cameras...\n")
    
    # Open all cameras
    for idx in camera_indices:
        print(f"Opening camera {idx}...", end=" ")
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)  # Force V4L2 backend
        
        if not cap.isOpened():
            print(f"❌ Failed")
            continue
        
        # Set resolution
        if idx == 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Innomaker-specific settings
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # Try auto exposure first
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Auto mode
        
        # If camera 0, try additional settings for dark image
        if idx == 0:
            print("(applying brightness boost)...", end=" ")
            # Manual exposure settings for dark cameras
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual mode
            cap.set(cv2.CAP_PROP_EXPOSURE, -4)  # Increase exposure (try -1 to -10)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Increase brightness
            cap.set(cv2.CAP_PROP_CONTRAST, 150)  # Increase contrast
            cap.set(cv2.CAP_PROP_GAIN, 100)  # Increase gain for low light
        else:
            # For cameras 2 and 4, use auto settings
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        
        captures[idx] = cap
        print(f"✓ Opened")
    
    if not captures:
        print("\n❌ No cameras could be opened!")
        return
    
    # Let cameras warm up - especially important for Innomaker
    print("\nWarming up cameras (this may take a moment)...", end=" ")
    for i in range(50):  # Longer warmup for Innomaker
        for cap in captures.values():
            cap.read()
        time.sleep(0.05)
    print("Done!")
    
    print("\n" + "="*60)
    print("Capturing snapshots every 3 seconds...")
    print("Press Ctrl+C to stop")
    print("="*60)
    print("\n⚠️  If Camera 0 is still black:")
    print("  1. Check for physical lens cover/cap")
    print("  2. Ensure good lighting on the subject")
    print("  3. Camera might need manual focus adjustment")
    print("="*60)
    
    snapshot_count = 0
    
    try:
        while True:
            snapshot_count += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            print(f"\nSnapshot #{snapshot_count} at {timestamp}:")
            
            for idx, cap in captures.items():
                ret, frame = cap.read()
                
                if ret:
                    filename = f"{camera_names[idx]}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    
                    brightness = frame.mean()
                    status = "✓" if brightness > 30 else "⚠️  DARK"
                    
                    # Show more detail for dark images
                    if brightness < 10:
                        status = "❌ BLACK"
                    
                    print(f"  {status} {filename} (brightness: {brightness:.1f})")
                else:
                    print(f"  ❌ {camera_names[idx]} - Failed to capture")
            
            print(f"\nWaiting 3 seconds... (Press Ctrl+C to stop)")
            time.sleep(3)
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        # Cleanup
        print("\nClosing cameras...")
        for cap in captures.values():
            cap.release()
        print("Done!")
        print(f"\nTotal snapshots saved: {snapshot_count * len(captures)}")


if __name__ == "__main__":
    headless_camera_viewer()