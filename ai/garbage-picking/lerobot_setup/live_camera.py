"""
Headless camera viewer - saves periodic snapshots without GUI.
"""
import cv2
import time
import os

def headless_camera_viewer():
    """Save periodic snapshots from all cameras."""
    
    # Your available cameras
    camera_indices = [0, 2, 4]
    camera_names = {0: "Camera_0", 2: "Camera_2", 4: "Camera_4"}
    captures = {}
    
    print("\n" + "="*60)
    print("HEADLESS CAMERA VIEWER")
    print("="*60)
    print("\nOpening cameras...\n")
    
    # Open all cameras
    for idx in camera_indices:
        print(f"Opening camera {idx}...", end=" ")
        cap = cv2.VideoCapture(idx)
        
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
        
        # Enable autofocus and auto exposure
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        
        captures[idx] = cap
        print(f"✓ Opened")
    
    if not captures:
        print("\n❌ No cameras could be opened!")
        return
    
    # Let cameras warm up
    print("\nWarming up cameras...", end=" ")
    for i in range(30):
        for cap in captures.values():
            cap.read()
    print("Done!")
    
    print("\n" + "="*60)
    print("Capturing snapshots every 3 seconds...")
    print("Press Ctrl+C to stop")
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
                    status = "✓" if brightness > 30 else "⚠️ "
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