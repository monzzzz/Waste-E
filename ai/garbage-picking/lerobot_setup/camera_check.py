"""
Simple camera snapshot tool - takes one picture from each camera with better settings.
"""
import cv2
import time

def take_snapshots():
    """Take a snapshot from each available camera."""
    
    # Your available cameras
    camera_indices = [0, 2, 4]
    camera_names = {0: "camera_0", 2: "camera_2", 4: "camera_4"}
    
    print("\n" + "="*60)
    print("CAMERA SNAPSHOT TOOL")
    print("="*60)
    print("\nTaking snapshots from all cameras...\n")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    for idx in camera_indices:
        print(f"Opening camera {idx}...", end=" ")
        cap = cv2.VideoCapture(idx)
        
        if not cap.isOpened():
            print(f"❌ Failed to open")
            continue
        
        # Set resolution (might help with black image)
        if idx == 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Enable autofocus
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        # Set auto exposure
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Auto mode
        
        print("warming up...", end=" ")
        
        # Let camera warm up and autofocus - read and discard several frames
        for i in range(30):  # Read 30 frames to let camera adjust
            cap.read()
            time.sleep(0.1)
        
        # Now capture the actual frame
        ret, frame = cap.read()
        
        if ret and frame is not None:
            # Check if image is too dark
            mean_brightness = frame.mean()
            if mean_brightness < 10:
                print(f"⚠️  Warning: Very dark image (brightness: {mean_brightness:.1f})")
            
            filename = f"{camera_names[idx]}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Saved as {filename} (brightness: {mean_brightness:.1f})")
        else:
            print(f"❌ Failed to capture frame")
        
        cap.release()
    
    print("\n" + "="*60)
    print("Done! Check the .jpg files to identify which camera is which.")
    print("="*60)
    print("\nTroubleshooting:")
    print("  - Black image: Check if camera lens is covered or blocked")
    print("  - Blurry image: Camera might need better lighting or manual focus")
    print("  - Try running the script again - cameras sometimes need time to initialize")


if __name__ == "__main__":
    take_snapshots()