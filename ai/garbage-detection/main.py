import sys
import os
import cv2
import argparse
from detection import GarbageDetection

def main():
    # Example Usage: python main.py --front data/front.jpg --left data/left.jpg --right data/right.jpg
    
    # 1. Define Image Paths (You can change these defaults)
    parser = argparse.ArgumentParser(description='Waste Detection Server Test')
    parser.add_argument('--front', type=str, default='data/front.jpg', help='Path to front camera image')
    parser.add_argument('--left', type=str, default='data/left.jpg', help='Path to left camera image')
    parser.add_argument('--right', type=str, default='data/right.jpg', help='Path to right camera image')
    args = parser.parse_args()

    # 2. Check if files exist
    print(f"Checking inputs:\n Front: {args.front}\n Left: {args.left}\n Right: {args.right}")
    
    # 3. Load Images
    front_img = cv2.imread(args.front) if os.path.exists(args.front) else None
    left_img = cv2.imread(args.left) if os.path.exists(args.left) else None
    right_img = cv2.imread(args.right) if os.path.exists(args.right) else None

    # Warning for missing images
    if front_img is None: print(f"Warning: Front image not found at {args.front}")
    if left_img is None: print(f"Warning: Left image not found at {args.left}")
    if right_img is None: print(f"Warning: Right image not found at {args.right}")

    # 4. Initialize Detector
    detector = GarbageDetection()

    # 5. Get Motor Commands
    result = detector.get_motor_speeds(front_img, left_img, right_img)

    # 6. Output Result
    print("-" * 50)
    print("MOTOR COMMANDS RESULT:")
    print(f"Left Motor:  {result['left']}")
    print(f"Right Motor: {result['right']}")
    print(f"Status:      {result['status']}")
    print("-" * 50)

if __name__ == "__main__":
    main()
