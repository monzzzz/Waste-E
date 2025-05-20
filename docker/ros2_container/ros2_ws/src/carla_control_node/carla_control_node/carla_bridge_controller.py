import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import carla
import numpy as np
import cv2
from cv_bridge import CvBridge
import time
import subprocess


class CarlaBridgeController(Node):
    def __init__(self):
        super().__init__('carla_bridge_controller')
        self.get_logger().info("CARLA Bridge Controller Node started - DEBUG MODE")
        
        # Create republisher for camera images
        self.image_pub = self.create_publisher(
            Image, 
            '/custom/pedestrian/camera/image', 
            10
        )
        
        # Set up bridge for image conversion
        self.bridge = CvBridge()
        
        # Track if any images have been received
        self.images_received = 0
        self.last_image_time = time.time()
        
        # Subscribe to the camera topic with consistent QoS settings
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.subscription = self.create_subscription(
            Image,
            '/carla/pedestrian_1/pedestrian_camera/image',
            self.image_callback,
            qos_profile=sensor_qos
        )
        
        # Add a test image publisher
        # self.create_timer(1.0, self.publish_test_image)
        
        # Add multiple diagnostics
        self.create_timer(5.0, self.check_topics)
        self.create_timer(10.0, self.check_actors)
        self.create_timer(15.0, self.check_carla_status)

    def image_callback(self, image):
        try:
            # Count received images and update timestamp
            self.images_received += 1
            self.last_image_time = time.time()
            
            # Detailed logging of received image
            self.get_logger().info(f"RECEIVED IMAGE #{self.images_received}: " +
                                  f"size={image.width}x{image.height}, " +
                                  f"encoding={image.encoding}, " +
                                  f"frame_id={image.header.frame_id}")
            
            # Update frame ID and publish
            image.header.frame_id = "map"
            image.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(image)
            
            # Log successful republishing
            self.get_logger().info(f"Successfully republished image #{self.images_received}")
            
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")
            import traceback
            traceback.print_exc()

    def check_topics(self):
        """Check camera-related topics and publishers"""
        try:
            # Log all published topics
            result = subprocess.run(["ros2", "topic", "list"], capture_output=True, text=True)
            topics = result.stdout.strip().split('\n')
            
            # Check for camera topics
            camera_topics = [t for t in topics if "image" in t or "camera" in t]
            self.get_logger().info(f"Available camera topics: {camera_topics}")
            
            # If source topic exists, get detailed info
            if '/carla/pedestrian_camera/image' in topics:
                self.get_logger().info("Source camera topic found, checking details...")
                
                # Get publisher count
                info_cmd = ["ros2", "topic", "info", "/carla/pedestrian_camera/image"]
                info_result = subprocess.run(info_cmd, capture_output=True, text=True)
                self.get_logger().info(f"Topic info: {info_result.stdout}")
                
                # Log image reception status
                elapsed = time.time() - self.last_image_time
                if self.images_received > 0:
                    self.get_logger().info(f"Received {self.images_received} images, last one {elapsed:.1f}s ago")
                else:
                    self.get_logger().warn(f"NO IMAGES RECEIVED from source topic yet!")
            else:
                self.get_logger().warn("Source camera topic NOT FOUND!")
                
        except Exception as e:
            self.get_logger().error(f"Error checking topics: {e}")

    def check_actors(self):
        """Check if pedestrian and camera actors exist in CARLA"""
        try:
            # Check CARLA actor list 
            actor_cmd = ["ros2", "topic", "echo", "/carla/actor_list", "--once"]
            actor_result = subprocess.run(actor_cmd, capture_output=True, text=True, timeout=2.0)
            
            # Look for pedestrian in actor list
            if "pedestrian" in actor_result.stdout.lower():
                self.get_logger().info("Pedestrian actor found in CARLA!")
            else:
                self.get_logger().warn("NO PEDESTRIAN found in CARLA actor list!")
                
            # Check for camera sensor
            if "camera" in actor_result.stdout.lower() and "pedestrian" in actor_result.stdout.lower():
                self.get_logger().info("Pedestrian camera sensor found in CARLA!")
            else:
                self.get_logger().warn("NO PEDESTRIAN CAMERA found in CARLA actor list!")
                
        except Exception as e:
            self.get_logger().error(f"Error checking CARLA actors: {e}")

    def check_carla_status(self):
        """Check if CARLA is running properly"""
        try:
            # Check if CARLA is publishing status info
            status_cmd = ["ros2", "topic", "echo", "/carla/status", "--once"]
            status_result = subprocess.run(status_cmd, capture_output=True, text=True, timeout=1.0)
            
            if status_result.stdout.strip():
                self.get_logger().info(f"CARLA status active: {status_result.stdout[:100]}...")
            else:
                self.get_logger().warn("NO CARLA STATUS available - bridge may not be connected!")
                
        except Exception as e:
            self.get_logger().error(f"Error checking CARLA status: {e}")

    def publish_test_image(self):
        """Publish a test image to verify RViz configuration"""
        try:
            # Only publish test image if no real images received recently
            if self.images_received == 0 or (time.time() - self.last_image_time) > 5.0:
                # Create test image
                test_img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray
                
                # Add rectangle
                cv2.rectangle(test_img, (50, 50), (590, 430), (0, 0, 255), 5)
                
                # Add text with current time
                timestamp = time.time()
                cv2.putText(test_img, "TEST IMAGE - NO REAL CAMERA DATA", 
                           (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(test_img, f"Time: {timestamp:.1f}", 
                           (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                # Convert to ROS Image
                ros_img = self.bridge.cv2_to_imgmsg(test_img, encoding="bgr8")
                ros_img.header.stamp = self.get_clock().now().to_msg()
                ros_img.header.frame_id = "map"
                
                # Publish test image
                self.image_pub.publish(ros_img)
                self.get_logger().info("Published test image (no real camera data available)")
        except Exception as e:
            self.get_logger().error(f"Error publishing test image: {e}")    

def main(args=None):
    rclpy.init(args=args)
    node = CarlaBridgeController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
