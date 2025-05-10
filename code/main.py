import rclpy
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped

def main():
    rclpy.init()
    node = rclpy.create_node('ai_agent')

    def callback(msg):
        print("Received image frame")
        cmd = AckermannDriveStamped()
        cmd.drive.speed = 1.0
        cmd.drive.steering_angle = 0.0
        pub.publish(cmd)
    # This publisher sends control commands (speed, steering) to the CARLA simulator via ROS 2
    pub = node.create_publisher(AckermannDriveStamped, '/carla/ego_vehicle/ackermann_cmd', 10)

    # This subscriber receives camera image data published by CARLA via carla_ros_bridge
    sub = node.create_subscription(Image, '/carla/ego_vehicle/camera/rgb/image_raw', callback, 10)
    
    rclpy.spin(node)

if __name__ == '__main__':
    main()
