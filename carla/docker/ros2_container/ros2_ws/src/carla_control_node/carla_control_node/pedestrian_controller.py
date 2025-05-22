#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from carla_msgs.msg import CarlaWalkerControl
import math
import time
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32


class PedestrianController(Node):
    def __init__(self):
        super().__init__('pedestrian_controller')
        self.get_logger().info("Pedestrian Controller Node started")
        
        # Publisher for walker control commands
        self.walker_pub = self.create_publisher(
            CarlaWalkerControl,
            '/carla/pedestrian_1/walker_control_cmd',
            10
        )
        
        # Create timer to send movement commands
        self.create_timer(0.1, self.send_walker_command)
        
        # State variables
        self.direction = 0.0  # Direction in radians
        self.speed = 1.0      # Walking speed
        self.is_walking = True
        self.last_direction_change = time.time()
        self.direction_change_interval = 5.0  # Change direction every 5 seconds
        
    def send_walker_command(self):
        """Send control command to the walker"""
        try:
            # Check if we should change direction
            current_time = time.time()
            if current_time - self.last_direction_change > self.direction_change_interval:
                # Change direction randomly
                self.direction += 0.5
                if self.direction > 2 * math.pi:
                    self.direction = 0.0
                self.last_direction_change = current_time
                self.get_logger().info(f"Changing direction to {self.direction:.2f} radians")
            
            # Create the control message
            control_msg = CarlaWalkerControl()
            control_msg.direction.x = math.cos(self.direction)
            control_msg.direction.y = math.sin(self.direction)
            control_msg.direction.z = 0.0
            control_msg.speed = self.speed
            control_msg.jump = False
            
            # Publish the control message
            self.walker_pub.publish(control_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error sending walker command: {e}")
            import traceback
            traceback.print_exc()

def main(args=None):
    rclpy.init(args=args)
    node = PedestrianController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
