#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
import subprocess

class MapVisualizer(Node):
    def __init__(self):
        super().__init__('map_visualizer')
        self.get_logger().info("Map Visualizer Node started")
        
        # Create a subscription to CARLA's map visualization
        self.create_subscription(
            MarkerArray,
            '/carla/map',
            self.map_callback,
            10
        )
        
        # Create a timer to check if map data is available
        self.create_timer(5.0, self.check_map_topic)
        
    def map_callback(self, marker_array):
        num_markers = len(marker_array.markers)
        self.get_logger().info(f"Received map data with {num_markers} markers")
        
    def check_map_topic(self):
        """Check if map data is available"""
        try:
            cmd = ["ros2", "topic", "info", "/carla/map"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.get_logger().info(f"Map topic info: {result.stdout}")
            
            # Check if map service is available
            cmd2 = ["ros2", "service", "list"]
            result2 = subprocess.run(cmd2, capture_output=True, text=True)
            map_services = [s for s in result2.stdout.split('\n') if 'map' in s]
            self.get_logger().info(f"Map-related services: {map_services}")
            
        except Exception as e:
            self.get_logger().error(f"Error checking map topic: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = MapVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()