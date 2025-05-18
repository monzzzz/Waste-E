import rclpy
from rclpy.node import Node
from carla_msgs.msg import CarlaEgoVehicleControl
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseWithCovarianceStamped

class EnhancedControlNode(Node):
    def __init__(self):
        super().__init__('enhanced_control_node')
        
        # Publisher for vehicle control commands
        self.publisher_ = self.create_publisher(
            CarlaEgoVehicleControl, 
            '/carla/ego_vehicle/vehicle_control_cmd', 
            10
        )
        
        # Subscribe to vehicle position
        self.position_subscription = self.create_subscription(
            NavSatFix,
            '/carla/ego_vehicle/gnss',
            self.position_callback,
            10
        )
        
        # Simple control loop
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Initialize control state
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.reverse = False
        
        self.get_logger().info("Enhanced Control Node started")

    def position_callback(self, msg):
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude
        self.get_logger().info(f"Current position: {self.current_lat}, {self.current_lon}")

    def timer_callback(self):
        # Create control message
        msg = CarlaEgoVehicleControl()
        msg.steer = self.steering
        msg.throttle = self.throttle
        msg.brake = self.brake
        msg.reverse = self.reverse
        
        # Publish control command
        self.publisher_.publish(msg)
        
        # Log current control state
        self.get_logger().info(f"Control: steer={msg.steer:.2f}, throttle={msg.throttle:.2f}, brake={msg.brake:.2f}")

    def drive_forward(self):
        self.steering = 0.0
        self.throttle = 0.5
        self.brake = 0.0
        self.reverse = False

    def turn_left(self):
        self.steering = -0.5
        self.throttle = 0.3
        self.brake = 0.0
        self.reverse = False

    def turn_right(self):
        self.steering = 0.5
        self.throttle = 0.3
        self.brake = 0.0
        self.reverse = False

    def stop(self):
        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 1.0
        self.reverse = False

def main(args=None):
    rclpy.init(args=args)
    control_node = EnhancedControlNode()
    
    # Example: Drive forward for a while
    control_node.drive_forward()
    
    rclpy.spin(control_node)
    control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()