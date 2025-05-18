import rclpy
from rclpy.node import Node
import carla
import random
import time

class VehicleSpawner(Node):
    def __init__(self):
        super().__init__('vehicle_spawner')
        self.get_logger().info("Vehicle Spawner Node started")
        
        # Connect to CARLA server
        self.client = carla.Client('carla_server', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        
        # Timer for spawn logic
        self.timer = self.create_timer(1.0, self.spawn_vehicle)
        self.vehicle = None

    def spawn_vehicle(self):
        if self.vehicle is not None:
            self.get_logger().info("Vehicle already spawned")
            self.timer.cancel()  # Stop the timer
            return
            
        # Get footpath waypoints (sidewalks in CARLA)
        footpath_waypoints = []
        for waypoint in self.world.get_map().generate_waypoints(2.0):
            if waypoint.lane_type == carla.LaneType.Sidewalk:
                footpath_waypoints.append(waypoint)
        
        if not footpath_waypoints:
            self.get_logger().error("No footpath waypoints found!")
            return
            
        # Choose a random footpath waypoint
        waypoint = random.choice(footpath_waypoints)
        
        # Create spawn point
        spawn_point = waypoint.transform
        spawn_point.location.z += 0.5  # Raise slightly to avoid collisions
        
        # Find a small car blueprint (models like Micro, Mini, etc.)
        small_car_blueprints = [bp for bp in self.blueprint_library.filter('vehicle.*') 
                             if 'mini' in bp.id.lower() or 'micro' in bp.id.lower() 
                             or 'isetta' in bp.id.lower() or 'carlacola' in bp.id.lower()]
        
        if not small_car_blueprints:
            # If no specific small cars found, use any small vehicle
            small_car_blueprints = [bp for bp in self.blueprint_library.filter('vehicle.micro')]
            
        if not small_car_blueprints:
            # Fallback to any vehicle
            small_car_blueprints = list(self.blueprint_library.filter('vehicle'))
        
        vehicle_bp = random.choice(small_car_blueprints)
        
        # Try to spawn the vehicle
        try:
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.get_logger().info(f"Spawned {vehicle_bp.id} on footpath at {spawn_point.location}")
            
            # Set it to autopilot (optional)
            # self.vehicle.set_autopilot(True)
        except Exception as e:
            self.get_logger().error(f"Failed to spawn vehicle: {e}")

def main(args=None):
    rclpy.init(args=args)
    spawner = VehicleSpawner()
    rclpy.spin(spawner)
    
    # Cleanup
    if spawner.vehicle is not None:
        spawner.vehicle.destroy()
    spawner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()