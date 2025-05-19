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
        
        # Initialize waypoint variables
        self.waypoints = []
        self.current_wp_index = 0
        self.vehicle = None
    
        # First spawn vehicle, then start following waypoints
        self.spawn_vehicle()
        if self.vehicle:
            self.generate_sidewalk_path()
            self.timer = self.create_timer(0.1, self.follow_waypoints)

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

    def generate_sidewalk_path(self):
        """Generate a path of waypoints on the sidewalk."""
        if not self.vehicle:
            self.get_logger().error("No vehicle to generate path for!")
            return
            
        # Get vehicle's current waypoint
        current_location = self.vehicle.get_location()
        current_wp = self.world.get_map().get_waypoint(
            current_location, 
            lane_type=carla.LaneType.Sidewalk,
            project_to_road=True
        )
        
        if not current_wp:
            self.get_logger().error("Could not find a sidewalk waypoint near the vehicle!")
            return
        
        # Generate 100 waypoints ahead, staying on sidewalk
        self.waypoints = []
        next_wp = current_wp
        
        for _ in range(100):
            # Get next waypoint, 2 meters ahead
            next_waypoints = next_wp.next(2.0)
            if not next_waypoints or next_waypoints[0].lane_type != carla.LaneType.Sidewalk:
                # Try to find another sidewalk nearby
                alternative_wps = next_wp.get_right_lane()
                if alternative_wps and alternative_wps.lane_type == carla.LaneType.Sidewalk:
                    next_wp = alternative_wps
                else:
                    # No more sidewalk ahead, end the path
                    break
            else:
                next_wp = next_waypoints[0]
                
            self.waypoints.append(next_wp)
        
        self.get_logger().info(f"Generated {len(self.waypoints)} sidewalk waypoints")
        self.current_wp_index = 0


    def follow_waypoints(self):
        """Follow the generated sidewalk waypoints."""
        if not self.vehicle or not self.waypoints:
            return
            
        if self.current_wp_index >= len(self.waypoints):
            self.get_logger().info("Reached the end of the waypoints.")
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0))
            return
        
        target_wp = self.waypoints[self.current_wp_index]
        vehicle_loc = self.vehicle.get_location()
        target_loc = target_wp.transform.location

        # Calculate direction vector to target
        direction = carla.Vector3D(
            x=target_loc.x - vehicle_loc.x,
            y=target_loc.y - vehicle_loc.y,
            z=0
        )
        
        # Normalize the direction vector
        distance = direction.length()
        if distance > 0.1:
            direction.x /= distance
            direction.y /= distance
        
        # Get vehicle's forward vector
        forward = self.vehicle.get_transform().get_forward_vector()
        
        # Calculate steering (dot product and cross product)
        dot = forward.x * direction.x + forward.y * direction.y
        cross = forward.x * direction.y - forward.y * direction.x
        
        # Convert to steering value (-1 to 1)
        steer = min(max(cross * 2.0, -1.0), 1.0)
        
        # Adjust speed based on steering angle and distance
        throttle = 0.5
        if abs(steer) > 0.2:
            throttle = 0.3  # Slow down for turns
        
        # Move to next waypoint when close enough
        if distance < 1.0:
            self.current_wp_index += 1
            self.get_logger().info(f"Reached waypoint {self.current_wp_index-1}, moving to next one")
        
        # Apply control
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        control.brake = 0.0
        
        self.vehicle.apply_control(control)



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