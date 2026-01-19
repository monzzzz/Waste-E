import numpy as np
from shapely.geometry import Point


class NavigationController:
    """
    Navigation controller for robot path following.
    Provides turn-by-turn directions based on current location.
    """
    
    def __init__(self, waypoints):
        """
        Initialize navigation controller with a list of waypoints.
        
        Args:
            waypoints: List of (x, y) coordinate tuples representing the path
        """
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.total_waypoints = len(waypoints)
        self.is_complete = False
        
        print(f"Navigation initialized with {self.total_waypoints} waypoints")
    
    def get_current_waypoint(self):
        """Get the current target waypoint."""
        if self.is_complete:
            return None
        return self.waypoints[self.current_waypoint_index]
    
    def get_next_waypoint(self):
        """Get the next waypoint after current."""
        if self.current_waypoint_index + 1 < self.total_waypoints:
            return self.waypoints[self.current_waypoint_index + 1]
        return None
    
    def calculate_bearing(self, point1, point2):
        """
        Calculate bearing from point1 to point2 in degrees.
        0° = North, 90° = East, 180° = South, 270° = West
        
        Args:
            point1: (x, y) starting point
            point2: (x, y) ending point
            
        Returns:
            Bearing in degrees (0-360)
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        
        # Calculate angle in radians, then convert to degrees
        angle = np.arctan2(dx, dy)  # Note: dx, dy order for North=0
        bearing = np.degrees(angle)
        
        # Normalize to 0-360
        if bearing < 0:
            bearing += 360
        
        return bearing
    
    def calculate_turn_angle(self, bearing1, bearing2):
        """
        Calculate turn angle from bearing1 to bearing2.
        Positive = right turn, Negative = left turn
        
        Args:
            bearing1: Current bearing in degrees
            bearing2: Target bearing in degrees
            
        Returns:
            Turn angle in degrees (-180 to 180)
        """
        diff = bearing2 - bearing1
        
        # Normalize to -180 to 180
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        
        return diff
    
    def get_direction_command(self, current_location, current_heading=None):
        """
        Get navigation command based on current location and heading.
        
        Args:
            current_location: (x, y) current robot position
            current_heading: Current heading in degrees (optional)
            
        Returns:
            Dictionary with navigation information:
            {
                'command': 'STRAIGHT' | 'LEFT' | 'RIGHT' | 'ARRIVED' | 'COMPLETE',
                'distance_to_waypoint': float,
                'bearing_to_waypoint': float,
                'turn_angle': float (if current_heading provided),
                'current_waypoint_index': int,
                'progress_percent': float
            }
        """
        if self.is_complete:
            return {
                'command': 'COMPLETE',
                'message': 'Navigation complete!',
                'progress_percent': 100.0
            }
        
        current_wp = self.get_current_waypoint()
        
        # Calculate distance to current waypoint
        distance = Point(current_location).distance(Point(current_wp))
        
        # Calculate bearing to current waypoint
        bearing_to_wp = self.calculate_bearing(current_location, current_wp)
        
        # Check if we've reached the current waypoint (threshold: 0.0001 units ~11 meters)
        threshold = 0.0001
        if distance < threshold:
            # Move to next waypoint
            self.current_waypoint_index += 1
            
            if self.current_waypoint_index >= self.total_waypoints:
                self.is_complete = True
                return {
                    'command': 'COMPLETE',
                    'message': 'Navigation complete! All waypoints reached.',
                    'progress_percent': 100.0
                }
            
            # Arrived at waypoint, but more waypoints ahead
            next_wp = self.get_current_waypoint()
            next_distance = Point(current_location).distance(Point(next_wp))
            next_bearing = self.calculate_bearing(current_location, next_wp)
            
            result = {
                'command': 'ARRIVED',
                'message': f'Reached waypoint {self.current_waypoint_index - 1}',
                'distance_to_waypoint': next_distance,
                'bearing_to_waypoint': next_bearing,
                'current_waypoint_index': self.current_waypoint_index,
                'progress_percent': (self.current_waypoint_index / self.total_waypoints) * 100
            }
            
            if current_heading is not None:
                turn_angle = self.calculate_turn_angle(current_heading, next_bearing)
                result['turn_angle'] = turn_angle
                result['command'] = self._get_turn_command(turn_angle)
            
            return result
        
        # Still heading to current waypoint
        result = {
            'distance_to_waypoint': distance,
            'bearing_to_waypoint': bearing_to_wp,
            'current_waypoint_index': self.current_waypoint_index,
            'progress_percent': (self.current_waypoint_index / self.total_waypoints) * 100,
            'waypoint_coordinates': current_wp
        }
        
        if current_heading is not None:
            # Calculate turn needed
            turn_angle = self.calculate_turn_angle(current_heading, bearing_to_wp)
            result['turn_angle'] = turn_angle
            result['command'] = self._get_turn_command(turn_angle)
        else:
            # No heading provided, just give bearing
            result['command'] = 'NAVIGATE'
            result['message'] = f'Navigate to bearing {bearing_to_wp:.1f}°'
        
        return result
    
    def _get_turn_command(self, turn_angle):
        """
        Convert turn angle to simple command.
        
        Args:
            turn_angle: Angle in degrees (-180 to 180)
            
        Returns:
            'STRAIGHT', 'LEFT', or 'RIGHT'
        """
        # Threshold for "straight" (within ±15 degrees)
        if abs(turn_angle) < 15:
            return 'STRAIGHT'
        elif turn_angle < 0:
            return 'LEFT'
        else:
            return 'RIGHT'
    
    def reset(self):
        """Reset navigation to start from beginning."""
        self.current_waypoint_index = 0
        self.is_complete = False
        print("Navigation reset to start")
    
    def get_status(self):
        """Get current navigation status."""
        return {
            'current_waypoint_index': self.current_waypoint_index,
            'total_waypoints': self.total_waypoints,
            'progress_percent': (self.current_waypoint_index / self.total_waypoints) * 100,
            'is_complete': self.is_complete,
            'waypoints_remaining': self.total_waypoints - self.current_waypoint_index
        }
    
    def simulate_navigation(self):
        """
        Simulate navigation through all waypoints.
        Useful for testing.
        """
        print("\n" + "="*70)
        print("NAVIGATION SIMULATION")
        print("="*70)
        
        for i in range(len(self.waypoints) - 1):
            current_pos = self.waypoints[i]
            
            # Simulate heading as the direction to next waypoint
            if i + 1 < len(self.waypoints):
                next_pos = self.waypoints[i + 1]
                heading = self.calculate_bearing(current_pos, next_pos)
            else:
                heading = 0
            
            cmd = self.get_direction_command(current_pos, heading)
            
            print(f"\nWaypoint {i} -> {i+1}:")
            print(f"  Position: ({current_pos[0]:.6f}, {current_pos[1]:.6f})")
            print(f"  Command: {cmd['command']}")
            print(f"  Progress: {cmd['progress_percent']:.1f}%")
            if 'turn_angle' in cmd:
                print(f"  Turn angle: {cmd['turn_angle']:.1f}°")
            if 'distance_to_waypoint' in cmd:
                print(f"  Distance to next: {cmd['distance_to_waypoint']:.6f}")
        
        print("\n" + "="*70)
        print("SIMULATION COMPLETE")
        print("="*70)


def main():
    """Example usage of NavigationController."""
    # Example waypoints (replace with actual path)
    example_waypoints = [
        (-123.370000, 48.420000),
        (-123.369000, 48.421000),
        (-123.368000, 48.421500),
        (-123.367000, 48.422000),
        (-123.366000, 48.421500),
        (-123.365000, 48.421000),
    ]
    
    # Create navigation controller
    nav = NavigationController(example_waypoints)
    
    # Example 1: Get direction at starting point
    print("\n--- Example 1: At starting point ---")
    current_pos = (-123.370000, 48.420000)
    current_heading = 45  # degrees (northeast)
    cmd = nav.get_direction_command(current_pos, current_heading)
    print(f"Command: {cmd['command']}")
    print(f"Turn angle: {cmd.get('turn_angle', 'N/A')}°")
    print(f"Distance: {cmd.get('distance_to_waypoint', 'N/A')}")
    
    # Example 2: Simulate full navigation
    nav.reset()
    nav.simulate_navigation()
    
    print("\n--- Usage in your robot code ---")
    print("# Initialize with your path")
    print("nav = NavigationController(coverage_path)")
    print("")
    print("# In your robot control loop:")
    print("while not nav.is_complete:")
    print("    current_pos = get_robot_position()  # Your function")
    print("    current_heading = get_robot_heading()  # Your function")
    print("    cmd = nav.get_direction_command(current_pos, current_heading)")
    print("    ")
    print("    if cmd['command'] == 'STRAIGHT':")
    print("        robot_go_straight()")
    print("    elif cmd['command'] == 'LEFT':")
    print("        robot_turn_left()")
    print("    elif cmd['command'] == 'RIGHT':")
    print("        robot_turn_right()")


if __name__ == "__main__":
    main()
