import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrow, Circle
import numpy as np
from pathlib import Path
from path_planner import PathPlanner
from navigation_controller import NavigationController
from tile_basemap import TileBasemap


class PathSimulator:
    """
    Visualize robot navigation simulation on the planned path.
    Shows a moving dot following the coverage path with real-time navigation commands.
    """
    
    def __init__(self, planner, coverage_path, area_polygon=None, roads_in_area=None):
        """
        Initialize path simulator.
        
        Args:
            planner: PathPlanner instance with road network
            coverage_path: List of waypoints to follow
            area_polygon: Selected area polygon (optional)
            roads_in_area: GeoDataFrame of roads in area (optional)
        """
        self.planner = planner
        self.coverage_path = coverage_path
        self.area_polygon = area_polygon
        self.roads_in_area = roads_in_area
        
        # Load Google Maps tiles
        self.tile_basemap = TileBasemap()
        
        # Create navigation controller
        self.nav_controller = NavigationController(coverage_path)
        
        # Simulation state
        self.current_position = coverage_path[0]
        self.current_heading = 0
        self.speed = 0.00002  # Speed in coordinate units per frame (slower for smoother animation)
        self.current_waypoint_idx = 0
        
        # Animation objects
        self.fig = None
        self.ax = None
        self.robot_dot = None
        self.heading_arrow = None
        self.status_text = None
        
    def setup_plot(self):
        """Setup the matplotlib figure and axes."""
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        
        # Add Google Maps satellite basemap first
        if self.tile_basemap.tiles:
            print(f"Adding {len(self.tile_basemap.tiles)} satellite tiles to simulation...")
            self.tile_basemap.add_to_plot(self.ax, alpha=0.7)
        else:
            print("No satellite tiles found. Run download_map_tiles.py first.")
        
        # Plot full road network in background (if available)
        if self.planner is not None:
            try:
                self.planner.gdf.plot(ax=self.ax, color='lightgray', linewidth=0.5, alpha=0.3)
            except:
                print("Could not plot road network")
        
        # Plot roads in selected area
        if self.roads_in_area is not None and len(self.roads_in_area) > 0:
            self.roads_in_area.plot(ax=self.ax, color='navy', linewidth=1.5, alpha=0.4)
        
        # Plot selected area boundary
        if self.area_polygon:
            x, y = self.area_polygon.exterior.xy
            self.ax.plot(x, y, 'r--', linewidth=2, label='Selected Area', alpha=0.5)
            self.ax.fill(x, y, color='red', alpha=0.05)
        
        # Plot the full path - MAKE IT MORE VISIBLE
        if len(self.coverage_path) > 1:
            path_x = [p[0] for p in self.coverage_path]
            path_y = [p[1] for p in self.coverage_path]
            
            # Plot path with thicker line
            self.ax.plot(path_x, path_y, 'b-', linewidth=3, label='Planned Path', 
                        alpha=0.7, zorder=3)
            
            # Plot waypoints as small dots
            self.ax.plot(path_x, path_y, 'c.', markersize=4, alpha=0.5, zorder=3)
            
            # Mark start and end - LARGER
            self.ax.plot(path_x[0], path_y[0], 'go', markersize=20, 
                        label='Start', zorder=4, markeredgecolor='darkgreen', markeredgewidth=3)
            self.ax.plot(path_x[-1], path_y[-1], 'rs', markersize=20, 
                        label='Goal', zorder=4, markeredgecolor='darkred', markeredgewidth=3)
        
        # Create robot marker - LARGER AND MORE VISIBLE
        self.robot_dot = Circle((0, 0), radius=0.00002, color='red', 
                               zorder=10, label='Robot', visible=False,
                               edgecolor='darkred', linewidth=2)
        self.ax.add_patch(self.robot_dot)
        
        # Create heading arrow (initially hidden)
        self.heading_arrow = None
        
        # Status text - LARGER
        self.status_text = self.ax.text(0.02, 0.02, '', transform=self.ax.transAxes,
                                       fontsize=14, verticalalignment='bottom',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.95),
                                       zorder=11, fontweight='bold')
        
        self.ax.set_xlabel('Longitude', fontsize=12)
        self.ax.set_ylabel('Latitude', fontsize=12)
        self.ax.set_title('Robot Navigation Simulation', fontsize=16, fontweight='bold')
        self.ax.legend(loc='upper right', fontsize=11)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set axis limits based on path
        if len(self.coverage_path) > 1:
            path_x = [p[0] for p in self.coverage_path]
            path_y = [p[1] for p in self.coverage_path]
            margin_x = (max(path_x) - min(path_x)) * 0.1
            margin_y = (max(path_y) - min(path_y)) * 0.1
            self.ax.set_xlim(min(path_x) - margin_x, max(path_x) + margin_x)
            self.ax.set_ylim(min(path_y) - margin_y, max(path_y) + margin_y)
        
        plt.tight_layout()
    
    def calculate_heading_to_target(self, current, target):
        """Calculate heading from current position to target."""
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        heading = np.arctan2(dx, dy)
        return np.degrees(heading) % 360
    
    def move_towards_target(self, current, target, speed):
        """Move from current position towards target at given speed."""
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance <= speed:
            # Reached target
            return target
        else:
            # Move towards target
            ratio = speed / distance
            new_x = current[0] + dx * ratio
            new_y = current[1] + dy * ratio
            return (new_x, new_y)
    
    def init_animation(self):
        """Initialize animation."""
        self.robot_dot.set_visible(True)
        self.robot_dot.center = self.coverage_path[0]
        self.current_position = self.coverage_path[0]
        self.current_waypoint_idx = 0
    
    def update_frame(self, frame):
        """Update animation frame."""
        if self.current_waypoint_idx >= len(self.coverage_path):
            # Animation complete
            self.status_text.set_text('NAVIGATION COMPLETE! ✓')
            return
        
        # Get target waypoint
        target = self.coverage_path[self.current_waypoint_idx]
        
        # Move towards target
        new_position = self.move_towards_target(self.current_position, target, self.speed)
        
        # Check if reached target
        distance_to_target = np.sqrt(
            (new_position[0] - target[0])**2 + 
            (new_position[1] - target[1])**2
        )
        
        if distance_to_target < self.speed:
            # Reached waypoint, move to next
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx < len(self.coverage_path):
                new_position = target
        
        # Update robot position
        self.current_position = new_position
        self.robot_dot.center = new_position
        
        # Update heading
        if self.current_waypoint_idx < len(self.coverage_path):
            next_target = self.coverage_path[self.current_waypoint_idx]
            self.current_heading = self.calculate_heading_to_target(
                self.current_position, next_target
            )
        
        # Get navigation command
        nav_cmd = self.nav_controller.get_direction_command(
            self.current_position, 
            self.current_heading
        )
        
        # Update heading arrow
        if self.heading_arrow:
            self.heading_arrow.remove()
        
        # Draw heading arrow - LARGER AND MORE VISIBLE
        arrow_length = 0.0001
        dx = arrow_length * np.sin(np.radians(self.current_heading))
        dy = arrow_length * np.cos(np.radians(self.current_heading))
        
        self.heading_arrow = FancyArrow(
            self.current_position[0], self.current_position[1],
            dx, dy,
            width=0.00003, head_width=0.00007, head_length=0.0001,
            fc='yellow', ec='orange', zorder=9, linewidth=2
        )
        self.ax.add_patch(self.heading_arrow)
        
        # Update status text
        command = nav_cmd.get('command', 'NAVIGATE')
        progress = nav_cmd.get('progress_percent', 0)
        waypoint_num = nav_cmd.get('current_waypoint_index', 0)
        total_waypoints = len(self.coverage_path)
        
        # Get command emoji
        command_emoji = {
            'STRAIGHT': '^',
            'LEFT': '<',
            'RIGHT': '>',
            'ARRIVED': '*',
            'COMPLETE': 'X',
            'NAVIGATE': 'o'
        }.get(command, '->')
        
        status = (
            f"{command_emoji} Command: {command}\n"
            f"Waypoint: {waypoint_num}/{total_waypoints}\n"
            f"Progress: {progress:.1f}%\n"
            f"Heading: {self.current_heading:.1f}deg"
        )
        
        if 'turn_angle' in nav_cmd:
            status += f"\nTurn: {nav_cmd['turn_angle']:.1f}deg"
        
        self.status_text.set_text(status)
    
    def run(self, save_animation=False, filename='navigation_simulation.gif'):
        """
        Run the simulation animation.
        
        Args:
            save_animation: Whether to save animation as GIF
            filename: Output filename for animation
        """
        self.setup_plot()
        
        # Calculate number of frames needed
        # Limit frames to reasonable number
        print(f"\nStarting simulation...")
        print(f"Total waypoints: {len(self.coverage_path)}")
        print(f"Animation speed: {self.speed} units/frame")
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, 
            self.update_frame,
            init_func=self.init_animation,
            frames=2000,  # Fixed number of frames
            interval=30,  # 30ms between frames
            blit=False,
            repeat=True
        )
        
        if save_animation:
            print(f"\nSaving animation to {filename}...")
            print("This may take a while...")
            anim.save(filename, writer='pillow', fps=20)
            print(f"Animation saved!")
        
        plt.show()
        
        return anim


def main():
    """Run simulation on the most recent path planning result."""
    print("="*70)
    print("ROBOT NAVIGATION SIMULATOR")
    print("="*70)
    
    # Path to the shapefile
    shapefile_path = Path(__file__).parent / "data/tile2net_output_SMUS/Victoria/network/Victoria-Network-18-01-2026_23_08/Victoria-Network-18-01-2026_23_08.shp"
    
    # Check if waypoints file exists
    waypoints_file = Path(__file__).parent / "waypoints.txt"
    
    if waypoints_file.exists():
        print("\nLoading path from waypoints.txt...")
        # Load waypoints from file
        coverage_path = []
        with open(waypoints_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split(',')
                if len(parts) == 3:
                    lon, lat = float(parts[1]), float(parts[2])
                    coverage_path.append((lon, lat))
        
        print(f"Loaded {len(coverage_path)} waypoints")
        
        # Load planner for road network (optional - just for context)
        try:
            planner = PathPlanner(shapefile_path)
        except Exception as e:
            print(f"Could not load road network: {e}")
            print("Continuing without road network visualization...")
            planner = None
        
        # Create and run simulator
        simulator = PathSimulator(planner, coverage_path) if planner else PathSimulator(None, coverage_path)
        
    else:
        print("\nNo waypoints.txt found. Please run main.py first to generate a path.")
        print("Or create a path manually...")
        return
    
    print("\n" + "="*70)
    print("Starting simulation...")
    print("="*70)
    print("\nControls:")
    print("  - Close window to stop simulation")
    print("  - Watch the red dot follow the blue path")
    print("  - Yellow arrow shows robot heading")
    print("  - Status box shows navigation commands")
    print("\n" + "="*70)
    
    # Run simulation
    simulator.run(save_animation=False)  # Set True to save as GIF


if __name__ == "__main__":
    main()
