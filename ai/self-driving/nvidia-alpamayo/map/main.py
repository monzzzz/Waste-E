from pathlib import Path
from path_planner import PathPlanner
from navigation_controller import NavigationController


def main():
    # Path to the shapefile
    shapefile_path = Path(__file__).parent / "data/tile2net_output_SMUS/Victoria/network/Victoria-Network-18-01-2026_23_08/Victoria-Network-18-01-2026_23_08.shp"
    
    # Initialize path planner
    planner = PathPlanner(shapefile_path)
    
    # Get network bounds for reference
    bounds = planner.get_bounds()
    print(f"\nNetwork bounds:")
    print(f"  Longitude: [{bounds['minx']:.6f}, {bounds['maxx']:.6f}]")
    print(f"  Latitude: [{bounds['miny']:.6f}, {bounds['maxy']:.6f}]")
    
    print("\n" + "="*70)
    print("AREA COVERAGE PATH PLANNING")
    print("="*70)
    print("\nThis tool will help you:")
    print("1. Select an area on the map")
    print("2. Find all roads in that area")
    print("3. Generate the shortest path to cover all those roads")
    print("="*70)
    
    # Step 1: Show the full map
    print("\n[Step 1] Displaying full road network...")
    planner.show_map_with_labels(save_path='full_network.png')
    
    # Step 2: Select area
    print("\n[Step 2] Select the area you want to cover...")
    print("Instructions: Drag a rectangle on the map to select the area")
    area_polygon = planner.select_area_interactive()
    
    if area_polygon is None:
        print("\nNo area selected. Exiting...")
        return
    
    # Step 3: Get roads in the area
    print("\n[Step 3] Extracting roads in the selected area...")
    roads_in_area, subgraph = planner.get_roads_in_area(area_polygon)
    
    if roads_in_area is None or len(roads_in_area) == 0:
        print("No roads found in the selected area!")
        return
    
    # Show the selected area with roads
    print("\n[Step 4] Displaying selected area with roads...")
    planner.show_map_with_labels(area_polygon=area_polygon, save_path='selected_area.png')
    
    # Step 4.5: Select starting point
    print("\n[Step 5] Select starting point for coverage path...")
    print("Instructions: Click on the map to select where you want to start")
    start_point = planner.select_starting_point_interactive(area_polygon)
    
    if start_point is None:
        print("No starting point selected. Using automatic starting point.")
        start_node = None
    else:
        # Find nearest node to the selected point
        start_node = planner.find_nearest_node(start_point)
        print(f"Starting node on road network: {start_node}")
    
    # Step 5: Find coverage path
    print("\n[Step 6] Planning optimal coverage path...")
    coverage_path = planner.find_coverage_path(subgraph, start_node=start_node)
    
    if coverage_path is None or len(coverage_path) == 0:
        print("Could not generate coverage path!")
        return
    
    # Step 6: Visualize the result
    print("\n[Step 7] Visualizing coverage path...")
    planner.visualize_coverage_path(area_polygon, roads_in_area, coverage_path, 
                                   save_path='coverage_path.png')
    
    # Step 7: Export waypoints and create navigation controller
    print("\n[Step 8] Setting up navigation controller...")
    planner.export_path_to_waypoints(coverage_path, 'waypoints.txt')
    
    # Create navigation controller for robot
    nav_controller = NavigationController(coverage_path)
    
    print("\n" + "="*70)
    print("COVERAGE PATH PLANNING COMPLETE!")
    print("="*70)
    print(f"\nResults:")
    print(f"  - Roads to cover: {len(roads_in_area)}")
    print(f"  - Path waypoints: {len(coverage_path)}")
    print(f"  - Output files saved:")
    print(f"    * full_network.png - Full road network")
    print(f"    * selected_area.png - Selected coverage area")
    print(f"    * coverage_path.png - Final coverage path")
    print(f"    * waypoints.txt - Waypoint coordinates")
    
    # Example robot navigation usage
    print("\n" + "="*70)
    print("ROBOT NAVIGATION INTERFACE")
    print("="*70)
    print("\nNavigation controller ready! Use it in your robot code:")
    print("")
    print("# Get current status")
    print(f"status = nav_controller.get_status()")
    print(f"  -> Waypoints: {nav_controller.total_waypoints}")
    print("")
    print("# In your robot control loop:")
    print("current_location = (robot_lon, robot_lat)  # Your robot's GPS position")
    print("current_heading = robot_compass_heading    # Your robot's heading (0-360°)")
    print("cmd = nav_controller.get_direction_command(current_location, current_heading)")
    print("")
    print("if cmd['command'] == 'STRAIGHT':")
    print("    # Go straight")
    print("elif cmd['command'] == 'LEFT':")
    print("    # Turn left")
    print("elif cmd['command'] == 'RIGHT':")
    print("    # Turn right")
    print("elif cmd['command'] == 'COMPLETE':")
    print("    # Path complete!")
    print("")
    print("# Example simulation:")
    print("nav_controller.simulate_navigation()")
    
    print("\n" + "="*70)
    print("PROGRAMMATIC USAGE")
    print("="*70)
    print("\n# Select area and generate coverage path:")
    print("planner = PathPlanner('path/to/shapefile.shp')")
    print("area = planner.select_area_interactive()")
    print("roads, subgraph = planner.get_roads_in_area(area)")
    print("start_point = planner.select_starting_point_interactive(area)")
    print("start_node = planner.find_nearest_node(start_point)")
    print("path = planner.find_coverage_path(subgraph, start_node=start_node)")
    print("planner.visualize_coverage_path(area, roads, path)")


if __name__ == "__main__":
    main()
