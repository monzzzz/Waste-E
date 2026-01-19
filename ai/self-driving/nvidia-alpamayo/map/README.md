# Coverage Path Planner for Waste Collection Robot

This module implements an interactive coverage path planning system for autonomous waste collection robots. It uses road network data (shapefiles) to generate optimal paths that cover every road segment in a selected area, complete with a navigation controller for path execution.

## Features

- **Interactive Map Visualization**: visualizes complex road networks from shapefiles.
- **Area Selection**: Interactive tool to select a specific working zone by drawing a rectangle on the map.
- **Optimal Path Planning**: Generates efficient coverage paths ensuring all roads in the selected area are visited.
- **Navigation Controller**: precise turn-by-turn navigation logic (`STRAIGHT`, `LEFT`, `RIGHT`) for robot control.
- **Simulation**: Built-in visual simulator to preview the robot's path execution.

## Requirements

This project relies on several Python libraries for geospatial analysis and graph theory:

- `geopandas`: For handling spatial data and shapefiles.
- `networkx`: For graph creation and path planning algorithms.
- `shapely`: For geometric operations.
- `matplotlib`: For visualization and interactive widgets.
- `numpy`: For numerical calculations.

## Project Structure

- **`main.py`**: The main entry point. Orchestrates the workflow from loading the map to generating navigation commands.
- **`path_planner.py`**: Core logic class. Handles:
  - Loading shapefiles.
  - Converting road networks to graphs.
  - Interactive area selection (drag-to-select).
  - Calculating coverage paths.
- **`navigation_controller.py`**: Controls the robot's movement. available for integration with robot hardware code.
  - Calculates bearings and turn angles.
  - Provides real-time commands based on current GPS location.
- **`simulate_navigation.py`**: Visual tool to simulate the path following behavior on screen.

## Usage

### 1. Run the Main Script
Start the interactive tool:
```bash
python main.py
```

### 2. Follow the Interactive Steps
1.  **View Map**: The full road network will be displayed.
2.  **Select Area**: A window will open asking you to "Drag to select an area". Click and drag a rectangle over the region you want the robot to cover.
3.  **Confirm Selection**: The selected roads will be extracted.
4.  **Select Start Point**: Click on the map to define where the robot will start its journey.
5.  **Generate Path**: The system will calculate the coverage path.
6.  **Navigation**: The system will export the path and initialize the navigation controller.

### 3. Output Files
The tool typically generates the following outputs for review/debugging:
- `full_network.png`: Image of the entire loaded map.
- `selected_area.png`: Image showing just the user-selected region.
- `coverage_path.png`: Visualization of the calculated path.
- `waypoints.txt`: A text file containing the ordered list of coordinates for the robot to follow.

## Integration with Robot Hardware

The `NavigationController` class is designed to be used directly in your robot's control loop.

```python
from navigation_controller import NavigationController

# Load generated waypoints
# ... (load your waypoints)

nav = NavigationController(waypoints)

# In your control loop:
location = get_gps_location()
heading = get_compass_heading()

cmd = nav.get_direction_command(location, heading)

if cmd['command'] == 'STRAIGHT':
    motor_forward()
elif cmd['command'] == 'LEFT':
    turn_left()
# ...
```