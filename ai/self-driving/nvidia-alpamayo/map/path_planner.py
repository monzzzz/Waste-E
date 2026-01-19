import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, box
from shapely.ops import nearest_points
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector


class PathPlanner:
    def __init__(self, shapefile_path):
        """
        Initialize the path planner with a road network shapefile.
        
        Args:
            shapefile_path: Path to the .shp file containing the road network
        """
        print("Loading road network...")
        self.gdf = gpd.read_file(shapefile_path)
        print(f"Loaded {len(self.gdf)} road segments")
        
        # Create network graph
        self.graph = nx.Graph()
        self._build_graph()
        
    def _build_graph(self):
        """Build a NetworkX graph from the road network."""
        print("Building graph from road network...")
        
        for idx, row in self.gdf.iterrows():
            geom = row.geometry
            
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                
                # Add edges between consecutive points
                for i in range(len(coords) - 1):
                    start_node = coords[i]
                    end_node = coords[i + 1]
                    
                    # Calculate distance (weight)
                    distance = Point(start_node).distance(Point(end_node))
                    
                    # Add edge to graph with segment info
                    self.graph.add_edge(start_node, end_node, 
                                      weight=distance, 
                                      segment_id=idx,
                                      geometry=LineString([start_node, end_node]))
            
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    coords = list(line.coords)
                    for i in range(len(coords) - 1):
                        start_node = coords[i]
                        end_node = coords[i + 1]
                        distance = Point(start_node).distance(Point(end_node))
                        self.graph.add_edge(start_node, end_node, 
                                          weight=distance,
                                          segment_id=idx,
                                          geometry=LineString([start_node, end_node]))
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def find_nearest_node(self, point):
        """
        Find the nearest node in the graph to a given point.
        
        Args:
            point: Tuple of (x, y) coordinates or Point object
            
        Returns:
            Nearest node coordinates as tuple
        """
        if isinstance(point, (list, tuple)):
            point = Point(point)
        
        # Get all nodes
        nodes = list(self.graph.nodes())
        
        # Find nearest node
        min_dist = float('inf')
        nearest_node = None
        
        for node in nodes:
            node_point = Point(node)
            dist = point.distance(node_point)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        
        return nearest_node
    
    def select_area_interactive(self):
        """
        Interactive area selection using rectangle selector.
        Drag to draw a rectangle on the map to select an area.
        
        Returns:
            Polygon representing the selected area
        """
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # Plot road network
        self.gdf.plot(ax=ax, color='navy', linewidth=0.8, alpha=0.6)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Drag to select an area (rectangle)\nPress Enter when done', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        selected_area = {'box': None, 'selector': None}
        
        def on_select(eclick, erelease):
            """Handle rectangle selection."""
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            
            # Create bounding box
            minx, maxx = min(x1, x2), max(x1, x2)
            miny, maxy = min(y1, y2), max(y1, y2)
            
            selected_area['box'] = box(minx, miny, maxx, maxy)
            
            print(f"\nArea selected:")
            print(f"  Longitude: [{minx:.6f}, {maxx:.6f}]")
            print(f"  Latitude: [{miny:.6f}, {maxy:.6f}]")
            print("Close the window to proceed...")
        
        # Create rectangle selector and keep reference to prevent garbage collection
        selector = RectangleSelector(
            ax, on_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True,
            props=dict(facecolor='red', edgecolor='red', alpha=0.3, fill=True)
        )
        
        # Keep reference to selector to prevent garbage collection
        selected_area['selector'] = selector
        
        plt.show()
        
        return selected_area['box']
    
    def select_starting_point_interactive(self, area_polygon=None):
        """
        Interactive starting point selection by clicking on the map.
        
        Args:
            area_polygon: Optional polygon to highlight the area
            
        Returns:
            Tuple of (x, y) coordinates for the starting point
        """
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # Plot road network
        self.gdf.plot(ax=ax, color='navy', linewidth=0.8, alpha=0.6)
        
        # Plot area if provided
        if area_polygon:
            x, y = area_polygon.exterior.xy
            ax.plot(x, y, 'r--', linewidth=2, label='Selected Area', zorder=4)
            ax.fill(x, y, color='red', alpha=0.15)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Click to select STARTING POINT for coverage path', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        selected_point = {'point': None}
        
        def on_click(event):
            """Handle mouse click."""
            if event.inaxes == ax and selected_point['point'] is None:
                x, y = event.xdata, event.ydata
                selected_point['point'] = (x, y)
                
                # Plot the selected point
                ax.plot(x, y, 'go', markersize=15, label='Starting Point', zorder=5)
                ax.annotate('START', xy=(x, y), xytext=(10, 10),
                          textcoords='offset points',
                          bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.8),
                          fontsize=12, fontweight='bold',
                          arrowprops=dict(arrowstyle='->', color='green'))
                
                ax.legend()
                fig.canvas.draw()
                
                print(f"\nStarting point selected: ({x:.6f}, {y:.6f})")
                print("Close the window to proceed...")
        
        # Connect the click event
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        plt.show()
        
        return selected_point['point']
    
    def get_roads_in_area(self, area_polygon):
        """
        Get all road segments within the specified area.
        
        Args:
            area_polygon: Shapely Polygon representing the area
            
        Returns:
            GeoDataFrame with roads in the area, and subgraph
        """
        if area_polygon is None:
            print("No area selected!")
            return None, None
        
        # Filter roads that intersect with the area
        roads_in_area = self.gdf[self.gdf.intersects(area_polygon)]
        
        print(f"\nFound {len(roads_in_area)} road segments in the selected area")
        
        # Create subgraph for the area
        subgraph = nx.Graph()
        
        for idx, row in roads_in_area.iterrows():
            geom = row.geometry
            
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                for i in range(len(coords) - 1):
                    start_node = coords[i]
                    end_node = coords[i + 1]
                    
                    # Check if both nodes are in the area
                    if area_polygon.contains(Point(start_node)) or area_polygon.contains(Point(end_node)):
                        distance = Point(start_node).distance(Point(end_node))
                        subgraph.add_edge(start_node, end_node, 
                                        weight=distance,
                                        segment_id=idx,
                                        geometry=LineString([start_node, end_node]))
            
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    coords = list(line.coords)
                    for i in range(len(coords) - 1):
                        start_node = coords[i]
                        end_node = coords[i + 1]
                        if area_polygon.contains(Point(start_node)) or area_polygon.contains(Point(end_node)):
                            distance = Point(start_node).distance(Point(end_node))
                            subgraph.add_edge(start_node, end_node, 
                                            weight=distance,
                                            segment_id=idx,
                                            geometry=LineString([start_node, end_node]))
        
        print(f"Subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
        
        return roads_in_area, subgraph
    
    def find_coverage_path(self, subgraph, start_node=None):
        """
        Find a path that covers all edges (roads) in the subgraph.
        Uses Chinese Postman Problem approach.
        
        Args:
            subgraph: NetworkX graph of roads to cover
            start_node: Optional starting node
            
        Returns:
            List of nodes representing the coverage path
        """
        if subgraph.number_of_edges() == 0:
            print("No roads to cover!")
            return None
        
        print("\nFinding optimal coverage path...")
        
        # Check if graph is connected
        if not nx.is_connected(subgraph):
            print("Warning: Graph is not fully connected. Finding path in largest component...")
            # Get largest connected component
            largest_cc = max(nx.connected_components(subgraph), key=len)
            subgraph = subgraph.subgraph(largest_cc).copy()
            print(f"Using largest component with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
        
        # Find nodes with odd degree (for Chinese Postman Problem)
        odd_degree_nodes = [node for node, degree in subgraph.degree() if degree % 2 == 1]
        
        if len(odd_degree_nodes) == 0:
            # Graph is Eulerian - perfect case!
            print("Graph is Eulerian! Finding Eulerian path...")
            path = list(nx.eulerian_path(subgraph, source=start_node))
            # Convert edge list to node list
            node_path = [path[0][0]]
            for edge in path:
                node_path.append(edge[1])
            return node_path
        
        else:
            print(f"Graph has {len(odd_degree_nodes)} odd-degree nodes. Creating coverage path...")
            
            # For simplicity, use a greedy approach or approximate solution
            # Find minimum spanning tree as a baseline
            if start_node is None:
                start_node = list(subgraph.nodes())[0]
            
            # Use DFS traversal to visit all edges
            visited_edges = set()
            path = [start_node]
            current = start_node
            
            while len(visited_edges) < subgraph.number_of_edges():
                # Find unvisited edges from current node
                unvisited_neighbors = []
                for neighbor in subgraph.neighbors(current):
                    edge = tuple(sorted([current, neighbor]))
                    if edge not in visited_edges:
                        unvisited_neighbors.append(neighbor)
                
                if unvisited_neighbors:
                    # Choose closest unvisited neighbor
                    next_node = min(unvisited_neighbors, 
                                  key=lambda n: subgraph[current][n]['weight'])
                    edge = tuple(sorted([current, next_node]))
                    visited_edges.add(edge)
                    path.append(next_node)
                    current = next_node
                else:
                    # Need to traverse to find unvisited edges
                    # Find shortest path to a node with unvisited edges
                    found_next = False
                    for node in subgraph.nodes():
                        for neighbor in subgraph.neighbors(node):
                            edge = tuple(sorted([node, neighbor]))
                            if edge not in visited_edges:
                                # Find path to this node
                                try:
                                    bridge_path = nx.shortest_path(subgraph, current, node, weight='weight')
                                    path.extend(bridge_path[1:])  # Skip current node
                                    current = node
                                    found_next = True
                                    break
                                except nx.NetworkXNoPath:
                                    continue
                        if found_next:
                            break
                    
                    if not found_next:
                        break
            
            print(f"Coverage path created with {len(path)} waypoints")
            return path
    
    def visualize_coverage_path(self, area_polygon, roads_in_area, coverage_path, save_path=None):
        """
        Visualize the area coverage path.
        
        Args:
            area_polygon: Selected area polygon
            roads_in_area: GeoDataFrame of roads in the area
            coverage_path: List of nodes in the coverage path
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(16, 16))
        
        # Plot full road network in background
        self.gdf.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.3, label='Other Roads')
        
        # Plot roads in selected area
        if roads_in_area is not None and len(roads_in_area) > 0:
            roads_in_area.plot(ax=ax, color='navy', linewidth=1.5, alpha=0.6, label='Roads to Cover')
        
        # Plot selected area boundary
        if area_polygon:
            x, y = area_polygon.exterior.xy
            ax.plot(x, y, 'r--', linewidth=2, label='Selected Area', zorder=4)
            ax.fill(x, y, color='red', alpha=0.1)
        
        # Plot coverage path
        if coverage_path and len(coverage_path) > 1:
            path_x = [p[0] for p in coverage_path]
            path_y = [p[1] for p in coverage_path]
            
            # Draw the path line
            ax.plot(path_x, path_y, 'b-', linewidth=3, label='Coverage Path', zorder=5, alpha=0.8)
            
            # Add directional arrows along the path
            arrow_spacing = max(1, len(coverage_path) // 20)  # Show ~20 arrows
            for i in range(0, len(coverage_path) - 1, arrow_spacing):
                dx = path_x[i+1] - path_x[i]
                dy = path_y[i+1] - path_y[i]
                ax.arrow(path_x[i], path_y[i], dx*0.8, dy*0.8,
                        head_width=0.0001, head_length=0.0001,
                        fc='blue', ec='blue', zorder=5, alpha=0.7)
            
            # Mark waypoints with numbers (show every 10th waypoint)
            waypoint_spacing = max(1, len(coverage_path) // 15)
            for i in range(0, len(coverage_path), waypoint_spacing):
                ax.plot(path_x[i], path_y[i], 'co', markersize=5, zorder=6)
                if i % (waypoint_spacing * 2) == 0:  # Label some waypoints
                    ax.annotate(f'{i}', xy=(path_x[i], path_y[i]),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, color='darkblue', weight='bold')
            
            # Mark start and end
            ax.plot(path_x[0], path_y[0], 'go', markersize=20, label='Start', zorder=7, markeredgecolor='darkgreen', markeredgewidth=2)
            ax.plot(path_x[-1], path_y[-1], 'ro', markersize=20, label='End', zorder=7, markeredgecolor='darkred', markeredgewidth=2)
            
            # Calculate total distance
            total_distance = sum(
                Point(coverage_path[i]).distance(Point(coverage_path[i+1]))
                for i in range(len(coverage_path) - 1)
            )
            
            ax.text(0.02, 0.98, 
                   f'Coverage Path Stats:\n'
                   f'Waypoints: {len(coverage_path)}\n'
                   f'Total Distance: {total_distance:.4f} units\n'
                   f'Roads Covered: {len(roads_in_area) if roads_in_area is not None else 0}',
                   transform=ax.transAxes,
                   fontsize=11,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Road Coverage Path Planning', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def plan_path(self, start_point, end_point):
        """
        Plan a path from start to end using A* algorithm.
        
        Args:
            start_point: Tuple of (x, y) coordinates for start
            end_point: Tuple of (x, y) coordinates for destination
            
        Returns:
            List of coordinates representing the path, or None if no path found
        """
        print(f"\nPlanning path from {start_point} to {end_point}...")
        
        # Find nearest nodes to start and end points
        start_node = self.find_nearest_node(start_point)
        end_node = self.find_nearest_node(end_point)
        
        print(f"Start node: {start_node}")
        print(f"End node: {end_node}")
        
        # Check if nodes are in the graph
        if start_node not in self.graph:
            print("Error: Start node not in graph")
            return None
        if end_node not in self.graph:
            print("Error: End node not in graph")
            return None
        
        # Use A* algorithm to find shortest path
        try:
            path = nx.astar_path(
                self.graph, 
                start_node, 
                end_node, 
                heuristic=lambda a, b: Point(a).distance(Point(b)),
                weight='weight'
            )
            
            # Calculate total distance
            total_distance = nx.astar_path_length(
                self.graph,
                start_node,
                end_node,
                heuristic=lambda a, b: Point(a).distance(Point(b)),
                weight='weight'
            )
            
            print(f"Path found! Total distance: {total_distance:.2f} units")
            print(f"Path has {len(path)} nodes")
            
            return path
            
        except nx.NetworkXNoPath:
            print("No path found between the two points!")
            return None
        except Exception as e:
            print(f"Error finding path: {e}")
            return None
    
    def visualize_path(self, path, start_point, end_point, save_path=None):
        """
        Visualize the planned path on the road network.
        
        Args:
            path: List of coordinates representing the path
            start_point: Starting point coordinates
            end_point: Ending point coordinates
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # Plot road network
        self.gdf.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.5, label='Road Network')
        
        if path:
            # Plot the path
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='Planned Path', zorder=3)
            
            # Plot waypoints
            ax.plot(path_x, path_y, 'co', markersize=3, label='Waypoints', zorder=4)
        
        # Plot start and end points
        ax.plot(start_point[0], start_point[1], 'go', markersize=15, label='Start', zorder=5)
        ax.plot(end_point[0], end_point[1], 'ro', markersize=15, label='Destination', zorder=5)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Path Planning Result')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def get_bounds(self):
        """Get the bounding box of the road network."""
        bounds = self.gdf.total_bounds
        return {
            'minx': bounds[0],
            'miny': bounds[1],
            'maxx': bounds[2],
            'maxy': bounds[3]
        }
    
    def show_map_with_labels(self, points_to_label=None, labels=None, area_polygon=None, save_path=None):
        """
        Display the road network map with labeled points and optional area.
        
        Args:
            points_to_label: List of tuples [(x1, y1), (x2, y2), ...] to mark on map
            labels: List of labels for each point ['Start', 'Waypoint', 'End', ...]
            area_polygon: Optional polygon to highlight
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # Plot road network
        self.gdf.plot(ax=ax, color='navy', linewidth=0.8, alpha=0.6)
        
        # Plot area if provided
        if area_polygon:
            x, y = area_polygon.exterior.xy
            ax.plot(x, y, 'r--', linewidth=2, label='Selected Area', zorder=4)
            ax.fill(x, y, color='red', alpha=0.15)
        
        # If points are provided, plot and label them
        if points_to_label:
            if labels is None:
                labels = [f'Point {i+1}' for i in range(len(points_to_label))]
            
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
            
            for i, (point, label) in enumerate(zip(points_to_label, labels)):
                color = colors[i % len(colors)]
                ax.plot(point[0], point[1], 'o', color=color, markersize=12, 
                       label=label, zorder=5)
                
                # Add text annotation with white background for readability
                ax.annotate(label, xy=point, xytext=(10, 10), 
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                           fontsize=10, fontweight='bold',
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Get bounds for display
        bounds = self.get_bounds()
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Road Network Map', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add coordinate info to the plot
        info_text = f"Network Coverage:\nLon: [{bounds['minx']:.6f}, {bounds['maxx']:.6f}]\nLat: [{bounds['miny']:.6f}, {bounds['maxy']:.6f}]"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Map saved to {save_path}")
        
        plt.show()
        
        return fig, ax
    
    def export_path_to_waypoints(self, coverage_path, filename='waypoints.txt'):
        """
        Export coverage path to a text file with waypoint coordinates.
        
        Args:
            coverage_path: List of nodes in the coverage path
            filename: Output filename
        """
        if coverage_path is None or len(coverage_path) == 0:
            print("No path to export!")
            return
        
        with open(filename, 'w') as f:
            f.write("# Waypoint Index, Longitude, Latitude\n")
            for i, point in enumerate(coverage_path):
                f.write(f"{i},{point[0]},{point[1]}\n")
        
        print(f"\nPath exported to {filename} ({len(coverage_path)} waypoints)")
