from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Create the launch description
    ld = LaunchDescription()
    
    # Add the carla_bridge_controller node
    pedestrian_node = Node(
        package='carla_control_node',
        executable='pedestrian',
        name='pedestrian',
        output='screen',
        emulate_tty=True
    )

    pedestrian_controller_node = Node(
        package='carla_control_node',
        executable='pedestrian_controller',
        name='pedestrian_controller',
        output='screen',
        emulate_tty=True,
    )

    map_visualizer_node = Node(
        package='carla_control_node',
        executable='map_visualizer',
        name='map_visualizer',
        output='screen',
        emulate_tty=True,
    )
    
    # Add the node to the launch description
    ld.add_action(pedestrian_node)
    ld.add_action(pedestrian_controller_node)
    ld.add_action(map_visualizer_node)

    return ld