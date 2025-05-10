FROM ros:humble



# Install system dependencies

RUN apt update && apt install -y \

    python3-pip \

    xdg-user-dirs \

    ros-humble-pcl-conversions \

    ros-humble-rviz2 \

    python3-colcon-common-extensions \

    ros-humble-rmw-cyclonedds-cpp


WORKDIR /workspace

COPY . /workspace



# Build all packages to avoid missing runtime dependencies

RUN . /opt/ros/humble/setup.sh & rosdep install --from-path src --ignore-src -r -y

RUN . /opt/ros/humble/setup.sh && colcon build



# Launch the bridge on container start

CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.sh && source install/setup.sh && ros2 launch carla_ros_bridge carla_ros_bridge.launch.py"]

