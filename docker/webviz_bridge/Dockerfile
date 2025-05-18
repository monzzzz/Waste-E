FROM osrf/ros:foxy-desktop-focal AS base

# Install rosbridge_suite and dependencies
RUN apt update && apt install -y \
    ros-foxy-rosbridge-server \
    python3-websockets \
    python3-tornado \
    python3-colcon-common-extensions

# Set up workspace
WORKDIR /ros2_ws

# Optional: Clone additional ROS 2 packages if needed
# RUN mkdir -p src && \
#     cd src && \
#     git clone <your-additional-package-repo> && \
#     cd .. && \
#     colcon build

# Expose the default WebSocket port for rosbridge
EXPOSE 9090

# Default command to launch rosbridge_websocket
CMD ["ros2", "launch", "rosbridge_server", "rosbridge_websocket_launch.xml"]