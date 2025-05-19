# First stage sets up the base, ie everything except the active dev repo
FROM  osrf/ros:foxy-desktop-focal AS base

# Setup ROS 2 Workspace
SHELL ["/bin/bash", "-c"]

RUN sudo apt install python3-colcon-common-extensions

COPY carla-0.9.13-py3.7-linux-x86_64.egg /opt/carla/

ENV PYTHONPATH="/opt/carla/carla-0.9.13-py3.7-linux-x86_64.egg:${PYTHONPATH:+$PYTHONPATH}"

RUN apt update && apt install -y ros-foxy-rosbridge-server

# RUN mkdir /ros2_ws
COPY ros2_ws ros2_ws 

RUN cd ros2_ws && \
    source /opt/ros/foxy/setup.bash && \
    colcon build

COPY ./ros_entrypoint.sh /

# Ensure script is executable
RUN ["chmod", "+x", "/ros_entrypoint.sh"]

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
