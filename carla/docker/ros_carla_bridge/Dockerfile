# First stage sets up the base, ie everything except the active dev repo
FROM osrf/ros:foxy-desktop-focal AS base

SHELL ["/bin/bash", "-c"]

RUN apt update && apt install -y python3-pip

RUN sudo apt install python3-colcon-common-extensions 

RUN mkdir -p ~/carla-ros-bridge
WORKDIR /root/carla-ros-bridge

# TODO: Check versions, should support up to CARLA 0.9.13
RUN git clone --recurse-submodules https://github.com/carla-simulator/ros-bridge.git src/ros-bridge

RUN rm -rf src/ros-bridge/carla_msgs

RUN git clone https://github.com/carla-simulator/ros-carla-msgs.git src/ros_carla_msgs

RUN rosdep update && rosdep install --from-paths src --ignore-src -r

RUN apt-get install ros-foxy-derived-object-msgs xdg-user-dirs

RUN source /opt/ros/foxy/setup.bash && \
    colcon build 

RUN source install/setup.bash

COPY ros_entrypoint.sh /

# Ensure script is executable
RUN ["sudo", "chmod", "+x", "/ros_entrypoint.sh"]

ENTRYPOINT ["/ros_entrypoint.sh"]

RUN python3 -m pip install --upgrade pip && python3 -m pip install numpy pygame
RUN python3 -m pip install 'carla==0.9.13'

