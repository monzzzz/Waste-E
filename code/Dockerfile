FROM ros:humble

RUN apt update && apt install -y python3-pip ros-humble-ackermann-msgs

RUN pip3 install numpy opencv-python scipy matplotlib

COPY . /workspace
WORKDIR /workspace

CMD ["bash"]
