# Base image
FROM ubuntu:24.04

# Basic tools
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    git \
    python3-pip \
    python3-venv \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 Jazzy installation
RUN apt-get update && apt-get install -y \
    ros-jazzy-desktop \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    && rm -rf /var/lib/apt/lists/*

# Set bash as default shell
SHELL ["/bin/bash", "-c"]

# Source ROS automatically
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

# Copy Python dependencies
COPY ../requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Set working directory inside container
WORKDIR /ros2_ws

# Default command
CMD ["/bin/bash"]
