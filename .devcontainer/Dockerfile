# Use Ubuntu 24.04 as base
FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive

# Basic tools
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    sudo \
    software-properties-common \
    git \
    wget \
    python3-pip \
    python3-venv \
    dbus-x11 \
    xfce4 xfce4-terminal \
    tightvncserver \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# ROS2 Jazzy setup
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list' \
    && apt-get update \
    && apt-get install -y ros-jazzy-desktop \
    && rm -rf /var/lib/apt/lists/*

# More tools
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-opencv \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Setup ROS2 environment
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

# Create VNC password (default: 'vncpassword')
RUN mkdir /root/.vnc && echo "vncpassword" | vncpasswd -f > /root/.vnc/passwd && chmod 600 /root/.vnc/passwd

# Expose VNC port
EXPOSE 5901

# Add start script
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
