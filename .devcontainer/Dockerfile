FROM osrf/ros:humble-desktop-full

ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN if ! id -u $USER_UID >/dev/null 2>&1; then \
        groupadd --gid $USER_GID $USERNAME && \
        useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi
# Add sudo support for the non-root user
RUN apt-get update && apt-get install -y \
    sudo git python3 python3-venv python3-dev python3-pip \
    portaudio19-dev ffmpeg && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME &&\
    rm -rf /var/lib/apt/lists/*

# Install system-level Python packages for audio (not in ML venv)
RUN pip3 install sounddevice

# Rosdep update
RUN rosdep update

# Add user to video group to allow access to webcam
# Add user to audio group to allow access to microphone
RUN usermod --append --groups video,audio $USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME

# Source the ROS setup file
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc