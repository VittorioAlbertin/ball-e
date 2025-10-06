FROM osrf/ros:humble-desktop-full

# Add ubuntu user with same UID and GID as your host system, if it doesn't already exist
# Since Ubuntu 24.04, a non-root user is created by default with the name vscode and UID=1000
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN if ! id -u $USER_UID >/dev/null 2>&1; then \
        groupadd --gid $USER_GID $USERNAME && \
        useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi
# Add sudo support for the non-root user
RUN apt-get update && \
    apt-get install -y sudo && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# System deps
RUN apt-get update && apt-get install -y \
    sudo git python3 python3-venv python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Switch from root to user
USER $USERNAME
WORKDIR /home/$USERNAME

# Add user to video group to allow access to webcam
RUN sudo usermod --append --groups video $USERNAME

# Create venv for ML packages
RUN python3 -m venv ~/ml_env
ENV PATH="/home/$USERNAME/ml_env/bin:$PATH"

# Install deps inside venv
COPY requirements-heavy.txt /tmp/requirements-heavy.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements-heavy.txt

COPY requirements-light.txt /tmp/requirements-light.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements-light.txt

# Update all packages
# RUN sudo apt update && sudo apt upgrade -y

# Rosdep update
RUN rosdep update

# Source the ROS setup file
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc