# Ball-e Quick Start Guide

This guide will get you up and running with Ball-e's person tracking and identification system in minutes.

## Prerequisites

### DevContainer Setup (Recommended)

**Required Software:**
- **Visual Studio Code** - Download from https://code.visualstudio.com/
- **Docker** - Install docker.io on Linux:
  ```bash
  sudo apt update
  sudo apt install docker.io
  sudo systemctl start docker
  sudo systemctl enable docker
  sudo usermod -aG docker $USER  # Add your user to docker group
  # Log out and back in for group changes to take effect
  ```
- **Dev Containers Extension** - Install in VS Code:
  - Open VS Code
  - Press `Ctrl+Shift+X` to open Extensions
  - Search for "Dev Containers"
  - Install the extension by Microsoft

**Hardware Requirements:**
- 4GB RAM minimum (8GB recommended)
- Webcam or USB camera
- (Optional) NVIDIA GPU for faster inference

### Manual Setup (Alternative)

**System Requirements:**
- Ubuntu 22.04 (recommended) or compatible Linux distribution
- ROS2 Humble installed
- Python 3.10 or higher
- 4GB RAM minimum (8GB recommended)
- Webcam or USB camera
- (Optional) NVIDIA GPU for faster inference

**Software Dependencies:**
```bash
# Install ROS2 Humble if not already installed
# Follow: https://docs.ros.org/en/humble/Installation.html

# Install Python dependencies
pip3 install scipy numpy onnxruntime opencv-python

# Install ros2_numpy
pip3 install ros2-numpy
```

## Installation

### Option A: DevContainer Setup (Recommended)

This is the easiest way to get started. The DevContainer handles all dependencies, ROS2 installation, and workspace building automatically.

#### Step 1: Clone the Repository

```bash
cd ~/Documents
git clone <repository-url> ball-e
cd ball-e
```

#### Step 2: Open in VS Code

```bash
code .
```

#### Step 3: Reopen in Container

When VS Code opens, you should see a notification:
> "Folder contains a Dev Container configuration file. Reopen folder to develop in a container."

Click **"Reopen in Container"**

Or manually trigger it:
- Press `Ctrl+Shift+P`
- Type "Dev Containers: Rebuild and Reopen in Container"
- Press Enter

#### Step 4: Wait for Automatic Setup

The container will now build and configure everything automatically. This process takes 5-10 minutes on first run and includes:

1. **Building Docker Image** (`.devcontainer/Dockerfile`):
   - Installs ROS2 Humble Desktop Full
   - Sets up Ubuntu user with sudo access
   - Configures video group for webcam access

2. **Creating Python Virtual Environment**:
   - Creates `/home/ubuntu/ml_env` for ML packages
   - Installs PyTorch, Ultralytics YOLO, OpenCV, and all dependencies from `.devcontainer/requirements.txt`

3. **Building ROS2 Workspace** (`.devcontainer/colcon_build.py`):
   - Scans all packages for `<group_depend>` tags in `package.xml`
   - Builds **system packages** with system Python (`/usr/bin/python3`)
   - Builds **ML packages** with virtual environment Python (`/home/ubuntu/ml_env/bin/python3`)
   - This dual-environment approach solves Python dependency conflicts between ROS2 and ML libraries

**You'll see output like:**
```
Building group 'system' with interpreter /usr/bin/python3: [msgs_interfaces, sensors_pkg, ...]
Building group 'ml' with interpreter /home/ubuntu/ml_env/bin/python3: [perception_pkg, ...]
```

#### Step 5: Verify Setup

Once the container is ready, open a terminal in VS Code (`Ctrl+` ` `) and verify:

```bash
# Check ROS2 is sourced
echo $ROS_DISTRO
# Should output: humble

# Check workspace is built
ls ros2_ws/install/
# Should show: setup.bash, setup.sh, and package folders

# Verify ml_env exists
ls /home/ubuntu/ml_env/bin/python3
# Should show: /home/ubuntu/ml_env/bin/python3
```

**That's it!** The system is ready to run. No manual environment activation needed.

---

### Understanding the DevContainer Setup

The DevContainer uses a sophisticated dual-Python-environment build system to handle incompatible dependencies:

**Why Two Environments?**
- **ROS2 packages** require system Python and specific ROS2 dependencies
- **ML packages** (PyTorch, Ultralytics) need newer versions that conflict with ROS2
- Solution: Build different package groups with different Python interpreters

**How It Works:**

1. **Package Grouping** - Each ROS2 package declares its group in `package.xml`:
   ```xml
   <!-- System packages (msgs_interfaces, sensors_pkg, etc.) -->
   <group_depend>system</group_depend>

   <!-- ML packages (perception_pkg with YOLO/face recognition) -->
   <group_depend>ml</group_depend>
   ```

2. **Automatic Build** - The `colcon_build.py` script:
   - Scans all packages for `<group_depend>` tags
   - Groups them by dependency environment
   - Runs `colcon build` for each group with the appropriate Python interpreter:
     - `system` group → `/usr/bin/python3`
     - `ml` group → `/home/ubuntu/ml_env/bin/python3`

3. **Runtime** - Launch files automatically use the correct Python environment:
   - ROS2 nodes in the `ml` group are configured to use `/home/ubuntu/ml_env/bin/python3`
   - No manual environment switching required

**Key Files:**
- `.devcontainer/devcontainer.json` - Container configuration and postCreateCommand
- `.devcontainer/Dockerfile` - ROS2 Humble base image with dependencies
- `.devcontainer/colcon_build.py` - Intelligent multi-environment build script
- `.devcontainer/requirements.txt` - Python ML dependencies for virtual environment

---

### Option B: Manual Installation

#### Step 1: Clone and Build

```bash
# Navigate to your workspace
cd ~/Documents/ball-e

# Build the ROS2 workspace
cd ros2_ws
colcon build --symlink-install

# Source the workspace
source install/setup.bash

# Add to your .bashrc for convenience
echo "source ~/Documents/ball-e/ros2_ws/install/setup.bash" >> ~/.bashrc
```

#### Step 2: Download Models

The face recognition models will be downloaded automatically on first run. Alternatively, download them manually:

```bash
cd ~/Documents/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models

# Face embedding model (~100MB)
wget https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx \
     -O facenet.onnx
```

#### Step 3: Initialize Database

```bash
# The people database will be created automatically on first run
# Default location: ~/Documents/ball-e/ros2_ws/robot_data/people.db
```

## Running Ball-e

### Launch the Full System

```bash
# Source your workspace
source ~/Documents/ball-e/ros2_ws/install/setup.bash

# Launch everything
ros2 launch robot_bringup ball_e_full_system_launch.py
```

You should see output indicating all nodes are starting:
```
[INFO] [camera_node]: Camera node initialized
[INFO] [yolo_node]: YOLO node initialized
[INFO] [person_tracker]: Person Tracker initialized
[INFO] [person_state_manager]: Person State Manager started
[INFO] [face_recognition_conditional]: Conditional Face Recognition Node started
[INFO] [visualization_node]: Visualization Node started
```

### Launch with Visualization

```bash
# Launch system in background
ros2 launch robot_bringup ball_e_full_system_launch.py &

# Start RViz2 with the tracking configuration
rviz2 -d ~/Documents/ball-e/ros2_ws/src/robot_bringup/rviz/ball_e_tracking.rviz
```

In RViz2, you'll see:
- Raw camera feed
- Annotated visualization with track IDs and identities
- Real-time person tracking

## Enrolling People

### Using the Enrollment CLI

```bash
# Run the enrollment tool
ros2 run interaction_pkg enroll_face_cli
```

Follow the prompts:
1. Position your face in front of the camera
2. Press 'c' to capture
3. Enter your name when prompted
4. Face will be added to the database

### Verify Enrollment

```bash
# Check the database
cat ~/Documents/ball-e/ros2_ws/src/interaction_pkg/people_database.json
```

## Configuration

### Basic Parameters

Launch with custom parameters:

```bash
ros2 launch robot_bringup ball_e_full_system_launch.py \
    max_age:=40 \
    min_hits:=2 \
    confidence_threshold:=0.6
```

### Parameter Reference

#### Tracking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_age` | 30 | Frames to keep track without detection |
| `min_hits` | 3 | Detections before confirming track |
| `iou_threshold` | 0.3 | IoU for matching detections to tracks |

#### State Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cleanup_timeout` | 5.0 | Seconds before removing inactive persons |
| `publish_rate` | 10.0 | Hz for publishing states |

#### Identification Coordination

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_requests_per_second` | 2.0 | Rate limit for identifications |
| `confidence_threshold` | 0.5 | Re-identify below this confidence |
| `reidentification_interval` | 60.0 | Seconds before re-identifying |

#### Face Recognition

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recognition_threshold` | 0.6 | Cosine similarity for matching |
| `min_face_size` | 20 | Minimum face size in pixels |
| `max_face_size` | 400 | Maximum face size in pixels |
| `frame_cache_size` | 10 | Frames to cache |

## Monitoring

### View Active Topics

```bash
# List all active topics
ros2 topic list

# Key topics:
# /camera/image_raw
# /yolo/detections
# /person_tracker/tracks
# /person_state/all
# /face_recognition/identity_update
# /visualization/annotated_image
```

### Monitor Person States

```bash
# Real-time person states
ros2 topic echo /person_state/all

# Identity updates
ros2 topic echo /face_recognition/identity_update

# Tracking data
ros2 topic echo /person_tracker/tracks
```

### Check Node Status

```bash
# List all running nodes
ros2 node list

# Get info about a specific node
ros2 node info /person_state_manager

# View node logs
ros2 run rqt_console rqt_console
```

### Query Services

```bash
# Get info about a person by track_id
ros2 service call /person_state/get_info \
    msgs_interfaces/srv/GetPersonInfo "{track_id: 1}"

# Request identification for a track
ros2 service call /person_state/request_identification \
    msgs_interfaces/srv/RequestIdentification "{track_id: 1}"
```

## Testing

### Test Individual Components

```bash
# Test person tracker only
ros2 launch robot_bringup camera_yolo_tracker_launch.py

# Test state manager only
ros2 launch robot_bringup person_state_manager_launch.py

# Test face recognition only
ros2 launch robot_bringup face_recognition_conditional_launch.py
```

### Verify Camera

```bash
# View camera feed
ros2 run rqt_image_view rqt_image_view /camera/image_raw
```

### Verify YOLO Detection

```bash
# View YOLO detections
ros2 run rqt_image_view rqt_image_view /yolo/image_detections
```

## Common Use Cases

### Scenario 1: Track People in a Room

```bash
# Launch with optimized tracking for indoor use
ros2 launch robot_bringup ball_e_full_system_launch.py \
    max_age:=40 \
    min_hits:=2 \
    cleanup_timeout:=3.0
```

### Scenario 2: High-Accuracy Identification

```bash
# Launch with strict recognition threshold
ros2 launch robot_bringup ball_e_full_system_launch.py \
    recognition_threshold:=0.7 \
    min_face_size:=30
```

### Scenario 3: Frequent Re-identification

```bash
# Re-identify people every 30 seconds
ros2 launch robot_bringup ball_e_full_system_launch.py \
    reidentification_interval:=30.0 \
    max_requests_per_second:=3.0
```

## Troubleshooting

### No Video Feed

```bash
# Check camera device
ls /dev/video*

# Test camera directly
ros2 run sensors_pkg camera_node
ros2 topic hz /camera/image_raw
```

### No Person Detections

```bash
# Check YOLO node
ros2 topic echo /yolo/detections

# Enable debug logging
ros2 run perception_pkg yolo_node --ros-args --log-level debug
```

### Face Recognition Not Working

```bash
# Verify models downloaded
ls ~/Documents/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/

# Check database service
ros2 service list | grep people_db

# Test recognition service
ros2 service call people_db/recognize_face \
    msgs_interfaces/srv/RecognizeFace "{face_embedding: [0.1, 0.2, ...], threshold: 0.6}"
```

### Poor Tracking Performance

```bash
# Adjust tracking parameters
ros2 launch robot_bringup ball_e_full_system_launch.py \
    iou_threshold:=0.4 \
    max_age:=50
```

## Performance Tuning

### For Low-End Hardware

```bash
# Reduce processing load
ros2 launch robot_bringup ball_e_full_system_launch.py \
    max_requests_per_second:=1.0 \
    frame_cache_size:=5 \
    publish_rate:=5.0
```

### For High-End Hardware

```bash
# Maximize performance
ros2 launch robot_bringup ball_e_full_system_launch.py \
    max_requests_per_second:=5.0 \
    frame_cache_size:=20 \
    publish_rate:=30.0
```

## Next Steps

- Read the [API Reference](API_REFERENCE.md) for detailed interface documentation
- Explore [TRACKING.md](docs/TRACKING.md) for advanced tracking configuration
- Check [STATE_MANAGEMENT.md](docs/STATE_MANAGEMENT.md) for state manager details
- Review [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues

## Getting Help

If you encounter issues:

1. Check the logs: `ros2 run rqt_console rqt_console`
2. Review [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
3. Enable debug logging: `--ros-args --log-level debug`
4. Open an issue on GitHub with logs and system info

---

Happy tracking! 🤖
