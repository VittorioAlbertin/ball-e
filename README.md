# Ball-e Social Robot

Ball-e is a ROS2 Humble social robot with advanced person tracking and face recognition capabilities. The system uses a modular architecture designed for real-time performance and extensibility.

![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)

## Overview

Ball-e implements a sophisticated person tracking and identification pipeline that achieves **30 FPS system throughput** with on-demand face recognition, solving the original ~1 FPS bottleneck.

### Key Features

- **Persistent Person Tracking**: ByteTrack algorithm for multi-person tracking with unique IDs
- **On-Demand Face Recognition**: Conditional processing triggered by smart coordination logic
- **Centralized State Management**: Single source of truth for all person states
- **Real-time Visualization**: Annotated video with track IDs, identities, and status
- **Modular Architecture**: Clean separation of concerns with well-defined interfaces
- **High Performance**: <200ms identification latency, ~60% CPU reduction

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    BALL-E SYSTEM                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Camera → YOLO → ByteTrack → State Manager             │
│                        ↓           ↓                    │
│                   Coordinator  Visualization            │
│                        ↓                                │
│               Face Recognition (conditional)            │
│                        ↓                                │
│                 People Database                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### System Components

1. **Detection Layer**
   - Camera Node: Video capture
   - YOLO Node: Person detection (COCO dataset)

2. **Tracking Layer**
   - Person Tracker: ByteTrack multi-object tracking
   - Persistent track ID assignment

3. **State Management Layer**
   - Person State Manager: Centralized world model
   - Service-based state queries and updates

4. **Recognition Layer**
   - Conditional Face Recognition: On-demand processing
   - Face embedding extraction (ONNX optimized)
   - Database matching service

5. **Coordination Layer**
   - Identification Coordinator: Smart triggering logic
   - Rate limiting and quality checks

6. **Visualization Layer**
   - Visualization Node: Annotated video output
   - RViz2 integration

## Quick Start

### Prerequisites

**For DevContainer Setup (Recommended)**:
- Visual Studio Code
- Docker (docker.io on Linux)
- VS Code Dev Containers extension

**For Manual Setup**:
- ROS2 Humble
- Python 3.10+
- Docker (optional)

### Installation with DevContainer (Recommended)

The project uses DevContainers for automated setup with proper ROS2 Humble and Python environments.

```bash
# 1. Clone the repository
cd ~/Documents
git clone <repository-url> ball-e
cd ball-e

# 2. Open in VS Code
code .

# 3. When prompted, click "Reopen in Container"
#    Or use: Ctrl+Shift+P -> "Dev Containers: Rebuild and Reopen in Container"
```

**What happens automatically:**
- Docker container with ROS2 Humble is built from `.devcontainer/Dockerfile`
- Python 3.10+ virtual environment (`ml_env`) is created at `/home/ubuntu/ml_env`
- ML dependencies (PyTorch, Ultralytics, etc.) are installed from `.devcontainer/requirements.txt`
- ROS2 workspace is built using `colcon_build.py` which manages two environments:
  - **System packages**: Built with `/usr/bin/python3`
  - **ML packages**: Built with `/home/ubuntu/ml_env/bin/python3` (for perception nodes)
- All packages are automatically compiled and ready to use

**Important Notes:**
- The `ml_env` virtual environment is automatically sourced when needed by ML nodes
- No manual environment activation required
- The build system (`.devcontainer/colcon_build.py`) detects package groups via `<group_depend>` tags in `package.xml`

### Manual Installation

```bash
# Clone the repository
cd ~/Documents
git clone <repository-url> ball-e
cd ball-e

# Build the workspace
cd ros2_ws
colcon build
source install/setup.bash
```

### Running the System

```bash
# Launch the full system
ros2 launch robot_bringup ball_e_full_system_launch.py

# Launch with RViz2 visualization
ros2 launch robot_bringup ball_e_full_system_launch.py &
rviz2 -d src/robot_bringup/rviz/ball_e_tracking.rviz
```

### Enrolling People

```bash
# Add a new person to the database
ros2 run interaction_pkg enroll_face_cli

# Follow the prompts to capture face and enter name
```

## Performance

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| System Throughput | ~1 FPS | ~30 FPS | **30x** |
| Face Recognition Rate | 30 FPS | 0.1-2 FPS | On-demand only |
| Identification Latency | N/A | <200ms | Real-time |
| CPU Usage | 100% | ~40% | 60% reduction |

## Configuration

Key parameters can be adjusted via launch file arguments:

```bash
ros2 launch robot_bringup ball_e_full_system_launch.py \
    max_age:=40 \
    min_hits:=2 \
    iou_threshold:=0.4 \
    max_requests_per_second:=3.0 \
    confidence_threshold:=0.6 \
    reidentification_interval:=30.0
```

See [QUICKSTART.md](QUICKSTART.md) for detailed parameter descriptions.

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: Getting started guide
- **[API_REFERENCE.md](API_REFERENCE.md)**: Topics, services, and messages
- **[docs/TRACKING.md](docs/TRACKING.md)**: Person tracking system
- **[docs/STATE_MANAGEMENT.md](docs/STATE_MANAGEMENT.md)**: State manager details
- **[docs/RECOGNITION.md](docs/RECOGNITION.md)**: Face recognition system
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**: Common issues and solutions

## Project Structure

```
ball-e/
├── ros2_ws/
│   └── src/
│       ├── sensors_pkg/           # Camera node
│       ├── perception_pkg/        # Tracking, recognition, visualization
│       │   ├── person_tracker.py
│       │   ├── person_state_manager.py
│       │   ├── face_recognition_conditional.py
│       │   ├── identification_coordinator.py
│       │   └── visualization_node.py
│       ├── interaction_pkg/       # People database, enrollment
│       ├── msgs_interfaces/       # Custom messages and services
│       └── robot_bringup/         # Launch files and configs
├── docs/                          # Documentation
├── README.md                      # This file
├── QUICKSTART.md                  # Quick start guide
└── API_REFERENCE.md              # API documentation
```

## Topics

### Subscribed Topics

- `/camera/image_raw` (sensor_msgs/Image): Camera feed
- `/yolo/detections` (vision_msgs/Detection2DArray): YOLO detections

### Published Topics

- `/person_tracker/tracks` (PersonTrackArray): Tracked persons with IDs
- `/person_state/all` (PersonStateArray): Complete person states
- `/face_recognition/identity_update` (IdentityUpdate): Recognition results
- `/visualization/annotated_image` (sensor_msgs/Image): Annotated video

### Services

- `/person_state/get_info` (GetPersonInfo): Query person by track_id
- `/person_state/request_identification` (RequestIdentification): Request face recognition
- `/person_state/update_identity` (UpdateIdentity): Update person identity
- `people_db/recognize_face` (RecognizeFace): Face matching service

See [API_REFERENCE.md](API_REFERENCE.md) for complete API documentation.

## Development

### Running Individual Nodes

```bash
# Person tracker only
ros2 launch robot_bringup person_tracker_launch.py

# State manager only
ros2 launch robot_bringup person_state_manager_launch.py

# Conditional face recognition only
ros2 launch robot_bringup face_recognition_conditional_launch.py
```

### Monitoring

```bash
# View person states
ros2 topic echo /person_state/all

# View identity updates
ros2 topic echo /face_recognition/identity_update

# View tracked persons
ros2 topic echo /person_tracker/tracks

# Check node status
ros2 node list
ros2 node info /person_state_manager
```

## Future Enhancements

- **Audio Integration**: Microphone array for sound source localization
- **Emotion Recognition**: Per-person emotion modeling
- **Multi-camera Fusion**: Track persons across multiple cameras
- **Trajectory Prediction**: Anticipate person movement
- **Social Behaviors**: Context-aware interaction
- **Voice Interaction**: TTS/STT integration

## License

Apache 2.0

## Authors

Ball-e Team - Social Robotics Perception System

---

**Status**: Production Ready ✓
**Last Updated**: 2025-01-21
**ROS2 Version**: Humble