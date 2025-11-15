# Ball-e Documentation

## Overview

Ball-e is an intelligent robot with multi-modal person identification capabilities, combining visual tracking, face recognition, voice recognition, and speech understanding.

## Documentation Index

### System Architecture

- **[MULTIMODAL_IDENTIFICATION.md](./MULTIMODAL_IDENTIFICATION.md)**
  - Complete multi-modal identification system architecture
  - Three-layer tracking (visual, biometric, Bayesian)
  - Bayesian identity fusion theory and implementation
  - Component descriptions and data flow
  - Configuration and tuning parameters

### User Guides

- **[ENROLLMENT_GUIDE.md](./ENROLLMENT_GUIDE.md)**
  - Step-by-step person enrollment tutorial
  - Multi-pose face capture instructions
  - Voice sample recording guide
  - Troubleshooting common issues
  - Database management commands

### API Reference

- **[AUDIO_API_REFERENCE.md](./AUDIO_API_REFERENCE.md)**
  - ROS 2 topics and message types
  - Service definitions and usage
  - Temporal Bayesian Tracker API
  - Code examples and integration patterns

## Quick Start

### 1. Build the Workspace
```bash
cd ~/Documents/ball-e/ros2_ws
colcon build
source install/setup.bash
```

### 2. Enroll a Person
```bash
# Terminal 1
ros2 launch robot_bringup enrollment_launch.py

# Terminal 2
ros2 run interaction_pkg enrollment_cli
```

### 3. Run Full Identification System
```bash
ros2 launch robot_bringup multimodal_identification_launch.py
```

## Key Features

- **Visual Tracking**: ByteTrack multi-object tracking
- **Face Recognition**: InsightFace with 512-dim embeddings
- **Voice Recognition**: Resemblyzer with 256-dim speaker embeddings
- **Speech-to-Text**: OpenAI Whisper transcription
- **Identity Fusion**: Recursive Bayesian filtering
- **Interactive Enrollment**: Multi-pose face + multi-phrase voice capture

## Dependencies

See individual documentation files for specific requirements:
- ROS 2 (Humble/Iron)
- Python 3.8+
- CUDA (optional, for GPU acceleration)

## Directory Structure

```
ball-e/
├── docs/                          # Documentation
│   ├── README.md                  # This file
│   ├── MULTIMODAL_IDENTIFICATION.md
│   ├── ENROLLMENT_GUIDE.md
│   └── AUDIO_API_REFERENCE.md
├── ros2_ws/
│   └── src/
│       ├── sensors_pkg/           # Sensor drivers (camera, microphone)
│       ├── perception_pkg/        # Recognition and tracking
│       ├── interaction_pkg/       # Database and enrollment
│       ├── msgs_interfaces/       # Custom messages and services
│       └── robot_bringup/         # Launch files
└── ...
```

## License

Apache-2.0
