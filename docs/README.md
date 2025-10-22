# Ball-E Documentation

Welcome to the Ball-E robot documentation. This directory contains comprehensive documentation for all system components.

## Table of Contents

1. [Face Recognition System](face_recognition_system.md) - Complete system overview
2. [Node Documentation](#node-documentation) - Individual node references
3. [Quick Start](#quick-start)
4. [Architecture](#architecture)

## Quick Start

### Launch the System

```bash
# In Docker container
cd /ball-e/ros2_ws
source install/setup.bash
ros2 launch robot_bringup robot_launch.py
```

### Enroll Your Face

1. Stand in front of camera
2. Wait for "UNKNOWN FACE DETECTED" message
3. Run: `ros2 run interaction_pkg enroll_face "Your Name"`
4. You should now be recognized with a green bounding box!

### Visualize in RViz

```bash
rviz2

# Add Image displays:
# - /yolo/image_detections
# - /face/debug_image
```

## Node Documentation

### Perception Nodes

| Node | Package | Description | Documentation |
|------|---------|-------------|---------------|
| YOLO Node | perception_pkg | Object detection (YOLO) | [docs](../ros2_ws/src/perception_pkg/docs/yolo_node.md) |
| Face Detection Node | perception_pkg | Face detection & recognition | [docs](../ros2_ws/src/perception_pkg/docs/face_detection_node.md) |

### Interaction Nodes

| Node | Package | Description | Documentation |
|------|---------|-------------|---------------|
| People Database Node | interaction_pkg | Face storage & recognition | [docs](../ros2_ws/src/interaction_pkg/docs/people_database_node.md) |
| Face Enrollment Node | interaction_pkg | Unknown face enrollment | [docs](../ros2_ws/src/interaction_pkg/docs/face_enrollment_node.md) |

## Architecture

```
Camera → YOLO Detection → Face Detection → Recognition
                              ↓              ↓
                         Visualization   Database
                                            ↓
                                       Enrollment
```

### Key Features

- ✅ Real-time face detection and recognition
- ✅ Persistent people database (SQLite)
- ✅ Interactive face enrollment
- ✅ Color-coded visualization
- ✅ Async processing for high frame rates
- ✅ NMS for duplicate removal
- ✅ GPU acceleration support

## System Requirements

### Hardware
- Camera (USB or compatible)
- Optional: NVIDIA GPU with CUDA (for acceleration)

### Software
- ROS2 Humble
- Python 3.10+
- Docker (recommended)

### Dependencies
- PyTorch
- OpenCV
- ONNX Runtime
- NumPy

## Topics Overview

### Image Topics
- `/camera/image_raw` - Raw camera feed
- `/yolo/image_detections` - Object detection visualization
- `/face/debug_image` - Face recognition visualization

### Detection Topics
- `/yolo/detections` - Object detection results
- `/face/detections` - Face bounding boxes
- `/face/recognition` - Recognition results with person info

## Services Overview

### Database Services
- `people_db/add_person` - Add person to database
- `people_db/recognize_face` - Match face embedding
- `people_db/get_person` - Get person details
- `people_db/update_last_seen` - Update interaction time
- `people_db/update_preferences` - Update preferences
- `people_db/get_all_people` - List all people
- `people_db/delete_person` - Remove person

### Enrollment Services
- `enroll_pending_face` - Enroll unknown face

## Common Commands

### Launch
```bash
# Full system
ros2 launch robot_bringup robot_launch.py

# Individual nodes
ros2 run perception_pkg yolo_node
ros2 run perception_pkg face_detection_node
ros2 run interaction_pkg people_database_node
ros2 run interaction_pkg face_enrollment_node
```

### Enrollment
```bash
# Enroll face
ros2 run interaction_pkg enroll_face "Name" "Optional notes"
```

### Database Management
```bash
# List all people
ros2 service call /people_db/get_all_people msgs_interfaces/srv/GetAllPeople

# Get person info
ros2 service call /people_db/get_person msgs_interfaces/srv/GetPerson "{person_id: 1}"

# Delete person
ros2 service call /people_db/delete_person msgs_interfaces/srv/DeletePerson "{person_id: 5}"
```

### Backup Database
```bash
# Backup
cp /ball-e/ros2_ws/robot_data/people.db \
   /ball-e/ros2_ws/robot_data/people_backup_$(date +%Y%m%d).db

# Restore
cp /ball-e/ros2_ws/robot_data/people_backup_YYYYMMDD.db \
   /ball-e/ros2_ws/robot_data/people.db
```

## Troubleshooting

### Issue: No faces detected
**Solution**: Lower `face_confidence_threshold` parameter (default 0.6 → 0.3)

### Issue: Faces not recognized
**Solution**: Lower `recognition_threshold` parameter (default 0.6 → 0.5)

### Issue: Slow performance
**Solution**:
- Enable GPU acceleration
- Reduce camera resolution
- Check logs for bottlenecks

### Issue: Enrollment fails
**Solution**:
- Ensure face_enrollment_node is running
- Wait for "UNKNOWN FACE DETECTED" prompt
- Enroll within 60 seconds

## Performance Tips

### Frame Rate Optimization
1. Use GPU (CUDA) if available
2. Lower camera resolution (e.g., 640x480)
3. Increase confidence thresholds
4. Limit field of view

### Recognition Accuracy
1. Good lighting conditions
2. Front-facing poses
3. Multiple enrollment angles
4. Adjust recognition threshold

## Visualization Color Codes

### Face Recognition
- **Yellow/Cyan**: Recognition in progress
- **Green**: Recognized (shows name)
- **Red**: Unknown (not in database)

### YOLO Detections
- Different colors per object class
- Label shows class name + confidence

## Configuration Files

### Launch Files
- `ros2_ws/src/robot_bringup/launch/robot_launch.py` - Main system launch
- `ros2_ws/src/robot_bringup/launch/camera_yolo_launch.py` - Camera + YOLO only

### Parameters
Edit launch files to change node parameters:
```python
Node(
    package='perception_pkg',
    executable='face_detection_node',
    parameters=[
        {'face_confidence_threshold': 0.7},
        {'recognition_threshold': 0.5}
    ]
)
```

## Database Location

- **Path**: `/ball-e/ros2_ws/robot_data/people.db`
- **Type**: SQLite3
- **Schema**: See [people_database_node.md](../ros2_ws/src/interaction_pkg/docs/people_database_node.md)

## Models

### YOLO
- **Model**: YOLOv5 Nano
- **Size**: ~4MB
- **Location**: `~/.cache/torch/hub/`
- **Download**: Automatic

### Face Detection
- **Model**: UltraFace RFB-320
- **Size**: ~1MB
- **Location**: `/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/`
- **Download**: Automatic on first run

### Face Embedding
- **Model**: ArcFace ResNet100
- **Size**: ~100MB
- **Location**: `/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/`
- **Download**: Automatic on first run

## Contributing

When adding new features:
1. Update relevant node documentation
2. Add examples to this README
3. Update face_recognition_system.md if architecture changes
4. Test all documented commands
5. Update changelog

## Support

For issues or questions:
1. Check individual node documentation
2. Review troubleshooting section
3. Check ROS logs for errors
4. Verify all dependencies installed

## License

[Add license information]

## Authors

[Add author information]

## Changelog

### v1.0 (Current)
- Initial documentation release
- Complete face recognition system
- Interactive enrollment
- Real-time visualization
- Persistent database
