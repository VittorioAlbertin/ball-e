# Ball-E Documentation

Welcome to the Ball-E robot documentation. This directory contains comprehensive documentation for all system components.

## Table of Contents

1. [Pipeline Flow](PIPELINE_FLOW.md) - Complete step-by-step vision pipeline walkthrough
2. [Person Tracking](TRACKING.md) - ByteTrack algorithm and tracking system details
3. [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
4. [Node Documentation](#node-documentation) - Individual node references
5. [Quick Start](#quick-start)
6. [Architecture](#architecture)

## Quick Start

### Launch the System

```bash
# In Docker container
cd /ball-e/ros2_ws
source install/setup.bash
ros2 launch robot_bringup ball_e_full_system_launch.py
```

### Enroll Your Face

1. Stand in front of camera
2. Check the visualization to see your track ID (e.g., "ID:1" on the bounding box)
3. Run: `ros2 run interaction_pkg enroll_by_track_id 1 "Your Name"`
4. You should now be recognized and displayed with your name!

### View Visualization

The system publishes annotated video with tracking and identification info:

```bash
# View in RViz2 or rqt_image_view
ros2 run rqt_image_view rqt_image_view /visualization/annotated_image

# Or use RViz2
rviz2
# Add Image display for: /visualization/annotated_image
```

## Node Documentation

### Perception Nodes

| Node | Package | Description | Documentation |
|------|---------|-------------|---------------|
| YOLO Node | perception_pkg | Object detection (YOLO) | [docs](../ros2_ws/src/perception_pkg/docs/yolo_node.md) |
| Person Tracker | perception_pkg | ByteTrack multi-person tracking | [docs](../ros2_ws/src/perception_pkg/docs/person_tracker.md) |
| Person State Manager | perception_pkg | Centralized person state | [docs](../ros2_ws/src/perception_pkg/docs/person_state_manager.md) |
| Face Recognition (Conditional) | perception_pkg | On-demand face recognition | [docs](../ros2_ws/src/perception_pkg/docs/face_recognition_conditional.md) |
| Identification Coordinator | perception_pkg | Smart identification triggers | [docs](../ros2_ws/src/perception_pkg/docs/identification_coordinator.md) |
| Visualization Node | perception_pkg | Annotated video output | [docs](../ros2_ws/src/perception_pkg/docs/visualization_node.md) |

### Interaction Nodes

| Node | Package | Description | Documentation |
|------|---------|-------------|---------------|
| People Database Node | interaction_pkg | Face storage & recognition | [docs](../ros2_ws/src/interaction_pkg/docs/people_database_node.md) |

## Architecture

```
Camera → YOLO → Person Tracker → Person State Manager → Identification Coordinator
                      ↓                    ↓                        ↓
                 ByteTrack         Centralized State      Face Recognition (conditional)
                                                                    ↓
                                                            People Database

                                        Visualization Node (annotated video)
```

### Key Features

- ✅ Real-time person tracking with persistent IDs (ByteTrack)
- ✅ On-demand face recognition (30 FPS system throughput)
- ✅ Centralized person state management
- ✅ Smart identification coordination
- ✅ Persistent people database (SQLite)
- ✅ Track-based enrollment system
- ✅ Rich visualization with track IDs and identities
- ✅ Modular and extensible architecture

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
- `/visualization/annotated_image` - Full tracking and identification visualization

### Detection Topics
- `/yolo/detections` - Object detection results (Detection2DArray)
- `/person_tracker/tracks` - Person tracking with persistent IDs (PersonTrackArray)
- `/person_state/all` - Complete person states (PersonStateArray)
- `/face_recognition/identity_update` - Identity update events (IdentityUpdate)

## Services Overview

### Database Services
- `people_db/add_person` - Add person to database
- `people_db/recognize_face` - Match face embedding
- `people_db/get_person` - Get person details
- `people_db/update_last_seen` - Update interaction time
- `people_db/update_preferences` - Update preferences
- `people_db/get_all_people` - List all people
- `people_db/delete_person` - Remove person

### State Management Services
- `/person_state/update_identity` - Update person identity
- `/person_state/request_identification` - Request face recognition for track

## Common Commands

### Launch
```bash
# Full system (recommended - includes all components)
ros2 launch robot_bringup ball_e_full_system_launch.py

# Perception + Recognition only
ros2 launch robot_bringup complete_pipeline_launch.py

# Perception only (no face recognition)
ros2 launch robot_bringup perception_pipeline_launch.py

# Minimal (Camera + YOLO only)
ros2 launch robot_bringup camera_yolo_launch.py

# Individual nodes (advanced usage)
ros2 run perception_pkg yolo_node
ros2 run perception_pkg person_tracker
ros2 run perception_pkg person_state_manager
ros2 run perception_pkg face_recognition_conditional
ros2 run interaction_pkg people_database_node
```

### Enrollment
```bash
# Enroll person by track ID
ros2 run interaction_pkg enroll_by_track_id <track_id> "Name" "Optional notes"

# Example: Enroll track ID 1 as "John Doe"
ros2 run interaction_pkg enroll_by_track_id 1 "John Doe" "Friend from work"
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

### Issue: Person not being tracked
**Solution**:
- Check YOLO detections are working
- Adjust `min_hits` parameter (default: 3 confirmations needed)
- Verify person is within camera view

### Issue: Faces not recognized
**Solution**:
- Check `recognition_threshold` parameter (default 0.75)
- Ensure face is visible and looking at camera
- Re-enroll with better quality face capture
- Check logs for similarity scores

### Issue: Slow performance
**Solution**:
- System is optimized for 30 FPS by default
- Face recognition is on-demand (not every frame)
- Check logs for actual bottlenecks
- Reduce camera resolution if needed

### Issue: Enrollment fails
**Solution**:
- Verify track ID is correct from visualization
- Ensure person is facing camera
- Check that face is in top portion of person box
- Look at logs for face extraction details

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

## Visualization Guide

### Track Display
- **Track ID**: Shown as "ID:X" on bounding box
- **Identity**: Person's name if recognized, "Unknown" if not, "Identifying..." during recognition
- **Status**: [NEW], [LOW-CONF], [MISS:X] for tracking status
- **Confidence Bar**: Visual indicator below bounding box

### Color Coding
- **Bright colors**: Identified persons (brighter when confident)
- **Yellow tint**: Currently identifying
- **Dimmer colors**: Unknown persons
- **Consistent colors**: Each track ID keeps same color

### Statistics Overlay
- **Tracked**: Total number of active tracks
- **Identified**: Number of recognized persons
- **Unknown**: Number of unidentified persons
- **Pending ID**: Number waiting for identification

## Configuration Files

### Launch Files
- `ros2_ws/src/robot_bringup/launch/ball_e_full_system_launch.py` - Complete system (recommended)
- `ros2_ws/src/robot_bringup/launch/complete_pipeline_launch.py` - Perception + Face recognition
- `ros2_ws/src/robot_bringup/launch/perception_pipeline_launch.py` - Camera + YOLO + Tracking + State
- `ros2_ws/src/robot_bringup/launch/camera_yolo_tracker_launch.py` - Camera + YOLO + Tracker
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
3. Update PIPELINE_FLOW.md if architecture changes
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
