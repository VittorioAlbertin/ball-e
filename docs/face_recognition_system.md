# Ball-E Person Tracking and Recognition System

## System Overview

The Ball-E system provides real-time person tracking, face recognition, and identity management with a modular, high-performance architecture. It uses ByteTrack for persistent person tracking, on-demand face recognition for efficiency, and centralized state management for coordinating all person-related information.

## Architecture

### Component Diagram

```
┌─────────────┐
│   Camera    │
└──────┬──────┘
       │ /camera/image_raw
       ↓
┌─────────────┐
│    YOLO     │  (Object Detection)
└──────┬──────┘
       │ /yolo/detections
       ↓
┌─────────────┐
│   Person    │  (ByteTrack - Persistent Track IDs)
│   Tracker   │
└──────┬──────┘
       │ /person_tracker/tracks
       ↓
┌─────────────┐
│   Person    │  (Centralized State Management)
│State Manager│  (Combines tracking + identity)
└──┬────┬─────┘
   │    │ /person_state/all
   │    └────────────────────────────┐
   │                                  ↓
   │                          ┌───────────────┐
   │                          │Visualization  │
   │                          │     Node      │
   │                          └───────────────┘
   │                           /visualization/annotated_image
   ↓
┌──────────────────┐
│ Identification   │  (Smart Triggering)
│  Coordinator     │
└────────┬─────────┘
         │ (triggers)
         ↓
┌─────────────────┐
│Face Recognition │  (On-Demand, <200ms)
│  (Conditional)  │
└────────┬────────┘
         │ /face_recognition/identity_update
         ├──────────────────┐
         ↓                  ↓
┌─────────────┐      ┌──────────────┐
│   People    │      │    Person    │
│  Database   │←─────│     State    │
│    Node     │      │   Manager    │
└─────────────┘      └──────────────┘
```

### Data Flow

1. **Camera** → Raw images
2. **YOLO** → Person detections
3. **Person Tracker** → Persistent track IDs (ByteTrack)
4. **Person State Manager** → Centralized person state (tracking + identity)
5. **Identification Coordinator** → Smart triggering (new tracks, confidence decay, periodic)
6. **Face Recognition** → On-demand embedding extraction + matching
7. **People Database** → Face embedding storage + similarity matching
8. **Visualization** → Annotated video with track IDs and identities

## Nodes

### perception_pkg

| Node | Description | Documentation |
|------|-------------|---------------|
| `yolo_node` | Real-time object detection (YOLO) | [yolo_node.md](../ros2_ws/src/perception_pkg/docs/yolo_node.md) |
| `person_tracker` | ByteTrack multi-person tracking | [person_tracker.md](../ros2_ws/src/perception_pkg/docs/person_tracker.md) |
| `person_state_manager` | Centralized person state management | [person_state_manager.md](../ros2_ws/src/perception_pkg/docs/person_state_manager.md) |
| `identification_coordinator` | Smart identification triggering | [identification_coordinator.md](../ros2_ws/src/perception_pkg/docs/identification_coordinator.md) |
| `face_recognition_conditional` | On-demand face recognition | [face_recognition_conditional.md](../ros2_ws/src/perception_pkg/docs/face_recognition_conditional.md) |
| `visualization_node` | Annotated video output | [visualization_node.md](../ros2_ws/src/perception_pkg/docs/visualization_node.md) |

### interaction_pkg

| Node | Description | Documentation |
|------|-------------|---------------|
| `people_database_node` | Face storage and recognition service | [people_database_node.md](../ros2_ws/src/interaction_pkg/docs/people_database_node.md) |

## Quick Start

### 1. Launch All Nodes

```bash
# In Docker container
cd /ball-e/ros2_ws
source install/setup.bash
ros2 launch robot_bringup ball_e_full_system_launch.py
```

This launches:
- Camera node
- YOLO detection node
- Person tracker (ByteTrack)
- Person state manager
- Identification coordinator
- Face recognition (conditional/on-demand)
- People database node
- Visualization node

### 2. View Visualization (RViz)

```bash
# Open RViz
rviz2

# Add Image displays:
# - /yolo/image_detections (object detection)
# - /face/debug_image (face recognition)
```

### 3. Enroll a Person

When a person is tracked, check the visualization for their track ID.

Enroll by track ID in a new terminal:
```bash
# Example: If person has track ID 1
ros2 run interaction_pkg enroll_by_track_id 1 "Your Name" "Optional notes"
```

You should see:
```
Waiting for identification to complete...
[INFO] Captured embedding for track_id=1
✓ SUCCESS: Successfully added Your Name with ID 1
```

### 4. Verify Recognition

The person's name should now appear on the bounding box in the visualization.

## Topic Reference

### Image Topics

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | 30 Hz | Raw camera feed (BGR8) |
| `/yolo/image_detections` | `sensor_msgs/Image` | 30 Hz | YOLO visualization |
| `/face/debug_image` | `sensor_msgs/Image` | 10-20 Hz | Face recognition visualization |

### Detection Topics

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/yolo/detections` | `vision_msgs/Detection2DArray` | 30 Hz | Object detections |
| `/face/detections` | `vision_msgs/Detection2DArray` | 10-20 Hz | Face bounding boxes |
| `/face/recognition` | `msgs_interfaces/FaceRecognition` | 10-20 Hz | Recognition results |

## Service Reference

### People Database Services

| Service | Purpose |
|---------|---------|
| `people_db/add_person` | Add person to database |
| `people_db/recognize_face` | Match face embedding |
| `people_db/get_person` | Retrieve person info |
| `people_db/update_last_seen` | Update interaction time |
| `people_db/update_preferences` | Update user preferences |
| `people_db/get_all_people` | List all people |
| `people_db/delete_person` | Remove person |

### Enrollment Services

| Service | Purpose |
|---------|---------|
| `enroll_pending_face` | Enroll unknown face with name |

## Configuration

### Key Parameters

#### YOLO Node
- Model: YOLOv5 Nano (automatic download)
- Input size: 640x640
- Device: Auto-select GPU/CPU

#### Face Detection Node
```yaml
face_confidence_threshold: 0.6    # Face detection confidence
recognition_threshold: 0.6        # Face matching threshold
nms_iou_threshold: 0.3           # Duplicate removal threshold
```

#### Face Enrollment Node
```yaml
cooldown_seconds: 10.0           # Time between enrollment prompts
min_confidence: 0.5              # Minimum confidence to offer enrollment
```

#### People Database Node
```yaml
db_path: /ball-e/ros2_ws/robot_data/people.db  # SQLite database location
```

### Launch File Configuration

Edit `ros2_ws/src/robot_bringup/launch/ball_e_full_system_launch.py`:

```python
# Example: Change recognition parameters
Node(
    package='perception_pkg',
    executable='face_recognition_conditional',
    parameters=[
        {'recognition_threshold': 0.5},           # Lower = easier to match
        {'reidentification_interval': 30.0},      # Re-identify every N seconds
        {'auto_identify_new_tracks': True}        # Auto-identify new people
    ]
)

# Example: Change tracker parameters
Node(
    package='perception_pkg',
    executable='person_tracker',
    parameters=[
        {'max_age': 30},              # Frames to keep lost tracks
        {'min_hits': 3},              # Confirmations needed for new track
        {'iou_threshold': 0.3}        # Matching threshold
    ]
)
```

## Performance Tuning

### Frame Rate Optimization

| Action | Effect | Trade-off |
|--------|--------|-----------|
| Use GPU | +200% FPS | Requires CUDA |
| Lower camera resolution | +50% FPS | Lower quality |
| Increase `nms_iou_threshold` | +10% FPS | More duplicates |
| Increase confidence thresholds | +5% FPS | Fewer detections |

### Recognition Accuracy

| Action | Effect | Trade-off |
|--------|--------|-----------|
| Lower `recognition_threshold` | More matches | More false positives |
| Raise `face_confidence_threshold` | Better quality faces | Fewer detections |
| Re-enroll with multiple angles | Better recognition | More database entries |
| Good lighting | Much better accuracy | Environmental |

## Visualization Color Codes

### YOLO Detections
- Different colors for different object classes
- Label shows class name and confidence

### Face Recognition
- **Yellow/Cyan**: "Detecting..." (recognition in progress)
- **Green**: Recognized person (shows name)
- **Red**: Unknown person (not in database)

## Common Workflows

### Workflow 1: Initial Setup

```bash
# 1. Launch system
ros2 launch robot_bringup ball_e_full_system_launch.py

# 2. View visualization
ros2 run rqt_image_view rqt_image_view /visualization/annotated_image

# 3. Position yourself in front of camera

# 4. Check visualization for your track ID (e.g., "ID:1")

# 5. Enroll by track ID in new terminal
ros2 run interaction_pkg enroll_by_track_id 1 "Your Name"

# 6. Verify recognition - your name appears on bounding box
```

### Workflow 2: Adding Multiple People

```bash
# Person 1 in front of camera
ros2 run interaction_pkg enroll_face "Alice"

# Person 2 in front of camera (wait 10 seconds)
ros2 run interaction_pkg enroll_face "Bob"

# Person 3 in front of camera (wait 10 seconds)
ros2 run interaction_pkg enroll_face "Charlie"
```

### Workflow 3: Database Management

```bash
# List all people
ros2 service call /people_db/get_all_people msgs_interfaces/srv/GetAllPeople

# Get person details
ros2 service call /people_db/get_person msgs_interfaces/srv/GetPerson "{person_id: 1}"

# Delete person (careful!)
ros2 service call /people_db/delete_person msgs_interfaces/srv/DeletePerson "{person_id: 5}"
```

### Workflow 4: Backup Database

```bash
# Backup current database
cp /ball-e/ros2_ws/robot_data/people.db \
   /ball-e/ros2_ws/robot_data/people_backup_$(date +%Y%m%d).db

# List backups
ls -lh /ball-e/ros2_ws/robot_data/people_backup_*.db

# Restore from backup
cp /ball-e/ros2_ws/robot_data/people_backup_20250101.db \
   /ball-e/ros2_ws/robot_data/people.db
```

## Troubleshooting

### No Faces Detected

**Symptoms**: No bounding boxes on `/face/debug_image`

**Check**:
1. ✓ YOLO detecting people? (check `/yolo/image_detections`)
2. ✓ Face visible and front-facing?
3. ✓ Good lighting?
4. ✓ Face confidence threshold not too high?

**Solution**:
```python
# Lower face confidence threshold
{'face_confidence_threshold': 0.3}  # Default: 0.6
```

### Faces Not Recognized

**Symptoms**: Red boxes with "Unknown" label

**Check**:
1. ✓ Person enrolled in database?
2. ✓ Face angle similar to enrolled image?
3. ✓ Lighting conditions similar?

**Solution**:
```python
# Lower recognition threshold
{'recognition_threshold': 0.5}  # Default: 0.6

# Or re-enroll with current conditions
ros2 run interaction_pkg enroll_face "PersonName"
```

### Slow Performance

**Symptoms**: <5 FPS on `/face/debug_image`

**Check**:
1. ✓ Using GPU? (check logs for "Using device: cuda")
2. ✓ Camera resolution reasonable? (<1080p)
3. ✓ Multiple people in frame?

**Solution**:
- Enable CUDA if available
- Reduce camera resolution
- Limit field of view to fewer people

### "Detecting..." Never Changes

**Symptoms**: Yellow boxes stay "Detecting..." forever

**Check**:
1. ✓ People database node running?
2. ✓ Service connectivity? (`ros2 service list | grep people_db`)
3. ✓ Any errors in face_detection_node logs?

**Solution**:
```bash
# Restart database node
ros2 run interaction_pkg people_database_node

# Check service availability
ros2 service list | grep people_db
```

### Enrollment Fails

**Symptoms**: "No pending face to enroll" error

**Check**:
1. ✓ Unknown face detected within last 60 seconds?
2. ✓ Face enrollment node running?
3. ✓ Enrollment prompt appeared in logs?

**Solution**:
- Re-detect unknown face (move in front of camera)
- Wait for enrollment prompt
- Enroll within 60 seconds

## Advanced Topics

### Custom Embedding Models

To use a different face embedding model:

1. Download ONNX model
2. Update `embedding_model_path` parameter
3. Ensure output is normalized 512-dim vector

### Multi-Camera Setup

For multiple cameras:

```python
# Launch separate face detection nodes per camera
Node(
    package='perception_pkg',
    executable='face_detection_node',
    name='face_detection_camera1',
    parameters=[...],
    remappings=[
        ('/camera/image_raw', '/camera1/image_raw'),
        ('/yolo/detections', '/camera1/yolo/detections'),
    ]
)
```

### Integration with Other Systems

The face recognition system exposes standard ROS2 topics and services that can be integrated with:

- Navigation systems (approach recognized people)
- Dialog systems (personalized greetings)
- Access control (door unlocking)
- Analytics (visitor tracking)

## Security Considerations

### Data Privacy
- Face embeddings stored locally only
- No cloud transmission
- Embeddings are irreversible (cannot reconstruct face)
- Consider encryption for production environments

### Access Control
- Database has no authentication (local services only)
- Add service authentication for multi-user systems
- Implement audit logging for data access

### GDPR Compliance
- Implement data retention policies
- Provide deletion functionality (already included)
- Obtain consent before enrollment
- Document data usage in privacy policy

## Performance Benchmarks

### Typical Performance (on GPU)

| Component | Metric | Value |
|-----------|--------|-------|
| YOLO | FPS | 30-50 |
| Face Detection | ms per face | 10-20 |
| Face Recognition | ms per face | 20-30 |
| Overall | FPS | 15-25 |

### Database Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Add person | 1-5 ms | Per person |
| Recognize face | 10-50 ms | Linear with DB size |
| Get person | 1 ms | By ID |

## Support

### Reporting Issues
- Check logs for error messages
- Verify all nodes are running
- Test with known good conditions
- Document steps to reproduce

### Further Documentation
- Individual node docs in package `docs/` folders
- ROS2 humble documentation: https://docs.ros.org/
- YOLO documentation: https://docs.ultralytics.com/

## Changelog

### v1.0 (Current)
- Initial release
- YOLO person detection
- Face detection with UltraFace
- Face recognition with ArcFace embeddings
- SQLite database storage
- Interactive enrollment system
- Real-time visualization
- NMS for duplicate removal
- Async recognition pipeline
