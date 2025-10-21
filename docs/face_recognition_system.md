# Face Recognition System Documentation

## System Overview

The Ball-E face recognition system provides real-time face detection, recognition, and person management capabilities. It consists of multiple ROS2 nodes working together to detect people, recognize known faces, and enroll new faces into a persistent database.

## Architecture

### Component Diagram

```
┌─────────────┐
│   Camera    │
│    Node     │
└──────┬──────┘
       │ /camera/image_raw
       ↓
┌─────────────┐
│    YOLO     │
│    Node     │──→ /yolo/image_detections (visualization)
└──────┬──────┘
       │ /yolo/detections
       ↓
┌─────────────┐
│Face Detection│
│    Node     │──→ /face/debug_image (visualization)
└──────┬──────┘     /face/detections
       │ /face/recognition
       ├────────────────────┐
       ↓                    ↓
┌─────────────┐      ┌─────────────┐
│  Enrollment │      │   People    │
│    Node     │─────→│  Database   │
└─────────────┘      │    Node     │
                     └─────────────┘
```

### Data Flow

1. **Image Acquisition**: Camera publishes raw images
2. **Object Detection**: YOLO detects people in images
3. **Face Detection**: Extracts faces from person ROIs
4. **Face Recognition**: Compares faces against database
5. **Enrollment**: Adds unknown faces to database on request
6. **Visualization**: Displays annotated images with recognition results

## Nodes

### perception_pkg

| Node | Description | Documentation |
|------|-------------|---------------|
| `yolo_node` | Real-time object detection (YOLO) | [yolo_node.md](../ros2_ws/src/perception_pkg/docs/yolo_node.md) |
| `face_detection_node` | Face detection and recognition | [face_detection_node.md](../ros2_ws/src/perception_pkg/docs/face_detection_node.md) |

### interaction_pkg

| Node | Description | Documentation |
|------|-------------|---------------|
| `people_database_node` | Face storage and recognition service | [people_database_node.md](../ros2_ws/src/interaction_pkg/docs/people_database_node.md) |
| `face_enrollment_node` | Unknown face enrollment interface | [face_enrollment_node.md](../ros2_ws/src/interaction_pkg/docs/face_enrollment_node.md) |

## Quick Start

### 1. Launch All Nodes

```bash
# In Docker container
cd /ball-e/ros2_ws
source install/setup.bash
ros2 launch robot_bringup robot_launch.py
```

This launches:
- Camera node
- YOLO detection node
- Face detection node
- People database node
- Face enrollment node

### 2. View Visualization (RViz)

```bash
# Open RViz
rviz2

# Add Image displays:
# - /yolo/image_detections (object detection)
# - /face/debug_image (face recognition)
```

### 3. Enroll a New Face

When an unknown face is detected, you'll see:
```
============================================================
UNKNOWN FACE DETECTED!
Confidence: 0.72
============================================================
Would you like to add this person to the database?
Example:
  ros2 run interaction_pkg enroll_face "John Doe"
============================================================
```

Enroll the face:
```bash
ros2 run interaction_pkg enroll_face "Your Name"
```

### 4. Verify Recognition

The face should now appear with a green bounding box and your name displayed.

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

Edit `ros2_ws/src/robot_bringup/launch/robot_launch.py`:

```python
# Example: Change face confidence threshold
Node(
    package='perception_pkg',
    executable='face_detection_node',
    parameters=[
        {'face_confidence_threshold': 0.7},  # Higher = fewer false positives
        {'recognition_threshold': 0.5}       # Lower = easier to match
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
ros2 launch robot_bringup robot_launch.py

# 2. Open RViz for visualization
rviz2

# 3. Position yourself in front of camera

# 4. Wait for "UNKNOWN FACE DETECTED" prompt

# 5. Enroll yourself
ros2 run interaction_pkg enroll_face "Your Name"

# 6. Verify recognition (green box with your name)
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
