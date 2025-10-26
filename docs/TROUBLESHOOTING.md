# Ball-e Troubleshooting Guide

Common issues and their solutions for the Ball-e tracking and identification system.

## Table of Contents

- [System Won't Start](#system-wont-start)
- [Camera Issues](#camera-issues)
- [Detection Problems](#detection-problems)
- [Tracking Issues](#tracking-issues)
- [Recognition Problems](#recognition-problems)
- [Performance Issues](#performance-issues)
- [RViz Visualization](#rviz-visualization)
- [Database Issues](#database-issues)

---

## System Won't Start

### Build Errors

**Symptom**: `colcon build` fails with errors

**Solutions**:

```bash
# Clean and rebuild
cd ~/Documents/ball-e/ros2_ws
rm -rf build install log
colcon build --symlink-install

# Check for missing dependencies
rosdep install --from-paths src --ignore-src -r -y

# Verify ROS2 environment
source /opt/ros/humble/setup.bash
```

### Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'scipy'`

**Solutions**:

```bash
# Install Python dependencies
pip3 install scipy numpy onnxruntime opencv-python ros2-numpy

# In Docker container
source ~/ml_env/bin/activate
pip3 install scipy numpy onnxruntime opencv-python ros2-numpy
```

### Node Won't Start

**Symptom**: Node crashes immediately after launch

**Solutions**:

1. Check logs:
```bash
ros2 run rqt_console rqt_console
```

2. Enable debug logging:
```bash
ros2 run perception_pkg person_tracker --ros-args --log-level debug
```

3. Verify topic availability:
```bash
ros2 topic list
```

---

## Camera Issues

### No Video Feed

**Symptom**: `/camera/image_raw` topic not publishing

**Diagnosis**:

```bash
# Check camera devices
ls /dev/video*

# Test camera directly
ros2 run sensors_pkg camera_node

# Check topic
ros2 topic hz /camera/image_raw
ros2 topic echo /camera/image_raw --field height
```

**Solutions**:

1. **Wrong device**: Check camera device number
```python
# In camera_node.py, verify:
cap = cv2.VideoCapture(0)  # Try 0, 1, 2, etc.
```

2. **Permission denied**:
```bash
sudo usermod -a -G video $USER
# Logout and login again
```

3. **Camera in use**:
```bash
# Check what's using the camera
sudo lsof /dev/video0

# Kill blocking processes
pkill -f camera
```

### Poor Image Quality

**Symptom**: Blurry or dark images

**Solutions**:

```bash
# Adjust camera settings
v4l2-ctl --device=/dev/video0 --set-ctrl=brightness=128
v4l2-ctl --device=/dev/video0 --set-ctrl=exposure_auto=3
```

---

## Detection Problems

### No YOLO Detections

**Symptom**: `/yolo/detections` empty or not publishing

**Diagnosis**:

```bash
# Check YOLO output
ros2 topic echo /yolo/detections

# View YOLO annotated image
ros2 run rqt_image_view rqt_image_view /yolo/image_detections
```

**Solutions**:

1. **Model not loaded**:
```bash
# Check model path
ls ~/Documents/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/

# Verify YOLO model exists
```

2. **No people in frame**:
- Position a person in front of camera
- Ensure good lighting
- Check detection confidence threshold

3. **YOLO crash**:
```bash
# Check YOLO logs
ros2 node info /yolo_node

# Restart YOLO node
ros2 run perception_pkg yolo_node
```

### Low Detection Confidence

**Symptom**: Detections have very low confidence scores

**Solutions**:

1. **Improve lighting**
2. **Reduce distance to camera**
3. **Adjust YOLO threshold** (in launch file or node code)
4. **Use better YOLO model** (YOLOv8 instead of YOLOv5)

---

## Tracking Issues

### No Tracks Published

**Symptom**: `/person_tracker/tracks` empty despite detections

**Diagnosis**:

```bash
# Check tracker is receiving detections
ros2 topic echo /yolo/detections

# Check tracker output
ros2 topic echo /person_tracker/tracks

# Enable debug logging
ros2 run perception_pkg person_tracker --ros-args --log-level debug
```

**Solutions**:

1. **Detections filtered out**: Person class ID mismatch
```bash
# Verify YOLO publishes person class (class 0 or "person")
ros2 topic echo /yolo/detections --field detections[0].results[0].hypothesis.class_id
```

2. **Not enough hits**:
```bash
# Reduce min_hits threshold
ros2 launch robot_bringup person_tracker_launch.py min_hits:=1
```

3. **Detection confidence too low**:
```bash
# Lower confidence thresholds
ros2 launch robot_bringup person_tracker_launch.py \
    high_conf_threshold:=0.4 \
    low_conf_threshold:=0.05
```

### Frequent ID Switches

**Symptom**: Same person gets different track_ids

**Solutions**:

```bash
# Increase IoU threshold
ros2 launch robot_bringup person_tracker_launch.py iou_threshold:=0.45

# Increase max_age to keep tracks alive longer
ros2 launch robot_bringup person_tracker_launch.py max_age:=50

# Both
ros2 launch robot_bringup person_tracker_launch.py \
    iou_threshold:=0.45 \
    max_age:=50
```

### Tracks Disappear Quickly

**Symptom**: Tracks deleted during brief occlusions

**Solutions**:

```bash
# Increase max_age
ros2 launch robot_bringup person_tracker_launch.py max_age:=60

# Lower low_conf_threshold for better recovery
ros2 launch robot_bringup person_tracker_launch.py low_conf_threshold:=0.05
```

---

## Recognition Problems

### Face Recognition Not Working

**Symptom**: No identity updates or all unknown

**Diagnosis**:

```bash
# Check face recognition node
ros2 node list | grep face_recognition

# Check identity updates
ros2 topic echo /face_recognition/identity_update

# Check database service
ros2 service list | grep people_db
```

**Solutions**:

1. **Models not loaded**:
```bash
# Check embedding model
ls ~/Documents/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/facenet.onnx

# Download if missing
cd ~/Documents/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/
wget https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx -O facenet.onnx
```

2. **Database empty**:
```bash
# Check database
cat ~/Documents/ball-e/ros2_ws/src/interaction_pkg/people_database.json

# Enroll someone
ros2 run interaction_pkg enroll_face_cli
```

3. **Face too small/large**:
```bash
# Adjust face size limits
ros2 launch robot_bringup face_recognition_conditional_launch.py \
    min_face_size:=15 \
    max_face_size:=500
```

### Low Recognition Confidence

**Symptom**: Identities detected but with low confidence

**Solutions**:

1. **Re-enroll with better quality**:
- Better lighting
- Face directly facing camera
- Neutral expression
- Remove glasses if possible

2. **Lower recognition threshold**:
```bash
ros2 launch robot_bringup face_recognition_conditional_launch.py \
    recognition_threshold:=0.5
```

3. **Multiple enrollment samples**: Enroll same person multiple times with different angles/expressions

### Identifications Not Triggered

**Symptom**: New tracks appear but no face recognition triggered

**Diagnosis**:

```bash
# Check face recognition node
ros2 node info /face_recognition_conditional

# Check person states
ros2 topic echo /person_state/all
```

**Solutions**:

1. **Face recognition node not running**:
```bash
ros2 run perception_pkg face_recognition_conditional
```

2. **Auto-identification disabled**:
```bash
# Ensure auto_identify_new_tracks is true
ros2 param set /face_recognition_conditional auto_identify_new_tracks true
```

3. **Re-identification interval too long**:
```bash
# Reduce re-identification interval (default: 30.0s)
ros2 param set /face_recognition_conditional reidentification_interval 15.0
```

---

## Performance Issues

### Low FPS

**Symptom**: System running at <10 FPS

**Diagnosis**:

```bash
# Check topic rates
ros2 topic hz /camera/image_raw
ros2 topic hz /yolo/detections
ros2 topic hz /person_tracker/tracks

# Check CPU usage
top
```

**Solutions**:

1. **YOLO bottleneck**:
- Use smaller YOLO model (YOLOv5n instead of YOLOv5s)
- Enable GPU acceleration
- Reduce camera resolution

2. **Face recognition overload**:
```bash
# Reduce identification rate
ros2 launch robot_bringup ball_e_full_system_launch.py \
    max_requests_per_second:=1.0
```

3. **Visualization overhead**:
```bash
# Reduce visualization publish rate
ros2 param set /visualization_node publish_rate 5.0
```

### High CPU Usage

**Symptom**: CPU at 100%

**Solutions**:

```bash
# Reduce frame cache size
ros2 param set /face_recognition_conditional frame_cache_size 5

# Lower state manager publish rate
ros2 param set /person_state_manager publish_rate 5.0

# Disable visualization if not needed
ros2 lifecycle set /visualization_node shutdown
```

### High Memory Usage

**Symptom**: Memory grows over time

**Solutions**:

1. **Track buildup**:
```bash
# Reduce max_age
ros2 param set /person_tracker max_age 20

# Reduce cleanup_timeout
ros2 param set /person_state_manager cleanup_timeout 3.0
```

2. **Frame cache**:
```bash
# Reduce cache size
ros2 param set /face_recognition_conditional frame_cache_size 5
```

---

## RViz Visualization

### No Image in RViz

**Symptom**: RViz shows no image in Image display

**Solutions**:

1. **Check topic name**:
   - Verify topic is `/visualization/annotated_image`
   - Check "Reliability Policy" is set to "Reliable"

2. **Topic not publishing**:
```bash
ros2 topic hz /visualization/annotated_image
ros2 topic echo /visualization/annotated_image --field height
```

3. **Restart RViz**:
```bash
pkill rviz2
rviz2 -d ~/Documents/ball-e/ros2_ws/src/robot_bringup/rviz/ball_e_tracking.rviz
```

### Laggy Visualization

**Symptom**: RViz display is slow or choppy

**Solutions**:

1. **Reduce image size** in camera node
2. **Lower publish rate** for visualization
3. **Use** `Unreliable` **reliability policy** in RViz

---

## Database Issues

### Cannot Add Person

**Symptom**: Enrollment fails

**Diagnosis**:

```bash
# Check database node
ros2 node info /people_database_node

# Check database file
ls -lh ~/Documents/ball-e/ros2_ws/src/interaction_pkg/people_database.json
```

**Solutions**:

1. **Database file locked**:
```bash
# Check file permissions
chmod 644 ~/Documents/ball-e/ros2_ws/src/interaction_pkg/people_database.json
```

2. **Corrupted database**:
```bash
# Backup and reset
mv people_database.json people_database.json.bak
echo '{"people": []}' > people_database.json
```

### Database Service Not Available

**Symptom**: `people_db/recognize_face` service not found

**Solutions**:

```bash
# Check if database node is running
ros2 node list | grep people_database

# Start database node
ros2 run interaction_pkg people_database_node

# Check service list
ros2 service list | grep people_db
```

---

## Getting More Help

If issues persist:

1. **Collect logs**:
```bash
ros2 run rqt_console rqt_console  # Save logs
```

2. **System information**:
```bash
ros2 doctor --report
ros2 wtf
```

3. **Check GitHub issues**: Search for similar problems
4. **Enable verbose logging**: `--ros-args --log-level debug`
5. **Open an issue** with:
   - ROS2 version
   - System specs
   - Error logs
   - Steps to reproduce

---

**Pro Tip**: Most issues can be diagnosed by checking node logs and topic data. Always start with:
```bash
ros2 node list
ros2 topic list
ros2 topic hz <topic>
ros2 run rqt_console rqt_console
```
