# Person Tracker for Ball-e

A lightweight, real-time person tracking node for the Ball-e social robot using the ByteTrack algorithm.

## Overview

The person tracker subscribes to YOLO detections and assigns persistent track IDs to detected people across frames. This decouples person detection from face recognition, significantly improving system performance by allowing recognition to be triggered conditionally rather than every frame.

## Algorithm: ByteTrack

### Why ByteTrack?

We chose **ByteTrack** over DeepSORT for the following reasons:

1. **Lightweight & Fast**: ByteTrack uses only IoU (Intersection over Union) for tracking, avoiding the computational overhead of deep ReID networks required by DeepSORT
2. **Real-time Performance**: Critical for Ball-e given the existing ~1 FPS bottleneck with face recognition
3. **Robust to Occlusions**: ByteTrack's two-stage association handles partial occlusions and temporary detection failures better than single-threshold approaches
4. **No Visual Features Required**: Works directly with bounding boxes, making it simpler to implement and maintain
5. **State-of-the-Art Results**: Despite its simplicity, ByteTrack achieves competitive performance with more complex trackers

### How ByteTrack Works

ByteTrack employs a two-stage association strategy:

**Stage 1: High-Confidence Association**
- Match high-confidence detections (>0.6) with existing tracks using IoU
- Uses Hungarian algorithm for optimal assignment
- Updates matched tracks, creates new tracks for unmatched detections

**Stage 2: Low-Confidence Recovery**
- Match low-confidence detections (0.1-0.6) with remaining unmatched tracks
- Recovers tracks during partial occlusions or poor detection conditions
- Prevents premature track termination

**Track Lifecycle:**
- **Creation**: New track created when unmatched high-confidence detection appears
- **Confirmation**: Track confirmed after `min_hits` consecutive detections
- **Update**: Track updated when matched with detection, confidence smoothed over time
- **Deletion**: Track deleted after `max_age` frames without detection

Reference: [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)

## Architecture Integration

```
Camera Node → YOLO Node → Person Tracker → Face Recognition (conditional)
                   ↓              ↓
            /yolo/detections  /person_tracker/tracks
```

The person tracker fits into Ball-e's architecture as:
- **Input**: YOLO detections from `/yolo/detections` (Detection2DArray)
- **Output**: Persistent person tracks on `/person_tracker/tracks` (PersonTrackArray)
- **Future Integration**: Track IDs will be used by the coordinator to decide when to trigger face recognition

## Message Definitions

### PersonTrack.msg

```msg
std_msgs/Header header

int32 track_id                      # Persistent tracking ID
float32 bbox_x                      # Bounding box top-left x
float32 bbox_y                      # Bounding box top-left y
float32 bbox_w                      # Bounding box width
float32 bbox_h                      # Bounding box height
float32 tracking_confidence         # Smoothed tracking confidence (0.0-1.0)
int32 frames_since_last_seen        # Frames since last detection
bool is_new_track                   # True if track just confirmed
float32 detection_confidence        # Latest YOLO detection confidence
```

### PersonTrackArray.msg

```msg
std_msgs/Header header
PersonTrack[] tracks                # Array of active tracks
```

## Parameters

Configure via launch file arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_age` | 30 | Maximum frames to keep track alive without detection |
| `min_hits` | 3 | Minimum consecutive detections before confirming track |
| `iou_threshold` | 0.3 | IoU threshold for matching detections to tracks |
| `high_conf_threshold` | 0.6 | Confidence threshold for high-quality detections |
| `low_conf_threshold` | 0.1 | Minimum confidence for low-quality detections |

### Tuning Guidelines

- **Increase `max_age`** (40-60) if people frequently move behind obstacles
- **Decrease `min_hits`** (1-2) for faster track confirmation in static scenes
- **Increase `iou_threshold`** (0.4-0.5) in crowded scenes to reduce ID switches
- **Adjust confidence thresholds** based on YOLO model performance

## Usage

### Launch with Default Parameters

```bash
# Launch full pipeline (camera + YOLO + tracker)
ros2 launch robot_bringup camera_yolo_tracker_launch.py

# Launch tracker only (requires YOLO already running)
ros2 launch robot_bringup person_tracker_launch.py
```

### Launch with Custom Parameters

```bash
ros2 launch robot_bringup person_tracker_launch.py \
    max_age:=40 \
    min_hits:=2 \
    iou_threshold:=0.4
```

### Subscribe to Tracks

```python
import rclpy
from rclpy.node import Node
from msgs_interfaces.msg import PersonTrackArray

class TrackSubscriber(Node):
    def __init__(self):
        super().__init__('track_subscriber')
        self.subscription = self.create_subscription(
            PersonTrackArray,
            '/person_tracker/tracks',
            self.track_callback,
            10
        )

    def track_callback(self, msg):
        for track in msg.tracks:
            print(f"Track {track.track_id}: bbox=({track.bbox_x}, {track.bbox_y}, "
                  f"{track.bbox_w}, {track.bbox_h}), conf={track.tracking_confidence:.2f}")
```

## Debugging

The node logs important tracking events:

```bash
# View logs in real-time
ros2 run perception_pkg person_tracker --ros-args --log-level debug

# Common log messages:
# [INFO] Track 5 created
# [INFO] Track 3 deleted (age: 45, missed: 31)
# [DEBUG] Published 2 tracks: [1, 4]
```

### Visualization with RViz2

Coming soon: Custom RViz2 plugin to visualize tracks with IDs

## Dependencies

- **ROS2 Humble**
- **Python 3.10+**
- **scipy**: For Hungarian algorithm (`linear_sum_assignment`)
- **numpy**: For numerical operations
- **cv_bridge**: For ROS-OpenCV conversion
- **vision_msgs**: For YOLO Detection2DArray

Install Python dependencies:
```bash
pip3 install scipy numpy
```

## Performance

- **Tracking Speed**: ~100-500 Hz (depends on number of people)
- **Memory**: ~1-2 MB per 100 active tracks
- **Latency**: <10ms for typical scenarios (<10 people)

ByteTrack adds minimal overhead compared to YOLO detection, making it suitable for real-time social robotics applications.

## Future Enhancements

1. **Visual Features**: Add optional ReID network for appearance-based matching (upgrade to DeepSORT-like)
2. **Kalman Filter**: Predict bounding box positions for smoother tracking
3. **Multi-Camera Fusion**: Merge tracks from multiple camera views
4. **Track History**: Maintain trajectory history for motion analysis
5. **RViz2 Plugin**: Visualize tracks with persistent colors and IDs

## Troubleshooting

### No tracks published
- Verify YOLO is publishing detections: `ros2 topic echo /yolo/detections`
- Check if people are detected by YOLO with sufficient confidence (>0.1)
- Ensure messages are flowing: `ros2 topic hz /yolo/detections`

### Frequent ID switches
- Increase `iou_threshold` (try 0.4-0.5)
- Increase `max_age` to keep tracks alive longer
- Check YOLO detection quality and consistency

### Tracks not confirmed
- Decrease `min_hits` parameter (try 1-2)
- Check detection confidence and adjust `high_conf_threshold`

### High memory usage
- Decrease `max_age` to remove stale tracks faster
- Verify no track ID overflow (restart node if track IDs reach very high numbers)

## References

- [ByteTrack Paper (ECCV 2022)](https://arxiv.org/abs/2110.06864)
- [ByteTrack GitHub](https://github.com/ifzhang/ByteTrack)
- [SORT Paper](https://arxiv.org/abs/1602.00763) - Original simple online tracking
- [DeepSORT Paper](https://arxiv.org/abs/1703.07402) - Deep learning based tracking

## License

Apache-2.0

## Authors

Ball-e Team - Social Robotics Perception System
