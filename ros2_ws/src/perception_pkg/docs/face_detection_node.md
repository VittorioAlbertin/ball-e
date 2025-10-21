# Face Detection Node

## Overview
The face detection node detects and recognizes faces within person detections from YOLO. It uses lightweight ONNX models for face detection and embedding extraction, then queries the people database for recognition.

## Node Information
- **Package**: `perception_pkg`
- **Executable**: `face_detection_node`
- **Node Name**: `face_detection_node`

## Topics

### Subscribed Topics
| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Raw camera feed (BGR8 encoding) |
| `/yolo/detections` | `vision_msgs/Detection2DArray` | Person detections from YOLO |

### Published Topics
| Topic | Type | Description |
|-------|------|-------------|
| `/face/detections` | `vision_msgs/Detection2DArray` | Face bounding boxes and confidence |
| `/face/recognition` | `msgs_interfaces/FaceRecognition` | Recognition results with person info |
| `/face/debug_image` | `sensor_msgs/Image` | Annotated image with face bboxes and names |

## Services

### Service Clients
| Service | Type | Description |
|---------|------|-------------|
| `people_db/recognize_face` | `msgs_interfaces/RecognizeFace` | Queries database for face recognition |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `yolo_person_class_id` | int | `0` | YOLO class ID for person |
| `face_model_path` | string | `/ball-e/...models/yoloface.onnx` | Path to face detection ONNX model |
| `face_model_url` | string | GitHub URL | URL to download face detection model |
| `embedding_model_path` | string | `/ball-e/...models/facenet.onnx` | Path to face embedding ONNX model |
| `embedding_model_url` | string | GitHub URL | URL to download embedding model |
| `face_confidence_threshold` | float | `0.6` | Minimum confidence to accept face detection |
| `recognition_threshold` | float | `0.6` | Similarity threshold for face recognition |
| `nms_iou_threshold` | float | `0.3` | IoU threshold for Non-Maximum Suppression |

## Pipeline

### 1. Person Detection
- Subscribes to YOLO detections
- Filters for person class (ID 0)
- Extracts person ROI from image

### 2. Face Detection
- Runs UltraFace model on person ROI
- Detects faces within the person bbox
- Applies NMS to remove duplicates
- Filters by confidence threshold

### 3. Face Embedding
- Extracts face patch from image
- Runs embedding model (ArcFace/FaceNet)
- Generates 512-dimensional embedding vector
- Normalizes vector for comparison

### 4. Face Recognition
- Calls people database service (async)
- Compares embedding with stored faces
- Returns person info if match found
- Updates visualization with results

## Features

### Non-Maximum Suppression (NMS)
Removes duplicate face detections:
- Calculates IoU between overlapping boxes
- Keeps highest confidence detection
- Removes boxes with IoU > threshold (0.3)

### Asynchronous Recognition
- Recognition runs in background (non-blocking)
- Visualization updates when results arrive
- Multiple faces processed in parallel
- Prevents frame rate degradation

### Persistent Results
- Recognition results cached for 5 seconds
- Handles small bbox movements between frames
- Fuzzy matching within 20 pixels
- Smooth transitions between frames

### Color-Coded Visualization
- **Yellow/Cyan**: "Detecting..." (recognition in progress)
- **Green**: Recognized person (shows name)
- **Red**: Unknown person (not in database)

## Output Format

### FaceRecognition Message
```
Header header
float32[] face_embedding      # 512-dimensional vector
float32 bbox_center_x/y       # Face bounding box center
float32 bbox_size_x/y         # Face bounding box size
float32 confidence            # Face detection confidence
bool found                    # True if recognized
int32 person_id               # Database ID (-1 if unknown)
string name                   # Person's name
string last_seen              # Last interaction timestamp
int32 interaction_count       # Number of times seen
string preferences_json       # User preferences
string notes                  # Additional notes
string message                # Status message
```

## Performance

### Frame Rate
- **Detection**: ~10-20 FPS per person ROI
- **Recognition**: Async, doesn't block pipeline
- **Overall**: Near real-time with GPU

### Models
- **Face Detection**: UltraFace (RFB-320)
  - Input: 320x240
  - Size: ~1MB
  - Speed: ~10ms per ROI

- **Face Embedding**: ArcFace/FaceNet
  - Input: 112x112
  - Output: 512-dim vector
  - Speed: ~20ms per face

## Usage Example

### Launch
```bash
ros2 run perception_pkg face_detection_node
```

### View Detections (RViz)
1. Open RViz
2. Add → Image display
3. Set topic to `/face/debug_image`

### Subscribe to Recognition (Python)
```python
import rclpy
from rclpy.node import Node
from msgs_interfaces.msg import FaceRecognition

class FaceSubscriber(Node):
    def __init__(self):
        super().__init__('face_subscriber')
        self.subscription = self.create_subscription(
            FaceRecognition,
            '/face/recognition',
            self.callback,
            10)

    def callback(self, msg):
        if msg.found:
            print(f"Recognized {msg.name} (ID: {msg.person_id})")
            print(f"Confidence: {msg.confidence:.2f}")
            print(f"Seen {msg.interaction_count} times")
        else:
            print(f"Unknown face detected (confidence: {msg.confidence:.2f})")
```

## Troubleshooting

### No Faces Detected
1. Check person detection is working (YOLO)
2. Lower `face_confidence_threshold` parameter
3. Verify face is clearly visible in camera
4. Check lighting conditions

### Faces Not Recognized
1. Check people database has enrolled faces
2. Lower `recognition_threshold` parameter
3. Ensure face is front-facing and well-lit
4. Re-enroll person with better quality images

### Multiple Bounding Boxes on Same Face
1. Check NMS is enabled
2. Lower `nms_iou_threshold` (more aggressive NMS)
3. Increase `face_confidence_threshold` (fewer weak detections)

### Slow Performance
1. Verify models are using ONNX Runtime optimizations
2. Reduce camera resolution
3. Limit number of concurrent person detections
4. Consider GPU acceleration for ONNX Runtime

### "Detecting..." Stays Too Long
1. Check people database service is running
2. Check network/service connectivity
3. Reduce `recognition_threshold` if no matches
4. Check logs for recognition service errors

## Model Downloads

### First Run
Models download automatically if not found:
1. UltraFace: ~1MB from GitHub
2. ArcFace/FaceNet: ~100MB from GitHub
3. Cached in `/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/`

### Manual Download
```bash
# Face detection model
wget https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx \
  -O /ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/yoloface.onnx

# Embedding model
wget https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx \
  -O /ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/facenet.onnx
```

## Dependencies
- `onnxruntime`
- `cv2` (OpenCV)
- `numpy`
- `ros2_numpy`
- `msgs_interfaces` (custom messages)
- `people_database_node` (for recognition)

## Notes
- Requires YOLO node to be running
- Requires people database node for recognition
- Face detection runs on person ROIs only (not full image)
- BGR color format (OpenCV standard)
- Async recognition prevents blocking
