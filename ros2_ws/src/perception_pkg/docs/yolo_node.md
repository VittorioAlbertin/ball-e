# YOLO Detection Node

## Overview
The YOLO node performs real-time object detection using YOLOv5. It subscribes to camera images and publishes both structured detection data and annotated visualization images.

## Node Information
- **Package**: `perception_pkg`
- **Executable**: `yolo_node`
- **Node Name**: `yolo_node`

## Topics

### Subscribed Topics
| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Raw camera feed (BGR8 encoding) |

### Published Topics
| Topic | Type | Description |
|-------|------|-------------|
| `/yolo/detections` | `vision_msgs/Detection2DArray` | Structured detection results (bounding boxes, classes, confidence scores) |
| `/yolo/image_detections` | `sensor_msgs/Image` | Annotated image with bounding boxes and labels drawn |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| N/A | - | - | Currently uses hardcoded model path |

## Configuration

### Model Path
- **Location**: `/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/yolov5n.pt`
- **Model**: YOLOv5 Nano (lightweight, fast inference)
- **Download**: Automatically downloads if not found

### Device Selection
Automatically selects CUDA GPU if available, otherwise falls back to CPU.

## Features

### Object Detection
- **Classes**: 80 COCO classes (person, car, dog, etc.)
- **Input Size**: 640x640 pixels
- **Threading**: Async processing to avoid blocking the main loop
- **Frame Buffering**: Only processes the latest frame (drops old frames)

### Visualization
- **Bounding Boxes**: Colored rectangles around detected objects
- **Labels**: Class name and confidence score
- **Color Coding**: Different colors for different object classes
- **Background**: Filled rectangles behind labels for readability

## Output Format

### Detection2DArray Message
Each detection contains:
- `bbox.center.position.x/y`: Center of bounding box (pixels)
- `bbox.size_x/y`: Width and height of bounding box (pixels)
- `results[0].hypothesis.class_id`: COCO class ID (as string)
- `results[0].hypothesis.score`: Confidence score (0.0-1.0)

### Class IDs (Common)
- `0`: Person
- `1`: Bicycle
- `2`: Car
- `15`: Cat
- `16`: Dog
- See [COCO dataset](https://cocodataset.org/#explore) for full list

## Performance

### Frame Rate
- **GPU**: ~30-50 FPS (depending on GPU)
- **CPU**: ~5-15 FPS

### Optimization
- Latest frame only processing
- Async threading for non-blocking inference
- YOLOv5n (nano) for speed/accuracy balance

## Usage Example

### Launch
```bash
ros2 run perception_pkg yolo_node
```

### View Detections (RViz)
1. Open RViz
2. Add → Image display
3. Set topic to `/yolo/image_detections`

### Subscribe to Detections (Python)
```python
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray

class DetectionSubscriber(Node):
    def __init__(self):
        super().__init__('detection_subscriber')
        self.subscription = self.create_subscription(
            Detection2DArray,
            '/yolo/detections',
            self.callback,
            10)

    def callback(self, msg):
        for detection in msg.detections:
            class_id = detection.results[0].hypothesis.class_id
            score = detection.results[0].hypothesis.score
            x = detection.bbox.center.position.x
            y = detection.bbox.center.position.y
            print(f"Detected class {class_id} at ({x}, {y}) with {score:.2f} confidence")
```

## Troubleshooting

### No Detections
- Check camera feed is publishing on `/camera/image_raw`
- Verify objects are in COCO classes
- Lower confidence threshold in model if needed

### Low Frame Rate
- Switch to GPU if available
- Reduce image resolution from camera
- Use smaller YOLO model (already using nano)

### Deprecation Warning (torch.cuda.amp.autocast)
- Warning is suppressed automatically
- Comes from YOLOv5 library, not our code
- Does not affect functionality

## Dependencies
- `torch`
- `cv2` (OpenCV)
- `ros2_numpy`
- `ultralytics/yolov5` (loaded via torch.hub)

## Notes
- Model is cached in `~/.cache/torch/hub/`
- First run downloads model (~4MB for YOLOv5n)
- Uses BGR color format (OpenCV standard)
