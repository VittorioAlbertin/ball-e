# Ball-e API Reference

Complete reference for all topics, services, and messages in the Ball-e tracking and identification system.

## Table of Contents

- [Topics](#topics)
  - [Published Topics](#published-topics)
  - [Subscribed Topics](#subscribed-topics)
- [Services](#services)
- [Messages](#messages)
- [Service Definitions](#service-definitions)
- [Node APIs](#node-apis)

---

## Topics

### Published Topics

#### `/camera/image_raw`
- **Type**: `sensor_msgs/Image`
- **Publisher**: `camera_node`
- **Rate**: ~30 Hz
- **Description**: Raw camera feed in BGR8 encoding

#### `/yolo/detections`
- **Type**: `vision_msgs/Detection2DArray`
- **Publisher**: `yolo_node`
- **Rate**: ~30 Hz
- **Description**: YOLO object detections (all classes)
- **Note**: Person detections (class 0) are filtered downstream

#### `/yolo/image_detections`
- **Type**: `sensor_msgs/Image`
- **Publisher**: `yolo_node`
- **Rate**: ~30 Hz
- **Description**: Annotated image with YOLO bounding boxes

#### `/person_tracker/tracks`
- **Type**: `msgs_interfaces/PersonTrackArray`
- **Publisher**: `person_tracker`
- **Rate**: ~30 Hz (when detections present)
- **Description**: Tracked persons with persistent IDs

**Example**:
```yaml
header:
  stamp: ...
  frame_id: "person_tracker"
tracks:
  - track_id: 1
    bbox_x: 320.5
    bbox_y: 240.0
    bbox_w: 80.0
    bbox_h: 120.0
    tracking_confidence: 0.95
    detection_confidence: 0.87
    frames_since_last_seen: 0
    is_new_track: false
```

#### `/person_state/all`
- **Type**: `msgs_interfaces/PersonStateArray`
- **Publisher**: `person_state_manager`
- **Rate**: 10 Hz
- **Description**: Complete world model of all tracked persons

**Example**:
```yaml
header: ...
persons:
  - track_id: 1
    identity: "Alice"
    identity_confidence: 0.92
    bbox_x: 320.5
    bbox_y: 240.0
    bbox_w: 80.0
    bbox_h: 120.0
    first_seen: {sec: 1234567890, nanosec: 0}
    last_seen: {sec: 1234567895, nanosec: 0}
    requires_identification: false
    tracking_confidence: 0.95
    frames_since_last_seen: 0
total_tracked: 2
identified_count: 1
unidentified_count: 1
pending_identification_count: 0
```

#### `/face_recognition/identity_update`
- **Type**: `msgs_interfaces/IdentityUpdate`
- **Publisher**: `face_recognition_conditional`
- **Rate**: On-demand (0.1-2 Hz)
- **Description**: Face recognition results with performance metrics

**Example**:
```yaml
header: ...
track_id: 1
identity: "Alice"
confidence: 0.92
processing_time_ms: 145.3
face_size_pixels: 85.0
face_quality_ok: true
found_in_database: true
person_id: 42
message: "Face recognized successfully"
```

#### `/visualization/annotated_image`
- **Type**: `sensor_msgs/Image`
- **Publisher**: `visualization_node`
- **Rate**: 10 Hz
- **Description**: Annotated video with track IDs, identities, and status

### Subscribed Topics

Nodes subscribe to the topics listed above as inputs for their processing pipelines.

---

## Services

### `/person_state/get_info`
- **Type**: `msgs_interfaces/srv/GetPersonInfo`
- **Server**: `person_state_manager`
- **Description**: Query information about a specific tracked person

**Request**:
```yaml
track_id: 1
```

**Response**:
```yaml
success: true
message: "Found person with track_id=1"
person_state:
  track_id: 1
  identity: "Alice"
  identity_confidence: 0.92
  # ... full PersonState
```

### `/person_state/request_identification`
- **Type**: `msgs_interfaces/srv/RequestIdentification`
- **Server**: `person_state_manager`
- **Description**: Request face recognition for a tracked person

**Request**:
```yaml
track_id: 1
```

**Response**:
```yaml
success: true
message: "Identification queued for track_id=1"
already_identified: false
```

### `/person_state/update_identity`
- **Type**: `msgs_interfaces/srv/UpdateIdentity`
- **Server**: `person_state_manager`
- **Description**: Update person identity (called by face recognition)

**Request**:
```yaml
track_id: 1
identity: "Alice"
confidence: 0.92
```

**Response**:
```yaml
success: true
message: "Identity updated for track_id=1"
```

### `people_db/recognize_face`
- **Type**: `msgs_interfaces/srv/RecognizeFace`
- **Server**: `people_database_node`
- **Description**: Match face embedding against database

**Request**:
```yaml
face_embedding: [0.123, -0.456, 0.789, ...]  # 512-dim vector
threshold: 0.6
```

**Response**:
```yaml
found: true
person_id: 42
name: "Alice"
last_seen: "2025-01-21 10:30:45"
interaction_count: 15
preferences_json: "{}"
notes: ""
message: "Match found with 92% confidence"
```

### `people_db/add_person`
- **Type**: `msgs_interfaces/srv/AddPerson`
- **Server**: `people_database_node`
- **Description**: Add a new person to the database

**Request**:
```yaml
name: "Bob"
face_embedding: [0.123, -0.456, ...]
notes: "Friend from university"
```

**Response**:
```yaml
success: true
person_id: 43
message: "Person added successfully"
```

---

## Messages

### PersonTrack.msg

```
std_msgs/Header header

int32 track_id
float32 bbox_x
float32 bbox_y
float32 bbox_w
float32 bbox_h
float32 tracking_confidence
int32 frames_since_last_seen
bool is_new_track
float32 detection_confidence
```

### PersonTrackArray.msg

```
std_msgs/Header header
PersonTrack[] tracks
```

### PersonState.msg

```
std_msgs/Header header

int32 track_id
string identity
float32 identity_confidence
float32 bbox_x
float32 bbox_y
float32 bbox_w
float32 bbox_h
builtin_interfaces/Time first_seen
builtin_interfaces/Time last_seen
bool requires_identification
float32 tracking_confidence
int32 frames_since_last_seen
```

### PersonStateArray.msg

```
std_msgs/Header header

PersonState[] persons
int32 total_tracked
int32 identified_count
int32 unidentified_count
int32 pending_identification_count
```

### IdentityUpdate.msg

```
std_msgs/Header header

int32 track_id
string identity
float32 confidence
float32 processing_time_ms
float32 face_size_pixels
bool face_quality_ok
bool found_in_database
int32 person_id
string message
```

---

## Service Definitions

### GetPersonInfo.srv

```
# Request
int32 track_id
---
# Response
bool success
string message
PersonState person_state
```

### RequestIdentification.srv

```
# Request
int32 track_id
---
# Response
bool success
string message
bool already_identified
```

### UpdateIdentity.srv

```
# Request
int32 track_id
string identity
float32 confidence
---
# Response
bool success
string message
```

### RecognizeFace.srv

```
# Request
float32[] face_embedding
float32 threshold
---
# Response
bool found
int32 person_id
string name
string last_seen
int32 interaction_count
string preferences_json
string notes
string message
```

---

## Node APIs

### person_tracker

**Subscribed Topics**:
- `/yolo/detections` (Detection2DArray)
- `/camera/image_raw` (Image) - for future use

**Published Topics**:
- `/person_tracker/tracks` (PersonTrackArray)

**Parameters**:
- `max_age` (int, default: 30)
- `min_hits` (int, default: 3)
- `iou_threshold` (float, default: 0.3)
- `high_conf_threshold` (float, default: 0.6)
- `low_conf_threshold` (float, default: 0.1)

### person_state_manager

**Subscribed Topics**:
- `/person_tracker/tracks` (PersonTrackArray)

**Published Topics**:
- `/person_state/all` (PersonStateArray)

**Services Provided**:
- `/person_state/get_info` (GetPersonInfo)
- `/person_state/request_identification` (RequestIdentification)
- `/person_state/update_identity` (UpdateIdentity)

**Parameters**:
- `cleanup_timeout` (float, default: 5.0)
- `publish_rate` (float, default: 10.0)

### identification_coordinator

**Subscribed Topics**:
- `/person_state/all` (PersonStateArray)
- `/face_recognition/identity_update` (IdentityUpdate)

**Services Called**:
- `/person_state/request_identification` (RequestIdentification)

**Parameters**:
- `max_requests_per_second` (float, default: 2.0)
- `confidence_threshold` (float, default: 0.5)
- `recheck_interval` (float, default: 60.0)
- `new_track_delay` (float, default: 1.0)
- `enable_auto_identification` (bool, default: true)

### face_recognition_conditional

**Subscribed Topics**:
- `/camera/image_raw` (Image)
- `/person_tracker/tracks` (PersonTrackArray)
- `/person_state/all` (PersonStateArray)

**Published Topics**:
- `/face_recognition/identity_update` (IdentityUpdate)

**Services Called**:
- `people_db/recognize_face` (RecognizeFace)
- `/person_state/update_identity` (UpdateIdentity)

**Parameters**:
- `recognition_threshold` (float, default: 0.6)
- `min_face_size` (int, default: 20)
- `max_face_size` (int, default: 400)
- `frame_cache_size` (int, default: 10)
- `reidentification_interval` (float, default: 30.0)
- `auto_identify_new_tracks` (bool, default: true)

### visualization_node

**Subscribed Topics**:
- `/camera/image_raw` (Image)
- `/person_state/all` (PersonStateArray)

**Published Topics**:
- `/visualization/annotated_image` (Image)

**Parameters**:
- `show_track_id` (bool, default: true)
- `show_identity` (bool, default: true)
- `show_confidence` (bool, default: true)
- `show_status` (bool, default: true)
- `box_thickness` (int, default: 2)
- `font_scale` (float, default: 0.6)

---

## Usage Examples

### Python Client Example

```python
import rclpy
from rclpy.node import Node
from msgs_interfaces.msg import PersonStateArray
from msgs_interfaces.srv import RequestIdentification

class PersonTrackerClient(Node):
    def __init__(self):
        super().__init__('person_tracker_client')

        # Subscribe to person states
        self.subscription = self.create_subscription(
            PersonStateArray,
            '/person_state/all',
            self.state_callback,
            10
        )

        # Create service client
        self.id_client = self.create_client(
            RequestIdentification,
            '/person_state/request_identification'
        )

    def state_callback(self, msg):
        for person in msg.persons:
            self.get_logger().info(
                f'Track {person.track_id}: {person.identity or "Unknown"}'
            )

    def request_identification(self, track_id):
        request = RequestIdentification.Request()
        request.track_id = track_id

        future = self.id_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response.success:
            self.get_logger().info(f'Identification requested: {response.message}')

def main():
    rclpy.init()
    client = PersonTrackerClient()
    rclpy.spin(client)
```

### Command Line Examples

```bash
# Monitor person states
ros2 topic echo /person_state/all

# Request identification
ros2 service call /person_state/request_identification \
  msgs_interfaces/srv/RequestIdentification "{track_id: 1}"

# Get person info
ros2 service call /person_state/get_info \
  msgs_interfaces/srv/GetPersonInfo "{track_id: 1}"

# Check message definition
ros2 interface show msgs_interfaces/msg/PersonState

# Check service definition
ros2 interface show msgs_interfaces/srv/GetPersonInfo
```

---

For more information, see the component-specific documentation in the `/docs` directory.
