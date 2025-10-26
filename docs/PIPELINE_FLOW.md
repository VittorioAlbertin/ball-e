# Ball-e Vision Pipeline: Complete Flow Documentation

This document provides a step-by-step walkthrough of the complete image processing pipeline, from raw camera sensor input to the final annotated image with person detection, tracking, and identification.

## Architecture Overview

```
[Camera Sensor] (30 Hz)
    ↓ /camera/image_raw
[YOLO Node] (async, ~30 FPS)
    ↓ /yolo/detections (all 80 COCO classes)
[Person Tracker] (ByteTrack, filters person class)
    ↓ /person_tracker/tracks (persistent track_ids)
[Person State Manager] (world model, 10 Hz publish)
    ↓ /person_state/all
    ├→ [Identification Coordinator] (smart triggering)
    │   ↓ service: /person_state/request_identification
    │   ↓
    ├→ [Face Recognition Conditional] (on-demand, <200ms)
    │   ↓ service: people_db/recognize_face
    │   ↓
    │  [People Database Node] (SQLite + cosine similarity)
    │   ↓ service response
    │   ↓
    │   ↓ /face_recognition/identity_update
    │   ↓
    │   ↓ service: /person_state/update_identity
    │   └→ [Person State Manager] (updates identity)
    │
    └→ [Visualization Node]
        ↓ /visualization/annotated_image (final output)
```

---

## Step 1: Image Capture

**Node:** `camera_node.py`
**Package:** `sensors_pkg`
**Location:** `ros2_ws/src/sensors_pkg/sensors_pkg/camera_node.py:8-42`

### Operation
- Captures frames from camera at **~30 Hz** using OpenCV (`cv2.VideoCapture(0)`)
- Timer callback runs at `1.0/30.0` seconds (line 22)
- Converts BGR image to ROS `sensor_msgs/Image` using `cv_bridge` (line 34)

### Output
**Topic:** `/camera/image_raw`
**Type:** `sensor_msgs/Image`
**Encoding:** `bgr8`
**Frequency:** ~30 Hz

---

## Step 2: Object Detection

**Node:** `yolo_node.py`
**Package:** `perception_pkg`
**Location:** `ros2_ws/src/perception_pkg/perception_pkg/yolo_node.py:19-169`

### Subscription
**Topic:** `/camera/image_raw`
**Type:** `sensor_msgs/Image`
**Location:** Lines 26-28

### Processing

1. **Async Threading** (lines 52-56)
   - Uses background thread to prevent blocking ROS callbacks
   - Main thread stores latest frame in buffer with thread lock (lines 57-59)

2. **Image Conversion** (lines 70-75)
   - Converts BGR → RGB for YOLO inference
   - Handles grayscale to RGB conversion if needed

3. **YOLO Inference** (line 76)
   - Model: **YOLOv5n** (nano version for speed)
   - Input size: **640 pixels**
   - Device: CUDA if available, else CPU
   - Detects **all 80 COCO classes** (person, car, dog, cat, etc.)

4. **Result Extraction** (lines 89-108)
   - Extracts bounding boxes from pandas dataframe
   - Format: `xmin, ymin, xmax, ymax, confidence, class, name`
   - Converts to Detection2D format:
     - `bbox.center.position.x/y` (center point)
     - `bbox.size_x/y` (width/height)
     - `results[].hypothesis.class_id` (class as string)
     - `results[].hypothesis.score` (confidence)

### Output
**Topic 1:** `/yolo/detections`
**Type:** `vision_msgs/Detection2DArray`
**Contains:** All detected objects (80 COCO classes)

**Topic 2:** `/yolo/image_detections`
**Type:** `sensor_msgs/Image`
**Contains:** Annotated image with all COCO class visualizations

---

## Step 3: Person Tracking

**Node:** `person_tracker.py`
**Package:** `perception_pkg`
**Location:** `ros2_ws/src/perception_pkg/perception_pkg/person_tracker.py:102-337`

### Subscription
**Topic:** `/yolo/detections`
**Type:** `vision_msgs/Detection2DArray`
**Location:** Lines 134-139

### Processing

#### 3.1 Person Filtering (lines 154-177)
- Filters detections for **person class only** (`class_id == 'person'` or `'0'`)
- Converts Detection2D format (center + size) to `[x, y, w, h]` format:
  ```python
  x = center_x - size_x / 2.0  # Top-left corner
  y = center_y - size_y / 2.0
  w = size_x
  h = size_y
  ```

#### 3.2 ByteTrack Algorithm (lines 185-239)

**Two-Stage Association:**

1. **Stage 1: High-Confidence Matching** (lines 193-203)
   - Separates high-confidence detections (conf ≥ 0.6, configurable)
   - Matches with existing tracks using IoU metric
   - Hungarian algorithm (linear_sum_assignment) for optimal assignment (line 261)

2. **Stage 2: Low-Confidence Matching** (lines 207-212)
   - Low-confidence detections (0.1 ≤ conf < 0.6)
   - Matched only with unmatched tracks from Stage 1
   - Recovers temporarily occluded tracks

3. **New Track Creation** (lines 215-219)
   - Unmatched high-confidence detections become new tracks
   - Assigns unique, persistent `track_id` (auto-incrementing)
   - Logs track creation

#### 3.3 Track Management

**Track Confirmation** (lines 236-238)
- New tracks require `min_hits` consecutive detections (default: 3)
- Only confirmed tracks are published
- Prevents false positives from spurious detections

**Track Deletion** (lines 227-233)
- Tracks deleted after `max_age` frames without detection (default: 30)
- ~1 second at 30 FPS
- Logs deletion with age and miss count

**Confidence Decay** (lines 61, 66)
- Tracking confidence decays when not updated: `conf *= 0.95`
- Missed detections: `conf *= 0.90`

#### 3.4 IoU Computation (lines 69-100)
```
IoU = intersection_area / union_area
```
- Used for data association
- Threshold: 0.3 (configurable)

### Parameters
- `max_age`: 30 frames (track lifetime without detection)
- `min_hits`: 3 frames (confirmation threshold)
- `iou_threshold`: 0.3 (matching threshold)
- `high_conf_threshold`: 0.6 (Stage 1 cutoff)
- `low_conf_threshold`: 0.1 (Stage 2 cutoff)

### Output
**Topic:** `/person_tracker/tracks`
**Type:** `msgs_interfaces/PersonTrackArray`
**Contains:**
- `track_id` (persistent, unique integer)
- `bbox_x, bbox_y, bbox_w, bbox_h` (bounding box)
- `tracking_confidence` (smoothed confidence)
- `detection_confidence` (latest YOLO confidence)
- `frames_since_last_seen` (miss count)
- `is_new_track` (true if just confirmed)

---

## Step 4: State Management

**Node:** `person_state_manager.py`
**Package:** `perception_pkg`
**Location:** `ros2_ws/src/perception_pkg/perception_pkg/person_state_manager.py:23-314`

### Subscription
**Topic:** `/person_tracker/tracks`
**Type:** `msgs_interfaces/PersonTrackArray`
**Location:** Lines 53-58

### Processing

#### 4.1 World Model (lines 99-135)
Maintains centralized dictionary: `{track_id: PersonState}`

**For New Tracks** (lines 108-124):
```python
{
    'track_id': track_id,
    'identity': '',                    # Unknown initially
    'identity_confidence': 0.0,
    'bbox_x/y/w/h': ...,               # From tracker
    'first_seen': timestamp,
    'last_seen': timestamp,
    'requires_identification': True,   # Flag for face recognition
    'tracking_confidence': ...,
    'frames_since_last_seen': ...
}
```

**For Existing Tracks** (lines 127-134):
- Updates bbox position
- Updates `last_seen` timestamp
- Updates tracking metrics
- Identity persists until updated by face recognition

#### 4.2 State Publishing (lines 223-250)
- Publishes all person states at **10 Hz**
- Includes statistics:
  - `total_tracked`: Total persons being tracked
  - `identified_count`: Persons with known identity
  - `unidentified_count`: Unknown persons
  - `pending_identification_count`: Queued for face recognition

#### 4.3 Cleanup (lines 252-275)
- Removes stale persons after `cleanup_timeout` (default: 5 seconds)
- Cleans up identification queue
- Logs track loss with identity info

### Services Provided

#### `/person_state/get_info` (lines 136-150)
**Type:** `msgs_interfaces/srv/GetPersonInfo`
- Query person state by `track_id`
- Returns complete `PersonState` message

#### `/person_state/request_identification` (lines 152-181)
**Type:** `msgs_interfaces/srv/RequestIdentification`
- Queue person for face recognition
- Checks if already identified
- Adds to identification queue
- Sets `requires_identification = True`

#### `/person_state/update_identity` (lines 183-221)
**Type:** `msgs_interfaces/srv/UpdateIdentity`
- Called by face recognition node
- Updates person's identity and confidence
- Removes from identification queue
- Logs identity changes

### Output
**Topic:** `/person_state/all`
**Type:** `msgs_interfaces/PersonStateArray`
**Frequency:** 10 Hz
**Contains:** Complete state for all tracked persons

---

## Step 5: Face Recognition (On-Demand)

**Node:** `face_recognition_conditional.py`
**Package:** `perception_pkg`
**Location:** `ros2_ws/src/perception_pkg/perception_pkg/face_recognition_conditional.py:30-429`

### Subscriptions
**Topic 1:** `/camera/image_raw` (lines 102-107) - Frame caching
**Topic 2:** `/person_tracker/tracks` (lines 109-114) - Trigger detection
**Topic 3:** `/person_state/all` (lines 116-121) - State monitoring

### Processing Flow

#### 5.1 Frame Caching (lines 162-169)
```python
self.frame_cache = deque(maxlen=10)  # Last 10 frames
frame_cache.append((timestamp, image, header))
```
- Maintains recent frames for async processing
- Prevents dropped frames during processing
- Stores timestamp for synchronization

#### 5.2 Automatic Identification Triggering

This node handles its own smart triggering logic:

**Trigger Sources:**
1. **New tracks**: Automatically identifies new tracks when `auto_identify_new_tracks=True` (lines 284-289)
2. **Explicit requests**: Monitors `person.requires_identification` flag (lines 311-313)
3. **Periodic re-identification**: Re-identifies known persons after `reidentification_interval` seconds (lines 315-323)

**Parameters:**
- `auto_identify_new_tracks`: true (automatically identify new tracks)
- `reidentification_interval`: 30.0s (re-identify known persons periodically)

#### 5.3 Face Detection in Person ROI (lines 232-311)

**Proper Face Detection Using UltraFace:**
```python
# 1. Extract person ROI from full-resolution image
person_roi = image[person_y:person_y+person_h, person_x:person_x+person_w]

# 2. Resize ROI to detector input (320x240)
resized_roi = cv2.resize(person_roi, (320, 240))

# 3. Run UltraFace detector
outputs = face_detector_session.run(...)

# 4. Get best face detection with confidence
best_face_bbox = boxes[best_idx]  # Normalized coordinates
best_confidence = scores[best_idx]

# 5. Convert to image coordinates
face_x, face_y, face_w, face_h = convert_to_image_coords(best_face_bbox, person_bbox)
```

**Key Features**:
- Uses lightweight UltraFace ONNX model (~1.2 MB)
- Detects faces within person ROI only (fast)
- Returns precise face coordinates (not estimated)
- Confidence threshold: 0.7 (configurable)

**Quality Checks** (lines 277-304):
- Face detection confidence >= 0.7
- Face size >= 40 pixels (configurable)
- Validates bbox is within image bounds
- Logs detection details for debugging

#### 5.4 Face Embedding Extraction (lines 342-370, 438-460)

**Uses Full-Resolution Face Crop:**
- Face crop extracted from **original camera resolution** (not downscaled)
- Maximizes facial detail for better embedding quality
- Camera resolution auto-detected (supports any resolution)

**Model:** ONNX-optimized FaceNet/ArcFace (lines 85-95)
- Path: `/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/facenet.onnx`
- Auto-downloads if not present (lines 156-178)

**Process:**
1. Extract face crop from full-resolution image using detected bbox
2. Resize face to model input size (typically 112x112)
3. Normalize: `input_data = face_rgb / 255.0`
4. Transpose: (H, W, C) → (C, H, W)
5. Add batch dimension: (1, C, H, W)
6. Run ONNX inference
7. Extract embedding (512-dim vector)
8. L2 normalize: `embedding / ||embedding||`

**Performance:** ~50-100ms per face (embedding extraction only)

#### 5.5 Database Matching (lines 373-436)

**Service Call:** `people_db/recognize_face`
**Type:** `msgs_interfaces/srv/RecognizeFace`
**Request:**
- `face_embedding`: float32[] (512 dimensions)
- `threshold`: 0.75 (cosine similarity threshold)

**Response:**
- `found`: true/false
- `person_id`, `name`: Matched person info
- `similarity`: **Actual cosine similarity score** (0.0-1.0)
- `message`: Status message

**Async Processing:**
- Non-blocking service call with callback (lines 380-431)
- Allows continued processing during database query
- Total latency: **<300ms** (including face detection + embedding + matching)

#### 5.6 State Update (lines 397, 410)

**Service Call:** `/person_state/update_identity`
**Type:** `msgs_interfaces/srv/UpdateIdentity`
**Request:**
- `track_id`: Person to update
- `identity`: Name from database (or '' if unknown)
- `confidence`: **Actual similarity score from database** (not threshold!)

**Important Changes:**
- Confidence now uses **real similarity score** (e.g., 0.9823)
- No longer hardcoded to threshold value (0.75)
- Provides accurate match quality to visualization

**Cleanup:**
- Removes from identification queue after processing
- Updates `last_identification_time` (line 425)
- Logs detailed results with all timing breakdowns (lines 413-422)

### Output
**Topic:** `/face_recognition/identity_update`
**Type:** `msgs_interfaces/IdentityUpdate`
**Contains:**
- `track_id`: Person identified
- `identity`: Name (or '' if unknown)
- `confidence`: Recognition confidence
- `face_embedding`: Original embedding (for enrollment)
- `processing_time_ms`: Total latency
- `face_size_pixels`: Face crop size
- `face_quality_ok`: Quality check result
- `found_in_database`: Match found?
- `person_id`: Database ID (if found)
- `message`: Status message

### Parameters
- `face_detection_threshold`: 0.7 (face detector confidence threshold)
- `recognition_threshold`: 0.75 (similarity threshold for matching)
- `min_face_size`: 40px (minimum detected face size)
- `frame_cache_size`: 10 frames (full-resolution image cache)
- `reidentification_interval`: 30.0s
- `auto_identify_new_tracks`: true

### Models Used
**Face Detector:** UltraFace RFB-320
- Size: ~1.2 MB (ONNX)
- Input: 320x240 RGB
- URL: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
- Performance: ~10-20ms per detection

**Face Embedder:** ArcFace ResNet100
- Size: ~100 MB (ONNX)
- Input: 112x112 RGB (configurable)
- Output: 512-dim normalized embedding
- URL: https://github.com/onnx/models (ArcFace)
- Performance: ~50-100ms per embedding

---

## Step 6: Database Matching

**Node:** `people_database_node.py`
**Package:** `interaction_pkg`
**Location:** `ros2_ws/src/interaction_pkg/interaction_pkg/people_database_node.py:23-240`

### Service Provided

**Service:** `people_db/recognize_face` (lines 44-45, 94-125)
**Type:** `msgs_interfaces/srv/RecognizeFace`

### Processing

#### 6.1 Database Structure
- **Database:** SQLite at `/ball-e/ros2_ws/robot_data/people.db`
- **Backend:** `interaction_pkg/people_database.py`
- **Schema:**
  ```sql
  people (
      id INTEGER PRIMARY KEY,
      name TEXT UNIQUE,
      face_embedding BLOB,      -- Stored as numpy array
      created_at TIMESTAMP,
      last_seen TIMESTAMP,
      interaction_count INTEGER,
      preferences TEXT,          -- JSON
      notes TEXT
  )
  ```

#### 6.2 Face Matching (line 100)
```python
match = db.find_similar_face(embedding, threshold)
```

**Algorithm:** Cosine Similarity
```
similarity = dot(query_embedding, stored_embedding)
             / (||query|| * ||stored||)
```
- Since embeddings are L2-normalized: `similarity = dot(query, stored)`
- Range: [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite
- Threshold: 0.6 (configurable)

**Process:**
1. Query all embeddings from database
2. Compute similarity with each stored embedding
3. Find maximum similarity
4. Return match if `max_similarity >= threshold`

#### 6.3 Response Handling (lines 102-118)

**If Match Found:**
- Returns person info: `id, name, last_seen, interaction_count, preferences, notes`
- Auto-updates `last_seen` timestamp (line 113)
- Logs recognition: "Recognized: {name} (ID: {id})"

**If No Match:**
- Returns `found = False`
- Message: "No matching person found"
- Identity remains empty in person state

### Other Services Provided

#### `people_db/add_person` (lines 41-42, 64-92)
- Add new person with face embedding
- Used by enrollment CLI

#### `people_db/get_person` (lines 47-48, 127-156)
- Query by ID or name

#### `people_db/update_last_seen` (lines 50-51, 158-170)
- Manual last_seen update

#### `people_db/update_preferences` (lines 53-54, 172-185)
- Update user preferences (JSON)

#### `people_db/get_all_people` (lines 56-57, 187-202)
- List all enrolled persons

#### `people_db/delete_person` (lines 59-60, 204-217)
- Remove person from database

---

## Step 7: Visualization & Annotation

**Node:** `visualization_node.py`
**Package:** `perception_pkg`
**Location:** `ros2_ws/src/perception_pkg/perception_pkg/visualization_node.py:22-310`

### Subscriptions
**Topic 1:** `/camera/image_raw` (lines 58-63) - Raw camera feed
**Topic 2:** `/person_state/all` (lines 65-70) - Person states

### Processing

#### 7.1 Image Composition (lines 89-110)
```python
annotated = current_image.copy()
_draw_statistics(annotated, msg)      # Top-left stats
for person in msg.persons:
    _draw_person(annotated, person)   # Per-person annotations
publish(annotated)
```

#### 7.2 Per-Person Annotations (lines 112-208)

**Bounding Box** (line 133):
- Color: HSV hue rotation based on `track_id` (lines 246-270)
  ```python
  hue = (track_id * 137.5) % 360  # Golden angle for good distribution
  ```
- Thickness: 2 pixels (configurable)
- Color modulation:
  - **Brighter** for identified persons (saturation=0.9, value=0.9)
  - **Dimmer** for unknown (saturation=0.7, value=0.7)
  - **Yellow tint** for identifying (multiply by 1.2)

**Labels Above Box** (lines 136-183):
Multi-line labels with colored backgrounds:

1. **Track ID** (if enabled):
   - Format: `ID:42`

2. **Identity** (if known):
   - Format: `John (85%)` (with confidence)
   - Or: `Unknown` (if identity == '')
   - Or: `Identifying...` (if requires_identification)

3. **Status Flags** (if enabled, lines 272-292):
   - `[NEW]`: Seen for <3 seconds
   - `[LOW-CONF]`: tracking_confidence < 0.5
   - `[MISS:5]`: frames_since_last_seen > 5

**Label Rendering:**
- Background: Filled rectangle in track color
- Text: White, OpenCV FONT_HERSHEY_SIMPLEX
- Auto-adjusts position if outside image bounds

**Confidence Bar** (lines 186-208):
- Location: Below bounding box
- Horizontal bar showing `tracking_confidence`
- Gray background, colored fill matches track color
- Height: 5 pixels

#### 7.3 Statistics Overlay (lines 210-244)

**Location:** Top-left corner

**Contents:**
```
Tracked: 3
Identified: 2
Unknown: 1
Pending ID: 0
```

**Rendering:**
- Black background boxes
- Green text (0, 255, 0)
- Font scale: 0.7
- Updates in real-time

### Parameters
- `show_track_id`: true (display track IDs)
- `show_identity`: true (display names)
- `show_confidence`: true (display confidence %)
- `show_status`: true (display status flags)
- `box_thickness`: 2 (bbox line width)
- `font_scale`: 0.6 (text size)

### Output
**Topic:** `/visualization/annotated_image`
**Type:** `sensor_msgs/Image`
**Encoding:** `bgr8`
**Contains:** Final annotated RGB image with:
- Color-coded bounding boxes
- Track IDs
- Identity names with confidence
- Status indicators
- Tracking confidence bars
- System statistics

---

## Complete Topic Flow

### Published Topics
| Topic | Type | Publisher | Frequency | Description |
|-------|------|-----------|-----------|-------------|
| `/camera/image_raw` | sensor_msgs/Image | camera_node | 30 Hz | Raw camera feed |
| `/yolo/detections` | vision_msgs/Detection2DArray | yolo_node | ~30 Hz | All COCO class detections |
| `/yolo/image_detections` | sensor_msgs/Image | yolo_node | ~30 Hz | YOLO visualization |
| `/person_tracker/tracks` | PersonTrackArray | person_tracker | ~30 Hz | Tracked persons with IDs |
| `/person_state/all` | PersonStateArray | person_state_manager | 10 Hz | Complete person states |
| `/face_recognition/identity_update` | IdentityUpdate | face_recognition_conditional | On-demand | Recognition results |
| `/visualization/annotated_image` | sensor_msgs/Image | visualization_node | 10 Hz | Final annotated output |

### Service Interfaces
| Service | Type | Server | Description |
|---------|------|--------|-------------|
| `/person_state/get_info` | GetPersonInfo | person_state_manager | Query person by track_id |
| `/person_state/request_identification` | RequestIdentification | person_state_manager | Queue face recognition |
| `/person_state/update_identity` | UpdateIdentity | person_state_manager | Update person identity |
| `people_db/recognize_face` | RecognizeFace | people_database_node | Face matching service |
| `people_db/add_person` | AddPerson | people_database_node | Enroll new person |
| `people_db/get_person` | GetPerson | people_database_node | Query database by ID/name |

---

## Performance Characteristics

### System Throughput
- **Overall:** ~30 FPS (30x improvement over original)
- **Bottleneck:** YOLO inference (~30 FPS)
- **Face Recognition:** On-demand only (0.1-2 FPS depending on activity)

### Latency Breakdown
| Component | Latency | Notes |
|-----------|---------|-------|
| Camera capture | 33ms | 30 Hz timer |
| YOLO inference | 20-40ms | YOLOv5n, async |
| Person tracking | <5ms | ByteTrack algorithm |
| State management | <1ms | Dictionary lookup |
| **Face detection** | **10-20ms** | **UltraFace on person ROI** |
| **Face embedding** | **50-100ms** | **ArcFace on high-res crop** |
| **Database matching** | **10-50ms** | **Cosine similarity, linear search** |
| Visualization | <10ms | OpenCV drawing |
| **Total (no face rec)** | **~50-80ms** | **12-20 FPS capable** |
| **Total (with face rec)** | **~250-320ms** | **Only when triggered** |

### Resource Usage
- **CPU:** ~40% (vs 100% in original always-on system)
- **Memory:** ~500MB (includes YOLO + FaceNet models)
- **GPU:** Optional (CUDA-accelerated YOLO if available)

### Scalability
- **Max simultaneous tracks:** ~10 persons (ByteTrack design)
- **Face recognition rate:** 2 requests/second (configurable)
- **Database size:** Tested with 100+ enrolled persons

---

## Key Design Decisions

### 1. On-Demand Face Recognition
**Problem:** Always-on face recognition caused ~1 FPS bottleneck
**Solution:** Conditional triggering based on person state
**Result:** 30x throughput improvement, 60% CPU reduction

### 2. Proper Face Detection (Not Estimation)
**Problem:** Crude geometric estimation (top 30% of person box) caused:
- Poor face crops (mostly background/clothing)
- Extremely high false similarities (0.99+ for different people)
- Identity flipping between people
- Hardcoded confidence values

**Solution:**
- UltraFace detector for precise face localization in person ROI
- Full-resolution image usage for maximum facial detail
- Actual similarity scores passed as confidence (not threshold)

**Result:**
- Accurate face detection with confidence scores
- Better embeddings from high-quality face crops
- True confidence values in visualization (not stuck at 0.75)
- Significantly reduced false matches

### 3. Centralized State Management
**Problem:** Distributed state across nodes caused inconsistencies
**Solution:** Single source of truth (person_state_manager)
**Result:** Clean separation of concerns, easier debugging

### 4. ByteTrack Two-Stage Association
**Problem:** Simple IoU matching lost tracks during occlusion
**Solution:** High-confidence + low-confidence matching stages
**Result:** More robust tracking, handles temporary occlusions

### 5. Smart Identification Triggering
**Problem:** When to trigger face recognition?
**Solution:** Multi-rule coordinator (new tracks, confidence decay, periodic re-check)
**Result:** Balanced coverage vs computational efficiency

### 6. Async Processing Throughout
**Problem:** Synchronous processing blocks ROS callbacks
**Solution:** Threading (YOLO), callbacks (services), caching (frames)
**Result:** System remains responsive under load

---

## Debugging & Monitoring

### View Active Topics
```bash
ros2 topic list
ros2 topic hz /person_tracker/tracks     # Check tracking rate
ros2 topic hz /visualization/annotated_image
```

### Echo Person States
```bash
ros2 topic echo /person_state/all
ros2 topic echo /face_recognition/identity_update
```

### Call Services Manually
```bash
# Request identification for track 42
ros2 service call /person_state/request_identification \
  msgs_interfaces/srv/RequestIdentification "{track_id: 42}"

# Query person info
ros2 service call /person_state/get_info \
  msgs_interfaces/srv/GetPersonInfo "{track_id: 42}"
```

### Check Node Status
```bash
ros2 node list
ros2 node info /person_state_manager
ros2 node info /face_recognition_conditional
```

### Visualize with RViz2
```bash
rviz2 -d src/robot_bringup/rviz/ball_e_tracking.rviz
```

---

## Common Issues & Solutions

### Issue: No faces recognized
**Possible causes:**
1. No persons enrolled in database
2. Face not detected in person ROI (check logs for "No face detected")
3. Face detection confidence too low (< 0.7)
4. Recognition threshold too high
5. Person not facing camera

**Solutions:**
```bash
# Check face detection logs
ros2 topic echo /face_recognition/identity_update

# Lower face detection threshold
face_detection_threshold:=0.5

# Lower recognition threshold (in launch file)
recognition_threshold:=0.65

# Enroll people with better quality faces
ros2 run interaction_pkg enroll_face_cli
```

### Issue: High false match rate (different people recognized as same)
**Possible causes:**
1. Recognition threshold too low
2. Poor lighting conditions
3. Similar appearance (siblings, etc.)

**Solutions:**
```bash
# Increase recognition threshold
recognition_threshold:=0.80

# Re-enroll with multiple angles and lighting conditions
# Check similarity scores in logs - should be >0.85 for good matches
```

### Issue: Tracking IDs jump frequently
**Possible causes:**
1. IoU threshold too high
2. Max age too low
3. Poor YOLO detection quality

**Solutions:**
```bash
# Adjust tracking parameters
iou_threshold:=0.2
max_age:=40
min_hits:=2
```

### Issue: Face recognition too slow
**Possible causes:**
1. No GPU acceleration
2. Large database
3. Rate limit too high

**Solutions:**
```bash
# Check GPU availability
nvidia-smi

# Reduce rate limit
max_requests_per_second:=1.0
```

---

## Future Enhancements

### Short-term
- [ ] GPU acceleration for YOLO (TensorRT)
- [ ] Face detection instead of estimation (MTCNN, RetinaFace)
- [ ] Kalman filter for track smoothing
- [ ] Depth integration for 3D tracking

### Long-term
- [ ] Multi-camera fusion
- [ ] Re-identification across cameras
- [ ] Trajectory prediction
- [ ] Emotion recognition per person
- [ ] Voice-face association

---

## References

- **ByteTrack Paper:** https://arxiv.org/abs/2110.06864
- **YOLOv5:** https://github.com/ultralytics/yolov5
- **FaceNet/ArcFace:** ONNX Model Zoo
- **ROS2 Humble:** https://docs.ros.org/en/humble/

---

**Document Version:** 1.0
**Last Updated:** 2025-10-23
**Maintainer:** Ball-e Team
