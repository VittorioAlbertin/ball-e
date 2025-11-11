# Architectural Refactor: Service-Based Identification Pipeline

## Current Problems
- Too many async message passing steps (IdentificationRequest → FaceRecognition → IdentityUpdate)
- Hard to track request/response flow
- Multiple nodes maintaining pending state
- Face crop saving used incorrectly for enrollment

## New Architecture Overview
**PersonStateManager** becomes the central orchestrator that:
- Subscribes to PersonTrackArray (only subscription)
- Calls services synchronously for face detection, embedding, and recognition
- Provides enrollment service
- No more IdentificationRequest/FaceRecognition/IdentityUpdate messages

---

## Step-by-Step Changes

### 1. Create New Service Definitions

**File**: `msgs_interfaces/srv/DetectFace.srv` (NEW)
```
# Request face detection in person ROI
std_msgs/Header header
int32 track_id
int32 bbox_x  # person bbox in low-res (640x360)
int32 bbox_y
int32 bbox_w
int32 bbox_h
---
# Response
bool success
int32 face_x  # face bbox in low-res (absolute coords)
int32 face_y
int32 face_w
int32 face_h
string message
```

**File**: `msgs_interfaces/srv/GenerateEmbedding.srv` (NEW)
```
# Request face embedding generation
std_msgs/Header header
int32 track_id
int32 face_x  # face bbox in low-res (640x360)
int32 face_y
int32 face_w
int32 face_h
---
# Response
bool success
float32[] embedding  # 512-dim vector
string message
```

**File**: `msgs_interfaces/srv/EnrollPerson.srv` (NEW)
```
# Enroll person from track_id
int32 track_id
string person_name
string notes
---
# Response
bool success
int32 person_id
string message
```

**File**: `msgs_interfaces/CMakeLists.txt`
- Add new service files to rosidl_generate_interfaces()

---

### 2. Refactor FaceDetectorNode → Service-Based

**File**: `perception_pkg/face_detector_node.py`

**Remove**:
- Subscription to `/person_state/identification_request`
- Publication to `/face_detector/detections`
- `pending_requests` dict
- `request_callback()` method
- `result_pub` publisher

**Keep**:
- Subscription to `/camera/image_low_res` (for frame caching)
- `frame_cache` dict
- YOLO model loading and inference
- `cleanup_cache()` timer

**Add**:
- Service server: `face_detection/detect_face` (DetectFace)
- Service callback:
  1. Get timestamp from header
  2. Lookup frame in cache by timestamp
  3. Extract ROI using person bbox
  4. Run YOLO face detection on ROI
  5. Transform face bbox from ROI coords to absolute low-res coords
  6. Return success + face_bbox or failure

**Signature**:
```python
def detect_face_callback(self, request, response):
    # Get frame from cache
    # Extract ROI
    # Detect face
    # Transform coords
    # Return result
```

---

### 3. Refactor FaceRecognizerNode → Service-Based

**File**: `perception_pkg/face_recognizer_node.py`

**Remove**:
- Subscription to `/face_detector/detections`
- Publication to `/face_recognizer/results`
- `detection_callback()` method
- `result_pub` publisher
- Face crop saving functionality (save_face_crop method)
- `face_crops_dir` parameter

**Keep**:
- Subscription to `/camera/image_raw` (high-res frame caching)
- `frame_cache` dict
- FaceNet model loading
- `scale_bbox()`, `extract_face()`, `generate_embedding()`, `preprocess_face()` methods
- `cleanup_cache()` timer
- GPU/CPU provider setup

**Remove service client**:
- No longer calls `people_db/recognize_face` (PersonStateManager does this)

**Add**:
- Service server: `face_recognition/generate_embedding` (GenerateEmbedding)
- Service callback:
  1. Get timestamp from header
  2. Lookup high-res frame in cache
  3. Scale face bbox from low-res to high-res
  4. Extract face crop
  5. Generate 512-dim embedding
  6. Return success + embedding or failure

**Signature**:
```python
def generate_embedding_callback(self, request, response):
    # Get high-res frame from cache
    # Scale bbox
    # Extract face
    # Generate embedding
    # Return result
```

---

### 4. Refactor PersonStateManager → Orchestrator

**File**: `perception_pkg/person_state_manager_node.py`

**Add to internal state dict**:
```python
self.person_states[track.track_id] = {
    # ... existing fields ...
    'requires_identification': False,  # NEW: external flag
}
```

**Add service clients**:
- `face_detection/detect_face` (DetectFace)
- `face_recognition/generate_embedding` (GenerateEmbedding)
- `people_db/recognize_face` (RecognizeFace) - already have
- `people_db/add_person` (AddPerson) - NEW

**Add service server**:
- `person_state/enroll_person` (EnrollPerson)

**Remove**:
- Subscription to `/face_recognizer/results`
- Publication to `/person_state/identification_request`
- `identity_callback()` method
- `request_identification()` method
- `identification_request_pub` publisher
- `identity_sub` subscriber

**Modify `track_callback()`**:
```python
# Update state from track
state = self.person_states[track.track_id]
state['last_seen'] = track.header.stamp
state['bbox'] = (...)
state['tracking_confidence'] = track.tracking_confidence

# Check if identification needed
needs_id_internal = self.requires_identification(track, state)
needs_id_external = state['requires_identification']

# Combine: set requires_identification if internal logic says yes
if needs_id_internal and not needs_id_external:
    state['requires_identification'] = True

# Process identification if needed and not pending
if state['requires_identification'] and not state['identification_pending']:
    state['identification_pending'] = True

    # Call identification pipeline synchronously
    identity_result = self.perform_identification(track, state)

    if identity_result['success']:
        state['identity'] = identity_result['identity']
        state['identity_confidence'] = identity_result['confidence']
        state['person_id'] = identity_result['person_id']
        state['last_identification_time'] = track.header.stamp

    state['requires_identification'] = False
    state['identification_pending'] = False
    state['identification_attempts'] += 1
```

**Add method `perform_identification()`**:
```python
def perform_identification(self, track, state):
    """Call detection → embedding → recognition services"""

    # 1. Detect face
    face_bbox = self.call_detect_face(track)
    if not face_bbox['success']:
        return {'success': False, 'identity': 'unknown', 'confidence': 0.0, 'person_id': -1}

    # 2. Generate embedding
    embedding = self.call_generate_embedding(track.header, face_bbox)
    if not embedding['success']:
        return {'success': False, 'identity': 'unknown', 'confidence': 0.0, 'person_id': -1}

    # 3. Recognize face
    identity = self.call_recognize_face(embedding['embedding'])

    return {
        'success': True,
        'identity': identity['person_name'] if identity['match_found'] else 'unknown',
        'confidence': identity['similarity_score'],
        'person_id': identity['person_id'] if identity['match_found'] else -1
    }
```

**Add helper methods**:
```python
def call_detect_face(self, track):
    """Call face detection service"""
    # Wait for service, call async, spin_until_complete, return result

def call_generate_embedding(self, header, face_bbox):
    """Call embedding generation service"""
    # Wait for service, call async, spin_until_complete, return result

def call_recognize_face(self, embedding):
    """Call recognition service (already exists in database)"""
    # Wait for service, call async, spin_until_complete, return result

def call_add_person(self, name, embedding, notes):
    """Call add person service"""
    # Wait for service, call async, spin_until_complete, return result
```

**Add service callback `enroll_person_callback()`**:
```python
def enroll_person_callback(self, request, response):
    """Enroll new person from track_id"""
    track_id = request.track_id

    # Get current state
    if track_id not in self.person_states:
        response.success = False
        response.message = f"Track {track_id} not found"
        return response

    state = self.person_states[track_id]

    # Force fresh identification
    state['requires_identification'] = True
    state['identification_pending'] = True

    # Create pseudo-track from current state
    track = self.create_track_from_state(track_id, state)

    # 1. Detect face
    face_bbox = self.call_detect_face(track)
    if not face_bbox['success']:
        response.success = False
        response.message = "Face detection failed"
        state['identification_pending'] = False
        state['requires_identification'] = False
        return response

    # 2. Generate embedding
    embedding = self.call_generate_embedding(track.header, face_bbox)
    if not embedding['success']:
        response.success = False
        response.message = "Embedding generation failed"
        state['identification_pending'] = False
        state['requires_identification'] = False
        return response

    # 3. Add to database
    add_result = self.call_add_person(
        request.person_name,
        embedding['embedding'],
        request.notes
    )

    if not add_result['success']:
        response.success = False
        response.message = add_result['message']
        state['identification_pending'] = False
        state['requires_identification'] = False
        return response

    # 4. Update state directly
    state['identity'] = request.person_name
    state['person_id'] = add_result['person_id']
    state['identity_confidence'] = 1.0
    state['last_identification_time'] = self.get_clock().now().to_msg()
    state['identification_pending'] = False
    state['requires_identification'] = False

    response.success = True
    response.person_id = add_result['person_id']
    response.message = f"Successfully enrolled '{request.person_name}'"

    return response
```

**Modify `publish_state_array()`**:
- No changes needed to PersonState message fields
- Still publishes `requires_identification` and all other fields
- Visualization will use `identification_pending` from internal state

---

### 5. Update IdentificationVisualizationNode

**File**: `perception_pkg/identification_visualization_node.py`

**Change in `draw_annotations()` method** (line 169):

**OLD**:
```python
# Determine box color based on identification status
if person.requires_identification:
    # YELLOW: Pending identification
    color = (0, 255, 255)
    status = "Identifying..."
```

**NEW**:
```python
# Determine box color based on identification status
# Note: PersonState.requires_identification now reflects identification_pending from internal state
if person.requires_identification:
    # YELLOW: Actively identifying
    color = (0, 255, 255)
    status = "Identifying..."
```

**Wait - clarification needed**: PersonState message has `requires_identification` field. In the new architecture:
- Internal state has both `requires_identification` (external trigger) and `identification_pending` (actively processing)
- When publishing PersonStateArray, which internal field maps to `person_state.requires_identification`?

**Solution**: Update `publish_state_array()` in PersonStateManager:

**OLD** (line 329):
```python
person_state.requires_identification = state['identification_pending']
```

**Keep as is** - this already maps correctly! When `identification_pending` is True, the PersonState message shows `requires_identification=True`, which the visualization interprets as "Identifying..."

**No changes needed to visualization node** - it already works correctly!

---

### 6. Update Messages/Remove Unused

**Remove** (no longer needed):
- `msgs_interfaces/msg/IdentificationRequest.msg`
- `msgs_interfaces/msg/FaceRecognition.msg`
- `msgs_interfaces/msg/IdentityUpdate.msg`

**Update CMakeLists.txt**:
- Remove these messages from rosidl_generate_interfaces()

---

### 7. Update Launch File

**File**: `robot_bringup/launch/identification_pipeline_launch.py`

**Remove from face_recognizer parameters**:
```python
'face_crops_dir': '/ball-e/ros2_ws/robot_data/face_crops',  # REMOVE
```

**No other changes** - all nodes still launch with same names

---

### 8. Update Setup.py (if needed)

No changes - all nodes keep same entry points

---

## Summary of New Data Flow

### A. Automatic Identification (track_callback):
```
PersonTrackArray → PersonStateManager
  ↓ (decides requires_identification=True)
  ↓
  ├─ call DetectFace service → FaceDetectorNode
  │   └─ returns face_bbox
  ├─ call GenerateEmbedding service → FaceRecognizerNode
  │   └─ returns embedding
  └─ call RecognizeFace service → PeopleDatabaseNode
      └─ returns identity/confidence/person_id
  ↓
PersonStateManager updates internal state
  ↓
PersonStateArray published (requires_identification = identification_pending)
  ↓
IdentificationVisualizationNode receives PersonStateArray
  ↓
If requires_identification=True → YELLOW "Identifying..."
Elif identity != 'unknown' → GREEN "{name}"
Else → RED "Unknown"
```

### B. Manual Enrollment:
```
ros2 service call /person_state/enroll_person EnrollPerson "{track_id: 1, person_name: 'John', notes: 'Friend'}"
  ↓
PersonStateManager.enroll_person_callback()
  ├─ Sets identification_pending=True (shows "Identifying..." in viz)
  ├─ call DetectFace
  ├─ call GenerateEmbedding
  └─ call AddPerson → PeopleDatabaseNode
  ↓
PersonStateManager updates internal state directly
  └─ Sets identification_pending=False (viz updates to show "John")
  ↓
Returns success/person_id
```

---

## Visualization Behavior Summary

**YELLOW "Identifying..."**:
- When `person.requires_identification == True` in PersonStateArray
- This happens when `identification_pending == True` in PersonStateManager internal state
- Shows during active face detection/recognition processing

**GREEN "{person_name}"**:
- When `person.identity != 'unknown'` and `person.identity != ''`
- Shows identified persons with their names
- Confidence score shown if enabled

**RED "Unknown"**:
- When `person.identity == 'unknown'` or `person.identity == ''`
- Shows unidentified persons

---

## Testing Plan

1. Build workspace: `colcon build --packages-select msgs_interfaces perception_pkg`
2. Test face detection service: `ros2 service call /face_detection/detect_face ...`
3. Test embedding service: `ros2 service call /face_recognition/generate_embedding ...`
4. Test automatic identification with new track - verify visualization shows:
   - YELLOW "Identifying..." during processing
   - GREEN "{name}" or RED "Unknown" after completion
5. Test enrollment service: `ros2 service call /person_state/enroll_person ...`
   - Verify visualization shows YELLOW during enrollment
   - Verify visualization shows GREEN with person's name after success

---

## Benefits

✅ Simplified architecture - PersonStateManager is single orchestrator
✅ Synchronous service calls - easier to debug and reason about
✅ No more complex message passing
✅ Single source of truth for person state
✅ Enrollment is built-in service, no separate node needed
✅ Face crops removed (were only for debugging)
✅ Visualization automatically shows "Identifying..." during processing via identification_pending flag
✅ No changes needed to visualization node - already works correctly!
